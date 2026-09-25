"""Inline feature construction from raw `input/` -- one function per plan node.

NORMALISATION NOTE
------------------
In the original workspace these blocks were materialised once into parquet
("the frozen feature store") and every pipeline read them back. That is not how
the other trajectories in this collection work, so the computation has been
inlined: `common.load_xy` wires the functions below into the recorded skrub
plan, and a pipeline now derives everything it needs from `input/` on its own.
Nothing about the features themselves changed -- `verify_inline_features.py`
checks every block against the original store.

GRANULARITY
-----------
Each function is one meaningful step and becomes one node via
`.skb.apply_func`, rather than one opaque blob (guide section 4 / pitfall 13):

    context -> link degrees -> seed edges -> {13 feature blocks} -> merge -> marks

The heavy lifting is numpy/polars because the inputs are a 623M-edge link graph
and a 36.7M-edge tracking graph; recording it as pandas row ops is not an
option at that size. What matters for the plan is that each *step* is its own
recorded node with explicit inputs.

WHERE THIS SITS RELATIVE TO THE CV (important, and unchanged by inlining)
------------------------------------------------------------------------
These blocks are built BEFORE `mark_as_X`, i.e. once per run, not per fold.
That is deliberate and is the semantically correct placement for this task, not
a shortcut:

  * A domain's features are a function of the GIVEN data (the link graph and the
    tracking graph over all 18.7M tracked domains) and its own identity -- never
    of its own label, which is always excluded (self-loops dropped, `tld_pop`
    leave-one-out corrected, `trk_cooc`/`tok_pop` estimated only on domains
    outside the modelled sample).
  * At real prediction time the 50,000 `target.tsv` domains are unseen, but
    every OTHER domain's trackers are known. Recomputing these blocks per fold
    from training-fold rows only would model a harder problem than the real one
    and make the CV pessimistic.

The consequence is that cross-validation covers the model and the encoding, but
not this file. A bug here is invisible to the score -- it just makes every fold
agree on the same wrong answer, which is exactly how the `nbr_2h` self-leak in
the original run reached the leaderboard. `../exploration/data_exploration_8.py`
is the guard for this layer; run it against any new block.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix

N_TRK = 355
MOD, REM = 37, 11         # the frozen row sample: tracked domains with id % 37 == 11
MAX_MID_DEG = 32          # 2-hop intermediaries above this are hubs, not signal
MIN_TOK_DF = 20           # hostname token needs this many out-of-sample domains
MAX_VOCAB = 200_000
TOK_ALPHA = 50.0          # empirical-Bayes shrinkage of a token profile to the prior
SLD = {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "gob", "in"}


# ===========================================================  shared context
def load_context(input_dir):
    """Row sample + a CSR of the tracking graph. The root every block hangs off.

    The row sample is deterministic and frozen: tracked domains with
    `domain_id % 37 == 11` (505,548 rows), then all 50,000 `target.tsv` domains,
    each block sorted by domain_id. Row ORDER is what makes scores comparable
    across the whole run, so this must never change.
    """
    input_dir = Path(input_dir)
    tg = pl.read_parquet(input_dir / "tracking_graph_train.parquet",
                         columns=["domain_id", "tracker_id"]).with_columns(
        pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int32))
    tracked = tg.select("domain_id").unique().sort("domain_id")
    train_ids = tracked.filter((pl.col("domain_id") % MOD) == REM)["domain_id"].to_numpy()
    target_ids = pl.read_csv(input_dir / "target.tsv", separator="\t").with_columns(
        pl.col("domain_id").cast(pl.Int32)).sort("domain_id")["domain_id"].to_numpy()

    all_ids = np.concatenate([train_ids, target_ids])
    n, n_train = len(all_ids), len(train_ids)
    maxid = int(max(tracked["domain_id"].max(), all_ids.max())) + 1
    row_of = np.full(maxid, -1, dtype=np.int32)
    row_of[all_ids] = np.arange(n, dtype=np.int32)
    is_train = np.zeros(n, dtype=bool)
    is_train[:n_train] = True
    in_sample = np.zeros(maxid, dtype=bool)
    in_sample[train_ids] = True

    tg_sorted = tg.sort("domain_id")
    tg_dom = tg_sorted["domain_id"].to_numpy()
    tg_trk = tg_sorted["tracker_id"].to_numpy()
    n_trk_of = np.bincount(tg_dom, minlength=maxid).astype(np.int32)
    offs = np.zeros(maxid + 1, dtype=np.int64)
    np.cumsum(n_trk_of, out=offs[1:])

    return SimpleNamespace(
        input_dir=input_dir, tg_dom=tg_dom, tg_trk=tg_trk, tg_sorted=tg_sorted,
        tracked_ids=tracked["domain_id"].to_numpy(), n_trk_of=n_trk_of, offs=offs,
        is_tracked=n_trk_of > 0, maxid=maxid, train_ids=train_ids,
        target_ids=target_ids, all_ids=all_ids, n=n, n_train=n_train,
        row_of=row_of, is_train=is_train, in_sample=in_sample)


def _expand(ctx, node_ids):
    """(nodes -> their trackers) via the CSR. Returns (index_into_nodes, tracker)."""
    cnt = ctx.n_trk_of[node_ids].astype(np.int64)
    src = np.repeat(np.arange(len(node_ids), dtype=np.int64), cnt)
    within = (np.arange(int(cnt.sum()), dtype=np.int64)
              - np.repeat(np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt))
    return src, ctx.tg_trk[np.repeat(ctx.offs[node_ids], cnt) + within]


def _scatter(ctx, rows, trk, w=None):
    out = np.bincount(rows.astype(np.int64) * N_TRK + trk, weights=w,
                      minlength=ctx.n * N_TRK)
    return out.reshape(ctx.n, N_TRK).astype(np.float32)


def _frame(ctx, arr, prefix):
    """(n, 355) array -> the block's DataFrame, keyed by domain_id."""
    return pd.concat(
        [pd.DataFrame({"domain_id": ctx.all_ids}),
         pd.DataFrame(arr, columns=[f"{prefix}_{i}" for i in range(N_TRK)])], axis=1)


# ===========================================================  link-graph nodes
def link_degrees(ctx):
    """Total (in+out) link-graph degree of every domain -- one full scan of 623M edges."""
    deg = np.zeros(ctx.maxid, dtype=np.int32)
    pf = pq.ParquetFile(ctx.input_dir / "link-graph.parquet")
    for b in pf.iter_batches(batch_size=40_000_000,
                             columns=["source_domain_id", "target_domain_id"]):
        for c in (0, 1):
            v = b.column(c).to_numpy()
            v = v[(v >= 0) & (v < ctx.maxid)]
            deg += np.bincount(v, minlength=ctx.maxid).astype(np.int32)
    return deg


def seed_edges(ctx):
    """Link edges touching a sampled/target domain, both directions, self-loops dropped."""
    lg = pl.scan_parquet(ctx.input_dir / "link-graph.parquet")
    seed = pl.Series(ctx.all_ids).implode()
    out = {}
    for direction, self_col, nbr_col in (
            ("out", "source_domain_id", "target_domain_id"),
            ("in", "target_domain_id", "source_domain_id")):
        ed = lg.filter(pl.col(self_col).is_in(seed)).collect(engine="streaming")
        si, ni = ed[self_col].to_numpy(), ed[nbr_col].to_numpy()
        keep = (si != ni) & (ni >= 0) & (ni < ctx.maxid)
        out[direction] = (ctx.row_of[si[keep]], ni[keep])
    out["rows"] = np.concatenate([out["out"][0], out["in"][0]])
    out["nbrs"] = np.concatenate([out["out"][1], out["in"][1]])
    return out


# ===========================================================  the 13 blocks
def block_labels(ctx):
    """The target: a 355-column binary tracker indicator per TRAIN row."""
    y = np.zeros((ctx.n, N_TRK), dtype=np.uint8)
    e = ctx.tg_sorted.filter(
        pl.col("domain_id").is_in(pl.Series(ctx.train_ids).implode()))
    y[ctx.row_of[e["domain_id"].to_numpy()], e["tracker_id"].to_numpy()] = 1
    return _frame(ctx, y, "y")


def block_nbr_out(ctx, edges):
    """Trackers of the domains this domain links TO (raw mention counts)."""
    rows, nbrs = edges["out"]
    src, trk = _expand(ctx, nbrs)
    return _frame(ctx, _scatter(ctx, rows[src], trk), "no")


def block_nbr_in(ctx, edges):
    """Trackers of the domains that link TO this domain."""
    rows, nbrs = edges["in"]
    src, trk = _expand(ctx, nbrs)
    return _frame(ctx, _scatter(ctx, rows[src], trk), "ni")


def block_nbr_hub(ctx, edges, linkdeg):
    """Neighbourhood with each link weighted 1/log2(2 + linkdeg(neighbour)).

    Discounts 50k-outlink directories against hand-made links. The single most
    valuable block in the ablation (-0.00115 to remove).
    """
    w = (1.0 / np.log2(2.0 + linkdeg[edges["nbrs"]])).astype(np.float64)
    src, trk = _expand(ctx, edges["nbrs"])
    return _frame(ctx, _scatter(ctx, edges["rows"][src], trk, np.repeat(w, ctx.n_trk_of[edges["nbrs"]])), "nh")


def block_nbr_rec(ctx, edges):
    """Trackers of RECIPROCAL neighbours only (d->n and n->d)."""
    key_out = edges["out"][0].astype(np.int64) * ctx.maxid + edges["out"][1]
    key_in = edges["in"][0].astype(np.int64) * ctx.maxid + edges["in"][1]
    rec = np.intersect1d(key_out, key_in, assume_unique=False)
    rows = (rec // ctx.maxid).astype(np.int32)
    nbrs = (rec % ctx.maxid).astype(np.int64)
    src, trk = _expand(ctx, nbrs)
    return _frame(ctx, _scatter(ctx, rows[src], trk), "nr")


def two_hop_pairs(ctx, edges, linkdeg):
    """(row, intermediary) pairs: untracked, low-degree neighbours to route through."""
    ok = ((~ctx.is_tracked[edges["nbrs"]]) & (linkdeg[edges["nbrs"]] <= MAX_MID_DEG)
          & (linkdeg[edges["nbrs"]] > 0))
    pk = np.unique(edges["rows"][ok].astype(np.int64) * ctx.maxid + edges["nbrs"][ok])
    return (pk // ctx.maxid).astype(np.int32), (pk % ctx.maxid).astype(np.int64)


def block_nbr_2h(ctx, edges, linkdeg):
    """2-hop reach through untracked low-degree intermediaries.

    THE `keep2` MASK IS LOAD-BEARING. `mid` was reached FROM the seed, so the
    seed is still in mid's own edge list; without dropping `seed -> mid -> seed`
    round trips this block writes every training row's own label vector into its
    own feature. That bug scored 0.91276 with a fold std of 0.00005 before
    `data_exploration_8.py` caught it (own-tracker coverage was exactly 1.0000).
    """
    pair_rows, pair_mid = two_hop_pairs(ctx, edges, linkdeg)
    mids = np.unique(pair_mid)
    lg = pl.scan_parquet(ctx.input_dir / "link-graph.parquet")
    mseed = pl.Series(mids.astype(np.int32)).implode()
    m_self, m_nbr = [], []
    for self_col, nbr_col in (("source_domain_id", "target_domain_id"),
                              ("target_domain_id", "source_domain_id")):
        ed = lg.filter(pl.col(self_col).is_in(mseed)).collect(engine="streaming")
        a, b = ed[self_col].to_numpy(), ed[nbr_col].to_numpy()
        keep = ((a != b) & (b >= 0) & (b < ctx.maxid)
                & ctx.is_tracked[np.clip(b, 0, ctx.maxid - 1)])
        m_self.append(a[keep])
        m_nbr.append(b[keep])
    m_self, m_nbr = np.concatenate(m_self), np.concatenate(m_nbr)

    mid_pos = np.full(ctx.maxid, -1, dtype=np.int32)
    mid_pos[mids] = np.arange(len(mids), dtype=np.int32)
    mp = mid_pos[m_self]
    order = np.argsort(mp, kind="stable")
    mp, m_nbr = mp[order], m_nbr[order]
    mcnt = np.bincount(mp, minlength=len(mids)).astype(np.int64)
    moffs = np.zeros(len(mids) + 1, dtype=np.int64)
    np.cumsum(mcnt, out=moffs[1:])

    acc = np.zeros((ctx.n, N_TRK), dtype=np.float64)
    mass = np.zeros(ctx.n, dtype=np.float64)
    pp = mid_pos[pair_mid]
    CH = 4_000_000
    for i in range(0, len(pp), CH):
        p, r = pp[i:i + CH], pair_rows[i:i + CH]
        c = mcnt[p]
        if c.sum() == 0:
            continue
        rr = np.repeat(r, c)
        within = (np.arange(int(c.sum()), dtype=np.int64)
                  - np.repeat(np.concatenate([[0], np.cumsum(c)[:-1]]), c))
        nb = m_nbr[np.repeat(moffs[p], c) + within]
        keep2 = nb != ctx.all_ids[rr]              # drop seed -> mid -> seed
        rr, nb = rr[keep2], nb[keep2]
        src, trk = _expand(ctx, nb)
        acc += np.bincount(rr[src].astype(np.int64) * N_TRK + trk,
                           minlength=ctx.n * N_TRK).reshape(ctx.n, N_TRK)
        mass += np.bincount(rr[src], minlength=ctx.n)
    return _frame(ctx, acc.astype(np.float32), "n2")


def block_nbr_frac(ctx, edges):
    """P(tracker | a random DISTINCT tracked neighbour).

    Not derivable from nbr_out/nbr_in/nbr_hub: those sum tracker MENTIONS, so a
    neighbour running 50 trackers outvotes 25 single-tracker neighbours.
    """
    m = ctx.is_tracked[edges["nbrs"]]
    pk = np.unique(edges["rows"][m].astype(np.int64) * ctx.maxid + edges["nbrs"][m])
    rows = (pk // ctx.maxid).astype(np.int32)
    nbrs = (pk % ctx.maxid).astype(np.int64)
    n_distinct = np.bincount(rows, minlength=ctx.n).astype(np.float32)
    src, trk = _expand(ctx, nbrs)
    f = _scatter(ctx, rows[src], trk)
    f /= np.maximum(n_distinct, 1.0)[:, None]
    return _frame(ctx, f, "nf")


def block_direct(ctx):
    """Direct hyperlinks between the domain and each tracker's OWN hostname.

    The only block built from data no other block reads, and the largest single
    feature gain in the run (+0.00487): such a link coincides with a true
    tracking edge 50.4% of the time against a 0.55% base rate.
    """
    trk_meta = pl.read_csv(ctx.input_dir / "trackers.tsv", separator="\t")
    trk_of_domain = np.full(ctx.maxid, -1, dtype=np.int32)
    trk_of_domain[trk_meta["tracking_domain_id"].to_numpy()] = \
        trk_meta["tracker_id"].to_numpy()
    tset = pl.Series(np.sort(trk_meta["tracking_domain_id"].to_numpy())) \
        .cast(pl.Int32).implode()
    seed = pl.Series(ctx.all_ids).implode()
    lg = pl.scan_parquet(ctx.input_dir / "link-graph.parquet")

    d = np.zeros((ctx.n, N_TRK), dtype=np.float32)
    for self_col, other_col in (("source_domain_id", "target_domain_id"),
                                ("target_domain_id", "source_domain_id")):
        ed = lg.filter(pl.col(self_col).is_in(seed)
                       & pl.col(other_col).is_in(tset)).collect(engine="streaming")
        r = ctx.row_of[ed[self_col].to_numpy()]
        t = trk_of_domain[ed[other_col].to_numpy()]
        ok = (r >= 0) & (t >= 0)
        np.add.at(d, (r[ok], t[ok]), 1.0)
    return _frame(ctx, d, "dl")


def _tld_map(ctx):
    dom = pl.read_parquet(ctx.input_dir / "domains.parquet").with_columns(
        pl.col("domain_id").cast(pl.Int32))
    dom = dom.with_columns(pl.col("domain").str.split(".").alias("_p")).with_columns(
        pl.col("_p").list.last().alias("tld")).drop("_p")
    uniq = sorted(dom["tld"].unique().to_list())
    idx = {t: i for i, t in enumerate(uniq)}
    m = np.full(ctx.maxid, -1, dtype=np.int32)
    sub = dom.filter(pl.col("domain_id") < ctx.maxid)
    m[sub["domain_id"].to_numpy()] = np.array(
        [idx.get(t, -1) for t in sub["tld"].to_list()], dtype=np.int32)
    return m, len(uniq)


def block_tld_pop(ctx, labels):
    """Leave-one-out P(tracker | TLD).

    LOO matters for rare TLDs: without subtracting the row's own labels, a
    domain that is the only one with its TLD would read its own answer back.
    """
    tld_map, n_tld = _tld_map(ctx)
    tld_of_edge = tld_map[ctx.tg_dom]
    ok = tld_of_edge >= 0
    tp = np.bincount(tld_of_edge[ok].astype(np.int64) * N_TRK + ctx.tg_trk[ok],
                     minlength=n_tld * N_TRK).reshape(n_tld, N_TRK).astype(np.float32)
    tld_of_tracked = tld_map[ctx.tracked_ids]
    n_dom = np.bincount(tld_of_tracked[tld_of_tracked >= 0], minlength=n_tld)

    y = labels[[f"y_{i}" for i in range(N_TRK)]].to_numpy().astype(np.float32)
    row_tld = tld_map[ctx.all_ids]
    out = np.zeros((ctx.n, N_TRK), dtype=np.float32)
    has = row_tld >= 0
    out[has] = tp[row_tld[has]]
    denom = n_dom[np.where(has, row_tld, 0)].astype(np.float32)
    out[ctx.is_train] -= y[ctx.is_train]                 # leave-one-out
    denom = np.where(ctx.is_train, denom - 1.0, denom)
    out /= np.maximum(denom, 1.0)[:, None]
    np.clip(out, 0.0, None, out=out)
    return _frame(ctx, out, "tp")


def block_trk_cooc(ctx, nbr_hub):
    """Neighbourhood propagated through tracker co-occurrence: nh_l1 @ P(b|a).

    P(b|a) is estimated ONLY on the 18.18M tracked domains OUTSIDE the modelled
    sample, so it is independent of every scored row's label by construction --
    stronger than leave-one-out, and identical in kind for target rows.
    """
    out_mask = ~ctx.in_sample[ctx.tg_dom]
    _, comp = np.unique(ctx.tg_dom[out_mask], return_inverse=True)
    m = csr_matrix((np.ones(len(comp), dtype=np.float32),
                    (comp, ctx.tg_trk[out_mask])), shape=(comp.max() + 1, N_TRK))
    c = np.asarray((m.T @ m).todense(), dtype=np.float64)
    diag = np.maximum(np.diag(c).copy(), 1.0)
    cn = (c / diag[:, None]).astype(np.float32)
    np.fill_diagonal(cn, 0.0)

    nh = nbr_hub[[f"nh_{i}" for i in range(N_TRK)]].to_numpy().astype(np.float32)
    nh = nh / np.maximum(nh.sum(1, keepdims=True), 1.0)
    return _frame(ctx, nh @ cn, "tc")


def block_tok_pop(ctx):
    """IDF-weighted, empirical-Bayes-smoothed P(tracker | hostname token).

    Shrinkage is not optional here: coverage needs a df floor as low as 20, and
    the IDF weighting gives the RAREST tokens the LARGEST weight -- precisely
    where the profiles are worst estimated. `(count + 50*prior)/(df + 50)` lets a
    thin token decay to the prior instead of injecting noise.
    """
    dom = pl.read_parquet(ctx.input_dir / "domains.parquet").with_columns(
        pl.col("domain_id").cast(pl.Int32))

    def tokenise(df):
        return df.with_columns(
            pl.col("domain").str.to_lowercase().str.replace(r"\.[a-z]+$", "")
            .str.extract_all(r"[a-z]{3,}").alias("tok"))

    seed = pl.Series(ctx.all_ids).implode()
    seed_tok = tokenise(dom.filter(pl.col("domain_id").is_in(seed))) \
        .select(["domain_id", "tok"])
    seed_vocab = seed_tok.explode("tok").drop_nulls("tok")["tok"].unique()

    out_ids = np.setdiff1d(ctx.tracked_ids, ctx.train_ids)
    out_tok = tokenise(
        dom.filter(pl.col("domain_id").is_in(pl.Series(out_ids).implode()))) \
        .select(["domain_id", "tok"]).explode("tok").drop_nulls("tok")
    out_tok = out_tok.filter(pl.col("tok").is_in(seed_vocab.implode()))
    df_tok = (out_tok.group_by("tok").len().rename({"len": "df"})
              .filter(pl.col("df") >= MIN_TOK_DF)
              .sort("df", descending=True).head(MAX_VOCAB))

    vocab_df = df_tok.select(["tok", "df"]).with_row_index("ti")
    n_vocab = vocab_df.height
    df_arr = vocab_df["df"].to_numpy().astype(np.float64)

    pairs = (out_tok.join(vocab_df.select(["tok", "ti"]), on="tok", how="inner")
             .join(ctx.tg_sorted, on="domain_id", how="inner"))
    tp = np.bincount(pairs["ti"].to_numpy().astype(np.int64) * N_TRK
                     + pairs["tracker_id"].to_numpy(),
                     minlength=n_vocab * N_TRK).reshape(n_vocab, N_TRK).astype(np.float64)
    prior = (np.bincount(
        ctx.tg_sorted.filter(pl.col("domain_id").is_in(pl.Series(out_ids).implode()))
        ["tracker_id"].to_numpy(), minlength=N_TRK).astype(np.float64)
        / float(len(out_ids)))
    tp = ((tp + TOK_ALPHA * prior[None, :]) / (df_arr[:, None] + TOK_ALPHA)).astype(np.float32)

    idf = np.log(float(len(out_ids)) / df_arr).astype(np.float32)
    se = (seed_tok.explode("tok").drop_nulls("tok")
          .join(vocab_df.select(["tok", "ti"]), on="tok", how="inner"))
    se_row = ctx.row_of[se["domain_id"].to_numpy()]
    se_tok = se["ti"].to_numpy().astype(np.int64)

    kp = np.zeros((ctx.n, N_TRK), dtype=np.float32)
    wsum = np.zeros(ctx.n, dtype=np.float32)
    CH = 2_000_000
    for i in range(0, len(se_row), CH):
        r, t = se_row[i:i + CH], se_tok[i:i + CH]
        w = idf[t]
        np.add.at(kp, r, tp[t] * w[:, None])
        np.add.at(wsum, r, w)
    kp /= np.maximum(wsum, 1e-6)[:, None]
    return _frame(ctx, kp, "kp")


# ===========================================================  scalar blocks
def block_meta(ctx, edges):
    """Hostname shape, TLD, degrees, press freedom, url-classification category."""
    dom = pl.read_parquet(ctx.input_dir / "domains.parquet").with_columns(
        pl.col("domain_id").cast(pl.Int32))
    dom = dom.with_columns(pl.col("domain").str.split(".").alias("_p")).with_columns(
        pl.col("_p").list.last().alias("tld"),
        pl.col("_p").list.len().cast(pl.Int32).alias("n_labels"),
        pl.col("domain").str.len_chars().cast(pl.Int32).alias("host_len"),
        pl.col("domain").str.count_matches(r"[0-9]").cast(pl.Int32).alias("n_digits"),
        pl.col("domain").str.count_matches("-").cast(pl.Int32).alias("n_hyphens"),
    ).with_columns(
        pl.when(pl.col("_p").list.get(-2, null_on_oob=True).is_in(list(SLD))
                & (pl.col("n_labels") >= 3))
        .then(pl.col("_p").list.slice(-3, 3).list.join("."))
        .otherwise(pl.col("_p").list.slice(-2, 2).list.join("."))
        .alias("reg_dom")).drop("_p")

    deg = {}
    for direction in ("out", "in"):
        rows, nbrs = edges[direction]
        deg[f"{direction}deg"] = np.bincount(rows, minlength=ctx.n).astype(np.int32)
        deg[f"n_trk_nbr_{direction}"] = np.bincount(
            rows[ctx.is_tracked[nbrs]], minlength=ctx.n).astype(np.int32)

    fp = pl.read_csv(ctx.input_dir / "freedom-of-the-press.csv", separator="\t")
    uc = pl.read_csv(ctx.input_dir / "url-classification.csv", infer_schema_length=10000)
    uc = uc.with_columns(
        pl.col("url").str.replace(r"^https?://", "").str.split("/").list.first()
        .str.replace(r"^www\.", "").str.to_lowercase().alias("host")
    ).select(["host", "category"]).unique(subset=["host"]).rename(
        {"category": "uc_category"})

    meta = pl.DataFrame({"domain_id": ctx.all_ids}).join(dom, on="domain_id", how="left")
    meta = meta.with_columns([pl.Series(k, v) for k, v in deg.items()])
    meta = meta.join(fp.select(["tld", "freedom_of_the_press"]), on="tld", how="left")
    meta = meta.with_columns(
        pl.col("domain").str.replace(r"^www\.", "").alias("_h")
    ).join(uc, left_on="_h", right_on="host", how="left").drop("_h")
    meta = meta.with_columns(
        (pl.col("n_trk_nbr_out") + pl.col("n_trk_nbr_in")).alias("n_trk_nbr_tot"))
    return meta.to_pandas()


def block_meta2(ctx, edges, linkdeg, nbr_hub, nbr_2h):
    """Scalars that let the model calibrate the 355-wide blocks.

    Takes `nbr_hub` / `nbr_2h` as inputs rather than recomputing their masses,
    so the numbers are the same objects the model sees.
    """
    key_out = edges["out"][0].astype(np.int64) * ctx.maxid + edges["out"][1]
    key_in = edges["in"][0].astype(np.int64) * ctx.maxid + edges["in"][1]
    rec_rows = (np.intersect1d(key_out, key_in, assume_unique=False)
                // ctx.maxid).astype(np.int32)
    pair_rows, _ = two_hop_pairs(ctx, edges, linkdeg)

    nbr_deg = linkdeg[edges["nbrs"]].astype(np.float64)
    cnt = np.bincount(edges["rows"], minlength=ctx.n).astype(np.float64)
    sum_deg = np.bincount(edges["rows"], weights=nbr_deg, minlength=ctx.n)
    max_deg = np.zeros(ctx.n, dtype=np.float64)
    np.maximum.at(max_deg, edges["rows"], nbr_deg)

    # ascontiguousarray is load-bearing, not cosmetic: pandas hands back an
    # F-contiguous block, and summing that along axis 1 accumulates float32 in a
    # different pairwise order than the C-contiguous scatter output the original
    # store summed. The values drift ~1e-6 otherwise -- harmless numerically,
    # but it breaks the bit-for-bit equivalence that lets results.json stand.
    hub_mass = np.ascontiguousarray(
        nbr_hub[[f"nh_{i}" for i in range(N_TRK)]].to_numpy(), dtype=np.float32).sum(1)
    two_mass = np.ascontiguousarray(
        nbr_2h[[f"n2_{i}" for i in range(N_TRK)]].to_numpy(), dtype=np.float32).sum(1)
    return pd.DataFrame({
        "domain_id": ctx.all_ids,
        "n_rec_nbr": np.bincount(rec_rows, minlength=ctx.n).astype(np.int32),
        "n_untracked_lowdeg_nbr": np.bincount(pair_rows, minlength=ctx.n).astype(np.int32),
        "hub_weight_mass": hub_mass.astype(np.float32),
        "two_hop_mass": two_mass.astype(np.float32),
        "mean_nbr_linkdeg": (sum_deg / np.maximum(cnt, 1)).astype(np.float32),
        "max_nbr_linkdeg": max_deg.astype(np.float32),
        "own_linkdeg": linkdeg[ctx.all_ids].astype(np.float32),
    })


def split_rows(frame, ctx, split):
    """Keep the train half (labelled) or the target half (for submission)."""
    n_train = ctx.n_train
    out = frame.iloc[:n_train] if split == "train" else frame.iloc[n_train:]
    return out.reset_index(drop=True)
