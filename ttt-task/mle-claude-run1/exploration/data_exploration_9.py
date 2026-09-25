"""Round 9: four NEW feature blocks, chosen by what the model cannot already derive.

Lesson from pipelines 07-09: a block that is a *linear function of inputs the MLP
already has* buys nothing -- the first dense layer can learn that map itself.
`nbr_rec`/`nbr_2h` were re-weightings of `nbr_hub` and came out net-negative, and
`tld_pop` (0.795 standalone!) has -0.00013 marginal value because the
hub-neighbourhood plus one-hot TLD already span it.

So every block here is either NEW DATA or a NON-LINEAR transform of existing data:

  direct    dl_*  Direct hyperlinks between the domain and each tracker's OWN
                  domain, from the link graph. Genuinely new data -- no block so
                  far looks at edges pointing AT the 355 tracker hostnames. A
                  prior probe measured 50.4% precision for the tracking relation
                  against a 0.55% base rate (~90x lift), on 10.9% of train and
                  16.9% of target domains.

  nbr_frac  nf_*  P(tracker | a random tracked neighbour) = (#distinct
                  neighbours using t) / (#distinct tracked neighbours).
                  NOT derivable from the existing blocks: those are sums of
                  tracker MENTIONS, so a neighbour carrying 50 trackers outvotes
                  25 single-tracker neighbours. Per-neighbour normalisation is
                  non-linear in them, and aims at the >100-neighbour regime,
                  which has the worst recall of all (0.798, exploration_5).

  trk_cooc  tc_*  Neighbour evidence propagated through the tracker-tracker
                  co-occurrence matrix: tc = nbr_hub_l1 @ P(b|a). This IS a
                  linear map of existing features -- the point is that its
                  coefficients are estimated from 18.18M out-of-sample domains
                  rather than the 337k rows an MLP fold sees, the same reason
                  target encoding beats letting a model learn a category itself.
                  Attacks the hard ceiling that only 79.7% of a domain's true
                  trackers appear anywhere in its neighbourhood: a correlated
                  tracker can now be surfaced even when unobserved nearby.

  tok_pop   kp_*  IDF-weighted mean of P(tracker | hostname token) over the
                  domain's own hostname tokens. New data (hostname text) in the
                  form that WORKED for TLDs, rather than the raw char n-grams
                  that lost 0.0033 in pipeline_08: a target-informed profile,
                  not surface text for the net to interpret.

LEAKAGE: `trk_cooc` and `tok_pop` are label-derived statistics, so both are
estimated ONLY from the 18.18M tracked domains OUTSIDE the frozen sample. That
is stronger than leave-one-out -- the statistic is independent of every scored
row's label by construction, and identical in kind for target rows. Run
data_exploration_8.py afterwards to confirm.
"""
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

WS = Path(__file__).resolve().parent.parent
IN, OUT = WS / "input", WS / "features"
MOD, REM = 37, 11
N_TRK = 355
MIN_TOK_DF = 20           # a token needs this many out-of-sample domains to count
MAX_VOCAB = 200_000
TOK_ALPHA = 50.0          # empirical-Bayes shrinkage of a token profile to the prior
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------- frozen row order (identical)
tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"]).with_columns(
    pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int32))
tracked = tg.select("domain_id").unique().sort("domain_id")
train_ids = tracked.filter((pl.col("domain_id") % MOD) == REM)["domain_id"].to_numpy()
target_ids = pl.read_csv(IN / "target.tsv", separator="\t").with_columns(
    pl.col("domain_id").cast(pl.Int32)).sort("domain_id")["domain_id"].to_numpy()
all_ids = np.concatenate([train_ids, target_ids])
N, N_TRAIN = len(all_ids), len(train_ids)
MAXID = int(max(tracked["domain_id"].max(), all_ids.max())) + 1
row_of = np.full(MAXID, -1, dtype=np.int32)
row_of[all_ids] = np.arange(N, dtype=np.int32)
IN_SAMPLE = np.zeros(MAXID, dtype=bool)
IN_SAMPLE[train_ids] = True          # the rows whose labels must never be used
log("rows:", N, " train:", N_TRAIN)

tg_sorted = tg.sort("domain_id")
tg_dom, tg_trk = tg_sorted["domain_id"].to_numpy(), tg_sorted["tracker_id"].to_numpy()
n_trk_of = np.bincount(tg_dom, minlength=MAXID).astype(np.int32)
offs = np.zeros(MAXID + 1, dtype=np.int64)
np.cumsum(n_trk_of, out=offs[1:])
IS_TRACKED = n_trk_of > 0


def expand_trackers(node_ids):
    cnt = n_trk_of[node_ids].astype(np.int64)
    src = np.repeat(np.arange(len(node_ids), dtype=np.int64), cnt)
    within = (np.arange(int(cnt.sum()), dtype=np.int64)
              - np.repeat(np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt))
    return src, tg_trk[np.repeat(offs[node_ids], cnt) + within]


def save(arr, prefix, fname):
    cols = [f"{prefix}_{i}" for i in range(N_TRK)]
    for tag, sl in (("train", slice(0, N_TRAIN)), ("target", slice(N_TRAIN, N))):
        pl.DataFrame({"domain_id": all_ids[sl]}).hstack(
            pl.DataFrame(arr[sl], schema=cols)).write_parquet(
            OUT / f"{fname}_{tag}.parquet", compression="zstd")
    log("wrote", fname, arr.shape, " nonzero rows:",
        int((arr.sum(1) > 0).sum()))


lg = pl.scan_parquet(IN / "link-graph.parquet")
seed = pl.Series(all_ids).implode()

# =============================================================== 1. direct
trk_meta = pl.read_csv(IN / "trackers.tsv", separator="\t")
trk_dom_ids = trk_meta["tracking_domain_id"].to_numpy()
trk_ids = trk_meta["tracker_id"].to_numpy()
trk_of_domain = np.full(MAXID, -1, dtype=np.int32)
trk_of_domain[trk_dom_ids] = trk_ids
tset = pl.Series(np.sort(trk_dom_ids)).cast(pl.Int32).implode()

D = np.zeros((N, N_TRK), dtype=np.float32)
for self_col, other_col in (("source_domain_id", "target_domain_id"),
                            ("target_domain_id", "source_domain_id")):
    ed = lg.filter(pl.col(self_col).is_in(seed)
                   & pl.col(other_col).is_in(tset)).collect(engine="streaming")
    r = row_of[ed[self_col].to_numpy()]
    t = trk_of_domain[ed[other_col].to_numpy()]
    ok = (r >= 0) & (t >= 0)
    np.add.at(D, (r[ok], t[ok]), 1.0)
    log("direct edges", self_col, "->", int(ok.sum()))
save(D, "dl", "direct")
del D

# =============================================================== 2. nbr_frac
sides = {}
for direction, self_col, nbr_col in (("out", "source_domain_id", "target_domain_id"),
                                     ("in", "target_domain_id", "source_domain_id")):
    ed = lg.filter(pl.col(self_col).is_in(seed)).collect(engine="streaming")
    si, ni = ed[self_col].to_numpy(), ed[nbr_col].to_numpy()
    keep = (si != ni) & (ni >= 0) & (ni < MAXID)
    sides[direction] = (row_of[si[keep]], ni[keep])
    log(direction, "seed edges:", int(keep.sum()))

rows_all = np.concatenate([sides["out"][0], sides["in"][0]])
nbrs_all = np.concatenate([sides["out"][1], sides["in"][1]])
# distinct (row, tracked neighbour) pairs -- direction and multiplicity collapsed
m = IS_TRACKED[nbrs_all]
pk = np.unique(rows_all[m].astype(np.int64) * MAXID + nbrs_all[m])
p_rows = (pk // MAXID).astype(np.int32)
p_nbrs = (pk % MAXID).astype(np.int64)
n_distinct = np.bincount(p_rows, minlength=N).astype(np.float32)
log("distinct (row, tracked nbr) pairs:", len(pk))

src, trk = expand_trackers(p_nbrs)
F = np.bincount(p_rows[src].astype(np.int64) * N_TRK + trk,
                minlength=N * N_TRK).reshape(N, N_TRK).astype(np.float32)
F /= np.maximum(n_distinct, 1.0)[:, None]
save(F, "nf", "nbr_frac")
del F, src, trk

# =============================================================== 3. trk_cooc
# co-occurrence estimated ONLY on domains outside the frozen sample
out_mask = ~IN_SAMPLE[tg_dom]
d_out = tg_dom[out_mask]
_, comp = np.unique(d_out, return_inverse=True)
M = csr_matrix((np.ones(len(comp), dtype=np.float32),
                (comp, tg_trk[out_mask])), shape=(comp.max() + 1, N_TRK))
log("out-of-sample domains for co-occurrence:", M.shape[0],
    " edges:", int(out_mask.sum()))
C = np.asarray((M.T @ M).todense(), dtype=np.float64)
diag = np.maximum(np.diag(C).copy(), 1.0)
Cn = (C / diag[:, None]).astype(np.float32)        # Cn[a, b] = P(b | a)
np.fill_diagonal(Cn, 0.0)                          # self term adds nothing new
del M, C

nh = pl.read_parquet(OUT / "nbr_hub_train.parquet")[
    [f"nh_{i}" for i in range(N_TRK)]].to_numpy().astype(np.float32)
nh_t = pl.read_parquet(OUT / "nbr_hub_target.parquet")[
    [f"nh_{i}" for i in range(N_TRK)]].to_numpy().astype(np.float32)
NH = np.vstack([nh, nh_t])
del nh, nh_t
NH /= np.maximum(NH.sum(1, keepdims=True), 1.0)    # same l1 the pipelines use
save(NH @ Cn, "tc", "trk_cooc")
del NH

# =============================================================== 4. tok_pop
dom = pl.read_parquet(IN / "domains.parquet").with_columns(
    pl.col("domain_id").cast(pl.Int32))
TOKRE = r"[a-z]{3,}"


def tokenise(df):
    """Alphabetic runs of >=3 chars, TLD dropped (tld_pop already covers it)."""
    return df.with_columns(
        pl.col("domain").str.to_lowercase().str.replace(r"\.[a-z]+$", "")
        .str.extract_all(TOKRE).alias("tok"))


seed_tok = tokenise(dom.filter(pl.col("domain_id").is_in(seed))) \
    .select(["domain_id", "tok"])
seed_vocab = seed_tok.explode("tok").drop_nulls("tok")["tok"].unique()
log("distinct tokens in seed hostnames:", seed_vocab.len())

# document frequency over OUT-OF-SAMPLE tracked domains only
out_ids = np.setdiff1d(tracked["domain_id"].to_numpy(), train_ids)
out_tok = tokenise(dom.filter(pl.col("domain_id").is_in(pl.Series(out_ids).implode()))) \
    .select(["domain_id", "tok"]).explode("tok").drop_nulls("tok")
out_tok = out_tok.filter(pl.col("tok").is_in(seed_vocab.implode()))
df_tok = out_tok.group_by("tok").len().rename({"len": "df"}) \
    .filter(pl.col("df") >= MIN_TOK_DF).sort("df", descending=True).head(MAX_VOCAB)
log("kept vocabulary:", df_tok.height)

vocab_df = df_tok.select(["tok", "df"]).with_row_index("ti")
n_vocab = vocab_df.height
df_arr = vocab_df["df"].to_numpy().astype(np.float64)

# P(tracker | token), EMPIRICAL-BAYES SMOOTHED toward the global prior.
# Without this, dropping MIN_TOK_DF to 20 (needed for coverage -- at df>=100 only
# 19% of rows had any in-vocabulary token, since most hostnames are brand-like)
# would be actively harmful: a 20-domain token yields a ~40-observation profile,
# and the IDF weighting below gives exactly those rare tokens the LARGEST weight.
# Shrinking by (count + a*prior)/(df + a) makes a thin token degrade gracefully
# to the prior instead of injecting noise, which is the standard fix for
# high-cardinality target encoding.
pairs = (out_tok.join(vocab_df.select(["tok", "ti"]), on="tok", how="inner")
         .join(tg_sorted, on="domain_id", how="inner"))
log("(token, tracker) observations:", pairs.height, " vocab:", n_vocab)
TP = np.bincount(pairs["ti"].to_numpy().astype(np.int64) * N_TRK
                 + pairs["tracker_id"].to_numpy(),
                 minlength=n_vocab * N_TRK).reshape(n_vocab, N_TRK).astype(np.float64)
out_ids_set = pl.Series(out_ids).implode()
prior = (np.bincount(tg_sorted.filter(pl.col("domain_id").is_in(out_ids_set))
                     ["tracker_id"].to_numpy(), minlength=N_TRK)
         .astype(np.float64) / float(len(out_ids)))
TP = ((TP + TOK_ALPHA * prior[None, :])
      / (df_arr[:, None] + TOK_ALPHA)).astype(np.float32)
del pairs, out_tok

# IDF-weighted mean of the profiles of each seed domain's own tokens
N_OUT = float(len(out_ids))
idf = np.log(N_OUT / df_arr).astype(np.float32)
se = (seed_tok.explode("tok").drop_nulls("tok")
      .join(vocab_df.select(["tok", "ti"]), on="tok", how="inner"))
se_row = row_of[se["domain_id"].to_numpy()]
se_tok = se["ti"].to_numpy().astype(np.int64)
log("(seed row, token) pairs:", len(se_row))

KP = np.zeros((N, N_TRK), dtype=np.float32)
wsum = np.zeros(N, dtype=np.float32)
CH = 2_000_000
for i in range(0, len(se_row), CH):
    r, t = se_row[i:i + CH], se_tok[i:i + CH]
    w = idf[t]
    np.add.at(KP, r, TP[t] * w[:, None])
    np.add.at(wsum, r, w)
KP /= np.maximum(wsum, 1e-6)[:, None]
save(KP, "kp", "tok_pop")
log("seed rows with any token profile:", int((wsum > 0).sum()))
log("DONE")
