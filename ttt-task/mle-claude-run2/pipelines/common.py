"""Plan-building helpers for the TrackTheTrackers workspace (skrub DataOps).

END-TO-END TASK: there is no design matrix in `input/`, only a pile of raw
files. Everything below builds one INSIDE the plan (guide section 15); nothing
is ever cached to disk.

Row design (locked for the workspace)
------------------------------------
One row per (domain, tracker) pair, over the FULL 355-tracker candidate set, for
a deterministic sample of domains::

    SAMPLE: domain_id % 1301 == 7   ->  14,335 domains  x 355  =  5.09M rows
    label : 1 if that domain uses that tracker in tracking_graph_train

Using all 355 candidates means there is no candidate-generation recall ceiling:
the scorer sees every tracker a domain could be given, so recall@10 measured
here is exactly the metric the leaderboard computes. `n_true` (the domain's true
tracker count) rides along in X as the scorer's denominator and is dropped from
the model's features.

CV: GroupKFold(3) on `domain_id`, so all 355 rows of a domain live in one fold.

Metric: recall@10, a custom plan scorer (`SCORER` / `attach_scoring`), locked by
the harness as `plan:recall@10`. Run ml-score WITHOUT --scoring.

LABEL PROVENANCE -- the one invariant of this workspace
------------------------------------------------------
Every feature here is built from labels of *other* domains (a neighbour's
trackers, a TLD's tracker distribution, ...), which is exactly the leakage route
`mark_as_X`/`mark_as_y` do NOT close (guide section 15, pitfall 17): the tracking
graph is a constant to skrub, so a feature that reaches the modelled rows' own
labels leaks identically in every fold and the CV looks clean.

The invariant that closes it, enforced in `load_context`:

    POOL = tracking_graph rows whose domain_id is NOT in the modelled sample.

POOL is provably disjoint from every modelled row, so no walk -- one hop, two
hops, tld aggregate, co-occurrence, anything -- can reach a modelled domain's
own label, and no `d -> n -> d` return path exists. EVERY label-derived
statistic in this file reads POOL and never `labels_all`. This also matches the
real prediction setting: target.tsv domains were removed from the tracking graph
entirely, so at submission time the same POOL-style statistics are what is
available.

Feature blocks
--------------
Each block is its own deferred node taking (ctx, X) and returning a frame with
X's index and only that block's columns, so blocks compose with `concat` and the
graph stays inspectable one meaningful step at a time (guide section 15, rule 3)
instead of collapsing into one opaque build.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import skrub
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

SEED = 42
N_FOLDS = 3
WS_ROOT = Path(__file__).resolve().parent.parent
INPUT = WS_ROOT / "input"

# --- locked row design -----------------------------------------------------
SAMPLE_MOD = 1301      # domain_id % SAMPLE_MOD == SAMPLE_REM defines the sample
SAMPLE_REM = 7
N_TRACKERS = 355
SMOOTH = 20.0          # pseudo-count for every conditional-prior shrinkage
N_JOBS = 16            # LightGBM threads -- see make_model's docstring

ID_COLS = ["domain_id", "tracker_id"]
NON_FEATURE = ["domain_id", "n_true"]   # kept in X for the scorer, never modelled


def make_cv():
    """The workspace's CV splitter -- GroupKFold on domain_id.

    A domain contributes 355 rows (one per candidate tracker). They must not be
    split across folds: recall@10 is a per-domain metric, and a domain with some
    of its candidates in train and the rest in test is both unscoreable and
    leaky. Groups are wired in `load_xy` via mark_as_X(split_kwargs=...), the
    only place they can live.
    """
    return GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# =========================================================================
# Raw tables + the disjoint label pool
# =========================================================================
def load_context(input_dir):
    """Read the raw files and split the tracking graph into SAMPLE vs POOL.

    Returns a dict of polars frames. This is the only node that touches disk.

    `edges` is restricted to link-graph edges with an endpoint in FOCUS (the
    modelled sample plus target.tsv). That is a row filter on a 623M-edge table,
    not a cached feature: it is recomputed in-plan on every fold and depends on
    no labels.
    """
    input_dir = Path(input_dir)

    labels_all = pl.read_parquet(input_dir / "tracking_graph_train.parquet",
                                 columns=["domain_id", "tracker_id"]).with_columns(
        pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int16))

    in_sample = (pl.col("domain_id") % SAMPLE_MOD) == SAMPLE_REM
    sample_labels = labels_all.filter(in_sample)
    # THE invariant: the label pool excludes every modelled domain.
    pool = labels_all.filter(~in_sample)

    sample_ids = sample_labels["domain_id"].unique().sort()
    target_ids = pl.read_csv(input_dir / "target.tsv", separator="\t")["domain_id"].cast(pl.Int32)
    focus = pl.concat([sample_ids, target_ids]).unique()

    lg = pl.read_parquet(input_dir / "link-graph.parquet")
    edges = lg.filter(pl.col("source_domain_id").is_in(focus.implode())
                      | pl.col("target_domain_id").is_in(focus.implode()))

    domains = (pl.read_parquet(input_dir / "domains.parquet")
               .with_columns(pl.col("domain_id").cast(pl.Int32))
               .with_columns(pl.col("domain").str.split(".").list.last().alias("tld")))

    trackers = pl.read_csv(input_dir / "trackers.tsv", separator="\t").with_columns(
        pl.col("tracker_id").cast(pl.Int16))
    fotp = pl.read_csv(input_dir / "freedom-of-the-press.csv", separator="\t")

    # Global link degree of EVERY domain, taken from the full 623M-edge graph,
    # not from `edges`: a neighbour's informativeness depends on how many
    # domains it links to overall, which the FOCUS-restricted view cannot see.
    gdeg = (lg.group_by("source_domain_id").len().rename(
                {"source_domain_id": "domain_id", "len": "gout"})
            .join(lg.group_by("target_domain_id").len().rename(
                {"target_domain_id": "domain_id", "len": "gin"}),
                on="domain_id", how="full", coalesce=True)
            .with_columns(pl.col("gout").fill_null(0), pl.col("gin").fill_null(0))
            .with_columns((pl.col("gout") + pl.col("gin")).alias("gdeg"))
            .select(pl.col("domain_id").cast(pl.Int32), "gdeg"))
    # How many trackers each POOL domain carries -- a 30-tracker portal is far
    # less specific evidence than a domain with one.
    ntrk = pool.group_by("domain_id").len().rename({"len": "ntrk"})

    return {"pool": pool, "sample_labels": sample_labels, "sample_ids": sample_ids,
            "target_ids": target_ids, "edges": edges, "domains": domains,
            "trackers": trackers, "fotp": fotp, "gdeg": gdeg, "ntrk": ntrk,
            "n_pool_domains": pool["domain_id"].n_unique()}


def build_rows(ctx):
    """The design matrix's rows: every (sampled domain, tracker) pair + the label.

    Also carries `n_true`, the domain's true tracker count -- the recall@10
    denominator, read by the scorer and dropped from the model's features.
    """
    ids = ctx["sample_ids"]
    grid = (ids.to_frame("domain_id")
            .join(pl.int_range(N_TRACKERS, eager=True).cast(pl.Int16).to_frame("tracker_id"),
                  how="cross"))
    lab = ctx["sample_labels"].with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    n_true = ctx["sample_labels"].group_by("domain_id").len().rename({"len": "n_true"})
    rows = (grid.join(lab, on=ID_COLS, how="left")
                .with_columns(pl.col("label").fill_null(0))
                .join(n_true, on="domain_id", how="left")
                .with_columns(pl.col("n_true").cast(pl.Int16))
                .sort(ID_COLS))
    return rows.to_pandas()


# =========================================================================
# Feature blocks -- each returns a frame aligned to X's index
# =========================================================================
def _keys(X):
    """X's (domain_id, tracker_id) keys as polars, with X's row order preserved."""
    return pl.DataFrame({"domain_id": X["domain_id"].to_numpy().astype(np.int32),
                         "tracker_id": X["tracker_id"].to_numpy().astype(np.int16)})


def _align(X, keys_with_feats, cols):
    """Attach a joined polars frame back to X's index, in X's row order."""
    out = keys_with_feats.select(cols).to_pandas()
    out.index = X.index
    return out


def _tracker_prior(ctx):
    """P(tracker) over POOL domains -- the backbone every lift is measured against."""
    n = ctx["n_pool_domains"]
    return (ctx["pool"].group_by("tracker_id").len()
            .with_columns((pl.col("len") / n).alias("pri_p"))
            .select("tracker_id", "pri_p"))


def block_prior(ctx, X):
    """Tracker popularity: how often each tracker appears at all (POOL only)."""
    pri = _tracker_prior(ctx).with_columns(
        pl.col("pri_p").log().alias("pri_logp"),
        pl.col("pri_p").rank(descending=True).cast(pl.Float32).alias("pri_rank"))
    j = _keys(X).join(pri, on="tracker_id", how="left")
    return _align(X, j, ["pri_p", "pri_logp", "pri_rank"])


def block_tld(ctx, X):
    """P(tracker | TLD), shrunk toward the global prior, plus the TLD's lift.

    Trackers are strongly regional (a .ru domain and a .de domain do not use the
    same analytics vendors), so this is the cheapest real conditioning available
    and needs no link graph.
    """
    pri = _tracker_prior(ctx)
    dom_tld = ctx["domains"].select("domain_id", "tld")
    pool_tld = ctx["pool"].join(dom_tld, on="domain_id", how="inner")
    tld_n = pool_tld.group_by("tld").agg(pl.col("domain_id").n_unique().alias("tld_n"))
    tld_cnt = pool_tld.group_by(["tld", "tracker_id"]).len().rename({"len": "tld_c"})
    tld_p = (tld_cnt.join(tld_n, on="tld").join(pri, on="tracker_id", how="left")
             .with_columns(((pl.col("tld_c") + SMOOTH * pl.col("pri_p"))
                            / (pl.col("tld_n") + SMOOTH)).alias("tld_p"))
             .select("tld", "tracker_id", "tld_p", "tld_c"))

    j = (_keys(X).join(dom_tld, on="domain_id", how="left")
         .join(tld_n, on="tld", how="left")
         .join(tld_p, on=["tld", "tracker_id"], how="left")
         .join(pri, on="tracker_id", how="left")
         .with_columns(
             pl.col("tld_c").fill_null(0).cast(pl.Float32),
             pl.col("tld_n").fill_null(0).log1p().alias("tld_logn"),
             # unseen (tld, tracker) pairs fall back to the shrunk global prior
             pl.col("tld_p").fill_null(
                 SMOOTH * pl.col("pri_p") / (SMOOTH + pl.col("tld_n").fill_null(0))),
         )
         .with_columns((pl.col("tld_p") / pl.col("pri_p")).log().alias("tld_lift")))
    return _align(X, j, ["tld_p", "tld_lift", "tld_c", "tld_logn"])


def _neighbour_profile(ctx, X, direction):
    """Tracker usage among a domain's link-graph neighbours (POOL labels only).

    direction="out": domains this domain links TO. "in": domains linking to it.
    Returns the joined frame plus the column names it added.
    """
    me, other = (("source_domain_id", "target_domain_id") if direction == "out"
                 else ("target_domain_id", "source_domain_id"))
    pfx = direction
    dom_ids = pl.Series("domain_id",
                        np.unique(X["domain_id"].to_numpy()).astype(np.int32))

    e = (ctx["edges"].filter(pl.col(me).is_in(dom_ids.implode()))
         .select(pl.col(me).cast(pl.Int32).alias("domain_id"),
                 pl.col(other).cast(pl.Int32).alias("nb"))
         .unique())
    deg = e.group_by("domain_id").len().rename({"len": f"{pfx}_deg"})
    lab = e.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    deg_lab = lab.group_by("domain_id").agg(
        pl.col("nb").n_unique().alias(f"{pfx}_deglab"))
    cnt = lab.group_by(["domain_id", "tracker_id"]).len().rename({"len": f"{pfx}_c"})

    pri = _tracker_prior(ctx)
    j = (_keys(X).join(deg, on="domain_id", how="left")
         .join(deg_lab, on="domain_id", how="left")
         .join(cnt, on=["domain_id", "tracker_id"], how="left")
         .join(pri, on="tracker_id", how="left")
         .with_columns(pl.col(f"{pfx}_deg").fill_null(0),
                       pl.col(f"{pfx}_deglab").fill_null(0),
                       pl.col(f"{pfx}_c").fill_null(0))
         .with_columns(
             ((pl.col(f"{pfx}_c") + SMOOTH * pl.col("pri_p"))
              / (pl.col(f"{pfx}_deglab") + SMOOTH)).alias(f"{pfx}_p"),
             pl.col(f"{pfx}_deg").log1p().alias(f"{pfx}_logdeg"),
             pl.col(f"{pfx}_deglab").log1p().alias(f"{pfx}_logdeglab"))
         .with_columns((pl.col(f"{pfx}_p") / pl.col("pri_p")).log().alias(f"{pfx}_lift")))
    cols = [f"{pfx}_c", f"{pfx}_p", f"{pfx}_lift", f"{pfx}_logdeg", f"{pfx}_logdeglab"]
    return j, cols


def block_nbr_out(ctx, X):
    """Trackers used by the domains this domain links to."""
    j, cols = _neighbour_profile(ctx, X, "out")
    return _align(X, j, cols)


def block_nbr_in(ctx, X):
    """Trackers used by the domains that link to this domain."""
    j, cols = _neighbour_profile(ctx, X, "in")
    return _align(X, j, cols)


def _und_neighbours(ctx, X):
    """The domain's neighbourhood as one undirected edge list, flagging direction.

    One row per (domain, neighbour) with `is_out` / `is_in` / `is_rec`, so the
    union view and the reciprocal view can be read off the same table. A mutual
    link is much stronger evidence of a real relationship than a one-way link
    from a directory page, and 11% of the focused edges are reciprocal.
    """
    dom_ids = pl.Series("domain_id",
                        np.unique(X["domain_id"].to_numpy()).astype(np.int32))
    e = ctx["edges"]
    a = (e.filter(pl.col("source_domain_id").is_in(dom_ids.implode()))
         .select(pl.col("source_domain_id").cast(pl.Int32).alias("domain_id"),
                 pl.col("target_domain_id").cast(pl.Int32).alias("nb"),
                 pl.lit(1, dtype=pl.Int8).alias("is_out"),
                 pl.lit(0, dtype=pl.Int8).alias("is_in")))
    b = (e.filter(pl.col("target_domain_id").is_in(dom_ids.implode()))
         .select(pl.col("target_domain_id").cast(pl.Int32).alias("domain_id"),
                 pl.col("source_domain_id").cast(pl.Int32).alias("nb"),
                 pl.lit(0, dtype=pl.Int8).alias("is_out"),
                 pl.lit(1, dtype=pl.Int8).alias("is_in")))
    return (pl.concat([a, b]).group_by(["domain_id", "nb"])
            .agg(pl.col("is_out").max(), pl.col("is_in").max())
            .with_columns(((pl.col("is_out") == 1) & (pl.col("is_in") == 1))
                          .cast(pl.Int8).alias("is_rec")))


def block_nbr_w(ctx, X):
    """Weighted views of the SAME 1-hop neighbourhood the plain profiles use.

    Residual analysis (data_exploration_5.py) found that 66.5% of the pairs
    pipeline_03 misses are already present in the 1-hop neighbourhood -- the
    tracker is there, it just does not rank top-10. 2-hop reaches only 10.7% of
    the misses and co-citation 6.1%, so the payoff is in reading the existing
    neighbourhood better, not in walking further.

    Three weightings, each down-weighting a different kind of uninformative
    neighbour:
      * Adamic-Adar, 1/log(2+gdeg): a hub that links to 50k domains says almost
        nothing about any one of them.
      * specificity, 1/log(2+ntrk): a portal carrying 30 trackers is weak
        evidence for any single one.
      * reciprocity: mutual links only.
    Plus the undirected union (better coverage than either direction alone) and
    the same-TLD neighbourhood (a German site's German neighbours are the ones
    that share its vendors).
    """
    pri = _tracker_prior(ctx)
    nb = _und_neighbours(ctx, X)
    nb = (nb.join(ctx["gdeg"], left_on="nb", right_on="domain_id", how="left")
            .with_columns(pl.col("gdeg").fill_null(0))
            .join(ctx["ntrk"], left_on="nb", right_on="domain_id", how="left")
            .with_columns(pl.col("ntrk").fill_null(0))
            .with_columns(
                (1.0 / (2.0 + pl.col("gdeg")).log()).alias("w_aa"),
                (1.0 / (2.0 + pl.col("ntrk")).log()).alias("w_sp")))

    dom_tld = ctx["domains"].select("domain_id", "tld")
    me_tld = (pl.DataFrame({"domain_id": np.unique(X["domain_id"].to_numpy()).astype(np.int32)})
              .join(dom_tld, on="domain_id", how="left"))
    nb = (nb.join(me_tld.rename({"tld": "my_tld"}), on="domain_id", how="left")
            .join(dom_tld.rename({"domain_id": "nb", "tld": "nb_tld"}), on="nb", how="left")
            .with_columns((pl.col("my_tld") == pl.col("nb_tld")).cast(pl.Int8).alias("same_tld")))

    # denominators: totals over the neighbours that actually carry labels
    lab = nb.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    tot = lab.group_by("domain_id").agg(
        pl.col("nb").n_unique().alias("u_n"),
        pl.col("w_aa").sum().alias("aa_tot"),
        pl.col("w_sp").sum().alias("sp_tot"),
        pl.col("is_rec").sum().alias("rec_n"),
        pl.col("same_tld").sum().alias("stld_n"))
    num = lab.group_by(["domain_id", "tracker_id"]).agg(
        pl.col("nb").n_unique().alias("u_c"),
        pl.col("w_aa").sum().alias("aa_c"),
        pl.col("w_sp").sum().alias("sp_c"),
        pl.col("is_rec").sum().alias("rec_c"),
        pl.col("same_tld").sum().alias("stld_c"))

    j = (_keys(X).join(tot, on="domain_id", how="left")
         .join(num, on=["domain_id", "tracker_id"], how="left")
         .join(pri, on="tracker_id", how="left"))
    zero = ["u_n", "aa_tot", "sp_tot", "rec_n", "stld_n",
            "u_c", "aa_c", "sp_c", "rec_c", "stld_c"]
    j = j.with_columns([pl.col(c).fill_null(0).cast(pl.Float64) for c in zero])
    out = []
    for c, n, name in [("u_c", "u_n", "u"), ("aa_c", "aa_tot", "aa"),
                       ("sp_c", "sp_tot", "sp"), ("rec_c", "rec_n", "rec"),
                       ("stld_c", "stld_n", "stld")]:
        j = j.with_columns(((pl.col(c) + SMOOTH * pl.col("pri_p"))
                            / (pl.col(n) + SMOOTH)).alias(f"{name}_p"))
        j = j.with_columns((pl.col(f"{name}_p") / pl.col("pri_p")).log().alias(f"{name}_lift"))
        out += [c, f"{name}_p", f"{name}_lift"]
    j = j.with_columns(pl.col("u_n").log1p().alias("u_logn"))
    out.append("u_logn")
    return _align(X, j, out)


def block_rank(ctx, X):
    """Within-domain relative views of the main scores.

    recall@10 only cares about the ORDER of a domain's 355 candidates, but a
    GBDT sees absolute feature values and has to reconstruct "high for this
    domain" from degree interactions. Handing it the rank and the share of mass
    directly removes that burden. Each source score is recomputed here rather
    than read off another block -- recomputation is expected (guide section 15,
    pitfall 18), and it keeps this block independently ablatable.
    """
    pri = _tracker_prior(ctx)
    keys = _keys(X)

    tld_blk = block_tld(ctx, X)
    out_blk = block_nbr_out(ctx, X)
    in_blk = block_nbr_in(ctx, X)
    j = keys.with_columns(
        pl.Series("tld_p", tld_blk["tld_p"].to_numpy()),
        pl.Series("out_p", out_blk["out_p"].to_numpy()),
        pl.Series("in_p", in_blk["in_p"].to_numpy()),
    ).join(pri, on="tracker_id", how="left")
    j = j.with_columns((pl.col("out_p") * pl.col("in_p") / pl.col("pri_p")).alias("mix_p"))

    cols = []
    for c in ["tld_p", "out_p", "in_p", "mix_p"]:
        j = j.with_columns(
            pl.col(c).rank(method="ordinal", descending=True).over("domain_id")
              .cast(pl.Float32).alias(f"r_{c}"),
            (pl.col(c) / pl.col(c).sum().over("domain_id")).alias(f"sh_{c}"),
            (pl.col(c) / pl.col(c).max().over("domain_id")).alias(f"mx_{c}"))
        cols += [f"r_{c}", f"sh_{c}", f"mx_{c}"]
    return _align(X, j, cols)


HUB_CAP = 1000   # a domain linking to more than this says nothing specific


def _hop_profile(ctx, X, kind):
    """Second-order neighbourhoods, as (domain_id, tracker_id, count) + totals.

    kind="h2": neighbours of neighbours (undirected, hub-capped).
    kind="cc": co-citation siblings -- domains sharing an in-linker with this
               one, which is the classic "same directory / same webring" signal.

    Both are hub-capped: expanding through a domain with >HUB_CAP links pulls in
    an arbitrary slice of the web and drowns the real neighbours. Labels come
    from POOL, so a walk that happens to come back to a modelled domain finds
    nothing -- the return path cannot leak.
    """
    dom_ids = pl.Series("domain_id",
                        np.unique(X["domain_id"].to_numpy()).astype(np.int32))
    e = ctx["edges"]
    und = pl.concat([
        e.select(pl.col("source_domain_id").alias("a"), pl.col("target_domain_id").alias("b")),
        e.select(pl.col("target_domain_id").alias("a"), pl.col("source_domain_id").alias("b")),
    ]).unique()
    if kind == "h2":
        mid = (und.filter(pl.col("a").is_in(dom_ids.implode()))
               .rename({"a": "domain_id", "b": "m"}))
        mdeg = und.group_by("a").len().rename({"a": "m", "len": "mdeg"})
        far = (mid.join(mdeg, on="m").filter(pl.col("mdeg") <= HUB_CAP)
               .join(und.rename({"a": "m", "b": "nb"}), on="m")
               .select("domain_id", "nb").unique()
               .filter(~pl.col("nb").is_in(dom_ids.implode())))
    else:
        inn = e.select(pl.col("source_domain_id").alias("src"),
                       pl.col("target_domain_id").alias("dst"))
        sdeg = inn.group_by("src").len().rename({"len": "sdeg"})
        mine = (inn.filter(pl.col("dst").is_in(dom_ids.implode()))
                .join(sdeg, on="src").filter(pl.col("sdeg") <= HUB_CAP))
        far = (mine.rename({"dst": "domain_id"}).join(inn.rename({"dst": "nb"}), on="src")
               .select("domain_id", "nb").unique())
    lab = far.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    tot = lab.group_by("domain_id").agg(pl.col("nb").n_unique().alias("n"))
    num = lab.group_by(["domain_id", "tracker_id"]).len().rename({"len": "c"})
    return tot, num


def _second_order_block(ctx, X, kind, pfx):
    pri = _tracker_prior(ctx)
    tot, num = _hop_profile(ctx, X, kind)
    j = (_keys(X).join(tot, on="domain_id", how="left")
         .join(num, on=["domain_id", "tracker_id"], how="left")
         .join(pri, on="tracker_id", how="left")
         .with_columns(pl.col("n").fill_null(0).cast(pl.Float64),
                       pl.col("c").fill_null(0).cast(pl.Float64))
         .with_columns(((pl.col("c") + SMOOTH * pl.col("pri_p"))
                        / (pl.col("n") + SMOOTH)).alias(f"{pfx}_p"),
                       pl.col("n").log1p().alias(f"{pfx}_logn"),
                       pl.col("c").alias(f"{pfx}_c"))
         .with_columns((pl.col(f"{pfx}_p") / pl.col("pri_p")).log().alias(f"{pfx}_lift")))
    return _align(X, j, [f"{pfx}_c", f"{pfx}_p", f"{pfx}_lift", f"{pfx}_logn"])


def block_hop2(ctx, X):
    """Neighbours of neighbours, hub-capped. Reaches 10.7% of pipeline_03's misses."""
    return _second_order_block(ctx, X, "h2", "h2")


def block_cocite(ctx, X):
    """Domains sharing an in-linker. Reaches 6.1% of pipeline_03's misses."""
    return _second_order_block(ctx, X, "cc", "cc")


def block_cooc(ctx, X):
    """Smooth the neighbour profile through tracker-tracker co-occurrence.

    Trackers travel in packs -- a site that runs one ad network usually runs its
    usual companions. So even when no neighbour uses tracker t, the neighbours'
    OTHER trackers can vote for it: score(d,t) = sum_s q(d,s) * P(t | s), where
    q is the domain's undirected neighbour profile and P(t|s) is estimated on
    POOL. Exploration round 5 measured this as reaching 13.9% of the misses,
    the largest of the three unused signals.

    The whole thing is one small dense matmul: (n_domains x 355) @ (355 x 355).
    """
    pri = _tracker_prior(ctx).sort("tracker_id")
    pri_v = np.zeros(N_TRACKERS, dtype=np.float64)
    pri_v[pri["tracker_id"].to_numpy()] = pri["pri_p"].to_numpy()

    # P(t | s) on POOL: of the domains carrying s, what share also carry t
    pool = ctx["pool"]
    pair = (pool.join(pool, on="domain_id").group_by(
        ["tracker_id", "tracker_id_right"]).len())
    n_s = pool.group_by("tracker_id").len().rename({"len": "n_s"})
    pair = pair.join(n_s, on="tracker_id").with_columns(
        (pl.col("len") / pl.col("n_s")).alias("pcond"))
    C = np.zeros((N_TRACKERS, N_TRACKERS), dtype=np.float64)
    C[pair["tracker_id"].to_numpy(), pair["tracker_id_right"].to_numpy()] = \
        pair["pcond"].to_numpy()
    np.fill_diagonal(C, 0.0)   # a tracker voting for itself is just u_p again

    # q: the undirected neighbour profile, as a dense (n_domains x 355) matrix
    nb = _und_neighbours(ctx, X)
    lab = nb.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    tot = lab.group_by("domain_id").agg(pl.col("nb").n_unique().alias("n"))
    num = lab.group_by(["domain_id", "tracker_id"]).agg(
        pl.col("nb").n_unique().alias("c"))
    doms = np.unique(X["domain_id"].to_numpy()).astype(np.int32)
    pos = {d: i for i, d in enumerate(doms)}
    Q = np.zeros((len(doms), N_TRACKERS), dtype=np.float64)
    ntot = np.zeros(len(doms), dtype=np.float64)
    ti = np.array([pos[d] for d in tot["domain_id"].to_numpy()])
    ntot[ti] = tot["n"].to_numpy()
    ni = np.array([pos[d] for d in num["domain_id"].to_numpy()])
    Q[ni, num["tracker_id"].to_numpy()] = num["c"].to_numpy()
    Q = Q / np.maximum(ntot, 1.0)[:, None]

    S = Q @ C                                  # co-occurrence-smoothed votes
    co_p = (S + SMOOTH * pri_v[None, :]) / (np.maximum(ntot, 0.0)[:, None] + SMOOTH)

    di = np.array([pos[d] for d in X["domain_id"].to_numpy()])
    tj = X["tracker_id"].to_numpy().astype(np.int64)
    out = pd.DataFrame({
        "co_s": S[di, tj],
        "co_p": co_p[di, tj],
        "co_lift": np.log(co_p[di, tj] / np.maximum(pri_v[tj], 1e-12)),
    }, index=X.index)
    return out


def block_nbrmax(ctx, X):
    """Extremes over the neighbours that use a tracker, not just their sum.

    76.5% of the misses are trackers with a POOL frequency under 1%, and the
    evidence for them is typically ONE neighbour. A sum or a mean buries that
    neighbour among the others; what matters is how good the single best witness
    is. So: the most specific neighbour that uses t (lowest global degree), the
    least cluttered one (fewest trackers), and the count of witnesses that are
    both reciprocal and same-TLD -- the strongest kind of witness there is.
    """
    pri = _tracker_prior(ctx)
    nb = _und_neighbours(ctx, X)
    nb = (nb.join(ctx["gdeg"], left_on="nb", right_on="domain_id", how="left")
            .with_columns(pl.col("gdeg").fill_null(0))
            .join(ctx["ntrk"], left_on="nb", right_on="domain_id", how="left")
            .with_columns(pl.col("ntrk").fill_null(0)))
    dom_tld = ctx["domains"].select("domain_id", "tld")
    me_tld = (pl.DataFrame({"domain_id": np.unique(X["domain_id"].to_numpy()).astype(np.int32)})
              .join(dom_tld, on="domain_id", how="left"))
    nb = (nb.join(me_tld.rename({"tld": "my_tld"}), on="domain_id", how="left")
            .join(dom_tld.rename({"domain_id": "nb", "tld": "nb_tld"}), on="nb", how="left")
            .with_columns((pl.col("my_tld") == pl.col("nb_tld")).cast(pl.Int8).alias("same_tld")))
    lab = nb.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    agg = lab.group_by(["domain_id", "tracker_id"]).agg(
        pl.col("gdeg").min().alias("mn_gdeg"),
        pl.col("ntrk").min().alias("mn_ntrk"),
        (pl.col("gdeg") <= 100).sum().alias("small_c"),
        ((pl.col("is_rec") == 1) & (pl.col("same_tld") == 1)).sum().alias("recstld_c"))
    j = (_keys(X).join(agg, on=["domain_id", "tracker_id"], how="left")
         .join(pri, on="tracker_id", how="left")
         .with_columns(
             pl.col("mn_gdeg").fill_null(-1).cast(pl.Float64).alias("mn_gdeg"),
             pl.col("mn_ntrk").fill_null(-1).cast(pl.Float64).alias("mn_ntrk"),
             pl.col("small_c").fill_null(0).cast(pl.Float64),
             pl.col("recstld_c").fill_null(0).cast(pl.Float64))
         .with_columns(pl.col("mn_gdeg").clip(lower_bound=0).log1p().alias("mn_loggdeg")))
    return _align(X, j, ["mn_loggdeg", "mn_ntrk", "small_c", "recstld_c"])


MIN_TOKEN_DOMAINS = 50   # a token seen on fewer POOL domains is noise


def _tokenize(df, col="domain"):
    """hostname -> its lowercase alphanumeric tokens of length >= 3, TLD dropped.

    "best-kettlebells-for-sale.com" -> ["best", "kettlebells", "for", "sale"].
    The name is the only description of a site's content that exists for every
    domain: url-classification.csv covers just 2.3% of them.
    """
    return (df.with_columns(
        pl.col(col).str.to_lowercase().str.replace(r"\.[a-z]+$", "")
          .str.extract_all(r"[a-z0-9]{3,}").alias("tok"))
        .explode("tok").filter(pl.col("tok").is_not_null()))


def block_host(ctx, X):
    """P(tracker | hostname token), aggregated over the domain's tokens.

    The link graph is exhausted -- exploration round 6 priced 2-hop, co-citation,
    co-occurrence and neighbour-extremes at between -0.0007 and +0.0011 on top of
    pipeline_04, all inside the noise. What no block has used yet is the hostname
    string, which says what the site is ABOUT: a "shop" domain, a "blog" domain
    and a "news" domain buy different trackers, and the name is the only content
    signal available for all 46M domains.

    Built exactly like the TLD block, which is the pattern that worked: a
    conditional prior on POOL, shrunk toward the global prior, summarised per
    domain by the best-evidenced token (max lift) and the mean over tokens.
    """
    pri = _tracker_prior(ctx)
    dom = ctx["domains"].select("domain_id", "domain")

    my_ids = pl.Series("domain_id", np.unique(X["domain_id"].to_numpy()).astype(np.int32))
    my_tok = _tokenize(dom.filter(pl.col("domain_id").is_in(my_ids.implode())))
    wanted = my_tok["tok"].unique()

    # POOL-side statistics, restricted to the tokens the modelled domains use
    pool_tok = (_tokenize(dom.join(ctx["pool"].select("domain_id").unique(), on="domain_id"))
                .filter(pl.col("tok").is_in(wanted.implode())))
    tok_n = pool_tok.group_by("tok").agg(pl.col("domain_id").n_unique().alias("tok_n"))
    tok_n = tok_n.filter(pl.col("tok_n") >= MIN_TOKEN_DOMAINS)
    tok_c = (pool_tok.join(tok_n, on="tok")
             .join(ctx["pool"], on="domain_id")
             .group_by(["tok", "tracker_id"]).len().rename({"len": "tok_c"}))
    tok_p = (tok_c.join(tok_n, on="tok").join(pri, on="tracker_id", how="left")
             .with_columns(((pl.col("tok_c") + SMOOTH * pl.col("pri_p"))
                            / (pl.col("tok_n") + SMOOTH)).alias("p"))
             .with_columns((pl.col("p") / pl.col("pri_p")).log().alias("lift"))
             .select("tok", "tracker_id", "p", "lift", "tok_n"))

    # aggregate over each domain's tokens
    per = (my_tok.select("domain_id", "tok").join(tok_p, on="tok")
           .group_by(["domain_id", "tracker_id"]).agg(
               pl.col("lift").max().alias("hst_maxlift"),
               pl.col("lift").mean().alias("hst_meanlift"),
               pl.col("p").max().alias("hst_maxp"),
               pl.len().alias("hst_ntok")))
    j = (_keys(X).join(per, on=["domain_id", "tracker_id"], how="left")
         .with_columns(pl.col("hst_maxlift").fill_null(0.0),
                       pl.col("hst_meanlift").fill_null(0.0),
                       pl.col("hst_ntok").fill_null(0).cast(pl.Float64))
         .join(pri, on="tracker_id", how="left")
         .with_columns(pl.col("hst_maxp").fill_null(pl.col("pri_p"))))
    return _align(X, j, ["hst_maxlift", "hst_meanlift", "hst_maxp", "hst_ntok"])


def block_content(ctx, X):
    """P(tracker | content category) + the TLD's press-freedom score.

    url-classification.csv reaches only 2.3% of domains, equally in the modelled
    sample and in target.tsv, so this can help a small slice at best; it is here
    because it is the only human-labelled description of what a site contains.
    freedom-of-the-press joins by TLD and lets rare TLDs borrow from countries
    with a similar regime rather than falling straight back to the global prior.
    """
    pri = _tracker_prior(ctx)
    uc = pl.read_csv(INPUT / "url-classification.csv")
    host = (uc.with_columns(
        pl.col("url").str.replace(r"^https?://", "").str.replace(r"^www\.", "")
          .str.split("/").list.first().str.split(":").list.first().alias("host"))
        .select("host", "category").unique(subset="host"))
    dom_cat = (ctx["domains"].join(host, left_on="domain", right_on="host", how="inner")
               .select("domain_id", "category"))

    pool_cat = ctx["pool"].join(dom_cat, on="domain_id", how="inner")
    cat_n = pool_cat.group_by("category").agg(pl.col("domain_id").n_unique().alias("cat_n"))
    cat_p = (pool_cat.group_by(["category", "tracker_id"]).len().rename({"len": "cat_c"})
             .join(cat_n, on="category").join(pri, on="tracker_id", how="left")
             .with_columns(((pl.col("cat_c") + SMOOTH * pl.col("pri_p"))
                            / (pl.col("cat_n") + SMOOTH)).alias("cat_p"))
             .with_columns((pl.col("cat_p") / pl.col("pri_p")).log().alias("cat_lift"))
             .select("category", "tracker_id", "cat_p", "cat_lift"))

    fotp = ctx["fotp"].select(pl.col("tld"), pl.col("freedom_of_the_press").alias("fotp"))
    dom_tld = ctx["domains"].select("domain_id", "tld")
    j = (_keys(X).join(dom_cat, on="domain_id", how="left")
         .join(cat_p, on=["category", "tracker_id"], how="left")
         .join(dom_tld, on="domain_id", how="left")
         .join(fotp, on="tld", how="left")
         .join(pri, on="tracker_id", how="left")
         .with_columns(pl.col("cat_lift").fill_null(0.0),
                       pl.col("fotp").fill_null(-1).cast(pl.Float64),
                       pl.col("category").is_not_null().cast(pl.Int8)
                         .cast(pl.Float64).alias("has_cat"))
         .with_columns(pl.col("cat_p").fill_null(pl.col("pri_p"))))
    return _align(X, j, ["cat_p", "cat_lift", "has_cat", "fotp"])


META_GROUPS = ["company", "brand", "category", "country"]


def block_meta(ctx, X):
    """Pool evidence across trackers that belong together (trackers.tsv).

    76.5% of pipeline_03's misses are trackers appearing on under 1% of domains,
    and with ~28k positives across 355 trackers a rare one has only a handful of
    training examples. But the 355 trackers are not independent: they belong to
    355 -> ~150 companies, a few brands and 5 categories. If a domain's
    neighbours all run Google products, an unseen Google tracker is a much better
    bet than its own frequency suggests.

    So for each candidate tracker this adds the neighbourhood's mass on the
    tracker's company / brand / category / country, ALWAYS EXCLUDING THE
    CANDIDATE'S OWN CONTRIBUTION -- otherwise the block would just restate u_p
    with extra steps, and for a single-tracker company it would restate it
    exactly. The group's own prior mass is subtracted the same way, so the lift
    is measured against the right reference.

    The tracker's category and country also go in as plain codes, giving the
    trees something to split on that is shared by many trackers at once.
    """
    pri = _tracker_prior(ctx)
    trk = ctx["trackers"].select(["tracker_id"] + META_GROUPS)

    nb = _und_neighbours(ctx, X)
    lab = nb.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    tot = lab.group_by("domain_id").agg(pl.col("nb").n_unique().alias("u_n"))
    num = (lab.group_by(["domain_id", "tracker_id"])
           .agg(pl.col("nb").n_unique().alias("c")))
    num_m = num.join(trk, on="tracker_id", how="left")

    j = _keys(X).join(trk, on="tracker_id", how="left").join(pri, on="tracker_id", how="left")
    j = (j.join(num.rename({"c": "own_c"}), on=["domain_id", "tracker_id"], how="left")
           .join(tot, on="domain_id", how="left")
           .with_columns(pl.col("own_c").fill_null(0).cast(pl.Float64),
                         pl.col("u_n").fill_null(0).cast(pl.Float64)))

    cols = []
    for g in META_GROUPS:
        gmass = num_m.group_by(["domain_id", g]).agg(pl.col("c").sum().alias(f"{g}_mass"))
        gpri = (pri.join(trk, on="tracker_id", how="left")
                .group_by(g).agg(pl.col("pri_p").sum().alias(f"{g}_pri")))
        j = (j.join(gmass, on=["domain_id", g], how="left")
               .join(gpri, on=g, how="left")
               .with_columns(pl.col(f"{g}_mass").fill_null(0).cast(pl.Float64),
                             pl.col(f"{g}_pri").fill_null(0.0)))
        # drop the candidate's own contribution from both the mass and the prior
        j = j.with_columns(
            (pl.col(f"{g}_mass") - pl.col("own_c")).clip(lower_bound=0).alias(f"{g}_oth"),
            (pl.col(f"{g}_pri") - pl.col("pri_p")).clip(lower_bound=1e-9).alias(f"{g}_opri"))
        j = j.with_columns(
            ((pl.col(f"{g}_oth") + SMOOTH * pl.col(f"{g}_opri"))
             / (pl.col("u_n") + SMOOTH)).alias(f"m_{g}_p"))
        j = j.with_columns(
            (pl.col(f"m_{g}_p") / pl.col(f"{g}_opri")).log().alias(f"m_{g}_lift"))
        cols += [f"m_{g}_p", f"m_{g}_lift"]

    for g in ["category", "country"]:
        # A deterministic code, NOT pl.Categorical.to_physical(): physical codes
        # are assigned in order of first appearance, so a train fold and a test
        # fold with different row orders would encode the same country as
        # different numbers and the model would read garbage at predict time.
        # Ranking the sorted distinct values of trackers.tsv is fold-independent.
        codes = (trk.select(g).unique().sort(g).with_row_index(f"code_{g}")
                 .with_columns(pl.col(f"code_{g}").cast(pl.Float64)))
        j = j.join(codes, on=g, how="left").with_columns(
            pl.col(f"code_{g}").fill_null(-1.0))
        cols.append(f"code_{g}")
    return _align(X, j, cols)


BLOCKS = {
    "prior": block_prior,
    "tld": block_tld,
    "nbr_out": block_nbr_out,
    "nbr_in": block_nbr_in,
    "nbr_w": block_nbr_w,
    "rank": block_rank,
    "hop2": block_hop2,
    "cocite": block_cocite,
    "cooc": block_cooc,
    "nbrmax": block_nbrmax,
    "host": block_host,
    "content": block_content,
    "meta": block_meta,
}


# =========================================================================
# Plan assembly
# =========================================================================
def load_context_op():
    """The single recorded read of input/ -- the root of every plan here."""
    return skrub.as_data_op(str(INPUT)).skb.apply_func(load_context)


def load_xy(subsample=None):
    """Recorded build of the (domain, tracker) row table + the marks.

    Returns (ctx, X, y). Marks come as early as the rows and the raw label
    exist; every feature is built downstream of them, so the CV re-runs the
    whole construction per fold.
    """
    ctx = load_context_op()
    rows = skrub.deferred(build_rows)(ctx)
    if subsample:
        rows = rows.skb.subsample(n=subsample)
    y = rows["label"].skb.mark_as_y()
    X = rows.drop(columns=["label"]).skb.mark_as_X(
        cv=make_cv(), split_kwargs={"groups": rows["domain_id"]})
    return ctx, X, y


def features(ctx, X, blocks=("prior",)):
    """Concatenate the requested feature blocks onto the modelled columns.

    `tracker_id` stays as a feature so the model can learn per-tracker offsets;
    `domain_id` and `n_true` are dropped (the scorer reads them off X instead).
    """
    parts = [skrub.deferred(BLOCKS[b])(ctx, X) for b in blocks]
    base = X.drop(columns=NON_FEATURE)
    return base.skb.concat(parts, axis=1)


# =========================================================================
# The workspace metric: recall@10
# =========================================================================
def _scores_from(estimator, X):
    """Per-row ranking score, whatever kind of estimator the plan ended with."""
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        return np.asarray(p)[:, 1] if np.ndim(p) == 2 and np.shape(p)[1] == 2 else np.ravel(p)
    if hasattr(estimator, "decision_function"):
        return np.ravel(estimator.decision_function(X))
    return np.ravel(estimator.predict(X))


def recall_at_10(estimator, X, y):
    """Mean over domains of |top-10 predicted trackers & true trackers| / n_true.

    A bare callable scorer (estimator, X, y) rather than `make_scorer`, because
    the metric needs the per-row grouping key: sklearn hands the scorer the test
    fold's marked X, which still carries `domain_id` and `n_true` even though
    the model never saw them.
    """
    s = _scores_from(estimator, X)
    df = pl.DataFrame({
        "domain_id": np.asarray(X["domain_id"]).astype(np.int64),
        "tracker_id": np.asarray(X["tracker_id"]).astype(np.int64),
        "n_true": np.asarray(X["n_true"]).astype(np.float64),
        "y": np.asarray(y).astype(np.float64),
        "s": np.asarray(s, dtype=np.float64),
    })
    top = (df.sort(["domain_id", "s", "tracker_id"], descending=[False, True, False])
             .group_by("domain_id", maintain_order=True).head(10))
    per_dom = top.group_by("domain_id").agg(
        (pl.col("y").sum() / pl.col("n_true").first()).alias("r"))
    return float(per_dom["r"].mean())


SCORER = recall_at_10
SCORER_NAME = "recall@10"


def attach_scoring(pred):
    """Declare the workspace metric on the final node (guide section 9).

    skrub honours a plan scorer only when make_grid_search gets scoring=None,
    which the harness does as soon as it sees this node -- so run ml-score
    WITHOUT --scoring. Every pipeline in this workspace must end with this call.
    """
    return pred.skb.with_scoring(SCORER, name=SCORER_NAME)


# =========================================================================
# The model: LightGBM, with the divergence guard this design matrix needs
# =========================================================================
def make_model(**kw):
    """Binary LightGBM over the (domain, tracker) rows; `predict_proba` ranks.

    WHY min_child_weight IS NOT A FREE HYPERPARAMETER HERE
    ------------------------------------------------------
    The first attempt at pipeline_01 used HistGradientBoostingClassifier with
    default settings and scored 0.329 with fold scores [0.225, 0.004, 0.757] --
    impossible for a plan whose every feature is a function of tracker_id alone
    (it can only express ONE global ranking, so all folds must score the same).

    The cause is Newton-step divergence, not a data bug, and it reproduces in
    twenty lines of numpy with no skrub involved. The base rate here is 0.56%
    positives, so the initial raw prediction is logit(0.0056) ~ -5.2. For the
    most popular tracker (id 129, present on 59% of domains) a leaf's Newton
    step is -G/H with H = sum p(1-p) ~ 0 at that starting point: the step is
    huge, p overshoots to ~1, the hessian collapses again in the other
    direction, and the leaf oscillates out to raw = -2091, i.e. probability
    exactly 0 for the single most useful tracker. Missing tracker 129 alone
    costs ~0.5 recall. HistGB and LightGBM at their defaults diverge
    identically; it is a property of the loss surface, not of either library.

    A floor on the per-leaf hessian sum is the fix that addresses the cause:
    min_child_weight=100 holds the model at the true rates (0.602 / 0.219 /
    0.122 / 0.001 against true 0.600 / 0.230 / 0.120 / 0.001) with no other
    change. A smaller learning rate with many more trees also converges, but it
    treats the symptom and costs fit time. Keep the floor; tune around it.

    n_jobs=N_JOBS is not a tuning knob either. LightGBM parallelises histogram
    building over FEATURES, and these plans have a handful of them; with
    n_jobs=-1 the other ~60 OpenMP threads busy-wait. Measured on pipeline_01's
    4-feature matrix: n_jobs=64 -> 343s per fit, n_jobs=16 -> 7.1s, same score
    to four decimals. A fixed, modest thread count is the difference between a
    runnable workspace and a 20-minute baseline.
    """
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  min_child_weight=100.0, subsample_freq=0, colsample_bytree=1.0,
                  random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


class LambdaRanker(RegressorMixin, BaseEstimator):
    """LGBMRanker over the (domain, tracker) rows, grouped by a column of X.

    recall@10 is a per-domain ranking metric, so the natural objective is
    lambdarank truncated at 10 rather than pointwise log-loss: it only ever
    compares candidates WITHIN a domain, which is exactly the comparison the
    metric makes, and it spends no capacity on calibrating probabilities across
    domains.

    LGBMRanker needs its rows grouped, as consecutive blocks plus a `group`
    array of block sizes. The group key cannot be passed as a fit parameter
    through a DataOps plan, so it rides in X as `group_col` and is removed from
    the features here -- the same trick `n_true` uses for the scorer. fit sorts
    by the key (a CV fold hands over whatever row order it likes) and predict
    simply drops it.

    Exposed as a regressor: the plan's output is a ranking score, not a
    probability, and `common._scores_from` falls through to `.predict` for it.
    """

    def __init__(self, group_col="domain_id", **params):
        self.group_col = group_col
        self.params = params

    def fit(self, X, y):
        g = np.asarray(X[self.group_col])
        order = np.argsort(g, kind="stable")
        Xs, ys = X.iloc[order], np.asarray(y)[order]
        _, sizes = np.unique(g[order], return_counts=True)
        params = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                      min_child_weight=100.0, random_state=SEED, n_jobs=N_JOBS,
                      verbose=-1, objective="lambdarank",
                      lambdarank_truncation_level=10)
        params.update(self.params)
        self.model_ = lgb.LGBMRanker(**params).fit(
            Xs.drop(columns=[self.group_col]), ys, group=sizes)
        return self

    def predict(self, X):
        return self.model_.predict(X.drop(columns=[self.group_col]))


class BlendRanker(RegressorMixin, BaseEstimator):
    """Rank-average of the pointwise classifier and the lambdarank ranker.

    pipeline_04 (log-loss) and pipeline_06 (lambdarank) scored 0.87290 and
    0.87253 on identical features -- a dead heat. But a tie in the mean does not
    mean the two models are the same model: one is fitted to make probabilities
    right across all domains, the other to order candidates within a domain, and
    where they disagree they disagree for different reasons.

    Their scores live on incomparable scales (a probability versus an unbounded
    lambdarank score), so they are combined by WITHIN-DOMAIN RANK rather than by
    averaging values -- which is also the only combination recall@10 can even
    see, since the metric reads nothing but the per-domain order.
    """

    def __init__(self, group_col="domain_id", **params):
        self.group_col = group_col
        self.params = params

    def fit(self, X, y):
        feats = X.drop(columns=[self.group_col])
        self.clf_ = make_model(**self.params).fit(feats, y)
        self.rnk_ = LambdaRanker(group_col=self.group_col, **self.params).fit(X, y)
        return self

    def predict(self, X):
        a = self.clf_.predict_proba(X.drop(columns=[self.group_col]))[:, 1]
        b = self.rnk_.predict(X)
        df = pl.DataFrame({"g": np.asarray(X[self.group_col]).astype(np.int64),
                           "a": np.asarray(a, dtype=np.float64),
                           "b": np.asarray(b, dtype=np.float64)})
        df = df.with_columns(
            pl.col("a").rank("average").over("g").alias("ra"),
            pl.col("b").rank("average").over("g").alias("rb"))
        return (df["ra"] + df["rb"]).to_numpy()


def block_trkcode(ctx, X):
    """Deterministic integer codes for a tracker's company / brand / category / country.

    Nothing a tree needs -- `block_meta` already turns these groups into numeric
    masses. This block exists for the neural model, which consumes them as
    EMBEDDING inputs: a rare tracker's score then has a component that comes from
    its company's embedding, trained on every domain that uses any of that
    company's trackers, rather than from its own handful of positives.

    Codes rank the sorted distinct values of trackers.tsv, so they are identical
    in every fold regardless of row order -- never `pl.Categorical.to_physical()`,
    which numbers by order of first appearance.
    """
    trk = ctx["trackers"].select(["tracker_id"] + META_GROUPS)
    j = _keys(X).join(trk, on="tracker_id", how="left")
    cols = []
    for g in META_GROUPS:
        codes = (trk.select(g).unique().sort(g).with_row_index(f"code2_{g}")
                 .with_columns(pl.col(f"code2_{g}").cast(pl.Float64)))
        j = j.join(codes, on=g, how="left").with_columns(
            pl.col(f"code2_{g}").fill_null(-1.0))
        cols.append(f"code2_{g}")
    return _align(X, j, cols)


BLOCKS["trkcode"] = block_trkcode


class _ListNet(nn.Module):
    """Per-candidate scorer: embeddings for the tracker's identity and groups,
    concatenated with the numeric pair features, through an MLP to one logit."""

    def __init__(self, n_numeric, cardinalities, emb_dim, hidden, n_layers, dropout):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        for e in self.embs:
            nn.init.normal_(e.weight, std=0.05)
        d = n_numeric + emb_dim * len(cardinalities)
        layers, prev = [], d
        for _ in range(n_layers):
            layers += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num, cat):
        e = [emb(cat[..., i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat([num] + e, dim=-1)).squeeze(-1)



def _train_listnet(num_cpu, cat_cpu, tgt_cpu, cards, hp, seed, device):
    """Train one _ListNet on `device`. Returns (net, final loss).

    Pulled out of TorchListRanker.fit so that seed-averaging runs the exact same
    training code, not a second copy of it that could drift.
    """
    dev = torch.device(device)
    torch.manual_seed(seed)
    num, cat, tgt = num_cpu.to(dev), cat_cpu.to(dev), tgt_cpu.to(dev)
    net = _ListNet(num.shape[-1], cards, hp["emb_dim"], hp["hidden"],
                   hp["n_layers"], hp["dropout"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    n, bd = num.shape[0], hp["batch_domains"]
    steps = hp["epochs"] * max(1, n // bd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    net.train()
    done = 0
    for _ in range(hp["epochs"]):
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n - bd + 1, bd):
            idx = perm[i:i + bd]
            logits = net(num[idx], cat[idx])
            loss = -(tgt[idx] * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            done += 1
            if done < steps:
                sched.step()
    return net, float(loss.detach())


class TorchListRanker(RegressorMixin, BaseEstimator):
    """GPU listwise ranker over each domain's full 355-candidate list.

    Why this model family, after the GBDT plateaued at ~0.876:

    * EMBEDDINGS SHARE STRENGTH ACROSS TRACKERS. The diagnosed bottleneck is
      rare trackers -- recovered 38.5% of the time against 94.6% for common
      ones, carrying 76.5% of all misses, because ~28k positives spread over
      355 trackers leaves a rare one with almost no examples. `block_meta`
      attacked this by hand-building company/brand/category masses. A learned
      embedding attacks it structurally: a rare tracker's logit is partly its
      company's embedding, which every domain using ANY of that company's
      trackers has trained.
    * A LISTWISE LOSS MATCHES THE METRIC. Each domain contributes exactly one
      list of 355 candidates, so the loss is a softmax over the whole list with
      soft targets y/n_true -- put probability mass on the true trackers, in
      competition with the other 354. That is much closer to recall@10 than
      pointwise log-loss, and unlike lambdarank (pipeline_06) it sees the entire
      candidate list at once rather than sampled pairs.

    Implementation notes. Rows are sorted by (domain, tracker) and reshaped to
    (n_domains, 355, n_features); the uniform group size is guaranteed by the row
    design and asserted, so a future change to the candidate set fails loudly
    instead of scoring garbage. Standardisation is fitted inside fit() on that
    fold's rows only. The whole training fold is ~0.7 GB of float32, so it lives
    on the GPU and minibatching is over domains.
    """

    def __init__(self, group_col="domain_id", emb_cols=("tracker_id", "code2_company",
                                                        "code2_brand", "code2_category",
                                                        "code2_country"),
                 emb_dim=32, hidden=256, n_layers=2, dropout=0.1, lr=2e-3,
                 epochs=40, batch_domains=512, weight_decay=1e-5,
                 device="cuda", random_state=SEED):
        self.group_col = group_col
        self.emb_cols = emb_cols
        self.emb_dim = emb_dim
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_domains = batch_domains
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state

    # --- shaping -----------------------------------------------------------
    def _split_cols(self, X):
        emb = [c for c in self.emb_cols if c in X.columns]
        num = [c for c in X.columns if c not in emb and c != self.group_col]
        return emb, num

    def _reshape(self, X):
        """Sort to (domain, tracker) order and fold into per-domain lists.

        Returns the sorted order (to map predictions back) and the list length.
        """
        g = np.asarray(X[self.group_col])
        t = np.asarray(X["tracker_id"]) if "tracker_id" in X.columns else np.zeros(len(g))
        order = np.lexsort((t, g))
        sizes = np.unique(g[order], return_counts=True)[1]
        k = int(sizes[0])
        if not np.all(sizes == k):
            raise ValueError(
                f"TorchListRanker needs one equal-length candidate list per domain; "
                f"got sizes {sizes.min()}..{sizes.max()}. The workspace row design "
                f"gives every domain all {N_TRACKERS} candidates.")
        return order, k

    def _tensors(self, X, order, k, fit):
        emb, num = self._split_cols(X)
        A = np.nan_to_num(np.asarray(X[num], dtype=np.float64)[order],
                          nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.mu_, self.sd_ = A.mean(0), A.std(0)
            self.sd_[self.sd_ < 1e-9] = 1.0
            self.num_cols_, self.emb_cols_ = num, emb
        A = np.clip((A - self.mu_) / self.sd_, -10, 10).astype(np.float32)
        C = np.asarray(X[emb], dtype=np.int64)[order]
        C = np.clip(C, 0, None)
        if fit:
            self.cards_ = [int(C[:, i].max()) + 2 for i in range(C.shape[1])]
        C = np.minimum(C, np.array(self.cards_)[None, :] - 1)
        d = len(A) // k
        return (torch.from_numpy(A).view(d, k, -1),
                torch.from_numpy(C).view(d, k, -1))

    # --- sklearn API -------------------------------------------------------
    def fit(self, X, y):
        dev = torch.device(self.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.random_state)
        order, k = self._reshape(X)
        num, cat = self._tensors(X, order, k, fit=True)
        yy = torch.from_numpy(np.asarray(y, dtype=np.float32)[order]).view(-1, k)
        # soft listwise target: the domain's probability mass split over its
        # true trackers (n_true >= 1 by construction, but guard anyway)
        tgt = yy / yy.sum(1, keepdim=True).clamp(min=1.0)

        self.net_, self.train_loss_ = _train_listnet(
            num, cat, tgt, self.cards_, self._hp(), self.random_state, dev)
        return self

    def _hp(self):
        return dict(emb_dim=self.emb_dim, hidden=self.hidden, n_layers=self.n_layers,
                    dropout=self.dropout, lr=self.lr, epochs=self.epochs,
                    batch_domains=self.batch_domains, weight_decay=self.weight_decay)

    def predict(self, X):
        dev = next(self.net_.parameters()).device
        order, k = self._reshape(X)
        num, cat = self._tensors(X, order, k, fit=False)
        self.net_.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, num.shape[0], 1024):
                outs.append(self.net_(num[i:i + 1024].to(dev),
                                      cat[i:i + 1024].to(dev)).float().cpu())
        s = torch.cat(outs).view(-1).numpy()
        out = np.empty(len(s), dtype=np.float64)
        out[order] = s          # undo the sort so scores line up with X's rows
        return out


class HybridRanker(RegressorMixin, BaseEstimator):
    """Rank-average of the GPU listwise ranker and the LightGBM classifier.

    pipeline_10's blend of log-loss and lambdarank added nothing, because both
    were the same trees on the same features and made the same mistakes. This
    blend is different in kind: on fold 0 the two models' within-domain rankings
    correlate at only 0.754, so they genuinely disagree about which candidates
    belong in the top 10, and the disagreement is informative --

        GBDT only   0.8740
        w_nn = 0.5  0.8791
        w_nn = 0.7  0.8801
        NN only     0.8789

    The weight is set to 2/3 rather than the grid's argmax (0.7): the curve is
    flat between 0.5 and 0.7, so "weight the stronger model twice as much" is a
    defensible choice, while reading 0.7 off a single fold's maximum would be
    tuning on noise.

    Combining by rank, not by value, for the same reason as BlendRanker -- a
    softmax logit and a probability are on incomparable scales, and recall@10
    reads nothing but the per-domain order anyway.
    """

    def __init__(self, group_col="domain_id", nn_weight=2 / 3, epochs=40,
                 n_estimators=800, n_seeds=1, random_state=SEED):
        self.group_col = group_col
        self.nn_weight = nn_weight
        self.epochs = epochs
        self.n_estimators = n_estimators
        self.n_seeds = n_seeds
        self.random_state = random_state

    def fit(self, X, y):
        kw = dict(group_col=self.group_col, epochs=self.epochs,
                  random_state=self.random_state)
        self.nn_ = (TorchListRanker(**kw) if self.n_seeds == 1
                    else SeedAveragedListRanker(n_seeds=self.n_seeds, **kw)).fit(X, y)
        self.gb_ = make_model(n_estimators=self.n_estimators).fit(
            X.drop(columns=[self.group_col]), y)
        return self

    def predict(self, X):
        a = self.nn_.predict(X)
        b = self.gb_.predict_proba(X.drop(columns=[self.group_col]))[:, 1]
        df = pl.DataFrame({"g": np.asarray(X[self.group_col]).astype(np.int64),
                           "a": np.asarray(a, dtype=np.float64),
                           "b": np.asarray(b, dtype=np.float64)})
        df = df.with_columns(pl.col("a").rank("average").over("g").alias("ra"),
                             pl.col("b").rank("average").over("g").alias("rb"))
        w = self.nn_weight
        return (w * df["ra"].to_numpy() + (1 - w) * df["rb"].to_numpy())


def available_devices():
    """Every visible CUDA device, or ["cpu"] if there are none."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [f"cuda:{i}" for i in range(n)] or ["cpu"]


class SeedAveragedListRanker(TorchListRanker):
    """TorchListRanker fitted with several seeds in parallel, one per GPU.

    A single fit uses ~20s and a fraction of one A100 out of the four on this
    box, so the cheapest remaining gain is not a bigger network but more of the
    same network. A listwise softmax on 9.5k training lists is a
    high-variance estimator: the embedding table alone has 355 + ~150 + ... rows
    whose gradients come from a long-tailed label distribution, so different
    seeds land on visibly different rankings even at identical loss.

    Members are averaged as PER-DOMAIN SOFTMAX PROBABILITIES, not ranks. All
    members share one architecture and one loss, so their logits are already on
    a comparable scale and the softmax average is the proper ensemble of the
    distributions the model was actually trained to produce -- averaging ranks
    would throw away the confidence that makes a 2-vs-3 ordering different from
    a 2-vs-200 one. (Ranks are still the right tool for the NN-vs-GBDT blend in
    HybridRanker, where the two scales are genuinely incomparable.)

    Parallelism is threads, not processes: the training loop is almost entirely
    CUDA kernel launches, which release the GIL, and each thread owns a distinct
    device. Shaping and standardisation happen ONCE on the CPU and the resulting
    tensors are copied to each card (~0.7 GB each), so the per-seed cost is just
    the training itself.
    """

    def __init__(self, n_seeds=8, devices=None, **kw):
        super().__init__(**kw)
        self.n_seeds = n_seeds
        self.devices = devices

    def fit(self, X, y):
        devs = list(self.devices) if self.devices else available_devices()
        order, k = self._reshape(X)
        num, cat = self._tensors(X, order, k, fit=True)   # once, on CPU
        yy = torch.from_numpy(np.asarray(y, dtype=np.float32)[order]).view(-1, k)
        tgt = yy / yy.sum(1, keepdim=True).clamp(min=1.0)

        hp, losses = self._hp(), {}
        jobs = [(self.random_state + 1000 * i, devs[i % len(devs)])
                for i in range(self.n_seeds)]

        def run(job):
            seed, dev = job
            net, loss = _train_listnet(num, cat, tgt, self.cards_, hp, seed, dev)
            losses[seed] = loss
            return net, dev

        with ThreadPoolExecutor(max_workers=len(devs)) as ex:
            self.members_ = list(ex.map(run, jobs))
        self.train_loss_ = float(np.mean(list(losses.values())))
        self.devices_used_ = sorted({d for _, d in self.members_})
        return self

    def predict(self, X):
        order, k = self._reshape(X)
        num, cat = self._tensors(X, order, k, fit=False)
        probs = np.zeros((num.shape[0], k), dtype=np.float64)
        for net, dev in self.members_:
            net.eval()
            dv = torch.device(dev)
            with torch.no_grad():
                for i in range(0, num.shape[0], 1024):
                    lg = net(num[i:i + 1024].to(dv), cat[i:i + 1024].to(dv))
                    probs[i:i + 1024] += torch.softmax(lg.float(), dim=-1).cpu().numpy()
        s = (probs / len(self.members_)).reshape(-1)
        out = np.empty(len(s), dtype=np.float64)
        out[order] = s
        return out
