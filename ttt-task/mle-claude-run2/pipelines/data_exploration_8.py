"""Exploration round 8 -- learning curve for the NEURAL model, done properly.

Exploration round 7 measured a training-size curve for the GBDT and found it
flat to slightly negative out to 8x (0.8718 -> 0.8665). That result was never
re-tested for TorchListRanker, whose per-tracker embedding parameters have a far
better reason to want more data than a tree's splits do -- and which trains in
20s, so the experiment is cheap.

TWO FLAWS IN ROUND 7'S DESIGN, BOTH FIXED HERE
----------------------------------------------
Round 7 grew the training set with domains from other `domain_id % 1301`
classes while leaving those domains inside POOL. That is subtly wrong in two
ways, and both push the curve DOWN, which is the direction round 7 reported:

1. LEAKY TRAINING FEATURES. A domain's 1-hop features never read its own label
   (no self-loops), but the TLD prior and the hostname-token prior are POOL
   aggregates that DO include it. For a token seen on the minimum 50 domains, a
   training domain contributes ~2% of its own token statistic -- and contributes
   exactly its own label. The training rows therefore carry slightly optimistic
   features that the test rows do not, so the model learns to over-trust those
   columns and generalises worse. The bias grows with the number of extra
   domains, i.e. exactly along the curve.
2. NO EDGES. `load_context` restricts the link graph to FOCUS = the modelled
   sample plus target.tsv. Extra training domains are outside FOCUS, so every
   neighbour feature they had was ZERO -- the added rows were feature-poor
   relative to the rows the model is scored on.

Here each curve point rebuilds POOL (excluding *every* modelled domain, the
workspace invariant), the focused edge set, and all features for train and test
together, so the two are constructed identically. The test domains are held
fixed as fold 0's test split throughout, and stay outside POOL at every point.

Both models are re-measured, so round 7's GBDT conclusion is re-tested under the
corrected design rather than assumed.
"""
import time

import numpy as np
import pandas as pd
import polars as pl

from common import (BLOCKS, INPUT, N_TRACKERS, SAMPLE_MOD, SAMPLE_REM, ID_COLS,
                    make_cv, make_model, recall_at_10, TorchListRanker)

BLKS = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w", "meta", "host",
        "content", "trkcode")
POINTS = [("1x", []), ("2x", [11]), ("4x", [11, 13, 17]),
          ("8x", [11, 13, 17, 19, 23, 29, 31])]

print("reading raw tables ...")
t0 = time.time()
labels_all = pl.read_parquet(INPUT / "tracking_graph_train.parquet",
                             columns=["domain_id", "tracker_id"]).with_columns(
    pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int16))
lg = pl.read_parquet(INPUT / "link-graph.parquet")
domains = (pl.read_parquet(INPUT / "domains.parquet")
           .with_columns(pl.col("domain_id").cast(pl.Int32))
           .with_columns(pl.col("domain").str.split(".").list.last().alias("tld")))
trackers = pl.read_csv(INPUT / "trackers.tsv", separator="\t").with_columns(
    pl.col("tracker_id").cast(pl.Int16))
fotp = pl.read_csv(INPUT / "freedom-of-the-press.csv", separator="\t")
target_ids = pl.read_csv(INPUT / "target.tsv", separator="\t")["domain_id"].cast(pl.Int32)
gdeg = (lg.group_by("source_domain_id").len().rename(
            {"source_domain_id": "domain_id", "len": "gout"})
        .join(lg.group_by("target_domain_id").len().rename(
            {"target_domain_id": "domain_id", "len": "gin"}),
            on="domain_id", how="full", coalesce=True)
        .with_columns(pl.col("gout").fill_null(0), pl.col("gin").fill_null(0))
        .with_columns((pl.col("gout") + pl.col("gin")).alias("gdeg"))
        .select(pl.col("domain_id").cast(pl.Int32), "gdeg"))
print(f"  {time.time() - t0:.1f}s")

# --- the fixed evaluation set: fold 0's test domains of the base sample ------
base_labels = labels_all.filter((pl.col("domain_id") % SAMPLE_MOD) == SAMPLE_REM)
base_ids = base_labels["domain_id"].unique().sort()
_grid = (base_ids.to_frame("domain_id").join(
    pl.int_range(N_TRACKERS, eager=True).cast(pl.Int16).to_frame("tracker_id"), how="cross")
    .sort(ID_COLS).to_pandas())
tr0, te0 = next(iter(make_cv().split(_grid, _grid["tracker_id"],
                                    groups=_grid["domain_id"])))
TEST_DOMS = np.unique(_grid["domain_id"].to_numpy()[te0])
BASE_TRAIN_DOMS = np.unique(_grid["domain_id"].to_numpy()[tr0])
print(f"fixed test domains: {len(TEST_DOMS):,}   base train domains: {len(BASE_TRAIN_DOMS):,}")


def make_rows(ids):
    """(domain, tracker) grid + label, for an arbitrary set of domain ids."""
    grid = (pl.Series("domain_id", np.asarray(ids, dtype=np.int32)).unique().sort()
            .to_frame("domain_id")
            .join(pl.int_range(N_TRACKERS, eager=True).cast(pl.Int16).to_frame("tracker_id"),
                  how="cross"))
    lab = labels_all.with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    n_true = labels_all.group_by("domain_id").len().rename({"len": "n_true"})
    r = (grid.join(lab, on=ID_COLS, how="left")
         .with_columns(pl.col("label").fill_null(0))
         .join(n_true, on="domain_id", how="left")           # recall@10 denominator
         .with_columns(pl.col("n_true").cast(pl.Int16))
         .sort(ID_COLS).to_pandas())
    return r.drop(columns=["label"]), r["label"]


class _Wrap:
    def __init__(self, model, feats, proba):
        self.m, self.f, self.proba = model, feats, proba

    def predict(self, Xt):
        f = self.f.loc[Xt.index]
        if self.proba:
            return self.m.predict_proba(f.drop(columns=["domain_id"]))[:, 1]
        return self.m.predict(f)


print("\n" + "=" * 78)
print(f"{'size':>5}  {'train domains':>14} {'rows':>12} {'positives':>10}  "
      f"{'NN':>8} {'GBDT':>8}")
print("=" * 78)
for tag, rems in POINTS:
    t0 = time.time()
    extra_ids = (labels_all.filter((pl.col("domain_id") % SAMPLE_MOD).is_in(rems))
                 ["domain_id"].unique() if rems else pl.Series("domain_id", [], dtype=pl.Int32))
    # THE INVARIANT: POOL excludes every modelled domain at this curve point
    modelled = pl.concat([base_ids, extra_ids]).unique()
    pool = labels_all.filter(~pl.col("domain_id").is_in(modelled.implode()))
    focus = pl.concat([modelled, target_ids]).unique()
    edges = lg.filter(pl.col("source_domain_id").is_in(focus.implode())
                      | pl.col("target_domain_id").is_in(focus.implode()))
    ctx = {"pool": pool, "edges": edges, "domains": domains, "trackers": trackers,
           "fotp": fotp, "gdeg": gdeg,
           "ntrk": pool.group_by("domain_id").len().rename({"len": "ntrk"}),
           "n_pool_domains": pool["domain_id"].n_unique()}

    train_ids = np.concatenate([BASE_TRAIN_DOMS, extra_ids.to_numpy()])
    Xtr, ytr = make_rows(train_ids)
    Xte, yte = make_rows(TEST_DOMS)
    ftr = pd.concat([Xtr[["tracker_id", "domain_id"]]]
                    + [BLOCKS[b](ctx, Xtr) for b in BLKS], axis=1)
    fte = pd.concat([Xte[["tracker_id", "domain_id"]]]
                    + [BLOCKS[b](ctx, Xte) for b in BLKS], axis=1)
    build = time.time() - t0

    t0 = time.time()
    nn = TorchListRanker(epochs=40).fit(ftr, ytr)
    s_nn = recall_at_10(_Wrap(nn, fte, False), Xte, yte)
    t_nn = time.time() - t0

    t0 = time.time()
    gb = make_model(n_estimators=800).fit(ftr.drop(columns=["domain_id"]), ytr)
    s_gb = recall_at_10(_Wrap(gb, fte, True), Xte, yte)
    t_gb = time.time() - t0

    print(f"{tag:>5}  {len(train_ids):>14,} {len(Xtr):>12,} {int(ytr.sum()):>10,}  "
          f"{s_nn:>8.4f} {s_gb:>8.4f}   "
          f"(build {build:.0f}s, nn {t_nn:.0f}s, gbdt {t_gb:.0f}s)")

print("=" * 78)
print("round 7 (flawed design, GBDT only): 0.8718 / 0.8690 / 0.8690 / 0.8665")
