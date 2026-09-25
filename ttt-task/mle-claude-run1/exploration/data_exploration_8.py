"""Round 8: LEAKAGE AUDIT of the nbr_2h block (pipeline_07's +0.023 looked wrong).

pipeline_07 gained +0.023 with a fold std of 0.00005 -- an order of magnitude
tighter than every earlier pipeline (0.0003-0.0011). A jump that large that is
also that uniform across folds is the signature of a feature that carries the
answer, not of better evidence.

Suspected mechanism: nbr_2h walks seed -> mid -> mid's tracked neighbours. `mid`
was *found* as a neighbour of the seed, so the edge between them still exists in
mid's own edge list, which means THE SEED ITSELF is one of mid's neighbours. Every
train seed is tracked, so its own tracker set is written back into its own 2-hop
feature. data_exploration_7 filtered self-loops (mid != its neighbour) but never
excluded the seed.

Target rows cannot leak this way (target domains are absent from the tracking
graph, so they are never "tracked" neighbours), which makes it worse than a
mis-scored CV: the model would lean on a feature that is systematically present
in training and absent at submission time.

This script measures the leak directly rather than arguing about it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

WS = Path(__file__).resolve().parent.parent
FEAT, IN = WS / "features", WS / "input"
N_TRK = 355

lab = pd.read_parquet(FEAT / "labels_train.parquet")
Y = lab[[f"y_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.uint8)
n2 = pd.read_parquet(FEAT / "nbr_2h_train.parquet")[
    [f"n2_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
m2 = pd.read_parquet(FEAT / "meta2_train.parquet")
has2 = m2["two_hop_mass"].to_numpy() > 0
print(f"train rows: {len(Y)},  with any 2-hop evidence: {has2.mean():.4f}")

# For rows WITH 2-hop evidence: what fraction of their own true trackers are
# present in their own 2-hop histogram? If the seed leaks into itself this is ~1.
sub_y, sub_n2 = Y[has2], n2[has2]
own_in_2h = ((sub_y == 1) & (sub_n2 > 0)).sum(1) / np.maximum(sub_y.sum(1), 1)
print(f"\nfraction of own true trackers present in own 2-hop histogram: "
      f"{own_in_2h.mean():.4f}")
print(f"  rows where ALL own trackers appear in their 2-hop block: "
      f"{(own_in_2h >= 0.999).mean():.4f}")

# Compare against the same quantity for the 1-hop block, which is leak-free by
# construction (self-loops dropped, a domain cannot be its own neighbour).
no = pd.read_parquet(FEAT / "nbr_out_train.parquet")[
    [f"no_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
ni = pd.read_parquet(FEAT / "nbr_in_train.parquet")[
    [f"ni_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
one_hop = (no + ni)[has2]
own_in_1h = ((sub_y == 1) & (one_hop > 0)).sum(1) / np.maximum(sub_y.sum(1), 1)
print(f"  same for the (leak-free) 1-hop block:              {own_in_1h.mean():.4f}")
del no, ni, one_hop

# Direct structural proof: for a handful of seeds, is the seed itself among the
# tracked neighbours of one of its own low-degree untracked mids?
ids = lab["domain_id"].to_numpy()
probe = ids[has2][:200]
lg = pl.scan_parquet(IN / "link-graph.parquet")
ps = pl.Series(probe.astype(np.int32)).implode()
nb = pl.concat([
    lg.filter(pl.col("source_domain_id").is_in(ps))
      .select(pl.col("source_domain_id").alias("seed"),
              pl.col("target_domain_id").alias("mid")).collect(engine="streaming"),
    lg.filter(pl.col("target_domain_id").is_in(ps))
      .select(pl.col("target_domain_id").alias("seed"),
              pl.col("source_domain_id").alias("mid")).collect(engine="streaming"),
]).filter(pl.col("seed") != pl.col("mid")).unique()
mids = pl.Series(nb["mid"].unique().to_numpy().astype(np.int32)).implode()
back = pl.concat([
    lg.filter(pl.col("source_domain_id").is_in(mids))
      .select(pl.col("source_domain_id").alias("mid"),
              pl.col("target_domain_id").alias("hop2")).collect(engine="streaming"),
    lg.filter(pl.col("target_domain_id").is_in(mids))
      .select(pl.col("target_domain_id").alias("mid"),
              pl.col("source_domain_id").alias("hop2")).collect(engine="streaming"),
])
round_trip = nb.join(back, on="mid", how="inner").filter(
    pl.col("seed") == pl.col("hop2"))
print(f"\nstructural check on {len(probe)} probe seeds: "
      f"{round_trip['seed'].n_unique()} of them are reachable as their OWN "
      f"2-hop neighbour (seed -> mid -> seed)")
print("\nVERDICT: the 2-hop block leaks the label whenever that count is > 0; "
      "the fix is to drop hop2 == seed when expanding, then rebuild the block.")


# ---------------------------------------------------------------------------
# PART 2 (added after the fix): train-vs-target distribution check.
#
# The leak above was dangerous precisely because it made a block informative on
# train rows and empty on target rows. Own-tracker coverage cannot be measured
# for target rows (no labels), so the general guard is structural: every block
# must have a SIMILAR distribution of evidence mass and support size on the two
# row sets. A block that is systematically richer on train rows is either
# leaking or mis-built.
# ---------------------------------------------------------------------------
def block_stats(fname, prefix):
    cols = [f"{prefix}{i}" for i in range(N_TRK)]
    out = {}
    for tag in ("train", "target"):
        M = pd.read_parquet(FEAT / f"{fname}_{tag}.parquet")[cols].to_numpy("float32")
        out[tag] = (float((M.sum(1) > 0).mean()), float(np.median(M.sum(1))),
                    float(np.median((M > 0).sum(1))))
    return out


print("\n" + "=" * 74)
print("block            any-evidence rate      median mass        median support")
print("                  train  target        train  target       train  target")
print("=" * 74)
for fname, prefix in (("nbr_out", "no_"), ("nbr_in", "ni_"), ("tld_pop", "tp_"),
                      ("nbr_hub", "nh_"), ("nbr_rec", "nr_"), ("nbr_2h", "n2_")):
    st = block_stats(fname, prefix)
    tr, tg_ = st["train"], st["target"]
    print(f"{fname:14s}  {tr[0]:8.4f} {tg_[0]:7.4f}   {tr[1]:10.2f} {tg_[1]:8.2f}"
          f"   {tr[2]:8.0f} {tg_[2]:7.0f}")
print("=" * 74)


# ---------------------------------------------------------------------------
# PART 3 (added for the round-9 blocks): STANDALONE Recall@10 per block.
#
# The sharpest general leak test available here. Rank a row's 355 trackers by
# that block alone and score it: a block that carries the answer scores ~1.0 on
# the rows it covers (the buggy nbr_2h would have), while a legitimately
# informative block lands far below. It doubles as a price list -- pipeline_02
# established that tld_pop alone is worth 0.795 -- so a new block that scores
# near zero standalone is unlikely to add anything in combination either.
# ---------------------------------------------------------------------------
def standalone_recall(fname, prefix, k=10):
    cols = [f"{prefix}{i}" for i in range(N_TRK)]
    S = pd.read_parquet(FEAT / f"{fname}_train.parquet")[cols].to_numpy("float32")
    covered = S.sum(1) > 0
    n_true = Y.sum(1)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(Y, top, axis=1).sum(1)
    rec = hits / np.maximum(n_true, 1)
    return float(covered.mean()), float(rec.mean()), float(rec[covered].mean())


print("\n" + "=" * 74)
print("standalone Recall@10 by block (leak => ~1.0 on covered rows)")
print(f"{'block':12s} {'coverage':>9s} {'all rows':>9s} {'covered rows':>13s}")
print("=" * 74)
for fname, prefix in (("nbr_out", "no_"), ("nbr_in", "ni_"), ("nbr_hub", "nh_"),
                      ("tld_pop", "tp_"), ("nbr_2h", "n2_"),
                      ("direct", "dl_"), ("nbr_frac", "nf_"),
                      ("trk_cooc", "tc_"), ("tok_pop", "kp_")):
    cov, ra, rc = standalone_recall(fname, prefix)
    print(f"{fname:12s} {cov:9.4f} {ra:9.4f} {rc:13.4f}")
print("=" * 74)
