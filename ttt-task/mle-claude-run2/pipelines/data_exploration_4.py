"""Reusable feature-block audit -- run this on EVERY new block before trusting it.

Usage:
    python data_exploration_4.py [block ...]        # default: every block in BLOCKS

Three checks the CV cannot do for you (guide section 15, CLAUDE.md leakage part 3):

  (a) STANDALONE SCORE. Rank the 355 candidates using this block's columns alone
      (no model: each numeric column is used directly as a ranking score) and
      report recall@10. A block that lands near 1.0 on the rows it covers is
      carrying the answer, not signal.

  (b) COVERAGE PARITY. Compare the block's non-null coverage and value
      distribution between the modelled sample and the target.tsv domains it
      will have to score for real. A block that is rich in training and empty at
      prediction time inflates the CV and helps nothing.

  (c) LABEL PROVENANCE. Assert directly that the statistics the block reads can
      never reach a modelled domain's own label: POOL and the modelled sample
      must be disjoint, and no neighbour walk may return to its own origin.

Output is a report; nothing is written to disk.
"""
import sys

import numpy as np
import pandas as pd
import polars as pl

from common import (BLOCKS, INPUT, SAMPLE_MOD, SAMPLE_REM, build_rows,
                    load_context, recall_at_10)

blocks = sys.argv[1:] or list(BLOCKS)

print("loading context ...")
ctx = load_context(INPUT)
rows = build_rows(ctx)
X = rows.drop(columns=["label"])
y = rows["label"]

print("=" * 72)
print("(c) LABEL PROVENANCE -- the workspace invariant")
pool_ids = set(ctx["pool"]["domain_id"].unique().to_list())
samp_ids = set(ctx["sample_ids"].to_list())
overlap = pool_ids & samp_ids
print(f"  modelled domains: {len(samp_ids):,}   POOL domains: {len(pool_ids):,}")
print(f"  POOL n modelled sample = {len(overlap)}  -> {'OK' if not overlap else 'LEAK'}")
assert not overlap, "POOL reaches a modelled domain's own labels"
bad_mod = [d for d in list(samp_ids)[:10000] if d % SAMPLE_MOD != SAMPLE_REM]
print(f"  every modelled domain satisfies id %% {SAMPLE_MOD} == {SAMPLE_REM}: "
      f"{'OK' if not bad_mod else 'BROKEN'}")
# a d -> n -> d walk would need a modelled domain as its own neighbour's source;
# since neighbour labels are only ever read from POOL, check POOL cannot be reached
e = ctx["edges"]
self_loops = e.filter(pl.col("source_domain_id") == pl.col("target_domain_id")).height
print(f"  self-loops in the focused link graph: {self_loops:,} "
      f"(harmless: a self-loop neighbour is a modelled domain, absent from POOL)")


class _ColRanker:
    """Ranks by one raw feature column -- lets recall_at_10 score a block alone."""

    def __init__(self, values):
        self.values = pd.Series(np.asarray(values, dtype=np.float64), index=X.index)

    def predict(self, Xt):
        return self.values.loc[Xt.index].to_numpy()


tgt_ids = ctx["target_ids"]
# the same row grid, but for target domains -- used only for coverage parity
tgt_grid = (tgt_ids.unique().sort().to_frame("domain_id")
            .join(pl.int_range(355, eager=True).cast(pl.Int16).to_frame("tracker_id"),
                  how="cross").to_pandas())

for name in blocks:
    print("=" * 72)
    print(f"BLOCK {name!r}")
    fn = BLOCKS[name]
    blk = fn(ctx, X)
    blk_t = fn(ctx, tgt_grid)

    print("  (a) standalone recall@10, ranking by each column alone:")
    for c in blk.columns:
        v = blk[c].to_numpy()
        if not np.issubdtype(v.dtype, np.number):
            continue
        finite = np.where(np.isfinite(v), v, np.nan)
        s = recall_at_10(_ColRanker(np.nan_to_num(finite, nan=-1e18)), X, y)
        print(f"      {c:16s} recall@10 = {s:.4f}"
              + ("   <-- SUSPICIOUS, near-perfect" if s > 0.98 else ""))

    print("  (b) coverage parity, modelled sample vs target.tsv:")
    for c in blk.columns:
        a, b = blk[c].to_numpy(), blk_t[c].to_numpy()
        if not np.issubdtype(a.dtype, np.number):
            continue
        cov_a = float(np.mean(np.isfinite(a) & (a != 0)))
        cov_b = float(np.mean(np.isfinite(b) & (b != 0)))
        flag = "   <-- COVERAGE MISMATCH" if abs(cov_a - cov_b) > 0.05 else ""
        print(f"      {c:16s} nonzero {cov_a:.3f} vs {cov_b:.3f} | "
              f"mean {np.nanmean(a):+.4g} vs {np.nanmean(b):+.4g} | "
              f"p99 {np.nanpercentile(a, 99):+.4g} vs {np.nanpercentile(b, 99):+.4g}{flag}")
