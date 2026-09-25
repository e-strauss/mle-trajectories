"""Exploration round 2 -- is target.tsv a uniform sample of tracked domains?

Round 1 found target domains carry ~73% more link edges than a uniform
`domain_id % 1301 == 7` sample (outdeg 36.6 vs 21.2), while the *fraction* with
at least one labelled neighbour matched almost exactly (0.703 vs 0.698). That
is the difference between a shifted distribution and a shifted tail, and it
decides whether the modelled sample needs a degree-matched definition.

Checks:
  1. Degree quantiles, target vs uniform sample vs all tracked domains.
  2. Does degree predict n_true? If targets are higher-degree they also carry
     more trackers, which moves the recall@10 operating point.
  3. Is the uniform sample's recall@10 baseline stable across degree strata?
  4. Would a degree-filtered sample definition match target better?
"""
import numpy as np
import polars as pl

from common import WS_ROOT

IN = WS_ROOT / "input"
M, R = 1301, 7
QS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"])
in_sample = (pl.col("domain_id") % M) == R
S, POOL = tg.filter(in_sample), tg.filter(~in_sample)
sample_ids = S["domain_id"].unique().cast(pl.Int32)
tgt_ids = pl.read_csv(IN / "target.tsv", separator="\t")["domain_id"].cast(pl.Int32)

lg = pl.read_parquet(IN / "link-graph.parquet")
outdeg = lg.group_by("source_domain_id").len().rename(
    {"source_domain_id": "domain_id", "len": "outdeg"})
indeg = lg.group_by("target_domain_id").len().rename(
    {"target_domain_id": "domain_id", "len": "indeg"})
deg = outdeg.join(indeg, on="domain_id", how="full", coalesce=True).with_columns(
    pl.col("outdeg").fill_null(0), pl.col("indeg").fill_null(0))

all_tracked = tg["domain_id"].unique().cast(pl.Int32)

print("=" * 70)
print("1. degree quantiles (0 for domains absent from the link graph)")
groups = {"target": tgt_ids, "uniform sample": sample_ids, "all tracked": all_tracked}
degs = {}
for name, ids in groups.items():
    d = (ids.to_frame("domain_id")
           .join(deg, on="domain_id", how="left")
           .with_columns(pl.col("outdeg").fill_null(0), pl.col("indeg").fill_null(0)))
    degs[name] = d
    print(f"  {name:15s} n={d.height:>8,}  "
          f"outdeg mean {d['outdeg'].mean():7.1f} q={[int(d['outdeg'].quantile(q)) for q in QS]}")
    print(f"  {'':15s}            "
          f"indeg  mean {d['indeg'].mean():7.1f} q={[int(d['indeg'].quantile(q)) for q in QS]}")
    print(f"  {'':15s}            frac outdeg==0 {(d['outdeg'] == 0).mean():.3f}   "
          f"frac indeg==0 {(d['indeg'] == 0).mean():.3f}")

print("=" * 70)
print("2. does degree predict the number of trackers?")
n_true_all = tg.group_by("domain_id").len().rename({"len": "n_true"}).with_columns(
    pl.col("domain_id").cast(pl.Int32))
dn = degs["all tracked"].join(n_true_all, on="domain_id", how="left")
bins = [0, 1, 3, 10, 30, 100, 300, 10**9]
dn = dn.with_columns(
    pl.col("outdeg").cut(bins[1:-1], labels=[f"<{b}" for b in bins[1:]]).alias("obin"))
print(dn.group_by("obin").agg(pl.len().alias("n"),
                              pl.col("n_true").mean().alias("n_true_mean"))
        .sort("obin"))
print("  corr(log1p(outdeg), n_true):",
      round(np.corrcoef(np.log1p(dn["outdeg"].to_numpy()), dn["n_true"].to_numpy())[0, 1], 4))

print("=" * 70)
print("3. implied n_true for target, if degree is the only difference")
# reweight all-tracked domains to the target outdeg distribution and read off n_true
tb = degs["target"].with_columns(
    pl.col("outdeg").cut(bins[1:-1], labels=[f"<{b}" for b in bins[1:]]).alias("obin"))
w = tb.group_by("obin").agg((pl.len() / tb.height).alias("w"))
ref = dn.group_by("obin").agg(pl.col("n_true").mean().alias("m"))
imp = w.join(ref, on="obin", how="left")
print(imp.sort("obin"))
print(f"  uniform-sample n_true mean: {S.group_by('domain_id').len()['len'].mean():.3f}")
print(f"  degree-reweighted implied n_true mean: {(imp['w'] * imp['m']).sum():.3f}")

print("=" * 70)
print("4. baseline recall@10 by degree stratum (uniform sample)")
prior_top10 = POOL["tracker_id"].value_counts(sort=True).head(10)["tracker_id"].to_list()
sd = degs["uniform sample"].with_columns(
    pl.col("outdeg").cut(bins[1:-1], labels=[f"<{b}" for b in bins[1:]]).alias("obin"))
hit = (S.with_columns(pl.col("domain_id").cast(pl.Int32),
                      pl.col("tracker_id").is_in(prior_top10).alias("h"))
        .group_by("domain_id").agg(pl.col("h").mean().alias("r"))
        .join(sd.select("domain_id", "obin", "outdeg"), on="domain_id", how="left"))
print(hit.group_by("obin").agg(pl.len().alias("n"), pl.col("r").mean().alias("recall@10"))
         .sort("obin"))
print(f"  overall: {hit['r'].mean():.4f}")

print("=" * 70)
print("5. candidate sample definitions vs the target degree profile")
for label, expr in [
    ("uniform  (id % 1301 == 7)", (pl.col("domain_id") % M) == R),
    ("outdeg>=1 & id % 940 == 7", ((pl.col("domain_id") % 940) == 7) & (pl.col("outdeg") >= 1)),
    ("indeg>=1  & id % 1090 == 7", ((pl.col("domain_id") % 1090) == 7) & (pl.col("indeg") >= 1)),
    ("deg>=1    & id % 1130 == 7", ((pl.col("domain_id") % 1130) == 7)
                                   & ((pl.col("indeg") + pl.col("outdeg")) >= 1)),
]:
    cand = (all_tracked.to_frame("domain_id").join(deg, on="domain_id", how="left")
            .with_columns(pl.col("outdeg").fill_null(0), pl.col("indeg").fill_null(0))
            .filter(expr))
    print(f"  {label:28s} n={cand.height:>7,}  rows@355={cand.height * 355:>10,}  "
          f"outdeg mean {cand['outdeg'].mean():6.1f}  indeg mean {cand['indeg'].mean():6.1f}  "
          f"med out {int(cand['outdeg'].median())} in {int(cand['indeg'].median())}")
t = degs["target"]
print(f"  {'TARGET':28s} n={t.height:>7,}  {'':19s}  "
      f"outdeg mean {t['outdeg'].mean():6.1f}  indeg mean {t['indeg'].mean():6.1f}  "
      f"med out {int(t['outdeg'].median())} in {int(t['indeg'].median())}")
