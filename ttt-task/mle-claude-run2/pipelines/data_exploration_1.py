"""Exploration round 1 -- size the row design and measure raw signal strength.

Reports only; writes nothing. Questions:
  1. How many domains fall in a `domain_id % M == r` sample, and how does the
     modelled sample compare to target.tsv (degree, tld, tracker count)?
  2. Baseline recall@10 of the global tracker prior.
  3. How much does the link graph carry? Out-neighbour / in-neighbour tracker
     profiles, ranked alone.
  4. Coverage parity: do target domains have the same neighbour coverage as the
     modelled sample?
  5. TLD-conditional prior + url-classification join rate.
"""
import numpy as np
import polars as pl

from common import WS_ROOT

IN = WS_ROOT / "input"
M, R = 1301, 7          # deterministic sample: domain_id % 1301 == 7


def recall_at10(sample_labels: pl.DataFrame, ranking: pl.DataFrame) -> float:
    """sample_labels: domain_id, tracker_id (truth). ranking: domain_id, tracker_id, score."""
    top = (ranking.sort(["domain_id", "score"], descending=[False, True])
                  .group_by("domain_id", maintain_order=True).head(10)
                  .select("domain_id", "tracker_id")
                  .with_columns(pl.lit(1, dtype=pl.Int8).alias("hit")))
    j = sample_labels.join(top, on=["domain_id", "tracker_id"], how="left")
    per_dom = j.group_by("domain_id").agg(pl.col("hit").fill_null(0).mean().alias("r"))
    # domains with no predictions at all score 0 and are already covered
    return per_dom["r"].mean()


print("=" * 70)
print("1. sample sizing")
tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"])
in_sample = (pl.col("domain_id") % M) == R
S = tg.filter(in_sample)
POOL = tg.filter(~in_sample)
sample_ids = S["domain_id"].unique()
print(f"modelled domains: {len(sample_ids):,}   label rows: {S.height:,}")
print(f"design matrix @355 candidates: {len(sample_ids) * 355:,}")
print(f"pool rows (disjoint label source): {POOL.height:,}  "
      f"domains: {POOL['domain_id'].n_unique():,}")
n_true = S.group_by("domain_id").len().rename({"len": "n_true"})
print("n_true per modelled domain:", n_true["n_true"].describe())

print("=" * 70)
print("2. global prior baseline")
prior = (POOL.group_by("tracker_id").len()
             .with_columns((pl.col("len") / POOL["domain_id"].n_unique()).alias("score"))
             .drop("len"))
glob = sample_ids.to_frame("domain_id").join(prior, how="cross")
print(f"recall@10 global prior: {recall_at10(S, glob):.4f}")

print("=" * 70)
print("3. link-graph neighbour signal")
lg = pl.read_parquet(IN / "link-graph.parquet")
sid = sample_ids.cast(pl.Int32)
sid_df = sid.to_frame("domain_id")
tgt_ids = pl.read_csv(IN / "target.tsv", separator="\t")["domain_id"].cast(pl.Int32)

# pool labels keyed by int32 for joining against the link graph
pool_lab = POOL.with_columns(pl.col("domain_id").cast(pl.Int32))

for direction, me, other in [("out", "source_domain_id", "target_domain_id"),
                             ("in", "target_domain_id", "source_domain_id")]:
    e = lg.filter(pl.col(me).is_in(sid)).select(
        pl.col(me).alias("domain_id"), pl.col(other).alias("nb"))
    deg = e.group_by("domain_id").len().rename({"len": "deg"})
    prof = (e.join(pool_lab, left_on="nb", right_on="domain_id", how="inner")
             .group_by(["domain_id", "tracker_id"]).len()
             .join(deg, on="domain_id")
             .with_columns((pl.col("len") / pl.col("deg")).alias("score")))
    cov = prof["domain_id"].n_unique()
    print(f"[{direction}] edges {e.height:,}  domains w/ >=1 edge "
          f"{deg.height:,}/{len(sid):,}  w/ >=1 labelled nb {cov:,}")
    print(f"     recall@10 by {direction}-neighbour profile alone: "
          f"{recall_at10(S, prof.select('domain_id', 'tracker_id', 'score')):.4f}")
    # blend with the global prior as a backoff so uncovered domains still rank
    blend = (prof.select("domain_id", "tracker_id", "score")
                 .join(sid_df.join(prior, how="cross"),
                       on=["domain_id", "tracker_id"], how="full", coalesce=True)
                 .with_columns((pl.col("score").fill_null(0.0) * 10
                                + pl.col("score_right").fill_null(0.0)).alias("score")))
    print(f"     recall@10 blended with global prior: "
          f"{recall_at10(S, blend.select('domain_id', 'tracker_id', 'score')):.4f}")

print("=" * 70)
print("4. coverage parity: modelled sample vs target.tsv")
for name, ids in [("sample", sid), ("target", tgt_ids)]:
    o = lg.filter(pl.col("source_domain_id").is_in(ids))
    i = lg.filter(pl.col("target_domain_id").is_in(ids))
    o_lab = o.join(pool_lab, left_on="target_domain_id", right_on="domain_id", how="inner")
    i_lab = i.join(pool_lab, left_on="source_domain_id", right_on="domain_id", how="inner")
    print(f"  {name:7s} n={len(ids):,}  outdeg mean "
          f"{o.height / len(ids):.1f}  indeg mean {i.height / len(ids):.1f}  "
          f"frac w/ labelled out-nb {o_lab['source_domain_id'].n_unique() / len(ids):.3f}  "
          f"frac w/ labelled in-nb {i_lab['target_domain_id'].n_unique() / len(ids):.3f}")

print("=" * 70)
print("5. tld prior + url-classification join rate")
dom = pl.read_parquet(IN / "domains.parquet")
tld = dom.with_columns(pl.col("domain").str.split(".").list.last().alias("tld"))
for name, ids in [("sample", sample_ids), ("target", tgt_ids.cast(pl.Int64))]:
    t = tld.filter(pl.col("domain_id").is_in(ids))
    print(f"  {name}: top tlds", t["tld"].value_counts(sort=True).head(6).to_dicts())

uc = pl.read_csv(IN / "url-classification.csv")
host = (uc.with_columns(
    pl.col("url").str.replace(r"^https?://", "").str.replace(r"^www\.", "")
      .str.split("/").list.first().str.split(":").list.first().alias("host"))
    .select("host", "category").unique(subset="host"))
print("  url-classification rows:", uc.height, "unique hosts:", host.height)
print("  categories:", uc["category"].value_counts(sort=True).head(10).to_dicts())
hj = dom.join(host, left_on="domain", right_on="host", how="inner")
print("  domains matched to a category:", hj.height)
for name, ids in [("sample", sample_ids), ("target", tgt_ids.cast(pl.Int64))]:
    print(f"  {name}: frac with category "
          f"{hj.filter(pl.col('domain_id').is_in(ids)).height / len(ids):.3f}")

print("=" * 70)
print("6. tld-conditional prior, ranked alone")
pool_tld = pool_lab.join(tld.select(pl.col("domain_id").cast(pl.Int32), "tld"),
                         on="domain_id", how="inner")
tld_n = pool_tld.group_by("tld").agg(pl.col("domain_id").n_unique().alias("n"))
tld_prior = (pool_tld.group_by(["tld", "tracker_id"]).len()
             .join(tld_n, on="tld")
             .with_columns((pl.col("len") / pl.col("n")).alias("score")))
s_tld = tld.select(pl.col("domain_id").cast(pl.Int32), "tld").filter(pl.col("domain_id").is_in(sid))
r = s_tld.join(tld_prior, on="tld", how="inner").select("domain_id", "tracker_id", "score")
print(f"  recall@10 by tld-conditional prior alone: {recall_at10(S, r):.4f}")
