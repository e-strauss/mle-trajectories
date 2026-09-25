"""Exploration round 5 -- where is pipeline_03 losing the remaining 0.14 recall?

pipeline_03 (prior + tld + out/in neighbour profiles) sits at 0.859. Rather than
guessing the next feature, refit it on fold 0 and take the misses apart:

  1. Recall by stratum: n_true, labelled-neighbour count, TLD size.
  2. Which trackers are missed, and are they rare or popular?
  3. Headroom of the signals not yet used, measured as "does the missed tracker
     appear at all in X?", for three candidate expansions:
       (a) 2-hop neighbours      (neighbours of neighbours, POOL labels)
       (b) co-citation           (domains sharing an in-linker with this one)
       (c) tracker co-occurrence (trackers that travel with the ones already
           predicted high, from POOL co-occurrence)
     A signal can only help where the answer is inside it, so this bounds each
     idea before any of them costs a pipeline.
"""
import numpy as np
import pandas as pd
import polars as pl

from common import (BLOCKS, INPUT, SMOOTH, build_rows, load_context, make_cv,
                    make_model, recall_at_10)

BLKS = ("prior", "tld", "nbr_out", "nbr_in")

print("loading + building ...")
ctx = load_context(INPUT)
rows = build_rows(ctx)
X = rows.drop(columns=["label"])
y = rows["label"]
feat = pd.concat([X[["tracker_id"]]] + [BLOCKS[b](ctx, X) for b in BLKS], axis=1)

cv = make_cv()
tr, te = next(iter(cv.split(X, y, groups=X["domain_id"])))
m = make_model().fit(feat.iloc[tr], y.iloc[tr])
p = m.predict_proba(feat.iloc[te])[:, 1]

te_df = pl.DataFrame({
    "domain_id": X["domain_id"].to_numpy()[te].astype(np.int64),
    "tracker_id": X["tracker_id"].to_numpy()[te].astype(np.int64),
    "n_true": X["n_true"].to_numpy()[te].astype(np.float64),
    "y": y.to_numpy()[te].astype(np.float64),
    "s": p,
})
top = (te_df.sort(["domain_id", "s"], descending=[False, True])
            .group_by("domain_id", maintain_order=True).head(10)
            .with_columns(pl.lit(1, dtype=pl.Int8).alias("in_top10")))
scored = te_df.join(top.select("domain_id", "tracker_id", "in_top10"),
                    on=["domain_id", "tracker_id"], how="left").with_columns(
    pl.col("in_top10").fill_null(0))
per_dom = scored.group_by("domain_id").agg(
    (pl.col("y") * pl.col("in_top10")).sum().alias("hits"),
    pl.col("n_true").first())
per_dom = per_dom.with_columns((pl.col("hits") / pl.col("n_true")).alias("r"))
print(f"fold-0 recall@10 = {per_dom['r'].mean():.4f}   test domains {per_dom.height:,}")

print("=" * 72)
print("1. recall by stratum")
edges = ctx["edges"]
dom_ids = pl.Series("domain_id", per_dom["domain_id"].cast(pl.Int32))
lab_nb = {}
for d, me, other in [("out", "source_domain_id", "target_domain_id"),
                     ("in", "target_domain_id", "source_domain_id")]:
    e = (edges.filter(pl.col(me).is_in(dom_ids.implode()))
         .select(pl.col(me).cast(pl.Int32).alias("domain_id"),
                 pl.col(other).cast(pl.Int32).alias("nb")).unique())
    lab_nb[d] = e.join(ctx["pool"], left_on="nb", right_on="domain_id", how="inner")
    cnt = lab_nb[d].group_by("domain_id").agg(pl.col("nb").n_unique().alias(f"{d}_nlab"))
    per_dom = per_dom.join(cnt.with_columns(pl.col("domain_id").cast(pl.Int64)),
                           on="domain_id", how="left").with_columns(
        pl.col(f"{d}_nlab").fill_null(0))

for col, cuts in [("n_true", [1, 2, 3, 5]), ("out_nlab", [0, 1, 3, 10, 30]),
                  ("in_nlab", [0, 1, 3, 10, 30])]:
    b = per_dom.with_columns(pl.col(col).cut(cuts).alias("b"))
    print(f"  by {col}:")
    print(b.group_by("b").agg(pl.len().alias("n"), pl.col("r").mean().alias("recall"))
           .sort("b").to_pandas().to_string(index=False))

print("=" * 72)
print("2. which trackers are missed")
missed = scored.filter((pl.col("y") == 1) & (pl.col("in_top10") == 0))
hitp = scored.filter((pl.col("y") == 1) & (pl.col("in_top10") == 1))
print(f"  true pairs {int(scored['y'].sum()):,}  missed {missed.height:,} "
      f"({missed.height / scored['y'].sum():.3f})")
pri = (ctx["pool"].group_by("tracker_id").len()
       .with_columns((pl.col("len") / ctx["n_pool_domains"]).alias("pri_p")))
mt = (missed.group_by("tracker_id").len().rename({"len": "n_missed"})
      .join(hitp.group_by("tracker_id").len().rename({"len": "n_hit"}),
            on="tracker_id", how="full", coalesce=True)
      .with_columns(pl.col("n_missed").fill_null(0), pl.col("n_hit").fill_null(0))
      .join(pri.with_columns(pl.col("tracker_id").cast(pl.Int64)).select("tracker_id", "pri_p"),
            on="tracker_id", how="left")
      .with_columns((pl.col("n_hit") / (pl.col("n_hit") + pl.col("n_missed"))).alias("hit_rate"))
      .sort("n_missed", descending=True))
print(mt.head(15).to_pandas().to_string(index=False))
print("  share of all misses carried by trackers with pri_p < 0.01:",
      round(mt.filter(pl.col("pri_p") < 0.01)["n_missed"].sum() / missed.height, 3))

print("=" * 72)
print("3. headroom of the unused signals -- is the missed tracker even reachable?")
miss_pairs = missed.select("domain_id", "tracker_id")
print(f"  missed (domain, tracker) pairs: {miss_pairs.height:,}")


def reach(name, pairs):
    """Fraction of missed pairs whose tracker appears in the given (domain, tracker) set."""
    j = miss_pairs.join(pairs.unique(), on=["domain_id", "tracker_id"], how="semi")
    print(f"  {name:38s} covers {j.height / miss_pairs.height:.3f} of misses")


# baseline: the 1-hop neighbourhood already in the model
one_hop = pl.concat([lab_nb["out"].select(pl.col("domain_id").cast(pl.Int64),
                                          pl.col("tracker_id").cast(pl.Int64)),
                     lab_nb["in"].select(pl.col("domain_id").cast(pl.Int64),
                                         pl.col("tracker_id").cast(pl.Int64))])
reach("(a0) 1-hop neighbours [already used]", one_hop)

# (a) 2-hop: neighbours of neighbours, either direction
und = pl.concat([
    edges.select(pl.col("source_domain_id").alias("a"), pl.col("target_domain_id").alias("b")),
    edges.select(pl.col("target_domain_id").alias("a"), pl.col("source_domain_id").alias("b")),
]).unique()
hop1 = (und.filter(pl.col("a").is_in(dom_ids.implode()))
        .rename({"a": "domain_id", "b": "m"}))
# cap hub expansion: a neighbour linking to >1000 domains says nothing specific
mdeg = und.group_by("a").len().rename({"a": "m", "len": "mdeg"})
hop2 = (hop1.join(mdeg, on="m").filter(pl.col("mdeg") <= 1000)
        .join(und.rename({"a": "m", "b": "nb2"}), on="m")
        .select("domain_id", "nb2").unique()
        .join(ctx["pool"], left_on="nb2", right_on="domain_id", how="inner")
        .select(pl.col("domain_id").cast(pl.Int64), pl.col("tracker_id").cast(pl.Int64)))
reach("(a) 2-hop neighbours (hub-capped)", hop2)

# (b) co-citation: domains sharing an in-linker
inn = edges.select(pl.col("source_domain_id").alias("src"),
                   pl.col("target_domain_id").alias("dst"))
src_deg = inn.group_by("src").len().rename({"len": "sdeg"})
mine = inn.filter(pl.col("dst").is_in(dom_ids.implode())).join(src_deg, on="src").filter(
    pl.col("sdeg") <= 1000)
cocite = (mine.rename({"dst": "domain_id"}).join(inn.rename({"dst": "sib"}), on="src")
          .select("domain_id", "sib").unique()
          .join(ctx["pool"], left_on="sib", right_on="domain_id", how="inner")
          .select(pl.col("domain_id").cast(pl.Int64), pl.col("tracker_id").cast(pl.Int64)))
reach("(b) co-citation siblings (hub-capped)", cocite)

# (c) tracker co-occurrence expansion of the model's current top-3
cooc = (ctx["pool"].join(ctx["pool"], on="domain_id")
        .filter(pl.col("tracker_id") != pl.col("tracker_id_right"))
        .group_by(["tracker_id", "tracker_id_right"]).len())
n_t = ctx["pool"].group_by("tracker_id").len().rename({"len": "nt"})
cooc = (cooc.join(n_t, on="tracker_id")
        .with_columns((pl.col("len") / pl.col("nt")).alias("pcond"))
        .filter(pl.col("pcond") > 0.05))
top3 = (te_df.sort(["domain_id", "s"], descending=[False, True])
        .group_by("domain_id", maintain_order=True).head(3).select("domain_id", "tracker_id"))
exp = (top3.with_columns(pl.col("tracker_id").cast(pl.Int16))
       .join(cooc, on="tracker_id")
       .select("domain_id", pl.col("tracker_id_right").cast(pl.Int64).alias("tracker_id")))
reach("(c) co-occurrence expansion of top-3", exp)
reach("(a)+(b) union", pl.concat([hop2, cocite]))
