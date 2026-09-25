"""Exploration round 7 -- is the plateau a feature problem or a sample-size problem?

pipelines 04, 05 and 06 all sit at ~0.872, and exploration round 6 priced every
new graph block inside the noise. Two hypotheses remain for the stuck 0.127:

  A. FEATURES. Rare trackers (76.5% of misses have POOL frequency < 1%) have too
     few training examples to learn individually -- so pool them through
     trackers.tsv (company / brand / category / country). Priced here.

  B. SAMPLE SIZE. The locked row design models 14,335 domains = ~28k positives
     spread over 355 trackers. That may simply be too little to learn the rare
     ones at all, in which case no feature fixes it and the answer is more rows.

B is measured as a learning curve: hold the fold-0 TEST domains fixed and grow
the TRAINING domain set with domains drawn from other `domain_id % 1301` classes.
Those extra domains stay in POOL, which is correct and not a leak -- every 1-hop
feature of a domain reads its NEIGHBOURS' labels, never its own, and the focused
link graph has zero self-loops (data_exploration_4.py check (c)). What the curve
cannot do is change the locked row design; if it bends upward, that is a finding
for the next workspace, not a pipeline for this one.
"""
import numpy as np
import pandas as pd
import polars as pl

from common import (BLOCKS, INPUT, N_TRACKERS, SAMPLE_MOD, SAMPLE_REM, build_rows,
                    load_context, make_cv, make_model, recall_at_10)

BASE = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w")

print("loading ...")
ctx = load_context(INPUT)
rows = build_rows(ctx)
X = rows.drop(columns=["label"])
y = rows["label"]
cv = make_cv()
tr, te = next(iter(cv.split(X, y, groups=X["domain_id"])))

cache = {}


def block(name, Xa):
    key = (name, id(Xa))
    if key not in cache:
        cache[key] = BLOCKS[name](ctx, Xa)
    return cache[key]


def feats(names, Xa):
    return pd.concat([Xa[["tracker_id"]]] + [block(n, Xa) for n in names], axis=1)


class _Fitted:
    def __init__(self, m, f):
        self.m, self.f = m, f

    def predict_proba(self, Xt):
        return self.m.predict_proba(self.f.loc[Xt.index])


print("=" * 72)
print("A. does pooling rare trackers through trackers.tsv help?")
f_base = feats(list(BASE), X)
m = make_model().fit(f_base.iloc[tr], y.iloc[tr])
base = recall_at_10(_Fitted(m, f_base), X.iloc[te], y.iloc[te])
print(f"  base                     {base:.4f}")
for extra in [["meta"], ["host", "content"], ["meta", "host", "content"]]:
    f = feats(list(BASE) + extra, X)
    m = make_model().fit(f.iloc[tr], y.iloc[tr])
    s = recall_at_10(_Fitted(m, f), X.iloc[te], y.iloc[te])
    print(f"  +{'+'.join(extra):24s}{s:.4f}   delta {s - base:+.4f}")

# recall restricted to rare trackers -- where the meta block is supposed to act
f = feats(list(BASE) + ["meta"], X)
m = make_model().fit(f.iloc[tr], y.iloc[tr])
p = m.predict_proba(f.iloc[te])[:, 1]
pri = (ctx["pool"].group_by("tracker_id").len()
       .with_columns((pl.col("len") / ctx["n_pool_domains"]).alias("pri_p")))
rare = set(pri.filter(pl.col("pri_p") < 0.01)["tracker_id"].to_list())
d = pl.DataFrame({"domain_id": X["domain_id"].to_numpy()[te].astype(np.int64),
                  "tracker_id": X["tracker_id"].to_numpy()[te].astype(np.int64),
                  "y": y.to_numpy()[te].astype(np.float64), "s": p})
top = (d.sort(["domain_id", "s"], descending=[False, True])
        .group_by("domain_id", maintain_order=True).head(10))
tp = d.filter(pl.col("y") == 1).join(top.select("domain_id", "tracker_id")
                                     .with_columns(pl.lit(1).alias("hit")),
                                     on=["domain_id", "tracker_id"], how="left")
tp = tp.with_columns(pl.col("hit").fill_null(0),
                     pl.col("tracker_id").is_in(list(rare)).alias("is_rare"))
print(tp.group_by("is_rare").agg(pl.len().alias("true_pairs"),
                                 pl.col("hit").mean().alias("pair_hit_rate")).to_pandas()
      .to_string(index=False))

print("=" * 72)
print("B. learning curve: same test domains, more training domains")
te_doms = np.unique(X["domain_id"].to_numpy()[te])
labels_all = pl.read_parquet(INPUT / "tracking_graph_train.parquet",
                             columns=["domain_id", "tracker_id"]).with_columns(
    pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int16))
tr_doms_base = np.unique(X["domain_id"].to_numpy()[tr])

for mult, extra_rems in [(1, []), (2, [11]), (4, [11, 13, 17]), (8, [11, 13, 17, 19, 23, 29, 31])]:
    extra = labels_all.filter((pl.col("domain_id") % SAMPLE_MOD).is_in(extra_rems)
                              if extra_rems else pl.lit(False))
    add_ids = extra["domain_id"].unique().sort()
    tr_ids = np.concatenate([tr_doms_base, add_ids.to_numpy()])
    grid = (pl.Series("domain_id", tr_ids).to_frame("domain_id")
            .join(pl.int_range(N_TRACKERS, eager=True).cast(pl.Int16).to_frame("tracker_id"),
                  how="cross"))
    lab = pl.concat([ctx["sample_labels"], extra]).with_columns(
        pl.lit(1, dtype=pl.Int8).alias("label"))
    Xtr = (grid.join(lab, on=["domain_id", "tracker_id"], how="left")
           .with_columns(pl.col("label").fill_null(0)).sort(["domain_id", "tracker_id"]))
    ytr = Xtr["label"].to_pandas()
    Xtr = Xtr.drop("label").to_pandas()
    ftr = feats(list(BASE), Xtr)
    m = make_model().fit(ftr, ytr)
    s = recall_at_10(_Fitted(m, f_base), X.iloc[te], y.iloc[te])
    print(f"  {mult}x  train domains {len(tr_ids):>7,}  rows {len(Xtr):>10,}  "
          f"positives {int(ytr.sum()):>6,}  recall@10 = {s:.4f}")
