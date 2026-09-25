"""Exploration round 3 -- debug pipeline_01's impossible fold spread.

pipeline_01 can only express ONE global ranking of the 355 trackers (every
feature is a function of tracker_id alone), so all three folds must score the
same ~0.755. They scored [0.225, 0.004, 0.757]. Something in the plan assembly,
the fold split, or the scorer is wrong; this script takes one fold apart by hand.
"""
import numpy as np
import pandas as pd
import polars as pl
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

from common import (BLOCKS, INPUT, build_rows, features, load_context, load_xy,
                    make_cv, recall_at_10)

print("=" * 70)
print("1. build the row table and the prior block directly")
ctx = load_context(INPUT)
rows = build_rows(ctx)
print("rows:", rows.shape, list(rows.columns))
print(rows.head(3))
print("label mean:", rows["label"].mean(), " positives:", int(rows["label"].sum()))
print("index unique:", rows.index.is_unique, " monotonic:", rows.index.is_monotonic_increasing)

X_all = rows.drop(columns=["label"])
y_all = rows["label"]
blk = BLOCKS["prior"](ctx, X_all)
print("prior block:", blk.shape, list(blk.columns))
print("  nulls per col:", blk.isna().sum().to_dict())
print(blk.head(3))

print("=" * 70)
print("2. hand-rolled fold: fit HistGB on fold-0 train, score fold-0 test")
feat = pd.concat([X_all[["tracker_id"]], blk], axis=1)
cv = make_cv()
tr, te = next(iter(cv.split(feat, y_all, groups=X_all["domain_id"])))
print(f"  train rows {len(tr):,}  test rows {len(te):,}")
print(f"  train domains {X_all['domain_id'].iloc[tr].nunique():,}  "
      f"test domains {X_all['domain_id'].iloc[te].nunique():,}")
print(f"  train pos rate {y_all.iloc[tr].mean():.5f}  test pos rate {y_all.iloc[te].mean():.5f}")

m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                   early_stopping=False, random_state=42)
m.fit(feat.iloc[tr], y_all.iloc[tr])
p = m.predict_proba(feat.iloc[te])[:, 1]
print("  predicted proba: min %.6g max %.6g n_unique %d"
      % (p.min(), p.max(), len(np.unique(np.round(p, 9)))))


class _Fitted:
    def __init__(self, model, feat):
        self.model, self.feat = model, feat

    def predict_proba(self, X):
        return self.model.predict_proba(self.feat.loc[X.index])


print("  recall@10 on this fold:", recall_at_10(_Fitted(m, feat), X_all.iloc[te], y_all.iloc[te]))

# the ranking this model implies, versus the true POOL popularity ranking
tr_rank = (pd.DataFrame({"tracker_id": X_all["tracker_id"].iloc[te].to_numpy(), "p": p})
           .groupby("tracker_id")["p"].mean().sort_values(ascending=False))
print("  model top10 trackers:", tr_rank.head(10).index.tolist())
pool_top10 = (ctx["pool"]["tracker_id"].value_counts(sort=True)
              .head(10)["tracker_id"].to_list())
print("  POOL  top10 trackers:", pool_top10)

print("=" * 70)
print("3. same thing through the skrub plan, one fold at a time")
with skrub.config_context(eager_data_ops=False):
    ctx_op, Xo, yo = load_xy()
    f = features(ctx_op, Xo, blocks=("prior",))
    pr = f.skb.apply(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                   early_stopping=False, random_state=42),
                     y=yo)
learner = pr.skb.make_learner()
env = pr.skb.get_data()
print("  env keys:", list(env.keys()))

split = learner.split(env, cv=make_cv())
print("  split type:", type(split))
