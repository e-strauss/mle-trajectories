"""pipeline_03 -- LightGBM classifier on the raw numeric features.

Same feature setup as pipeline_02 (drop Id, all-numeric passthrough) but a
leaf-wise gradient booster, which trains much faster than sklearn's HGB on a
table this large and is usually at least as accurate. Default hyperparameters.
"""
from lightgbm import LGBMClassifier

from common import load_xy, SEED

X, y = load_xy(target="Cover_Type")
X = X.skb.drop(cols="Id")
pred = X.skb.apply(
    LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
    y=y,
)

DESCRIPTION = "LightGBM classifier, default params, raw numeric features"
PARENT = "pipeline_02"
