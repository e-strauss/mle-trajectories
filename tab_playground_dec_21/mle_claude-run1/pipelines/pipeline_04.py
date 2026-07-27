"""pipeline_04 -- XGBoost classifier on the raw numeric features.

Third booster family for comparison (histogram tree method). XGBoost needs
0-indexed contiguous integer labels for multiclass, but Cover_Type is 1..7 with
class 5 nearly absent -- rather than remap inside the plan, we let its label
encoder handle it via the sklearn wrapper, which fits a LabelEncoder internally.
Default hyperparameters, tree_method='hist' for speed on 4M rows.
"""
from xgboost import XGBClassifier

from common import load_xy, SEED

X, y = load_xy(target="Cover_Type")
X = X.skb.drop(cols="Id")
pred = X.skb.apply(
    XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1),
    y=y,
)

DESCRIPTION = "XGBoost classifier (hist), default params, raw numeric features"
PARENT = "pipeline_02"
