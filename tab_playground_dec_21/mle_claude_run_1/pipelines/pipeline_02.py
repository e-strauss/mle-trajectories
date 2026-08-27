"""pipeline_02 -- HistGradientBoostingClassifier on the raw numeric features.

Data is already all-numeric (10 continuous + 44 binary indicators), so no
encoding is needed: drop Id and feed the columns straight to a NaN-tolerant
histogram GBM with default hyperparameters. This is the real baseline.

early_stopping=False: the default 'auto' does an internal STRATIFIED validation
split, which raises on the 1-member class 5. Disabling it sidesteps that (and
early stopping is superfluous at 4M rows anyway).
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from common import load_xy, SEED

X, y = load_xy(target="Cover_Type")
X = X.skb.drop(cols="Id")
pred = X.skb.apply(
    HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
    y=y,
)

DESCRIPTION = "HistGradientBoostingClassifier, default params, raw numeric features"
PARENT = "pipeline_01"
