import skrub
from lightgbm import LGBMClassifier
from skrub import TableVectorizer, ToCategorical

from common import features_v15, load_xy, model_features

DESCRIPTION = ("model family: LightGBM on pipeline_15 features (fused sweep: "
               "num_leaves x n_estimators, lr 0.03, subsample/colsample 0.8)")
PARENT = "pipeline_15"

X, y, viol = load_xy()
feats = features_v15(X, viol)
Xv = model_features(feats).skb.apply(TableVectorizer(low_cardinality=ToCategorical()))
model = LGBMClassifier(
    learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    min_child_samples=50, reg_lambda=1.0, n_jobs=32, random_state=0, verbose=-1,
    num_leaves=skrub.choose_from([15, 63], name="num_leaves"),
    n_estimators=skrub.choose_from([400, 1200], name="n_estimators"),
)
pred = Xv.skb.apply(model, y=y)
