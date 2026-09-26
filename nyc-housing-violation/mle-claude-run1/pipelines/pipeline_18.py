import skrub
from catboost import CatBoostClassifier
from skrub import TableVectorizer

from common import features_v15, load_xy, model_features

DESCRIPTION = ("model family: CatBoost (GPU) on pipeline_15 features, whole-estimator "
               "variants over depth x iterations (lr 0.05)")
PARENT = "pipeline_15"


class CatBoostClassifierCloneable(CatBoostClassifier):
    def __sklearn_clone__(self):
        return CatBoostClassifierCloneable(**self.get_params(deep=False))


X, y, viol = load_xy()
feats = features_v15(X, viol)
Xv = model_features(feats).skb.apply(TableVectorizer())
variants = {f"d{d}_it{it}": CatBoostClassifierCloneable(
                depth=d, iterations=it, learning_rate=0.05, task_type="GPU", devices="0",
                random_seed=0, verbose=0)
            for d in (6, 8) for it in (1000, 2500)}
pred = Xv.skb.apply(skrub.choose_from(variants, name="model"), y=y)
