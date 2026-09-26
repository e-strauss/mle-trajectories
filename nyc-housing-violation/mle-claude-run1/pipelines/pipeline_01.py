import skrub

from common import base_features, load_xy, make_model, model_features

DESCRIPTION = ("anchor: v1 pipeline_06 feature set (PLUTO + HPD violation/complaint "
               "history) on 3 cutoffs, expanding-window time CV")
PARENT = None

X, y, viol = load_xy()
feats = base_features(X, viol)
pred = model_features(feats).skb.apply(make_model(), y=y)
