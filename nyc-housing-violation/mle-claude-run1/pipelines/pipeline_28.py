from common import features_v28, load_xy, make_lgbm, model_features

DESCRIPTION = "long history (pipeline_26) + ACRIS ownership/financing (pipeline_24), LightGBM"
PARENT = "pipeline_26"

X, y, viol = load_xy()
feats = features_v28(X, viol)
pred = model_features(feats).skb.apply(make_lgbm(), y=y)
