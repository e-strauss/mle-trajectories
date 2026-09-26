import skrub

from common import (LAKE, features_v28, load_acris, load_xy, make_lgbm, model_features,
                    owner_portfolio, read_acris_grantees)

DESCRIPTION = ("+ owner portfolio (ACRIS grantee of the latest deed before cutoff): "
               "portfolio size, violations (all / Class C) per other lot of the same "
               "owner, 1y/3y, own lot excluded")
PARENT = "pipeline_28"

X, y, viol = load_xy()
feats = features_v28(X, viol)
lake = skrub.as_data_op(LAKE)
grantees = lake.skb.apply_func(read_acris_grantees)
feats = skrub.deferred(owner_portfolio)(feats, load_acris(lake), grantees, viol)
pred = model_features(feats).skb.apply(make_lgbm(), y=y)
