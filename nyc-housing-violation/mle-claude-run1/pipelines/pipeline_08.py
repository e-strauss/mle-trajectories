from common import base_features, load_xy, make_model, model_features, table_block

DESCRIPTION = "+ DOB enforcement: dob_violations (top-5 type codes) + dob_ecb_violations (severity), BBL or BIN->BBL"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "dob_violations")
feats = table_block(feats, "dob_ecb_violations", k_cats=3)
pred = model_features(feats).skb.apply(make_model(), y=y)
