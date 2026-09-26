from common import base_features, load_xy, make_model, model_features, table_block

DESCRIPTION = "+ dob_complaints block (BIN->BBL; counts, top-5 categories, recency)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "dob_complaints")
pred = model_features(feats).skb.apply(make_model(), y=y)
