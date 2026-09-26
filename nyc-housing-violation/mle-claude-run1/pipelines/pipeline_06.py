from common import base_features, load_xy, make_model, model_features, table_block

DESCRIPTION = "+ hpd_bedbug_reports block (filings, recency, infested-unit sum)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "hpd_bedbug_reports", k_cats=0, value=True)
pred = model_features(feats).skb.apply(make_model(), y=y)
