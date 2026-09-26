from common import base_features, load_xy, make_model, model_features, table_block

DESCRIPTION = "+ rare HPD orders: hpd_vacate_orders (counts, reasons, vacated units) + hpd_aep_buildings (AEP start, dated)"
PARENT = "pipeline_01"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, "hpd_vacate_orders", value=True)
feats = table_block(feats, "hpd_aep_buildings", k_cats=0)
pred = model_features(feats).skb.apply(make_model(), y=y)
