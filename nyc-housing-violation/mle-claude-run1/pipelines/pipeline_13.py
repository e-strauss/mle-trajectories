import skrub

from common import (base_features, event_features, load_events, load_union, load_xy,
                    make_model, model_features, table_block)

DESCRIPTION = ("+ short-horizon activity: 30-day counts before cutoff for HPD violations "
               "(all, Class C), HPD complaints and non-HPD 311")
PARENT = "pipeline_07"

X, y, viol = load_xy()
feats = base_features(X, viol)
feats = table_block(feats, ["sr311_2010", "sr311_2020"])
short = dict(windows=(30,), recency=False)
feats = skrub.deferred(event_features)(feats, viol, "viol30", **short)
feats = skrub.deferred(event_features)(feats, viol[viol["cat"] == "C"], "violC30", **short)
feats = skrub.deferred(event_features)(feats, load_events("hpd_complaints"), "comp30", **short)
feats = skrub.deferred(event_features)(feats, load_union(["sr311_2010", "sr311_2020"]),
                                       "sr311_30", **short)
pred = model_features(feats).skb.apply(make_model(), y=y)
