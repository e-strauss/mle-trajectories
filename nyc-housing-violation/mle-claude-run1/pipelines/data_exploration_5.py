"""Exploration 5: audit of the long-history block (pipeline_26).

  - HPD violations / complaints per inspection year 2008-2016: the violations base
    is "all violations open as of 2012-10-01", so pre-2013 years are an incomplete,
    open-only subset -> how big is the cliff?
  - coverage (> 0) of each long-window column per cutoff, INCLUDING the test cutoff
    (22v3 @ 2023) -> parity,
  - standalone AP of each long-window column per training cutoff (near-perfect =
    carries the label; also shows which window carries the gain).
Fine-grained skrub DataOps, one output node evaluated once. Writes nothing.
"""
import pandas as pd
import skrub
from sklearn.metrics import average_precision_score

from common import (CUTOFFS, LAKE, TEST_CUTOFF, TEST_RELEASE, attach_label, event_features,
                    load_events, read_lots)

LONG_SINCE = pd.Timestamp("2008-01-01")
ALL_CUTS = {**CUTOFFS, TEST_CUTOFF: TEST_RELEASE}


def per_year(ev):
    return ev["date"].dt.year.value_counts().sort_index().loc[2008:2016]


def coverage(feats, cols):
    return (feats[cols] > 0).groupby(feats["cutoff"]).mean().round(4).T


def standalone_ap(scored, cols):
    out = {}
    for cut, d in scored.groupby("cutoff"):
        out[cut.year] = {c: average_precision_score(d["y"], d[c].fillna(0) if "days" not in c
                                                    else -d[c].fillna(10_000)) for c in cols}
    return pd.DataFrame(out).round(4)


lake = skrub.as_data_op(LAKE)
lots = lake.skb.apply_func(read_lots, ALL_CUTS)
viol = load_events("hpd_violations", lake)
viol_long = load_events("hpd_violations", lake, since=LONG_SINCE)
comp_long = load_events("hpd_complaints", lake, since=LONG_SINCE)
labelled = skrub.deferred(attach_label)(lots[lots["cutoff"] < TEST_CUTOFF], viol)

feats = skrub.deferred(event_features)(lots, viol_long, "violL", windows=(1095, 1825, 3650),
                                       recency=False)
feats = skrub.deferred(event_features)(feats, viol_long[viol_long["cat"] == "C"], "violCL",
                                       windows=(1095, 1825, 3650))
feats = skrub.deferred(event_features)(feats, comp_long, "compL", windows=(1095, 1825, 3650),
                                       recency=False)
cols = ["violL_n1095d", "violL_n1825d", "violL_n3650d", "violCL_n1095d", "violCL_n1825d",
        "violCL_n3650d", "violCL_days_since", "compL_n1095d", "compL_n1825d", "compL_n3650d"]
scored = feats.loc[labelled.index].assign(y=labelled["y"])

result = skrub.as_data_op({
    "violations per inspection year": viol_long.skb.apply_func(per_year),
    "complaints per received year": comp_long.skb.apply_func(per_year),
    "coverage (>0) per cutoff": skrub.deferred(coverage)(feats, cols),
    "standalone AP per training cutoff": skrub.deferred(standalone_ap)(scored, cols),
})

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for k, v in result.skb.eval().items():
        print(f"== {k}\n{v}\n")
