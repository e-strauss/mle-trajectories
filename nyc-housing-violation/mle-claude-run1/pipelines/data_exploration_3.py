"""Exploration 3: error analysis of the current best (pipeline_07 features/model).

Reproduces fold 2 of the CV by hand: fit on cutoffs 2020+2021, score cutoff 2022.
Reports AP / positives / captured positives by segment:
  - prior history: Class C in the last year / older C only / violations but no C /
    no HPD violation in 3y ("cold start"),
  - building size (unitsres bins),
and where the positives rank (share of all positives inside the top 5% / 10%).

Features carry no label information (pre-cutoff events only), so building them on
the full row set is safe; only the model fit is split by cutoff. Fine-grained
skrub DataOps, evaluated once with .skb.eval(). Writes nothing.
"""
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import average_precision_score

from common import (LAKE, attach_label, base_features, load_events, make_model, read_lots,
                    table_block)

TEST_CUT = pd.Timestamp("2022-01-01")


def fit_predict(feats, labelled):
    train = (labelled["cutoff"] < TEST_CUT).to_numpy()
    Xm = feats.drop(columns=["bbl", "cutoff"])
    model = make_model().fit(Xm[train], labelled["y"][train])
    return pd.Series(model.predict_proba(Xm[~train])[:, 1], index=Xm.index[~train])


def segment(d):
    c1 = d["viol_C_n365d"] > 0
    c3 = d["viol_C_n1095d"] > 0
    anyv = d["viol_n1095d"] > 0
    return np.select([c1, c3, anyv], ["C in last 1y", "C 1-3y ago only", "violations, no C"],
                     default="no HPD violation 3y")


def seg_report(d):
    return pd.Series({"lots": len(d), "pos": int(d["y"].sum()), "rate": d["y"].mean(),
                      "AP_within": average_precision_score(d["y"], d["p"]) if d["y"].any() else np.nan,
                      "pos_share": d["y"].sum()}).round(4)


def add_share(r, total):
    return r.assign(pos_share=(r["pos_share"] / total).round(3))


def rank_report(d):
    top5, top10 = d["p"] >= d["p"].quantile(0.95), d["p"] >= d["p"].quantile(0.90)
    return pd.Series({"AP": average_precision_score(d["y"], d["p"]),
                      "recall@5%": d.loc[top5, "y"].sum() / d["y"].sum(),
                      "recall@10%": d.loc[top10, "y"].sum() / d["y"].sum(),
                      "prec@5%": d.loc[top5, "y"].mean()}).round(4)


lake = skrub.as_data_op(LAKE)
lots = lake.skb.apply_func(read_lots)
viol = load_events("hpd_violations", lake)
labelled = skrub.deferred(attach_label)(lots, viol)
feats = base_features(lots, viol, lake)
feats = table_block(feats, ["sr311_2010", "sr311_2020"], lake)
p = skrub.deferred(fit_predict)(feats, labelled)
scored = feats.loc[p.index].assign(y=labelled["y"], p=p)
scored = scored.assign(segment=scored.skb.apply_func(segment),
                       size=scored["unitsres"].skb.apply_func(
                           pd.cut, [2, 5, 10, 20, 50, 100, 1e9],
                           labels=["3-5", "6-10", "11-20", "21-50", "51-100", ">100"]))
total_pos = scored["y"].sum()
by_segment = skrub.deferred(add_share)(
    scored.groupby("segment")[["y", "p"]].apply(seg_report), total_pos)
by_size = skrub.deferred(add_share)(
    scored.groupby("size", observed=True)[["y", "p"]].apply(seg_report), total_pos)
overall = skrub.deferred(rank_report)(scored)
result = skrub.as_data_op({"overall": overall, "by_segment": by_segment, "by_size": by_size})

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    out = result.skb.eval()
    for k, v in out.items():
        print(f"== {k}\n{v}\n")
