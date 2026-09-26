"""Exploration 2: lake triage part 2 + reusable per-table audit.

For every event table in common.SPECS (or those given on the command line):
  - rows read in the history window, share keyed to a lot of the universe,
  - lot coverage (>= 1 event in the 365d before cutoff) at each backtest cutoff
    AND at the test cutoff (22v3 @ 2023-01-01) -> coverage parity,
  - standalone AP of the 365d event count per training cutoff (a near-perfect
    value would mean the block carries the label).

Fine-grained skrub DataOps (one recorded node per step), all outputs combined
into one final node and evaluated ONCE with .skb.eval(). Writes nothing.

    python data_exploration_2.py [table ...]
"""
import sys

import pandas as pd
import skrub
from sklearn.metrics import average_precision_score

from common import (CUTOFFS, LAKE, SPECS, TEST_CUTOFF, TEST_RELEASE, attach_label,
                    event_features, load_events, read_lots)

ALL_CUTS = {**CUTOFFS, TEST_CUTOFF: TEST_RELEASE}


def ap_of(d):
    return average_precision_score(d["y"], d["e"])


def report_row(name, n_rows, keyed, first, cov, ap):
    rec = {"table": name, "rows": n_rows, "keyed_to_lot": round(keyed, 3),
           "first": first.date() if pd.notna(first) else None}
    rec |= {f"cov{c.year}": round(v, 4) for c, v in cov.items()}
    rec |= {f"AP{c.year}": round(v, 4) for c, v in ap.items()}
    return rec


def to_frame(*rows):
    return pd.DataFrame(list(rows))


def table_audit(name, lake, lots, labelled):
    ev = load_events(name, lake)                                   # read ... assemble
    n_rows = ev["bbl"].size
    keyed = ev["bbl"].isin(lots["bbl"]).mean()                     # key match rate
    first = ev["date"].min()
    feats = skrub.deferred(event_features)(lots, ev, "e", windows=(365,), recency=False)
    cov = (feats["e_n365d"] > 0).groupby(feats["cutoff"]).mean()   # coverage per cutoff
    scored = labelled[["cutoff", "y"]].assign(e=feats["e_n365d"])  # aligned on row index
    ap = scored.groupby("cutoff")[["y", "e"]].apply(ap_of)         # standalone AP
    return skrub.deferred(report_row)(name, n_rows, keyed, first, cov, ap)


names = sys.argv[1:] or [n for n in SPECS if n != "hpd_violations"]
lake = skrub.as_data_op(LAKE)
lots = lake.skb.apply_func(read_lots, ALL_CUTS)
viol = load_events("hpd_violations", lake)
labelled = skrub.deferred(attach_label)(lots[lots["cutoff"] < TEST_CUTOFF], viol)
base_rate = labelled.groupby("cutoff")["y"].agg(["size", "mean"])
report = skrub.deferred(to_frame)(*[table_audit(n, lake, lots, labelled) for n in names])
result = skrub.as_data_op({"base_rate": base_rate, "report": report})

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    out = result.skb.eval()
    print("rows / base rate per cutoff:\n", out["base_rate"], "\n")
    print(out["report"].to_string(index=False))
