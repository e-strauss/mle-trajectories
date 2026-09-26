"""Exploration 6: audit of the owner-portfolio block (pipeline_30).

  - coverage per cutoff INCLUDING test (22v3 @ 2023): share of lots with an owner,
    share in a multi-lot portfolio (>= 2 lots), portfolio-size quantiles,
  - standalone AP of each owner feature per training cutoff,
  - largest portfolios as of the 2022 cutoff (sanity check of name normalization).
Fine-grained skrub DataOps, one output node evaluated once. Writes nothing.
"""
import pandas as pd
import skrub
from sklearn.metrics import average_precision_score

from common import (CUTOFFS, LAKE, TEST_CUTOFF, TEST_RELEASE, attach_label, load_acris,
                    load_events, owner_portfolio, read_acris_grantees, read_lots)

ALL_CUTS = {**CUTOFFS, TEST_CUTOFF: TEST_RELEASE}


def coverage(feats):
    g = feats.groupby("cutoff")["owner_n_lots"]
    return pd.DataFrame({"has_owner": g.apply(lambda s: s.notna().mean()),
                         "portfolio>=2": g.apply(lambda s: (s >= 2).mean()),
                         "portfolio>=10": g.apply(lambda s: (s >= 10).mean()),
                         "median_size": g.median(), "p99_size": g.quantile(0.99)}).round(4)


def standalone_ap(scored):
    cols = [c for c in scored.columns if c.startswith("owner_")]
    return pd.DataFrame({cut.year: {c: average_precision_score(d["y"], d[c].fillna(0))
                                    for c in cols}
                         for cut, d in scored.groupby("cutoff")}).round(4)


def top_owners(acris, grantees, cut=pd.Timestamp("2022-01-01"), n=15):
    d = acris[(acris["cat"] == "DEED") & (acris["date"] < cut)]
    d = d.sort_values("date").drop_duplicates("bbl", keep="last")
    first = grantees.sort_values("owner").drop_duplicates("document_id")
    return d.merge(first, on="document_id")["owner"].value_counts().head(n)


lake = skrub.as_data_op(LAKE)
lots = lake.skb.apply_func(read_lots, ALL_CUTS)
viol = load_events("hpd_violations", lake)
acris = load_acris(lake)
grantees = lake.skb.apply_func(read_acris_grantees)
labelled = skrub.deferred(attach_label)(lots[lots["cutoff"] < TEST_CUTOFF], viol)
feats = skrub.deferred(owner_portfolio)(lots, acris, grantees, viol)
scored = feats.loc[labelled.index].assign(y=labelled["y"])

result = skrub.as_data_op({
    "coverage per cutoff": skrub.deferred(coverage)(feats),
    "standalone AP per training cutoff": skrub.deferred(standalone_ap)(scored),
    "largest owners as of 2022-01-01 (all lots, latest deed)": skrub.deferred(top_owners)(
        acris, grantees),
})

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for k, v in result.skb.eval().items():
        print(f"== {k}\n{v}\n")
