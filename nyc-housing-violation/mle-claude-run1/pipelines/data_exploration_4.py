"""Exploration 4: ownership (ACRIS) and violation-type (HPD order codes) signals.

ACRIS: document types recorded 2017-2022 (+ amounts), legals -> multi-dwelling
lot match rate, party types. HPD: for lots with a Class C violation in 2021, the
2022 positive rate per top order number of that C violation (which C types
recur?), and the same for B/A order codes among lots WITHOUT a 2021 C.

Fine-grained skrub DataOps, one output node evaluated once with .skb.eval().
Reads the lake, writes nothing.
"""
import pandas as pd
import skrub

from common import LAKE, STORAGE, attach_label, bbl_key, load_events, read_lots

T = pd.Timestamp("2022-01-01")


def read_master(lake):
    return pd.read_parquet(f"{lake}/acris_master", storage_options=STORAGE,
                           columns=["document_id", "doc_type", "document_amt",
                                    "recorded_datetime"],
                           filters=[("year", ">=", 2017)])


def read_legals(lake):
    return pd.read_parquet(f"{lake}/acris_legals", storage_options=STORAGE,
                           columns=["document_id", "borough", "block", "lot",
                                    "property_type"])


def read_parties_sample(lake):
    return pd.read_parquet(f"{lake}/acris_parties", storage_options=STORAGE,
                           columns=["document_id", "party_type"]).sample(
        2_000_000, random_state=0)


def read_orders(lake):
    return pd.read_parquet(f"{lake}/hpd_violations", storage_options=STORAGE,
                           columns=["bbl", "boroid", "block", "lot", "class",
                                    "inspectiondate", "ordernumber", "novdescription"],
                           filters=[("year", "==", 2021)])


def recurrence(orders, labelled, cls, among):
    """2022 positive rate of lots grouped by the 2021 order numbers of class `cls`."""
    o = orders[orders["class"] == cls]
    lots = labelled[labelled["bbl"].isin(among)]
    pairs = o[o["key"].isin(set(lots["bbl"]))].drop_duplicates(["key", "ordernumber"])
    pairs = pairs.merge(lots[["bbl", "y"]], left_on="key", right_on="bbl")
    desc = o.groupby("ordernumber")["novdescription"].first().str[:60]
    r = pairs.groupby("ordernumber")["y"].agg(lots="size", rate_2022="mean")
    return r[r["lots"] >= 200].join(desc).sort_values("lots", ascending=False).head(15).round(3)


lake = skrub.as_data_op(LAKE)
master = lake.skb.apply_func(read_master)
legals = lake.skb.apply_func(read_legals)
legals = legals.assign(bbl=legals.skb.apply_func(bbl_key, boro="borough", block="block",
                                                  lot="lot"))
lots = lake.skb.apply_func(read_lots, {T: "21v4"})
multi = lots["bbl"]

# ACRIS
recent = master[master["recorded_datetime"] < pd.Timestamp("2023-01-01")]
doc_types = recent.groupby("doc_type")["document_amt"].agg(docs="size", median_amt="median")
doc_types = doc_types.sort_values("docs", ascending=False).head(20)
on_multi = legals[legals["bbl"].isin(multi)]
legal_match = skrub.as_data_op({"legals_rows": legals["bbl"].size,
                                "share_on_multi_dwelling_lot": legals["bbl"].isin(multi).mean(),
                                "multi_lots_with_any_doc": multi.isin(on_multi["bbl"]).mean()})
docs_on_multi = recent[recent["document_id"].isin(on_multi["document_id"])]
multi_doc_types = docs_on_multi["doc_type"].value_counts().head(15)
party_types = lake.skb.apply_func(read_parties_sample)["party_type"].value_counts()

# HPD order codes
viol = load_events("hpd_violations", lake)
labelled = skrub.deferred(attach_label)(lots, viol)
orders = lake.skb.apply_func(read_orders)
orders = orders.assign(key=orders.skb.apply_func(bbl_key, bbl="bbl", boro="boroid",
                                                  block="block", lot="lot"))
c2021 = orders.loc[orders["class"] == "C", "key"]
no_c = labelled.loc[~labelled["bbl"].isin(c2021), "bbl"]
c_recur = skrub.deferred(recurrence)(orders, labelled, "C", labelled["bbl"])
b_no_c = skrub.deferred(recurrence)(orders, labelled, "B", no_c)
base = labelled.assign(had_c=labelled["bbl"].isin(c2021)).groupby("had_c")["y"].mean()

result = skrub.as_data_op({
    "acris doc types 2017-2022 (top 20)": doc_types,
    "acris legals match": legal_match,
    "acris doc types on multi-dwelling lots": multi_doc_types,
    "acris party types (2M sample)": party_types,
    "2022 rate by had-C-in-2021": base,
    "C order codes 2021 -> 2022 rate": c_recur,
    "B order codes 2021, lots without 2021 C -> 2022 rate": b_no_c,
})

if __name__ == "__main__":
    pd.set_option("display.width", 220, "display.max_colwidth", 60)
    for k, v in result.skb.eval().items():
        print(f"== {k}\n{v}\n")
