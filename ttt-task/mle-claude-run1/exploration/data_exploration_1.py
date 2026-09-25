"""Round 1: what is actually in input/ — shapes, key overlaps, target structure.

Goal: understand the prediction problem (cold-start? partially-observed?) and
decide what training rows / features / CV are even possible.
"""
from pathlib import Path

import polars as pl

IN = Path(__file__).resolve().parent.parent / "input"


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


hr("trackers.tsv")
trk = pl.read_csv(IN / "trackers.tsv", separator="\t")
print(trk.shape, trk.columns)
print(trk.head(5))
print("n unique tracking_domain_id:", trk["tracking_domain_id"].n_unique())
print("n unique tracker_id:", trk["tracker_id"].n_unique())
print(trk["category"].value_counts(sort=True))
print(trk["company"].value_counts(sort=True).head(15))
print("country nulls:", trk["country"].null_count(), "brand nulls:", trk["brand"].null_count())

hr("target.tsv")
tgt = pl.read_csv(IN / "target.tsv", separator="\t")
print(tgt.shape, tgt.columns)
print(tgt.head())
print("unique:", tgt["domain_id"].n_unique())

hr("tracking_graph_train.parquet")
tg = pl.read_parquet(IN / "tracking_graph_train.parquet")
print(tg.shape, tg.columns, tg.dtypes)
print(tg.head())
print("unique domains:", tg["domain_id"].n_unique())
print("unique trackers:", tg["tracking_domain_id"].n_unique(), tg["tracker_id"].n_unique())
print("duplicate (domain,tracker) rows:", tg.shape[0] - tg.select(["domain_id", "tracker_id"]).n_unique())

per_dom = tg.group_by("domain_id").len().rename({"len": "n_trk"})
print("\ntrackers-per-domain describe:")
print(per_dom["n_trk"].describe())
print(per_dom["n_trk"].value_counts(sort=True).head(15))

hr("target vs train overlap")
tgt_ids = set(tgt["domain_id"].to_list())
tg_doms = set(tg["domain_id"].to_list())
print("target domains:", len(tgt_ids))
print("target domains present in tracking_graph_train:", len(tgt_ids & tg_doms))
print("train domains total:", len(tg_doms))

hr("per-tracker popularity")
pop = tg.group_by("tracker_id").len().sort("len", descending=True)
print(pop.head(20).join(trk.select(["tracker_id", "domain", "category"]), on="tracker_id", how="left"))
n_dom = len(tg_doms)
print("\ntop-10 trackers cover share of all (domain,tracker) edges:",
      pop["len"].head(10).sum() / tg.shape[0])
# naive recall@10 upper bound estimate: predicting global top-10 for everyone
top10 = pop["tracker_id"].head(10).to_list()
hit = tg.filter(pl.col("tracker_id").is_in(top10)).group_by("domain_id").len().rename({"len": "hits"})
rec = per_dom.join(hit, on="domain_id", how="left").with_columns(
    pl.col("hits").fill_null(0)
).with_columns((pl.col("hits") / pl.col("n_trk")).alias("rec"))
print("global-top10 recall@10 over train domains:", rec["rec"].mean())
# ceiling: domains with >10 trackers cannot reach recall 1
print("mean min(10,n)/n ceiling:", (pl.min_horizontal(pl.lit(10), per_dom["n_trk"]) / per_dom["n_trk"]).mean())

hr("domains.parquet")
dom = pl.read_parquet(IN / "domains.parquet")
print(dom.shape, dom.columns, dom.dtypes)
print(dom.head())
print("unique domain_id:", dom["domain_id"].n_unique())

hr("url-classification.csv")
uc = pl.read_csv(IN / "url-classification.csv", infer_schema_length=10000)
print(uc.shape, uc.columns)
print(uc.head())
print(uc["category"].value_counts(sort=True).head(30))

hr("freedom-of-the-press.csv")
fp = pl.read_csv(IN / "freedom-of-the-press.csv")
print(fp.shape, fp.columns)
print(fp.head())

hr("link-graph.parquet (schema + row count only)")
lg_lazy = pl.scan_parquet(IN / "link-graph.parquet")
print(lg_lazy.collect_schema())
print("rows:", lg_lazy.select(pl.len()).collect().item())
