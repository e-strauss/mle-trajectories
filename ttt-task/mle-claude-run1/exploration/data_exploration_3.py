"""Round 3: is a random sample of tracked domains representative of target.tsv?

target.tsv domains are 100% unseen in tracking_graph_train, but 90.8% of them
have >=1 TRACKED link-graph neighbour. If randomly-drawn *training* domains are
far less connected, a KFold over them would be a badly calibrated proxy for the
real task (and the model would be tuned for the wrong regime). Measure it.

Also checks the TLD distribution match and the n_labels match.
"""
from pathlib import Path

import numpy as np
import polars as pl

IN = Path(__file__).resolve().parent.parent / "input"
MOD, REM = 37, 11          # deterministic sample: domain_id % 37 == 11


def hr(t):
    print("\n" + "=" * 70, t, "=" * 70, sep="\n", flush=True)


tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"])
tg = tg.with_columns(pl.col("domain_id").cast(pl.Int32))
tracked = tg.select("domain_id").unique()
print("tracked domains:", tracked.shape[0])

samp = tracked.filter((pl.col("domain_id") % MOD) == REM)
print(f"deterministic sample (id %% {MOD} == {REM}):", samp.shape[0])

tgt = pl.read_csv(IN / "target.tsv", separator="\t").with_columns(
    pl.col("domain_id").cast(pl.Int32))

hr("link-graph connectivity: sample vs target")
ALL = pl.concat([samp.with_columns(pl.lit(0, dtype=pl.Int8).alias("grp")),
                 tgt.with_columns(pl.lit(1, dtype=pl.Int8).alias("grp"))])
lg = pl.scan_parquet(IN / "link-graph.parquet")
allids = ALL["domain_id"].implode()

out_e = lg.filter(pl.col("source_domain_id").is_in(allids)).collect(engine="streaming")
in_e = lg.filter(pl.col("target_domain_id").is_in(allids)).collect(engine="streaming")
print("out-edges:", out_e.shape[0], " in-edges:", in_e.shape[0])

trk_flag = tracked.with_columns(pl.lit(True).alias("is_trk"))


def cov(edges, self_col, nbr_col):
    e = edges.rename({self_col: "domain_id", nbr_col: "nbr"})
    e = e.filter(pl.col("domain_id") != pl.col("nbr"))          # drop self loops
    e = e.join(trk_flag.rename({"domain_id": "nbr"}), on="nbr", how="left") \
         .with_columns(pl.col("is_trk").fill_null(False))
    return e.group_by("domain_id").agg(
        pl.len().alias("deg"), pl.col("is_trk").sum().alias("n_trk_nbr"))


co = cov(out_e, "source_domain_id", "target_domain_id").rename(
    {"deg": "outdeg", "n_trk_nbr": "n_trk_out"})
ci = cov(in_e, "target_domain_id", "source_domain_id").rename(
    {"deg": "indeg", "n_trk_nbr": "n_trk_in"})
stats = (ALL.join(co, on="domain_id", how="left").join(ci, on="domain_id", how="left")
         .with_columns([pl.col(c).fill_null(0) for c in
                        ("outdeg", "n_trk_out", "indeg", "n_trk_in")]))

for g, lbl in ((0, "SAMPLE(train)"), (1, "TARGET")):
    s = stats.filter(pl.col("grp") == g)
    print(f"\n--- {lbl}  n={s.shape[0]}")
    print("  has out-edge      :", (s["outdeg"] > 0).mean())
    print("  has in-edge       :", (s["indeg"] > 0).mean())
    print("  >=1 tracked nbr   :", ((s["n_trk_out"] + s["n_trk_in"]) > 0).mean())
    print("  median outdeg/indeg:", s["outdeg"].median(), s["indeg"].median())
    print("  mean n_trk_out/in :", s["n_trk_out"].mean(), s["n_trk_in"].mean())

hr("TLD + n_labels match")
dom = pl.read_parquet(IN / "domains.parquet").with_columns(
    pl.col("domain_id").cast(pl.Int32))
dom = dom.with_columns(pl.col("domain").str.split(".").alias("_p")).with_columns(
    pl.col("_p").list.last().alias("tld"),
    pl.col("_p").list.len().alias("n_labels")).drop("_p")
j = ALL.join(dom, on="domain_id", how="left")
for g, lbl in ((0, "SAMPLE"), (1, "TARGET")):
    s = j.filter(pl.col("grp") == g)
    vc = s["tld"].value_counts(sort=True).head(10).with_columns(
        (pl.col("count") / s.shape[0]).alias("share"))
    print(f"\n--- {lbl} top TLDs"); print(vc)
    print("  n_labels share:", s["n_labels"].value_counts(sort=True).head(4).to_dicts())

hr("label stats on the sample")
ys = tg.join(samp, on="domain_id", how="semi")
pd_ = ys.group_by("domain_id").len().rename({"len": "n_trk"})
nn = pd_["n_trk"].to_numpy()
print("sample edges:", ys.shape[0], "mean n_trk:", nn.mean())
print("recall@10 ceiling:", (np.minimum(10, nn) / nn).mean())
pop = ys.group_by("tracker_id").len().sort("len", descending=True)
top10 = pop["tracker_id"].head(10).to_list()
hit = ys.filter(pl.col("tracker_id").is_in(top10)).group_by("domain_id").len().rename({"len": "h"})
r = pd_.join(hit, on="domain_id", how="left").with_columns(pl.col("h").fill_null(0))
print("global-top10 recall@10 on sample:", (r["h"] / r["n_trk"]).mean())
print("n distinct trackers used in sample:", ys["tracker_id"].n_unique())
