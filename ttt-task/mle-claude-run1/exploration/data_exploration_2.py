"""Round 2: what signal exists for a COLD-START domain?

Target domains never appear in tracking_graph_train, so every feature must come
from data that is available for an unseen hostname:
  (a) the hostname string itself (TLD, registrable domain, tokens)
  (b) the link graph (its neighbours' trackers ARE known)
  (c) url-classification category, press-freedom by TLD

This round measures the *coverage and lift* of each, before committing to a
feature build.
"""
from pathlib import Path

import numpy as np
import polars as pl

IN = Path(__file__).resolve().parent.parent / "input"

# second-level suffixes that need 3 labels for the registrable domain
SLD = {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "gob", "in"}


def hr(t):
    print("\n" + "=" * 70, t, "=" * 70, sep="\n", flush=True)


tgt = pl.read_csv(IN / "target.tsv", separator="\t")
tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"])
per_dom = tg.group_by("domain_id").len().rename({"len": "n_trk"})

hr("recall@10 ceiling + n_trk tail")
n = per_dom["n_trk"].to_numpy()
print("edges:", tg.shape[0], "domains:", n.size)
print("ceiling mean(min(10,n)/n):", np.minimum(10, n).mean() / 1.0 and (np.minimum(10, n) / n).mean())
for q in (0.5, 0.9, 0.99, 0.999, 1.0):
    print(f"  q{q}: {np.quantile(n, q)}")
print("share of domains with n<=10:", (n <= 10).mean())

hr("hostnames: target vs train")
dom = pl.read_parquet(IN / "domains.parquet")


def add_host_feats(df):
    return df.with_columns(
        pl.col("domain").str.split(".").alias("_p")
    ).with_columns(
        pl.col("_p").list.last().alias("tld"),
        pl.col("_p").list.len().alias("n_labels"),
        pl.col("domain").str.len_chars().alias("host_len"),
    ).with_columns(
        # registrable domain: last 2 labels, or 3 when the 2nd-to-last is an SLD
        pl.when(pl.col("_p").list.get(-2, null_on_oob=True).is_in(list(SLD)) & (pl.col("n_labels") >= 3))
        .then(pl.col("_p").list.slice(-3, 3).list.join("."))
        .otherwise(pl.col("_p").list.slice(-2, 2).list.join("."))
        .alias("reg_dom")
    ).drop("_p")


dom = add_host_feats(dom)
tgt_d = tgt.join(dom, on="domain_id", how="left")
print("target sample hostnames:")
print(tgt_d.select(["domain_id", "domain", "tld", "reg_dom"]).head(10))
print("target null hostname:", tgt_d["domain"].null_count())
print("\ntarget TLD dist:")
print(tgt_d["tld"].value_counts(sort=True).head(12))

train_d = per_dom.join(dom, on="domain_id", how="left")
print("\ntrain-graph TLD dist:")
print(train_d["tld"].value_counts(sort=True).head(12))
print("\nn_labels: target vs train")
print(tgt_d["n_labels"].value_counts(sort=True).head(6))
print(train_d["n_labels"].value_counts(sort=True).head(6))
print("\nhost_len mean: target", tgt_d["host_len"].mean(), "train", train_d["host_len"].mean())

hr("registrable-domain siblings: does a target share reg_dom with a TRACKED domain?")
train_reg = train_d.select("reg_dom").unique()
tgt_reg_hit = tgt_d.join(train_reg.with_columns(pl.lit(True).alias("hit")),
                         on="reg_dom", how="left")
print("target domains whose reg_dom appears in the tracking graph:",
      tgt_reg_hit["hit"].fill_null(False).mean())
# how big are those sibling groups
sib = train_d.group_by("reg_dom").len().rename({"len": "n_sib"})
j = tgt_d.join(sib, on="reg_dom", how="left")
print("n_sib describe for targets with a hit:")
print(j.filter(pl.col("n_sib").is_not_null())["n_sib"].describe())

hr("link graph: degree of target domains, and how many neighbours are tracked")
lg = pl.scan_parquet(IN / "link-graph.parquet")
tgt_ids = tgt["domain_id"].cast(pl.Int32)
out_e = (lg.filter(pl.col("source_domain_id").is_in(tgt_ids))
         .collect(engine="streaming"))
in_e = (lg.filter(pl.col("target_domain_id").is_in(tgt_ids))
        .collect(engine="streaming"))
print("out-edges from target domains:", out_e.shape[0])
print("in-edges into target domains:", in_e.shape[0])
outdeg = out_e.group_by("source_domain_id").len()
indeg = in_e.group_by("target_domain_id").len()
print("targets with >=1 out-edge:", outdeg.shape[0] / 50000,
      " >=1 in-edge:", indeg.shape[0] / 50000)
print("outdeg describe:"); print(outdeg["len"].describe())
print("indeg describe:"); print(indeg["len"].describe())

tracked = per_dom.select(pl.col("domain_id").cast(pl.Int32)).with_columns(pl.lit(True).alias("t"))
on = (out_e.rename({"target_domain_id": "domain_id"})
      .join(tracked, on="domain_id", how="left"))
print("share of out-neighbours that are tracked:", on["t"].fill_null(False).mean())
ni = (in_e.rename({"source_domain_id": "domain_id"})
      .join(tracked, on="domain_id", how="left"))
print("share of in-neighbours that are tracked:", ni["t"].fill_null(False).mean())
cov_out = on.filter(pl.col("t")).select("source_domain_id").unique().shape[0] / 50000
cov_in = ni.filter(pl.col("t")).select("target_domain_id").unique().shape[0] / 50000
print("targets with >=1 TRACKED out-neighbour:", cov_out)
print("targets with >=1 TRACKED in-neighbour:", cov_in)
both = set(on.filter(pl.col("t"))["source_domain_id"].to_list()) | \
       set(ni.filter(pl.col("t"))["target_domain_id"].to_list())
print("targets with >=1 tracked neighbour (either dir):", len(both) / 50000)

hr("url-classification coverage on target domains")
uc = pl.read_csv(IN / "url-classification.csv", infer_schema_length=10000)
uc = uc.with_columns(
    pl.col("url").str.replace(r"^https?://", "").str.split("/").list.first()
    .str.replace(r"^www\.", "").str.to_lowercase().alias("host")
).select(["host", "category"]).unique(subset=["host"])
print("unique hosts in url-classification:", uc.shape[0])
tgt_uc = tgt_d.with_columns(pl.col("domain").str.replace(r"^www\.", "").alias("host")) \
              .join(uc, on="host", how="left")
print("target coverage by url-classification:", tgt_uc["category"].is_not_null().mean())
print("via reg_dom:", tgt_d.join(uc.rename({"host": "reg_dom"}), on="reg_dom", how="left")["category"].is_not_null().mean())
