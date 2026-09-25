"""Round 5: WHERE does the current model lose recall? (loss attribution)

Refits pipeline_03's Ridge on fold 0 of the workspace CV (outside the harness --
exploration only, nothing is recorded) and attributes the missing recall to
buckets of (number of tracked link neighbours) and (number of true trackers).
That says which feature to build next instead of guessing.

Also checks the payoff of a possible 2-hop block: do the domains with no tracked
1-hop neighbour have tracked neighbours two hops away?
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

WS = Path(__file__).resolve().parent.parent
FEAT = WS / "features"
IN = WS / "input"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


N_TRK = 355
lab = pd.read_parquet(FEAT / "labels_train.parquet")
Y = lab[[f"y_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
dom_ids = lab["domain_id"].to_numpy()
meta = pd.read_parquet(FEAT / "meta_train.parquet")
log("labels", Y.shape)

blocks = []
for name, pref in (("nbr_out", "no_"), ("nbr_in", "ni_"), ("tld_pop", "tp_")):
    b = pd.read_parquet(FEAT / f"{name}_train.parquet")
    M = b[[f"{pref}{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)
    if pref != "tp_":
        M = M / np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    blocks.append(M)
Xf = np.hstack(blocks)
del blocks
log("features", Xf.shape)

tr, te = next(iter(KFold(3, shuffle=True, random_state=42).split(Xf)))
sc = StandardScaler().fit(Xf[tr])
r = Ridge(alpha=1.0, solver="cholesky").fit(sc.transform(Xf[tr]), Y[tr])
S = r.predict(sc.transform(Xf[te]))
log("ridge fitted; fold-0 test rows:", len(te))

top = np.argpartition(-S, 9, axis=1)[:, :10]
yt = Y[te].astype(np.int32)
n_true = yt.sum(1)
hits = np.take_along_axis(yt, top, axis=1).sum(1)
rec = hits / n_true
print(f"\nfold-0 recall@10 = {rec.mean():.5f}   (loss = {1 - rec.mean():.5f})")

mt = meta.iloc[te]
total_loss = (1 - rec).sum()


def attribute(name, bucket):
    print(f"\n--- by {name}")
    df = pd.DataFrame({"b": np.asarray(bucket), "rec": rec, "n": 1})
    g = df.groupby("b", observed=True).agg(rows=("n", "sum"), recall=("rec", "mean"))
    g["share_rows"] = g["rows"] / len(rec)
    g["loss"] = (1 - g["recall"]) * g["rows"]
    g["share_of_loss"] = g["loss"] / total_loss
    print(g.round(4).to_string())


attribute("n tracked link-neighbours",
          pd.cut(mt["n_trk_nbr_tot"].to_numpy(), [-1, 0, 1, 2, 5, 20, 100, 10 ** 9],
                 labels=["0", "1", "2", "3-5", "6-20", "21-100", ">100"]))
attribute("n true trackers",
          pd.cut(n_true, [0, 1, 2, 3, 5, 10, 10 ** 9],
                 labels=["1", "2", "3", "4-5", "6-10", ">10"]))
attribute("TLD", np.where(mt["tld"].isin(["com", "net", "org", "ru", "de", "uk"]),
                          mt["tld"], "other"))

# how often is the true tracker present in the 1-hop histogram at all?
no = pd.read_parquet(FEAT / "nbr_out_train.parquet")[
    [f"no_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)[te]
ni = pd.read_parquet(FEAT / "nbr_in_train.parquet")[
    [f"ni_{i}" for i in range(N_TRK)]].to_numpy(dtype=np.float32)[te]
seen = ((no + ni) > 0)
cov = (yt * seen).sum(1) / n_true
print(f"\nmean fraction of a domain's true trackers that appear anywhere in its "
      f"1-hop neighbour histogram: {cov.mean():.4f}")
k = seen.sum(1)
print("mean distinct trackers in the 1-hop histogram:", k.mean(),
      " median:", np.median(k))
# ceiling if we could always pick the right 10 out of the 1-hop support
print("recall ceiling of a perfect ranker restricted to 1-hop support + top-10 pop:",
      np.mean(np.minimum(cov, 1.0)))
del no, ni

# ---- would a 2-hop block reach the unreachable domains? -------------------
log("checking 2-hop reach for domains with no tracked 1-hop neighbour")
tg = pl.read_parquet(IN / "tracking_graph_train.parquet", columns=["domain_id"])
tracked = tg.select(pl.col("domain_id").cast(pl.Int32)).unique()
dead = dom_ids[te][mt["n_trk_nbr_tot"].to_numpy() == 0]
print("dead-end test domains:", len(dead))
lg = pl.scan_parquet(IN / "link-graph.parquet")
seed = pl.Series(dead.astype(np.int32)).implode()
h1 = pl.concat([
    lg.filter(pl.col("source_domain_id").is_in(seed))
      .select(pl.col("target_domain_id").alias("n")).collect(engine="streaming"),
    lg.filter(pl.col("target_domain_id").is_in(seed))
      .select(pl.col("source_domain_id").alias("n")).collect(engine="streaming"),
]).unique()
print("distinct 1-hop neighbours of dead-ends:", h1.shape[0])
s1 = h1["n"].implode()
h2 = pl.concat([
    lg.filter(pl.col("source_domain_id").is_in(s1))
      .select(pl.col("target_domain_id").alias("n")).collect(engine="streaming"),
    lg.filter(pl.col("target_domain_id").is_in(s1))
      .select(pl.col("source_domain_id").alias("n")).collect(engine="streaming"),
]).unique()
print("distinct 2-hop neighbours:", h2.shape[0],
      " of which tracked:", h2.join(tracked, left_on="n", right_on="domain_id",
                                    how="semi").shape[0])
log("done")
