"""Round 4: BUILD THE FROZEN FEATURE STORE (run once; never re-run).

The raw data is far too big to cross-validate directly (18.7M tracked domains,
623M link edges), so per CLAUDE.md we build ONE deterministic working sample and
give every pipeline a shared recorded read of it. Comparability across the whole
workspace depends on these files never changing.

Row sample (deterministic, no randomness):
  train rows  = tracked domains with domain_id % 37 == 11   -> 505,548
  target rows = all 50,000 domains of target.tsv
Both sets sorted by domain_id, so row order is fixed forever.

Leakage discipline — every feature is computable for an unseen hostname, and no
feature ever reads the row's OWN trackers:
  * neighbour histograms exclude self-loops; a neighbour's trackers are given
    data (the tracking graph), not the row's label.
  * TLD tracker profiles are leave-one-out: the row's own label vector is
    subtracted from its TLD's counts.

Blocks written to ../features/  (train + target variants, same columns):
  labels.parquet      domain_id + y_0..y_354                (train only, uint8)
  nbr_out.parquet     domain_id + no_0..no_354              out-neighbour tracker counts
  nbr_in.parquet      domain_id + ni_0..ni_354              in-neighbour tracker counts
  tld_pop.parquet     domain_id + tp_0..tp_354              LOO P(tracker | TLD)
  meta.parquet        domain_id + hostname / degree / press-freedom / url-category
  global_pop.parquet  tracker_id + n_domains                (355 rows, full graph)
"""
import time
from pathlib import Path

import numpy as np
import polars as pl

WS = Path(__file__).resolve().parent.parent
IN = WS / "input"
OUT = WS / "features"
OUT.mkdir(exist_ok=True)

MOD, REM = 37, 11
N_TRK = 355
SLD = {"co", "com", "net", "org", "gov", "edu", "ac", "or", "ne", "gob", "in"}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- row sample
tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"]).with_columns(
    pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int32))
log("tracking graph edges:", tg.shape[0])

tracked = tg.select("domain_id").unique().sort("domain_id")
train_ids = tracked.filter((pl.col("domain_id") % MOD) == REM)["domain_id"].to_numpy()
target_ids = pl.read_csv(IN / "target.tsv", separator="\t").with_columns(
    pl.col("domain_id").cast(pl.Int32)).sort("domain_id")["domain_id"].to_numpy()
log("train rows:", len(train_ids), "target rows:", len(target_ids))

all_ids = np.concatenate([train_ids, target_ids])       # train block first
N = len(all_ids)
MAXID = int(max(tracked["domain_id"].max(), all_ids.max())) + 1
row_of = np.full(MAXID, -1, dtype=np.int32)
row_of[all_ids] = np.arange(N, dtype=np.int32)
IS_TRAIN = np.zeros(N, dtype=bool)
IS_TRAIN[: len(train_ids)] = True
log("store rows:", N, "maxid:", MAXID)


def save(arr, prefix, fname):
    """Split an (N, 355) array into the train / target parquet pair."""
    cols = [f"{prefix}_{i}" for i in range(N_TRK)]
    for tag, sl in (("train", slice(0, len(train_ids))),
                    ("target", slice(len(train_ids), N))):
        df = pl.DataFrame({"domain_id": all_ids[sl]}).hstack(
            pl.DataFrame(arr[sl], schema=cols))
        df.write_parquet(OUT / f"{fname}_{tag}.parquet", compression="zstd")
        log("wrote", f"{fname}_{tag}.parquet", df.shape)


# ---------------------------------------------------------------- labels
Y = np.zeros((N, N_TRK), dtype=np.uint8)
e = tg.filter(pl.col("domain_id").is_in(pl.Series(train_ids).implode()))
r = row_of[e["domain_id"].to_numpy()]
Y[r, e["tracker_id"].to_numpy()] = 1
log("label edges:", e.shape[0], "positives:", int(Y.sum()))
cols = [f"y_{i}" for i in range(N_TRK)]
pl.DataFrame({"domain_id": train_ids}).hstack(
    pl.DataFrame(Y[: len(train_ids)], schema=cols)).write_parquet(
    OUT / "labels_train.parquet", compression="zstd")
log("wrote labels_train.parquet")

# global popularity over the FULL graph (355 rows)
tg.group_by("tracker_id").len().rename({"len": "n_domains"}).sort(
    "tracker_id").write_parquet(OUT / "global_pop.parquet")

# ---------------------------------------------------------------- link graph
# tracker list per domain, as flat arrays, for fast neighbour aggregation
tg_sorted = tg.sort("domain_id")
tg_dom = tg_sorted["domain_id"].to_numpy()
tg_trk = tg_sorted["tracker_id"].to_numpy()
# CSR-style offsets over domain_id space
deg_all = np.bincount(tg_dom, minlength=MAXID).astype(np.int64)
offs = np.zeros(MAXID + 1, dtype=np.int64)
np.cumsum(deg_all, out=offs[1:])
log("built CSR over tracking graph")

allids_pl = pl.Series(all_ids).implode()
lg = pl.scan_parquet(IN / "link-graph.parquet")

degrees = {}
for direction, self_col, nbr_col in (("out", "source_domain_id", "target_domain_id"),
                                     ("in", "target_domain_id", "source_domain_id")):
    ed = lg.filter(pl.col(self_col).is_in(allids_pl)).collect(engine="streaming")
    self_id = ed[self_col].to_numpy()
    nbr_id = ed[nbr_col].to_numpy()
    keep = self_id != nbr_id                                   # drop self loops
    self_id, nbr_id = self_id[keep], nbr_id[keep]
    log(direction, "edges (self-loops dropped):", len(self_id))

    sr = row_of[self_id]
    degrees[f"{direction}deg"] = np.bincount(sr, minlength=N).astype(np.int32)
    # a neighbour's tracker count (0 => untracked)
    nbr_ok = (nbr_id >= 0) & (nbr_id < MAXID)
    nbr_deg = np.zeros(len(nbr_id), dtype=np.int64)
    nbr_deg[nbr_ok] = deg_all[nbr_id[nbr_ok]]
    degrees[f"n_trk_nbr_{direction}"] = np.bincount(
        sr[nbr_deg > 0], minlength=N).astype(np.int32)

    # expand each (self, nbr) edge into (self, tracker) pairs via the CSR
    rep = np.repeat(sr, nbr_deg)
    starts = np.zeros(len(nbr_id), dtype=np.int64)
    starts[nbr_ok] = offs[nbr_id[nbr_ok]]
    # positions inside each neighbour's tracker slice
    idx = np.repeat(starts, nbr_deg) + (
        np.arange(nbr_deg.sum(), dtype=np.int64)
        - np.repeat(np.concatenate([[0], np.cumsum(nbr_deg)[:-1]]), nbr_deg))
    trk = tg_trk[idx]
    log(direction, "(self, tracker) pairs:", len(trk))
    flat = np.bincount(rep.astype(np.int64) * N_TRK + trk, minlength=N * N_TRK)
    H = flat.reshape(N, N_TRK).astype(np.float32)
    del flat, rep, idx, trk, starts, nbr_deg, self_id, nbr_id, sr, ed
    save(H, "no" if direction == "out" else "ni",
         "nbr_out" if direction == "out" else "nbr_in")
    del H

# ---------------------------------------------------------------- hostnames
dom = pl.read_parquet(IN / "domains.parquet").with_columns(
    pl.col("domain_id").cast(pl.Int32))
dom = dom.with_columns(pl.col("domain").str.split(".").alias("_p")).with_columns(
    pl.col("_p").list.last().alias("tld"),
    pl.col("_p").list.len().cast(pl.Int32).alias("n_labels"),
    pl.col("domain").str.len_chars().cast(pl.Int32).alias("host_len"),
    pl.col("domain").str.count_matches(r"[0-9]").cast(pl.Int32).alias("n_digits"),
    pl.col("domain").str.count_matches("-").cast(pl.Int32).alias("n_hyphens"),
).with_columns(
    pl.when(pl.col("_p").list.get(-2, null_on_oob=True).is_in(list(SLD))
            & (pl.col("n_labels") >= 3))
    .then(pl.col("_p").list.slice(-3, 3).list.join("."))
    .otherwise(pl.col("_p").list.slice(-2, 2).list.join("."))
    .alias("reg_dom")).drop("_p")
log("hostname features built")

# ---------------------------------------------------------------- TLD profiles
tld_codes, tld_uniq = dom["tld"].to_physical(), None
tld_cat = dom.select(["domain_id", "tld"])
tld_map = np.full(MAXID, -1, dtype=np.int32)
uniq_tlds = tld_cat["tld"].unique().sort().to_list()
tld_index = {t: i for i, t in enumerate(uniq_tlds)}
sub = tld_cat.filter(pl.col("domain_id") < MAXID)
tld_map[sub["domain_id"].to_numpy()] = np.array(
    [tld_index.get(t, -1) for t in sub["tld"].to_list()], dtype=np.int32)
n_tld = len(uniq_tlds)
log("distinct TLDs:", n_tld)

# counts over ALL tracked domains
tld_of_edge = tld_map[tg_dom]
ok = tld_of_edge >= 0
TP = np.bincount(tld_of_edge[ok].astype(np.int64) * N_TRK + tg_trk[ok],
                 minlength=n_tld * N_TRK).reshape(n_tld, N_TRK).astype(np.float32)
tld_of_tracked = tld_map[tracked["domain_id"].to_numpy()]
n_dom_per_tld = np.bincount(tld_of_tracked[tld_of_tracked >= 0], minlength=n_tld)
log("tld profile matrix:", TP.shape)

row_tld = tld_map[all_ids]
TPR = np.zeros((N, N_TRK), dtype=np.float32)
has = row_tld >= 0
TPR[has] = TP[row_tld[has]]
denom = n_dom_per_tld[np.where(has, row_tld, 0)].astype(np.float32)
# leave-one-out: train rows are inside their TLD's counts, target rows are not
TPR[IS_TRAIN] -= Y[IS_TRAIN]
denom = np.where(IS_TRAIN, denom - 1.0, denom)
TPR /= np.maximum(denom, 1.0)[:, None]
np.clip(TPR, 0.0, None, out=TPR)
save(TPR, "tp", "tld_pop")
del TPR, TP

# ---------------------------------------------------------------- meta block
fp = pl.read_csv(IN / "freedom-of-the-press.csv", separator="\t")
uc = pl.read_csv(IN / "url-classification.csv", infer_schema_length=10000)
uc = uc.with_columns(
    pl.col("url").str.replace(r"^https?://", "").str.split("/").list.first()
    .str.replace(r"^www\.", "").str.to_lowercase().alias("host")
).select(["host", "category"]).unique(subset=["host"]).rename({"category": "uc_category"})

meta = pl.DataFrame({"domain_id": all_ids}).join(dom, on="domain_id", how="left")
meta = meta.with_columns([
    pl.Series(k, v) for k, v in degrees.items()
])
meta = meta.join(fp.select(["tld", "freedom_of_the_press"]), on="tld", how="left")
meta = meta.with_columns(pl.col("domain").str.replace(r"^www\.", "").alias("_h")) \
           .join(uc, left_on="_h", right_on="host", how="left").drop("_h")
meta = meta.with_columns(
    (pl.col("n_trk_nbr_out") + pl.col("n_trk_nbr_in")).alias("n_trk_nbr_tot"),
    pl.Series("is_train", IS_TRAIN),
)
log("meta cols:", meta.columns)
for tag, sl in (("train", slice(0, len(train_ids))), ("target", slice(len(train_ids), N))):
    meta[sl].drop("is_train").write_parquet(OUT / f"meta_{tag}.parquet",
                                            compression="zstd")
    log("wrote", f"meta_{tag}.parquet", meta[sl].shape)

log("DONE")
print("\nfiles:")
for p in sorted(OUT.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size / 1e6:8.1f} MB")
