"""Build a small, self-consistent sample of the trackthetrackers input folder.

Writes ttt-task/input/ (gitignored) from the full task data, so a skrubified
pipeline and its original can be run side by side in ~1 minute instead of hours.

Random domain sampling would destroy the link graph (623M edges over 46M nodes:
two random domains are almost never connected), so the sample is a NEIGHBOURHOOD:
seed domains plus their link-graph neighbours that are themselves tracked, then
every edge whose endpoints both survive. That keeps the graph-derived features
(degrees, direct tracker links, neighbour tracker adoption) non-trivial.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = Path(os.environ.get("TTT_SRC", "/home/estrauss-ldap/repos/mle-star/machine_learning_engineering/tasks/trackthetrackers-task"))
OUT = Path(os.environ.get("TTT_OUT", Path(__file__).resolve().parent / "input"))

N_SEED = int(os.environ.get("TTT_N_SEED", 20_000))      # tracked domains to grow the neighbourhood from
N_DOMAINS = int(os.environ.get("TTT_N_DOMAINS", 20_000))  # cap on sampled candidate (training) domains
N_TARGET = int(os.environ.get("TTT_N_TARGET", 2_000))     # rows kept from target.tsv

OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

print("reading trackers/target ...", flush=True)
trackers = pd.read_csv(f"{SRC}/trackers.tsv", sep="\t")
tracker_tdids = trackers["tracking_domain_id"].to_numpy(np.int64)
target_all = pd.read_csv(f"{SRC}/target.tsv", sep="\t")["domain_id"].to_numpy(np.int64)

print("reading tracking graph ...", flush=True)
tg = pq.read_table(f"{SRC}/tracking_graph_train.parquet").to_pandas()
candidates = np.unique(tg["domain_id"].to_numpy(np.int64))
print(f"  {len(tg)} rows, {len(candidates)} candidate domains", flush=True)

print("reading link graph ...", flush=True)
lg = pq.read_table(f"{SRC}/link-graph.parquet")
src = lg.column(0).to_numpy().astype(np.int64)
dst = lg.column(1).to_numpy().astype(np.int64)
del lg
print(f"  {len(src)} edges", flush=True)

seeds = rng.choice(candidates, N_SEED, replace=False)
seeds.sort()
touch = np.isin(src, seeds) | np.isin(dst, seeds)
nbr = np.unique(np.concatenate([src[touch], dst[touch]]))
print(f"  seeds {len(seeds)} -> touching edges {touch.sum()}, neighbours {len(nbr)}", flush=True)

nbr_tracked = np.intersect1d(nbr, candidates)
pool = np.union1d(seeds, nbr_tracked)
print(f"  tracked neighbours {len(nbr_tracked)}, pool {len(pool)}", flush=True)
if len(pool) > N_DOMAINS:
    pool = np.sort(rng.choice(pool, N_DOMAINS, replace=False))

target_sample = np.sort(rng.choice(target_all, N_TARGET, replace=False))
selected = np.union1d(pool, target_sample)

keep_nodes = np.union1d(selected, tracker_tdids)
emask = np.isin(src, keep_nodes) & np.isin(dst, keep_nodes)
sub_src, sub_dst = src[emask], dst[emask]
print(f"  sampled edges {len(sub_src)}", flush=True)
del src, dst, emask, touch

edge_nodes = np.unique(np.concatenate([sub_src, sub_dst]))
need_names = np.union1d(np.union1d(selected, tracker_tdids), edge_nodes)
print(f"  domains needing a name: {len(need_names)}", flush=True)

print("writing link-graph / tracking-graph / target / trackers ...", flush=True)
pd.DataFrame({"source_domain_id": sub_src.astype(np.int32),
              "target_domain_id": sub_dst.astype(np.int32)}
             ).to_parquet(f"{OUT}/link-graph.parquet", index=False)

tg_sub = tg[tg["domain_id"].isin(pd.Index(selected))].reset_index(drop=True)
tg_sub.to_parquet(f"{OUT}/tracking_graph_train.parquet", index=False)
print(f"  tracking graph rows {len(tg_sub)}, domains {tg_sub['domain_id'].nunique()}", flush=True)
del tg

pd.DataFrame({"domain_id": target_sample}).to_csv(f"{OUT}/target.tsv", sep="\t", index=False)
trackers.to_csv(f"{OUT}/trackers.tsv", sep="\t", index=False)

print("reading domains ...", flush=True)
dom = pq.read_table(f"{SRC}/domains.parquet").to_pandas()
dom_sub = dom[dom["domain_id"].isin(pd.Index(need_names))].reset_index(drop=True)
dom_sub.to_parquet(f"{OUT}/domains.parquet", index=False)
print(f"  domains rows {len(dom_sub)}", flush=True)
names = set(dom_sub["domain"].dropna().astype(str))
del dom

print("filtering url-classification ...", flush=True)
kept = []
for chunk in pd.read_csv(f"{SRC}/url-classification.csv", chunksize=500_000):
    u = chunk["url"].astype(str)
    host = (u.str.split("://").str[-1].str.split("/").str[0]
             .str.split(":").str[0].str.strip().str.lower())
    hit = host.isin(names) | host.str.removeprefix("www.").isin(names)
    kept.append(chunk[hit])
url_sub = pd.concat(kept, ignore_index=True)
url_sub.to_csv(f"{OUT}/url-classification.csv", index=False)
print(f"  url rows {len(url_sub)}", flush=True)

shutil.copy(f"{SRC}/freedom-of-the-press.csv", f"{OUT}/freedom-of-the-press.csv")
print(f"done -> {OUT}", flush=True)
