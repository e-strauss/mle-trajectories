"""Round 7: build THREE new graph feature blocks (additive -- rows/labels untouched).

data_exploration_5 showed the 1-hop histogram contains only 79.7% of a domain's
true trackers, and that loss is spread across all neighbourhood sizes: thin
neighbourhoods lack evidence, huge ones are dominated by hubs. These blocks
attack both ends. Adding block FILES is safe for comparability: the frozen row
sample and labels in ../features/ are not touched, pipelines just opt in via
`load_xy(blocks=...)`.

  nbr_hub  nh_0..nh_354   hub-discounted neighbour histogram. Every link
                          (either direction) contributes its neighbour's
                          trackers with weight 1/log2(2 + linkdeg(neighbour)),
                          so a 50k-outlink directory counts far less than a
                          hand-made link between two real sites.
  nbr_rec  nr_0..nr_354   trackers of RECIPROCAL neighbours only (d->n and
                          n->d). Mutual links are a much stronger similarity
                          signal than one-way links.
  nbr_2h   n2_0..n2_354   2-hop reach THROUGH UNTRACKED, low-degree
                          intermediaries (linkdeg <= 32). Tracked 1-hop
                          neighbours are already in nbr_out/nbr_in, so routing
                          only through untracked ones adds strictly new
                          evidence -- exactly what the 8.4% of domains with no
                          tracked neighbour need (exploration_5 found 2.15M
                          tracked domains two hops from those dead ends).
  meta2                   scalar context for the above: reciprocal-neighbour
                          count, untracked-neighbour count, hub-weight mass,
                          neighbour-degree summaries, 2-hop evidence mass.

Leakage discipline: self-loops are dropped everywhere, a domain's own trackers
never enter its own features, and every block is computable for an unseen
hostname from the given link + tracking graphs.

NOTE -- the 2-hop expansion below carries a subtle trap that the first version of
this script fell into: because `mid` is reached FROM the seed, the seed is itself
one of mid's neighbours, so walking mid's edges hands the seed its own label
vector back. data_exploration_8 caught it (own-tracker coverage of the block was
exactly 1.0000) and the `keep2` mask below is the fix. Rerunning this script
rewrites nbr_2h/meta2 and therefore invalidates any pipeline that scored against
the buggy version -- pipeline_07 was re-scored afterwards; pipelines 01-06 never
read these two blocks and are unaffected.
"""
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

WS = Path(__file__).resolve().parent.parent
IN = WS / "input"
OUT = WS / "features"
MOD, REM = 37, 11
N_TRK = 355
MAX_MID_DEG = 32          # intermediaries above this are hubs, not signal
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------- the frozen row order
tg = pl.read_parquet(IN / "tracking_graph_train.parquet",
                     columns=["domain_id", "tracker_id"]).with_columns(
    pl.col("domain_id").cast(pl.Int32), pl.col("tracker_id").cast(pl.Int32))
tracked = tg.select("domain_id").unique().sort("domain_id")
train_ids = tracked.filter((pl.col("domain_id") % MOD) == REM)["domain_id"].to_numpy()
target_ids = pl.read_csv(IN / "target.tsv", separator="\t").with_columns(
    pl.col("domain_id").cast(pl.Int32)).sort("domain_id")["domain_id"].to_numpy()
all_ids = np.concatenate([train_ids, target_ids])
N, N_TRAIN = len(all_ids), len(train_ids)
MAXID = int(max(tracked["domain_id"].max(), all_ids.max())) + 1
row_of = np.full(MAXID, -1, dtype=np.int32)
row_of[all_ids] = np.arange(N, dtype=np.int32)
log("rows:", N, "(train", N_TRAIN, ")")

# CSR of the tracking graph: domain -> its trackers
tg_sorted = tg.sort("domain_id")
tg_dom, tg_trk = tg_sorted["domain_id"].to_numpy(), tg_sorted["tracker_id"].to_numpy()
n_trk_of = np.bincount(tg_dom, minlength=MAXID).astype(np.int32)
offs = np.zeros(MAXID + 1, dtype=np.int64)
np.cumsum(n_trk_of, out=offs[1:])
IS_TRACKED = n_trk_of > 0
log("tracking CSR built;", int(IS_TRACKED.sum()), "tracked domains")


def expand_trackers(node_ids, weights=None):
    """(nodes -> their trackers) expansion via the CSR. Returns (idx_into_nodes, tracker).

    `idx_into_nodes` lets the caller map each produced tracker back to whatever
    seed the node came from, so one helper serves 1-hop and 2-hop alike.
    """
    cnt = n_trk_of[node_ids].astype(np.int64)
    total = int(cnt.sum())
    src = np.repeat(np.arange(len(node_ids), dtype=np.int64), cnt)
    starts = offs[node_ids]
    within = (np.arange(total, dtype=np.int64)
              - np.repeat(np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt))
    trk = tg_trk[np.repeat(starts, cnt) + within]
    w = None if weights is None else np.repeat(weights, cnt)
    return src, trk, w


def scatter(rows, trk, w=None):
    """Accumulate (row, tracker) -> (N, 355) float32."""
    flat = rows.astype(np.int64) * N_TRK + trk
    out = np.bincount(flat, weights=w, minlength=N * N_TRK)
    return out.reshape(N, N_TRK).astype(np.float32)


def save(arr, prefix, fname):
    cols = [f"{prefix}_{i}" for i in range(N_TRK)]
    for tag, sl in (("train", slice(0, N_TRAIN)), ("target", slice(N_TRAIN, N))):
        pl.DataFrame({"domain_id": all_ids[sl]}).hstack(
            pl.DataFrame(arr[sl], schema=cols)).write_parquet(
            OUT / f"{fname}_{tag}.parquet", compression="zstd")
    log("wrote", fname, arr.shape)


# --------------------------------------------- global link-graph degrees (46M)
linkdeg = np.zeros(MAXID, dtype=np.int32)
pf = pq.ParquetFile(IN / "link-graph.parquet")
for b in pf.iter_batches(batch_size=40_000_000,
                         columns=["source_domain_id", "target_domain_id"]):
    for c in (0, 1):
        v = b.column(c).to_numpy()
        v = v[(v >= 0) & (v < MAXID)]
        linkdeg += np.bincount(v, minlength=MAXID).astype(np.int32)
log("link degrees done; max:", int(linkdeg.max()), "mean:", float(linkdeg.mean()))

# ------------------------------------------------- seed edges (both directions)
lg = pl.scan_parquet(IN / "link-graph.parquet")
seed = pl.Series(all_ids).implode()
sides = {}
for direction, self_col, nbr_col in (("out", "source_domain_id", "target_domain_id"),
                                     ("in", "target_domain_id", "source_domain_id")):
    ed = lg.filter(pl.col(self_col).is_in(seed)).collect(engine="streaming")
    si, ni = ed[self_col].to_numpy(), ed[nbr_col].to_numpy()
    keep = (si != ni) & (ni >= 0) & (ni < MAXID)
    sides[direction] = (row_of[si[keep]], ni[keep])
    log(direction, "seed edges:", int(keep.sum()))

rows_all = np.concatenate([sides["out"][0], sides["in"][0]])
nbrs_all = np.concatenate([sides["out"][1], sides["in"][1]])

# ------------------------------------------------------------- 1. nbr_hub
w_edge = (1.0 / np.log2(2.0 + linkdeg[nbrs_all])).astype(np.float64)
src, trk, w = expand_trackers(nbrs_all, w_edge)
H = scatter(rows_all[src], trk, w)
save(H, "nh", "nbr_hub")
hub_mass = H.sum(axis=1)
del H, src, trk, w

# ------------------------------------------------------------- 2. nbr_rec
# reciprocal = the same (row, neighbour) pair present in BOTH directions
key_out = sides["out"][0].astype(np.int64) * MAXID + sides["out"][1]
key_in = sides["in"][0].astype(np.int64) * MAXID + sides["in"][1]
rec_key = np.intersect1d(key_out, key_in, assume_unique=False)
rec_rows = (rec_key // MAXID).astype(np.int32)
rec_nbrs = (rec_key % MAXID).astype(np.int64)
log("reciprocal pairs:", len(rec_key))
src, trk, _ = expand_trackers(rec_nbrs)
R = scatter(rec_rows[src], trk)
save(R, "nr", "nbr_rec")
n_rec = np.bincount(rec_rows, minlength=N).astype(np.int32)
del R, src, trk, key_out, key_in, rec_key

# ------------------------------------------------------------- 3. nbr_2h
mid_ok = (~IS_TRACKED[nbrs_all]) & (linkdeg[nbrs_all] <= MAX_MID_DEG) \
         & (linkdeg[nbrs_all] > 0)
pair_rows, pair_mid = rows_all[mid_ok], nbrs_all[mid_ok]
# dedup (row, mid): a mid reached by both directions should count once
pk = np.unique(pair_rows.astype(np.int64) * MAXID + pair_mid)
pair_rows = (pk // MAXID).astype(np.int32)
pair_mid = (pk % MAXID).astype(np.int64)
n_untracked_mid = np.bincount(pair_rows, minlength=N).astype(np.int32)
log("(row, low-degree untracked mid) pairs:", len(pk))

mids = np.unique(pair_mid)
mseed = pl.Series(mids.astype(np.int32)).implode()
m_edges = []
for self_col, nbr_col in (("source_domain_id", "target_domain_id"),
                          ("target_domain_id", "source_domain_id")):
    ed = lg.filter(pl.col(self_col).is_in(mseed)).collect(engine="streaming")
    a, b2 = ed[self_col].to_numpy(), ed[nbr_col].to_numpy()
    keep = (a != b2) & (b2 >= 0) & (b2 < MAXID) & IS_TRACKED[np.clip(b2, 0, MAXID - 1)]
    m_edges.append((a[keep], b2[keep]))
    log("mid edges to tracked:", int(keep.sum()))
m_self = np.concatenate([m_edges[0][0], m_edges[1][0]])
m_nbr = np.concatenate([m_edges[0][1], m_edges[1][1]])
del m_edges

# CSR over the mid set: mid -> its tracked neighbours
mid_pos = np.full(MAXID, -1, dtype=np.int32)
mid_pos[mids] = np.arange(len(mids), dtype=np.int32)
mp = mid_pos[m_self]
order = np.argsort(mp, kind="stable")
mp, m_nbr = mp[order], m_nbr[order]
mcnt = np.bincount(mp, minlength=len(mids)).astype(np.int64)
moffs = np.zeros(len(mids) + 1, dtype=np.int64)
np.cumsum(mcnt, out=moffs[1:])
log("2-hop CSR built over", len(mids), "mids")

# expand (row, mid) -> (row, mid's tracked neighbour) -> (row, tracker), chunked
H2 = np.zeros((N, N_TRK), dtype=np.float64)
two_mass = np.zeros(N, dtype=np.float64)
pp = mid_pos[pair_mid]
CH = 4_000_000
for i in range(0, len(pp), CH):
    p, r = pp[i:i + CH], pair_rows[i:i + CH]
    c = mcnt[p]
    if c.sum() == 0:
        continue
    rr = np.repeat(r, c)
    within = (np.arange(int(c.sum()), dtype=np.int64)
              - np.repeat(np.concatenate([[0], np.cumsum(c)[:-1]]), c))
    nb = m_nbr[np.repeat(moffs[p], c) + within]
    # CRITICAL (see data_exploration_8): `mid` was found AS a neighbour of the
    # seed, so that edge is still in mid's own edge list and the seed itself
    # comes back as one of mid's tracked neighbours -- writing the seed's own
    # label vector into its own feature. Drop those seed -> mid -> seed round
    # trips; without this, 100% of every train row's true trackers appear in
    # its own 2-hop block (and target rows, being untracked, never leak), which
    # inflated pipeline_07 to a bogus 0.91276.
    keep2 = nb != all_ids[rr]
    rr, nb = rr[keep2], nb[keep2]
    src, trk, _ = expand_trackers(nb)
    flat = rr[src].astype(np.int64) * N_TRK + trk
    H2 += np.bincount(flat, minlength=N * N_TRK).reshape(N, N_TRK)
    two_mass += np.bincount(rr[src], minlength=N)
    log(f"  2-hop chunk {i // CH + 1}/{(len(pp) + CH - 1) // CH}")
save(H2.astype(np.float32), "n2", "nbr_2h")
del H2

# ------------------------------------------------------------- 4. meta2
nbr_deg = linkdeg[nbrs_all].astype(np.float64)
cnt_nbr = np.bincount(rows_all, minlength=N).astype(np.float64)
sum_deg = np.bincount(rows_all, weights=nbr_deg, minlength=N)
max_deg = np.zeros(N, dtype=np.float64)
np.maximum.at(max_deg, rows_all, nbr_deg)
m2 = pl.DataFrame({
    "domain_id": all_ids,
    "n_rec_nbr": n_rec,
    "n_untracked_lowdeg_nbr": n_untracked_mid,
    "hub_weight_mass": hub_mass.astype(np.float32),
    "two_hop_mass": two_mass.astype(np.float32),
    "mean_nbr_linkdeg": (sum_deg / np.maximum(cnt_nbr, 1)).astype(np.float32),
    "max_nbr_linkdeg": max_deg.astype(np.float32),
    "own_linkdeg": linkdeg[all_ids].astype(np.float32),
})
for tag, sl in (("train", slice(0, N_TRAIN)), ("target", slice(N_TRAIN, N))):
    m2[sl].write_parquet(OUT / f"meta2_{tag}.parquet", compression="zstd")
log("wrote meta2", m2.shape)
print(m2.head(5))
log("DONE")
