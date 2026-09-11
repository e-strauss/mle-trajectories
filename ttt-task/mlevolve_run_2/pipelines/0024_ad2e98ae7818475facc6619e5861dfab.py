from collections import Counter, defaultdict
import gc
import json
import math
import os
import re
import sys
import numpy as np

# Force unbuffered / line-buffered stdout so logs are never suppressed
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Set fixed random seeds for strict reproducibility
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.makedirs("working", exist_ok=True)
os.makedirs("submission", exist_ok=True)

# =========================================================================
# 1. Load Metadata & Setup Problem Targets
# =========================================================================
trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
trackers_df["tracker_id"] = trackers_df["tracker_id"].astype(int)
trackers_df["tracking_domain_id"] = trackers_df["tracking_domain_id"].astype(np.int64)

target_df = pd.read_csv("input/target.tsv", sep="\t")
test_domains = target_df["domain_id"].values
num_trackers = len(trackers_df)

tracker_id_to_tdid = dict(
    zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
)
tdid_to_tracker_id = dict(
    zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
)
tracker_domain_ids_set = set(trackers_df["tracking_domain_id"].values)

with open("working/tracker_id_to_tdid.json", "w") as f:
    json.dump({int(k): int(v) for k, v in tracker_id_to_tdid.items()}, f)
with open("working/tdid_to_tracker_id.json", "w") as f:
    json.dump({int(k): int(v) for k, v in tdid_to_tracker_id.items()}, f)

# =========================================================================
# 2. Leak-Free Train / Validation Dataset Splitting
# =========================================================================
train_graph_table = pq.read_table(
    "input/tracking_graph_train.parquet",
    columns=["domain_id", "tracking_domain_id", "tracker_id"],
)
train_graph_df = train_graph_table.to_pandas()
del train_graph_table
gc.collect()

train_graph_df["domain_id"] = train_graph_df["domain_id"].astype(np.int64)
train_graph_df["tracker_id"] = train_graph_df["tracker_id"].astype(int)

# Filter candidate training domains strictly excluding test set (utilizing 100% of candidate domains)
all_candidate_domains = np.setdiff1d(train_graph_df["domain_id"].unique(), test_domains)
np.random.shuffle(all_candidate_domains)

n_val = min(30000, int(len(all_candidate_domains) * 0.2))

val_domains = all_candidate_domains[:n_val]
train_domains = all_candidate_domains[n_val:]

train_domains_set = set(train_domains)
val_domains_set = set(val_domains)
test_domains_set = set(test_domains)

all_selected_domains = np.concatenate([train_domains, val_domains, test_domains])
selected_domains_set = set(all_selected_domains)

# Build binary target matrices (N x 355)
train_domain_to_idx = {d: i for i, d in enumerate(train_domains)}
val_domain_to_idx = {d: i for i, d in enumerate(val_domains)}

train_targets = np.zeros((len(train_domains), num_trackers), dtype=np.uint8)
val_targets = np.zeros((len(val_domains), num_trackers), dtype=np.uint8)

train_sub = train_graph_df[train_graph_df["domain_id"].isin(train_domains_set)]
train_r = [train_domain_to_idx[d] for d in train_sub["domain_id"].values]
train_c = train_sub["tracker_id"].values
train_targets[train_r, train_c] = 1

val_sub = train_graph_df[train_graph_df["domain_id"].isin(val_domains_set)]
val_r = [val_domain_to_idx[d] for d in val_sub["domain_id"].values]
val_c = val_sub["tracker_id"].values
val_targets[val_r, val_c] = 1

np.savez_compressed("working/train_targets.npz", targets=train_targets)
np.savez_compressed("working/val_targets.npz", targets=val_targets)

# Compute global empirical tracker priors strictly from training set
global_tracker_priors = train_targets.mean(axis=0).astype(np.float32)
np.save("working/global_tracker_priors.npy", global_tracker_priors)

del train_graph_df, train_sub, val_sub
gc.collect()

# =========================================================================
# 3. Domain Lexical & Press Freedom Features (Vectorized & Fast)
# =========================================================================
domains_table = pq.read_table("input/domains.parquet", columns=["domain", "domain_id"])
domains_df = domains_table.to_pandas()
del domains_table
gc.collect()

domains_df = domains_df[domains_df["domain_id"].isin(selected_domains_set)].copy()
domain_id_to_name = dict(zip(domains_df["domain_id"], domains_df["domain"]))
name_to_domain_id = dict(zip(domains_df["domain"], domains_df["domain_id"]))
del domains_df
gc.collect()

press_freedom_map = {}
if os.path.exists("input/freedom-of-the-press.csv"):
    try:
        press_df = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
        if len(press_df.columns) < 3:
            press_df = pd.read_csv(
                "input/freedom-of-the-press.csv", sep=None, engine="python"
            )
        for _, row in press_df.iterrows():
            tld_val = str(row.iloc[0]).strip().lower().lstrip(".")
            score_val = float(row.iloc[2])
            press_freedom_map[tld_val] = score_val
    except Exception:
        pass

median_press_score = (
    float(np.median(list(press_freedom_map.values()))) if press_freedom_map else 50.0
)

top_tlds = [
    "com", "ru", "org", "net", "de", "uk", "jp", "fr", "it", "pl",
    "br", "cn", "in", "nl", "es", "cz", "eu", "ua", "ca", "au",
    "ch", "se", "ro", "gr", "at", "tv", "io", "me", "co", "info",
]
top_tlds_set = set(top_tlds)
tld_to_col_idx = {t: i for i, t in enumerate(top_tlds)}

keywords_ecommerce = ["shop", "store", "cart", "buy", "market", "mall", "deal", "pay"]
keywords_media = ["news", "press", "media", "times", "post", "daily", "gazette", "journal"]
keywords_video = ["video", "tv", "movie", "film", "tube", "stream"]
keywords_adult = ["adult", "sex", "porn", "xxx"]
keywords_tech = ["tech", "dev", "code", "cloud", "soft", "app", "web"]
keywords_finance = ["bank", "finance", "invest", "loan", "crypto", "coin"]

two_level_tlds = {
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "ed.jp", "go.jp", "gr.jp", "lg.jp",
    "com.br", "net.br", "org.br", "gov.br", "edu.br",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.ru", "net.ru", "org.ru", "pp.ru",
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in",
    "com.mx", "net.mx", "org.mx", "edu.mx", "gob.mx",
    "com.tr", "net.tr", "org.tr", "edu.tr", "gov.tr",
    "com.pl", "net.pl", "org.pl", "info.pl",
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
    "co.za", "net.za", "org.za", "web.za",
    "co.kr", "ne.kr", "or.kr", "re.kr",
    "com.ar", "net.ar", "org.ar",
    "com.tw", "org.tw", "idv.tw",
    "com.ua", "net.ua", "org.ua", "kiev.ua",
    "co.il", "org.il", "net.il",
    "com.sg", "org.sg", "net.sg",
    "com.hk", "org.hk", "net.hk",
}


def extract_root_and_tld(domain_name):
    if not domain_name or not isinstance(domain_name, str):
        return "unknown", "unknown"
    parts = domain_name.lower().strip().split(".")
    if len(parts) == 1:
        return parts[0], parts[0]
    tld = parts[-1]
    if len(parts) >= 3:
        last_two = f"{parts[-2]}.{parts[-1]}"
        if last_two in two_level_tlds:
            root = f"{parts[-3]}.{last_two}"
            return root, last_two
    root = f"{parts[-2]}.{parts[-1]}"
    return root, tld


num_selected = len(all_selected_domains)
all_names = [domain_id_to_name.get(did, "") for did in all_selected_domains]
all_roots = []
all_tlds = []

for name in all_names:
    root, tld = extract_root_and_tld(name)
    all_roots.append(root)
    all_tlds.append(tld)

domain_tlds = dict(zip(all_selected_domains, all_tlds))
domain_roots = dict(zip(all_selected_domains, all_roots))

# Vectorized creation of lexical features
d_lens = np.array([max(len(n), 1) for n in all_names], dtype=np.float32)
num_dots = np.array([n.count(".") for n in all_names], dtype=np.float32)
num_hyphens = np.array([n.count("-") for n in all_names], dtype=np.float32)
num_digits = np.array([sum(c.isdigit() for c in n) for n in all_names], dtype=np.float32)
has_www = np.array([1.0 if n.lower().startswith("www.") else 0.0 for n in all_names], dtype=np.float32)
has_multi_sub = np.array([1.0 if d > 1 else 0.0 for d in num_dots], dtype=np.float32)

press_scores = np.array([press_freedom_map.get(tld, median_press_score) for tld in all_tlds], dtype=np.float32)
has_press = np.array([1.0 if tld in press_freedom_map else 0.0 for tld in all_tlds], dtype=np.float32)
is_auth = np.array([1.0 if s > 60.0 else 0.0 for s in press_scores], dtype=np.float32)

tld_onehot = np.zeros((num_selected, len(top_tlds) + 1), dtype=np.float32)
for i, tld in enumerate(all_tlds):
    col = tld_to_col_idx.get(tld, len(top_tlds))
    tld_onehot[i, col] = 1.0

kw_ecom = np.array([sum(1.0 for kw in keywords_ecommerce if kw in n.lower()) for n in all_names], dtype=np.float32)
kw_med = np.array([sum(1.0 for kw in keywords_media if kw in n.lower()) for n in all_names], dtype=np.float32)
kw_vid = np.array([sum(1.0 for kw in keywords_video if kw in n.lower()) for n in all_names], dtype=np.float32)
kw_adlt = np.array([sum(1.0 for kw in keywords_adult if kw in n.lower()) for n in all_names], dtype=np.float32)
kw_tch = np.array([sum(1.0 for kw in keywords_tech if kw in n.lower()) for n in all_names], dtype=np.float32)
kw_fin = np.array([sum(1.0 for kw in keywords_finance if kw in n.lower()) for n in all_names], dtype=np.float32)
kw_total = kw_ecom + kw_med + kw_vid + kw_adlt + kw_tch + kw_fin

X_lexical = np.column_stack([
    d_lens,
    num_dots,
    num_hyphens,
    num_digits,
    num_digits / d_lens,
    has_www,
    has_multi_sub,
    press_scores,
    has_press,
    is_auth,
    tld_onehot,
    kw_ecom,
    kw_med,
    kw_vid,
    kw_adlt,
    kw_tch,
    kw_fin,
    kw_total,
]).astype(np.float32)
del all_names, d_lens, num_dots, num_hyphens, num_digits, has_www, has_multi_sub, press_scores, has_press, is_auth, tld_onehot
gc.collect()

# =========================================================================
# 4. URL Content Classification Features
# =========================================================================
url_cat_categories = [
    "Arts",
    "Business",
    "Computers",
    "Games",
    "Health",
    "Home",
    "Kids_and_Teens",
    "News",
    "Recreation",
    "Reference",
    "Regional",
    "Science",
    "Shopping",
    "Society",
    "Sports",
]
cat_to_idx = {c: i for i, c in enumerate(url_cat_categories)}
domain_cat_counts = defaultdict(
    lambda: np.zeros(len(url_cat_categories), dtype=np.float32)
)

if os.path.exists("input/url-classification.csv"):
    url_chunk_iter = pd.read_csv(
        "input/url-classification.csv", chunksize=200000, usecols=["url", "category"]
    )
    for chunk in url_chunk_iter:
        for u, cat in zip(chunk["url"].values, chunk["category"].values):
            if not isinstance(u, str) or not isinstance(cat, str):
                continue
            s = u.split("://", 1)[1] if "://" in u else u
            h = s.split("/", 1)[0].split(":", 1)[0].strip().lower()
            target_id = name_to_domain_id.get(h, None)
            if target_id is None and h.startswith("www."):
                target_id = name_to_domain_id.get(h[4:], None)
            if target_id is not None and cat in cat_to_idx:
                domain_cat_counts[target_id][cat_to_idx[cat]] += 1.0

X_cat = np.zeros((num_selected, len(url_cat_categories) + 1), dtype=np.float32)
domain_to_sel_idx_map = {did: idx for idx, did in enumerate(all_selected_domains)}
for did, counts in domain_cat_counts.items():
    s_idx = domain_to_sel_idx_map.get(did, None)
    if s_idx is not None:
        c_sum = counts.sum()
        if c_sum > 0:
            X_cat[s_idx, : len(url_cat_categories)] = counts / c_sum
            X_cat[s_idx, len(url_cat_categories)] = 1.0

del domain_cat_counts, name_to_domain_id, domain_to_sel_idx_map
gc.collect()

# =========================================================================
# 5. Link Graph Topological & Collaborative Neighbor Features
# =========================================================================
link_table = pq.read_table(
    "input/link-graph.parquet",
    columns=["source_domain_id", "target_domain_id"],
)
src_arr = link_table["source_domain_id"].to_numpy()
dst_arr = link_table["target_domain_id"].to_numpy()
del link_table
gc.collect()

# Ensure bounds safety across all selected domains
max_id = int(max(src_arr.max(), dst_arr.max(), all_selected_domains.max())) + 1
num_selected = len(all_selected_domains)
num_train_domains = len(train_domains)

out_degrees = np.bincount(src_arr, minlength=max_id)
in_degrees = np.bincount(dst_arr, minlength=max_id)
tot_degrees = out_degrees + in_degrees

domain_to_sel_idx = np.full(max_id, -1, dtype=np.int32)
domain_to_sel_idx[all_selected_domains] = np.arange(num_selected, dtype=np.int32)

tdid_to_tr_id_arr = np.full(max_id, -1, dtype=np.int16)
for tr_id, tdid in zip(
    trackers_df["tracker_id"].values, trackers_df["tracking_domain_id"].values
):
    if tdid < max_id:
        tdid_to_tr_id_arr[tdid] = tr_id

domain_to_train_idx = np.full(max_id, -1, dtype=np.int32)
domain_to_train_idx[train_domains] = np.arange(num_train_domains, dtype=np.int32)

# Direct Tracker Link Indicators (all 355 trackers)
src_sel_idx = domain_to_sel_idx[src_arr]
dst_tr_id = tdid_to_tr_id_arr[dst_arr]
is_tracker_edge = dst_tr_id >= 0
tracker_out_counts = np.bincount(src_arr[is_tracker_edge], minlength=max_id)

direct_link_mask = (src_sel_idx >= 0) & is_tracker_edge
valid_src_sel = src_sel_idx[direct_link_mask]
valid_dst_tr = dst_tr_id[direct_link_mask]

direct_links = np.zeros((num_selected, num_trackers), dtype=np.float32)
direct_links[valid_src_sel, valid_dst_tr] = 1.0

# Fast Sparse-Sparse Neighbor Tracker Adoption with Adamic-Adar Weighting
train_targets_csr = csr_matrix(train_targets, dtype=np.float32)

# Outgoing neighbors: src in selected_domains, dst in train_domains
dst_train_idx = domain_to_train_idx[dst_arr]
out_nbr_mask = (src_sel_idx >= 0) & (dst_train_idx >= 0) & (src_arr != dst_arr)
u_out = src_sel_idx[out_nbr_mask]
v_out = dst_train_idx[out_nbr_mask]
w_out = (1.0 / np.log1p(np.maximum(tot_degrees[dst_arr[out_nbr_mask]], 1))).astype(np.float32)

A_out = csr_matrix(
    (w_out, (u_out, v_out)),
    shape=(num_selected, num_train_domains),
)
nbr_train_out_deg = np.array(A_out.sum(axis=1)).ravel()
# Ultra-fast sparse-sparse matrix multiplication
out_nbr_tracker_counts = (A_out @ train_targets_csr).toarray()
out_nbr_tracker_norm = np.where(
    nbr_train_out_deg[:, None] > 0,
    out_nbr_tracker_counts / np.maximum(nbr_train_out_deg[:, None], 1e-7),
    0.0,
).astype(np.float32)
del A_out, w_out, u_out, v_out, out_nbr_mask, out_nbr_tracker_counts
gc.collect()

# Incoming neighbors: src in train_domains, dst in selected_domains
src_train_idx = domain_to_train_idx[src_arr]
dst_sel_idx = domain_to_sel_idx[dst_arr]
in_nbr_mask = (src_train_idx >= 0) & (dst_sel_idx >= 0) & (src_arr != dst_arr)
v_in = src_train_idx[in_nbr_mask]
u_in = dst_sel_idx[in_nbr_mask]
w_in = (1.0 / np.log1p(np.maximum(tot_degrees[src_arr[in_nbr_mask]], 1))).astype(np.float32)

A_in = csr_matrix(
    (w_in, (u_in, v_in)),
    shape=(num_selected, num_train_domains),
)
nbr_train_in_deg = np.array(A_in.sum(axis=1)).ravel()
# Ultra-fast sparse-sparse matrix multiplication
in_nbr_tracker_counts = (A_in @ train_targets_csr).toarray()
in_nbr_tracker_norm = np.where(
    nbr_train_in_deg[:, None] > 0,
    in_nbr_tracker_counts / np.maximum(nbr_train_in_deg[:, None], 1e-7),
    0.0,
).astype(np.float32)
del A_in, w_in, u_in, v_in, in_nbr_mask, in_nbr_tracker_counts, train_targets_csr
gc.collect()

nbr_tot_deg = nbr_train_out_deg + nbr_train_in_deg

# Vectorized graph scalar features
sel_out_deg = out_degrees[all_selected_domains].astype(np.float32)
sel_in_deg = in_degrees[all_selected_domains].astype(np.float32)
sel_tr_links = tracker_out_counts[all_selected_domains].astype(np.float32)

X_graph = np.column_stack([
    sel_out_deg,
    np.log1p(sel_out_deg),
    sel_in_deg,
    np.log1p(sel_in_deg),
    sel_out_deg + sel_in_deg,
    (sel_in_deg + 1.0) / (sel_out_deg + 1.0),
    (sel_out_deg + sel_in_deg == 0).astype(np.float32),
    sel_tr_links,
    np.log1p(sel_tr_links),
    sel_tr_links / np.maximum(sel_out_deg, 1.0),
    nbr_train_out_deg.astype(np.float32),
    np.log1p(nbr_train_out_deg).astype(np.float32),
    nbr_train_in_deg.astype(np.float32),
    np.log1p(nbr_train_in_deg).astype(np.float32),
    nbr_tot_deg.astype(np.float32),
    np.log1p(nbr_tot_deg).astype(np.float32),
]).astype(np.float32)

del (
    src_arr,
    dst_arr,
    src_sel_idx,
    dst_tr_id,
    is_tracker_edge,
    direct_link_mask,
    valid_src_sel,
    valid_dst_tr,
    dst_train_idx,
    src_train_idx,
    dst_sel_idx,
    domain_to_sel_idx,
    domain_to_train_idx,
    tdid_to_tr_id_arr,
    out_degrees,
    in_degrees,
    tot_degrees,
    tracker_out_counts,
    nbr_train_out_deg,
    nbr_train_in_deg,
    nbr_tot_deg,
    sel_out_deg,
    sel_in_deg,
    sel_tr_links,
)
gc.collect()

# =========================================================================
# 6. Bayesian Leave-One-Out Root-Domain & TLD Tracker Priors (Zero Leakage)
# =========================================================================
tld_train_counts = defaultdict(int)
tld_tracker_counts = defaultdict(lambda: np.zeros(num_trackers, dtype=np.float32))

root_train_counts = defaultdict(int)
root_tracker_counts = defaultdict(lambda: np.zeros(num_trackers, dtype=np.float32))

for did in train_domains:
    tld = domain_tlds.get(did, "unknown")
    root = domain_roots.get(did, "unknown")
    idx = train_domain_to_idx[did]
    targ = train_targets[idx]

    tld_train_counts[tld] += 1
    tld_tracker_counts[tld] += targ

    root_train_counts[root] += 1
    root_tracker_counts[root] += targ

alpha_tld = 10.0
tld_priors = {}
for tld, counts in tld_tracker_counts.items():
    total = tld_train_counts[tld]
    smoothed = (counts + alpha_tld * global_tracker_priors) / (total + alpha_tld)
    tld_priors[tld] = smoothed

beta_root = 2.0
root_prior_matrix = np.zeros((num_selected, num_trackers), dtype=np.float32)
tld_prior_matrix = np.zeros((num_selected, num_trackers), dtype=np.float32)
has_root_match = np.zeros(num_selected, dtype=np.float32)
root_cnt_arr = np.zeros(num_selected, dtype=np.float32)

# Vectorized TLD prior lookup
unique_tlds = list(tld_priors.keys())
tld_to_int = {t: idx for idx, t in enumerate(unique_tlds)}
tld_table = np.zeros((len(unique_tlds) + 1, num_trackers), dtype=np.float32)
for t, idx in tld_to_int.items():
    tld_table[idx] = tld_priors[t]
tld_table[-1] = global_tracker_priors

default_tld_int = len(unique_tlds)
sel_tld_indices = np.array(
    [tld_to_int.get(domain_tlds.get(did, "unknown"), default_tld_int) for did in all_selected_domains],
    dtype=np.int32,
)
tld_prior_matrix = tld_table[sel_tld_indices]
root_prior_matrix = tld_prior_matrix.copy()

# Fast targeted update for root domains
for i in range(num_train_domains):
    did = all_selected_domains[i]
    root = domain_roots.get(did, "unknown")
    tot = root_train_counts.get(root, 0)
    if tot > 1:
        tot_loo = tot - 1
        cnt_loo = root_tracker_counts[root] - train_targets[i]
        tld_base = tld_prior_matrix[i]
        root_prior_matrix[i] = (cnt_loo + beta_root * tld_base) / (tot_loo + beta_root)
        has_root_match[i] = 1.0
        root_cnt_arr[i] = float(tot_loo)

for i in range(num_train_domains, num_selected):
    did = all_selected_domains[i]
    root = domain_roots.get(did, "unknown")
    if root in root_train_counts:
        tot = root_train_counts[root]
        cnt = root_tracker_counts[root]
        tld_base = tld_prior_matrix[i]
        root_prior_matrix[i] = (cnt + beta_root * tld_base) / (tot + beta_root)
        has_root_match[i] = 1.0
        root_cnt_arr[i] = float(tot)
del tld_table, sel_tld_indices, tld_to_int, unique_tlds
gc.collect()

X_prior = np.column_stack([
    has_root_match,
    root_cnt_arr,
    np.log1p(root_cnt_arr),
    -np.sum(root_prior_matrix * np.log(root_prior_matrix + 1e-12), axis=1).astype(np.float32),
    root_prior_matrix.max(axis=1).astype(np.float32),
]).astype(np.float32)

del root_train_counts, root_tracker_counts, tld_train_counts, tld_tracker_counts, tld_priors, domain_roots, domain_tlds
gc.collect()

# =========================================================================
# 7. Merge Feature Sets & Materialize Tensors
# =========================================================================
X_scalar = np.hstack([X_lexical, X_cat, X_graph, X_prior]).astype(np.float32)
del X_lexical, X_cat, X_graph, X_prior
gc.collect()
np.nan_to_num(X_scalar, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# Concatenate scalar features with the 5 high-signal 355-dimensional tracker-aligned blocks:
# 1) Direct tracker link indicators (N, 355)
# 2) Outgoing neighbor tracker adoption (N, 355)
# 3) Incoming neighbor tracker adoption (N, 355)
# 4) Bayesian root-domain tracker prior probabilities (N, 355)
# 5) Bayesian TLD tracker prior probabilities (N, 355)
X_all = np.hstack(
    [
        X_scalar,
        direct_links,
        out_nbr_tracker_norm,
        in_nbr_tracker_norm,
        root_prior_matrix,
        tld_prior_matrix,
    ]
).astype(np.float32)

del (
    X_scalar,
    direct_links,
    out_nbr_tracker_norm,
    in_nbr_tracker_norm,
    root_prior_matrix,
    tld_prior_matrix,
)
gc.collect()

# Compute empirical conditional co-occurrence matrix C for relational refinement
Y_tr_f = train_targets.astype(np.float32)
cooc_counts = Y_tr_f.T.dot(Y_tr_f)
tracker_freqs = np.diag(cooc_counts).copy()

epsilon_cooc = 10.0
cond_cooc = cooc_counts / (tracker_freqs[:, None] + epsilon_cooc)
np.fill_diagonal(cond_cooc, 0.0)

row_max = cond_cooc.max(axis=1, keepdims=True)
cond_cooc_norm = np.where(row_max > 0, cond_cooc / np.maximum(row_max, 1.0), 0.0).astype(np.float32)
del Y_tr_f, cooc_counts, tracker_freqs, cond_cooc, row_max
gc.collect()

# Build taxonomically initialized tracker prototypes from corporate metadata
meta_rows = []
for tid in range(num_trackers):
    sub = trackers_df[trackers_df["tracker_id"] == tid]
    if len(sub) > 0:
        r = sub.iloc[0]
        meta_rows.append({
            "tracker_id": tid,
            "company": str(r["company"]).strip().lower() if pd.notna(r["company"]) else "unknown",
            "category": str(r["category"]).strip().lower() if pd.notna(r["category"]) else "unknown",
            "country": str(r["country"]).strip().lower() if pd.notna(r["country"]) else "unknown",
            "brand": str(r["brand"]).strip().lower() if pd.notna(r["brand"]) else "unknown",
        })
    else:
        meta_rows.append({
            "tracker_id": tid,
            "company": "unknown",
            "category": "unknown",
            "country": "unknown",
            "brand": "unknown",
        })
meta_df = pd.DataFrame(meta_rows)
comp_dummies = pd.get_dummies(meta_df["company"], prefix="comp", dtype=np.float32)
cat_dummies = pd.get_dummies(meta_df["category"], prefix="cat", dtype=np.float32)
cntry_dummies = pd.get_dummies(meta_df["country"], prefix="cntry", dtype=np.float32)
brand_dummies = pd.get_dummies(meta_df["brand"], prefix="brand", dtype=np.float32)

taxonomic_matrix = np.hstack([
    comp_dummies.values * 1.5,
    brand_dummies.values * 1.5,
    cat_dummies.values * 1.0,
    cntry_dummies.values * 0.8,
]).astype(np.float32)

target_proto_dim = 128
svd_components = min(target_proto_dim, taxonomic_matrix.shape[1] - 1)
svd_proto = TruncatedSVD(n_components=svd_components, random_state=42)
P_tax = svd_proto.fit_transform(taxonomic_matrix).astype(np.float32)
if P_tax.shape[1] < target_proto_dim:
    pad_dim = target_proto_dim - P_tax.shape[1]
    pad_vecs = np.zeros((num_trackers, pad_dim), dtype=np.float32)
    P_tax = np.hstack([P_tax, pad_vecs])

tax_norms = np.linalg.norm(P_tax, axis=1, keepdims=True)
taxonomic_prototypes = np.where(
    tax_norms > 1e-7, P_tax / np.maximum(tax_norms, 1e-7), 0.0
).astype(np.float32)
del meta_rows, meta_df, comp_dummies, cat_dummies, cntry_dummies, brand_dummies, taxonomic_matrix, P_tax, tax_norms
gc.collect()

n_train_len = len(train_domains)
n_val_len = len(val_domains)

X_train = np.ascontiguousarray(X_all[:n_train_len], dtype=np.float32)
X_val = np.ascontiguousarray(
    X_all[n_train_len : n_train_len + n_val_len], dtype=np.float32
)
X_test = np.ascontiguousarray(X_all[n_train_len + n_val_len :], dtype=np.float32)
del X_all
gc.collect()

Y_train = train_targets.astype(np.float32)
Y_val = val_targets.astype(np.float32)

np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_val, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

in_features = X_train.shape[1]

# =========================================================================
# 8. Model Architecture & Loss Function
# =========================================================================
prior_eps = 1e-5
clipped_priors = np.clip(global_tracker_priors, prior_eps, 1.0 - prior_eps)
init_biases = np.log(clipped_priors / (1.0 - clipped_priors)).astype(np.float32)


class ResidualTabularBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.act2 = nn.SiLU()
        self.shortcut = (
            nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act2(out + res)


class TrackerRankNet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 355,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout_rate: float = 0.2,
        initial_bias: np.ndarray = None,
        num_prior_channels: int = 5,
        cooccurrence_matrix: np.ndarray = None,
        tracker_prototypes: np.ndarray = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.num_prior_channels = num_prior_channels
        self.has_tracker_priors = in_dim >= num_prior_channels * num_classes

        self.input_norm = nn.BatchNorm1d(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_block1 = ResidualTabularBlock(
            hidden_dim, hidden_dim, dropout_rate=dropout_rate
        )
        self.res_block2 = ResidualTabularBlock(
            hidden_dim, hidden_dim // 2, dropout_rate=dropout_rate
        )

        bottleneck_dim = hidden_dim // 2
        self.direct_head = nn.Linear(bottleneck_dim, num_classes)
        self.domain_latent_proj = nn.Linear(bottleneck_dim, latent_dim)

        if tracker_prototypes is not None:
            self.tracker_prototypes = nn.Parameter(
                torch.tensor(tracker_prototypes, dtype=torch.float32)
            )
        else:
            self.tracker_prototypes = nn.Parameter(
                torch.randn(num_classes, latent_dim) / math.sqrt(latent_dim)
            )
        self.head_blend = nn.Parameter(torch.tensor([0.5]))

        if self.has_tracker_priors:
            init_weights = torch.tensor(
                [1.5, 1.0, 1.0, 1.5, 0.8], dtype=torch.float32
            ).unsqueeze(0).repeat(num_classes, 1)
            self.tracker_prior_weights = nn.Parameter(init_weights)
            self.tracker_prior_bias = nn.Parameter(torch.zeros(num_classes))
            self.tracker_res_net = nn.Sequential(
                nn.Linear(num_prior_channels, 32),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
            )
            with torch.no_grad():
                nn.init.normal_(self.tracker_res_net[0].weight, std=0.01)
                nn.init.zeros_(self.tracker_res_net[0].bias)
                nn.init.zeros_(self.tracker_res_net[3].weight)
                nn.init.zeros_(self.tracker_res_net[3].bias)
            self.res_scale = nn.Parameter(torch.tensor([0.5]))

        # Contextual Relational Tracker Refinement Module
        self.cooc_layer = nn.Linear(num_classes, num_classes, bias=False)
        self.cooc_gate = nn.Linear(latent_dim, num_classes)
        if cooccurrence_matrix is not None:
            with torch.no_grad():
                self.cooc_layer.weight.copy_(
                    torch.from_numpy(cooccurrence_matrix.T).float()
                )
            self.cooc_layer.weight.requires_grad = False
        else:
            with torch.no_grad():
                nn.init.normal_(self.cooc_layer.weight, std=0.01)
        with torch.no_grad():
            nn.init.normal_(self.cooc_gate.weight, std=0.01)
            nn.init.constant_(self.cooc_gate.bias, -2.0)

        if initial_bias is not None:
            with torch.no_grad():
                self.direct_head.bias.copy_(torch.tensor(initial_bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_norm(x)
        h = self.input_proj(h)
        h = self.res_block1(h)
        h = self.res_block2(h)

        direct_logits = self.direct_head(h)
        domain_latent = F.normalize(self.domain_latent_proj(h), p=2, dim=-1)
        tracker_latent = F.normalize(self.tracker_prototypes, p=2, dim=-1)
        proto_logits = torch.matmul(domain_latent, tracker_latent.t()) * (
            math.sqrt(self.latent_dim) / 2.0
        )

        alpha = torch.sigmoid(self.head_blend)
        base_logits = alpha * direct_logits + (1.0 - alpha) * proto_logits

        if self.has_tracker_priors:
            prior_signals = x[
                :, -self.num_prior_channels * self.num_classes :
            ].reshape(-1, self.num_prior_channels, self.num_classes)
            prior_perm = prior_signals.permute(0, 2, 1)
            direct_prior_logits = (
                prior_perm * self.tracker_prior_weights
            ).sum(dim=-1) + self.tracker_prior_bias
            mlp_prior_logits = self.tracker_res_net(prior_perm).squeeze(-1)
            tracker_res_logits = direct_prior_logits + mlp_prior_logits
            base_logits = base_logits + self.res_scale * tracker_res_logits

        probs = torch.sigmoid(base_logits)
        cooc_msg = self.cooc_layer(probs)
        proto_sim = torch.matmul(domain_latent, tracker_latent.t())
        cooc_gate = torch.sigmoid(self.cooc_gate(domain_latent) + 0.5 * proto_sim)
        relational_refinement = cooc_gate * cooc_msg
        final_logits = base_logits + relational_refinement
        return final_logits


class TaskAlignedSoftRankLoss(nn.Module):

    def __init__(
        self,
        tau: float = 0.5,
        cutoff_rank: float = 9.0,
        rank_weight: float = 0.25,
    ):
        super().__init__()
        self.tau = tau
        self.cutoff_rank = cutoff_rank
        self.rank_weight = rank_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).sum(dim=-1).mean()

        diff = (logits.unsqueeze(1) - logits.unsqueeze(2)) / self.tau
        soft_ranks = torch.sigmoid(diff).sum(dim=-1) - 0.5

        rank_violations = F.relu(soft_ranks - self.cutoff_rank)
        pos_counts = targets.sum(dim=-1).clamp(min=1.0)
        rank_loss = ((rank_violations * targets).sum(dim=-1) / pos_counts).mean()

        return bce_loss + self.rank_weight * rank_loss


def compute_recall_at_k(
    preds_logits: np.ndarray, targets: np.ndarray, k: int = 10
) -> float:
    n_samples = preds_logits.shape[0]
    topk_indices = np.argpartition(preds_logits, -k, axis=1)[:, -k:]
    row_indices = np.arange(n_samples)[:, None]
    hits = targets[row_indices, topk_indices].sum(axis=1)
    true_counts = targets.sum(axis=1)
    recalls = np.where(true_counts > 0, hits / true_counts, 0.0)
    return float(np.mean(recalls))


# =========================================================================
# 9. Training, Validation & Checkpoint Selection (High-Speed In-Memory GPU)
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrackerRankNet(
    in_dim=in_features,
    num_classes=num_trackers,
    hidden_dim=512,
    latent_dim=128,
    dropout_rate=0.2,
    initial_bias=init_biases,
    num_prior_channels=5,
    cooccurrence_matrix=cond_cooc_norm,
    tracker_prototypes=taxonomic_prototypes,
).to(device)

criterion = TaskAlignedSoftRankLoss(
    tau=0.5,
    cutoff_rank=9.0,
    rank_weight=0.25,
)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name or "norm" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = AdamW(
    [
        {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
)

num_epochs = 10
base_lr = 3e-4
min_lr = 1e-5
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - 1, eta_min=min_lr)

# Keep datasets in host memory (RAM) and stream contiguous slices to GPU
X_train_t = torch.from_numpy(X_train)
Y_train_t = torch.from_numpy(Y_train)
X_val_t = torch.from_numpy(X_val)
X_test_t = torch.from_numpy(X_test)

del X_train, X_val, X_test
gc.collect()

batch_size = 2048
n_train_samples = X_train_t.shape[0]
num_batches_per_epoch = math.ceil(n_train_samples / batch_size)
best_val_recall = -1.0
best_model_path = "working/best_tracker_ranknet.pt"

print(f"Beginning training on {n_train_samples} samples across {num_epochs} epochs...", flush=True)

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    # Shuffle once per epoch so batch slicing is contiguous and ultra-fast
    perm = torch.randperm(n_train_samples)
    X_train_epoch = X_train_t[perm]
    Y_train_epoch = Y_train_t[perm]

    for start_idx in range(0, n_train_samples, batch_size):
        if epoch == 1:
            warmup_step = num_batches + 1
            warmup_lr = min_lr + (base_lr - min_lr) * (warmup_step / num_batches_per_epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        end_idx = min(start_idx + batch_size, n_train_samples)
        batch_x = X_train_epoch[start_idx:end_idx].to(device, non_blocking=True)
        batch_y = Y_train_epoch[start_idx:end_idx].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    del X_train_epoch, Y_train_epoch, perm
    gc.collect()

    if epoch > 1:
        scheduler.step()
    avg_train_loss = total_train_loss / max(num_batches, 1)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for start_idx in range(0, len(X_val_t), 4096):
            end_idx = min(start_idx + 4096, len(X_val_t))
            batch_x = X_val_t[start_idx:end_idx].to(device)
            batch_logits = model(batch_x)
            val_preds_list.append(batch_logits.cpu().numpy())

    val_preds = np.vstack(val_preds_list)
    val_recall_10 = compute_recall_at_k(val_preds, Y_val, k=10)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
        f"Val Recall@10: {val_recall_10:.5f} | LR: {current_lr:.6f}",
        flush=True,
    )

    if val_recall_10 > best_val_recall:
        best_val_recall = val_recall_10
        torch.save(model.state_dict(), best_model_path)

# =========================================================================
# 10. Load Best Model, Compute Final Score & Generate Submission
# =========================================================================
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

final_val_preds_list = []
with torch.no_grad():
    for start_idx in range(0, len(X_val_t), 2048):
        end_idx = min(start_idx + 2048, len(X_val_t))
        batch_x = X_val_t[start_idx:end_idx].to(device)
        batch_logits = model(batch_x)
        final_val_preds_list.append(batch_logits.cpu().numpy())

final_val_preds = np.vstack(final_val_preds_list)
final_val_score = compute_recall_at_k(final_val_preds, Y_val, k=10)

test_preds_list = []
with torch.no_grad():
    for start_idx in range(0, len(X_test_t), 2048):
        end_idx = min(start_idx + 2048, len(X_test_t))
        batch_x = X_test_t[start_idx:end_idx].to(device)
        batch_logits = model(batch_x)
        test_preds_list.append(batch_logits.cpu().numpy())

test_preds = np.vstack(test_preds_list)
top10_test_tracker_ids = np.argsort(test_preds, axis=1)[:, -10:][:, ::-1]

target_domain_ids = target_df["domain_id"].values
flat_tracker_ids = top10_test_tracker_ids.flatten()
flat_tracking_domain_ids = [tracker_id_to_tdid[tid] for tid in flat_tracker_ids]
flat_domain_ids = np.repeat(target_domain_ids, 10)

sub_df = pd.DataFrame(
    {"domain_id": flat_domain_ids, "tracking_domain_id": flat_tracking_domain_ids}
)

submission_csv_path = "submission/submission.csv"
submission_tsv_path = "submission/submission.tsv"

sub_df.to_csv(submission_csv_path, sep="\t", index=False)
sub_df.to_csv(submission_tsv_path, sep="\t", index=False)

assert os.path.exists(submission_csv_path), "Submission file was not created!"
assert len(sub_df) == len(target_domain_ids) * 10, "Submission row count mismatch!"
assert sub_df["tracking_domain_id"].isnull().sum() == 0, "Null tracking IDs found!"

print(f"Final Validation Score: {final_val_score:.5f}", flush=True)
