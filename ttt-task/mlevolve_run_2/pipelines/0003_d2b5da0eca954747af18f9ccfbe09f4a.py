from collections import Counter, defaultdict
import gc
import json
import math
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Set fixed random seeds for strict reproducibility
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

# Filter candidate training domains strictly excluding test set
all_candidate_domains = np.setdiff1d(train_graph_df["domain_id"].unique(), test_domains)
np.random.shuffle(all_candidate_domains)

n_val = min(30000, int(len(all_candidate_domains) * 0.2))
n_train = min(170000, len(all_candidate_domains) - n_val)

val_domains = all_candidate_domains[:n_val]
train_domains = all_candidate_domains[n_val : n_val + n_train]

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
# 3. Domain Lexical & Press Freedom Features
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

# Press freedom scores
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
    "com",
    "ru",
    "org",
    "net",
    "de",
    "uk",
    "jp",
    "fr",
    "it",
    "pl",
    "br",
    "cn",
    "in",
    "nl",
    "es",
    "cz",
    "eu",
    "ua",
    "ca",
    "au",
    "ch",
    "se",
    "ro",
    "gr",
    "at",
    "tv",
    "io",
    "me",
    "co",
    "info",
]
top_tlds_set = set(top_tlds)

keywords_ecommerce = [
    "shop",
    "store",
    "cart",
    "buy",
    "market",
    "mall",
    "deal",
    "pay",
]
keywords_media = [
    "news",
    "press",
    "media",
    "times",
    "post",
    "daily",
    "gazette",
    "journal",
]
keywords_video = ["video", "tv", "movie", "film", "tube", "stream"]
keywords_adult = ["adult", "sex", "porn", "xxx"]
keywords_tech = ["tech", "dev", "code", "cloud", "soft", "app", "web"]
keywords_finance = ["bank", "finance", "invest", "loan", "crypto", "coin"]
all_keywords = (
    keywords_ecommerce
    + keywords_media
    + keywords_video
    + keywords_adult
    + keywords_tech
    + keywords_finance
)


def calc_entropy(s):
    if not s:
        return 0.0
    length = len(s)
    counts = Counter(s)
    ent = 0.0
    for c in counts.values():
        p = c / length
        ent -= p * math.log2(p)
    return ent


lexical_rows = []
domain_tlds = {}

for did in all_selected_domains:
    name = domain_id_to_name.get(did, "")
    name_lower = name.lower()
    parts = name_lower.split(".")
    tld = parts[-1] if len(parts) > 1 else "unknown"
    domain_tlds[did] = tld

    d_len = max(len(name_lower), 1)
    num_dots = name_lower.count(".")
    num_hyphens = name_lower.count("-")
    num_digits = sum(c.isdigit() for c in name_lower)
    vowel_count = sum(c in "aeiou" for c in name_lower)
    consonant_count = sum(c in "bcdfghjklmnpqrstvwxyz" for c in name_lower)

    press_score = press_freedom_map.get(tld, median_press_score)
    has_press = 1.0 if tld in press_freedom_map else 0.0

    row = {
        "domain_id": did,
        "domain_len": d_len,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_digits": num_digits,
        "digit_ratio": num_digits / d_len,
        "has_www": 1.0 if name_lower.startswith("www.") else 0.0,
        "vowel_ratio": vowel_count / d_len,
        "consonant_ratio": consonant_count / d_len,
        "entropy": calc_entropy(name_lower),
        "has_multiple_subdomains": 1.0 if num_dots > 1 else 0.0,
        "press_freedom_score": press_score,
        "has_press_freedom": has_press,
        "is_authoritarian_tld": 1.0 if press_score > 60.0 else 0.0,
    }

    for t in top_tlds:
        row[f"tld_{t}"] = 1.0 if tld == t else 0.0
    row["tld_other"] = 1.0 if tld not in top_tlds_set else 0.0

    kw_ecom = sum(1.0 for kw in keywords_ecommerce if kw in name_lower)
    kw_med = sum(1.0 for kw in keywords_media if kw in name_lower)
    kw_vid = sum(1.0 for kw in keywords_video if kw in name_lower)
    kw_adlt = sum(1.0 for kw in keywords_adult if kw in name_lower)
    kw_tch = sum(1.0 for kw in keywords_tech if kw in name_lower)
    kw_fin = sum(1.0 for kw in keywords_finance if kw in name_lower)

    row["kw_ecom_sum"] = kw_ecom
    row["kw_media_sum"] = kw_med
    row["kw_video_sum"] = kw_vid
    row["kw_adult_sum"] = kw_adlt
    row["kw_tech_sum"] = kw_tch
    row["kw_finance_sum"] = kw_fin
    row["kw_total_sum"] = kw_ecom + kw_med + kw_vid + kw_adlt + kw_tch + kw_fin

    for kw in all_keywords:
        row[f"kw_{kw}"] = 1.0 if kw in name_lower else 0.0

    lexical_rows.append(row)

lexical_df = pd.DataFrame(lexical_rows)
del lexical_rows
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

cat_rows = []
for did in all_selected_domains:
    counts = domain_cat_counts.get(did, None)
    row = {"domain_id": did}
    if counts is not None and counts.sum() > 0:
        props = counts / counts.sum()
        row["has_url_category"] = 1.0
        for idx, c in enumerate(url_cat_categories):
            row[f"url_cat_{c}"] = float(props[idx])
    else:
        row["has_url_category"] = 0.0
        for c in url_cat_categories:
            row[f"url_cat_{c}"] = 0.0
    cat_rows.append(row)

cat_df = pd.DataFrame(cat_rows)
del cat_rows, domain_cat_counts, name_to_domain_id
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

# Bidirectional Neighbor Tracker Adoption (Outgoing & Incoming)
train_targets_float = train_targets.astype(np.float32)

# Outgoing neighbors: src in selected_domains, dst in train_domains
dst_train_idx = domain_to_train_idx[dst_arr]
out_nbr_mask = (src_sel_idx >= 0) & (dst_train_idx >= 0) & (src_arr != dst_arr)
u_out = src_sel_idx[out_nbr_mask]
v_out = dst_train_idx[out_nbr_mask]

A_out = csr_matrix(
    (np.ones(len(u_out), dtype=np.float32), (u_out, v_out)),
    shape=(num_selected, num_train_domains),
)
A_out.data = np.ones_like(A_out.data)
nbr_train_out_deg = np.array(A_out.sum(axis=1)).ravel()
out_nbr_tracker_counts = A_out.dot(train_targets_float)
out_nbr_tracker_norm = out_nbr_tracker_counts / np.maximum(
    nbr_train_out_deg[:, None], 1.0
)
del A_out
gc.collect()

# Incoming neighbors: src in train_domains, dst in selected_domains
src_train_idx = domain_to_train_idx[src_arr]
dst_sel_idx = domain_to_sel_idx[dst_arr]
in_nbr_mask = (src_train_idx >= 0) & (dst_sel_idx >= 0) & (src_arr != dst_arr)
v_in = src_train_idx[in_nbr_mask]
u_in = dst_sel_idx[in_nbr_mask]

A_in = csr_matrix(
    (np.ones(len(u_in), dtype=np.float32), (u_in, v_in)),
    shape=(num_selected, num_train_domains),
)
A_in.data = np.ones_like(A_in.data)
nbr_train_in_deg = np.array(A_in.sum(axis=1)).ravel()
in_nbr_tracker_counts = A_in.dot(train_targets_float)
in_nbr_tracker_norm = in_nbr_tracker_counts / np.maximum(
    nbr_train_in_deg[:, None], 1.0
)
del A_in
gc.collect()

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
    out_nbr_mask,
    u_out,
    v_out,
    src_train_idx,
    dst_sel_idx,
    in_nbr_mask,
    v_in,
    u_in,
    domain_to_sel_idx,
    domain_to_train_idx,
    tdid_to_tr_id_arr,
    train_targets_float,
)
gc.collect()

graph_rows = []
for idx, did in enumerate(all_selected_domains):
    out_deg = float(out_degrees[did])
    in_deg = float(in_degrees[did])
    tr_links = float(tracker_out_counts[did])
    nbr_out = float(nbr_train_out_deg[idx])
    nbr_in = float(nbr_train_in_deg[idx])
    nbr_tot = nbr_out + nbr_in

    row = {
        "domain_id": did,
        "out_degree": out_deg,
        "log_out_degree": math.log1p(out_deg),
        "in_degree": in_deg,
        "log_in_degree": math.log1p(in_deg),
        "total_degree": out_deg + in_deg,
        "degree_ratio": (in_deg + 1.0) / (out_deg + 1.0),
        "is_isolated": 1.0 if (out_deg + in_deg) == 0 else 0.0,
        "tracker_out_links": tr_links,
        "log_tracker_out_links": math.log1p(tr_links),
        "tracker_link_ratio": tr_links / max(out_deg, 1.0),
        "nbr_train_out_count": nbr_out,
        "log_nbr_train_out_count": math.log1p(nbr_out),
        "nbr_train_in_count": nbr_in,
        "log_nbr_train_in_count": math.log1p(nbr_in),
        "nbr_train_total_count": nbr_tot,
        "log_nbr_train_total_count": math.log1p(nbr_tot),
    }
    graph_rows.append(row)

graph_df = pd.DataFrame(graph_rows)
del (
    graph_rows,
    out_degrees,
    in_degrees,
    tracker_out_counts,
    nbr_train_out_deg,
    nbr_train_in_deg,
)
gc.collect()

# =========================================================================
# 6. Bayesian TLD Tracker Priors (Training Split Only)
# =========================================================================
tld_train_counts = defaultdict(int)
tld_tracker_counts = defaultdict(lambda: np.zeros(num_trackers, dtype=np.float32))

for did in train_domains:
    tld = domain_tlds.get(did, "unknown")
    tld_train_counts[tld] += 1
    idx = train_domain_to_idx[did]
    tld_tracker_counts[tld] += train_targets[idx]

alpha = 10.0
tld_priors = {}
for tld, counts in tld_tracker_counts.items():
    total = tld_train_counts[tld]
    smoothed = (counts + alpha * global_tracker_priors) / (total + alpha)
    tld_priors[tld] = smoothed

tld_prior_matrix = np.zeros((num_selected, num_trackers), dtype=np.float32)
prior_rows = []

for idx, did in enumerate(all_selected_domains):
    tld = domain_tlds.get(did, "unknown")
    prior = tld_priors.get(tld, global_tracker_priors)
    tld_prior_matrix[idx] = prior

    row = {
        "domain_id": did,
        "tld_prior_entropy": float(-np.sum(prior * np.log(prior + 1e-12))),
        "tld_prior_max": float(np.max(prior)),
        "tld_prior_top5_sum": float(np.sum(np.sort(prior)[::-1][:5])),
    }
    prior_rows.append(row)

prior_df = pd.DataFrame(prior_rows)
del prior_rows
gc.collect()

# =========================================================================
# 7. Merge Feature Sets & Materialize Tensors
# =========================================================================
tabular_df = lexical_df.merge(cat_df, on="domain_id", how="left")
tabular_df = tabular_df.merge(graph_df, on="domain_id", how="left")
tabular_df = tabular_df.merge(prior_df, on="domain_id", how="left")
tabular_df.fillna(0.0, inplace=True)

del lexical_df, cat_df, graph_df, prior_df
gc.collect()

scalar_cols = [c for c in tabular_df.columns if c != "domain_id"]
tabular_df.set_index("domain_id", inplace=True)
tabular_df = tabular_df.loc[all_selected_domains].reset_index()

X_scalar = tabular_df[scalar_cols].to_numpy(dtype=np.float32)
np.nan_to_num(X_scalar, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
del tabular_df
gc.collect()

# Concatenate scalar features with the 4 full 355-dimensional tracker-aligned blocks
# Block ordering:
# 1) Direct tracker link indicators (N, 355)
# 2) Outgoing neighbor tracker adoption (N, 355)
# 3) Incoming neighbor tracker adoption (N, 355)
# 4) Bayesian TLD tracker prior probabilities (N, 355)
X_all = np.hstack(
    [
        X_scalar,
        direct_links,
        out_nbr_tracker_norm,
        in_nbr_tracker_norm,
        tld_prior_matrix,
    ]
)

del X_scalar, direct_links, out_nbr_tracker_norm, in_nbr_tracker_norm, tld_prior_matrix
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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.has_tracker_priors = in_dim >= 4 * num_classes

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
        self.tracker_prototypes = nn.Parameter(
            torch.randn(num_classes, latent_dim) / math.sqrt(latent_dim)
        )
        self.head_blend = nn.Parameter(torch.tensor([0.5]))

        if self.has_tracker_priors:
            self.tracker_direct_proj = nn.Linear(4, 1, bias=False)
            with torch.no_grad():
                self.tracker_direct_proj.weight.copy_(
                    torch.tensor([[1.5, 1.0, 1.0, 0.5]])
                )
            self.tracker_res_net = nn.Sequential(
                nn.Linear(4, 32),
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
            prior_signals = x[:, -4 * self.num_classes :].reshape(
                -1, 4, self.num_classes
            )
            prior_perm = prior_signals.permute(0, 2, 1)
            direct_prior_logits = self.tracker_direct_proj(prior_perm).squeeze(-1)
            mlp_prior_logits = self.tracker_res_net(prior_perm).squeeze(-1)
            tracker_res_logits = direct_prior_logits + mlp_prior_logits
            return base_logits + self.res_scale * tracker_res_logits

        return base_logits


class AsymmetricFocalRecallLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip_margin: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        loss_pos = (
            -targets * (1.0 - pos_probs).pow(self.gamma_pos) * torch.log(pos_probs)
        )

        neg_probs = (probs - self.clip_margin).clamp(min=self.eps, max=1.0 - self.eps)
        neg_probs = torch.where(
            probs <= self.clip_margin, torch.zeros_like(neg_probs), neg_probs
        )
        loss_neg = (
            -(1.0 - targets)
            * neg_probs.pow(self.gamma_neg)
            * torch.log(1.0 - neg_probs)
        )
        return (loss_pos + loss_neg).sum(dim=-1).mean()


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
# 9. Training, Validation & Checkpoint Selection
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TrackerRankNet(
    in_dim=in_features,
    num_classes=num_trackers,
    hidden_dim=512,
    latent_dim=128,
    dropout_rate=0.2,
    initial_bias=init_biases,
).to(device)

criterion = AsymmetricFocalRecallLoss(gamma_pos=0.0, gamma_neg=2.0, clip_margin=0.05)

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
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
)

num_epochs = 12
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

batch_size = 512
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
test_dataset = TensorDataset(torch.from_numpy(X_test))

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

best_val_recall = -1.0
best_model_path = "working/best_tracker_ranknet.pt"

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_train_loss / max(num_batches, 1)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_logits = model(batch_x)
            val_preds_list.append(batch_logits.cpu().numpy())

    val_preds = np.vstack(val_preds_list)
    val_recall_10 = compute_recall_at_k(val_preds, Y_val, k=10)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
        f"Val Recall@10: {val_recall_10:.5f} | LR: {current_lr:.6f}"
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
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_logits = model(batch_x)
        final_val_preds_list.append(batch_logits.cpu().numpy())

final_val_preds = np.vstack(final_val_preds_list)
final_val_score = compute_recall_at_k(final_val_preds, Y_val, k=10)

test_preds_list = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)
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

print(f"Final Validation Score: {final_val_score:.5f}")
