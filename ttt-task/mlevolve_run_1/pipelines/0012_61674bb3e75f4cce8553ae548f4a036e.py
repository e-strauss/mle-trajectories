from collections import Counter, defaultdict
import copy
import gc
import json
import math
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for strict reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("working", exist_ok=True)
os.makedirs("submission", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Load Trackers Metadata & Target Domain IDs
# -----------------------------------------------------------------------------
trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
tracker_id_to_tracking_domain_id = dict(
    zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
)
num_trackers = len(trackers_df)

target_df = pd.read_csv("input/target.tsv", sep="\t")
target_domain_ids = target_df["domain_id"].unique()
target_domain_set = set(target_domain_ids)

# -----------------------------------------------------------------------------
# 2. Load Training Tracking Graph and Construct Disjoint Splits
# -----------------------------------------------------------------------------
train_graph_df = pd.read_parquet(
    "input/tracking_graph_train.parquet", columns=["domain_id", "tracker_id"]
)

# Strict isolation: filter out any target domains that might appear in the training graph
train_graph_df = train_graph_df[~train_graph_df["domain_id"].isin(target_domain_set)]
all_train_domains = train_graph_df["domain_id"].unique()

# Stratified sampling: guarantee full representation across all candidate trackers
stratified_domains = set()
for t_id in range(num_trackers):
    t_domains = train_graph_df.loc[
        train_graph_df["tracker_id"] == t_id, "domain_id"
    ].unique()
    if len(t_domains) > 0:
        sample_size = min(250, len(t_domains))
        sampled = np.random.choice(t_domains, size=sample_size, replace=False)
        stratified_domains.update(sampled)

remaining_domains = list(set(all_train_domains) - stratified_domains)
needed_additional = max(0, 260000 - len(stratified_domains))
sampled_additional = np.random.choice(
    remaining_domains,
    size=min(needed_additional, len(remaining_domains)),
    replace=False,
)

selected_pool = np.array(list(stratified_domains) + list(sampled_additional))
np.random.shuffle(selected_pool)

train_size = min(250000, len(selected_pool) - 10000)
val_size = min(10000, len(selected_pool) - train_size)

train_domain_ids = selected_pool[:train_size]
val_domain_ids = selected_pool[train_size : train_size + val_size]

train_domain_set = set(train_domain_ids)
val_domain_set = set(val_domain_ids)
all_needed_domain_set = train_domain_set | val_domain_set | target_domain_set
all_needed_domain_arr = np.array(sorted(all_needed_domain_set), dtype=np.int64)

# Build multi-hot target matrices and validation ground-truth dictionary
train_edges = train_graph_df[train_graph_df["domain_id"].isin(train_domain_set)]
val_edges = train_graph_df[train_graph_df["domain_id"].isin(val_domain_set)]

train_domain_to_idx = {d_id: i for i, d_id in enumerate(train_domain_ids)}
val_domain_to_idx = {d_id: i for i, d_id in enumerate(val_domain_ids)}

Y_train = np.zeros((len(train_domain_ids), num_trackers), dtype=np.float32)
for row in train_edges.itertuples(index=False):
    Y_train[train_domain_to_idx[row.domain_id], row.tracker_id] = 1.0

Y_val = np.zeros((len(val_domain_ids), num_trackers), dtype=np.float32)
val_ground_truth = {}
for row in val_edges.itertuples(index=False):
    idx = val_domain_to_idx[row.domain_id]
    Y_val[idx, row.tracker_id] = 1.0
    d_str = str(row.domain_id)
    if d_str not in val_ground_truth:
        val_ground_truth[d_str] = []
    val_ground_truth[d_str].append(int(row.tracker_id))

# Save validation ground truth for auditable tracking
with open("working/val_ground_truth.json", "w") as f:
    json.dump(val_ground_truth, f)

# Calculate empirical tracker priors and log-odds strictly on training set
train_tracker_counts = Y_train.sum(axis=0)
train_tracker_priors = train_tracker_counts / len(train_domain_ids)
clipped_priors = np.clip(train_tracker_priors, 1e-4, 1.0 - 1e-4)
prior_log_odds = np.log(clipped_priors / (1.0 - clipped_priors))

# Precompute empirical tracker co-occurrence conditional probability matrix P(tracker_j | tracker_i)
cooccur = Y_train.T @ Y_train
diag_counts = np.maximum(np.diag(cooccur).copy(), 1.0)
cooccur_cond_prob = (cooccur / diag_counts[:, None]).astype(np.float32)
np.fill_diagonal(cooccur_cond_prob, 0.0)

# Precompute thresholded Normalized Pointwise Mutual Information (NPMI) matrix
num_train_samples = float(len(train_domain_ids))
p_i = diag_counts / num_train_samples
p_ij = cooccur / num_train_samples
p_i_outer = np.outer(p_i, p_i)

with np.errstate(divide="ignore", invalid="ignore"):
    pmi = np.log(np.maximum(p_ij, 1e-12)) - np.log(np.maximum(p_i_outer, 1e-12))
    norm_factor = -np.log(np.maximum(p_ij, 1e-12))
    npmi = pmi / np.maximum(norm_factor, 1e-12)

npmi[p_ij == 0] = 0.0
npmi = np.clip(npmi, 0.0, 1.0)
np.fill_diagonal(npmi, 0.0)

# Row-normalize thresholded NPMI to create a scale-invariant diffusion operator
npmi_row_sums = npmi.sum(axis=1, keepdims=True)
npmi_norm = np.where(
    npmi_row_sums > 0, npmi / np.maximum(npmi_row_sums, 1e-6), 0.0
).astype(np.float32)

# Precompute Company-Level Co-ownership Diffusion Matrix M_company derived from trackers.tsv
tracker_companies = trackers_df.set_index("tracker_id")["company"].to_dict()
tracker_brands = trackers_df.set_index("tracker_id")["brand"].to_dict()

company_cooccur = np.zeros((num_trackers, num_trackers), dtype=np.float32)
for i in range(num_trackers):
    c_i = str(tracker_companies.get(i, "")).strip()
    b_i = str(tracker_brands.get(i, "")).strip()
    grp_i = c_i if c_i and c_i != "#" else (b_i if b_i and b_i != "#" else f"unknown_{i}")
    for j in range(num_trackers):
        if i == j:
            continue
        c_j = str(tracker_companies.get(j, "")).strip()
        b_j = str(tracker_brands.get(j, "")).strip()
        grp_j = c_j if c_j and c_j != "#" else (b_j if b_j and b_j != "#" else f"unknown_{j}")
        if grp_i == grp_j and not grp_i.startswith("unknown_"):
            company_cooccur[i, j] = 1.0

company_row_sums = company_cooccur.sum(axis=1, keepdims=True)
company_norm = np.where(
    company_row_sums > 0, company_cooccur / np.maximum(company_row_sums, 1e-6), 0.0
).astype(np.float32)

del train_graph_df, train_edges, val_edges, cooccur, diag_counts, company_cooccur
gc.collect()

# -----------------------------------------------------------------------------
# 3. Load Hostnames & Compute Domain Features
# -----------------------------------------------------------------------------
domains_df = pd.read_parquet("input/domains.parquet")
domains_df = domains_df[domains_df["domain_id"].isin(all_needed_domain_set)].copy()
domains_df = domains_df.drop_duplicates(subset=["domain_id"])
domains_df["domain"] = domains_df["domain"].astype(str).str.lower()

# Morphological features
domain_series = domains_df["domain"]
length = domain_series.str.len()
num_dots = domain_series.str.count(r"\.")
num_hyphens = domain_series.str.count(r"-")
num_digits = domain_series.str.count(r"\d")
digit_ratio = num_digits / (length + 1e-5)
num_vowels = domain_series.str.count(r"[aeiou]")
vowel_ratio = num_vowels / (length + 1e-5)
has_www = domain_series.str.startswith("www.").astype(float)
char_diversity = (domain_series.apply(lambda s: len(set(s))) / (length + 1e-5)).astype(
    float
)

tld_series = domain_series.str.split(".").str[-1]
is_cctld = (tld_series.str.len() == 2).astype(float)

domains_df["feat_len"] = length.astype(float)
domains_df["feat_num_dots"] = num_dots.astype(float)
domains_df["feat_num_hyphens"] = num_hyphens.astype(float)
domains_df["feat_num_digits"] = num_digits.astype(float)
domains_df["feat_digit_ratio"] = digit_ratio.astype(float)
domains_df["feat_vowel_ratio"] = vowel_ratio.astype(float)
domains_df["feat_has_www"] = has_www
domains_df["feat_char_diversity"] = char_diversity
domains_df["feat_is_cctld"] = is_cctld

# Character entropy computation
def calc_char_entropy(s: str) -> float:
    if not s:
        return 0.0
    s_len = len(s)
    counts = Counter(s)
    return -sum((c / s_len) * math.log2(c / s_len) for c in counts.values())

domains_df["feat_entropy"] = [calc_char_entropy(s) for s in domain_series]

# Second-Level Domain (SLD) morphology extraction
COMMON_SECOND_LEVEL_TLDS = {"co", "com", "org", "net", "edu", "gov", "ac", "ne", "or"}

def extract_sld(domain_str: str) -> str:
    parts = domain_str.strip().lower().split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in COMMON_SECOND_LEVEL_TLDS:
        return parts[-3]
    elif len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""

sld_series = domain_series.apply(extract_sld)
sld_len = sld_series.str.len()
sld_hyphens = sld_series.str.count(r"-")
sld_digits = sld_series.str.count(r"\d")
sld_vowels = sld_series.str.count(r"[aeiou]")

domains_df["feat_sld_len"] = sld_len.astype(float)
domains_df["feat_sld_digit_ratio"] = (sld_digits / (sld_len + 1e-5)).astype(float)
domains_df["feat_sld_hyphen_ratio"] = (sld_hyphens / (sld_len + 1e-5)).astype(float)
domains_df["feat_sld_vowel_ratio"] = (sld_vowels / (sld_len + 1e-5)).astype(float)
domains_df["feat_sld_entropy"] = [calc_char_entropy(s) for s in sld_series]

# Vertical keyword indicators expanded to 24 commercial verticals
vertical_keywords = [
    "shop", "blog", "news", "forum", "tv", "gov", "ad", "media",
    "video", "tech", "health", "game", "finance", "press", "travel",
    "music", "sports", "food", "auto", "edu", "realestate", "job",
    "fashion", "crypto"
]
for kw in vertical_keywords:
    domains_df[f"feat_has_kw_{kw}"] = domain_series.str.contains(kw, regex=False).astype(float)

# Compute TLD frequency strictly on training data
train_mask = domains_df["domain_id"].isin(train_domain_set)
train_tlds = tld_series.loc[train_mask]
tld_freq_map = train_tlds.value_counts().to_dict()
domains_df["feat_tld_log_freq"] = np.log1p(
    tld_series.map(tld_freq_map).fillna(0).astype(float)
)

# Press freedom score mapping
try:
    fop_df = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
    if "freedom_of_the_press" not in fop_df.columns:
        fop_df = pd.read_csv("input/freedom-of-the-press.csv")
except Exception:
    fop_df = pd.DataFrame(columns=["tld", "country", "freedom_of_the_press"])

fop_map = dict(zip(fop_df["tld"].str.lower(), fop_df["freedom_of_the_press"]))
train_fop_scores = [
    fop_map[t] for t in train_tlds if t in fop_map and pd.notna(fop_map[t])
]
median_fop = float(np.median(train_fop_scores)) if train_fop_scores else 50.0

domains_df["feat_press_freedom"] = (
    tld_series.map(fop_map).fillna(median_fop).astype(float)
)
domains_df["feat_has_press_freedom"] = (tld_series.isin(fop_map).astype(float)).astype(
    float
)
domains_df["feat_press_freedom_cctld"] = (
    domains_df["feat_press_freedom"] * domains_df["feat_is_cctld"]
)

top_tlds = train_tlds.value_counts().head(80).index.tolist()
tld_dummies = pd.DataFrame(
    {f"feat_tld_is_{t}": (tld_series == t).astype(float) for t in top_tlds},
    index=domains_df.index,
)
domains_df = pd.concat([domains_df, tld_dummies], axis=1)

# -----------------------------------------------------------------------------
# 4. Feature Engineering: URL Classification Categories
# -----------------------------------------------------------------------------
url_df = pd.read_csv("input/url-classification.csv")
urls_clean = (
    url_df["url"].astype(str).str.lower().str.replace(r"^https?://", "", regex=True)
)
url_hosts = (
    urls_clean.str.split("/").str[0]
    .str.split(":").str[0]
    .str.replace(r"^www\d*\.", "", regex=True)
    .str.strip()
)
url_df["clean_host"] = url_hosts
url_df["apex_domain"] = url_hosts.apply(
    lambda h: ".".join(h.split(".")[-2:]) if "." in h else h
)

url_categories = sorted(url_df["category"].dropna().unique())

# Exact host category distribution
cat_matrix_host = (
    url_df.groupby(["clean_host", "category"]).size().unstack(fill_value=0)
)
total_cat_counts_host = cat_matrix_host[url_categories].sum(axis=1)
cat_prob_host = cat_matrix_host[url_categories].div(
    total_cat_counts_host.replace(0, 1), axis=0
)
cat_prob_host["feat_url_cat_log_total"] = np.log1p(total_cat_counts_host.astype(float))

# Registered apex domain category distribution
cat_matrix_apex = (
    url_df.groupby(["apex_domain", "category"]).size().unstack(fill_value=0)
)
total_cat_counts_apex = cat_matrix_apex[url_categories].sum(axis=1)
cat_prob_apex = cat_matrix_apex[url_categories].div(
    total_cat_counts_apex.replace(0, 1), axis=0
)
cat_prob_apex["feat_url_cat_log_total"] = np.log1p(total_cat_counts_apex.astype(float))

domains_df["clean_host"] = (
    domain_series.str.replace(r"^www\d*\.", "", regex=True).str.strip().str.lower()
)
domains_df["apex_domain"] = domains_df["clean_host"].apply(
    lambda h: ".".join(h.split(".")[-2:]) if "." in h else h
)

# First merge exact host distributions
domains_df = domains_df.merge(
    cat_prob_host,
    left_on="clean_host",
    right_index=True,
    how="left",
)

# Two-tier fallback: for unmatched hosts, merge registered apex domain distributions
unmatched_mask = domains_df["feat_url_cat_log_total"].isna()
if unmatched_mask.any():
    apex_prob_dict = {cat: cat_prob_apex[cat].to_dict() for cat in url_categories}
    apex_total_dict = cat_prob_apex["feat_url_cat_log_total"].to_dict()
    unmatched_apex = domains_df.loc[unmatched_mask, "apex_domain"]

    for cat in url_categories:
        domains_df.loc[unmatched_mask, cat] = unmatched_apex.map(apex_prob_dict[cat])
    domains_df.loc[unmatched_mask, "feat_url_cat_log_total"] = unmatched_apex.map(
        apex_total_dict
    )

for cat in url_categories:
    domains_df[f"feat_cat_{cat}"] = domains_df[cat].fillna(0.0).astype(float)
    if cat in domains_df.columns:
        domains_df.drop(columns=[cat], inplace=True)

domains_df["feat_url_cat_log_total"] = (
    domains_df["feat_url_cat_log_total"].fillna(0.0).astype(float)
)
domains_df["feat_has_url_cat"] = (
    domains_df["feat_url_cat_log_total"] > 0
).astype(float)

domains_df.drop(columns=["apex_domain"], inplace=True)

del url_df, urls_clean, url_hosts, cat_matrix_host, cat_matrix_apex
del cat_prob_host, cat_prob_apex, total_cat_counts_host, total_cat_counts_apex
gc.collect()

# -----------------------------------------------------------------------------
# 5. Feature Engineering: Bidirectional Link Graph Streaming & Direct Tracker Hyperlinks
# -----------------------------------------------------------------------------
track_dom_to_trackers = defaultdict(list)
for row in trackers_df.itertuples(index=False):
    track_dom_to_trackers[row.tracking_domain_id].append(row.tracker_id)
tracking_domain_ids_arr = np.array(sorted(track_dom_to_trackers.keys()), dtype=np.int64)

in_degrees = {}
out_degrees = {}
direct_out_counts = np.zeros((len(domains_df), num_trackers), dtype=np.float32)
direct_in_counts = np.zeros((len(domains_df), num_trackers), dtype=np.float32)

# Disentangled directional neighbor tracker affinity accumulators
domain_ids_list = domains_df["domain_id"].values
domain_id_to_row_idx = {d_id: i for i, d_id in enumerate(domain_ids_list)}
num_target_domains = len(domain_ids_list)

neighbor_out_tracker_counts = np.zeros((num_target_domains, num_trackers), dtype=np.float32)
neighbor_out_train_count = np.zeros(num_target_domains, dtype=np.float32)
neighbor_in_tracker_counts = np.zeros((num_target_domains, num_trackers), dtype=np.float32)
neighbor_in_train_count = np.zeros(num_target_domains, dtype=np.float32)

max_dom_id = max(domains_df["domain_id"].max(), tracking_domain_ids_arr.max())
train_domain_arr = np.array(sorted(train_domain_set), dtype=np.int64)

if max_dom_id < 80000000:
    all_dom_lookup = np.full(int(max_dom_id) + 1, -1, dtype=np.int32)
    all_dom_lookup[domain_ids_list] = np.arange(num_target_domains, dtype=np.int32)

    train_dom_lookup = np.full(int(max_dom_id) + 1, -1, dtype=np.int32)
    for d_id, idx in train_domain_to_idx.items():
        train_dom_lookup[d_id] = idx
else:
    all_dom_lookup = None
    train_dom_lookup = None

pfile = pq.ParquetFile("input/link-graph.parquet")
for batch in pfile.iter_batches(
    batch_size=2500000, columns=["source_domain_id", "target_domain_id"]
):
    src = batch.column("source_domain_id").to_numpy()
    dst = batch.column("target_domain_id").to_numpy()

    src_mask = np.isin(src, all_needed_domain_arr)
    if np.any(src_mask):
        matched_src, counts_src = np.unique(src[src_mask], return_counts=True)
        for d_id, c in zip(matched_src, counts_src):
            out_degrees[d_id] = out_degrees.get(d_id, 0) + c

        # Extract direct host-to-tracker out-links with frequency accumulation
        sub_src = src[src_mask]
        sub_dst = dst[src_mask]
        tracker_hit_mask = np.isin(sub_dst, tracking_domain_ids_arr)
        if np.any(tracker_hit_mask):
            for s_id, t_dom in zip(sub_src[tracker_hit_mask], sub_dst[tracker_hit_mask]):
                r_idx = domain_id_to_row_idx.get(s_id, -1)
                if r_idx >= 0:
                    for t_id in track_dom_to_trackers[t_dom]:
                        direct_out_counts[r_idx, t_id] += 1.0

    dst_mask = np.isin(dst, all_needed_domain_arr)
    if np.any(dst_mask):
        matched_dst, counts_dst = np.unique(dst[dst_mask], return_counts=True)
        for d_id, c in zip(matched_dst, counts_dst):
            in_degrees[d_id] = in_degrees.get(d_id, 0) + c

        # Extract direct tracker-to-host in-links with frequency accumulation
        sub_src_in = src[dst_mask]
        sub_dst_in = dst[dst_mask]
        tracker_src_mask = np.isin(sub_src_in, tracking_domain_ids_arr)
        if np.any(tracker_src_mask):
            for t_dom, d_id in zip(sub_src_in[tracker_src_mask], sub_dst_in[tracker_src_mask]):
                r_idx = domain_id_to_row_idx.get(d_id, -1)
                if r_idx >= 0:
                    for t_id in track_dom_to_trackers[t_dom]:
                        direct_in_counts[r_idx, t_id] += 1.0

    # Accumulate out-of-fold neighbor tracker affinity from known training domain neighbors
    if all_dom_lookup is not None:
        src_valid = (src <= max_dom_id) & (src >= 0)
        dst_valid = (dst <= max_dom_id) & (dst >= 0)

        s_all = np.full(len(src), -1, dtype=np.int32)
        s_all[src_valid] = all_dom_lookup[src[src_valid]]

        d_all = np.full(len(dst), -1, dtype=np.int32)
        d_all[dst_valid] = all_dom_lookup[dst[dst_valid]]

        s_tr = np.full(len(src), -1, dtype=np.int32)
        s_tr[src_valid] = train_dom_lookup[src[src_valid]]

        d_tr = np.full(len(dst), -1, dtype=np.int32)
        d_tr[dst_valid] = train_dom_lookup[dst[dst_valid]]

        # Direction 1: target/val/train src connects to train dst (outgoing neighbor affinity)
        m1 = (s_all >= 0) & (d_tr >= 0) & (src != dst)
        if np.any(m1):
            s_idx = s_all[m1]
            d_tidx = d_tr[m1]
            np.add.at(neighbor_out_tracker_counts, s_idx, Y_train[d_tidx])
            np.add.at(neighbor_out_train_count, s_idx, 1.0)

        # Direction 2: target/val/train dst connects from train src (incoming backlink affinity)
        m2 = (d_all >= 0) & (s_tr >= 0) & (src != dst)
        if np.any(m2):
            d_idx = d_all[m2]
            s_tidx = s_tr[m2]
            np.add.at(neighbor_in_tracker_counts, d_idx, Y_train[s_tidx])
            np.add.at(neighbor_in_train_count, d_idx, 1.0)
    else:
        m1 = np.isin(src, all_needed_domain_arr) & np.isin(dst, train_domain_arr) & (src != dst)
        if np.any(m1):
            s_idx = np.array([domain_id_to_row_idx[x] for x in src[m1]], dtype=np.int64)
            d_tidx = np.array([train_domain_to_idx[x] for x in dst[m1]], dtype=np.int64)
            np.add.at(neighbor_out_tracker_counts, s_idx, Y_train[d_tidx])
            np.add.at(neighbor_out_train_count, s_idx, 1.0)

        m2 = np.isin(dst, all_needed_domain_arr) & np.isin(src, train_domain_arr) & (src != dst)
        if np.any(m2):
            d_idx = np.array([domain_id_to_row_idx[x] for x in dst[m2]], dtype=np.int64)
            s_tidx = np.array([train_domain_to_idx[x] for x in src[m2]], dtype=np.int64)
            np.add.at(neighbor_in_tracker_counts, d_idx, Y_train[s_tidx])
            np.add.at(neighbor_in_train_count, d_idx, 1.0)

domains_df["feat_in_degree"] = (
    domains_df["domain_id"].map(in_degrees).fillna(0).astype(float)
)
domains_df["feat_out_degree"] = (
    domains_df["domain_id"].map(out_degrees).fillna(0).astype(float)
)
domains_df["feat_total_degree"] = (
    domains_df["feat_in_degree"] + domains_df["feat_out_degree"]
)
domains_df["feat_log_in_degree"] = np.log1p(domains_df["feat_in_degree"])
domains_df["feat_log_out_degree"] = np.log1p(domains_df["feat_out_degree"])
domains_df["feat_degree_ratio"] = domains_df["feat_in_degree"] / (
    domains_df["feat_out_degree"] + 1.0
)
domains_df["feat_is_isolated"] = (domains_df["feat_total_degree"] == 0).astype(float)

# Directional neighbor degree statistics
domains_df["feat_neighbor_out_train_count"] = neighbor_out_train_count.astype(float)
domains_df["feat_log_neighbor_out_train_count"] = np.log1p(neighbor_out_train_count).astype(float)
domains_df["feat_has_neighbor_out"] = (neighbor_out_train_count > 0).astype(float)
domains_df["feat_neighbor_in_train_count"] = neighbor_in_train_count.astype(float)
domains_df["feat_log_neighbor_in_train_count"] = np.log1p(neighbor_in_train_count).astype(float)
domains_df["feat_has_neighbor_in"] = (neighbor_in_train_count > 0).astype(float)

# Binary indicators and log1p frequency counts for direct tracker links
direct_out_bin_matrix = (direct_out_counts > 0).astype(np.float32)
direct_out_log_matrix = np.log1p(direct_out_counts).astype(np.float32)
direct_in_bin_matrix = (direct_in_counts > 0).astype(np.float32)
direct_in_log_matrix = np.log1p(direct_in_counts).astype(np.float32)

num_direct_out = direct_out_bin_matrix.sum(axis=1)
num_direct_in = direct_in_bin_matrix.sum(axis=1)
domains_df["feat_num_direct_out"] = num_direct_out
domains_df["feat_num_direct_in"] = num_direct_in
domains_df["feat_has_direct_out"] = (num_direct_out > 0).astype(float)
domains_df["feat_has_direct_in"] = (num_direct_in > 0).astype(float)
domains_df["feat_total_direct_trackers"] = num_direct_out + num_direct_in
domains_df["feat_has_direct_tracker"] = (
    domains_df["feat_total_direct_trackers"] > 0
).astype(float)

# Disentangled directional normalized neighbor tracker affinity
deg_out_denom = np.maximum(neighbor_out_train_count[:, None], 1.0)
neighbor_out_aff_matrix = np.where(
    neighbor_out_train_count[:, None] > 0, neighbor_out_tracker_counts / deg_out_denom, 0.0
).astype(np.float32)

deg_in_denom = np.maximum(neighbor_in_train_count[:, None], 1.0)
neighbor_in_aff_matrix = np.where(
    neighbor_in_train_count[:, None] > 0, neighbor_in_tracker_counts / deg_in_denom, 0.0
).astype(np.float32)

direct_out_bin_cols = [f"feat_direct_out_bin_{i}" for i in range(num_trackers)]
direct_out_log_cols = [f"feat_direct_out_log_{i}" for i in range(num_trackers)]
direct_in_bin_cols = [f"feat_direct_in_bin_{i}" for i in range(num_trackers)]
direct_in_log_cols = [f"feat_direct_in_log_{i}" for i in range(num_trackers)]
neighbor_out_aff_cols = [f"feat_neighbor_out_aff_{i}" for i in range(num_trackers)]
neighbor_in_aff_cols = [f"feat_neighbor_in_aff_{i}" for i in range(num_trackers)]

tracker_feat_dict = {}
for i in range(num_trackers):
    tracker_feat_dict[direct_out_bin_cols[i]] = direct_out_bin_matrix[:, i]
    tracker_feat_dict[direct_out_log_cols[i]] = direct_out_log_matrix[:, i]
    tracker_feat_dict[direct_in_bin_cols[i]] = direct_in_bin_matrix[:, i]
    tracker_feat_dict[direct_in_log_cols[i]] = direct_in_log_matrix[:, i]
    tracker_feat_dict[neighbor_out_aff_cols[i]] = neighbor_out_aff_matrix[:, i]
    tracker_feat_dict[neighbor_in_aff_cols[i]] = neighbor_in_aff_matrix[:, i]

tracker_feat_df = pd.DataFrame(tracker_feat_dict, index=domains_df.index)
domains_df = pd.concat([domains_df, tracker_feat_df], axis=1)

del (
    in_degrees,
    out_degrees,
    direct_out_counts,
    direct_in_counts,
    direct_out_bin_matrix,
    direct_out_log_matrix,
    direct_in_bin_matrix,
    direct_in_log_matrix,
    neighbor_out_tracker_counts,
    neighbor_out_train_count,
    neighbor_in_tracker_counts,
    neighbor_in_train_count,
    neighbor_out_aff_matrix,
    neighbor_in_aff_matrix,
    tracker_feat_dict,
    tracker_feat_df,
    num_direct_out,
    num_direct_in,
    domain_id_to_row_idx,
    track_dom_to_trackers,
    domain_ids_list,
)
if all_dom_lookup is not None:
    del all_dom_lookup, train_dom_lookup
gc.collect()

# -----------------------------------------------------------------------------
# 6. Feature Engineering: TF-IDF Character N-Grams + TruncatedSVD
# -----------------------------------------------------------------------------
tfidf = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5), max_features=5000, sublinear_tf=True
)
train_domains_list = domains_df.loc[train_mask, "domain"].astype(str).values.tolist()
tfidf.fit(train_domains_list)

svd = TruncatedSVD(n_components=128, random_state=42)
train_tfidf_mat = tfidf.transform(train_domains_list)
svd.fit(train_tfidf_mat)

all_tfidf_mat = tfidf.transform(domains_df["domain"].astype(str).values.tolist())
all_svd_emb = svd.transform(all_tfidf_mat)

svd_cols = [f"feat_svd_{i}" for i in range(128)]
svd_df = pd.DataFrame(
    all_svd_emb.astype(np.float32), columns=svd_cols, index=domains_df.index
)
domains_df = pd.concat([domains_df, svd_df], axis=1)

del tfidf, svd, all_tfidf_mat, all_svd_emb, train_tfidf_mat, svd_df
gc.collect()

# -----------------------------------------------------------------------------
# 7. Partition Features & Scale Strictly Fit on Train & Build Tensors
# -----------------------------------------------------------------------------
domains_df_indexed = domains_df.set_index("domain_id")

direct_out_cols = direct_out_bin_cols + direct_out_log_cols
direct_in_cols = direct_in_bin_cols + direct_in_log_cols
neighbor_out_cols = neighbor_out_aff_cols
neighbor_in_cols = neighbor_in_aff_cols
raw_tracker_cols_set = set(
    direct_out_cols + direct_in_cols + neighbor_out_cols + neighbor_in_cols
)

dense_feature_cols = [
    c for c in domains_df.columns if c.startswith("feat_") and c not in raw_tracker_cols_set
]

train_dense = (
    domains_df_indexed.reindex(train_domain_ids)[dense_feature_cols]
    .fillna(0)
    .values.astype(np.float32)
)
val_dense = (
    domains_df_indexed.reindex(val_domain_ids)[dense_feature_cols]
    .fillna(0)
    .values.astype(np.float32)
)
test_dense = (
    domains_df_indexed.reindex(target_domain_ids)[dense_feature_cols]
    .fillna(0)
    .values.astype(np.float32)
)

scaler = StandardScaler()
scaler.fit(train_dense)

train_dense_scaled = scaler.transform(train_dense).astype(np.float32)
val_dense_scaled = scaler.transform(val_dense).astype(np.float32)
test_dense_scaled = scaler.transform(test_dense).astype(np.float32)

# Preserve unstandardized non-negative tracker graph signals to prevent zero-value negative biases
train_raw_out = (
    domains_df_indexed.reindex(train_domain_ids)[direct_out_cols]
    .fillna(0)
    .values.astype(np.float32)
)
val_raw_out = (
    domains_df_indexed.reindex(val_domain_ids)[direct_out_cols]
    .fillna(0)
    .values.astype(np.float32)
)
test_raw_out = (
    domains_df_indexed.reindex(target_domain_ids)[direct_out_cols]
    .fillna(0)
    .values.astype(np.float32)
)

train_raw_in = (
    domains_df_indexed.reindex(train_domain_ids)[direct_in_cols]
    .fillna(0)
    .values.astype(np.float32)
)
val_raw_in = (
    domains_df_indexed.reindex(val_domain_ids)[direct_in_cols]
    .fillna(0)
    .values.astype(np.float32)
)
test_raw_in = (
    domains_df_indexed.reindex(target_domain_ids)[direct_in_cols]
    .fillna(0)
    .values.astype(np.float32)
)

train_raw_neighbor_out = (
    domains_df_indexed.reindex(train_domain_ids)[neighbor_out_cols]
    .fillna(0)
    .values.astype(np.float32)
)
val_raw_neighbor_out = (
    domains_df_indexed.reindex(val_domain_ids)[neighbor_out_cols]
    .fillna(0)
    .values.astype(np.float32)
)
test_raw_neighbor_out = (
    domains_df_indexed.reindex(target_domain_ids)[neighbor_out_cols]
    .fillna(0)
    .values.astype(np.float32)
)

train_raw_neighbor_in = (
    domains_df_indexed.reindex(train_domain_ids)[neighbor_in_cols]
    .fillna(0)
    .values.astype(np.float32)
)
val_raw_neighbor_in = (
    domains_df_indexed.reindex(val_domain_ids)[neighbor_in_cols]
    .fillna(0)
    .values.astype(np.float32)
)
test_raw_neighbor_in = (
    domains_df_indexed.reindex(target_domain_ids)[neighbor_in_cols]
    .fillna(0)
    .values.astype(np.float32)
)

num_dense = train_dense_scaled.shape[1]
len_out = len(direct_out_cols)
len_in = len(direct_in_cols)
len_nout = len(neighbor_out_cols)
len_nin = len(neighbor_in_cols)

dense_indices = list(range(0, num_dense))
direct_out_indices = list(range(num_dense, num_dense + len_out))
direct_in_indices = list(range(num_dense + len_out, num_dense + len_out + len_in))
neighbor_out_indices = list(
    range(num_dense + len_out + len_in, num_dense + len_out + len_in + len_nout)
)
neighbor_in_indices = list(
    range(
        num_dense + len_out + len_in + len_nout,
        num_dense + len_out + len_in + len_nout + len_nin,
    )
)
neighbor_indices = neighbor_out_indices + neighbor_in_indices

X_train = np.hstack(
    [
        train_dense_scaled,
        train_raw_out,
        train_raw_in,
        train_raw_neighbor_out,
        train_raw_neighbor_in,
    ]
)
X_val = np.hstack(
    [
        val_dense_scaled,
        val_raw_out,
        val_raw_in,
        val_raw_neighbor_out,
        val_raw_neighbor_in,
    ]
)
X_test = np.hstack(
    [
        test_dense_scaled,
        test_raw_out,
        test_raw_in,
        test_raw_neighbor_out,
        test_raw_neighbor_in,
    ]
)

del (
    domains_df,
    domains_df_indexed,
    train_dense,
    val_dense,
    test_dense,
    train_dense_scaled,
    val_dense_scaled,
    test_dense_scaled,
    train_raw_out,
    val_raw_out,
    test_raw_out,
    train_raw_in,
    val_raw_in,
    test_raw_in,
    train_raw_neighbor_out,
    val_raw_neighbor_out,
    test_raw_neighbor_out,
    train_raw_neighbor_in,
    val_raw_neighbor_in,
    test_raw_neighbor_in,
)
gc.collect()


# -----------------------------------------------------------------------------
# 8. Model Architecture & Smooth Pairwise Ranking + Asymmetric Loss
# -----------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout_rate)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop1(self.act1(self.ln1(self.fc1(x))))
        out = self.drop2(self.ln2(self.fc2(out)))
        return self.act2(out + residual)


class TrackerRankingResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_trackers: int = 355,
        hidden_dim: int = 384,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        init_log_odds: np.ndarray = None,
        cooccur_matrix: np.ndarray = None,
        npmi_matrix: np.ndarray = None,
        company_matrix: np.ndarray = None,
        direct_out_indices: list = None,
        direct_in_indices: list = None,
        neighbor_out_indices: list = None,
        neighbor_in_indices: list = None,
        neighbor_indices: list = None,
        dense_indices: list = None,
    ):
        super().__init__()
        dense_dim = len(dense_indices) if dense_indices is not None else in_features
        if dense_indices is not None:
            self.register_buffer(
                "dense_indices", torch.tensor(dense_indices, dtype=torch.long)
            )
        else:
            self.dense_indices = None

        self.input_proj = nn.Sequential(
            nn.Linear(dense_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.res_blocks = nn.ModuleList(
            [
                ResidualDenseBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Linear(hidden_dim, num_trackers)
        if init_log_odds is not None:
            with torch.no_grad():
                self.head.bias.copy_(torch.tensor(init_log_odds, dtype=torch.float32))

        # Direct residual skip paths for unstandardized direct tracker in/out links
        if direct_out_indices is not None and len(direct_out_indices) > 0:
            self.register_buffer(
                "out_indices", torch.tensor(direct_out_indices, dtype=torch.long)
            )
            self.direct_out_proj = nn.Linear(len(direct_out_indices), num_trackers, bias=False)
            with torch.no_grad():
                w_out = torch.zeros(num_trackers, len(direct_out_indices))
                if len(direct_out_indices) >= 2 * num_trackers:
                    w_out[:, :num_trackers] = torch.eye(num_trackers) * 1.5
                    w_out[:, num_trackers : 2 * num_trackers] = torch.eye(num_trackers) * 0.5
                else:
                    w_out[:, :num_trackers] = torch.eye(num_trackers) * 1.5
                self.direct_out_proj.weight.copy_(
                    w_out + torch.clamp(torch.randn_like(w_out) * 0.01, min=0.0)
                )
        else:
            self.out_indices = None
            self.direct_out_proj = None

        if direct_in_indices is not None and len(direct_in_indices) > 0:
            self.register_buffer(
                "in_indices", torch.tensor(direct_in_indices, dtype=torch.long)
            )
            self.direct_in_proj = nn.Linear(len(direct_in_indices), num_trackers, bias=False)
            with torch.no_grad():
                w_in = torch.zeros(num_trackers, len(direct_in_indices))
                if len(direct_in_indices) >= 2 * num_trackers:
                    w_in[:, :num_trackers] = torch.eye(num_trackers) * 1.0
                    w_in[:, num_trackers : 2 * num_trackers] = torch.eye(num_trackers) * 0.3
                else:
                    w_in[:, :num_trackers] = torch.eye(num_trackers) * 1.0
                self.direct_in_proj.weight.copy_(
                    w_in + torch.clamp(torch.randn_like(w_in) * 0.01, min=0.0)
                )
        else:
            self.in_indices = None
            self.direct_in_proj = None

        # Dedicated linear projections for directional neighbor affinities
        if neighbor_out_indices is None and neighbor_indices is not None:
            neighbor_out_indices = neighbor_indices

        if neighbor_out_indices is not None and len(neighbor_out_indices) > 0:
            self.register_buffer(
                "neighbor_out_indices",
                torch.tensor(neighbor_out_indices, dtype=torch.long),
            )
            self.neighbor_out_proj = nn.Linear(
                len(neighbor_out_indices), num_trackers, bias=False
            )
            with torch.no_grad():
                self.neighbor_out_proj.weight.copy_(
                    torch.eye(num_trackers) * 1.0
                    + torch.clamp(torch.randn(num_trackers, num_trackers) * 0.01, min=0.0)
                )
        else:
            self.neighbor_out_indices = None
            self.neighbor_out_proj = None

        if neighbor_in_indices is not None and len(neighbor_in_indices) > 0:
            self.register_buffer(
                "neighbor_in_indices",
                torch.tensor(neighbor_in_indices, dtype=torch.long),
            )
            self.neighbor_in_proj = nn.Linear(
                len(neighbor_in_indices), num_trackers, bias=False
            )
            with torch.no_grad():
                self.neighbor_in_proj.weight.copy_(
                    torch.eye(num_trackers) * 1.0
                    + torch.clamp(torch.randn(num_trackers, num_trackers) * 0.01, min=0.0)
                )
        else:
            self.neighbor_in_indices = None
            self.neighbor_in_proj = None

        # Empirical Tracker Co-occurrence Diffusion Operator
        if cooccur_matrix is not None:
            self.register_buffer(
                "cooccur_matrix", torch.tensor(cooccur_matrix, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "cooccur_matrix",
                torch.zeros((num_trackers, num_trackers), dtype=torch.float32),
            )

        # Thresholded NPMI Co-occurrence Diffusion Operator
        if npmi_matrix is not None:
            self.register_buffer(
                "npmi_matrix", torch.tensor(npmi_matrix, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "npmi_matrix",
                torch.zeros((num_trackers, num_trackers), dtype=torch.float32),
            )

        # Company-Level Co-ownership Diffusion Operator M_company
        if company_matrix is not None:
            self.register_buffer(
                "company_matrix", torch.tensor(company_matrix, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "company_matrix",
                torch.zeros((num_trackers, num_trackers), dtype=torch.float32),
            )

        self.diffusion_ln = nn.LayerNorm(num_trackers)
        self.npmi_ln = nn.LayerNorm(num_trackers)
        self.company_ln = nn.LayerNorm(num_trackers)

        # Context-conditioned dynamic gating linear layers sigmoid(W_gate h + b_gate)
        self.gate_cooccur = nn.Linear(hidden_dim, num_trackers)
        self.gate_npmi = nn.Linear(hidden_dim, num_trackers)
        self.gate_company = nn.Linear(hidden_dim, num_trackers)

        for gate_layer in [self.gate_cooccur, self.gate_npmi, self.gate_company]:
            nn.init.normal_(gate_layer.weight, mean=0.0, std=0.01)
            nn.init.constant_(gate_layer.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dense = x[:, self.dense_indices] if self.dense_indices is not None else x
        h = self.input_proj(x_dense)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)

        # Unbottlenecked residual skip paths for direct tracker links & directional affinities
        if self.direct_out_proj is not None and self.out_indices is not None:
            logits = logits + self.direct_out_proj(x[:, self.out_indices])

        if self.direct_in_proj is not None and self.in_indices is not None:
            logits = logits + self.direct_in_proj(x[:, self.in_indices])

        if self.neighbor_out_proj is not None and self.neighbor_out_indices is not None:
            logits = logits + self.neighbor_out_proj(x[:, self.neighbor_out_indices])

        if self.neighbor_in_proj is not None and self.neighbor_in_indices is not None:
            logits = logits + self.neighbor_in_proj(x[:, self.neighbor_in_indices])

        # Dynamic context-conditioned corporate & co-occurrence graph diffusion
        probs = torch.sigmoid(logits)
        diff_cooccur = self.diffusion_ln(torch.matmul(probs, self.cooccur_matrix))
        diff_npmi = self.npmi_ln(torch.matmul(probs, self.npmi_matrix))
        diff_comp = self.company_ln(torch.matmul(probs, self.company_matrix))

        gate_c = torch.sigmoid(self.gate_cooccur(h))
        gate_n = torch.sigmoid(self.gate_npmi(h))
        gate_co = torch.sigmoid(self.gate_company(h))

        diffused = gate_c * diff_cooccur + gate_n * diff_npmi + gate_co * diff_comp
        return logits + diffused


class AsymmetricLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 2.5,
        gamma_pos: float = 0.0,
        clip_margin: float = 0.02,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_loss = (
            -targets
            * ((1.0 - probs) ** self.gamma_pos)
            * torch.log(probs.clamp(min=self.eps))
        )
        neg_probs = (probs - self.clip_margin).clamp(min=0.0)
        neg_loss = (
            -(1.0 - targets)
            * (neg_probs**self.gamma_neg)
            * torch.log((1.0 - neg_probs).clamp(min=self.eps))
        )
        return (pos_loss + neg_loss).sum(dim=-1).mean()


class TrackerRankingLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 2.5,
        gamma_pos: float = 0.0,
        clip_margin: float = 0.02,
        rank_weight: float = 0.5,
        boundary_start: int = 7,
        boundary_end: int = 11,
        margin: float = 1.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__()
        self.asym_loss = AsymmetricLoss(
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip_margin=clip_margin,
            eps=eps,
        )
        self.rank_weight = rank_weight
        self.boundary_start = boundary_start
        self.boundary_end = boundary_end
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_asym = self.asym_loss(logits, targets)

        # Rank-10 Boundary-Targeted Margin Ranking Loss
        has_pos = targets.sum(dim=-1) > 0
        has_neg = (1.0 - targets).sum(dim=-1) > 0
        valid_mask = has_pos & has_neg

        if not valid_mask.any():
            return l_asym

        # Extract negative logits strictly
        neg_logits = logits.masked_fill(targets > 0.5, -1e9)
        k_max = min(self.boundary_end, neg_logits.size(-1))
        topk_negs, _ = torch.topk(neg_logits, k=k_max, dim=-1)

        # Negative logits specifically selected from the rank-10 boundary window (ranks 8 through 11)
        b_start = min(self.boundary_start, k_max - 1)
        boundary_negs = topk_negs[:, b_start:k_max]

        # Pairwise difference: softplus(neg_i - pos_j + margin)
        diff = boundary_negs.unsqueeze(2) - logits.unsqueeze(1) + self.margin
        pair_loss = F.softplus(diff)

        # Filter strictly for positive targets
        pos_mask = (targets > 0.5).unsqueeze(1)
        pair_loss = pair_loss * pos_mask

        # Normalize per sample by number of positive pairs (num_pos * window_size)
        num_pos = targets.sum(dim=-1)
        w_size = boundary_negs.size(1)
        sample_loss = pair_loss.sum(dim=(1, 2)) / (num_pos * w_size + 1e-6)

        l_rank = torch.where(valid_mask, sample_loss, torch.zeros_like(sample_loss)).mean()

        return l_asym + self.rank_weight * l_rank


# -----------------------------------------------------------------------------
# 9. Training Loop & Validation Recall@10 Evaluation
# -----------------------------------------------------------------------------
batch_size = 512
train_dataset = TensorDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()
)
test_dataset = TensorDataset(torch.from_numpy(X_test).float())

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size * 2, shuffle=False, drop_last=False
)

model = TrackerRankingResNet(
    in_features=X_train.shape[1],
    num_trackers=num_trackers,
    hidden_dim=384,
    num_blocks=3,
    dropout_rate=0.2,
    init_log_odds=prior_log_odds,
    cooccur_matrix=cooccur_cond_prob,
    npmi_matrix=npmi_norm,
    company_matrix=company_norm,
    direct_out_indices=direct_out_indices,
    direct_in_indices=direct_in_indices,
    neighbor_out_indices=neighbor_out_indices,
    neighbor_in_indices=neighbor_in_indices,
    dense_indices=dense_indices,
).to(device)

criterion = TrackerRankingLoss(
    gamma_neg=2.5,
    gamma_pos=0.0,
    clip_margin=0.02,
    rank_weight=0.5,
    boundary_start=7,
    boundary_end=11,
    margin=1.0,
)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
num_epochs = 32
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)


def evaluate_model(eval_model, data_loader, domain_ids, ground_truth_dict):
    eval_model.eval()
    all_top10 = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            logits = eval_model(batch_x)
            top10_batch = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
            all_top10.append(top10_batch)

    all_top10 = np.vstack(all_top10)
    recalls = []
    for i, d_id in enumerate(domain_ids):
        true_trackers = set(ground_truth_dict.get(str(d_id), []))
        if len(true_trackers) > 0:
            pred_trackers = set(all_top10[i])
            recall = len(true_trackers & pred_trackers) / len(true_trackers)
            recalls.append(recall)
        else:
            recalls.append(0.0)

    return float(np.mean(recalls))


best_val_score = 0.0
best_model_path = "working/best_tracker_resnet.pt"
patience = 10
patience_counter = 0
top_checkpoints = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_train_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_train_loss / max(1, num_batches)
    val_recall = evaluate_model(model, val_loader, val_domain_ids, val_ground_truth)

    state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if len(top_checkpoints) < 3:
        top_checkpoints.append((val_recall, epoch + 1, state_copy))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
    elif val_recall > top_checkpoints[-1][0]:
        top_checkpoints[-1] = (val_recall, epoch + 1, state_copy)
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

    if val_recall > best_val_score:
        best_val_score = val_recall
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
        status = "BEST"
    else:
        patience_counter += 1
        status = f"patience {patience_counter}/{patience}"

    print(
        f"Epoch {epoch + 1:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Recall@10: {val_recall:.5f} | Best: {best_val_score:.5f} ({status})"
    )

    if patience_counter >= patience:
        break

# -----------------------------------------------------------------------------
# 10. Top-3 Checkpoint Weight Averaging & Test Inference
# -----------------------------------------------------------------------------
if len(top_checkpoints) >= 3:
    print(
        f"Averaging Top-3 Checkpoints: Epoch {top_checkpoints[0][1]} (Recall@10: {top_checkpoints[0][0]:.5f}), "
        f"Epoch {top_checkpoints[1][1]} (Recall@10: {top_checkpoints[1][0]:.5f}), "
        f"and Epoch {top_checkpoints[2][1]} (Recall@10: {top_checkpoints[2][0]:.5f})"
    )
    avg_state_dict = {}
    for key in top_checkpoints[0][2].keys():
        if torch.is_floating_point(top_checkpoints[0][2][key]):
            avg_state_dict[key] = (
                top_checkpoints[0][2][key].to(device).float()
                + top_checkpoints[1][2][key].to(device).float()
                + top_checkpoints[2][2][key].to(device).float()
            ) / 3.0
        else:
            avg_state_dict[key] = top_checkpoints[0][2][key].to(device)
    model.load_state_dict(avg_state_dict)
elif len(top_checkpoints) == 2:
    print(
        f"Averaging Top-2 Checkpoints: Epoch {top_checkpoints[0][1]} (Recall@10: {top_checkpoints[0][0]:.5f}) "
        f"and Epoch {top_checkpoints[1][1]} (Recall@10: {top_checkpoints[1][0]:.5f})"
    )
    avg_state_dict = {}
    for key in top_checkpoints[0][2].keys():
        if torch.is_floating_point(top_checkpoints[0][2][key]):
            avg_state_dict[key] = (
                top_checkpoints[0][2][key].to(device).float()
                + top_checkpoints[1][2][key].to(device).float()
            ) / 2.0
        else:
            avg_state_dict[key] = top_checkpoints[0][2][key].to(device)
    model.load_state_dict(avg_state_dict)
elif os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))

final_val_score = evaluate_model(model, val_loader, val_domain_ids, val_ground_truth)

model.eval()
test_top10_list = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        top10_batch = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
        test_top10_list.append(top10_batch)

test_top10_preds = np.vstack(test_top10_list)

tracker_to_domain_map = np.zeros(num_trackers, dtype=np.int64)
for t_id, d_id in tracker_id_to_tracking_domain_id.items():
    tracker_to_domain_map[t_id] = d_id

predicted_tracking_domains = tracker_to_domain_map[test_top10_preds]

repeated_domain_ids = np.repeat(target_domain_ids, 10)
flattened_tracking_domains = predicted_tracking_domains.flatten()

submission_df = pd.DataFrame(
    {
        "domain_id": repeated_domain_ids,
        "tracking_domain_id": flattened_tracking_domains,
    }
)

submission_df.to_csv("submission/submission.csv", sep="\t", index=False)
submission_df.to_csv("submission/submission.tsv", sep="\t", index=False)

print(f"Final Validation Score: {final_val_score:.5f}")
