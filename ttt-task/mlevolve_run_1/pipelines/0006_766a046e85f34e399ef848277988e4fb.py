import gc
import json
import math
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from collections import Counter
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ---------------------------------------------------------------------------
# 0. Setup and Directory Paths
# ---------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1. Load Trackers Metadata and Build Mappings
# ---------------------------------------------------------------------------
trackers_df = pd.read_csv(os.path.join(INPUT_DIR, "trackers.tsv"), sep="\t")
tracker_id_to_domain = {}
tracking_domain_to_id = {}
for _, row in trackers_df.iterrows():
    tid = int(row["tracker_id"])
    tdid = int(row["tracking_domain_id"])
    tracker_id_to_domain[tid] = tdid
    tracking_domain_to_id[tdid] = tid

num_trackers = len(tracker_id_to_domain)
tracker_id_to_domain_arr = np.zeros(num_trackers, dtype=np.int64)
for tid, tdid in tracker_id_to_domain.items():
    tracker_id_to_domain_arr[tid] = tdid

tracker_domain_ids_set = set(tracking_domain_to_id.keys())
tracker_domain_ids_arr = np.array(list(tracker_domain_ids_set), dtype=np.int64)

# ---------------------------------------------------------------------------
# 2. Load Targets (Test Domain IDs)
# ---------------------------------------------------------------------------
target_df = pd.read_csv(os.path.join(INPUT_DIR, "target.tsv"), sep="\t")
test_domain_ids = target_df["domain_id"].to_numpy(dtype=np.int64)
test_domain_set = set(test_domain_ids)
test_id_to_idx = {d: i for i, d in enumerate(test_domain_ids)}

# ---------------------------------------------------------------------------
# 3. Load Training Graph and Build Train/Val Splits
# ---------------------------------------------------------------------------
train_graph_table = pq.read_table(
    os.path.join(INPUT_DIR, "tracking_graph_train.parquet"),
    columns=["domain_id", "tracker_id"],
)
train_domain_col = train_graph_table["domain_id"].to_numpy()
train_tracker_col = train_graph_table["tracker_id"].to_numpy()

unique_train_domains = np.unique(train_domain_col)
unique_train_domains = np.array(
    [d for d in unique_train_domains if d not in test_domain_set], dtype=np.int64
)

# Stratified random split: hold out 30,000 domains for validation
np.random.shuffle(unique_train_domains)
val_size = min(30000, int(len(unique_train_domains) * 0.15))
val_domains_selected = unique_train_domains[:val_size]
train_candidates = unique_train_domains[val_size:]

# Select up to 350,000 domains for training
max_train_domains = 350000
if len(train_candidates) > max_train_domains:
    train_domains_selected = train_candidates[:max_train_domains]
else:
    train_domains_selected = train_candidates

train_domain_set = set(train_domains_selected)
val_domain_set = set(val_domains_selected)
all_active_domain_ids = set.union(train_domain_set, val_domain_set, test_domain_set)
active_domain_arr = np.sort(np.array(list(all_active_domain_ids), dtype=np.int64))


def is_in_sorted(arr: np.ndarray, sorted_arr: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_arr, arr)
    idx = np.clip(idx, 0, len(sorted_arr) - 1)
    return sorted_arr[idx] == arr

# ---------------------------------------------------------------------------
# 4. Build Sparse Target Matrices (Multi-hot Tracker Sets)
# ---------------------------------------------------------------------------
train_id_to_idx = {d: i for i, d in enumerate(train_domains_selected)}
val_id_to_idx = {d: i for i, d in enumerate(val_domains_selected)}

train_rows, train_cols = [], []
val_rows, val_cols = [], []
tracker_counts = np.zeros(num_trackers, dtype=np.float64)

for d, t in zip(train_domain_col, train_tracker_col):
    if d in train_id_to_idx:
        train_rows.append(train_id_to_idx[d])
        train_cols.append(t)
        tracker_counts[t] += 1
    elif d in val_id_to_idx:
        val_rows.append(val_id_to_idx[d])
        val_cols.append(t)

del train_graph_table, train_domain_col, train_tracker_col
gc.collect()

y_train_sparse = sparse.csr_matrix(
    (
        np.ones(len(train_rows), dtype=np.uint8),
        (train_rows, train_cols),
    ),
    shape=(len(train_domains_selected), num_trackers),
)

y_val_sparse = sparse.csr_matrix(
    (
        np.ones(len(val_rows), dtype=np.uint8),
        (val_rows, val_cols),
    ),
    shape=(len(val_domains_selected), num_trackers),
)

empirical_priors = (tracker_counts + 1.0) / (len(train_domains_selected) + 2.0)
tracker_priors = np.clip(empirical_priors, 1e-6, 1.0 - 1e-6)
prior_biases = np.log(tracker_priors / (1.0 - tracker_priors)).astype(np.float32)

# ---------------------------------------------------------------------------
# 5. Extract Domain Hostnames from domains.parquet
# ---------------------------------------------------------------------------
domain_names_map = {}
domains_file = pq.ParquetFile(os.path.join(INPUT_DIR, "domains.parquet"))

for batch in domains_file.iter_batches(
    batch_size=2000000, columns=["domain_id", "domain"]
):
    b_ids = batch["domain_id"].to_numpy()
    mask = is_in_sorted(b_ids, active_domain_arr)
    if np.any(mask):
        matched_ids = b_ids[mask]
        matched_indices = np.flatnonzero(mask)
        matched_names = batch["domain"].take(matched_indices).to_pylist()
        for did, dname in zip(matched_ids, matched_names):
            domain_names_map[int(did)] = str(dname)

# ---------------------------------------------------------------------------
# 6. Stream Link-Graph: Degree Centrality & Direct Tracker Links
# ---------------------------------------------------------------------------
link_graph_file = pq.ParquetFile(os.path.join(INPUT_DIR, "link-graph.parquet"))

in_degree_dict = {d: 0 for d in all_active_domain_ids}
out_degree_dict = {d: 0 for d in all_active_domain_ids}
direct_tracker_link_counts = {d: 0 for d in all_active_domain_ids}
tracker_domain_ids_arr_sorted = np.sort(tracker_domain_ids_arr)

direct_links_train = np.zeros(
    (len(train_domains_selected), num_trackers), dtype=np.float32
)
direct_links_val = np.zeros(
    (len(val_domains_selected), num_trackers), dtype=np.float32
)
direct_links_test = np.zeros(
    (len(test_domain_ids), num_trackers), dtype=np.float32
)

for batch in link_graph_file.iter_batches(
    batch_size=2000000, columns=["source_domain_id", "target_domain_id"]
):
    src_arr = batch["source_domain_id"].to_numpy()
    tgt_arr = batch["target_domain_id"].to_numpy()

    # Out-degree accumulation
    mask_src = is_in_sorted(src_arr, active_domain_arr)
    if np.any(mask_src):
        vals, cnts = np.unique(src_arr[mask_src], return_counts=True)
        for v, c in zip(vals, cnts):
            out_degree_dict[v] += int(c)

    # In-degree accumulation
    mask_tgt = is_in_sorted(tgt_arr, active_domain_arr)
    if np.any(mask_tgt):
        vals, cnts = np.unique(tgt_arr[mask_tgt], return_counts=True)
        for v, c in zip(vals, cnts):
            in_degree_dict[v] += int(c)

    # Direct tracker link presence
    is_tracker_tgt = is_in_sorted(tgt_arr, tracker_domain_ids_arr_sorted)
    if np.any(is_tracker_tgt):
        tracker_sources = src_arr[is_tracker_tgt]
        tracker_targets = tgt_arr[is_tracker_tgt]
        mask_trk_src = is_in_sorted(tracker_sources, active_domain_arr)
        if np.any(mask_trk_src):
            vals, cnts = np.unique(tracker_sources[mask_trk_src], return_counts=True)
            for v, c in zip(vals, cnts):
                direct_tracker_link_counts[v] += int(c)

            matched_sources = tracker_sources[mask_trk_src]
            matched_targets = tracker_targets[mask_trk_src]
            for s, t in zip(matched_sources, matched_targets):
                tid = tracking_domain_to_id.get(int(t))
                if tid is not None:
                    sid = int(s)
                    if sid in train_id_to_idx:
                        direct_links_train[train_id_to_idx[sid], tid] = 1.0
                    elif sid in val_id_to_idx:
                        direct_links_val[val_id_to_idx[sid], tid] = 1.0
                    elif sid in test_id_to_idx:
                        direct_links_test[test_id_to_idx[sid], tid] = 1.0

gc.collect()

# ---------------------------------------------------------------------------
# 7. URL Content Classification Features
# ---------------------------------------------------------------------------
url_df = pd.read_csv(
    os.path.join(INPUT_DIR, "url-classification.csv"),
    usecols=["url", "category"],
)
unique_categories = sorted(url_df["category"].dropna().unique().tolist())
cat_to_idx = {c: i for i, c in enumerate(unique_categories)}
num_categories = len(unique_categories)

domain_cat_counts = {}
url_pattern = re.compile(r"^(?:https?://)?(?:www\.)?([^/:\s]+)")

for url, cat in zip(url_df["url"].astype(str), url_df["category"]):
    match = url_pattern.match(url)
    if match:
        hostname = match.group(1).lower()
        if cat in cat_to_idx:
            c_idx = cat_to_idx[cat]
            if hostname not in domain_cat_counts:
                domain_cat_counts[hostname] = np.zeros(num_categories, dtype=np.float32)
            domain_cat_counts[hostname][c_idx] += 1.0

del url_df
gc.collect()

# ---------------------------------------------------------------------------
# 8. Freedom of the Press Metadata Features
# ---------------------------------------------------------------------------
fop_file = os.path.join(INPUT_DIR, "freedom-of-the-press.csv")
try:
    fop_df = pd.read_csv(fop_file, sep="\t")
    if len(fop_df.columns) < 3:
        fop_df = pd.read_csv(fop_file)
except Exception:
    fop_df = pd.read_csv(fop_file, delim_whitespace=True)

fop_df.columns = [c.strip().lower() for c in fop_df.columns]
tld_to_fop = {}
fop_scores = []
for _, row in fop_df.iterrows():
    tld = str(row.get("tld", "")).strip().lower()
    score_val = row.get("freedom_of_the_press", np.nan)
    try:
        score = float(score_val)
        tld_to_fop[tld] = score
        fop_scores.append(score)
    except Exception:
        continue

median_fop = float(np.median(fop_scores)) if len(fop_scores) > 0 else 50.0

# ---------------------------------------------------------------------------
# 9. Lexical & Information-Theoretic Feature Engineering
# ---------------------------------------------------------------------------
KEYWORDS = [
    "shop",
    "store",
    "news",
    "blog",
    "cdn",
    "api",
    "app",
    "media",
    "tech",
    "forum",
    "portal",
    "game",
    "video",
    "tv",
    "mail",
    "edu",
    "gov",
    "org",
    "info",
    "travel",
]


def compute_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(s)
    ent = 0.0
    for cnt in counts.values():
        p = cnt / total
        ent -= p * math.log2(p)
    return ent


def extract_features_for_domain_list(domain_ids: np.ndarray) -> np.ndarray:
    num_samples = len(domain_ids)
    num_kw = len(KEYWORDS)
    total_base_feats = 12 + num_kw + 8 + num_categories + 1 + 2
    feats = np.zeros((num_samples, total_base_feats), dtype=np.float32)
    vowels = set("aeiou")

    for i, did in enumerate(domain_ids):
        raw_name = domain_names_map.get(did, "")
        name = raw_name.lower()
        parts = name.split(".")
        tld = parts[-1] if len(parts) > 1 else ""
        sld = parts[-2] if len(parts) > 1 else parts[0]

        length = len(name)
        sld_len = len(sld)
        num_dots = name.count(".")
        num_hyphens = name.count("-")
        num_digits = sum(1 for c in name if c.isdigit())
        num_vowels = sum(1 for c in name if c in vowels)
        num_consonants = sum(1 for c in name if c.isalpha() and c not in vowels)

        col = 0
        feats[i, col] = length
        col += 1
        feats[i, col] = sld_len
        col += 1
        feats[i, col] = num_dots
        col += 1
        feats[i, col] = num_hyphens
        col += 1
        feats[i, col] = num_digits
        col += 1
        feats[i, col] = num_digits / (length + 1e-5)
        col += 1
        feats[i, col] = num_vowels / (length + 1e-5)
        col += 1
        feats[i, col] = num_consonants / (length + 1e-5)
        col += 1
        feats[i, col] = compute_entropy(name)
        col += 1
        feats[i, col] = 1.0 if name.startswith("www.") else 0.0
        col += 1
        feats[i, col] = 1.0 if num_hyphens > 0 else 0.0
        col += 1
        feats[i, col] = 1.0 if num_digits > 0 else 0.0
        col += 1

        for kw in KEYWORDS:
            feats[i, col] = 1.0 if kw in name else 0.0
            col += 1

        in_d = in_degree_dict.get(did, 0)
        out_d = out_degree_dict.get(did, 0)
        tot_d = in_d + out_d
        dir_t_cnt = direct_tracker_link_counts.get(did, 0)

        feats[i, col] = math.log1p(in_d)
        col += 1
        feats[i, col] = math.log1p(out_d)
        col += 1
        feats[i, col] = math.log1p(tot_d)
        col += 1
        feats[i, col] = math.log1p(in_d) / (math.log1p(out_d) + 1.0)
        col += 1
        feats[i, col] = 1.0 if in_d == 0 else 0.0
        col += 1
        feats[i, col] = 1.0 if out_d == 0 else 0.0
        col += 1
        feats[i, col] = 1.0 if tot_d == 0 else 0.0
        col += 1
        feats[i, col] = math.log1p(dir_t_cnt)
        col += 1

        clean_host = name[4:] if name.startswith("www.") else name
        cat_dist = domain_cat_counts.get(clean_host, None)
        if cat_dist is not None and cat_dist.sum() > 0:
            feats[i, col : col + num_categories] = cat_dist / cat_dist.sum()
            feats[i, col + num_categories] = 1.0
        else:
            feats[i, col + num_categories] = 0.0
        col += num_categories + 1

        if tld in tld_to_fop:
            feats[i, col] = tld_to_fop[tld]
            feats[i, col + 1] = 1.0
        else:
            feats[i, col] = median_fop
            feats[i, col + 1] = 0.0

    return feats


base_train_feats = extract_features_for_domain_list(train_domains_selected)
base_val_feats = extract_features_for_domain_list(val_domains_selected)
base_test_feats = extract_features_for_domain_list(test_domain_ids)

# ---------------------------------------------------------------------------
# 10. Subword Character n-gram TF-IDF & Truncated SVD Manifold
# ---------------------------------------------------------------------------
train_domain_strings = [domain_names_map.get(d, "") for d in train_domains_selected]
val_domain_strings = [domain_names_map.get(d, "") for d in val_domains_selected]
test_domain_strings = [domain_names_map.get(d, "") for d in test_domain_ids]

tfidf_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    max_features=10000,
    min_df=5,
    sublinear_tf=True,
)

tfidf_train = tfidf_vectorizer.fit_transform(train_domain_strings)
tfidf_val = tfidf_vectorizer.transform(val_domain_strings)
tfidf_test = tfidf_vectorizer.transform(test_domain_strings)

svd_dim = 32
svd = TruncatedSVD(n_components=svd_dim, random_state=42)
svd_train = svd.fit_transform(tfidf_train).astype(np.float32)
svd_val = svd.transform(tfidf_val).astype(np.float32)
svd_test = svd.transform(tfidf_test).astype(np.float32)

del (
    tfidf_vectorizer,
    tfidf_train,
    tfidf_val,
    tfidf_test,
    train_domain_strings,
    val_domain_strings,
    test_domain_strings,
)
gc.collect()

# ---------------------------------------------------------------------------
# 11. TLD Features, Empirical Tracker Priors & Feature Concatenation
# ---------------------------------------------------------------------------
def extract_tld(name: str) -> str:
    if not name:
        return ""
    parts = name.strip().lower().split(".")
    return parts[-1] if len(parts) > 1 else ""


train_tlds = [
    extract_tld(domain_names_map.get(d, "")) for d in train_domains_selected
]
val_tlds = [
    extract_tld(domain_names_map.get(d, "")) for d in val_domains_selected
]
test_tlds = [
    extract_tld(domain_names_map.get(d, "")) for d in test_domain_ids
]

tld_counter = Counter(train_tlds)
top_50_tlds = [tld for tld, _ in tld_counter.most_common(50) if tld]
top_50_tld_map = {tld: i for i, tld in enumerate(top_50_tlds)}
num_top_tlds = len(top_50_tld_map)


def encode_tld_one_hot(tlds: list) -> np.ndarray:
    one_hot = np.zeros((len(tlds), num_top_tlds), dtype=np.float32)
    for i, t in enumerate(tlds):
        if t in top_50_tld_map:
            one_hot[i, top_50_tld_map[t]] = 1.0
    return one_hot


tld_one_hot_train = encode_tld_one_hot(train_tlds)
tld_one_hot_val = encode_tld_one_hot(val_tlds)
tld_one_hot_test = encode_tld_one_hot(test_tlds)

# Out-of-fold TLD-conditional empirical tracker prior distributions
n_train = len(train_domains_selected)
tld_priors_train = np.zeros((n_train, num_trackers), dtype=np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_fold_idx, oof_fold_idx in kf.split(np.arange(n_train)):
    fold_y = y_train_sparse[train_fold_idx]
    fold_tlds = [train_tlds[i] for i in train_fold_idx]
    fold_n = len(train_fold_idx)
    fold_global_prior = (np.array(fold_y.sum(axis=0)).ravel() + 1.0) / (
        fold_n + 2.0
    )

    unique_fold_tlds = list(set(fold_tlds))
    fold_tld_map = {t: i for i, t in enumerate(unique_fold_tlds)}
    row_idx = [fold_tld_map[t] for t in fold_tlds]
    col_idx = np.arange(fold_n)
    m_mat = sparse.csr_matrix(
        (np.ones(fold_n, dtype=np.float32), (row_idx, col_idx)),
        shape=(len(unique_fold_tlds), fold_n),
    )
    fold_tld_sums = (m_mat @ fold_y).toarray()
    fold_tld_counts = np.array(m_mat.sum(axis=1)).ravel()

    fold_smoothed = (fold_tld_sums + 20.0 * fold_global_prior[None, :]) / (
        fold_tld_counts[:, None] + 20.0
    )

    oof_tlds = [train_tlds[i] for i in oof_fold_idx]
    oof_rows = np.array([fold_tld_map.get(t, -1) for t in oof_tlds])
    valid_mask = oof_rows >= 0
    tld_priors_train[oof_fold_idx[valid_mask]] = fold_smoothed[
        oof_rows[valid_mask]
    ]
    tld_priors_train[oof_fold_idx[~valid_mask]] = fold_global_prior

# Full training set smoothed priors for Val and Test
unique_train_tlds = list(set(train_tlds))
train_tld_map = {t: i for i, t in enumerate(unique_train_tlds)}
row_idx = [train_tld_map[t] for t in train_tlds]
col_idx = np.arange(n_train)
m_mat_full = sparse.csr_matrix(
    (np.ones(n_train, dtype=np.float32), (row_idx, col_idx)),
    shape=(len(unique_train_tlds), n_train),
)
full_tld_sums = (m_mat_full @ y_train_sparse).toarray()
full_tld_counts = np.array(m_mat_full.sum(axis=1)).ravel()
full_global_prior = (np.array(y_train_sparse.sum(axis=0)).ravel() + 1.0) / (
    n_train + 2.0
)

full_smoothed = (full_tld_sums + 20.0 * full_global_prior[None, :]) / (
    full_tld_counts[:, None] + 20.0
)


def assign_full_tld_priors(target_tlds: list) -> np.ndarray:
    n = len(target_tlds)
    row_indices = np.array([train_tld_map.get(t, -1) for t in target_tlds])
    out = np.zeros((n, num_trackers), dtype=np.float32)
    valid = row_indices >= 0
    out[valid] = full_smoothed[row_indices[valid]]
    out[~valid] = full_global_prior
    return out


tld_priors_val = assign_full_tld_priors(val_tlds)
tld_priors_test = assign_full_tld_priors(test_tlds)

# Concatenate dense features and apply leak-free standard scaling
X_train_dense_raw = np.hstack(
    [base_train_feats, svd_train, tld_one_hot_train, tld_priors_train]
)
X_val_dense_raw = np.hstack(
    [base_val_feats, svd_val, tld_one_hot_val, tld_priors_val]
)
X_test_dense_raw = np.hstack(
    [base_test_feats, svd_test, tld_one_hot_test, tld_priors_test]
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_dense_raw).astype(np.float32)
X_val_scaled = scaler.transform(X_val_dense_raw).astype(np.float32)
X_test_scaled = scaler.transform(X_test_dense_raw).astype(np.float32)

np.nan_to_num(X_train_scaled, copy=False)
np.nan_to_num(X_val_scaled, copy=False)
np.nan_to_num(X_test_scaled, copy=False)

# Append direct tracker link indicator slices at known indices [-num_trackers:]
X_train = np.hstack([X_train_scaled, direct_links_train]).astype(np.float32)
X_val = np.hstack([X_val_scaled, direct_links_val]).astype(np.float32)
X_test = np.hstack([X_test_scaled, direct_links_test]).astype(np.float32)

del (
    X_train_dense_raw,
    X_val_dense_raw,
    X_test_dense_raw,
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
)
gc.collect()

in_features = X_train.shape[1]


# ---------------------------------------------------------------------------
# 12. Model Architecture: TrackerInteractionRankingNet
# ---------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.15):
        super().__init__()
        self.fc_gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        gated = self.fc_gate(x)
        val, gate = gated.chunk(2, dim=-1)
        x = val * torch.sigmoid(gate)
        x = self.dropout(self.fc_proj(x))
        return self.norm(residual + x)


class TrackerInteractionRankingNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_trackers: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        num_blocks: int = 3,
        dropout_rate: float = 0.2,
        initial_priors: np.ndarray = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim

        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate * 0.5),
        )

        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_blocks)
            ]
        )

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )

        self.tracker_embeddings = nn.Parameter(
            torch.randn(num_trackers, embed_dim) / math.sqrt(embed_dim)
        )

        self.tracker_correlation = nn.Parameter(torch.zeros(num_trackers, num_trackers))

        self.prior_bias = nn.Parameter(torch.zeros(num_trackers))
        if initial_priors is not None:
            with torch.no_grad():
                self.prior_bias.copy_(torch.from_numpy(initial_priors))

        self.direct_link_proj = nn.Linear(num_trackers, num_trackers, bias=False)

        self.corr_scale = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        query = self.query_proj(h)

        base_logits = torch.matmul(query, self.tracker_embeddings.t())
        corr_logits = torch.matmul(torch.sigmoid(base_logits), self.tracker_correlation)

        direct_link_slice = x[:, -self.num_trackers:]
        direct_logits = self.direct_link_proj(direct_link_slice)

        logits = (
            base_logits
            + self.corr_scale * corr_logits
            + self.prior_bias.unsqueeze(0)
            + direct_logits
        )
        return logits


# ---------------------------------------------------------------------------
# 13. Hybrid Ranking Loss Function
# ---------------------------------------------------------------------------
class AsymmetricFocalLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = (probs - self.clip).clamp(min=0.0)

        pos_loss = (
            targets * torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        )
        neg_loss = (
            (1.0 - targets)
            * torch.pow(probs_neg, self.gamma_neg)
            * torch.log((1.0 - probs_neg).clamp(min=self.eps))
        )

        loss = -1.0 * (pos_loss + neg_loss)
        return loss.sum(dim=-1).mean()


class TopKMarginRankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0, top_k: int = 10):
        super().__init__()
        self.margin = margin
        self.top_k = top_k

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets > 0.5
        num_pos = pos_mask.sum(dim=-1)
        num_neg = (~pos_mask).sum(dim=-1)

        valid_domain_mask = (num_pos > 0) & (num_neg > 0)
        if not valid_domain_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        neg_logits = torch.where(
            pos_mask, torch.tensor(-1e9, device=logits.device), logits
        )
        top_k_neg, _ = torch.topk(neg_logits, k=self.top_k, dim=-1)

        diffs = top_k_neg.unsqueeze(-1) - logits.unsqueeze(1) + self.margin
        diffs_relu = F.relu(diffs)

        pos_mask_expanded = pos_mask.unsqueeze(1)
        diffs_pos = diffs_relu * pos_mask_expanded.float()

        loss_per_sample = diffs_pos.sum(dim=(1, 2)) / (
            num_pos.float() * float(self.top_k) + 1e-6
        )
        return loss_per_sample[valid_domain_mask].mean()


class HybridTrackerRankingLoss(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        margin: float = 1.0,
        top_k: int = 10,
        ranking_weight: float = 0.5,
    ):
        super().__init__()
        self.asl_loss = AsymmetricFocalLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        self.rank_loss = TopKMarginRankingLoss(margin=margin, top_k=top_k)
        self.ranking_weight = ranking_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_asl = self.asl_loss(logits, targets)
        loss_rank = self.rank_loss(logits, targets)
        return loss_asl + self.ranking_weight * loss_rank


# ---------------------------------------------------------------------------
# 14. Optimizer Setup
# ---------------------------------------------------------------------------
def build_optimizer(
    model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 1e-4
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith("bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


# Empirical Tracker Co-occurrence and PPMI Matrix
train_cooccurrence = (
    (y_train_sparse.T @ y_train_sparse).toarray().astype(np.float64)
)
tracker_freqs = np.diag(train_cooccurrence).copy()
tracker_freqs = np.maximum(tracker_freqs, 1.0)
n_train_total = float(len(train_domains_selected))

p_ij = train_cooccurrence / n_train_total
p_i = tracker_freqs / n_train_total
p_expected = np.outer(p_i, p_i)

eps = 1e-8
pmi = np.log((p_ij + eps) / (p_expected + eps))
ppmi = np.maximum(0.0, pmi)
np.fill_diagonal(ppmi, 0.0)
if ppmi.max() > 0:
    ppmi = ppmi / ppmi.max()
ppmi = ppmi.astype(np.float32)

model = TrackerInteractionRankingNet(
    in_features=in_features,
    num_trackers=num_trackers,
    hidden_dim=256,
    embed_dim=128,
    num_blocks=3,
    dropout_rate=0.2,
    initial_priors=prior_biases,
).to(device)

with torch.no_grad():
    model.tracker_correlation.copy_(torch.from_numpy(ppmi).to(device))

criterion = HybridTrackerRankingLoss(
    gamma_neg=4.0,
    gamma_pos=1.0,
    margin=1.0,
    top_k=10,
    ranking_weight=0.5,
).to(device)

optimizer = build_optimizer(model=model, learning_rate=1e-3, weight_decay=1e-4)


# ---------------------------------------------------------------------------
# 15. Dataset Wrapper and DataLoaders
# ---------------------------------------------------------------------------
y_train_dense = torch.from_numpy(y_train_sparse.toarray().astype(np.float32))
train_dataset = TensorDataset(torch.from_numpy(X_train), y_train_dense)
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

val_dataset = TensorDataset(torch.from_numpy(X_val))
val_loader = DataLoader(
    val_dataset,
    batch_size=2048,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_dataset = TensorDataset(torch.from_numpy(X_test))
test_loader = DataLoader(
    test_dataset,
    batch_size=2048,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)


# ---------------------------------------------------------------------------
# 16. Official Metric Computation: Recall@10
# ---------------------------------------------------------------------------
def compute_validation_recall_at_10(
    model: torch.nn.Module, data_loader: DataLoader, y_sparse: sparse.csr_matrix
) -> float:
    model.eval()
    all_top10_preds = []

    with torch.no_grad():
        for (batch_x,) in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_logits = model(batch_x)
            top10_batch = torch.topk(batch_logits, k=10, dim=-1).indices.cpu().numpy()
            all_top10_preds.append(top10_batch)

    all_top10_preds = np.vstack(all_top10_preds)
    num_samples = y_sparse.shape[0]
    recalls = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        start_idx = y_sparse.indptr[i]
        end_idx = y_sparse.indptr[i + 1]
        n_true = end_idx - start_idx
        if n_true > 0:
            true_trackers = set(y_sparse.indices[start_idx:end_idx])
            hits = sum(1 for p in all_top10_preds[i] if p in true_trackers)
            recalls[i] = hits / n_true
        else:
            recalls[i] = 0.0

    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# 17. Training Loop with Metric Monitoring
# ---------------------------------------------------------------------------
num_epochs = 12
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
best_model_path = os.path.join(WORKING_DIR, "best_tracker_ranking_net.pt")

best_val_recall = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    scheduler.step()
    epoch_train_loss = running_loss / max(1, num_batches)
    val_recall = compute_validation_recall_at_10(model, val_loader, y_val_sparse)
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Recall@10: {val_recall:.4f}"
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(model.state_dict(), best_model_path)

# ---------------------------------------------------------------------------
# 18. Model Evaluation & Inference on Test Domains
# ---------------------------------------------------------------------------
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))

final_val_score = compute_validation_recall_at_10(model, val_loader, y_val_sparse)

model.eval()
test_top10_list = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_logits = model(batch_x)
        top10_batch = torch.topk(batch_logits, k=10, dim=-1).indices.cpu().numpy()
        test_top10_list.append(top10_batch)

test_top10_tracker_ids = np.vstack(test_top10_list)
test_top10_tracking_domains = tracker_id_to_domain_arr[test_top10_tracker_ids]

sub_domain_ids = np.repeat(test_domain_ids, 10)
sub_tracking_domain_ids = test_top10_tracking_domains.reshape(-1)

submission_df = pd.DataFrame(
    {
        "domain_id": sub_domain_ids,
        "tracking_domain_id": sub_tracking_domain_ids,
    }
)

submission_path_csv = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_path_tsv = os.path.join(SUBMISSION_DIR, "submission.tsv")

submission_df.to_csv(submission_path_csv, sep="\t", index=False)
submission_df.to_csv(submission_path_tsv, sep="\t", index=False)

print(f"Final Validation Score: {final_val_score}")
