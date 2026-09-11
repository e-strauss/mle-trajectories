import gc
import math
import os
import shutil
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Global Configuration & Seeds
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load Data & Build Canonical Mappings
# ---------------------------------------------------------------------------
target_df = pd.read_csv(os.path.join(INPUT_DIR, "target.tsv"), sep="\t")
test_domain_ids = target_df["domain_id"].to_numpy(dtype=np.int64)

trackers_df = pd.read_csv(os.path.join(INPUT_DIR, "trackers.tsv"), sep="\t")
trackers_df["tracker_id"] = trackers_df["tracker_id"].astype(np.int32)
trackers_df["tracking_domain_id"] = trackers_df["tracking_domain_id"].astype(np.int64)

NUM_TRACKERS = 355
tracker_id_to_domain_id = dict(
    zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
)
domain_id_to_tracker_id = dict(
    zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
)

train_tracking_df = pd.read_parquet(
    os.path.join(INPUT_DIR, "tracking_graph_train.parquet"),
    columns=["domain_id", "tracking_domain_id", "tracker_id"],
)
train_tracking_df["domain_id"] = train_tracking_df["domain_id"].astype(np.int64)
train_tracking_df["tracker_id"] = train_tracking_df["tracker_id"].astype(np.int32)

train_tracking_df = train_tracking_df[
    (train_tracking_df["tracker_id"] >= 0)
    & (train_tracking_df["tracker_id"] < NUM_TRACKERS)
].drop_duplicates(subset=["domain_id", "tracker_id"])

# ---------------------------------------------------------------------------
# 2. Leak-Free Train / Validation Split
# ---------------------------------------------------------------------------
all_known_domains = np.sort(train_tracking_df["domain_id"].unique())
test_domain_set = set(test_domain_ids)

candidate_train_val_domains = np.array(
    [d for d in all_known_domains if d not in test_domain_set], dtype=np.int64
)
np.random.shuffle(candidate_train_val_domains)

VAL_SIZE = 25000
TRAIN_SIZE = min(200000, len(candidate_train_val_domains) - VAL_SIZE)

val_domain_ids = np.sort(candidate_train_val_domains[:VAL_SIZE])
train_domain_ids = np.sort(
    candidate_train_val_domains[VAL_SIZE : VAL_SIZE + TRAIN_SIZE]
)

train_domain_set = set(train_domain_ids)
val_domain_set = set(val_domain_ids)

train_labels_df = train_tracking_df[
    train_tracking_df["domain_id"].isin(train_domain_set)
]
val_labels_df = train_tracking_df[train_tracking_df["domain_id"].isin(val_domain_set)]

train_domain_to_row = {did: idx for idx, did in enumerate(train_domain_ids)}
val_domain_to_row = {did: idx for idx, did in enumerate(val_domain_ids)}

Y_train = np.zeros((len(train_domain_ids), NUM_TRACKERS), dtype=np.float32)
r_idx_tr = train_labels_df["domain_id"].map(train_domain_to_row).to_numpy()
c_idx_tr = train_labels_df["tracker_id"].to_numpy()
Y_train[r_idx_tr, c_idx_tr] = 1.0

Y_val = np.zeros((len(val_domain_ids), NUM_TRACKERS), dtype=np.float32)
r_idx_val = val_labels_df["domain_id"].map(val_domain_to_row).to_numpy()
c_idx_val = val_labels_df["tracker_id"].to_numpy()
Y_val[r_idx_val, c_idx_val] = 1.0

val_ground_truth = (
    val_labels_df.groupby("domain_id")["tracker_id"]
    .apply(lambda s: set(s.tolist()))
    .to_dict()
)
for did in val_domain_ids:
    if did not in val_ground_truth:
        val_ground_truth[did] = set()

tracker_priors = (Y_train.sum(axis=0) + 1.0) / (len(train_domain_ids) + 2.0)

# Tracker co-occurrence matrix from Y_train with zeroed diagonal
C_cooc = Y_train.T.dot(Y_train)
np.fill_diagonal(C_cooc, 0.0)
row_sums = C_cooc.sum(axis=1, keepdims=True)
T_cooc = np.divide(C_cooc, np.maximum(row_sums, 1e-6), dtype=np.float32)

non_val_labels_df = train_tracking_df[
    ~train_tracking_df["domain_id"].isin(val_domain_set)
]
all_non_val_domains = np.sort(non_val_labels_df["domain_id"].unique())

del train_labels_df, val_labels_df, train_tracking_df
gc.collect()

# ---------------------------------------------------------------------------
# 3. Hyperlink Graph & Bidirectional Tracker Diffusion
# ---------------------------------------------------------------------------
link_graph_df = pd.read_parquet(
    os.path.join(INPUT_DIR, "link-graph.parquet"),
    columns=["source_domain_id", "target_domain_id"],
)
link_graph_df["source_domain_id"] = link_graph_df["source_domain_id"].astype(np.int64)
link_graph_df["target_domain_id"] = link_graph_df["target_domain_id"].astype(np.int64)

out_degree_map = link_graph_df["source_domain_id"].value_counts().to_dict()
in_degree_map = link_graph_df["target_domain_id"].value_counts().to_dict()

active_domain_ids = np.unique(
    np.concatenate(
        [
            all_non_val_domains,
            val_domain_ids,
            test_domain_ids,
            trackers_df["tracking_domain_id"].to_numpy(),
        ]
    )
)
node_to_idx = {did: idx for idx, did in enumerate(active_domain_ids)}
N_ACTIVE = len(active_domain_ids)

mask_valid_edges = link_graph_df["source_domain_id"].isin(node_to_idx) & link_graph_df[
    "target_domain_id"
].isin(node_to_idx)
filtered_edges = link_graph_df[mask_valid_edges]

src_nodes = filtered_edges["source_domain_id"].map(node_to_idx).to_numpy(dtype=np.int32)
dst_nodes = filtered_edges["target_domain_id"].map(node_to_idx).to_numpy(dtype=np.int32)
del link_graph_df, filtered_edges
gc.collect()

A_graph = sparse.csr_matrix(
    (np.ones(len(src_nodes), dtype=np.float32), (src_nodes, dst_nodes)),
    shape=(N_ACTIVE, N_ACTIVE),
)
A_T = A_graph.transpose().tocsr()

Y_global = np.zeros((N_ACTIVE, NUM_TRACKERS), dtype=np.float32)
r_idx_global = non_val_labels_df["domain_id"].map(node_to_idx).to_numpy()
c_idx_global = non_val_labels_df["tracker_id"].to_numpy()
valid_global = ~np.isnan(r_idx_global)
Y_global[r_idx_global[valid_global].astype(np.int64), c_idx_global[valid_global]] = 1.0
del non_val_labels_df
gc.collect()

direct_link_matrix = np.zeros((N_ACTIVE, NUM_TRACKERS), dtype=np.float32)
for t_id, tracking_did in tracker_id_to_domain_id.items():
    if tracking_did in node_to_idx:
        col_idx = node_to_idx[tracking_did]
        in_neighbors = A_T[col_idx].indices
        direct_link_matrix[in_neighbors, t_id] = 1.0

has_label = (Y_global.sum(axis=1, keepdims=True) > 0).astype(np.float32)

# Multi-hop 2-step normalized out-diffusion
out_diff_1 = A_graph.dot(Y_global)
out_deg_1 = A_graph.dot(has_label) + 1e-5
diff_out_1 = out_diff_1 / out_deg_1

out_diff_2 = A_graph.dot(out_diff_1)
out_deg_2 = A_graph.dot(A_graph.dot(has_label)) + 1e-5
diff_out_2 = out_diff_2 / out_deg_2

diff_neighbor_out = 0.7 * diff_out_1 + 0.3 * diff_out_2

# Multi-hop 2-step normalized in-diffusion
in_diff_1 = A_T.dot(Y_global)
in_deg_1 = A_T.dot(has_label) + 1e-5
diff_in_1 = in_diff_1 / in_deg_1

in_diff_2 = A_T.dot(in_diff_1)
in_deg_2 = A_T.dot(A_T.dot(has_label)) + 1e-5
diff_in_2 = in_diff_2 / in_deg_2

diff_neighbor_in = 0.7 * diff_in_1 + 0.3 * diff_in_2

del (
    A_graph,
    A_T,
    Y_global,
    has_label,
    out_diff_1,
    out_deg_1,
    diff_out_1,
    out_diff_2,
    out_deg_2,
    diff_out_2,
    in_diff_1,
    in_deg_1,
    diff_in_1,
    in_diff_2,
    in_deg_2,
    diff_in_2,
)
gc.collect()


def extract_graph_features(domain_arr):
    n = len(domain_arr)
    neighbor_out = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    neighbor_in = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    direct_link = np.zeros((n, NUM_TRACKERS), dtype=np.float32)
    topo_feats = np.zeros((n, 6), dtype=np.float32)

    for i, did in enumerate(domain_arr):
        o_deg = float(out_degree_map.get(did, 0))
        i_deg = float(in_degree_map.get(did, 0))
        tot_deg = o_deg + i_deg
        log_o = np.log1p(o_deg)
        log_i = np.log1p(i_deg)
        deg_ratio = (log_i + 1e-4) / (log_o + 1e-4)
        topo_feats[i] = [o_deg, i_deg, tot_deg, log_o, log_i, deg_ratio]

        if did in node_to_idx:
            idx = node_to_idx[did]
            neighbor_out[i] = diff_neighbor_out[idx]
            neighbor_in[i] = diff_neighbor_in[idx]
            direct_link[i] = direct_link_matrix[idx]

    return neighbor_out, neighbor_in, direct_link, topo_feats


tr_neighbor_out, tr_neighbor_in, tr_direct_link, tr_topo = extract_graph_features(
    train_domain_ids
)
val_neighbor_out, val_neighbor_in, val_direct_link, val_topo = extract_graph_features(
    val_domain_ids
)
ts_neighbor_out, ts_neighbor_in, ts_direct_link, ts_topo = extract_graph_features(
    test_domain_ids
)

del diff_neighbor_out, diff_neighbor_in, direct_link_matrix
gc.collect()

# ---------------------------------------------------------------------------
# 4. Domain Hostnames, Lexical, TLD, & Press Freedom Features
# ---------------------------------------------------------------------------
domains_df = pd.read_parquet(
    os.path.join(INPUT_DIR, "domains.parquet"), columns=["domain_id", "domain"]
)
domain_lookup = dict(zip(domains_df["domain_id"], domains_df["domain"]))
del domains_df
gc.collect()

try:
    freedom_df = pd.read_csv(
        os.path.join(INPUT_DIR, "freedom-of-the-press.csv"), sep="\t"
    )
    if freedom_df.shape[1] < 3:
        freedom_df = pd.read_csv(
            os.path.join(INPUT_DIR, "freedom-of-the-press.csv"),
            sep=r"\s+",
            engine="python",
        )
except Exception:
    freedom_df = pd.read_csv(
        os.path.join(INPUT_DIR, "freedom-of-the-press.csv"),
        sep=r"\s+",
        engine="python",
    )

freedom_map = dict(
    zip(
        freedom_df["tld"].str.strip().str.lower(),
        freedom_df["freedom_of_the_press"].astype(float),
    )
)

EU_TLDS = {
    "de",
    "fr",
    "nl",
    "it",
    "es",
    "pl",
    "se",
    "eu",
    "at",
    "cz",
    "dk",
    "fi",
    "be",
}
RU_TLDS = {"ru", "su", "by", "kz", "ua"}
ANGLO_TLDS = {"us", "uk", "ca", "au", "nz"}
ASIA_TLDS = {"cn", "jp", "kr", "in", "tw", "vn", "id", "th"}


def extract_domain_metadata(domain_name):
    if not isinstance(domain_name, str) or not domain_name:
        return "", "other", 0, 0, 0, 0.0, 0.0, 0.0

    d_clean = domain_name.strip().lower()
    parts = d_clean.split(".")
    tld = parts[-1] if len(parts) > 1 else "other"
    if len(parts) >= 3 and parts[-2] in {
        "co",
        "com",
        "org",
        "net",
        "edu",
        "gov",
    }:
        tld = f"{parts[-2]}.{parts[-1]}"

    root = parts[0] if parts else ""
    length = len(d_clean)
    num_dots = d_clean.count(".")
    num_hyphens = d_clean.count("-")
    num_digits = sum(c.isdigit() for c in d_clean)
    digit_ratio = num_digits / max(1, length)
    vowels = sum(c in "aeiou" for c in d_clean)
    vowel_ratio = vowels / max(1, length)

    char_counts = {}
    for c in d_clean:
        char_counts[c] = char_counts.get(c, 0) + 1
    entropy = -sum(
        (cnt / length) * np.log2(cnt / length) for cnt in char_counts.values()
    )

    return (
        root,
        tld,
        length,
        num_dots,
        num_hyphens,
        digit_ratio,
        vowel_ratio,
        entropy,
    )


train_meta = [
    extract_domain_metadata(domain_lookup.get(did, "")) for did in train_domain_ids
]
val_meta = [
    extract_domain_metadata(domain_lookup.get(did, "")) for did in val_domain_ids
]
test_meta = [
    extract_domain_metadata(domain_lookup.get(did, "")) for did in test_domain_ids
]

train_tlds = [m[1] for m in train_meta]
top_tlds = set(pd.Series(train_tlds).value_counts().head(30).index)
tld_to_code = {tld: i + 1 for i, tld in enumerate(sorted(top_tlds))}

known_train_freedom = [
    freedom_map[t] if t in freedom_map else freedom_map[t.split(".")[-1]]
    for t in train_tlds
    if t in freedom_map or t.split(".")[-1] in freedom_map
]
median_freedom = float(np.median(known_train_freedom)) if known_train_freedom else 45.0


def build_domain_tabular_features(meta_list):
    feats = np.zeros((len(meta_list), 14), dtype=np.float32)
    for i, (
        root,
        tld,
        length,
        num_dots,
        num_hyphens,
        digit_ratio,
        vowel_ratio,
        entropy,
    ) in enumerate(meta_list):
        tld_base = tld.split(".")[-1]
        tld_code = tld_to_code.get(tld, 0)
        has_freedom = 1.0 if (tld in freedom_map or tld_base in freedom_map) else 0.0
        freedom_score = freedom_map.get(tld, freedom_map.get(tld_base, median_freedom))

        is_eu = 1.0 if tld_base in EU_TLDS else 0.0
        is_ru = 1.0 if tld_base in RU_TLDS else 0.0
        is_anglo = 1.0 if tld_base in ANGLO_TLDS else 0.0
        is_asia = 1.0 if tld_base in ASIA_TLDS else 0.0
        is_com = 1.0 if tld_base == "com" else 0.0

        feats[i] = [
            float(length),
            float(num_dots),
            float(num_hyphens),
            float(digit_ratio),
            float(vowel_ratio),
            float(entropy),
            float(tld_code),
            has_freedom,
            float(freedom_score),
            is_eu,
            is_ru,
            is_anglo,
            is_asia,
            is_com,
        ]
    return feats


tr_dom_feats = build_domain_tabular_features(train_meta)
val_dom_feats = build_domain_tabular_features(val_meta)
ts_dom_feats = build_domain_tabular_features(test_meta)

# ---------------------------------------------------------------------------
# 5. Domain Name Subword Character N-Gram Features
# ---------------------------------------------------------------------------
train_texts = [
    m[0] if m[0] else domain_lookup.get(did, "")
    for did, m in zip(train_domain_ids, train_meta)
]
val_texts = [
    m[0] if m[0] else domain_lookup.get(did, "")
    for did, m in zip(val_domain_ids, val_meta)
]
test_texts = [
    m[0] if m[0] else domain_lookup.get(did, "")
    for did, m in zip(test_domain_ids, test_meta)
]

ngram_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=64,
    sublinear_tf=True,
)
ngram_vectorizer.fit(train_texts)

tr_ngram = ngram_vectorizer.transform(train_texts).toarray().astype(np.float32)
val_ngram = ngram_vectorizer.transform(val_texts).toarray().astype(np.float32)
ts_ngram = ngram_vectorizer.transform(test_texts).toarray().astype(np.float32)

# ---------------------------------------------------------------------------
# 6. URL Content Category Classification Features
# ---------------------------------------------------------------------------
url_cat_df = pd.read_csv(
    os.path.join(INPUT_DIR, "url-classification.csv"),
    usecols=["url", "category"],
)
categories = sorted(url_cat_df["category"].dropna().unique().tolist())
NUM_CATS = len(categories)


def parse_url_netloc(u):
    if not isinstance(u, str) or not u:
        return ""
    idx = u.find("://")
    if idx != -1:
        u = u[idx + 3 :]
    slash_idx = u.find("/")
    if slash_idx != -1:
        u = u[:slash_idx]
    colon_idx = u.find(":")
    if colon_idx != -1:
        u = u[:colon_idx]
    u = u.lower().strip()
    if u.startswith("www."):
        u = u[4:]
    return u


url_cat_df["domain_clean"] = url_cat_df["url"].apply(parse_url_netloc)
domain_cat_counts = (
    url_cat_df.groupby(["domain_clean", "category"]).size().unstack(fill_value=0)
)
del url_cat_df
gc.collect()

cat_matrix = (
    domain_cat_counts[categories].values
    / (domain_cat_counts[categories].values.sum(axis=1, keepdims=True) + 1e-6)
).astype(np.float32)
domain_to_cat_prob = dict(zip(domain_cat_counts.index, cat_matrix))

known_cat_tr = [
    domain_to_cat_prob[domain_lookup[did]]
    for did in train_domain_ids
    if did in domain_lookup and domain_lookup[did] in domain_to_cat_prob
]
cat_prior = (
    np.mean(known_cat_tr, axis=0)
    if known_cat_tr
    else np.full(NUM_CATS, 1.0 / NUM_CATS, dtype=np.float32)
)


def build_category_features(domain_arr):
    feats = np.zeros((len(domain_arr), NUM_CATS + 1), dtype=np.float32)
    for i, did in enumerate(domain_arr):
        d_name = domain_lookup.get(did, "")
        if d_name in domain_to_cat_prob:
            feats[i, :NUM_CATS] = domain_to_cat_prob[d_name]
            feats[i, NUM_CATS] = 1.0
        else:
            feats[i, :NUM_CATS] = cat_prior
            feats[i, NUM_CATS] = 0.0
    return feats


tr_cat_feats = build_category_features(train_domain_ids)
val_cat_feats = build_category_features(val_domain_ids)
ts_cat_feats = build_category_features(test_domain_ids)

del domain_cat_counts, domain_to_cat_prob, domain_lookup
gc.collect()

# ---------------------------------------------------------------------------
# 7. Dense Assembly & Scaling
# ---------------------------------------------------------------------------
dense_tr_raw = np.hstack([tr_topo, tr_dom_feats, tr_cat_feats, tr_ngram])
dense_val_raw = np.hstack([val_topo, val_dom_feats, val_cat_feats, val_ngram])
dense_ts_raw = np.hstack([ts_topo, ts_dom_feats, ts_cat_feats, ts_ngram])

scaler = StandardScaler()
dense_tr_scaled = np.nan_to_num(scaler.fit_transform(dense_tr_raw).astype(np.float32))
dense_val_scaled = np.nan_to_num(scaler.transform(dense_val_raw).astype(np.float32))
dense_ts_scaled = np.nan_to_num(scaler.transform(dense_ts_raw).astype(np.float32))

del (
    dense_tr_raw,
    dense_val_raw,
    dense_ts_raw,
    tr_topo,
    val_topo,
    ts_topo,
    tr_dom_feats,
    val_dom_feats,
    ts_dom_feats,
    tr_cat_feats,
    val_cat_feats,
    ts_cat_feats,
    tr_ngram,
    val_ngram,
    ts_ngram,
)
gc.collect()


# ---------------------------------------------------------------------------
# 8. TrackerSyndicateNet Architecture
# ---------------------------------------------------------------------------
class TrackerSyndicateNet(nn.Module):

    def __init__(
        self,
        dense_dim: int,
        num_trackers: int = 355,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.25,
        tracker_priors: np.ndarray = None,
        T_cooc: np.ndarray = None,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim

        # 1. Domain Context Backbone
        self.dom_input_proj = nn.Linear(dense_dim, 256)
        self.dom_bn1 = nn.BatchNorm1d(256)
        self.dom_act1 = nn.GELU()
        self.dom_drop1 = nn.Dropout(dropout)

        self.dom_res_block = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )
        self.dom_act2 = nn.GELU()

        self.dom_out_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5),
        )

        # 2. Tracker Prototype Syndicate Module
        self.tracker_prototypes = nn.Parameter(torch.empty(num_trackers, embed_dim))
        nn.init.xavier_uniform_(self.tracker_prototypes)

        self.syndicate_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(dropout * 0.5)

        # 3. Multi-Channel Topological Diffusion with Per-Tracker Projections and Syndicate Co-occurrence
        self.diff_weights = nn.Parameter(torch.empty(num_trackers, 3))
        self.diff_bias = nn.Parameter(torch.zeros(num_trackers))
        nn.init.normal_(self.diff_weights, mean=1.0, std=0.1)

        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        if T_cooc is not None:
            self.register_buffer(
                "T_cooc", torch.tensor(T_cooc, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "T_cooc",
                torch.zeros((num_trackers, num_trackers), dtype=torch.float32),
            )

        # 4. Domain-Conditioned Dynamic Modality Gating
        self.domain_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # 5. Prior Biases & Learned Tracker Offsets
        if tracker_priors is not None:
            priors = np.clip(tracker_priors, 1e-4, 1.0 - 1e-4)
            log_odds = np.log(priors / (1.0 - priors))
            self.prior_bias = nn.Parameter(
                torch.tensor(log_odds, dtype=torch.float32), requires_grad=False
            )
        else:
            self.prior_bias = nn.Parameter(
                torch.zeros(num_trackers, dtype=torch.float32),
                requires_grad=False,
            )

        self.prior_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.tracker_bias = nn.Parameter(torch.zeros(num_trackers, dtype=torch.float32))

    def forward(
        self,
        feat_dense: torch.Tensor,
        feat_neighbor_out: torch.Tensor,
        feat_neighbor_in: torch.Tensor,
        feat_direct_link: torch.Tensor,
    ) -> torch.Tensor:
        h = self.dom_drop1(self.dom_act1(self.dom_bn1(self.dom_input_proj(feat_dense))))
        h = self.dom_act2(h + self.dom_res_block(h))
        h_dom = self.dom_out_proj(h)

        proto_expanded = self.tracker_prototypes.unsqueeze(0)
        attn_out, _ = self.syndicate_attn(
            proto_expanded, proto_expanded, proto_expanded
        )
        syndicate_trackers = self.attn_norm(
            self.tracker_prototypes + self.attn_drop(attn_out.squeeze(0))
        )

        s_semantic = torch.matmul(h_dom, syndicate_trackers.t()) / math.sqrt(
            self.embed_dim
        )

        diff_stack = torch.stack(
            [feat_neighbor_out, feat_neighbor_in, feat_direct_link], dim=-1
        )
        s_proj = (
            diff_stack * self.diff_weights.unsqueeze(0)
        ).sum(dim=-1) + self.diff_bias
        s_graph = s_proj + self.beta * torch.matmul(s_proj, self.T_cooc)

        gates = torch.sigmoid(self.domain_gate(h_dom))
        alpha_semantic = gates[:, 0:1]
        alpha_graph = gates[:, 1:2]

        logits = (
            alpha_semantic * s_semantic
            + alpha_graph * s_graph
            + self.prior_scale * self.prior_bias
            + self.tracker_bias
        )
        return logits


# ---------------------------------------------------------------------------
# 9. Top10ThresholdMarginRecallLoss
# ---------------------------------------------------------------------------
class Top10ThresholdMarginRecallLoss(nn.Module):

    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.5,
        focal_weight: float = 0.35,
        gamma_pos: float = 0.0,
        gamma_neg: float = 1.5,
        neg_weight: float = 0.25,
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.focal_weight = focal_weight
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.neg_weight = neg_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_positives = targets.sum()

        top_k = min(10, logits.size(1))
        thresh_10th = torch.topk(logits, k=top_k, dim=1).values[:, -1:].detach()

        margin_violations = F.softplus(
            (thresh_10th - logits + self.margin) / self.temperature
        )
        margin_loss = (margin_violations * targets).sum() / num_positives.clamp(min=1.0)

        probs = torch.sigmoid(logits)
        pos_bce = (
            -targets
            * torch.log(probs.clamp(min=1e-7))
            * ((1.0 - probs) ** self.gamma_pos)
        )
        neg_bce = (
            -(1.0 - targets)
            * torch.log((1.0 - probs).clamp(min=1e-7))
            * (probs**self.gamma_neg)
        )
        focal_loss = (pos_bce + self.neg_weight * neg_bce).mean()

        return margin_loss + self.focal_weight * focal_loss


# ---------------------------------------------------------------------------
# 10. Dataset & DataLoaders
# ---------------------------------------------------------------------------
class MultiViewTrackerDataset(Dataset):

    def __init__(
        self,
        feat_dense: np.ndarray,
        feat_neighbor_out: np.ndarray,
        feat_neighbor_in: np.ndarray,
        feat_direct_link: np.ndarray,
        targets: np.ndarray = None,
    ):
        self.feat_dense = torch.from_numpy(feat_dense)
        self.feat_neighbor_out = torch.from_numpy(feat_neighbor_out)
        self.feat_neighbor_in = torch.from_numpy(feat_neighbor_in)
        self.feat_direct_link = torch.from_numpy(feat_direct_link)
        self.targets = torch.from_numpy(targets) if targets is not None else None

    def __len__(self) -> int:
        return len(self.feat_dense)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return (
                self.feat_dense[idx],
                self.feat_neighbor_out[idx],
                self.feat_neighbor_in[idx],
                self.feat_direct_link[idx],
                self.targets[idx],
            )
        return (
            self.feat_dense[idx],
            self.feat_neighbor_out[idx],
            self.feat_neighbor_in[idx],
            self.feat_direct_link[idx],
        )


train_dataset = MultiViewTrackerDataset(
    dense_tr_scaled,
    tr_neighbor_out,
    tr_neighbor_in,
    tr_direct_link,
    Y_train,
)
val_dataset = MultiViewTrackerDataset(
    dense_val_scaled,
    val_neighbor_out,
    val_neighbor_in,
    val_direct_link,
)
test_dataset = MultiViewTrackerDataset(
    dense_ts_scaled,
    ts_neighbor_out,
    ts_neighbor_in,
    ts_direct_link,
)

BATCH_SIZE = 512
NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


# ---------------------------------------------------------------------------
# 11. Validation Recall@10 Evaluation
# ---------------------------------------------------------------------------
def compute_val_recall10(
    eval_model: torch.nn.Module,
    loader: DataLoader,
    domain_ids: np.ndarray,
    ground_truth_dict: dict,
    eval_device: torch.device,
) -> float:
    eval_model.eval()
    pred_top10_list = []

    with torch.no_grad():
        for d_feat, n_out, n_in, direct in loader:
            d_feat = d_feat.to(eval_device, non_blocking=True)
            n_out = n_out.to(eval_device, non_blocking=True)
            n_in = n_in.to(eval_device, non_blocking=True)
            direct = direct.to(eval_device, non_blocking=True)

            logits = eval_model(d_feat, n_out, n_in, direct)
            top10_idx = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
            pred_top10_list.append(top10_idx)

    pred_matrix = np.vstack(pred_top10_list)
    recalls = []

    for i, did in enumerate(domain_ids):
        true_trackers = ground_truth_dict.get(did, set())
        if not true_trackers:
            recalls.append(0.0)
        else:
            predicted_trackers = set(pred_matrix[i].tolist())
            hits = len(true_trackers.intersection(predicted_trackers))
            recalls.append(hits / float(len(true_trackers)))

    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# 12. Model Training with Metric-Aligned Checkpointing
# ---------------------------------------------------------------------------
dense_feature_dim = dense_tr_scaled.shape[1]
model = TrackerSyndicateNet(
    dense_dim=dense_feature_dim,
    num_trackers=NUM_TRACKERS,
    tracker_priors=tracker_priors,
    T_cooc=T_cooc,
).to(device)

criterion = Top10ThresholdMarginRecallLoss(
    margin=1.0, temperature=0.5, focal_weight=0.35
)

NUM_EPOCHS = 12
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

CHECKPOINT_PATH = os.path.join(WORKING_DIR, "tracker_syndicate_net_best.pt")
best_val_recall = -1.0
best_epoch = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for d_feat, n_out, n_in, direct, targets in train_loader:
        d_feat = d_feat.to(device, non_blocking=True)
        n_out = n_out.to(device, non_blocking=True)
        n_in = n_in.to(device, non_blocking=True)
        direct = direct.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(d_feat, n_out, n_in, direct)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        b_len = d_feat.size(0)
        running_loss += loss.item() * b_len
        total_samples += b_len

    scheduler.step()
    epoch_train_loss = running_loss / max(1, total_samples)

    val_recall = compute_val_recall10(
        model, val_loader, val_domain_ids, val_ground_truth, device
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_epoch = epoch
        torch.save(model.state_dict(), CHECKPOINT_PATH)

    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} - Train Loss: {epoch_train_loss:.4f} - "
        f"Val Recall@10: {val_recall:.5f} (Best: {best_val_recall:.5f} @ Epoch {best_epoch}) - "
        f"LR: {current_lr:.6f}"
    )

# ---------------------------------------------------------------------------
# 13. Checkpoint Restoration & Final Evaluation
# ---------------------------------------------------------------------------
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
final_val_score = compute_val_recall10(
    model, val_loader, val_domain_ids, val_ground_truth, device
)

# ---------------------------------------------------------------------------
# 14. Test Inference & Submission Generation
# ---------------------------------------------------------------------------
model.eval()
test_top10_batches = []

with torch.no_grad():
    for d_feat, n_out, n_in, direct in test_loader:
        d_feat = d_feat.to(device, non_blocking=True)
        n_out = n_out.to(device, non_blocking=True)
        n_in = n_in.to(device, non_blocking=True)
        direct = direct.to(device, non_blocking=True)

        logits = model(d_feat, n_out, n_in, direct)
        top10_tracker_ids = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
        test_top10_batches.append(top10_tracker_ids)

test_top10_matrix = np.vstack(test_top10_batches)

submission_rows = []
for i, domain_id in enumerate(test_domain_ids):
    top10_trackers = test_top10_matrix[i]
    for tid in top10_trackers:
        tracking_did = tracker_id_to_domain_id[int(tid)]
        submission_rows.append((domain_id, tracking_did))

sub_df = pd.DataFrame(submission_rows, columns=["domain_id", "tracking_domain_id"])

EXPECTED_ROWS = len(test_domain_ids) * 10
assert len(sub_df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(sub_df)}"
assert sub_df["domain_id"].nunique() == len(
    test_domain_ids
), "Unique domain count mismatch!"
assert not sub_df.isnull().any().any(), "Found NaN values in submission!"

sub_csv_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df.to_csv(sub_csv_path, sep="\t", index=False)
shutil.copyfile(sub_csv_path, os.path.join(SUBMISSION_DIR, "submission.tsv"))

print(f"Final Validation Score: {final_val_score}")
