import copy
import gc
import json
import os
import re
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------------
# Global Configuration and Seed Setup
# -------------------------------------------------------------------------
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# Step 1: Data Processing and Feature Engineering
# -------------------------------------------------------------------------
# 1. Load Trackers Metadata
trackers_df = pd.read_csv(os.path.join(INPUT_DIR, "trackers.tsv"), sep="\t")
trackers_df["tracker_id"] = trackers_df["tracker_id"].astype(np.int32)
trackers_df["tracking_domain_id"] = trackers_df["tracking_domain_id"].astype(np.int64)
trackers_df.to_parquet(
    os.path.join(WORKING_DIR, "tracker_metadata.parquet"), index=False
)
tracker_domain_ids_set = set(trackers_df["tracking_domain_id"].unique())
num_trackers = len(trackers_df)

# 2. Load Target (Test) Domains
target_df = pd.read_csv(os.path.join(INPUT_DIR, "target.tsv"), sep="\t")
test_domain_ids = target_df["domain_id"].astype(np.int64).values
test_domains_set = set(test_domain_ids)
np.save(os.path.join(WORKING_DIR, "test_domain_ids.npy"), test_domain_ids)

# 3. Load Tracking Graph & Create Train/Validation Split
tg_table = pq.read_table(
    os.path.join(INPUT_DIR, "tracking_graph_train.parquet"),
    columns=["domain_id", "tracker_id"],
)
tg_domain_ids = tg_table["domain_id"].to_numpy()
tg_tracker_ids = tg_table["tracker_id"].to_numpy().astype(np.int32)
del tg_table
gc.collect()

# Exclude any test domains from training candidates
unique_tg_domains = np.unique(tg_domain_ids)
valid_candidate_domains = unique_tg_domains[
    ~np.isin(unique_tg_domains, test_domain_ids)
]

# Shuffle candidates deterministically
shuffled_candidates = np.random.RandomState(SEED).permutation(valid_candidate_domains)
n_candidates = len(shuffled_candidates)
val_size = min(30000, int(n_candidates * 0.15))
train_size = min(200000, n_candidates - val_size)

val_domain_ids = shuffled_candidates[:val_size]
train_domain_ids = shuffled_candidates[val_size : val_size + train_size]

np.save(os.path.join(WORKING_DIR, "val_domain_ids.npy"), val_domain_ids)
np.save(os.path.join(WORKING_DIR, "train_domain_ids.npy"), train_domain_ids)


# Fast CSR ground truth matrix construction
def build_ground_truth_csr(domain_ids_arr, tg_doms, tg_tracks, num_tr):
    dom_to_idx = {d: i for i, d in enumerate(domain_ids_arr)}
    mask = np.isin(tg_doms, domain_ids_arr)
    sub_doms = tg_doms[mask]
    sub_tracks = tg_tracks[mask]

    rows = np.array([dom_to_idx[d] for d in sub_doms], dtype=np.int32)
    cols = sub_tracks
    vals = np.ones(len(rows), dtype=np.uint8)

    csr = csr_matrix(
        (vals, (rows, cols)), shape=(len(domain_ids_arr), num_tr), dtype=np.uint8
    )
    csr.data = np.minimum(csr.data, 1)
    return csr


val_labels_csr = build_ground_truth_csr(
    val_domain_ids, tg_domain_ids, tg_tracker_ids, num_trackers
)
train_labels_csr = build_ground_truth_csr(
    train_domain_ids, tg_domain_ids, tg_tracker_ids, num_trackers
)

save_npz(os.path.join(WORKING_DIR, "val_labels.npz"), val_labels_csr)
save_npz(os.path.join(WORKING_DIR, "train_labels.npz"), train_labels_csr)

# Global tracker frequencies strictly on training domains
train_tracker_counts = np.asarray(train_labels_csr.sum(axis=0)).ravel()
global_tracker_prior = train_tracker_counts / len(train_domain_ids)
top_trackers_global = np.argsort(-global_tracker_prior)[:10]

# Compute Tracker Co-occurrence Matrix (355 x 355) strictly on training set
co_occ = (train_labels_csr.T @ train_labels_csr).astype(np.float32).toarray()
co_occ_norm = co_occ / (train_tracker_counts[:, None] + 1e-6)
np.savez_compressed(
    os.path.join(WORKING_DIR, "tracker_priors.npz"),
    global_prior=global_tracker_prior,
    co_occurrence=co_occ_norm,
)
del tg_domain_ids, tg_tracker_ids
gc.collect()

# 4. Load Hostname Strings for Required Domains
all_needed_ids = set(train_domain_ids) | set(val_domain_ids) | set(test_domain_ids)
domains_table = pq.read_table(
    os.path.join(INPUT_DIR, "domains.parquet"),
    columns=["domain_id", "domain"],
)
dom_ids = domains_table["domain_id"].to_numpy()
dom_names = domains_table["domain"].to_numpy()
del domains_table
gc.collect()

mask_needed = np.isin(dom_ids, list(all_needed_ids))
needed_dom_ids = dom_ids[mask_needed]
needed_dom_names = dom_names[mask_needed]
del dom_ids, dom_names
gc.collect()

domain_lookup = dict(zip(needed_dom_ids, needed_dom_names))
del needed_dom_ids, needed_dom_names
gc.collect()

# 5. Load Freedom of the Press Dataset
press_freedom_map = {}
press_path = os.path.join(INPUT_DIR, "freedom-of-the-press.csv")
if os.path.exists(press_path):
    try:
        df_press = pd.read_csv(press_path, sep="\t")
        if len(df_press.columns) == 1:
            df_press = pd.read_csv(press_path, sep=",")
    except Exception:
        df_press = pd.read_csv(press_path)
    df_press.columns = [c.strip() for c in df_press.columns]
    for _, row in df_press.iterrows():
        tld_val = str(row["tld"]).strip().lower()
        try:
            score_val = float(row["freedom_of_the_press"])
            press_freedom_map[tld_val] = score_val
        except (ValueError, TypeError):
            continue

# 6. Load and Aggregate URL Classification Categories
url_cats_df = pd.read_csv(
    os.path.join(INPUT_DIR, "url-classification.csv"),
    usecols=["url", "category"],
)
known_categories = sorted(url_cats_df["category"].dropna().unique().tolist())
cat_to_idx = {cat: i for i, cat in enumerate(known_categories)}
url_domain_pattern = re.compile(r"^(?:https?://)?(?:www\.)?([^/:\?]+)", re.IGNORECASE)


def clean_url_to_domain(u):
    if not isinstance(u, str):
        return ""
    m = url_domain_pattern.match(u.strip())
    return m.group(1).lower() if m else ""


url_cats_df["clean_domain"] = url_cats_df["url"].apply(clean_url_to_domain)
url_cats_df = url_cats_df[url_cats_df["clean_domain"] != ""]

# Aggregate category counts per domain string
grouped = url_cats_df.groupby(["clean_domain", "category"]).size().unstack(fill_value=0)
domain_category_map = grouped.to_dict(orient="index")
del url_cats_df, grouped
gc.collect()

# 7. Load Link Graph Centrality & Tracker Outlink Features
lg_table = pq.read_table(
    os.path.join(INPUT_DIR, "link-graph.parquet"),
    columns=["source_domain_id", "target_domain_id"],
)
src_arr = lg_table["source_domain_id"].to_numpy()
dst_arr = lg_table["target_domain_id"].to_numpy()
del lg_table
gc.collect()

# Direct links to trackers
tracker_dst_mask = np.isin(
    dst_arr, np.array(list(tracker_domain_ids_set), dtype=np.int64)
)
tracker_src_ids = src_arr[tracker_dst_mask]
tracker_dst_ids = dst_arr[tracker_dst_mask]

# Top 10 most linked-to tracker domains
top_linked_tracker_domains = (
    pd.Series(tracker_dst_ids).value_counts().head(10).index.tolist()
)

# Out-degree and in-degree Series
out_degrees = pd.Series(src_arr).value_counts()
in_degrees = pd.Series(dst_arr).value_counts()
tracker_out_degrees = pd.Series(tracker_src_ids).value_counts()

# Top tracker specific link indicators
tracker_link_pairs = pd.DataFrame({"src": tracker_src_ids, "dst": tracker_dst_ids})
tracker_link_pairs = tracker_link_pairs[
    tracker_link_pairs["dst"].isin(top_linked_tracker_domains)
].drop_duplicates()
tracker_pair_set = set(zip(tracker_link_pairs["src"], tracker_link_pairs["dst"]))

del src_arr, dst_arr, tracker_src_ids, tracker_dst_ids, tracker_link_pairs
gc.collect()


# 8. Helper Functions for Feature Extraction
def extract_tld(d_str):
    if not isinstance(d_str, str) or "." not in d_str:
        return "unknown"
    parts = d_str.lower().strip().split(".")
    if (
        len(parts) >= 3
        and parts[-2]
        in {
            "co",
            "com",
            "org",
            "net",
            "edu",
            "gov",
            "ac",
            "mil",
            "gv",
            "ne",
        }
        and len(parts[-1]) == 2
    ):
        return f"{parts[-2]}.{parts[-1]}"
    return parts[-1]


# Extract TLDs for train set to build TLD priors
train_hostnames = [domain_lookup.get(did, "") for did in train_domain_ids]
train_tlds = [extract_tld(h) for h in train_hostnames]
tld_freq_map = pd.Series(train_tlds).value_counts(normalize=True).to_dict()
top_25_tlds = pd.Series(train_tlds).value_counts().head(25).index.tolist()

# Compute TLD-conditional tracker rates strictly on training set
train_tld_df = pd.DataFrame(
    {"tld": train_tlds, "row_idx": np.arange(len(train_domain_ids))}
)
tld_tracker_probs = {}
tld_expected_count = {}
for tld_name, group in train_tld_df.groupby("tld"):
    if len(group) >= 15:
        group_rows = group["row_idx"].values
        sub_labels = train_labels_csr[group_rows]
        mean_probs = np.asarray(sub_labels.mean(axis=0)).ravel()
        tld_tracker_probs[tld_name] = mean_probs
        tld_expected_count[tld_name] = float(sub_labels.sum(axis=1).mean())

train_press_scores = [
    press_freedom_map.get(t, np.nan) for t in train_tlds if t in press_freedom_map
]
median_press_score = (
    float(np.nanmedian(train_press_scores)) if len(train_press_scores) > 0 else 25.0
)

# 9. Subword Character N-Gram SVD Representation
tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    min_df=5,
    max_features=5000,
    dtype=np.float32,
)
svd = TruncatedSVD(n_components=32, random_state=SEED)
train_tfidf = tfidf.fit_transform(train_hostnames)
svd.fit(train_tfidf)
del train_tfidf
gc.collect()


# 10. Vectorized Feature Extraction Routine
def extract_feature_matrix(domain_id_list, hostnames_list):
    n = len(domain_id_list)
    features_dict = {}

    # Lexical features
    lens = np.array([len(h) for h in hostnames_list], dtype=np.float32)
    dots = np.array([h.count(".") for h in hostnames_list], dtype=np.float32)
    hyphens = np.array([h.count("-") for h in hostnames_list], dtype=np.float32)
    digits = np.array(
        [sum(c.isdigit() for c in h) for h in hostnames_list], dtype=np.float32
    )
    vowels = np.array(
        [sum(c in "aeiou" for c in h.lower()) for h in hostnames_list],
        dtype=np.float32,
    )

    features_dict["domain_len"] = lens
    features_dict["num_dots"] = dots
    features_dict["num_hyphens"] = hyphens
    features_dict["num_digits"] = digits
    features_dict["digit_ratio"] = digits / (lens + 1.0)
    features_dict["vowel_ratio"] = vowels / (lens + 1.0)

    # Keywords / Intent Indicators
    features_dict["has_www"] = np.array(
        [1.0 if h.lower().startswith("www.") else 0.0 for h in hostnames_list],
        dtype=np.float32,
    )
    features_dict["has_cdn"] = np.array(
        [
            1.0 if any(k in h.lower() for k in ["cdn", "static", "assets"]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_blog"] = np.array(
        [
            1.0 if any(k in h.lower() for k in ["blog", "news", "press"]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_shop"] = np.array(
        [
            (
                1.0
                if any(k in h.lower() for k in ["shop", "store", "market", "cart"])
                else 0.0
            )
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_app"] = np.array(
        [
            1.0 if any(k in h.lower() for k in ["api", "app", "dev", "cloud"]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )
    features_dict["has_gov_or_edu"] = np.array(
        [
            1.0 if any(k in h.lower() for k in [".gov", ".edu", ".ac."]) else 0.0
            for h in hostnames_list
        ],
        dtype=np.float32,
    )

    # TLD features
    tlds = [extract_tld(h) for h in hostnames_list]
    features_dict["is_cctld"] = np.array(
        [1.0 if len(t) == 2 else 0.0 for t in tlds], dtype=np.float32
    )
    features_dict["tld_freq"] = np.array(
        [tld_freq_map.get(t, 0.0) for t in tlds], dtype=np.float32
    )

    for top_tld in top_25_tlds:
        features_dict[f"tld_is_{top_tld}"] = np.array(
            [1.0 if t == top_tld else 0.0 for t in tlds], dtype=np.float32
        )

    # Freedom of the Press features
    press_scores = []
    press_missing = []
    for t in tlds:
        if t in press_freedom_map:
            press_scores.append(press_freedom_map[t])
            press_missing.append(0.0)
        else:
            press_scores.append(median_press_score)
            press_missing.append(1.0)
    features_dict["press_freedom"] = np.array(press_scores, dtype=np.float32)
    features_dict["press_missing"] = np.array(press_missing, dtype=np.float32)

    # URL Classification features
    has_cat = []
    cat_matrix = np.zeros((n, len(known_categories)), dtype=np.float32)
    for i, h in enumerate(hostnames_list):
        h_clean = h.lower().strip()
        if h_clean.startswith("www."):
            h_clean = h_clean[4:]
        if h_clean in domain_category_map:
            has_cat.append(1.0)
            c_dict = domain_category_map[h_clean]
            for cat_name, cnt in c_dict.items():
                if cat_name in cat_to_idx:
                    cat_matrix[i, cat_to_idx[cat_name]] = float(cnt)
        else:
            has_cat.append(0.0)

    features_dict["has_url_category"] = np.array(has_cat, dtype=np.float32)
    for idx_cat, cat_name in enumerate(known_categories):
        features_dict[f"url_cat_{cat_name}"] = cat_matrix[:, idx_cat]

    # Link Graph features
    out_deg = out_degrees.reindex(domain_id_list, fill_value=0).values.astype(
        np.float32
    )
    in_deg = in_degrees.reindex(domain_id_list, fill_value=0).values.astype(np.float32)
    tr_out = tracker_out_degrees.reindex(domain_id_list, fill_value=0).values.astype(
        np.float32
    )

    features_dict["out_degree"] = out_deg
    features_dict["in_degree"] = in_deg
    features_dict["log_out_degree"] = np.log1p(out_deg)
    features_dict["log_in_degree"] = np.log1p(in_deg)
    features_dict["degree_ratio"] = (
        features_dict["log_in_degree"] - features_dict["log_out_degree"]
    )
    features_dict["total_degree"] = np.log1p(out_deg + in_deg)
    features_dict["is_isolated"] = np.array(
        (out_deg == 0) & (in_deg == 0), dtype=np.float32
    )
    features_dict["tracker_outlinks"] = tr_out
    features_dict["has_tracker_outlinks"] = np.array(tr_out > 0, dtype=np.float32)

    # Top tracker specific links
    for tr_dom_id in top_linked_tracker_domains:
        features_dict[f"links_to_tracker_{tr_dom_id}"] = np.array(
            [
                1.0 if (did, tr_dom_id) in tracker_pair_set else 0.0
                for did in domain_id_list
            ],
            dtype=np.float32,
        )

    # TLD Tracker Likelihood Prior features
    mean_expected = float(np.mean(list(tld_expected_count.values()) or [3.0]))
    features_dict["tld_expected_trackers"] = np.array(
        [tld_expected_count.get(t, mean_expected) for t in tlds], dtype=np.float32
    )

    top5_tld_probs = np.zeros((n, 5), dtype=np.float32)
    for i, t in enumerate(tlds):
        p_vec = tld_tracker_probs.get(t, global_tracker_prior)
        top5_tld_probs[i] = p_vec[top_trackers_global[:5]]

    for k in range(5):
        features_dict[f"tld_prior_tracker_{k}"] = top5_tld_probs[:, k]

    # Character N-gram SVD features
    tfidf_mat = tfidf.transform(hostnames_list)
    svd_mat = svd.transform(tfidf_mat).astype(np.float32)
    for k in range(svd_mat.shape[1]):
        features_dict[f"svd_char_{k}"] = svd_mat[:, k]

    return pd.DataFrame(features_dict)


# Extract Feature Matrices
train_features_df = extract_feature_matrix(train_domain_ids, train_hostnames)

val_hostnames = [domain_lookup.get(did, "") for did in val_domain_ids]
val_features_df = extract_feature_matrix(val_domain_ids, val_hostnames)

test_hostnames = [domain_lookup.get(did, "") for did in test_domain_ids]
test_features_df = extract_feature_matrix(test_domain_ids, test_hostnames)

# Fill residual NaNs with training medians
train_medians = train_features_df.median()
train_features_df.fillna(train_medians, inplace=True)
val_features_df.fillna(train_medians, inplace=True)
test_features_df.fillna(train_medians, inplace=True)

num_features = train_features_df.shape[1]

# Prior log odds for output layer initialization
eps_p = 1e-4
p_clipped = np.clip(global_tracker_prior, eps_p, 1.0 - eps_p)
prior_log_odds = np.log(p_clipped / (1.0 - p_clipped)).astype(np.float32)


# -------------------------------------------------------------------------
# Step 2: Model Architecture Design
# -------------------------------------------------------------------------
class SwishGLUResidualBlock(nn.Module):
    """Gated Linear Unit residual block with Swish activation for tabular representations."""

    def __init__(self, dim: int, expansion_factor: int = 2, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = dim * expansion_factor
        self.linear_gate = nn.Linear(dim, hidden_dim)
        self.linear_val = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        gated = F.silu(self.linear_val(x_norm)) * self.linear_gate(x_norm)
        out = self.dropout(self.out_proj(gated))
        return residual + out


class LabelInteractionModule(nn.Module):
    """Low-rank tracker correlation refinement module to capture co-occurrence synergies."""

    def __init__(self, num_tr: int, rank: int = 32):
        super().__init__()
        self.down_proj = nn.Linear(num_tr, rank, bias=False)
        self.up_proj = nn.Linear(rank, num_tr, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        corr_delta = self.up_proj(F.gelu(self.down_proj(logits)))
        return logits + self.alpha * corr_delta


class TrackerGatedRankNet(nn.Module):
    """Deep Tabular Residual Ranking Network with Gated Blocks and Label Interaction."""

    def __init__(
        self,
        in_features: int,
        num_tr: int = 355,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.15,
        prior_bias: np.ndarray = None,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features)
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.proj_act = nn.SiLU()

        self.blocks = nn.ModuleList(
            [
                SwishGLUResidualBlock(hidden_dim, expansion_factor=2, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_tr)

        if prior_bias is not None and len(prior_bias) == num_tr:
            with torch.no_grad():
                self.classifier.bias.copy_(torch.from_numpy(prior_bias))

        self.label_interaction = LabelInteractionModule(num_tr=num_tr, rank=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_bn(x)
        h = self.proj_act(self.input_proj(h))

        for block in self.blocks:
            h = block(h)

        h = self.final_dropout(self.final_norm(h))
        raw_logits = self.classifier(h)
        refined_logits = self.label_interaction(raw_logits)
        return refined_logits


class TopKAsymmetricRecallLoss(nn.Module):
    """Composite loss combining Asymmetric Focal BCE and Top-K Hard Negative Margin Ranking."""

    def __init__(
        self,
        gamma_neg: float = 2.0,
        margin: float = 1.0,
        top_k_neg: int = 15,
        ranking_weight: float = 0.6,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.ranking_weight = ranking_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_f = targets.float()

        # 1. Asymmetric Focal Component
        pos_loss = -targets_f * torch.log(probs + self.eps)
        neg_weights = torch.pow(probs, self.gamma_neg)
        neg_loss = -(1.0 - targets_f) * neg_weights * torch.log(1.0 - probs + self.eps)
        asym_bce = (pos_loss + neg_loss).sum(dim=1).mean()

        # 2. Differentiable Top-K Negative Margin Ranking Component
        neg_mask = targets_f == 0.0
        neg_scores = torch.where(
            neg_mask, logits, torch.tensor(-1e6, device=logits.device)
        )
        top_k_val = min(self.top_k_neg, logits.size(1))
        top_negatives, _ = torch.topk(neg_scores, k=top_k_val, dim=1)

        pos_counts = targets_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        has_positives = (targets_f.sum(dim=1) > 0).float()

        diff = self.margin + top_negatives.unsqueeze(1) - logits.unsqueeze(2)
        pairwise_margin = F.softplus(diff)
        pos_mask = targets_f.unsqueeze(2)

        ranking_loss_per_sample = (pairwise_margin * pos_mask).sum(dim=(1, 2)) / (
            pos_counts.squeeze(1) * top_k_val
        )
        rank_loss = (ranking_loss_per_sample * has_positives).sum() / (
            has_positives.sum() + self.eps
        )

        return asym_bce + self.ranking_weight * rank_loss


# Instantiate Model, Loss, Optimizer, and Scheduler
model = TrackerGatedRankNet(
    in_features=num_features,
    num_tr=num_trackers,
    hidden_dim=256,
    num_blocks=3,
    dropout=0.15,
    prior_bias=prior_log_odds,
).to(device)

criterion = TopKAsymmetricRecallLoss(
    gamma_neg=2.0,
    margin=1.0,
    top_k_neg=15,
    ranking_weight=0.6,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

epochs = 12
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-5,
)

# -------------------------------------------------------------------------
# Step 3: Model Training and Evaluation
# -------------------------------------------------------------------------
X_train_tensor = torch.from_numpy(train_features_df.values.astype(np.float32))
y_train_tensor = torch.from_numpy(train_labels_csr.toarray().astype(np.float32))

X_val_tensor = torch.from_numpy(val_features_df.values.astype(np.float32))
y_val_tensor = torch.from_numpy(val_labels_csr.toarray().astype(np.float32))

X_test_tensor = torch.from_numpy(test_features_df.values.astype(np.float32))

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)


def compute_recall_at_10(
    eval_model: torch.nn.Module,
    features_t: torch.Tensor,
    targets_t: torch.Tensor,
    batch_size: int = 1024,
) -> float:
    eval_model.eval()
    all_hits = []
    all_totals = []
    with torch.no_grad():
        n = features_t.size(0)
        for i in range(0, n, batch_size):
            bx = features_t[i : i + batch_size].to(device, non_blocking=True)
            by = targets_t[i : i + batch_size].to(device, non_blocking=True)

            logits = eval_model(bx)
            _, top10_preds = torch.topk(logits, k=10, dim=1)

            hits = torch.gather(by, 1, top10_preds).sum(dim=1).float()
            totals = by.sum(dim=1).float()

            all_hits.append(hits.cpu())
            all_totals.append(totals.cpu())

    hits_vec = torch.cat(all_hits, dim=0)
    totals_vec = torch.cat(all_totals, dim=0)

    domain_recalls = torch.where(
        totals_vec > 0, hits_vec / totals_vec, torch.zeros_like(hits_vec)
    )
    return float(domain_recalls.mean().item())


best_val_recall = -1.0
best_model_weights = None
patience = 4
patience_counter = 0

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for bx, by in train_loader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    scheduler.step()
    epoch_loss = running_loss / max(num_batches, 1)

    val_recall = compute_recall_at_10(
        model, X_val_tensor, y_val_tensor, batch_size=1024
    )
    print(
        f"Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss:.4f} - Val Recall@10: {val_recall:.4f}"
    )

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Load Best Model Checkpoint
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, os.path.join(WORKING_DIR, "best_tracker_model.pt"))

final_val_score = compute_recall_at_10(
    model, X_val_tensor, y_val_tensor, batch_size=1024
)

# -------------------------------------------------------------------------
# Test Inference and Submission Generation
# -------------------------------------------------------------------------
model.eval()
test_predictions_list = []

with torch.no_grad():
    n_test = X_test_tensor.size(0)
    for i in range(0, n_test, 1024):
        bx = X_test_tensor[i : i + 1024].to(device, non_blocking=True)
        logits = model(bx)
        _, top10_preds = torch.topk(logits, k=10, dim=1)
        test_predictions_list.append(top10_preds.cpu().numpy())

test_pred_tracker_ids = np.concatenate(test_predictions_list, axis=0)

# Build Tracker ID to Tracking Domain ID Lookup
tracker_id_to_domain_id = np.zeros(num_trackers, dtype=np.int64)
tracker_id_to_domain_id[trackers_df["tracker_id"].values] = trackers_df[
    "tracking_domain_id"
].values

# Map predicted tracker indices to full tracking domain IDs
test_pred_tracking_domains = tracker_id_to_domain_id[test_pred_tracker_ids]

# Format submission table
repeated_domain_ids = np.repeat(test_domain_ids, 10)
flat_tracking_domain_ids = test_pred_tracking_domains.ravel()

submission_df = pd.DataFrame(
    {
        "domain_id": repeated_domain_ids,
        "tracking_domain_id": flat_tracking_domain_ids,
    }
)

# Save submission in both CSV and TSV formats
sub_csv_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_tsv_path = os.path.join(SUBMISSION_DIR, "submission.tsv")

submission_df.to_csv(sub_csv_path, sep="\t", index=False)
submission_df.to_csv(sub_tsv_path, sep="\t", index=False)

# Clean up memory
del (
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    X_test_tensor,
    train_dataset,
    train_loader,
)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------------------------------------------------------------
# Final Score Output
# -------------------------------------------------------------------------
print(f"Final Validation Score: {final_val_score}")
