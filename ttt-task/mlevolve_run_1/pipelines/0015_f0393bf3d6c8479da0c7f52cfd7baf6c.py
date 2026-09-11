import collections
import gc
import math
import os
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Ensure output directories exist
os.makedirs("working", exist_ok=True)
os.makedirs("submission", exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Configuring pipeline on execution device: {device}")

# -------------------------------------------------------------------------
# 1. Load Trackers Taxonomy and Target Domains
# -------------------------------------------------------------------------
print("Loading trackers metadata and target domains...")
trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
num_trackers = len(trackers_df)

tracker_mapping = np.zeros(num_trackers, dtype=np.int64)
tracker_domain_ids_set = set()
for _, row in trackers_df.iterrows():
    tid = int(row["tracker_id"])
    tdid = int(row["tracking_domain_id"])
    tracker_mapping[tid] = tdid
    tracker_domain_ids_set.add(tdid)

target_df = pd.read_csv("input/target.tsv", sep="\t")
test_domain_ids = target_df["domain_id"].to_numpy(dtype=np.int64)
test_domain_set = set(test_domain_ids)

# -------------------------------------------------------------------------
# 2. Load Training Tracking Graph and Build Multi-Hot Targets
# -------------------------------------------------------------------------
print("Loading tracking_graph_train.parquet...")
train_graph_df = pd.read_parquet("input/tracking_graph_train.parquet")
train_unique_domains = train_graph_df["domain_id"].unique()
total_labeled_domains = len(train_unique_domains)

MAX_TRAIN_DOMAINS = 250000
if total_labeled_domains > MAX_TRAIN_DOMAINS:
    rng = np.random.RandomState(42)
    selected_domain_ids = rng.choice(
        train_unique_domains, size=MAX_TRAIN_DOMAINS, replace=False
    )
    filtered_graph_df = train_graph_df[
        train_graph_df["domain_id"].isin(selected_domain_ids)
    ]
else:
    selected_domain_ids = train_unique_domains
    filtered_graph_df = train_graph_df

filtered_graph_df = filtered_graph_df.drop_duplicates(
    subset=["domain_id", "tracker_id"]
)

domain_to_row_idx = {did: idx for idx, did in enumerate(selected_domain_ids)}
row_indices = (
    filtered_graph_df["domain_id"].map(domain_to_row_idx).to_numpy(dtype=np.int32)
)
col_indices = filtered_graph_df["tracker_id"].to_numpy(dtype=np.int32)
data_ones = np.ones(len(filtered_graph_df), dtype=np.uint8)

full_sparse_targets = csr_matrix(
    (data_ones, (row_indices, col_indices)),
    shape=(len(selected_domain_ids), num_trackers),
    dtype=np.uint8,
)

del train_graph_df, filtered_graph_df, row_indices, col_indices, data_ones
gc.collect()

# -------------------------------------------------------------------------
# 3. Leak-Free Train / Hold-Out Validation Split
# -------------------------------------------------------------------------
train_indices, val_indices = train_test_split(
    np.arange(len(selected_domain_ids)), test_size=0.15, random_state=42
)

train_domain_ids = selected_domain_ids[train_indices]
val_domain_ids = selected_domain_ids[val_indices]

y_train = (full_sparse_targets[train_indices].toarray() > 0).astype(np.float32)
y_val = (full_sparse_targets[val_indices].toarray() > 0).astype(np.float32)
del full_sparse_targets
gc.collect()

# Empirical marginal log-odds prior bias computed strictly from train split
train_tracker_counts = y_train.sum(axis=0)
tracker_priors = (train_tracker_counts + 1e-4) / (len(y_train) + 2e-4)
tracker_log_odds = np.log(tracker_priors / (1.0 - tracker_priors)).astype(np.float32)

all_active_domain_ids = set(train_domain_ids) | set(val_domain_ids) | test_domain_set

# -------------------------------------------------------------------------
# 4. Ingest Hostnames, Press Freedom, and URL Taxonomies
# -------------------------------------------------------------------------
print("Ingesting domain hostnames and auxiliary taxonomies...")
domains_df = pd.read_parquet("input/domains.parquet")
domains_df = domains_df[domains_df["domain_id"].isin(all_active_domain_ids)]

id_to_domain_str = dict(zip(domains_df["domain_id"], domains_df["domain"].astype(str)))
domain_str_to_id = dict(
    zip(domains_df["domain"].astype(str).str.lower(), domains_df["domain_id"])
)
del domains_df
gc.collect()

# Freedom of the press scores
try:
    fop_df = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
    if "tld" not in fop_df.columns or "freedom_of_the_press" not in fop_df.columns:
        fop_df = pd.read_csv("input/freedom-of-the-press.csv")
    tld_to_fop = dict(
        zip(
            fop_df["tld"].astype(str).str.lower(),
            pd.to_numeric(fop_df["freedom_of_the_press"], errors="coerce"),
        )
    )
    valid_scores = [v for v in tld_to_fop.values() if not np.isnan(v)]
    fop_median = float(np.median(valid_scores)) if valid_scores else 45.0
except Exception:
    tld_to_fop = {}
    fop_median = 45.0

# URL category distribution vector
try:
    url_df = pd.read_csv("input/url-classification.csv")
    url_hostnames = (
        url_df["url"]
        .astype(str)
        .str.extract(r"https?://(?:www\.)?([^/:]+)", expand=False)
        .str.lower()
    )
    url_df["domain_id"] = url_hostnames.map(domain_str_to_id)
    matched_url_df = url_df.dropna(subset=["domain_id"]).copy()
    matched_url_df["domain_id"] = matched_url_df["domain_id"].astype(np.int64)
    matched_url_df = matched_url_df[
        matched_url_df["domain_id"].isin(all_active_domain_ids)
    ]

    category_names = sorted(url_df["category"].dropna().unique().tolist())
    cat_counts = (
        matched_url_df.groupby(["domain_id", "category"]).size().unstack(fill_value=0)
    )
    for cat in category_names:
        if cat not in cat_counts.columns:
            cat_counts[cat] = 0
    cat_counts = cat_counts[category_names]
    cat_sums = cat_counts.sum(axis=1).replace(0, 1)
    cat_dist = cat_counts.div(cat_sums, axis=0)

    domain_ids_arr = cat_dist.index.to_numpy()
    cat_values = cat_dist.to_numpy(dtype=np.float32)
    domain_to_category_features = {
        int(did): cat_values[i] for i, did in enumerate(domain_ids_arr)
    }
    del url_df, matched_url_df, cat_counts, cat_dist
    gc.collect()
except Exception:
    domain_to_category_features = {}
    category_names = [f"cat_{i}" for i in range(15)]

# -------------------------------------------------------------------------
# 5. Stream Link Graph for Topological Metrics
# -------------------------------------------------------------------------
print("Streaming link-graph.parquet for topology centrality...")
in_degrees = collections.defaultdict(int)
out_degrees = collections.defaultdict(int)
tracker_out_links = collections.defaultdict(int)

pq_file = pq.ParquetFile("input/link-graph.parquet")
batch_size = 5_000_000

for batch in pq_file.iter_batches(
    batch_size=batch_size, columns=["source_domain_id", "target_domain_id"]
):
    src_arr = batch.column("source_domain_id").to_numpy(zero_copy_only=False)
    tgt_arr = batch.column("target_domain_id").to_numpy(zero_copy_only=False)

    src_series = pd.Series(src_arr)
    src_vc = src_series.value_counts()
    for did, count in src_vc[src_vc.index.isin(all_active_domain_ids)].items():
        out_degrees[did] += int(count)

    tgt_series = pd.Series(tgt_arr)
    tgt_vc = tgt_series.value_counts()
    for did, count in tgt_vc[tgt_vc.index.isin(all_active_domain_ids)].items():
        in_degrees[did] += int(count)

    is_tracker_target = np.isin(tgt_arr, list(tracker_domain_ids_set))
    if np.any(is_tracker_target):
        tracker_srcs = src_arr[is_tracker_target]
        tsrc_vc = pd.Series(tracker_srcs).value_counts()
        for did, count in tsrc_vc[tsrc_vc.index.isin(all_active_domain_ids)].items():
            tracker_out_links[did] += int(count)

del pq_file
gc.collect()

# -------------------------------------------------------------------------
# 6. Tabular and Lexical Domain Feature Extraction
# -------------------------------------------------------------------------
TOP_TLDS = [
    "com",
    "org",
    "net",
    "de",
    "ru",
    "uk",
    "nl",
    "fr",
    "cn",
    "br",
    "it",
    "pl",
    "es",
    "ca",
    "au",
    "cz",
    "in",
    "jp",
    "se",
    "eu",
    "io",
    "info",
    "biz",
    "tv",
    "cc",
]
tld_to_idx = {tld: i for i, tld in enumerate(TOP_TLDS)}


def compute_entropy(s):
    if not s:
        return 0.0
    counts = collections.Counter(s)
    length = len(s)
    return -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())


def extract_domain_tabular_features(domain_id):
    hostname = id_to_domain_str.get(domain_id, f"unknown-{domain_id}.com").lower()
    parts = hostname.split(".")
    sld = parts[-2] if len(parts) >= 2 else parts[0]
    tld = parts[-1] if len(parts) >= 1 else ""

    length = len(hostname)
    num_dots = hostname.count(".")
    num_hyphens = hostname.count("-")
    num_digits = sum(c.isdigit() for c in hostname)
    num_vowels = sum(c in "aeiou" for c in hostname)
    num_consonants = sum(c in "bcdfghjklmnpqrstvwxyz" for c in hostname)
    entropy = compute_entropy(hostname)

    lexical = [
        float(length),
        float(num_dots),
        float(num_hyphens),
        float(num_digits),
        float(num_digits / (length + 1e-5)),
        float(num_vowels),
        float(num_vowels / (length + 1e-5)),
        float(num_consonants),
        float(num_consonants / (length + 1e-5)),
        float(entropy),
        float(len(sld)),
        1.0 if "shop" in hostname else 0.0,
        1.0 if "blog" in hostname else 0.0,
        1.0 if "news" in hostname else 0.0,
        1.0 if "forum" in hostname else 0.0,
        1.0 if "app" in hostname else 0.0,
        1.0 if "dev" in hostname else 0.0,
        1.0 if hostname.startswith("m.") else 0.0,
        1.0 if hostname.endswith((".org", ".net")) else 0.0,
        (
            1.0
            if (".gov" in hostname or ".edu" in hostname or ".ac." in hostname)
            else 0.0
        ),
    ]

    tld_onehot = [0.0] * len(TOP_TLDS)
    if tld in tld_to_idx:
        tld_onehot[tld_to_idx[tld]] = 1.0
    is_cctld = 1.0 if len(tld) == 2 else 0.0

    fop_score = float(tld_to_fop.get(tld, fop_median))
    has_fop = 1.0 if tld in tld_to_fop else 0.0

    if domain_id in domain_to_category_features:
        cat_vec = domain_to_category_features[domain_id].tolist()
        has_cat = [1.0]
    else:
        cat_vec = [0.0] * len(category_names)
        has_cat = [0.0]

    in_deg = float(in_degrees.get(domain_id, 0))
    out_deg = float(out_degrees.get(domain_id, 0))
    trk_links = float(tracker_out_links.get(domain_id, 0))
    graph_feats = [
        in_deg,
        out_deg,
        math.log1p(in_deg),
        math.log1p(out_deg),
        in_deg + out_deg,
        (in_deg + 1.0) / (out_deg + 1.0),
        trk_links,
        1.0 if trk_links > 0 else 0.0,
    ]

    return (
        lexical
        + tld_onehot
        + [is_cctld, fop_score, has_fop]
        + cat_vec
        + has_cat
        + graph_feats
    )


print("Extracting domain tabular features...")
X_train_tab = np.array(
    [extract_domain_tabular_features(did) for did in train_domain_ids],
    dtype=np.float32,
)
X_val_tab = np.array(
    [extract_domain_tabular_features(did) for did in val_domain_ids],
    dtype=np.float32,
)
X_test_tab = np.array(
    [extract_domain_tabular_features(did) for did in test_domain_ids],
    dtype=np.float32,
)

# -------------------------------------------------------------------------
# 7. Hostname TF-IDF & TruncatedSVD (Fitted Strictly on Train)
# -------------------------------------------------------------------------
print("Fitting TruncatedSVD representation on hostnames...")
train_hostnames = [
    id_to_domain_str.get(did, f"unknown-{did}.com").lower() for did in train_domain_ids
]
val_hostnames = [
    id_to_domain_str.get(did, f"unknown-{did}.com").lower() for did in val_domain_ids
]
test_hostnames = [
    id_to_domain_str.get(did, f"unknown-{did}.com").lower() for did in test_domain_ids
]

tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=5,
    max_features=15000,
    sublinear_tf=True,
)
tfidf.fit(train_hostnames)

train_tfidf = tfidf.transform(train_hostnames)
val_tfidf = tfidf.transform(val_hostnames)
test_tfidf = tfidf.transform(test_hostnames)

svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(train_tfidf)

X_train_svd = svd.transform(train_tfidf).astype(np.float32)
X_val_svd = svd.transform(val_tfidf).astype(np.float32)
X_test_svd = svd.transform(test_tfidf).astype(np.float32)

del tfidf, svd, train_tfidf, val_tfidf, test_tfidf
gc.collect()

# -------------------------------------------------------------------------
# 8. Feature Fusion and Standardization
# -------------------------------------------------------------------------
X_train_raw = np.hstack([X_train_tab, X_train_svd])
X_val_raw = np.hstack([X_val_tab, X_val_svd])
X_test_raw = np.hstack([X_test_tab, X_test_svd])

del X_train_tab, X_val_tab, X_test_tab, X_train_svd, X_val_svd, X_test_svd
gc.collect()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val = scaler.transform(X_val_raw).astype(np.float32)
X_test = scaler.transform(X_test_raw).astype(np.float32)

np.nan_to_num(X_train, copy=False)
np.nan_to_num(X_val, copy=False)
np.nan_to_num(X_test, copy=False)

del X_train_raw, X_val_raw, X_test_raw
gc.collect()

input_features_dim = int(X_train.shape[1])
num_candidate_trackers = num_trackers


# -------------------------------------------------------------------------
# 9. Model Architecture: GatedTrackerInteractionNet
# -------------------------------------------------------------------------
class GatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.silu(self.fc1(x))
        gate, val = torch.chunk(self.fc2(h), 2, dim=-1)
        gated = val * torch.sigmoid(gate)
        out = residual + self.dropout(gated)
        return self.layer_norm(out)


class TrackerCorrelationLayer(nn.Module):
    def __init__(self, num_trks: int, rank: int = 32):
        super().__init__()
        self.down = nn.Linear(num_trks, rank, bias=False)
        self.up = nn.Linear(rank, num_trks, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.down.weight, gain=0.01)
        nn.init.xavier_uniform_(self.up.weight, gain=0.01)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        corr = self.up(F.relu(self.down(logits)))
        return logits + self.alpha * corr


class GatedTrackerInteractionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_trks: int = 355,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        dropout_rate: float = 0.25,
        initial_log_odds: np.ndarray = None,
    ):
        super().__init__()
        self.num_trackers = num_trks

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.block1 = GatedResidualBlock(hidden_dim, dropout_rate)
        self.block2 = GatedResidualBlock(hidden_dim, dropout_rate)
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )

        self.tracker_embeddings = nn.Parameter(torch.empty(num_trks, latent_dim))
        nn.init.xavier_uniform_(self.tracker_embeddings)

        self.correlation_layer = TrackerCorrelationLayer(num_trks, rank=32)
        self.tracker_bias = nn.Parameter(torch.zeros(num_trks))
        if initial_log_odds is not None:
            with torch.no_grad():
                init_bias = torch.from_numpy(initial_log_odds).float()
                init_bias = torch.clamp(init_bias, min=-8.0, max=4.0)
                self.tracker_bias.copy_(init_bias)

        self.scale = 1.0 / (latent_dim**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.block1(h)
        h = self.block2(h)
        z = self.to_latent(h)
        interaction_logits = torch.matmul(z, self.tracker_embeddings.t()) * self.scale
        logits = interaction_logits + self.tracker_bias
        logits = self.correlation_layer(logits)
        return logits


# -------------------------------------------------------------------------
# 10. Asymmetric Recall Objective Function
# -------------------------------------------------------------------------
class AsymmetricRecallLoss(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.5,
        margin: float = 0.05,
        ranking_weight: float = 0.25,
        top_k: int = 10,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.margin = margin
        self.ranking_weight = ranking_weight
        self.top_k = top_k
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        p_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        p_neg = 1.0 - probs
        p_neg_shifted = torch.clamp(p_neg + self.margin, max=1.0)
        p_pos_neg = torch.clamp(1.0 - p_neg_shifted, min=0.0, max=1.0)

        focal_pos = (1.0 - p_pos).pow(self.gamma_pos)
        focal_neg = p_pos_neg.pow(self.gamma_neg)

        loss_pos = -targets * focal_pos * torch.log(p_pos)
        loss_neg = (
            -(1.0 - targets)
            * focal_neg
            * torch.log(p_neg_shifted.clamp(min=self.eps, max=1.0))
        )
        asl_loss = (loss_pos + loss_neg).sum(dim=1).mean()

        if self.ranking_weight > 0:
            pos_mask = targets > 0.5
            neg_mask = ~pos_mask
            has_pos = pos_mask.any(dim=1)
            if has_pos.any():
                logits_selected = logits[has_pos]
                pos_mask_sel = pos_mask[has_pos]
                neg_mask_sel = neg_mask[has_pos]

                neg_logits = torch.where(
                    neg_mask_sel,
                    logits_selected,
                    torch.tensor(-1e4, device=logits.device, dtype=logits.dtype),
                )
                top_k_neg, _ = torch.topk(
                    neg_logits, k=min(self.top_k, neg_logits.size(1)), dim=1
                )

                pos_logits = torch.where(
                    pos_mask_sel,
                    logits_selected,
                    torch.tensor(1e4, device=logits.device, dtype=logits.dtype),
                )
                min_pos_logits = pos_logits.min(dim=1, keepdim=True).values

                margin_violation = F.relu(top_k_neg - min_pos_logits + 1.0)
                rank_loss = margin_violation.mean()
            else:
                rank_loss = torch.tensor(0.0, device=logits.device)
            return asl_loss + self.ranking_weight * rank_loss
        return asl_loss


# -------------------------------------------------------------------------
# 11. Model, Optimizer, and DataLoader Initialization
# -------------------------------------------------------------------------
model = GatedTrackerInteractionNet(
    input_dim=input_features_dim,
    num_trks=num_candidate_trackers,
    hidden_dim=512,
    latent_dim=256,
    dropout_rate=0.25,
    initial_log_odds=tracker_log_odds,
).to(device)

criterion = AsymmetricRecallLoss(
    gamma_pos=0.0,
    gamma_neg=2.5,
    margin=0.05,
    ranking_weight=0.25,
    top_k=10,
).to(device)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name or "layer_norm" in name or "alpha" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

optimizer = AdamW(
    [
        {"params": decay_params, "weight_decay": 1e-4, "lr": 1e-3},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": 1e-3},
    ],
    eps=1e-8,
)

num_epochs = 10
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
y_train_c = np.ascontiguousarray(y_train, dtype=np.float32)
X_val_c = np.ascontiguousarray(X_val, dtype=np.float32)
y_val_c = np.ascontiguousarray(y_val, dtype=np.float32)
X_test_c = np.ascontiguousarray(X_test, dtype=np.float32)

train_dataset = TensorDataset(torch.from_numpy(X_train_c), torch.from_numpy(y_train_c))
val_dataset = TensorDataset(torch.from_numpy(X_val_c))
test_dataset = TensorDataset(torch.from_numpy(X_test_c))

pin_memory = torch.cuda.is_available()
train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=2048,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=2048,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory,
)


def evaluate_recall10(
    net: torch.nn.Module,
    loader: DataLoader,
    targets: np.ndarray,
    dev: torch.device,
) -> float:
    net.eval()
    top10_batches = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(dev, non_blocking=True)
            logits = net(bx)
            top10 = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
            top10_batches.append(top10)

    top10_indices = np.concatenate(top10_batches, axis=0)
    true_counts = targets.sum(axis=1)
    recalled_hits = np.take_along_axis(targets, top10_indices, axis=1).sum(axis=1)
    valid_mask = true_counts > 0

    recalls = np.zeros(len(targets), dtype=np.float64)
    recalls[valid_mask] = recalled_hits[valid_mask] / true_counts[valid_mask]
    return float(np.mean(recalls))


# -------------------------------------------------------------------------
# 12. Training Loop with Model Checkpointing
# -------------------------------------------------------------------------
best_val_score = -1.0
best_state = None
checkpoint_path = "working/best_tracker_model.pt"

print(f"Starting model training for {num_epochs} epochs...")

for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for bx, by in train_loader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(bx)
        loss = criterion(logits, by)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_loss / max(num_batches, 1)
    val_recall = evaluate_recall10(model, val_loader, y_val_c, device)
    current_lr = optimizer.param_groups[0]["lr"]

    if val_recall > best_val_score:
        best_val_score = val_recall
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, checkpoint_path)

    print(
        f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Recall@10: {val_recall:.5f} | LR: {current_lr:.6f}"
    )

# -------------------------------------------------------------------------
# 13. Load Best Checkpoint & Run Test Inference
# -------------------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

final_val_score = evaluate_recall10(model, val_loader, y_val_c, device)

model.eval()
test_top10_list = []
with torch.no_grad():
    for (bx,) in test_loader:
        bx = bx.to(device, non_blocking=True)
        logits = model(bx)
        top10_test = torch.topk(logits, k=10, dim=1).indices.cpu().numpy()
        test_top10_list.append(top10_test)

test_top10_indices = np.concatenate(test_top10_list, axis=0)
test_top10_tdids = tracker_mapping[test_top10_indices]

expanded_domain_ids = np.repeat(test_domain_ids, 10)
expanded_tracking_ids = test_top10_tdids.flatten()

sub_df = pd.DataFrame(
    {
        "domain_id": expanded_domain_ids,
        "tracking_domain_id": expanded_tracking_ids,
    }
)

assert len(sub_df) == len(test_domain_ids) * 10
assert sub_df["domain_id"].nunique() == len(test_domain_ids)
assert sub_df["tracking_domain_id"].isnull().sum() == 0
assert sub_df["domain_id"].isnull().sum() == 0

sub_df.to_csv("submission/submission.csv", sep="\t", index=False)
sub_df.to_csv("submission/submission.tsv", sep="\t", index=False)

print(f"Final Validation Score: {final_val_score}")
