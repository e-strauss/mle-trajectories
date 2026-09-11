import os
import gc
import re
import copy
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# -------------------------------------------------------------------------
# 0. Global Configuration and Reproducibility
# -------------------------------------------------------------------------
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# -------------------------------------------------------------------------
# 1. Load Trackers Metadata and Target Domains
# -------------------------------------------------------------------------
trackers_df = pd.read_csv("input/trackers.tsv", sep="\t")
tracker_id_to_domain_id = dict(
    zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
)
tracker_domain_id_to_tracker_id = dict(
    zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
)
tracker_domain_ids_set = set(trackers_df["tracking_domain_id"].unique())
NUM_TRACKERS = len(trackers_df)

target_df = pd.read_csv("input/target.tsv", sep="\t")
test_domain_ids = target_df["domain_id"].to_numpy(dtype=np.int64)
NUM_TEST = len(test_domain_ids)

# -------------------------------------------------------------------------
# 2. Load Tracking Graph (Train) and Create Leak-Free Splits
# -------------------------------------------------------------------------
train_graph_df = pd.read_parquet("input/tracking_graph_train.parquet")
domain_trackers = (
    train_graph_df.groupby("domain_id")["tracker_id"].apply(list).to_dict()
)
all_train_domains = np.array(list(domain_trackers.keys()), dtype=np.int64)

# Stratified random split
np.random.seed(42)
shuffled_indices = np.random.permutation(len(all_train_domains))
all_train_domains_shuffled = all_train_domains[shuffled_indices]

N_TRAIN = min(100000, int(len(all_train_domains_shuffled) * 0.85))
N_VAL = min(20000, len(all_train_domains_shuffled) - N_TRAIN)

train_domain_ids = all_train_domains_shuffled[:N_TRAIN]
val_domain_ids = all_train_domains_shuffled[N_TRAIN : N_TRAIN + N_VAL]


def build_multi_hot_matrix(domain_ids, domain_trackers_dict, num_trackers):
    matrix = np.zeros((len(domain_ids), num_trackers), dtype=np.float32)
    for i, d_id in enumerate(domain_ids):
        t_ids = domain_trackers_dict.get(d_id, [])
        if len(t_ids) > 0:
            matrix[i, t_ids] = 1.0
    return matrix


y_train = build_multi_hot_matrix(train_domain_ids, domain_trackers, NUM_TRACKERS)
y_val = build_multi_hot_matrix(val_domain_ids, domain_trackers, NUM_TRACKERS)

# Marginal tracker priors strictly from Train split
tracker_priors = np.clip(y_train.mean(axis=0), 1e-6, 1.0)

del train_graph_df
gc.collect()

# ---------------------------------------------------------
# 3. Extract Graph Topology Features (link-graph.parquet)
# ---------------------------------------------------------
link_df = pd.read_parquet(
    "input/link-graph.parquet", columns=["source_domain_id", "target_domain_id"]
)
out_degrees = link_df["source_domain_id"].value_counts().to_dict()
in_degrees = link_df["target_domain_id"].value_counts().to_dict()

tracker_links_df = link_df[link_df["target_domain_id"].isin(tracker_domain_ids_set)]
tracker_outlink_count = tracker_links_df["source_domain_id"].value_counts().to_dict()
tracker_unique_linked = (
    tracker_links_df.groupby("source_domain_id")["target_domain_id"].nunique().to_dict()
)

top_30_tracker_domain_ids = trackers_df["tracking_domain_id"].iloc[:30].to_numpy()
tracker_links_top30 = tracker_links_df[
    tracker_links_df["target_domain_id"].isin(set(top_30_tracker_domain_ids))
]
top30_tracker_pairs = set(
    zip(
        tracker_links_top30["source_domain_id"], tracker_links_top30["target_domain_id"]
    )
)

del link_df, tracker_links_df, tracker_links_top30
gc.collect()


def extract_graph_features(domain_ids):
    features = []
    for d_id in domain_ids:
        out_d = out_degrees.get(d_id, 0)
        in_d = in_degrees.get(d_id, 0)
        t_out = tracker_outlink_count.get(d_id, 0)
        t_uniq = tracker_unique_linked.get(d_id, 0)

        row = [
            np.log1p(out_d),
            np.log1p(in_d),
            np.log1p(out_d + in_d),
            out_d / (out_d + in_d + 1.0),
            np.log1p(t_out),
            np.log1p(t_uniq),
            t_uniq / (t_out + 1.0),
            1.0 if t_out > 0 else 0.0,
            1.0 if in_d > 0 else 0.0,
            1.0 if out_d > 0 else 0.0,
        ]
        for tid in top_30_tracker_domain_ids:
            row.append(1.0 if (d_id, tid) in top30_tracker_pairs else 0.0)
        features.append(row)
    return np.array(features, dtype=np.float32)


X_graph_train = extract_graph_features(train_domain_ids)
X_graph_val = extract_graph_features(val_domain_ids)
X_graph_test = extract_graph_features(test_domain_ids)

# ---------------------------------------------------------
# 4. Hostname Lookup and Lexical Subword Representations
# ---------------------------------------------------------
domains_lookup = pd.read_parquet(
    "input/domains.parquet", columns=["domain_id", "domain"]
)
all_active_ids = (
    set(train_domain_ids).union(set(val_domain_ids)).union(set(test_domain_ids))
)
domains_lookup = domains_lookup[domains_lookup["domain_id"].isin(all_active_ids)]
id_to_hostname = dict(zip(domains_lookup["domain_id"], domains_lookup["domain"]))
del domains_lookup
gc.collect()


def get_hostnames(domain_ids):
    return [str(id_to_hostname.get(d_id, f"unknown{d_id}")) for d_id in domain_ids]


train_hostnames = get_hostnames(train_domain_ids)
val_hostnames = get_hostnames(val_domain_ids)
test_hostnames = get_hostnames(test_domain_ids)


def extract_domain_structures(hostnames):
    feats = []
    for h in hostnames:
        clean_h = h.strip().lower()
        parts = clean_h.split(".")
        length = len(clean_h)
        num_dots = clean_h.count(".")
        num_hyphens = clean_h.count("-")
        num_digits = sum(c.isdigit() for c in clean_h)
        digit_ratio = num_digits / max(length, 1)
        subdomain_depth = len(parts)
        sld_len = len(parts[-2]) if len(parts) >= 2 else length
        has_cdn = (
            1.0
            if ("cdn" in clean_h or "static" in clean_h or "media" in clean_h)
            else 0.0
        )
        has_blog = (
            1.0
            if ("blog" in clean_h or "news" in clean_h or "press" in clean_h)
            else 0.0
        )
        has_shop = (
            1.0
            if ("shop" in clean_h or "store" in clean_h or "pay" in clean_h)
            else 0.0
        )

        feats.append(
            [
                float(length),
                float(num_dots),
                float(num_hyphens),
                float(num_digits),
                float(digit_ratio),
                float(subdomain_depth),
                float(sld_len),
                has_cdn,
                has_blog,
                has_shop,
            ]
        )
    return np.array(feats, dtype=np.float32)


X_struct_train = extract_domain_structures(train_hostnames)
X_struct_val = extract_domain_structures(val_hostnames)
X_struct_test = extract_domain_structures(test_hostnames)

tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=25000,
    sublinear_tf=True,
    min_df=3,
)
tfidf_char.fit(train_hostnames)

svd_char = TruncatedSVD(n_components=64, random_state=42)
X_char_tfidf_train = tfidf_char.transform(train_hostnames)
X_subword_train = svd_char.fit_transform(X_char_tfidf_train).astype(np.float32)

X_char_tfidf_val = tfidf_char.transform(val_hostnames)
X_subword_val = svd_char.transform(X_char_tfidf_val).astype(np.float32)

X_char_tfidf_test = tfidf_char.transform(test_hostnames)
X_subword_test = svd_char.transform(X_char_tfidf_test).astype(np.float32)

del tfidf_char, svd_char, X_char_tfidf_train, X_char_tfidf_val, X_char_tfidf_test
gc.collect()

# ---------------------------------------------------------
# 5. Jurisdictional Press Freedom and TLD Features
# ---------------------------------------------------------
try:
    press_freedom_df = pd.read_csv("input/freedom-of-the-press.csv", sep="\t")
    if len(press_freedom_df.columns) < 3:
        press_freedom_df = pd.read_csv(
            "input/freedom-of-the-press.csv", sep=r"\s+", engine="python"
        )
except Exception:
    press_freedom_df = pd.read_csv(
        "input/freedom-of-the-press.csv", sep=r"\s+", engine="python"
    )

press_cols = press_freedom_df.columns.tolist()
tld_col = press_cols[0]
score_col = press_cols[-1]
tld_to_freedom = dict(
    zip(
        press_freedom_df[tld_col].astype(str).str.lower().str.strip(),
        pd.to_numeric(press_freedom_df[score_col], errors="coerce").fillna(30.0),
    )
)


def get_tld(hostname):
    parts = hostname.strip().lower().split(".")
    if len(parts) >= 2:
        return parts[-1]
    return "unknown"


train_tlds = [get_tld(h) for h in train_hostnames]
val_tlds = [get_tld(h) for h in val_hostnames]
test_tlds = [get_tld(h) for h in test_hostnames]

tld_counts_train = pd.Series(train_tlds).value_counts().to_dict()
total_train_tlds = len(train_tlds)
known_scores = [tld_to_freedom[t] for t in train_tlds if t in tld_to_freedom]
median_freedom = float(np.median(known_scores)) if len(known_scores) > 0 else 30.0


def extract_jurisdiction_features(tlds):
    feats = []
    for t in tlds:
        freq = tld_counts_train.get(t, 1) / total_train_tlds
        has_freedom = 1.0 if t in tld_to_freedom else 0.0
        score = tld_to_freedom.get(t, median_freedom)

        is_com = 1.0 if t == "com" else 0.0
        is_net = 1.0 if t == "net" else 0.0
        is_org = 1.0 if t == "org" else 0.0
        is_de = 1.0 if t == "de" else 0.0
        is_ru = 1.0 if t == "ru" else 0.0
        is_uk = 1.0 if t == "uk" else 0.0

        feats.append(
            [
                np.log1p(freq * 10000.0),
                has_freedom,
                score / 100.0,
                is_com,
                is_net,
                is_org,
                is_de,
                is_ru,
                is_uk,
            ]
        )
    return np.array(feats, dtype=np.float32)


X_jur_train = extract_jurisdiction_features(train_tlds)
X_jur_val = extract_jurisdiction_features(val_tlds)
X_jur_test = extract_jurisdiction_features(test_tlds)

# ---------------------------------------------------------
# 6. URL Content Category Classification Features
# ---------------------------------------------------------
url_cat_df = pd.read_csv("input/url-classification.csv")
cat_names = sorted(url_cat_df["category"].dropna().unique().tolist())
url_domain_pattern = re.compile(r"https?://(?:www\.)?([^/:]+)")


def extract_domain_from_url(u):
    m = url_domain_pattern.match(str(u).strip().lower())
    return m.group(1) if m else None


url_cat_df["clean_domain"] = url_cat_df["url"].apply(extract_domain_from_url)
url_cat_df = url_cat_df.dropna(subset=["clean_domain", "category"])

domain_cat_counts = (
    url_cat_df.groupby(["clean_domain", "category"]).size().unstack(fill_value=0)
)
domain_cat_probs = domain_cat_counts.div(domain_cat_counts.sum(axis=1), axis=0).to_dict(
    orient="index"
)

del url_cat_df, domain_cat_counts
gc.collect()


def extract_url_category_features(hostnames):
    feats = []
    num_cats = len(cat_names)
    for h in hostnames:
        clean_h = h.strip().lower()
        if clean_h.startswith("www."):
            clean_h = clean_h[4:]

        prob_dict = domain_cat_probs.get(clean_h, None)
        if prob_dict is not None:
            cat_row = [prob_dict.get(c, 0.0) for c in cat_names]
            has_cat = 1.0
        else:
            cat_row = [0.0] * num_cats
            has_cat = 0.0
        feats.append([has_cat] + cat_row)
    return np.array(feats, dtype=np.float32)


X_cat_train = extract_url_category_features(train_hostnames)
X_cat_val = extract_url_category_features(val_hostnames)
X_cat_test = extract_url_category_features(test_hostnames)

# ---------------------------------------------------------
# 7. Concatenate, Standardize, and Calculate Co-occurrence
# ---------------------------------------------------------
X_dense_train = np.hstack(
    [X_graph_train, X_struct_train, X_subword_train, X_jur_train, X_cat_train]
)
X_dense_val = np.hstack(
    [X_graph_val, X_struct_val, X_subword_val, X_jur_val, X_cat_val]
)
X_dense_test = np.hstack(
    [X_graph_test, X_struct_test, X_subword_test, X_jur_test, X_cat_test]
)

X_dense_train = np.nan_to_num(X_dense_train, nan=0.0, posinf=0.0, neginf=0.0)
X_dense_val = np.nan_to_num(X_dense_val, nan=0.0, posinf=0.0, neginf=0.0)
X_dense_test = np.nan_to_num(X_dense_test, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_dense_train).astype(np.float32)
X_val_scaled = scaler.transform(X_dense_val).astype(np.float32)
X_test_scaled = scaler.transform(X_dense_test).astype(np.float32)

co_occurrence = np.dot(y_train.T, y_train)
tracker_counts = np.diag(co_occurrence)
co_occurrence_norm = co_occurrence / (
    tracker_counts[:, None] + tracker_counts[None, :] - co_occurrence + 1e-6
)
co_occurrence_norm = co_occurrence_norm.astype(np.float32)

del (
    X_dense_train,
    X_dense_val,
    X_dense_test,
    X_graph_train,
    X_graph_val,
    X_graph_test,
    X_struct_train,
    X_struct_val,
    X_struct_test,
    X_subword_train,
    X_subword_val,
    X_subword_test,
    X_jur_train,
    X_jur_val,
    X_jur_test,
    X_cat_train,
    X_cat_val,
    X_cat_test,
)
gc.collect()


# -------------------------------------------------------------------------
# 8. Model Architecture and Loss Formulation
# -------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.fc(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + residual


class DomainBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.block1 = ResidualDenseBlock(
            hidden_dim, hidden_dim, dropout_rate=dropout_rate
        )
        self.block2 = ResidualDenseBlock(
            hidden_dim, embed_dim, dropout_rate=dropout_rate
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.block1(h)
        h = self.block2(h)
        return self.final_norm(h)


class TrackerPrototypeDiffusion(nn.Module):
    def __init__(
        self,
        num_trackers: int,
        embed_dim: int = 256,
        co_occurrence_matrix: np.ndarray = None,
        diffusion_weight: float = 0.25,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim
        self.diffusion_weight = diffusion_weight

        self.prototypes = nn.Parameter(
            torch.empty(num_trackers, embed_dim).uniform_(
                -1.0 / np.sqrt(embed_dim), 1.0 / np.sqrt(embed_dim)
            )
        )

        if co_occurrence_matrix is not None:
            A = np.copy(co_occurrence_matrix).astype(np.float32)
            np.fill_diagonal(A, 1.0)
            deg = np.sum(A, axis=1)
            deg_inv_sqrt = np.power(np.maximum(deg, 1e-6), -0.5)
            D_mat = np.diag(deg_inv_sqrt)
            norm_adj = D_mat @ A @ D_mat
            adj_tensor = torch.from_numpy(norm_adj).float()
        else:
            adj_tensor = torch.eye(num_trackers, dtype=torch.float32)

        self.register_buffer("norm_adjacency", adj_tensor)
        self.diffusion_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self) -> torch.Tensor:
        diffused = torch.matmul(self.norm_adjacency, self.prototypes)
        diffused = self.diffusion_linear(diffused)
        refined_prototypes = self.prototypes + self.diffusion_weight * F.gelu(diffused)
        return F.normalize(refined_prototypes, p=2, dim=-1)


class TrackerCrossAttentionRanker(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_trackers: int = 355,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        dropout_rate: float = 0.2,
        co_occurrence_matrix: np.ndarray = None,
        tracker_priors: np.ndarray = None,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim

        self.backbone = DomainBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
        )

        self.prototype_bank = TrackerPrototypeDiffusion(
            num_trackers=num_trackers,
            embed_dim=embed_dim,
            co_occurrence_matrix=co_occurrence_matrix,
        )

        self.syndicate_gate = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_trackers),
            nn.Sigmoid(),
        )

        if tracker_priors is not None:
            safe_priors = np.clip(tracker_priors, 1e-4, 1.0 - 1e-4)
            init_bias = np.log(safe_priors / (1.0 - safe_priors))
            self.prior_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        else:
            self.prior_bias = nn.Parameter(torch.zeros(num_trackers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        domain_emb = self.backbone(x)
        domain_emb_norm = F.normalize(domain_emb, p=2, dim=-1)
        tracker_prototypes = self.prototype_bank()
        proto_scores = torch.matmul(domain_emb_norm, tracker_prototypes.t()) * 10.0
        gates = self.syndicate_gate(domain_emb)
        logits = (proto_scores * (0.5 + 0.5 * gates)) + self.prior_bias
        return logits


class AsymmetricTopKRecallLoss(nn.Module):
    def __init__(
        self,
        k: int = 10,
        margin: float = 0.5,
        margin_weight: float = 1.5,
        gamma_neg: float = 3.0,
        gamma_pos: float = 0.0,
        clip_neg: float = 0.05,
    ):
        super().__init__()
        self.k = k
        self.margin = margin
        self.margin_weight = margin_weight
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_neg = clip_neg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = torch.clamp(probs - self.clip_neg, min=0.0)

        loss_pos = (
            -targets
            * torch.pow(1.0 - probs_pos, self.gamma_pos)
            * torch.log(probs_pos + eps)
        )
        loss_neg = (
            -(1.0 - targets)
            * torch.pow(probs_neg, self.gamma_neg)
            * torch.log(1.0 - probs_neg + eps)
        )
        focal_loss = (loss_pos + loss_neg).sum(dim=1).mean()

        top_k_values, _ = torch.topk(logits, k=self.k, dim=1)
        kth_score = top_k_values[:, self.k - 1 : self.k].detach()

        violation = F.relu(kth_score - logits + self.margin)
        pos_mask = targets > 0.5
        pos_counts = pos_mask.sum(dim=1).clamp(min=1.0)
        margin_loss_per_domain = (violation * pos_mask.float()).sum(dim=1) / pos_counts
        margin_loss = margin_loss_per_domain.mean()

        return focal_loss + self.margin_weight * margin_loss


# -------------------------------------------------------------------------
# 9. Model and Optimizer Initialization
# -------------------------------------------------------------------------
input_dim = X_train_scaled.shape[1]
num_trackers = NUM_TRACKERS

model = TrackerCrossAttentionRanker(
    input_dim=input_dim,
    num_trackers=num_trackers,
    hidden_dim=512,
    embed_dim=256,
    dropout_rate=0.2,
    co_occurrence_matrix=co_occurrence_norm,
    tracker_priors=tracker_priors,
).to(device)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "norm" in name or "bias" in name:
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

criterion = AsymmetricTopKRecallLoss(
    k=10, margin=0.5, margin_weight=1.5, gamma_neg=3.0, gamma_pos=0.0
)

# -------------------------------------------------------------------------
# 10. DataLoaders and Evaluation Routine
# -------------------------------------------------------------------------
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024

train_dataset = TensorDataset(
    torch.from_numpy(X_train_scaled), torch.from_numpy(y_train)
)
val_dataset = TensorDataset(torch.from_numpy(X_val_scaled), torch.from_numpy(y_val))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False,
)


def evaluate_recall_at_10(model, data_loader, dev):
    model.eval()
    all_domain_recalls = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(dev, non_blocking=True)
            batch_y = batch_y.to(dev, non_blocking=True)
            logits = model(batch_x)
            top10_indices = torch.topk(logits, k=10, dim=1).indices

            hits = torch.gather(batch_y, 1, top10_indices).sum(dim=1)
            num_true = batch_y.sum(dim=1).clamp(min=1.0)
            domain_recalls = hits / num_true
            all_domain_recalls.append(domain_recalls.cpu())

    all_domain_recalls = torch.cat(all_domain_recalls, dim=0)
    return float(all_domain_recalls.mean().item())


# -------------------------------------------------------------------------
# 11. Training Loop with Metric Tracking & Checkpointing
# -------------------------------------------------------------------------
EPOCHS = 12
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val_recall = -1.0
best_model_weights = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    scheduler.step()
    avg_train_loss = total_loss / max(num_batches, 1)
    val_recall = evaluate_recall_at_10(model, val_loader, device)

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model_weights = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Recall@10: {val_recall:.5f}"
    )

# -------------------------------------------------------------------------
# 12. Model Restoration, Inference, and Submission Creation
# -------------------------------------------------------------------------
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)
    torch.save(best_model_weights, "./working/best_tracker_model.pt")

final_val_score = evaluate_recall_at_10(model, val_loader, device)

test_tensor = torch.from_numpy(X_test_scaled)
test_dataset = TensorDataset(test_tensor)
test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False,
)

model.eval()
all_top10_tracker_ids = []
with torch.no_grad():
    for (batch_x,) in test_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        test_logits = model(batch_x)
        top10_indices = torch.topk(test_logits, k=10, dim=1).indices
        all_top10_tracker_ids.append(top10_indices.cpu().numpy())

all_top10_tracker_ids = np.vstack(all_top10_tracker_ids)

tracker_domain_lookup = np.array(
    [tracker_id_to_domain_id[i] for i in range(num_trackers)], dtype=np.int64
)
top10_tracking_domain_ids = tracker_domain_lookup[all_top10_tracker_ids]

domain_col = np.repeat(test_domain_ids, 10)
tracking_domain_col = top10_tracking_domain_ids.reshape(-1)

submission_df = pd.DataFrame(
    {"domain_id": domain_col, "tracking_domain_id": tracking_domain_col}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, sep="\t", index=False)
submission_df.to_csv("./submission/submission.tsv", sep="\t", index=False)

assert os.path.exists(submission_path), "Submission file not found!"
assert len(submission_df) == len(test_domain_ids) * 10, "Row count mismatch!"
assert submission_df["domain_id"].nunique() == len(
    test_domain_ids
), "Unique domain count mismatch!"
assert not submission_df.isnull().any().any(), "Submission contains null values!"

print(f"Final Validation Score: {final_val_score:.5f}")
