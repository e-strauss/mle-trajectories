import collections
import gc
import json
import math
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. Feature Extraction Helper Functions
# =========================================================================
def extract_tld_and_sld(hostname):
    """Robust extraction of top-level and second-level domain from hostname."""
    if not isinstance(hostname, str) or not hostname:
        return "unknown", "unknown"
    parts = hostname.lower().strip().split(".")
    if len(parts) == 1:
        return parts[0], parts[0]
    two_part = ".".join(parts[-2:])
    common_two_part = {
        "co.uk",
        "gov.uk",
        "ac.uk",
        "org.uk",
        "com.au",
        "net.au",
        "org.au",
        "co.nz",
        "co.jp",
        "com.br",
        "com.mx",
        "com.tr",
        "com.ru",
    }
    if two_part in common_two_part and len(parts) >= 3:
        return two_part, parts[-3]
    return parts[-1], parts[-2]


def extract_registered_domain(hostname):
    """Extract registered domain (SLD.TLD) from hostname."""
    tld, sld = extract_tld_and_sld(hostname)
    if tld == "unknown" or sld == "unknown":
        return "unknown"
    return f"{sld}.{tld}"


def compute_lexical_features(hostname):
    """Extract 9 distinct lexical & morphological features from domain hostname."""
    if not isinstance(hostname, str) or len(hostname) == 0:
        return [0.0] * 9
    h = hostname.lower().strip()
    length = len(h)
    num_dots = h.count(".")
    num_digits = sum(c.isdigit() for c in h)
    num_hyphens = h.count("-")
    num_vowels = sum(c in "aeiou" for c in h)
    vowel_ratio = num_vowels / max(length, 1)
    digit_ratio = num_digits / max(length, 1)

    counts = collections.Counter(h)
    entropy = -sum((cnt / length) * math.log2(cnt / length) for cnt in counts.values())
    is_ip = 1.0 if all(p.isdigit() for p in h.split(".") if p) else 0.0
    has_www = 1.0 if "www." in h else 0.0

    return [
        float(length),
        float(num_dots),
        float(num_digits),
        float(num_hyphens),
        float(vowel_ratio),
        float(digit_ratio),
        float(entropy),
        float(is_ip),
        float(has_www),
    ]


# =========================================================================
# 2. Model Architecture & Loss Design
# =========================================================================
class ResidualGatedBlock(nn.Module):
    """Residual Gated Linear Unit (SwiGLU) block for dense multi-modal feature transformation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        gate, val = self.fc1(x_norm).chunk(2, dim=-1)
        hidden = self.act(gate) * val
        out = self.dropout(self.fc2(hidden))
        return residual + out


class TrackerRelationalEmbedding(nn.Module):
    """Learned tracker embeddings conditioned on categorical metadata and enriched via relational self-attention

    to capture functional dependencies and co-occurrence bundles across the 355 trackers.
    """

    def __init__(
        self,
        num_trackers: int = 355,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        company_ids: torch.Tensor = None,
        category_ids: torch.Tensor = None,
        country_ids: torch.Tensor = None,
        brand_ids: torch.Tensor = None,
        num_companies: int = 150,
        num_categories: int = 30,
        num_countries: int = 50,
        num_brands: int = 150,
    ):
        super().__init__()
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim
        self.tracker_embeddings = nn.Parameter(
            torch.randn(num_trackers, embed_dim) * (1.0 / math.sqrt(embed_dim))
        )

        meta_dim = embed_dim // 4
        self.company_emb = nn.Embedding(max(1, num_companies), meta_dim)
        self.category_emb = nn.Embedding(max(1, num_categories), meta_dim)
        self.country_emb = nn.Embedding(max(1, num_countries), meta_dim)
        self.brand_emb = nn.Embedding(max(1, num_brands), meta_dim)

        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

        if company_ids is not None:
            self.register_buffer(
                "company_ids", torch.as_tensor(company_ids, dtype=torch.long)
            )
            self.register_buffer(
                "category_ids", torch.as_tensor(category_ids, dtype=torch.long)
            )
            self.register_buffer(
                "country_ids", torch.as_tensor(country_ids, dtype=torch.long)
            )
            self.register_buffer(
                "brand_ids", torch.as_tensor(brand_ids, dtype=torch.long)
            )
        else:
            self.company_ids = None
            self.category_ids = None
            self.country_ids = None
            self.brand_ids = None

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self) -> torch.Tensor:
        base_emb = self.tracker_embeddings
        if self.company_ids is not None:
            c = self.company_emb(self.company_ids)
            cat = self.category_emb(self.category_ids)
            co = self.country_emb(self.country_ids)
            b = self.brand_emb(self.brand_ids)
            meta = torch.cat([c, cat, co, b], dim=-1)
            base_emb = base_emb + self.meta_proj(meta)

        emb = base_emb.unsqueeze(0)
        attn_out, _ = self.self_attn(emb, emb, emb)
        tracker_reps = self.norm(emb + attn_out).squeeze(0)
        return tracker_reps


class GBTrackerNet(nn.Module):
    """Gated Bilinear Tracker Network (GB-TrackerNet).

    Combines:
    1. Multi-modal domain feature projection via stacked SwiGLU residual blocks.
    2. Metadata-conditioned relational tracker embedding attention head.
    3. Bilinear tracker-domain scoring.
    4. Multi-prior gating head with inbound diffusion, outbound diffusion, and direct link indicators.
    """

    def __init__(
        self,
        in_features: int = 112,
        num_trackers: int = 355,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 3,
        dropout: float = 0.15,
        tracker_metadata: dict = None,
        transition_matrix: np.ndarray = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_trackers = num_trackers
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList(
            [
                ResidualGatedBlock(
                    dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )
        self.domain_norm = nn.LayerNorm(embed_dim)

        tracker_kwargs = {
            "num_trackers": num_trackers,
            "embed_dim": embed_dim,
            "dropout": dropout,
        }
        if tracker_metadata is not None:
            tracker_kwargs.update(tracker_metadata)
        self.tracker_module = TrackerRelationalEmbedding(**tracker_kwargs)
        self.tracker_bias = nn.Parameter(torch.zeros(num_trackers))

        # Learnable 355-dimensional tracker-specific scaling vectors
        self.direct_scale = nn.Parameter(torch.ones(num_trackers))
        self.in_scale = nn.Parameter(torch.ones(num_trackers))
        self.out_scale = nn.Parameter(torch.ones(num_trackers))
        self.tld_scale = nn.Parameter(torch.ones(num_trackers))
        self.diffusion_scale = self.in_scale

        # Multi-Source Gated Prior Fusion module: decoupled gating networks
        self.direct_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_trackers),
            nn.Sigmoid(),
        )
        self.in_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_trackers),
            nn.Sigmoid(),
        )
        self.out_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_trackers),
            nn.Sigmoid(),
        )
        self.tld_gate = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_trackers),
            nn.Sigmoid(),
        )
        self.prior_gate = self.in_gate
        self.diffusion_gate = self.in_gate

        # Multi-Label Co-occurrence Transition Layer (ML-CTL)
        if transition_matrix is not None:
            self.transition_matrix = nn.Parameter(
                torch.as_tensor(transition_matrix, dtype=torch.float32)
            )
        else:
            self.transition_matrix = nn.Parameter(
                torch.zeros(num_trackers, num_trackers)
            )
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        x: torch.Tensor,
        in_diff: torch.Tensor = None,
        out_diff: torch.Tensor = None,
        direct_links: torch.Tensor = None,
        tld_prior: torch.Tensor = None,
        diffusion_prior: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h_domain = self.domain_norm(h)

        tracker_reps = self.tracker_module()
        logits = (
            torch.matmul(h_domain, tracker_reps.t()) / math.sqrt(self.embed_dim)
            + self.tracker_bias
        )

        # Multi-Source Gated Prior Fusion: decoupled gating networks and learnable scaling vectors
        if direct_links is not None:
            logits = logits + self.direct_gate(h_domain) * (direct_links * self.direct_scale)
        if in_diff is not None:
            logits = logits + self.in_gate(h_domain) * (in_diff * self.in_scale)
        elif diffusion_prior is not None:
            logits = logits + self.in_gate(h_domain) * (diffusion_prior * self.in_scale)
        if out_diff is not None:
            logits = logits + self.out_gate(h_domain) * (out_diff * self.out_scale)
        if tld_prior is not None:
            logits = logits + self.tld_gate(h_domain) * (tld_prior * self.tld_scale)

        # Multi-Label Co-occurrence Transition Layer residual refinement
        if hasattr(self, "transition_matrix") and self.transition_matrix is not None:
            logits = logits + self.alpha * F.linear(
                torch.sigmoid(logits), self.transition_matrix
            )

        return logits


class SoftRank10BoundaryMarginLoss(nn.Module):
    """Soft Rank-10 Boundary Margin Ranking Loss combined with Asymmetric Focal Loss.

    Computes the smooth rank of each true tracker against negatives and penalizes predictions
    where the estimated rank exceeds the top-10 threshold.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        asl_margin: float = 0.05,
        ranking_weight: float = 0.5,
        tau: float = 1.0,
        margin: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.asl_margin = asl_margin
        self.ranking_weight = ranking_weight
        self.tau = tau
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # 1. Asymmetric Focal Cross-Entropy
        pos_probs = probs.clamp(min=1e-7, max=1.0 - 1e-7)
        pos_loss = (
            -targets * torch.pow(1.0 - pos_probs, self.gamma_pos) * torch.log(pos_probs)
        )

        neg_probs = (probs - self.asl_margin).clamp(min=0.0, max=1.0 - 1e-7)
        neg_loss = (
            -(1.0 - targets)
            * torch.pow(neg_probs, self.gamma_neg)
            * torch.log((1.0 - neg_probs).clamp(min=1e-7))
        )
        asl_loss = (pos_loss + neg_loss).sum(dim=-1).mean()

        if self.ranking_weight <= 0:
            return asl_loss

        # 2. Soft Rank-10 Boundary Margin Ranking Loss
        pos_mask = targets >= 0.5
        neg_mask = ~pos_mask

        # diff[b, p, n] = (logits[b, n] - logits[b, p]) / tau
        # s_neg has shape (B, 1, K), s_pos has shape (B, K, 1)
        diff = (logits.unsqueeze(1) - logits.unsqueeze(2)) / self.tau
        sigmoid_diff = torch.sigmoid(diff)

        # soft_rank[b, p] = sum_{n in negatives} sigmoid((logits[b, n] - logits[b, p]) / tau)
        soft_rank = (sigmoid_diff * neg_mask.unsqueeze(1).float()).sum(dim=2)

        # Penalize predictions where soft negative rank exceeds top-10 threshold (10.0 - margin)
        rank_penalty = F.softplus(soft_rank - (10.0 - self.margin)) / self.tau

        num_positives = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sample_rank_loss = (rank_penalty * pos_mask.float()).sum(
            dim=-1, keepdim=True
        ) / num_positives

        has_pos = (pos_mask.sum(dim=-1) > 0).float()
        rank_loss = (sample_rank_loss.squeeze(-1) * has_pos).sum() / (
            has_pos.sum() + 1e-7
        )

        total_loss = asl_loss + self.ranking_weight * rank_loss
        return total_loss


AsymmetricRecallRankingLoss = SoftRank10BoundaryMarginLoss


def compute_recall_at_10(logits, targets) -> float:
    """Official competition evaluation metric: Recall@10.

    For each domain, the fraction of its true trackers that appear anywhere in the top 10
    predicted trackers, averaged across all domains.
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    top10_preds = np.argpartition(logits, -10, axis=-1)[:, -10:]

    if sp.issparse(targets):
        targets_csr = targets.tocsr()
        indptr = targets_csr.indptr
        indices = targets_csr.indices
        recalls = []
        for i in range(len(top10_preds)):
            true_trackers = set(indices[indptr[i] : indptr[i + 1]])
            if len(true_trackers) == 0:
                continue
            hits = len(set(top10_preds[i]).intersection(true_trackers))
            recalls.append(hits / len(true_trackers))
        return float(np.mean(recalls)) if recalls else 0.0

    recalls = []
    for pred_row, true_row in zip(top10_preds, targets):
        if sp.issparse(true_row):
            true_trackers = set(true_row.indices)
        elif isinstance(true_row, np.ndarray):
            true_trackers = set(np.where(true_row > 0.5)[0])
        else:
            true_trackers = set(true_row)

        if len(true_trackers) == 0:
            continue
        hits = len(set(pred_row).intersection(true_trackers))
        recalls.append(hits / len(true_trackers))

    return float(np.mean(recalls)) if recalls else 0.0


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    total_steps: int = 1000,
    pct_start: float = 0.1,
):
    """Constructs AdamW optimizer with decoupled weight decay and OneCycleLR schedule."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            "bias" in name
            or "norm" in name
            or "tracker_bias" in name
            or "scale" in name
            or "alpha" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.98), eps=1e-6
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max(1, total_steps),
        pct_start=pct_start,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e2,
    )

    return optimizer, scheduler


# =========================================================================
# 3. Main End-to-End Pipeline Execution
# =========================================================================
def main():
    print("Starting data processing and feature engineering...")
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_trackers = 355

    # 1. Load Trackers metadata
    print("Loading tracker metadata...")
    trackers_df = pd.read_csv("./input/trackers.tsv", sep="\t")
    tracker_to_domain_map = dict(
        zip(trackers_df["tracker_id"], trackers_df["tracking_domain_id"])
    )
    tracking_domain_to_tracker = dict(
        zip(trackers_df["tracking_domain_id"], trackers_df["tracker_id"])
    )
    trackers_df.to_parquet(
        "./working/tracker_metadata.parquet", index=False, engine="pyarrow"
    )

    trackers_sorted = trackers_df.sort_values("tracker_id").reset_index(drop=True)
    comp_codes, comp_uniques = pd.factorize(trackers_sorted["company"])
    cat_codes, cat_uniques = pd.factorize(trackers_sorted["category"])
    country_codes, country_uniques = pd.factorize(trackers_sorted["country"])
    brand_codes, brand_uniques = pd.factorize(trackers_sorted["brand"])

    tracker_metadata = {
        "company_ids": torch.tensor(comp_codes, dtype=torch.long),
        "category_ids": torch.tensor(cat_codes, dtype=torch.long),
        "country_ids": torch.tensor(country_codes, dtype=torch.long),
        "brand_ids": torch.tensor(brand_codes, dtype=torch.long),
        "num_companies": len(comp_uniques),
        "num_categories": len(cat_uniques),
        "num_countries": len(country_uniques),
        "num_brands": len(brand_uniques),
    }

    # 2. Load Target domains (Test set)
    print("Loading test target domains...")
    target_df = pd.read_csv("./input/target.tsv", sep="\t")
    test_domain_ids = target_df["domain_id"].values.astype(np.int64)
    test_domain_set = set(test_domain_ids)
    print(f"Total test domains: {len(test_domain_ids)}")

    # 3. Load tracking graph (Train set candidate pool)
    print("Loading tracking_graph_train.parquet...")
    train_graph_df = pd.read_parquet("./input/tracking_graph_train.parquet")
    all_train_domains = train_graph_df["domain_id"].unique().astype(np.int64)

    # Strict isolation: ensure no overlap with test set
    candidate_train_domains = np.array(
        [d for d in all_train_domains if d not in test_domain_set]
    )
    np.random.shuffle(candidate_train_domains)

    # Split into train pool and validation pool
    val_size = min(50000, int(len(candidate_train_domains) * 0.15))
    val_domain_ids = candidate_train_domains[:val_size]
    train_domain_ids_pool = candidate_train_domains[val_size:]
    print(
        f"Validation domain count: {len(val_domain_ids)}, Train pool candidate"
        f" count: {len(train_domain_ids_pool)}"
    )

    val_domain_set = set(val_domain_ids)
    train_domain_pool_set = set(train_domain_ids_pool)

    # Build ground truth tracking dict strictly for train pool and val pool
    train_labels_map = collections.defaultdict(list)
    val_labels_map = collections.defaultdict(list)

    for row in train_graph_df.itertuples(index=False):
        d_id = row.domain_id
        t_id = row.tracker_id
        if d_id in train_domain_pool_set:
            train_labels_map[d_id].append(t_id)
        elif d_id in val_domain_set:
            val_labels_map[d_id].append(t_id)

    del train_graph_df
    gc.collect()

    # Subsample balanced training set of 500,000 domains
    max_train_samples = min(500000, len(train_domain_ids_pool))
    selected_train_domains = set()

    # Tracker-stratified sampling: ensure coverage across trackers
    domains_by_tracker = collections.defaultdict(list)
    for d_id, t_list in train_labels_map.items():
        for t_id in t_list:
            domains_by_tracker[t_id].append(d_id)

    for t_id in range(num_trackers):
        cand = domains_by_tracker.get(t_id, [])
        if cand:
            chosen = np.random.choice(cand, size=min(len(cand), 800), replace=False)
            selected_train_domains.update(chosen)

    remaining_quota = max_train_samples - len(selected_train_domains)
    if remaining_quota > 0:
        remaining_pool = [
            d for d in train_domain_ids_pool if d not in selected_train_domains
        ]
        chosen_rem = np.random.choice(
            remaining_pool,
            size=min(len(remaining_pool), remaining_quota),
            replace=False,
        )
        selected_train_domains.update(chosen_rem)

    train_domain_ids = np.array(list(selected_train_domains), dtype=np.int64)
    print(f"Final sampled training domains: {len(train_domain_ids)}")

    # 4. Load Domain Strings from domains.parquet
    print("Loading domain hostnames...")
    needed_domains = set(train_domain_ids) | set(val_domain_ids) | set(test_domain_ids)
    domains_pq = pq.read_table(
        "./input/domains.parquet", columns=["domain_id", "domain"]
    )
    domains_df = domains_pq.to_pandas()
    del domains_pq
    gc.collect()

    domains_df = domains_df[domains_df["domain_id"].isin(needed_domains)]
    domain_to_str = dict(zip(domains_df["domain_id"], domains_df["domain"]))
    del domains_df
    gc.collect()

    # 5. Load Press Freedom metadata & join by TLD
    print("Processing press freedom data...")
    fop_map = {}
    try:
        fop_df = pd.read_csv("./input/freedom-of-the-press.csv", sep="\t")
    except Exception:
        fop_df = pd.read_csv("./input/freedom-of-the-press.csv")
    fop_df.columns = [c.strip().lower() for c in fop_df.columns]
    for row in fop_df.itertuples(index=False):
        tld_val = str(getattr(row, "tld", "")).strip().lstrip(".").lower()
        score_val = getattr(row, "freedom_of_the_press", np.nan)
        try:
            fop_map[tld_val] = float(score_val)
        except (ValueError, TypeError):
            continue

    # 6. Load URL Categorization data with two-tier hierarchical domain lookup
    print("Processing URL category fingerprints with hierarchical domain matching...")
    url_cat_df = pd.read_csv("./input/url-classification.csv")
    extracted_domains = (
        url_cat_df["url"]
        .str.extract(r"https?://(?:www\.)?([^/:\?]+)", expand=False)
        .str.lower()
    )
    url_cat_df["clean_domain"] = extracted_domains
    valid_urls = url_cat_df.dropna(subset=["clean_domain", "category"]).copy()
    valid_urls["reg_domain"] = valid_urls["clean_domain"].apply(extract_registered_domain)

    # Tier 1: Exact hostname category distribution
    cat_counts_exact = (
        valid_urls.groupby(["clean_domain", "category"]).size().unstack(fill_value=0)
    )
    url_categories = sorted(list(cat_counts_exact.columns))
    cat_dist_exact = cat_counts_exact.div(cat_counts_exact.sum(axis=1), axis=0).to_dict(orient="index")

    # Tier 2: Registered domain (SLD+TLD) category distribution
    cat_counts_reg = (
        valid_urls.groupby(["reg_domain", "category"]).size().unstack(fill_value=0)
    )
    for c in url_categories:
        if c not in cat_counts_reg.columns:
            cat_counts_reg[c] = 0
    cat_counts_reg = cat_counts_reg[url_categories]
    cat_dist_reg = cat_counts_reg.div(cat_counts_reg.sum(axis=1), axis=0).to_dict(orient="index")

    del url_cat_df, valid_urls, cat_counts_exact, cat_counts_reg
    gc.collect()

    # 7. Extract Lexical, TLD, Press Freedom, and URL Category Features
    def extract_base_features(domain_id_list):
        lexical_feats = []
        fop_feats = []
        cat_feats = []
        hostnames = []
        tlds = []

        for d_id in domain_id_list:
            hostname = domain_to_str.get(d_id, "")
            hostnames.append(hostname)
            lexical_feats.append(compute_lexical_features(hostname))

            tld, sld = extract_tld_and_sld(hostname)
            tlds.append(tld)
            if tld in fop_map:
                fop_feats.append([fop_map[tld], 1.0])
            else:
                fop_feats.append([np.nan, 0.0])

            # Hierarchical category lookup: exact hostname -> registered SLD.TLD domain
            clean_h = hostname.lower().strip()
            if clean_h.startswith("www."):
                clean_h = clean_h[4:]

            cat_dict = None
            if hostname in cat_dist_exact:
                cat_dict = cat_dist_exact[hostname]
            elif clean_h in cat_dist_exact:
                cat_dict = cat_dist_exact[clean_h]
            else:
                reg_d = f"{sld}.{tld}"
                if reg_d in cat_dist_reg:
                    cat_dict = cat_dist_reg[reg_d]

            if cat_dict is not None:
                cat_vec = [cat_dict.get(c, 0.0) for c in url_categories]
                cat_vec.append(1.0)
            else:
                cat_vec = [0.0] * len(url_categories) + [0.0]
            cat_feats.append(cat_vec)

        return (
            np.array(lexical_feats, dtype=np.float32),
            np.array(fop_feats, dtype=np.float32),
            np.array(cat_feats, dtype=np.float32),
            hostnames,
            tlds,
        )

    print("Extracting domain base features...")
    train_lex, train_fop, train_cat, train_hosts, train_tlds = extract_base_features(
        train_domain_ids
    )
    val_lex, val_fop, val_cat, val_hosts, val_tlds = extract_base_features(
        val_domain_ids
    )
    test_lex, test_fop, test_cat, test_hosts, test_tlds = extract_base_features(
        test_domain_ids
    )

    valid_train_fop = train_fop[~np.isnan(train_fop[:, 0]), 0]
    fop_median = float(np.median(valid_train_fop)) if len(valid_train_fop) > 0 else 50.0
    fop_scaler = StandardScaler()
    train_fop[np.isnan(train_fop[:, 0]), 0] = fop_median
    val_fop[np.isnan(val_fop[:, 0]), 0] = fop_median
    test_fop[np.isnan(test_fop[:, 0]), 0] = fop_median

    train_fop[:, 0] = fop_scaler.fit_transform(train_fop[:, [0]]).flatten()
    val_fop[:, 0] = fop_scaler.transform(val_fop[:, [0]]).flatten()
    test_fop[:, 0] = fop_scaler.transform(test_fop[:, [0]]).flatten()

    # 8. Character N-Gram TF-IDF + SVD on Hostnames
    print("Computing subword character n-gram embeddings...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=2500,
        min_df=5,
        sublinear_tf=True,
    )
    train_tfidf = tfidf.fit_transform(train_hosts)
    val_tfidf = tfidf.transform(val_hosts)
    test_tfidf = tfidf.transform(test_hosts)

    svd_char = TruncatedSVD(n_components=24, random_state=42)
    train_char_svd = svd_char.fit_transform(train_tfidf).astype(np.float32)
    val_char_svd = svd_char.transform(val_tfidf).astype(np.float32)
    test_char_svd = svd_char.transform(test_tfidf).astype(np.float32)
    del train_tfidf, val_tfidf, test_tfidf
    gc.collect()

    # 9. Stream link-graph.parquet to extract Degrees, Direct Tracker Links & Directional Diffusion
    print("Streaming link-graph.parquet for graph topology, directional diffusion & direct links...")
    in_degrees = collections.Counter()
    out_degrees = collections.Counter()

    train_tracker_lookup = {
        int(d_id): train_labels_map[d_id]
        for d_id in train_domain_ids_pool
        if d_id in train_labels_map
    }

    domain_to_idx = {int(d): ("train", i) for i, d in enumerate(train_domain_ids)}
    domain_to_idx.update({int(d): ("val", i) for i, d in enumerate(val_domain_ids)})
    domain_to_idx.update({int(d): ("test", i) for i, d in enumerate(test_domain_ids)})

    # Directional diffusion matrices: inbound and outbound
    train_in_neigh_tracker = np.zeros(
        (len(train_domain_ids), num_trackers), dtype=np.float32
    )
    val_in_neigh_tracker = np.zeros(
        (len(val_domain_ids), num_trackers), dtype=np.float32
    )
    test_in_neigh_tracker = np.zeros(
        (len(test_domain_ids), num_trackers), dtype=np.float32
    )

    train_out_neigh_tracker = np.zeros(
        (len(train_domain_ids), num_trackers), dtype=np.float32
    )
    val_out_neigh_tracker = np.zeros(
        (len(val_domain_ids), num_trackers), dtype=np.float32
    )
    test_out_neigh_tracker = np.zeros(
        (len(test_domain_ids), num_trackers), dtype=np.float32
    )

    train_in_neigh_count = np.zeros(len(train_domain_ids), dtype=np.float32)
    val_in_neigh_count = np.zeros(len(val_domain_ids), dtype=np.float32)
    test_in_neigh_count = np.zeros(len(test_domain_ids), dtype=np.float32)

    train_out_neigh_count = np.zeros(len(train_domain_ids), dtype=np.float32)
    val_out_neigh_count = np.zeros(len(val_domain_ids), dtype=np.float32)
    test_out_neigh_count = np.zeros(len(test_domain_ids), dtype=np.float32)

    # Direct hyperlinks pointing to candidate tracker domain IDs
    train_direct_links = np.zeros(
        (len(train_domain_ids), num_trackers), dtype=np.float32
    )
    val_direct_links = np.zeros(
        (len(val_domain_ids), num_trackers), dtype=np.float32
    )
    test_direct_links = np.zeros(
        (len(test_domain_ids), num_trackers), dtype=np.float32
    )

    pfile = pq.ParquetFile("./input/link-graph.parquet")
    batch_counter = 0

    for batch in pfile.iter_batches(
        batch_size=2500000, columns=["source_domain_id", "target_domain_id"]
    ):
        batch_counter += 1
        src_arr = batch["source_domain_id"].to_pylist()
        tgt_arr = batch["target_domain_id"].to_pylist()

        for s, t in zip(src_arr, tgt_arr):
            s_in_needed = s in domain_to_idx
            t_in_needed = t in domain_to_idx

            if s_in_needed:
                out_degrees[s] += 1
            if t_in_needed:
                in_degrees[t] += 1

            # (1) Direct hyperlinks pointing to candidate tracker domain IDs
            if s_in_needed and t in tracking_domain_to_tracker:
                trk_id = tracking_domain_to_tracker[t]
                split_name, idx = domain_to_idx[s]
                if split_name == "train":
                    train_direct_links[idx, trk_id] = 1.0
                elif split_name == "val":
                    val_direct_links[idx, trk_id] = 1.0
                else:
                    test_direct_links[idx, trk_id] = 1.0

            # (2) Directional neighbor tracker diffusion
            # Inbound: s -> t (s links to t, trackers on s diffuse to t as in-neighbors)
            if s in train_tracker_lookup and t_in_needed:
                split_name, idx = domain_to_idx[t]
                if not (split_name == "train" and train_domain_ids[idx] == s):
                    if split_name == "train":
                        t_mat, t_cnt = train_in_neigh_tracker, train_in_neigh_count
                    elif split_name == "val":
                        t_mat, t_cnt = val_in_neigh_tracker, val_in_neigh_count
                    else:
                        t_mat, t_cnt = test_in_neigh_tracker, test_in_neigh_count

                    for trk in train_tracker_lookup[s]:
                        t_mat[idx, trk] += 1.0
                    t_cnt[idx] += 1.0

            # Outbound: s -> t (s links to t, trackers on t diffuse to s as out-neighbors)
            if t in train_tracker_lookup and s_in_needed:
                split_name, idx = domain_to_idx[s]
                if not (split_name == "train" and train_domain_ids[idx] == t):
                    if split_name == "train":
                        s_mat, s_cnt = train_out_neigh_tracker, train_out_neigh_count
                    elif split_name == "val":
                        s_mat, s_cnt = val_out_neigh_tracker, val_out_neigh_count
                    else:
                        s_mat, s_cnt = test_out_neigh_tracker, test_out_neigh_count

                    for trk in train_tracker_lookup[t]:
                        s_mat[idx, trk] += 1.0
                    s_cnt[idx] += 1.0

    print(f"Graph streaming completed. Processed {batch_counter} link-graph batches.")

    # 10. Assemble Graph Topological Features
    def extract_graph_features(domain_id_list):
        feats = []
        for d_id in domain_id_list:
            ind = in_degrees.get(int(d_id), 0)
            outd = out_degrees.get(int(d_id), 0)
            totd = ind + outd
            log_ind = math.log1p(ind)
            log_outd = math.log1p(outd)
            log_totd = math.log1p(totd)
            deg_ratio = log_ind / (log_outd + 1.0)
            feats.append(
                [
                    float(ind),
                    float(outd),
                    float(totd),
                    log_ind,
                    log_outd,
                    log_totd,
                    deg_ratio,
                ]
            )
        return np.array(feats, dtype=np.float32)

    train_graph_feats = extract_graph_features(train_domain_ids)
    val_graph_feats = extract_graph_features(val_domain_ids)
    test_graph_feats = extract_graph_features(test_domain_ids)

    graph_scaler = StandardScaler()
    train_graph_feats = graph_scaler.fit_transform(train_graph_feats)
    val_graph_feats = graph_scaler.transform(val_graph_feats)
    test_graph_feats = graph_scaler.transform(test_graph_feats)

    # Precompute global tracker frequencies, co-occurrence, ITF weights & ML-CTL transition matrix
    print("Computing global tracker frequencies, co-occurrence matrix & ITF weights...")
    global_tracker_counts = np.zeros(num_trackers, dtype=np.float32)
    tracker_cooccur = np.zeros((num_trackers, num_trackers), dtype=np.float32)

    for d_id in train_domain_ids:
        t_list = train_labels_map.get(d_id, [])
        for t in t_list:
            global_tracker_counts[t] += 1.0
        for t1 in t_list:
            for t2 in t_list:
                tracker_cooccur[t1, t2] += 1.0

    global_tracker_priors = global_tracker_counts / max(1.0, len(train_domain_ids))
    itf_weights = np.log(
        1.0 + len(train_domain_ids) / (global_tracker_counts + 1.0)
    ).astype(np.float32)

    transition_matrix = np.zeros((num_trackers, num_trackers), dtype=np.float32)
    for k in range(num_trackers):
        if global_tracker_counts[k] > 0:
            transition_matrix[:, k] = tracker_cooccur[:, k] / global_tracker_counts[k]
    np.fill_diagonal(transition_matrix, 0.0)

    # 11. Normalize Directional Neighbor Tracker Diffusion & Compute SVD
    def process_directional_diffusion(
        neigh_tracker_mat, neigh_count_vec, itf_vec=None, svd_model=None, n_components=24
    ):
        norm_mat = np.zeros_like(neigh_tracker_mat)
        mask = neigh_count_vec > 0
        norm_mat[mask] = neigh_tracker_mat[mask] / neigh_count_vec[mask, None]

        log_cnt = np.log1p(neigh_count_vec)[:, None]
        top1_prob = np.max(norm_mat, axis=1)[:, None]
        sorted_probs = np.sort(norm_mat, axis=1)[:, ::-1]
        top3_prob = np.sum(sorted_probs[:, :3], axis=1)[:, None]
        nonzero_count = np.sum(norm_mat > 0, axis=1)[:, None]

        p_safe = np.where(norm_mat > 0, norm_mat, 1.0)
        entropy = -np.sum(norm_mat * np.log2(p_safe), axis=1)[:, None]

        summary_feats = np.hstack(
            [
                neigh_count_vec[:, None],
                log_cnt,
                top1_prob,
                top3_prob,
                nonzero_count,
                entropy,
            ]
        ).astype(np.float32)

        if itf_vec is not None:
            norm_mat = norm_mat * itf_vec

        if svd_model is None:
            svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            svd_feats = svd_model.fit_transform(norm_mat).astype(np.float32)
            return summary_feats, svd_feats, norm_mat, svd_model
        else:
            svd_feats = svd_model.transform(norm_mat).astype(np.float32)
            return summary_feats, svd_feats, norm_mat, svd_model

    print("Computing directional neighbor tracker diffusion embeddings...")
    train_in_summary, train_in_svd, train_in_norm, svd_in = (
        process_directional_diffusion(
            train_in_neigh_tracker,
            train_in_neigh_count,
            itf_vec=itf_weights,
            svd_model=None,
            n_components=24,
        )
    )
    val_in_summary, val_in_svd, val_in_norm, _ = process_directional_diffusion(
        val_in_neigh_tracker,
        val_in_neigh_count,
        itf_vec=itf_weights,
        svd_model=svd_in,
        n_components=24,
    )
    test_in_summary, test_in_svd, test_in_norm, _ = process_directional_diffusion(
        test_in_neigh_tracker,
        test_in_neigh_count,
        itf_vec=itf_weights,
        svd_model=svd_in,
        n_components=24,
    )

    train_out_summary, train_out_svd, train_out_norm, svd_out = (
        process_directional_diffusion(
            train_out_neigh_tracker,
            train_out_neigh_count,
            itf_vec=itf_weights,
            svd_model=None,
            n_components=24,
        )
    )
    val_out_summary, val_out_svd, val_out_norm, _ = process_directional_diffusion(
        val_out_neigh_tracker,
        val_out_neigh_count,
        itf_vec=itf_weights,
        svd_model=svd_out,
        n_components=24,
    )
    test_out_summary, test_out_svd, test_out_norm, _ = process_directional_diffusion(
        test_out_neigh_tracker,
        test_out_neigh_count,
        itf_vec=itf_weights,
        svd_model=svd_out,
        n_components=24,
    )

    def compute_cross_directional_stats(in_summary, out_summary, direct_mat):
        in_cnt = in_summary[:, 0:1]
        out_cnt = out_summary[:, 0:1]
        diff_ratio = (in_cnt + 1.0) / (out_cnt + 1.0)
        in_nz = in_summary[:, 4:5]
        out_nz = out_summary[:, 4:5]
        nz_ratio = (in_nz + 1.0) / (out_nz + 1.0)
        direct_cnt = np.sum(direct_mat > 0, axis=1, keepdims=True).astype(np.float32)
        return np.hstack([diff_ratio, nz_ratio, direct_cnt]).astype(np.float32)

    train_cross_stats = compute_cross_directional_stats(
        train_in_summary, train_out_summary, train_direct_links
    )
    val_cross_stats = compute_cross_directional_stats(
        val_in_summary, val_out_summary, val_direct_links
    )
    test_cross_stats = compute_cross_directional_stats(
        test_in_summary, test_out_summary, test_direct_links
    )

    cross_scaler = StandardScaler()
    train_cross_stats = cross_scaler.fit_transform(train_cross_stats)
    val_cross_stats = cross_scaler.transform(val_cross_stats)
    test_cross_stats = cross_scaler.transform(test_cross_stats)

    # 12. Calculate Full 355-dim Dirichlet-Smoothed TLD Conditional Priors
    print("Computing full 355-dim Dirichlet-smoothed TLD conditional priors...")
    tld_tracker_counts = collections.defaultdict(
        lambda: np.zeros(num_trackers, dtype=np.float32)
    )
    tld_domain_counts = collections.Counter()

    for d_id, tld in zip(train_domain_ids, train_tlds):
        t_list = train_labels_map.get(d_id, [])
        for t in t_list:
            tld_tracker_counts[tld][t] += 1.0
        tld_domain_counts[tld] += 1.0

    alpha_dirichlet = 20.0
    smoothed_tld_priors = {}
    for tld, count in tld_domain_counts.items():
        smoothed_tld_priors[tld] = (
            (tld_tracker_counts[tld] + alpha_dirichlet * global_tracker_priors)
            / (count + alpha_dirichlet)
        ).astype(np.float32)

    def extract_full_tld_priors(tld_list):
        priors = np.zeros((len(tld_list), num_trackers), dtype=np.float32)
        for i, tld in enumerate(tld_list):
            if tld in smoothed_tld_priors:
                priors[i] = smoothed_tld_priors[tld]
            else:
                priors[i] = global_tracker_priors
        return priors

    tld_prior_train = extract_full_tld_priors(train_tlds)
    tld_prior_val = extract_full_tld_priors(val_tlds)
    tld_prior_test = extract_full_tld_priors(test_tlds)

    top_16_trackers = np.argsort(global_tracker_priors)[-16:]
    train_tld_prior_feats = tld_prior_train[:, top_16_trackers]
    val_tld_prior_feats = tld_prior_val[:, top_16_trackers]
    test_tld_prior_feats = tld_prior_test[:, top_16_trackers]

    # 13. Scale Lexical and URL Category features
    lex_scaler = StandardScaler()
    train_lex = lex_scaler.fit_transform(train_lex)
    val_lex = lex_scaler.transform(val_lex)
    test_lex = lex_scaler.transform(test_lex)

    cat_scaler = StandardScaler()
    train_cat = cat_scaler.fit_transform(train_cat)
    val_cat = cat_scaler.transform(val_cat)
    test_cat = cat_scaler.transform(test_cat)

    # 14. Concatenate Final Dense Feature Matrices
    print("Assembling final feature matrices...")
    X_train = np.hstack(
        [
            train_lex,
            train_fop,
            train_cat,
            train_char_svd,
            train_graph_feats,
            train_in_summary,
            train_in_svd,
            train_out_summary,
            train_out_svd,
            train_cross_stats,
            train_tld_prior_feats,
        ]
    ).astype(np.float32)

    X_val = np.hstack(
        [
            val_lex,
            val_fop,
            val_cat,
            val_char_svd,
            val_graph_feats,
            val_in_summary,
            val_in_svd,
            val_out_summary,
            val_out_svd,
            val_cross_stats,
            val_tld_prior_feats,
        ]
    ).astype(np.float32)

    X_test = np.hstack(
        [
            test_lex,
            test_fop,
            test_cat,
            test_char_svd,
            test_graph_feats,
            test_in_summary,
            test_in_svd,
            test_out_summary,
            test_out_svd,
            test_cross_stats,
            test_tld_prior_feats,
        ]
    ).astype(np.float32)

    print(
        f"X_train shape: {X_train.shape} | X_val shape: {X_val.shape} | X_test shape: {X_test.shape}"
    )

    # 15. Create Multi-hot Sparse Target Labels for Train and Validation
    def build_sparse_targets(domain_id_list, labels_map):
        rows, cols = [], []
        for row_idx, d_id in enumerate(domain_id_list):
            for t_id in labels_map.get(d_id, []):
                rows.append(row_idx)
                cols.append(t_id)
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(domain_id_list), num_trackers),
            dtype=np.float32,
        )

    Y_train = build_sparse_targets(train_domain_ids, train_labels_map)
    Y_val = build_sparse_targets(val_domain_ids, val_labels_map)

    diff_in_train = sp.csr_matrix(train_in_norm)
    diff_in_val = sp.csr_matrix(val_in_norm)
    diff_in_test = sp.csr_matrix(test_in_norm)

    diff_out_train = sp.csr_matrix(train_out_norm)
    diff_out_val = sp.csr_matrix(val_out_norm)
    diff_out_test = sp.csr_matrix(test_out_norm)

    direct_train = sp.csr_matrix(train_direct_links)
    direct_val = sp.csr_matrix(val_direct_links)
    direct_test = sp.csr_matrix(test_direct_links)

    # Save artifacts for auditable validation
    np.savez_compressed(
        "./working/features_dense.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
    )
    sp.save_npz("./working/Y_train.npz", Y_train)
    sp.save_npz("./working/Y_val.npz", Y_val)
    sp.save_npz("./working/diff_in_train.npz", diff_in_train)
    sp.save_npz("./working/diff_in_val.npz", diff_in_val)
    sp.save_npz("./working/diff_in_test.npz", diff_in_test)
    sp.save_npz("./working/diff_out_train.npz", diff_out_train)
    sp.save_npz("./working/diff_out_val.npz", diff_out_val)
    sp.save_npz("./working/diff_out_test.npz", diff_out_test)
    sp.save_npz("./working/direct_train.npz", direct_train)
    sp.save_npz("./working/direct_val.npz", direct_val)
    sp.save_npz("./working/direct_test.npz", direct_test)
    np.save("./working/tld_prior_train.npy", tld_prior_train)
    np.save("./working/tld_prior_val.npy", tld_prior_val)
    np.save("./working/tld_prior_test.npy", tld_prior_test)
    np.save("./working/test_domain_ids.npy", test_domain_ids)

    # 16. Instantiate Model, Loss & Optimizer
    in_features = X_train.shape[1]
    model = GBTrackerNet(
        in_features=in_features,
        num_trackers=num_trackers,
        embed_dim=256,
        hidden_dim=512,
        num_blocks=3,
        dropout=0.15,
        tracker_metadata=tracker_metadata,
        transition_matrix=transition_matrix,
    ).to(device)

    criterion = SoftRank10BoundaryMarginLoss(
        gamma_pos=1.0,
        gamma_neg=4.0,
        asl_margin=0.05,
        ranking_weight=0.5,
        tau=1.0,
        margin=0.0,
    ).to(device)

    batch_size = 1024
    num_epochs = 12
    eval_batch_size = 2048
    num_train = X_train.shape[0]
    steps_per_epoch = (num_train + batch_size - 1) // batch_size
    total_steps = num_epochs * steps_per_epoch

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=1.2e-3,
        weight_decay=1e-4,
        total_steps=total_steps,
        pct_start=0.1,
    )

    best_val_recall = -1.0
    best_model_path = "./working/best_gb_trackernet.pt"
    patience = 4
    patience_counter = 0

    print("Beginning model training and validation loop...")
    # 17. Main Training & Validation Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        perm = np.random.permutation(num_train)

        for step in range(steps_per_epoch):
            batch_indices = perm[step * batch_size : (step + 1) * batch_size]
            bx = torch.from_numpy(X_train[batch_indices]).to(device)
            bin_train = torch.from_numpy(
                diff_in_train[batch_indices].toarray()
            ).to(device)
            bout_train = torch.from_numpy(
                diff_out_train[batch_indices].toarray()
            ).to(device)
            bdir_train = torch.from_numpy(
                direct_train[batch_indices].toarray()
            ).to(device)
            btld_train = torch.from_numpy(
                tld_prior_train[batch_indices]
            ).to(device)
            by = torch.from_numpy(Y_train[batch_indices].toarray()).to(device)

            optimizer.zero_grad()
            logits = model(
                bx,
                in_diff=bin_train,
                out_diff=bout_train,
                direct_links=bdir_train,
                tld_prior=btld_train,
            )
            loss = criterion(logits, by)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * len(batch_indices)

        train_loss = running_loss / num_train

        # Validation Evaluation Pass
        model.eval()
        val_logits_list = []
        num_val = X_val.shape[0]
        val_steps = (num_val + eval_batch_size - 1) // eval_batch_size

        with torch.no_grad():
            for v_step in range(val_steps):
                v_start = v_step * eval_batch_size
                v_end = min((v_step + 1) * eval_batch_size, num_val)
                bx_val = torch.from_numpy(X_val[v_start:v_end]).to(device)
                bin_val = torch.from_numpy(
                    diff_in_val[v_start:v_end].toarray()
                ).to(device)
                bout_val = torch.from_numpy(
                    diff_out_val[v_start:v_end].toarray()
                ).to(device)
                bdir_val = torch.from_numpy(
                    direct_val[v_start:v_end].toarray()
                ).to(device)
                btld_val = torch.from_numpy(
                    tld_prior_val[v_start:v_end]
                ).to(device)
                batch_val_logits = model(
                    bx_val,
                    in_diff=bin_val,
                    out_diff=bout_val,
                    direct_links=bdir_val,
                    tld_prior=btld_val,
                )
                val_logits_list.append(batch_val_logits.cpu().numpy())

        val_logits = np.vstack(val_logits_list)
        val_recall = compute_recall_at_10(val_logits, Y_val)

        print(
            f"Epoch {epoch + 1:02d}/{num_epochs:02d} - Loss: {train_loss:.4f} - Val"
            f" Recall@10: {val_recall:.4f}"
        )

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 18. Load Best Checkpoint & Final Validation Score
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    val_logits_list = []
    with torch.no_grad():
        for v_step in range(val_steps):
            v_start = v_step * eval_batch_size
            v_end = min((v_step + 1) * eval_batch_size, num_val)
            bx_val = torch.from_numpy(X_val[v_start:v_end]).to(device)
            bin_val = torch.from_numpy(
                diff_in_val[v_start:v_end].toarray()
            ).to(device)
            bout_val = torch.from_numpy(
                diff_out_val[v_start:v_end].toarray()
            ).to(device)
            bdir_val = torch.from_numpy(
                direct_val[v_start:v_end].toarray()
            ).to(device)
            btld_val = torch.from_numpy(
                tld_prior_val[v_start:v_end]
            ).to(device)
            val_logits_list.append(
                model(
                    bx_val,
                    in_diff=bin_val,
                    out_diff=bout_val,
                    direct_links=bdir_val,
                    tld_prior=btld_val,
                ).cpu().numpy()
            )

    final_val_logits = np.vstack(val_logits_list)
    final_val_score = compute_recall_at_10(final_val_logits, Y_val)

    # 19. Full Test Inference
    test_logits_list = []
    num_test = X_test.shape[0]
    test_steps = (num_test + eval_batch_size - 1) // eval_batch_size

    with torch.no_grad():
        for t_step in range(test_steps):
            t_start = t_step * eval_batch_size
            t_end = min((t_step + 1) * eval_batch_size, num_test)
            bx_test = torch.from_numpy(X_test[t_start:t_end]).to(device)
            bin_test = torch.from_numpy(
                diff_in_test[t_start:t_end].toarray()
            ).to(device)
            bout_test = torch.from_numpy(
                diff_out_test[t_start:t_end].toarray()
            ).to(device)
            bdir_test = torch.from_numpy(
                direct_test[t_start:t_end].toarray()
            ).to(device)
            btld_test = torch.from_numpy(
                tld_prior_test[t_start:t_end]
            ).to(device)
            batch_test_logits = model(
                bx_test,
                in_diff=bin_test,
                out_diff=bout_test,
                direct_links=bdir_test,
                tld_prior=btld_test,
            )
            top10_batch = (
                torch.topk(batch_test_logits, k=10, dim=-1).indices.cpu().numpy()
            )
            test_logits_list.append(top10_batch)

    all_top10_test_preds = np.vstack(test_logits_list)

    # 20. Format and Export Submission Files
    sub_domains = []
    sub_trackers = []

    for d_id, preds in zip(test_domain_ids, all_top10_test_preds):
        for trk_id in preds:
            sub_domains.append(d_id)
            sub_trackers.append(tracker_to_domain_map[trk_id])

    sub_df = pd.DataFrame(
        {"domain_id": sub_domains, "tracking_domain_id": sub_trackers}
    )

    sub_df.to_csv("./submission/submission.csv", sep="\t", index=False)
    sub_df.to_csv("./submission/submission.tsv", sep="\t", index=False)

    print(f"Final Validation Score: {final_val_score:.6f}")


if __name__ == "__main__":
    main()
