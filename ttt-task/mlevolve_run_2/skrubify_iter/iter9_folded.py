import math

import numpy as np
import pandas as pd
import stratum as skrub  # drop-in for skrub: same .skb API, faster evaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


NUM_TRACKERS = 355  # Concrete value documented throughout the original script.
TOP_TLDS = [
    "com", "ru", "org", "net", "de", "uk", "jp", "fr", "it", "pl",
    "br", "cn", "in", "nl", "es", "cz", "eu", "ua", "ca", "au",
    "ch", "se", "ro", "gr", "at", "tv", "io", "me", "co", "info",
]
URL_CATEGORIES = [
    "Arts", "Business", "Computers", "Games", "Health", "Home",
    "Kids_and_Teens", "News", "Recreation", "Reference", "Regional",
    "Science", "Shopping", "Society", "Sports",
]
TWO_LEVEL_TLDS = [
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk",
    "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "asn.au", "id.au", "co.jp", "ne.jp", "or.jp", "ac.jp", "ed.jp",
    "go.jp", "gr.jp", "lg.jp", "com.br", "net.br", "org.br", "gov.br",
    "edu.br", "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn", "com.ru",
    "net.ru", "org.ru", "pp.ru", "co.in", "net.in", "org.in", "gen.in",
    "firm.in", "ind.in", "com.mx", "net.mx", "org.mx", "edu.mx", "gob.mx",
    "com.tr", "net.tr", "org.tr", "edu.tr", "gov.tr", "com.pl", "net.pl",
    "org.pl", "info.pl", "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
    "co.za", "net.za", "org.za", "web.za", "co.kr", "ne.kr", "or.kr",
    "re.kr", "com.ar", "net.ar", "org.ar", "com.tw", "org.tw", "idv.tw",
    "com.ua", "net.ua", "org.ua", "kiev.ua", "co.il", "org.il", "net.il",
    "com.sg", "org.sg", "net.sg", "com.hk", "org.hk", "net.hk",
]

LEXICAL_COLUMNS = [
    "domain_length", "num_dots", "num_hyphens", "num_digits", "digit_ratio",
    "has_www", "has_multi_subdomain", "press_score", "has_press_score",
    "is_authoritarian",
] + [f"tld_{tld}" for tld in TOP_TLDS] + ["tld_other"] + [
    "kw_ecommerce", "kw_media", "kw_video", "kw_adult", "kw_tech",
    "kw_finance", "kw_total",
]
CATEGORY_COLUMNS = [
    f"url_category_{category}" for category in URL_CATEGORIES
] + ["has_url_category"]
GRAPH_COLUMNS = [
    "out_degree", "log_out_degree", "in_degree", "log_in_degree",
    "total_degree", "in_out_ratio", "is_isolated", "tracker_link_count",
    "log_tracker_link_count", "tracker_link_ratio", "neighbor_out_degree",
    "log_neighbor_out_degree", "neighbor_in_degree",
    "log_neighbor_in_degree", "neighbor_total_degree",
    "log_neighbor_total_degree",
]
PRIOR_SCALAR_COLUMNS = [
    "has_root_match", "root_count", "log_root_count", "root_prior_entropy",
    "root_prior_max",
]
DIRECT_COLUMNS = [f"direct_{i}" for i in range(NUM_TRACKERS)]
OUT_NEIGHBOR_COLUMNS = [f"neighbor_out_{i}" for i in range(NUM_TRACKERS)]
IN_NEIGHBOR_COLUMNS = [f"neighbor_in_{i}" for i in range(NUM_TRACKERS)]
ROOT_PRIOR_COLUMNS = [f"root_prior_{i}" for i in range(NUM_TRACKERS)]
TLD_PRIOR_COLUMNS = [f"tld_prior_{i}" for i in range(NUM_TRACKERS)]
TARGET_COLUMNS = [f"target_{i}" for i in range(NUM_TRACKERS)]
FEATURE_COLUMNS = (
    LEXICAL_COLUMNS
    + CATEGORY_COLUMNS
    + GRAPH_COLUMNS
    + PRIOR_SCALAR_COLUMNS
    + DIRECT_COLUMNS
    + OUT_NEIGHBOR_COLUMNS
    + IN_NEIGHBOR_COLUMNS
    + ROOT_PRIOR_COLUMNS
    + TLD_PRIOR_COLUMNS
)


class CappedShuffledHoldout(BaseCrossValidator):
    """Seeded shuffle followed by min(30_000, floor(20%)) validation rows."""

    def __init__(self, test_fraction=0.2, max_test_size=30000, random_state=42):
        self.test_fraction = test_fraction
        self.max_test_size = max_test_size
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        n_test = min(self.max_test_size, int(n_samples * self.test_fraction))
        permutation = np.random.RandomState(self.random_state).permutation(n_samples)
        yield permutation[n_test:], permutation[:n_test]


class NeighborTrackerFeatures(TransformerMixin, BaseEstimator):
    """Learn graph-neighbor tracker adoption from each fold's training labels."""

    def __init__(self, num_trackers=NUM_TRACKERS):
        self.num_trackers = num_trackers

    def fit(self, X, y):
        self.train_domain_ids_ = X["domain_id"].to_numpy(dtype=np.int64)
        self.train_targets_ = np.asarray(y, dtype=np.float32)
        self.domain_to_train_idx_ = {
            int(domain_id): idx
            for idx, domain_id in enumerate(self.train_domain_ids_)
        }
        return self

    def fit_transform(self, X, y, links=None, degree_table=None):
        self.fit(X, y)
        return self._augment(X, links, degree_table)

    def transform(self, X, links=None, degree_table=None):
        return self._augment(X, links, degree_table)

    def _augment(self, X, links, degree_table):
        result = X.copy()
        selected_ids = result["domain_id"].to_numpy(dtype=np.int64)
        selected_to_idx = {
            int(domain_id): idx for idx, domain_id in enumerate(selected_ids)
        }

        src = links["source_domain_id"].to_numpy(dtype=np.int64)
        dst = links["target_domain_id"].to_numpy(dtype=np.int64)
        degree_lookup = dict(
            zip(
                degree_table["domain_id"].to_numpy(dtype=np.int64),
                degree_table["total_degree"].to_numpy(dtype=np.float32),
            )
        )

        out_rows, out_cols, out_weights = [], [], []
        in_rows, in_cols, in_weights = [], [], []

        for source, target in zip(src, dst):
            if source == target:
                continue

            selected_source = selected_to_idx.get(int(source))
            train_target = self.domain_to_train_idx_.get(int(target))
            if selected_source is not None and train_target is not None:
                degree = max(float(degree_lookup.get(int(target), 0.0)), 1.0)
                out_rows.append(selected_source)
                out_cols.append(train_target)
                out_weights.append(1.0 / np.log1p(degree))

            train_source = self.domain_to_train_idx_.get(int(source))
            selected_target = selected_to_idx.get(int(target))
            if train_source is not None and selected_target is not None:
                degree = max(float(degree_lookup.get(int(source), 0.0)), 1.0)
                in_rows.append(selected_target)
                in_cols.append(train_source)
                in_weights.append(1.0 / np.log1p(degree))

        n_selected = len(result)
        n_train = len(self.train_domain_ids_)
        A_out = csr_matrix(
            (
                np.asarray(out_weights, dtype=np.float32),
                (
                    np.asarray(out_rows, dtype=np.int64),
                    np.asarray(out_cols, dtype=np.int64),
                ),
            ),
            shape=(n_selected, n_train),
            dtype=np.float32,
        )
        A_in = csr_matrix(
            (
                np.asarray(in_weights, dtype=np.float32),
                (
                    np.asarray(in_rows, dtype=np.int64),
                    np.asarray(in_cols, dtype=np.int64),
                ),
            ),
            shape=(n_selected, n_train),
            dtype=np.float32,
        )

        out_degree = np.asarray(A_out.sum(axis=1)).ravel().astype(np.float32)
        in_degree = np.asarray(A_in.sum(axis=1)).ravel().astype(np.float32)
        out_counts = (A_out @ csr_matrix(self.train_targets_)).toarray()
        in_counts = (A_in @ csr_matrix(self.train_targets_)).toarray()

        out_norm = np.divide(
            out_counts,
            np.maximum(out_degree[:, None], 1e-7),
            out=np.zeros_like(out_counts, dtype=np.float32),
            where=out_degree[:, None] > 0,
        ).astype(np.float32)
        in_norm = np.divide(
            in_counts,
            np.maximum(in_degree[:, None], 1e-7),
            out=np.zeros_like(in_counts, dtype=np.float32),
            where=in_degree[:, None] > 0,
        ).astype(np.float32)

        result = result.assign(
            neighbor_out_degree=out_degree,
            log_neighbor_out_degree=np.log1p(out_degree).astype(np.float32),
            neighbor_in_degree=in_degree,
            log_neighbor_in_degree=np.log1p(in_degree).astype(np.float32),
            neighbor_total_degree=(out_degree + in_degree).astype(np.float32),
            log_neighbor_total_degree=np.log1p(out_degree + in_degree).astype(
                np.float32
            ),
        )
        out_frame = pd.DataFrame(
            out_norm, columns=OUT_NEIGHBOR_COLUMNS, index=result.index
        )
        in_frame = pd.DataFrame(
            in_norm, columns=IN_NEIGHBOR_COLUMNS, index=result.index
        )
        return pd.concat([result, out_frame, in_frame], axis=1)


class BayesianDomainPriors(TransformerMixin, BaseEstimator):
    """Fold-local TLD and root priors with leave-one-out training semantics."""

    def __init__(
        self,
        num_trackers=NUM_TRACKERS,
        alpha_tld=10.0,
        beta_root=2.0,
    ):
        self.num_trackers = num_trackers
        self.alpha_tld = alpha_tld
        self.beta_root = beta_root

    def fit(self, X, y):
        targets = np.asarray(y, dtype=np.float32)
        self.global_priors_ = targets.mean(axis=0).astype(np.float32)
        tlds = X["tld"].astype("string").fillna("unknown").to_numpy()
        roots = X["root"].astype("string").fillna("unknown").to_numpy()

        self.tld_counts_ = {}
        self.tld_tracker_counts_ = {}
        self.root_counts_ = {}
        self.root_tracker_counts_ = {}

        for tld, root, target in zip(tlds, roots, targets):
            self.tld_counts_[tld] = self.tld_counts_.get(tld, 0) + 1
            self.tld_tracker_counts_[tld] = (
                self.tld_tracker_counts_.get(
                    tld, np.zeros(self.num_trackers, dtype=np.float32)
                )
                + target
            )
            self.root_counts_[root] = self.root_counts_.get(root, 0) + 1
            self.root_tracker_counts_[root] = (
                self.root_tracker_counts_.get(
                    root, np.zeros(self.num_trackers, dtype=np.float32)
                )
                + target
            )

        self.tld_priors_ = {
            tld: (
                counts + self.alpha_tld * self.global_priors_
            ) / (self.tld_counts_[tld] + self.alpha_tld)
            for tld, counts in self.tld_tracker_counts_.items()
        }
        self.training_targets_ = targets
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self._augment(X, training=True)

    def transform(self, X):
        return self._augment(X, training=False)

    def _augment(self, X, training):
        result = X.copy()
        tlds = result["tld"].astype("string").fillna("unknown").to_numpy()
        roots = result["root"].astype("string").fillna("unknown").to_numpy()
        n_rows = len(result)

        tld_matrix = np.zeros((n_rows, self.num_trackers), dtype=np.float32)
        root_matrix = np.zeros_like(tld_matrix)
        has_root = np.zeros(n_rows, dtype=np.float32)
        root_counts = np.zeros(n_rows, dtype=np.float32)

        for idx, (tld, root) in enumerate(zip(tlds, roots)):
            tld_prior = self.tld_priors_.get(tld, self.global_priors_)
            tld_matrix[idx] = tld_prior
            root_matrix[idx] = tld_prior
            total = self.root_counts_.get(root, 0)

            if training:
                if total > 1:
                    total_loo = total - 1
                    count_loo = (
                        self.root_tracker_counts_[root]
                        - self.training_targets_[idx]
                    )
                    root_matrix[idx] = (
                        count_loo + self.beta_root * tld_prior
                    ) / (total_loo + self.beta_root)
                    has_root[idx] = 1.0
                    root_counts[idx] = float(total_loo)
            elif total > 0:
                root_matrix[idx] = (
                    self.root_tracker_counts_[root]
                    + self.beta_root * tld_prior
                ) / (total + self.beta_root)
                has_root[idx] = 1.0
                root_counts[idx] = float(total)

        entropy = -np.sum(
            root_matrix * np.log(root_matrix + 1e-12), axis=1
        ).astype(np.float32)

        result = result.assign(
            has_root_match=has_root,
            root_count=root_counts,
            log_root_count=np.log1p(root_counts).astype(np.float32),
            root_prior_entropy=entropy,
            root_prior_max=root_matrix.max(axis=1).astype(np.float32),
        )
        root_frame = pd.DataFrame(
            root_matrix, columns=ROOT_PRIOR_COLUMNS, index=result.index
        )
        tld_frame = pd.DataFrame(
            tld_matrix, columns=TLD_PRIOR_COLUMNS, index=result.index
        )
        return pd.concat([result, root_frame, tld_frame], axis=1)


class ResidualTabularBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
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

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act2(out + residual)


class TrackerRankNet(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes=NUM_TRACKERS,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        initial_bias=None,
        num_prior_channels=5,
        cooccurrence_matrix=None,
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
            hidden_dim, hidden_dim, dropout_rate
        )
        self.res_block2 = ResidualTabularBlock(
            hidden_dim, hidden_dim // 2, dropout_rate
        )

        bottleneck_dim = hidden_dim // 2
        self.direct_head = nn.Linear(bottleneck_dim, num_classes)
        self.domain_latent_proj = nn.Linear(bottleneck_dim, latent_dim)
        self.tracker_prototypes = nn.Parameter(
            torch.randn(num_classes, latent_dim) / math.sqrt(latent_dim)
        )
        self.head_blend = nn.Parameter(torch.tensor([0.5]))

        if self.has_tracker_priors:
            self.film_prior = nn.Linear(
                bottleneck_dim, num_prior_channels * 2
            )
            with torch.no_grad():
                nn.init.zeros_(self.film_prior.weight)
                nn.init.zeros_(self.film_prior.bias)

            initial_weights = torch.tensor(
                [1.5, 1.0, 1.0, 1.5, 0.8], dtype=torch.float32
            ).unsqueeze(0).repeat(num_classes, 1)
            self.tracker_prior_weights = nn.Parameter(initial_weights)
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

        self.cooc_layer = nn.Linear(num_classes, num_classes, bias=False)
        with torch.no_grad():
            if cooccurrence_matrix is not None:
                self.cooc_layer.weight.copy_(
                    torch.from_numpy(cooccurrence_matrix.T).float()
                )
            else:
                nn.init.normal_(self.cooc_layer.weight, std=0.01)

        self.rel_refine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_classes, num_classes),
        )
        with torch.no_grad():
            nn.init.zeros_(self.rel_refine[1].weight)
            nn.init.zeros_(self.rel_refine[1].bias)

        self.cooc_gate = nn.Linear(latent_dim, num_classes)
        with torch.no_grad():
            nn.init.normal_(self.cooc_gate.weight, std=0.01)
            nn.init.constant_(self.cooc_gate.bias, -2.0)

        if initial_bias is not None:
            with torch.no_grad():
                self.direct_head.bias.copy_(
                    torch.as_tensor(initial_bias, dtype=torch.float32)
                )

    def forward(self, x):
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
                :, -self.num_prior_channels * self.num_classes:
            ].reshape(-1, self.num_prior_channels, self.num_classes)
            prior_perm = prior_signals.permute(0, 2, 1)

            film_params = self.film_prior(h)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            modulated = prior_perm * gamma + beta

            direct_prior_logits = (
                modulated * self.tracker_prior_weights
            ).sum(dim=-1) + self.tracker_prior_bias
            mlp_prior_logits = self.tracker_res_net(modulated).squeeze(-1)
            base_logits = base_logits + self.res_scale * (
                direct_prior_logits + mlp_prior_logits
            )

        probabilities = torch.sigmoid(base_logits)
        message_1 = self.cooc_layer(probabilities)
        message_2 = self.cooc_layer(message_1)
        relational_message = message_1 + self.rel_refine(message_2)
        gate = torch.sigmoid(self.cooc_gate(domain_latent))
        return base_logits + gate * relational_message


class AsymmetricFocalRecallLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=2.0,
        clip_margin=0.05,
        eps=1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        positive_probabilities = probabilities.clamp(
            min=self.eps, max=1.0 - self.eps
        )
        positive_loss = (
            -targets
            * (1.0 - positive_probabilities).pow(self.gamma_pos)
            * torch.log(positive_probabilities)
        )

        negative_probabilities = (
            probabilities - self.clip_margin
        ).clamp(min=self.eps, max=1.0 - self.eps)
        negative_probabilities = torch.where(
            probabilities <= self.clip_margin,
            torch.zeros_like(negative_probabilities),
            negative_probabilities,
        )
        negative_loss = (
            -(1.0 - targets)
            * negative_probabilities.pow(self.gamma_neg)
            * torch.log(1.0 - negative_probabilities)
        )
        return (positive_loss + negative_loss).sum(dim=-1).mean()


class HybridTopKRankingLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=2.0,
        clip_margin=0.05,
        margin=1.0,
        top_k_neg=10,
        rank_weight=0.15,
        eps=1e-7,
    ):
        super().__init__()
        self.asym_focal = AsymmetricFocalRecallLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip_margin=clip_margin,
            eps=eps,
        )
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.rank_weight = rank_weight

    def forward(self, logits, targets):
        focal_loss = self.asym_focal(logits, targets)
        negative_logits = torch.where(
            targets == 0, logits, torch.full_like(logits, -1e9)
        )
        hard_negative_logits, _ = torch.topk(
            negative_logits, k=self.top_k_neg, dim=-1
        )
        margin_difference = self.margin - (
            logits.unsqueeze(-1) - hard_negative_logits.unsqueeze(1)
        )
        ranking_violation = F.relu(margin_difference)
        positive_mask = targets.unsqueeze(-1)
        positive_count = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sample_rank_loss = (
            ranking_violation * positive_mask
        ).sum(dim=(1, 2)) / (
            positive_count.squeeze(-1) * self.top_k_neg
        )
        return focal_loss + self.rank_weight * sample_rank_loss.mean()


class TrackerRankEstimator(ClassifierMixin, BaseEstimator):
    """The original custom PyTorch training and checkpoint-selection loop."""

    def __init__(
        self,
        num_classes=NUM_TRACKERS,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        num_prior_channels=5,
        num_epochs=12,
        batch_size=4096,
        base_lr=3e-4,
        min_lr=1e-5,
        random_state=42,
        inner_validation_fraction=0.2,
        inner_validation_max_size=30000,
    ):
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_prior_channels = num_prior_channels
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.random_state = random_state
        self.inner_validation_fraction = inner_validation_fraction
        self.inner_validation_max_size = inner_validation_max_size

    def fit(self, X, y, initial_bias=None, cooccurrence_matrix=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        features = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        targets = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
        initial_bias = np.asarray(initial_bias, dtype=np.float32).reshape(-1)
        cooccurrence_matrix = np.asarray(
            cooccurrence_matrix, dtype=np.float32
        )

        # The original selected checkpoints on the reported validation rows.
        # To avoid that leakage, checkpoint selection uses an equivalent split
        # carved only from this outer fold's training rows.
        n_samples = len(features)
        n_validation = min(
            self.inner_validation_max_size,
            int(n_samples * self.inner_validation_fraction),
        )
        permutation = np.random.RandomState(
            self.random_state
        ).permutation(n_samples)
        validation_idx = permutation[:n_validation]
        training_idx = permutation[n_validation:]

        X_train = features[training_idx]
        y_train = targets[training_idx]
        X_validation = features[validation_idx]
        y_validation = targets[validation_idx]

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_ = TrackerRankNet(
            in_dim=features.shape[1],
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            initial_bias=initial_bias,
            num_prior_channels=self.num_prior_channels,
            cooccurrence_matrix=cooccurrence_matrix,
        ).to(self.device_)

        criterion = HybridTopKRankingLoss(
            gamma_pos=0.0,
            gamma_neg=2.0,
            clip_margin=0.05,
            margin=1.0,
            top_k_neg=10,
            rank_weight=0.15,
        )

        decay_params, no_decay_params = [], []
        for name, parameter in self.model_.named_parameters():
            if not parameter.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-4},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs - 1,
            eta_min=self.min_lr,
        )

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_validation_t = torch.from_numpy(X_validation)

        n_train = len(X_train_t)
        batches_per_epoch = max(math.ceil(n_train / self.batch_size), 1)
        best_recall = -1.0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model_.state_dict().items()
        }

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            permutation = torch.randperm(n_train)
            batch_number = 0

            for start in range(0, n_train, self.batch_size):
                if epoch == 1:
                    warmup_step = batch_number + 1
                    warmup_lr = self.min_lr + (
                        self.base_lr - self.min_lr
                    ) * (warmup_step / batches_per_epoch)
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr

                end = min(start + self.batch_size, n_train)
                indices = permutation[start:end]
                batch_x = X_train_t[indices].to(
                    self.device_, non_blocking=True
                )
                batch_y = y_train_t[indices].to(
                    self.device_, non_blocking=True
                )

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(), max_norm=1.0
                )
                optimizer.step()
                batch_number += 1

            if epoch > 1:
                scheduler.step()

            validation_logits = self._predict_tensor(X_validation_t)
            validation_recall = recall_at_10(
                y_validation, validation_logits
            )
            if validation_recall > best_recall:
                best_recall = validation_recall
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }

        self.model_.load_state_dict(
            {
                key: value.to(self.device_)
                for key, value in best_state.items()
            }
        )
        self.model_.eval()
        self.classes_ = np.arange(self.num_classes, dtype=np.int64)
        return self

    def _predict_tensor(self, tensor):
        predictions = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), self.batch_size):
                end = min(start + self.batch_size, len(tensor))
                batch = tensor[start:end].to(self.device_)
                predictions.append(self.model_(batch).cpu().numpy())
        return np.vstack(predictions)

    def predict(self, X):
        features = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return self._predict_tensor(torch.from_numpy(features))


def recall_at_10(y_true, y_pred):
    targets = np.asarray(y_true)
    logits = np.asarray(y_pred)
    top_indices = np.argpartition(logits, -10, axis=1)[:, -10:]
    row_indices = np.arange(len(logits))[:, None]
    hits = targets[row_indices, top_indices].sum(axis=1)
    true_counts = targets.sum(axis=1)
    recalls = np.divide(
        hits,
        true_counts,
        out=np.zeros_like(hits, dtype=np.float64),
        where=true_counts > 0,
    )
    return float(np.mean(recalls))


recall_scorer = make_scorer(
    recall_at_10,
    response_method="predict",
    greater_is_better=True,
)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — reads are recorded. Chunking, intermediate files,
    # checkpoints, test prediction, and submission generation are omitted.
    trackers = (
        skrub.as_data_op("input/trackers.tsv")
        .skb.apply_func(pd.read_csv, sep="\t")
    )
    trackers = trackers.assign(
        tracker_id=trackers["tracker_id"].astype(int),
        tracking_domain_id=trackers["tracking_domain_id"].astype(np.int64),
    )

    target_domains = (
        skrub.as_data_op("input/target.tsv")
        .skb.apply_func(pd.read_csv, sep="\t")
    )
    tracking_graph = (
        skrub.as_data_op("input/tracking_graph_train.parquet")
        .skb.apply_func(
            pd.read_parquet,
            columns=["domain_id", "tracking_domain_id", "tracker_id"],
        )
    )
    tracking_graph = tracking_graph.assign(
        domain_id=tracking_graph["domain_id"].astype(np.int64),
        tracker_id=tracking_graph["tracker_id"].astype(int),
    )

    domains = (
        skrub.as_data_op("input/domains.parquet")
        .skb.apply_func(
            pd.read_parquet,
            columns=["domain", "domain_id"],
        )
        .drop_duplicates(subset="domain_id", keep="last")
    )
    links = (
        skrub.as_data_op("input/link-graph.parquet")
        .skb.apply_func(
            pd.read_parquet,
            columns=["source_domain_id", "target_domain_id"],
        )
    )

    # The original guarded these documented inputs with os.path.exists. Those
    # environment checks are dropped and each file is read directly.
    press = (
        skrub.as_data_op("input/freedom-of-the-press.csv")
        .skb.apply_func(pd.read_csv, sep=None, engine="python")
    )
    url_classes = (
        skrub.as_data_op("input/url-classification.csv")
        .skb.apply_func(
            pd.read_csv,
            usecols=["url", "category"],
        )
    )

    # 2. Prepare data — exclude test domains, construct the dense multi-label
    # target, mark the raw target, and attach the original capped holdout split.
    candidate_domains = (
        tracking_graph[["domain_id"]]
        .drop_duplicates()
        .sort_values("domain_id")
    )
    candidate_domains = candidate_domains[
        ~candidate_domains["domain_id"].isin(target_domains["domain_id"])
    ].reset_index(drop=True)

    target_long = (
        tracking_graph[["domain_id", "tracker_id"]]
        .drop_duplicates()
        .assign(present=np.uint8(1))
    )
    target_wide = (
        target_long.set_index(["domain_id", "tracker_id"])["present"]
        .unstack(fill_value=np.uint8(0))
        .reindex(
            columns=range(NUM_TRACKERS),
            fill_value=np.uint8(0),
        )
        .rename(
            columns={
                idx: f"target_{idx}" for idx in range(NUM_TRACKERS)
            }
        )
        .reset_index()
    )
    labeled_domains = candidate_domains.merge(
        target_wide.drop_duplicates(subset="domain_id", keep="last"),
        on="domain_id",
        how="left",
    ).fillna(
        {column: np.uint8(0) for column in TARGET_COLUMNS}
    )
    y = labeled_domains[TARGET_COLUMNS].skb.mark_as_y()

    raw_X = candidate_domains.merge(
        domains,
        on="domain_id",
        how="left",
    )
    raw_X = raw_X.assign(
        domain=raw_X["domain"].astype("string").fillna("")
    )
    X = raw_X.skb.mark_as_X(
        cv=CappedShuffledHoldout(
            test_fraction=0.2,
            max_test_size=30000,
            random_state=42,
        ),
        split_kwargs={},
    )

    # 3. Recorded lexical and press-freedom feature engineering.
    domain_name = X["domain"].astype("string").fillna("").str.lower().str.strip()
    safe_domain = domain_name.where(domain_name.str.len() > 0, "unknown")
    tld_last = safe_domain.str.extract(r"([^.]*)$", expand=False)
    second_last = safe_domain.str.extract(
        r"([^.]*)\.[^.]*$", expand=False
    )
    third_last = safe_domain.str.extract(
        r"([^.]*)\.[^.]*\.[^.]*$", expand=False
    )
    dot_count = safe_domain.str.count(r"\.")
    last_two = second_last.fillna("") + "." + tld_last
    has_two_level_tld = (
        (dot_count >= 2) & last_two.isin(TWO_LEVEL_TLDS)
    )
    ordinary_root = last_two.where(dot_count >= 1, tld_last)
    root = ordinary_root.where(
        ~has_two_level_tld,
        third_last.fillna("") + "." + last_two,
    )
    tld = tld_last.where(~has_two_level_tld, last_two)

    press_lookup = (
        press.iloc[:, [0, 2]]
        .set_axis(["tld", "press_score_lookup"], axis=1)
    )
    press_lookup = press_lookup.assign(
        tld=press_lookup["tld"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.lstrip("."),
        press_score_lookup=press_lookup[
            "press_score_lookup"
        ].skb.apply_func(pd.to_numeric, errors="coerce"),
        has_press_score=np.float32(1.0),
    ).drop_duplicates(subset="tld", keep="last")
    press_median = press_lookup["press_score_lookup"].median()

    lexical = X.assign(root=root, tld=tld).merge(
        press_lookup,
        on="tld",
        how="left",
    )
    domain_length = lexical["domain"].str.len().clip(lower=1)
    num_digits = lexical["domain"].str.count(r"\d")
    press_score = (
        lexical["press_score_lookup"]
        .fillna(press_median)
        .fillna(50.0)
    )

    lexical = lexical.assign(
        domain_length=domain_length.astype(np.float32),
        num_dots=lexical["domain"].str.count(r"\.").astype(np.float32),
        num_hyphens=lexical["domain"].str.count("-").astype(np.float32),
        num_digits=num_digits.astype(np.float32),
        digit_ratio=(num_digits / domain_length).astype(np.float32),
        has_www=lexical["domain"]
        .str.lower()
        .str.startswith("www.")
        .astype(np.float32),
        has_multi_subdomain=(
            lexical["domain"].str.count(r"\.") > 1
        ).astype(np.float32),
        press_score=press_score.astype(np.float32),
        has_press_score=lexical["has_press_score"]
        .fillna(0.0)
        .astype(np.float32),
        is_authoritarian=(press_score > 60.0).astype(np.float32),
    )
    lexical = lexical.assign(
        **{
            f"tld_{value}": (lexical["tld"] == value).astype(np.float32)
            for value in TOP_TLDS
        },
        tld_other=(~lexical["tld"].isin(TOP_TLDS)).astype(np.float32),
    )

    kw_ecommerce = sum(
        lexical["domain"].str.lower().str.contains(
            keyword, regex=False, na=False
        ).astype(np.float32)
        for keyword in [
            "shop", "store", "cart", "buy", "market", "mall", "deal", "pay"
        ]
    )
    kw_media = sum(
        lexical["domain"].str.lower().str.contains(
            keyword, regex=False, na=False
        ).astype(np.float32)
        for keyword in [
            "news", "press", "media", "times", "post", "daily",
            "gazette", "journal",
        ]
    )
    kw_video = sum(
        lexical["domain"].str.lower().str.contains(
            keyword, regex=False, na=False
        ).astype(np.float32)
        for keyword in ["video", "tv", "movie", "film", "tube", "stream"]
    )
    kw_adult = sum(
        lexical["domain"].str.lower().str.contains(
            keyword, regex=False, na=False
        ).astype(np.float32)
        for keyword in ["adult", "sex", "porn", "xxx"]
    )
    kw_tech = sum(
        lexical["domain"].str.lower().str.contains(
            keyword, regex=False, na=False
        ).astype(np.float32)
        for keyword in ["tech", "dev", "code", "cloud", "soft", "app", "web"]
    )
    kw_finance = sum(
        lexical["domain"].str.lower().str.contains(
            keyword, regex=False, na=False
        ).astype(np.float32)
        for keyword in ["bank", "finance", "invest", "loan", "crypto", "coin"]
    )
    lexical = lexical.assign(
        kw_ecommerce=kw_ecommerce,
        kw_media=kw_media,
        kw_video=kw_video,
        kw_adult=kw_adult,
        kw_tech=kw_tech,
        kw_finance=kw_finance,
        kw_total=(
            kw_ecommerce
            + kw_media
            + kw_video
            + kw_adult
            + kw_tech
            + kw_finance
        ),
    ).drop(columns=["press_score_lookup"])

    # 4. URL classification features as recorded parsing and reshaping.
    domain_name_lookup = domains.assign(
        lookup_domain=domains["domain"]
        .astype("string")
        .str.lower()
        .str.strip()
    )[["lookup_domain", "domain_id"]].drop_duplicates(
        subset="lookup_domain", keep="last"
    )
    domain_name_lookup = domain_name_lookup.rename(
        columns={"domain_id": "exact_domain_id"}
    )
    fallback_lookup = domain_name_lookup.rename(
        columns={
            "lookup_domain": "fallback_domain",
            "exact_domain_id": "fallback_domain_id",
        }
    )

    url_string = url_classes["url"].astype("string")
    host = (
        url_string.str.replace(r"^[^:]*://", "", regex=True)
        .str.extract(r"^([^/:]*)", expand=False)
        .str.strip()
        .str.lower()
    )
    parsed_urls = url_classes.assign(
        host=host,
        fallback_domain=host.str.replace(r"^www\.", "", regex=True),
        category=url_classes["category"].astype("string"),
    )
    parsed_urls = parsed_urls.merge(
        domain_name_lookup,
        left_on="host",
        right_on="lookup_domain",
        how="left",
    ).merge(
        fallback_lookup,
        on="fallback_domain",
        how="left",
    )
    parsed_urls = parsed_urls.assign(
        domain_id=parsed_urls["exact_domain_id"].fillna(
            parsed_urls["fallback_domain_id"]
        )
    )
    parsed_urls = parsed_urls[
        parsed_urls["domain_id"].notna()
        & parsed_urls["category"].isin(URL_CATEGORIES)
    ]

    url_counts = (
        parsed_urls.groupby(["domain_id", "category"])
        .size()
        .rename("category_count")
        .reset_index()
    )
    category_totals = url_counts.groupby(
        "domain_id"
    )["category_count"].transform("sum")
    url_counts = url_counts.assign(
        proportion=url_counts["category_count"] / category_totals
    )
    url_wide = (
        url_counts.set_index(["domain_id", "category"])["proportion"]
        .unstack(fill_value=np.float32(0.0))
        .reindex(
            columns=URL_CATEGORIES,
            fill_value=np.float32(0.0),
        )
        .rename(
            columns={
                category: f"url_category_{category}"
                for category in URL_CATEGORIES
            }
        )
        .assign(has_url_category=np.float32(1.0))
        .reset_index()
        .drop_duplicates(subset="domain_id", keep="last")
    )
    lexical = lexical.merge(url_wide, on="domain_id", how="left")
    lexical = lexical.assign(
        **{
            column: lexical[column].fillna(0.0).astype(np.float32)
            for column in CATEGORY_COLUMNS
        }
    )

    # 5. Static graph topology and direct tracker-link indicators.
    out_degree_table = (
        links.groupby("source_domain_id")
        .size()
        .rename("out_degree")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )
    in_degree_table = (
        links.groupby("target_domain_id")
        .size()
        .rename("in_degree")
        .reset_index()
        .rename(columns={"target_domain_id": "domain_id"})
    )
    degree_table = out_degree_table.merge(
        in_degree_table,
        on="domain_id",
        how="outer",
    ).fillna({"out_degree": 0, "in_degree": 0})
    degree_table = degree_table.assign(
        total_degree=(
            degree_table["out_degree"] + degree_table["in_degree"]
        ).astype(np.float32)
    )

    tracker_lookup = trackers[
        ["tracking_domain_id", "tracker_id"]
    ].drop_duplicates(subset="tracking_domain_id", keep="last")
    tracker_edges = links.merge(
        tracker_lookup,
        left_on="target_domain_id",
        right_on="tracking_domain_id",
        how="left",
    )
    tracker_edges = tracker_edges[tracker_edges["tracker_id"].notna()]

    tracker_link_counts = (
        tracker_edges.groupby("source_domain_id")
        .size()
        .rename("tracker_link_count")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )
    direct_wide = (
        tracker_edges[["source_domain_id", "tracker_id"]]
        .drop_duplicates()
        .assign(present=np.float32(1.0))
        .set_index(["source_domain_id", "tracker_id"])["present"]
        .unstack(fill_value=np.float32(0.0))
        .reindex(
            columns=range(NUM_TRACKERS),
            fill_value=np.float32(0.0),
        )
        .rename(
            columns={idx: f"direct_{idx}" for idx in range(NUM_TRACKERS)}
        )
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
        .drop_duplicates(subset="domain_id", keep="last")
    )

    features = (
        lexical.merge(
            degree_table.drop_duplicates(subset="domain_id", keep="last"),
            on="domain_id",
            how="left",
        )
        .merge(
            tracker_link_counts.drop_duplicates(
                subset="domain_id", keep="last"
            ),
            on="domain_id",
            how="left",
        )
        .merge(direct_wide, on="domain_id", how="left")
    )
    out_degree = features["out_degree"].fillna(0.0)
    in_degree = features["in_degree"].fillna(0.0)
    tracker_link_count = features["tracker_link_count"].fillna(0.0)

    features = features.assign(
        out_degree=out_degree.astype(np.float32),
        log_out_degree=out_degree.skb.apply_func(np.log1p).astype(np.float32),
        in_degree=in_degree.astype(np.float32),
        log_in_degree=in_degree.skb.apply_func(np.log1p).astype(np.float32),
        total_degree=(out_degree + in_degree).astype(np.float32),
        in_out_ratio=(
            (in_degree + 1.0) / (out_degree + 1.0)
        ).astype(np.float32),
        is_isolated=((out_degree + in_degree) == 0).astype(np.float32),
        tracker_link_count=tracker_link_count.astype(np.float32),
        log_tracker_link_count=tracker_link_count.skb.apply_func(
            np.log1p
        ).astype(np.float32),
        tracker_link_ratio=(
            tracker_link_count / out_degree.clip(lower=1.0)
        ).astype(np.float32),
        **{
            column: features[column].fillna(0.0).astype(np.float32)
            for column in DIRECT_COLUMNS
        },
    )

    # Neighbor adoption and Bayesian priors depend on fold-training labels.
    features = features.skb.apply(
        NeighborTrackerFeatures(num_trackers=NUM_TRACKERS),
        y=y,
        fit_transform_kwargs={
            "links": links,
            "degree_table": degree_table,
        },
        transform_kwargs={
            "links": links,
            "degree_table": degree_table,
        },
    )
    features = features.skb.apply(
        BayesianDomainPriors(
            num_trackers=NUM_TRACKERS,
            alpha_tld=10.0,
            beta_root=2.0,
        ),
        y=y,
    )

    design = (
        features[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )

    # Fold-local target statistics are fit-only inputs to the custom predictor.
    global_tracker_priors = y.mean(axis=0).astype(np.float32)
    clipped_priors = global_tracker_priors.clip(
        lower=1e-5, upper=1.0 - 1e-5
    )
    initial_bias = (
        clipped_priors / (1.0 - clipped_priors)
    ).skb.apply_func(np.log).astype(np.float32)

    y_float = y.astype(np.float32)
    cooccurrence_counts = y_float.T.dot(y_float)
    tracker_frequencies = cooccurrence_counts.skb.apply_func(np.diag)
    conditional_cooccurrence = cooccurrence_counts.div(
        tracker_frequencies + 10.0,
        axis=0,
    )
    conditional_cooccurrence = conditional_cooccurrence.where(
        ~np.eye(NUM_TRACKERS, dtype=bool),
        0.0,
    )
    cooccurrence_row_max = conditional_cooccurrence.max(axis=1)
    normalized_cooccurrence = conditional_cooccurrence.div(
        cooccurrence_row_max.clip(lower=1.0),
        axis=0,
    ).astype(np.float32)

    model = TrackerRankEstimator(
        num_classes=NUM_TRACKERS,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        num_prior_channels=5,
        num_epochs=12,
        batch_size=4096,
        base_lr=3e-4,
        min_lr=1e-5,
        random_state=42,
        inner_validation_fraction=0.2,
        inner_validation_max_size=30000,
    )
    pred = design.skb.apply(
        model,
        y=y,
        fit_kwargs={
            "initial_bias": initial_bias,
            "cooccurrence_matrix": normalized_cooccurrence,
        },
    )

    # 6. Score. The splitter on mark_as_X drives; no cv= is passed here.
    if __name__ == "__main__":
        with skrub.config(scheduler=True):
            search = pred.skb.make_grid_search(
                n_jobs=1,
                fitted=True,
                refit=False,
                scoring=recall_scorer,
            )
        results = search.results_
        print(results)
        for variant_score in results["scores"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {results['scores'][0]}")
