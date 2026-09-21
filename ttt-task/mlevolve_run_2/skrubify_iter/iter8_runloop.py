import math

import numpy as np
import pandas as pd
import stratum as skrub  # drop-in for skrub: same .skb API, faster evaluator
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


NUM_TRACKERS = 355  # Inferred from the original's documented N x 355 targets.
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
TWO_LEVEL_TLDS = {
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
}

TARGET_COLUMNS = [f"target_{i}" for i in range(NUM_TRACKERS)]
DIRECT_COLUMNS = [f"direct_{i}" for i in range(NUM_TRACKERS)]
OUT_NEIGHBOR_COLUMNS = [f"out_neighbor_{i}" for i in range(NUM_TRACKERS)]
IN_NEIGHBOR_COLUMNS = [f"in_neighbor_{i}" for i in range(NUM_TRACKERS)]
ROOT_PRIOR_COLUMNS = [f"root_prior_{i}" for i in range(NUM_TRACKERS)]
TLD_PRIOR_COLUMNS = [f"tld_prior_{i}" for i in range(NUM_TRACKERS)]

LEXICAL_COLUMNS = [
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "has_www",
    "has_multi_subdomain",
    "press_score",
    "has_press_score",
    "is_authoritarian",
    *[f"tld_{t}" for t in TOP_TLDS],
    "tld_other",
    "kw_ecommerce",
    "kw_media",
    "kw_video",
    "kw_adult",
    "kw_tech",
    "kw_finance",
    "kw_total",
]
URL_FEATURE_COLUMNS = [
    *[f"url_category_{category}" for category in URL_CATEGORIES],
    "has_url_category",
]
GRAPH_COLUMNS = [
    "out_degree",
    "log_out_degree",
    "in_degree",
    "log_in_degree",
    "total_degree",
    "in_out_ratio",
    "is_isolated",
    "tracker_out_links",
    "log_tracker_out_links",
    "tracker_link_ratio",
    "out_neighbor_degree",
    "log_out_neighbor_degree",
    "in_neighbor_degree",
    "log_in_neighbor_degree",
    "neighbor_total_degree",
    "log_neighbor_total_degree",
]
PRIOR_SCALAR_COLUMNS = [
    "has_root_match",
    "root_count",
    "log_root_count",
    "root_prior_entropy",
    "root_prior_max",
]
FEATURE_COLUMNS = [
    *LEXICAL_COLUMNS,
    *URL_FEATURE_COLUMNS,
    *GRAPH_COLUMNS,
    *PRIOR_SCALAR_COLUMNS,
    *DIRECT_COLUMNS,
    *OUT_NEIGHBOR_COLUMNS,
    *IN_NEIGHBOR_COLUMNS,
    *ROOT_PRIOR_COLUMNS,
    *TLD_PRIOR_COLUMNS,
]


class OriginalDomainHoldout(BaseCrossValidator):
    """Reproduce the original shuffled 20%-or-30,000-domain holdout."""

    def __init__(self, random_state=42, max_validation_size=30_000):
        self.random_state = random_state
        self.max_validation_size = max_validation_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)
        n_val = min(self.max_validation_size, int(len(indices) * 0.2))
        yield indices[n_val:], indices[:n_val]


class NeighborAdoption(TransformerMixin, BaseEstimator):
    """Learn tracker adoption from this fold's training-domain neighbors."""

    def __init__(self, direction="out", num_trackers=NUM_TRACKERS):
        self.direction = direction
        self.num_trackers = num_trackers

    def fit(self, X, y):
        self.train_domains_ = np.asarray(X["domain_id"], dtype=np.int64)
        self.train_targets_ = np.asarray(y, dtype=np.float32)
        return self

    def fit_transform(self, X, y, table=None):
        self.fit(X, y)
        return self.transform(X, table=table)

    def transform(self, X, table=None):
        domains = np.asarray(X["domain_id"], dtype=np.int64)
        edges = pd.DataFrame(table).loc[
            :, ["domain_id", "neighbor_domain_id", "weight"]
        ].copy()

        train_lookup = pd.DataFrame(
            {
                "neighbor_domain_id": self.train_domains_,
                "_train_idx": np.arange(len(self.train_domains_), dtype=np.int64),
            }
        )
        edges = edges.merge(
            train_lookup, on="neighbor_domain_id", how="inner", sort=False
        )
        edges = edges[edges["domain_id"] != edges["neighbor_domain_id"]]

        row_lookup = pd.DataFrame(
            {
                "domain_id": domains,
                "_row_idx": np.arange(len(domains), dtype=np.int64),
            }
        )
        edges = row_lookup.merge(edges, on="domain_id", how="inner", sort=False)

        if len(edges):
            adjacency = csr_matrix(
                (
                    edges["weight"].to_numpy(dtype=np.float32),
                    (
                        edges["_row_idx"].to_numpy(dtype=np.int64),
                        edges["_train_idx"].to_numpy(dtype=np.int64),
                    ),
                ),
                shape=(len(domains), len(self.train_domains_)),
                dtype=np.float32,
            )
            weighted_degree = np.asarray(adjacency.sum(axis=1)).ravel()
            counts = (adjacency @ self.train_targets_).astype(np.float32)
            normalized = np.where(
                weighted_degree[:, None] > 0,
                counts / np.maximum(weighted_degree[:, None], 1e-7),
                0.0,
            ).astype(np.float32)
        else:
            weighted_degree = np.zeros(len(domains), dtype=np.float32)
            normalized = np.zeros(
                (len(domains), self.num_trackers), dtype=np.float32
            )

        prefix = "out_neighbor" if self.direction == "out" else "in_neighbor"
        columns = (
            OUT_NEIGHBOR_COLUMNS
            if self.direction == "out"
            else IN_NEIGHBOR_COLUMNS
        )
        result = pd.DataFrame(normalized, columns=columns, index=X.index)
        result[f"{prefix}_degree"] = weighted_degree.astype(np.float32)
        return result


class TldTrackerPrior(TransformerMixin, BaseEstimator):
    """Smoothed TLD tracker priors fitted independently in each outer fold."""

    def __init__(self, alpha=10.0, num_trackers=NUM_TRACKERS):
        self.alpha = alpha
        self.num_trackers = num_trackers

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=np.float32)
        self.global_prior_ = y_array.mean(axis=0).astype(np.float32)

        frame = pd.DataFrame(y_array, columns=TARGET_COLUMNS)
        frame["_tld"] = (
            pd.Series(X["tld"], index=X.index)
            .astype("string")
            .fillna("unknown")
            .to_numpy()
        )
        grouped = frame.groupby("_tld", sort=False)
        self.tld_counts_ = grouped.size()
        self.tld_sums_ = grouped[TARGET_COLUMNS].sum()
        return self

    def transform(self, X):
        tlds = (
            pd.Series(X["tld"], index=X.index)
            .astype("string")
            .fillna("unknown")
        )
        output = np.empty((len(X), self.num_trackers), dtype=np.float32)

        for row_idx, tld in enumerate(tlds):
            if tld in self.tld_counts_.index:
                count = float(self.tld_counts_.loc[tld])
                sums = self.tld_sums_.loc[tld].to_numpy(dtype=np.float32)
                output[row_idx] = (
                    sums + self.alpha * self.global_prior_
                ) / (count + self.alpha)
            else:
                output[row_idx] = self.global_prior_

        return pd.DataFrame(output, columns=TLD_PRIOR_COLUMNS, index=X.index)


class RootTrackerPrior(TransformerMixin, BaseEstimator):
    """Bayesian root priors with the original training-row leave-one-out branch."""

    def __init__(self, beta=2.0, num_trackers=NUM_TRACKERS):
        self.beta = beta
        self.num_trackers = num_trackers

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=np.float32)
        roots = (
            pd.Series(X["root"], index=X.index)
            .astype("string")
            .fillna("unknown")
            .to_numpy()
        )
        frame = pd.DataFrame(y_array, columns=TARGET_COLUMNS)
        frame["_root"] = roots
        grouped = frame.groupby("_root", sort=False)
        self.root_counts_ = grouped.size()
        self.root_sums_ = grouped[TARGET_COLUMNS].sum()
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self._transform(
            X, y_array=np.asarray(y, dtype=np.float32), leave_one_out=True
        )

    def transform(self, X):
        return self._transform(X, y_array=None, leave_one_out=False)

    def _transform(self, X, y_array, leave_one_out):
        roots = (
            pd.Series(X["root"], index=X.index)
            .astype("string")
            .fillna("unknown")
            .to_numpy()
        )
        tld_prior = np.asarray(X[TLD_PRIOR_COLUMNS], dtype=np.float32)

        root_prior = tld_prior.copy()
        has_match = np.zeros(len(X), dtype=np.float32)
        root_count = np.zeros(len(X), dtype=np.float32)

        for i, root in enumerate(roots):
            if root not in self.root_counts_.index:
                continue

            total = int(self.root_counts_.loc[root])
            sums = self.root_sums_.loc[root].to_numpy(dtype=np.float32)

            if leave_one_out:
                if total <= 1:
                    continue
                total_used = total - 1
                sums_used = sums - y_array[i]
            else:
                total_used = total
                sums_used = sums

            root_prior[i] = (
                sums_used + self.beta * tld_prior[i]
            ) / (total_used + self.beta)
            has_match[i] = 1.0
            root_count[i] = float(total_used)

        result = pd.DataFrame(
            root_prior, columns=ROOT_PRIOR_COLUMNS, index=X.index
        )
        result["has_root_match"] = has_match
        result["root_count"] = root_count
        return result


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
        out = self.dropout(self.act1(self.norm1(self.fc1(x))))
        out = self.norm2(self.fc2(out))
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
            self.film_prior = nn.Linear(bottleneck_dim, num_prior_channels * 2)
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
                self.direct_head.bias.copy_(torch.tensor(initial_bias))

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
            prior_permuted = prior_signals.permute(0, 2, 1)

            film_parameters = self.film_prior(h)
            gamma, beta = film_parameters.chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            prior_modulated = prior_permuted * gamma + beta

            direct_prior_logits = (
                prior_modulated * self.tracker_prior_weights
            ).sum(dim=-1) + self.tracker_prior_bias
            mlp_prior_logits = self.tracker_res_net(prior_modulated).squeeze(-1)
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

        negative_probabilities = (probabilities - self.clip_margin).clamp(
            min=self.eps, max=1.0 - self.eps
        )
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
        sample_loss = (ranking_violation * positive_mask).sum(dim=(1, 2)) / (
            positive_count.squeeze(-1) * self.top_k_neg
        )
        return focal_loss + self.rank_weight * sample_loss.mean()


def recall_at_10(y_true, y_pred):
    targets = np.asarray(y_true)
    logits = np.asarray(y_pred)
    top_k = 10
    top_indices = np.argpartition(logits, -top_k, axis=1)[:, -top_k:]
    rows = np.arange(len(logits))[:, None]
    hits = targets[rows, top_indices].sum(axis=1)
    true_counts = targets.sum(axis=1)
    recalls = np.divide(
        hits,
        true_counts,
        out=np.zeros_like(hits, dtype=np.float64),
        where=true_counts > 0,
    )
    return float(np.mean(recalls))


class TorchTrackerRanker(RegressorMixin, BaseEstimator):
    """The original PyTorch loop, re-run independently for every outer fold."""

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

    def fit(self, X, y, global_priors=None, cooccurrence_matrix=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_array = np.array(X, dtype=np.float32, copy=True)
        y_array = np.array(y, dtype=np.float32, copy=True)
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        priors = np.asarray(global_priors, dtype=np.float32).reshape(-1)
        cooccurrence = np.asarray(cooccurrence_matrix, dtype=np.float32)

        prior_eps = 1e-5
        clipped = np.clip(priors, prior_eps, 1.0 - prior_eps)
        initial_bias = np.log(clipped / (1.0 - clipped)).astype(np.float32)

        # The original chose checkpoints on the same rows it scored. Outer CV
        # cannot reproduce that leak, so checkpoint selection uses an inner
        # holdout carved only from this fold's training rows.
        indices = np.arange(len(X_array))
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(indices)
        n_validation = min(30_000, int(len(indices) * 0.2))
        validation_idx = indices[:n_validation]
        training_idx = indices[n_validation:]

        X_train = X_array[training_idx]
        y_train = y_array[training_idx]
        X_validation = X_array[validation_idx]
        y_validation = y_array[validation_idx]

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = TrackerRankNet(
            in_dim=X_array.shape[1],
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            initial_bias=initial_bias,
            num_prior_channels=self.num_prior_channels,
            cooccurrence_matrix=cooccurrence,
        ).to(self.device_)

        criterion = HybridTopKRankingLoss(
            gamma_pos=0.0,
            gamma_neg=2.0,
            clip_margin=0.05,
            margin=1.0,
            top_k_neg=10,
            rank_weight=0.15,
        )

        decay_params = []
        no_decay_params = []
        for name, parameter in model.named_parameters():
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

        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        X_validation_tensor = torch.from_numpy(X_validation)

        n_train = len(X_train_tensor)
        batches_per_epoch = math.ceil(n_train / self.batch_size)
        best_validation_recall = -1.0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            permutation = torch.randperm(n_train)
            batch_number = 0

            for start in range(0, n_train, self.batch_size):
                if epoch == 1:
                    warmup_step = batch_number + 1
                    warmup_lr = self.min_lr + (
                        self.base_lr - self.min_lr
                    ) * (warmup_step / batches_per_epoch)
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = warmup_lr

                end = min(start + self.batch_size, n_train)
                batch_indices = permutation[start:end]
                batch_X = X_train_tensor[batch_indices].to(
                    self.device_, non_blocking=True
                )
                batch_y = y_train_tensor[batch_indices].to(
                    self.device_, non_blocking=True
                )

                optimizer.zero_grad()
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                batch_number += 1

            if epoch > 1:
                scheduler.step()

            model.eval()
            validation_parts = []
            with torch.no_grad():
                for start in range(0, len(X_validation_tensor), 4096):
                    end = min(start + 4096, len(X_validation_tensor))
                    batch = X_validation_tensor[start:end].to(self.device_)
                    validation_parts.append(model(batch).cpu().numpy())

            validation_logits = np.vstack(validation_parts)
            validation_recall = recall_at_10(
                y_validation, validation_logits
            )
            if validation_recall > best_validation_recall:
                best_validation_recall = validation_recall
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        model.load_state_dict(
            {key: value.to(self.device_) for key, value in best_state.items()}
        )
        model.eval()
        self.model_ = model
        return self

    def predict(self, X):
        X_array = np.array(X, dtype=np.float32, copy=True)
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        tensor = torch.from_numpy(X_array)
        predictions = []

        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), 4096):
                end = min(start + 4096, len(tensor))
                predictions.append(
                    self.model_(tensor[start:end].to(self.device_)).cpu().numpy()
                )
        return np.vstack(predictions)


recall_scorer = make_scorer(recall_at_10, greater_is_better=True)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — record all documented input reads. Existence guards around
    #    documented inputs are dropped. Chunked reads, intermediate files,
    #    checkpoints, test prediction, and submission output do not contribute
    #    to the cross-validated score and are omitted.
    trackers = skrub.as_data_op("input/trackers.tsv").skb.apply_func(
        pd.read_csv, sep="\t"
    )
    target_domains = skrub.as_data_op("input/target.tsv").skb.apply_func(
        pd.read_csv, sep="\t"
    )
    tracking_graph = skrub.as_data_op(
        "input/tracking_graph_train.parquet"
    ).skb.apply_func(
        pd.read_parquet,
        columns=["domain_id", "tracking_domain_id", "tracker_id"],
    )
    domains = skrub.as_data_op("input/domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain", "domain_id"]
    )
    press_tab = skrub.as_data_op(
        "input/freedom-of-the-press.csv"
    ).skb.apply_func(pd.read_csv, sep="\t")
    press_auto = skrub.as_data_op(
        "input/freedom-of-the-press.csv"
    ).skb.apply_func(pd.read_csv, sep=None, engine="python")
    press = (press_tab.shape[1] < 3).skb.if_else(press_auto, press_tab)

    url_data = skrub.as_data_op(
        "input/url-classification.csv"
    ).skb.apply_func(
        pd.read_csv, usecols=["url", "category"]
    )
    links = skrub.as_data_op("input/link-graph.parquet").skb.apply_func(
        pd.read_parquet,
        columns=["source_domain_id", "target_domain_id"],
    )

    trackers = trackers.assign(
        tracker_id=trackers["tracker_id"].astype(int),
        tracking_domain_id=trackers["tracking_domain_id"].astype(np.int64),
    )
    tracking_graph = tracking_graph.assign(
        domain_id=tracking_graph["domain_id"].astype(np.int64),
        tracker_id=tracking_graph["tracker_id"].astype(int),
    )

    # 2. Prepare data — exclude test domains, reshape the multi-label target,
    #    mark the raw target, and attach the original capped holdout splitter.
    candidates = tracking_graph[["domain_id"]].drop_duplicates()
    candidates = candidates[
        ~candidates["domain_id"].isin(target_domains["domain_id"])
    ]
    candidates = candidates.sort_values("domain_id").reset_index(drop=True)

    target_flags = (
        tracking_graph[["domain_id", "tracker_id"]]
        .drop_duplicates()
        .assign(present=np.uint8(1))
        .set_index(["domain_id", "tracker_id"])["present"]
    )
    target_wide = target_flags.unstack(fill_value=np.uint8(0))
    target_wide = target_wide.reindex(
        columns=range(NUM_TRACKERS), fill_value=np.uint8(0)
    )
    target_wide = target_wide.set_axis(TARGET_COLUMNS, axis=1).reset_index()

    dataset = candidates.merge(
        target_wide.drop_duplicates(subset="domain_id", keep="last"),
        on="domain_id",
        how="left",
        sort=False,
    ).fillna(0)

    y = dataset[TARGET_COLUMNS].skb.mark_as_y()
    X = dataset[["domain_id"]].skb.mark_as_X(
        cv=OriginalDomainHoldout(random_state=42, max_validation_size=30_000),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    domain_lookup = domains[["domain_id", "domain"]].drop_duplicates(
        subset="domain_id", keep="last"
    )
    features = X.merge(domain_lookup, on="domain_id", how="left", sort=False)
    domain_name = features["domain"].astype("string").fillna("")
    normalized_domain = domain_name.str.lower().str.strip()

    num_parts = normalized_domain.str.count(r"\.") + 1
    last = normalized_domain.str.extract(r"([^.]*)$", expand=False)
    second_last = normalized_domain.str.extract(
        r"([^.]*)\.[^.]*$", expand=False
    )
    third_last = normalized_domain.str.extract(
        r"([^.]*)\.[^.]*\.[^.]*$", expand=False
    )
    last_two = second_last + "." + last
    is_two_level = (num_parts >= 3) & last_two.isin(TWO_LEVEL_TLDS)

    tld = last.where(~is_two_level, last_two)
    root = last.where(
        num_parts < 2,
        last_two.where(~is_two_level, third_last + "." + last_two),
    )
    tld = tld.where(normalized_domain != "", "unknown").astype("string")
    root = root.where(normalized_domain != "", "unknown").astype("string")
    features = features.assign(tld=tld, root=root)

    press_key = (
        press.iloc[:, 0]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.lstrip(".")
        .rename("tld")
    )
    press_score_raw = press.iloc[:, 2].astype(float).rename("_press_score")

    # DataOps must be concatenated through the .skb namespace; calling eager
    # pandas.concat on lazy DataOps would execute pandas during plan building.
    press_lookup = press_key.skb.concat(
        [press_score_raw], axis=1
    ).drop_duplicates(subset="tld", keep="last")

    press_median_raw = press_lookup["_press_score"].median()
    press_median = (press_lookup.shape[0] == 0).skb.if_else(
        50.0, press_median_raw
    )

    features = features.merge(
        press_lookup, on="tld", how="left", sort=False
    )
    has_press = features["_press_score"].notna().astype(np.float32)
    press_score = (
        features["_press_score"].fillna(press_median).astype(np.float32)
    )

    domain_length = normalized_domain.str.len().clip(lower=1).astype(np.float32)
    num_dots = normalized_domain.str.count(r"\.").astype(np.float32)
    num_hyphens = normalized_domain.str.count("-").astype(np.float32)
    num_digits = normalized_domain.str.count(r"\d").astype(np.float32)

    kw_ecommerce = sum(
        normalized_domain.str.contains(keyword, regex=False).astype(np.float32)
        for keyword in [
            "shop", "store", "cart", "buy", "market", "mall", "deal", "pay"
        ]
    )
    kw_media = sum(
        normalized_domain.str.contains(keyword, regex=False).astype(np.float32)
        for keyword in [
            "news", "press", "media", "times", "post", "daily", "gazette",
            "journal",
        ]
    )
    kw_video = sum(
        normalized_domain.str.contains(keyword, regex=False).astype(np.float32)
        for keyword in ["video", "tv", "movie", "film", "tube", "stream"]
    )
    kw_adult = sum(
        normalized_domain.str.contains(keyword, regex=False).astype(np.float32)
        for keyword in ["adult", "sex", "porn", "xxx"]
    )
    kw_tech = sum(
        normalized_domain.str.contains(keyword, regex=False).astype(np.float32)
        for keyword in ["tech", "dev", "code", "cloud", "soft", "app", "web"]
    )
    kw_finance = sum(
        normalized_domain.str.contains(keyword, regex=False).astype(np.float32)
        for keyword in ["bank", "finance", "invest", "loan", "crypto", "coin"]
    )

    features = features.assign(
        domain_length=domain_length,
        num_dots=num_dots,
        num_hyphens=num_hyphens,
        num_digits=num_digits,
        digit_ratio=num_digits / domain_length,
        has_www=normalized_domain.str.startswith("www.").astype(np.float32),
        has_multi_subdomain=(num_dots > 1).astype(np.float32),
        press_score=press_score,
        has_press_score=has_press,
        is_authoritarian=(press_score > 60.0).astype(np.float32),
        **{
            f"tld_{value}": (tld == value).astype(np.float32)
            for value in TOP_TLDS
        },
        tld_other=(~tld.isin(TOP_TLDS)).astype(np.float32),
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
    )

    # Parse URL hosts without object-dtype list columns.
    url = url_data["url"].astype("string")
    category = url_data["category"].astype("string")
    host = (
        url.str.replace(r"^.*?://", "", regex=True)
        .str.extract(r"^([^/:]*)", expand=False)
        .str.strip()
        .str.lower()
        .astype("string")
    )
    url_rows = url_data.assign(host=host, category=category)
    url_rows = url_rows[
        url_rows["host"].notna() & url_rows["category"].notna()
    ]

    name_lookup = domains.assign(
        host=domains["domain"].astype("string").str.lower().str.strip()
    )[["host", "domain_id"]].drop_duplicates(subset="host", keep="last")
    fallback_lookup = name_lookup.rename(
        columns={
            "host": "host_without_www",
            "domain_id": "_fallback_domain_id",
        }
    )
    url_rows = url_rows.merge(
        name_lookup.rename(columns={"domain_id": "_domain_id"}),
        on="host",
        how="left",
        sort=False,
    )
    url_rows = url_rows.assign(
        host_without_www=url_rows["host"].str.replace(
            r"^www\.", "", regex=True
        )
    ).merge(
        fallback_lookup,
        on="host_without_www",
        how="left",
        sort=False,
    )
    url_rows = url_rows.assign(
        domain_id=url_rows["_domain_id"].fillna(
            url_rows["_fallback_domain_id"]
        )
    )
    url_rows = url_rows[
        url_rows["domain_id"].notna()
        & url_rows["category"].isin(URL_CATEGORIES)
    ]

    url_counts = (
        url_rows.groupby(["domain_id", "category"], sort=False)
        .size()
        .rename("count")
    )
    url_wide = url_counts.unstack(fill_value=0.0)
    url_wide = url_wide.reindex(columns=URL_CATEGORIES, fill_value=0.0)
    url_totals = url_wide.sum(axis=1)
    url_fractions = url_wide.div(url_totals, axis=0)
    url_fractions = url_fractions.set_axis(
        [f"url_category_{c}" for c in URL_CATEGORIES], axis=1
    )
    url_fractions = url_fractions.assign(
        has_url_category=(url_totals > 0).astype(np.float32)
    ).reset_index()
    features = features.merge(
        url_fractions.drop_duplicates(subset="domain_id", keep="last"),
        on="domain_id",
        how="left",
        sort=False,
    )

    out_degree = (
        links.groupby("source_domain_id", sort=False)
        .size()
        .rename("out_degree")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )
    in_degree = (
        links.groupby("target_domain_id", sort=False)
        .size()
        .rename("in_degree")
        .reset_index()
        .rename(columns={"target_domain_id": "domain_id"})
    )
    degree_table = out_degree.merge(
        in_degree, on="domain_id", how="outer", sort=False
    ).fillna(0)
    degree_table = degree_table.assign(
        total_degree=degree_table["out_degree"] + degree_table["in_degree"]
    )

    tracker_lookup = trackers[
        ["tracking_domain_id", "tracker_id"]
    ].drop_duplicates(subset="tracking_domain_id", keep="last")
    tracker_edges = links.merge(
        tracker_lookup,
        left_on="target_domain_id",
        right_on="tracking_domain_id",
        how="inner",
        sort=False,
    )
    tracker_link_counts = (
        tracker_edges.groupby("source_domain_id", sort=False)
        .size()
        .rename("tracker_out_links")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )
    direct_flags = (
        tracker_edges[["source_domain_id", "tracker_id"]]
        .drop_duplicates()
        .assign(present=np.float32(1.0))
        .set_index(["source_domain_id", "tracker_id"])["present"]
    )
    direct_wide = direct_flags.unstack(fill_value=np.float32(0.0))
    direct_wide = direct_wide.reindex(
        columns=range(NUM_TRACKERS), fill_value=np.float32(0.0)
    )
    direct_wide = (
        direct_wide.set_axis(DIRECT_COLUMNS, axis=1)
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )

    features = (
        features.merge(
            degree_table.drop_duplicates(subset="domain_id", keep="last"),
            on="domain_id",
            how="left",
            sort=False,
        )
        .merge(
            tracker_link_counts.drop_duplicates(
                subset="domain_id", keep="last"
            ),
            on="domain_id",
            how="left",
            sort=False,
        )
        .merge(
            direct_wide.drop_duplicates(subset="domain_id", keep="last"),
            on="domain_id",
            how="left",
            sort=False,
        )
    )

    out_deg = features["out_degree"].fillna(0).astype(np.float32)
    in_deg = features["in_degree"].fillna(0).astype(np.float32)
    tracker_links = (
        features["tracker_out_links"].fillna(0).astype(np.float32)
    )

    out_neighbor_edges = links.rename(
        columns={
            "source_domain_id": "domain_id",
            "target_domain_id": "neighbor_domain_id",
        }
    )
    out_neighbor_edges = out_neighbor_edges.merge(
        degree_table[["domain_id", "total_degree"]]
        .rename(
            columns={
                "domain_id": "neighbor_domain_id",
                "total_degree": "_neighbor_degree",
            }
        )
        .drop_duplicates(subset="neighbor_domain_id", keep="last"),
        on="neighbor_domain_id",
        how="left",
        sort=False,
    )
    out_neighbor_edges = out_neighbor_edges.assign(
        weight=1.0
        / (
            out_neighbor_edges["_neighbor_degree"]
            .fillna(0)
            .clip(lower=1)
            .skb.apply_func(np.log1p)
        )
    )[["domain_id", "neighbor_domain_id", "weight"]]

    in_neighbor_edges = links.rename(
        columns={
            "target_domain_id": "domain_id",
            "source_domain_id": "neighbor_domain_id",
        }
    )
    in_neighbor_edges = in_neighbor_edges.merge(
        degree_table[["domain_id", "total_degree"]]
        .rename(
            columns={
                "domain_id": "neighbor_domain_id",
                "total_degree": "_neighbor_degree",
            }
        )
        .drop_duplicates(subset="neighbor_domain_id", keep="last"),
        on="neighbor_domain_id",
        how="left",
        sort=False,
    )
    in_neighbor_edges = in_neighbor_edges.assign(
        weight=1.0
        / (
            in_neighbor_edges["_neighbor_degree"]
            .fillna(0)
            .clip(lower=1)
            .skb.apply_func(np.log1p)
        )
    )[["domain_id", "neighbor_domain_id", "weight"]]

    out_neighbor = features[["domain_id"]].skb.apply(
        NeighborAdoption(direction="out"),
        y=y,
        fit_transform_kwargs={"table": out_neighbor_edges},
        transform_kwargs={"table": out_neighbor_edges},
    )
    in_neighbor = features[["domain_id"]].skb.apply(
        NeighborAdoption(direction="in"),
        y=y,
        fit_transform_kwargs={"table": in_neighbor_edges},
        transform_kwargs={"table": in_neighbor_edges},
    )

    features = features.skb.concat([out_neighbor, in_neighbor], axis=1)
    out_neighbor_degree = features["out_neighbor_degree"].astype(np.float32)
    in_neighbor_degree = features["in_neighbor_degree"].astype(np.float32)
    neighbor_total_degree = out_neighbor_degree + in_neighbor_degree

    features = features.assign(
        out_degree=out_deg,
        log_out_degree=out_deg.skb.apply_func(np.log1p),
        in_degree=in_deg,
        log_in_degree=in_deg.skb.apply_func(np.log1p),
        total_degree=out_deg + in_deg,
        in_out_ratio=(in_deg + 1.0) / (out_deg + 1.0),
        is_isolated=((out_deg + in_deg) == 0).astype(np.float32),
        tracker_out_links=tracker_links,
        log_tracker_out_links=tracker_links.skb.apply_func(np.log1p),
        tracker_link_ratio=tracker_links / out_deg.clip(lower=1.0),
        log_out_neighbor_degree=out_neighbor_degree.skb.apply_func(np.log1p),
        log_in_neighbor_degree=in_neighbor_degree.skb.apply_func(np.log1p),
        neighbor_total_degree=neighbor_total_degree,
        log_neighbor_total_degree=neighbor_total_degree.skb.apply_func(
            np.log1p
        ),
    )

    tld_prior = features[["tld"]].skb.apply(
        TldTrackerPrior(alpha=10.0), y=y
    )
    root_input = features[["root"]].skb.concat([tld_prior], axis=1)
    root_prior = root_input.skb.apply(
        RootTrackerPrior(beta=2.0), y=y
    )
    features = features.skb.concat([tld_prior, root_prior], axis=1)

    root_count = features["root_count"].astype(np.float32)
    root_matrix = features[ROOT_PRIOR_COLUMNS]
    root_entropy = -(
        root_matrix * (root_matrix + 1e-12).skb.apply_func(np.log)
    ).sum(axis=1)

    features = features.assign(
        log_root_count=root_count.skb.apply_func(np.log1p),
        root_prior_entropy=root_entropy.astype(np.float32),
        root_prior_max=root_matrix.max(axis=1).astype(np.float32),
    )

    # Re-impose the original column_stack order because TrackerRankNet consumes
    # the final five 355-column blocks by positional slicing.
    model_features = (
        features[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )

    # These fold-training-only statistics are recorded separately and supplied
    # to the predictor with fit_kwargs.
    global_tracker_priors = y.mean(axis=0).astype(np.float32)
    cooccurrence_counts = y.T.dot(y).astype(np.float32)
    tracker_frequencies = cooccurrence_counts.skb.apply_func(np.diag)
    conditional_cooccurrence = cooccurrence_counts.div(
        tracker_frequencies + 10.0, axis=0
    )
    conditional_cooccurrence = conditional_cooccurrence.where(
        ~np.eye(NUM_TRACKERS, dtype=bool), 0.0
    )
    row_maximum = conditional_cooccurrence.max(axis=1)
    row_denominator = row_maximum.skb.apply_func(np.maximum, 1.0)
    normalized_cooccurrence = (
        conditional_cooccurrence.div(row_denominator, axis=0)
        .where(row_maximum > 0, 0.0)
        .astype(np.float32)
    )

    model = TorchTrackerRanker(
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
    )
    pred = model_features.skb.apply(
        model,
        y=y,
        fit_kwargs={
            "global_priors": global_tracker_priors,
            "cooccurrence_matrix": normalized_cooccurrence,
        },
    )

    # 4. Score. No cv= here; the splitter on mark_as_X drives.
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
