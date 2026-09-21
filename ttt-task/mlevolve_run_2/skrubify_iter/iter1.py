import math
import os

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.metrics._scorer import _SCORERS
from sklearn.model_selection import BaseCrossValidator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# The original explicitly describes 355 tracker targets. Estimator dimensions
# must be concrete while the lazy plan is constructed.
NUM_TRACKERS = 355
TRACKER_COLS = [f"tracker_{i}" for i in range(NUM_TRACKERS)]

TOP_TLDS = [
    "com", "ru", "org", "net", "de", "uk", "jp", "fr", "it", "pl",
    "br", "cn", "in", "nl", "es", "cz", "eu", "ua", "ca", "au",
    "ch", "se", "ro", "gr", "at", "tv", "io", "me", "co", "info",
]
TLD_COLS = [f"tld_{tld}" for tld in TOP_TLDS] + ["tld_other"]

URL_CATEGORIES = [
    "Arts", "Business", "Computers", "Games", "Health", "Home",
    "Kids_and_Teens", "News", "Recreation", "Reference", "Regional",
    "Science", "Shopping", "Society", "Sports",
]
URL_CAT_COLS = [f"url_cat_{category}" for category in URL_CATEGORIES]

DIRECT_COLS = [f"direct_{i}" for i in range(NUM_TRACKERS)]
OUT_NEIGHBOR_COLS = [f"out_neighbor_{i}" for i in range(NUM_TRACKERS)]
IN_NEIGHBOR_COLS = [f"in_neighbor_{i}" for i in range(NUM_TRACKERS)]
ROOT_PRIOR_COLS = [f"root_prior_{i}" for i in range(NUM_TRACKERS)]
TLD_PRIOR_COLS = [f"tld_prior_{i}" for i in range(NUM_TRACKERS)]

LEXICAL_COLS = [
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
    *TLD_COLS,
    "kw_ecommerce",
    "kw_media",
    "kw_video",
    "kw_adult",
    "kw_tech",
    "kw_finance",
    "kw_total",
]

GRAPH_COLS = [
    "out_degree",
    "log_out_degree",
    "in_degree",
    "log_in_degree",
    "total_degree",
    "in_out_ratio",
    "is_isolated",
    "tracker_link_count",
    "log_tracker_link_count",
    "tracker_link_ratio",
    "weighted_out_neighbor_degree",
    "log_weighted_out_neighbor_degree",
    "weighted_in_neighbor_degree",
    "log_weighted_in_neighbor_degree",
    "weighted_total_neighbor_degree",
    "log_weighted_total_neighbor_degree",
]

PRIOR_SCALAR_COLS = [
    "has_root_match",
    "root_count",
    "log_root_count",
    "root_prior_entropy",
    "root_prior_max",
]

SCALAR_COLS = (
    LEXICAL_COLS
    + URL_CAT_COLS
    + ["has_url_category"]
    + GRAPH_COLS
    + PRIOR_SCALAR_COLS
)

# TrackerRankNet reads the final 5 * 355 columns positionally, so the original
# np.hstack order is reimposed explicitly before fitting.
FEATURE_COLS = (
    SCALAR_COLS
    + DIRECT_COLS
    + OUT_NEIGHBOR_COLS
    + IN_NEIGHBOR_COLS
    + ROOT_PRIOR_COLS
    + TLD_PRIOR_COLS
)

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


def read_optional_press_freedom(path):
    """Read and normalize the original optional press-freedom source.

    This intentionally remains one recorded I/O operation: file existence,
    parser fallback, and positional source-column interpretation cannot be
    expressed without first executing the read.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["tld", "press_score"])

    try:
        frame = pd.read_csv(path, sep="\t")
        if len(frame.columns) < 3:
            frame = pd.read_csv(path, sep=None, engine="python")

        return pd.DataFrame(
            {
                "tld": (
                    frame.iloc[:, 0]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.lstrip(".")
                ),
                "press_score": pd.to_numeric(
                    frame.iloc[:, 2],
                    errors="coerce",
                ),
            }
        ).dropna(subset=["press_score"])
    except Exception:
        return pd.DataFrame(columns=["tld", "press_score"])


def read_optional_url_categories(path):
    """Read the optional URL source.

    The original chunking controlled memory only and did not alter its rows.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=["url", "category"])
    return pd.read_csv(path, usecols=["url", "category"])


def extract_tld(domain):
    if not isinstance(domain, str) or not domain:
        return "unknown"

    parts = domain.lower().strip().split(".")
    if len(parts) == 1:
        return parts[0]

    if len(parts) >= 3:
        last_two = f"{parts[-2]}.{parts[-1]}"
        if last_two in TWO_LEVEL_TLDS:
            return last_two

    return parts[-1]


def extract_root(domain):
    if not isinstance(domain, str) or not domain:
        return "unknown"

    parts = domain.lower().strip().split(".")
    if len(parts) == 1:
        return parts[0]

    if len(parts) >= 3:
        last_two = f"{parts[-2]}.{parts[-1]}"
        if last_two in TWO_LEVEL_TLDS:
            return f"{parts[-3]}.{last_two}"

    return f"{parts[-2]}.{parts[-1]}"


def extract_tld_series(series):
    # Public-suffix-like parsing is one operation not provided by pandas.
    return series.map(extract_tld)


def extract_root_series(series):
    # Root-domain parsing is one operation not provided by pandas.
    return series.map(extract_root)


def normalize_cooccurrence(cooccurrence):
    """Perform the original ndarray co-occurrence normalization."""
    values = np.asarray(cooccurrence, dtype=np.float32).copy()
    tracker_frequencies = np.diag(values).copy()
    conditional = values / (tracker_frequencies[:, None] + 10.0)
    np.fill_diagonal(conditional, 0.0)

    row_max = conditional.max(axis=1, keepdims=True)
    return np.where(
        row_max > 0.0,
        conditional / np.maximum(row_max, 1.0),
        0.0,
    ).astype(np.float32)


class OriginalDomainHoldout(BaseCrossValidator):
    """The original seeded shuffle followed by a capped 20% holdout."""

    def __init__(self, random_state=42, max_validation_rows=30000):
        self.random_state = random_state
        self.max_validation_rows = max_validation_rows

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        indices = np.arange(len(X))
        random_state = np.random.RandomState(self.random_state)
        random_state.shuffle(indices)
        n_validation = min(
            self.max_validation_rows,
            int(len(indices) * 0.2),
        )
        yield indices[n_validation:], indices[:n_validation]


class TLDTrackerPrior(TransformerMixin, BaseEstimator):
    """Learn smoothed TLD tracker priors from each fold's training rows."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, y):
        targets = np.asarray(y, dtype=np.float32)
        self.global_prior_ = targets.mean(axis=0)

        work = pd.DataFrame(
            targets,
            columns=TRACKER_COLS,
            index=X.index,
        )
        work["tld"] = X["tld"].astype(str).to_numpy()

        self.tracker_counts_ = work.groupby(
            "tld", sort=False
        )[TRACKER_COLS].sum()
        self.domain_counts_ = work.groupby(
            "tld", sort=False
        ).size()
        return self

    def transform(self, X):
        output = np.empty(
            (len(X), NUM_TRACKERS),
            dtype=np.float32,
        )

        for row_index, tld in enumerate(X["tld"].astype(str).to_numpy()):
            if tld in self.domain_counts_.index:
                output[row_index] = (
                    self.tracker_counts_.loc[tld].to_numpy(dtype=np.float32)
                    + self.alpha * self.global_prior_
                ) / (
                    float(self.domain_counts_.loc[tld]) + self.alpha
                )
            else:
                output[row_index] = self.global_prior_

        return pd.DataFrame(
            output,
            columns=TLD_PRIOR_COLS,
            index=X.index,
        )


class RootTrackerPrior(TransformerMixin, BaseEstimator):
    """Learn root priors with LOO fit_transform and ordinary transform."""

    def __init__(self, beta=2.0):
        self.beta = beta

    def fit(self, X, y):
        targets = np.asarray(y, dtype=np.float32)

        work = pd.DataFrame(
            targets,
            columns=TRACKER_COLS,
            index=X.index,
        )
        work["root"] = X["root"].astype(str).to_numpy()

        self.tracker_counts_ = work.groupby(
            "root", sort=False
        )[TRACKER_COLS].sum()
        self.domain_counts_ = work.groupby(
            "root", sort=False
        ).size()
        return self

    def _make_output(self, X, training_targets=None):
        tld_priors = X[TLD_PRIOR_COLS].to_numpy(dtype=np.float32)
        roots = X["root"].astype(str).to_numpy()

        root_priors = tld_priors.copy()
        has_root_match = np.zeros(len(X), dtype=np.float32)
        root_counts = np.zeros(len(X), dtype=np.float32)

        for row_index, root in enumerate(roots):
            if root not in self.domain_counts_.index:
                continue

            count = int(self.domain_counts_.loc[root])
            tracker_count = self.tracker_counts_.loc[root].to_numpy(
                dtype=np.float32
            )

            if training_targets is not None:
                if count <= 1:
                    continue
                count -= 1
                tracker_count = tracker_count - training_targets[row_index]

            root_priors[row_index] = (
                tracker_count + self.beta * tld_priors[row_index]
            ) / (float(count) + self.beta)
            has_root_match[row_index] = 1.0
            root_counts[row_index] = float(count)

        entropy = -np.sum(
            root_priors * np.log(root_priors + 1e-12),
            axis=1,
        ).astype(np.float32)

        return pd.DataFrame(
            np.column_stack(
                [
                    has_root_match,
                    root_counts,
                    np.log1p(root_counts),
                    entropy,
                    root_priors.max(axis=1).astype(np.float32),
                    root_priors,
                ]
            ),
            columns=PRIOR_SCALAR_COLS + ROOT_PRIOR_COLS,
            index=X.index,
        )

    def fit_transform(self, X, y, **fit_params):
        self.fit(X, y)
        return self._make_output(
            X,
            training_targets=np.asarray(y, dtype=np.float32),
        )

    def transform(self, X):
        return self._make_output(X)


class NeighborTrackerFeatures(TransformerMixin, BaseEstimator):
    """Fold-trained Adamic-Adar neighbor target aggregation."""

    def __init__(self, direction="out"):
        self.direction = direction

    def fit(self, X, y, links=None):
        self.links_ = links[
            ["source_domain_id", "target_domain_id", "total_degree"]
        ].copy()

        self.target_by_domain_ = pd.DataFrame(
            np.asarray(y, dtype=np.float32),
            columns=TRACKER_COLS,
        )
        self.target_by_domain_.insert(
            0,
            "neighbor_domain_id",
            X["domain_id"].to_numpy(),
        )
        return self

    def fit_transform(self, X, y, links=None, **fit_params):
        self.fit(X, y, links=links)
        return self.transform(X)

    def transform(self, X):
        domains = pd.DataFrame(
            {
                "row_id": np.arange(len(X)),
                "domain_id": X["domain_id"].to_numpy(),
            }
        )

        if self.direction == "out":
            edges = domains.merge(
                self.links_,
                left_on="domain_id",
                right_on="source_domain_id",
                how="left",
            )
            edges = edges[
                edges["source_domain_id"] != edges["target_domain_id"]
            ].copy()
            edges["neighbor_domain_id"] = edges["target_domain_id"]
            output_columns = OUT_NEIGHBOR_COLS
            degree_column = "weighted_out_neighbor_degree"
            log_degree_column = "log_weighted_out_neighbor_degree"
        else:
            edges = domains.merge(
                self.links_,
                left_on="domain_id",
                right_on="target_domain_id",
                how="left",
            )
            edges = edges[
                edges["source_domain_id"] != edges["target_domain_id"]
            ].copy()
            edges["neighbor_domain_id"] = edges["source_domain_id"]
            output_columns = IN_NEIGHBOR_COLS
            degree_column = "weighted_in_neighbor_degree"
            log_degree_column = "log_weighted_in_neighbor_degree"

        edges["weight"] = (
            1.0
            / np.log1p(
                np.maximum(
                    edges["total_degree"].fillna(0.0).to_numpy(),
                    1.0,
                )
            )
        ).astype(np.float32)

        edges = edges.merge(
            self.target_by_domain_,
            on="neighbor_domain_id",
            how="inner",
        )

        weighted_targets = (
            edges[TRACKER_COLS].to_numpy(dtype=np.float32)
            * edges["weight"].to_numpy(dtype=np.float32)[:, None]
        )

        weighted = pd.DataFrame(
            weighted_targets,
            columns=TRACKER_COLS,
        )
        weighted["row_id"] = edges["row_id"].to_numpy()
        weighted["weight"] = edges["weight"].to_numpy(dtype=np.float32)

        target_sums = weighted.groupby(
            "row_id", sort=False
        )[TRACKER_COLS].sum()
        weight_sums = weighted.groupby(
            "row_id", sort=False
        )["weight"].sum()

        target_sums = target_sums.reindex(
            range(len(X)),
            fill_value=0.0,
        )
        weight_sums = weight_sums.reindex(
            range(len(X)),
            fill_value=0.0,
        )

        weight_values = weight_sums.to_numpy(dtype=np.float32)
        normalized = np.where(
            weight_values[:, None] > 0.0,
            target_sums.to_numpy(dtype=np.float32)
            / np.maximum(weight_values[:, None], 1e-7),
            0.0,
        ).astype(np.float32)

        return pd.DataFrame(
            np.column_stack(
                [
                    weight_values,
                    np.log1p(weight_values),
                    normalized,
                ]
            ),
            columns=[
                degree_column,
                log_degree_column,
                *output_columns,
            ],
            index=X.index,
        )


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
            nn.Linear(in_dim, out_dim)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        output = self.fc1(x)
        output = self.norm1(output)
        output = self.act1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.norm2(output)
        return self.act2(output + residual)


class TrackerRankNet(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes=355,
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
        self.has_tracker_priors = (
            in_dim >= num_prior_channels * num_classes
        )

        self.input_norm = nn.BatchNorm1d(in_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        self.res_block1 = ResidualTabularBlock(
            hidden_dim,
            hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.res_block2 = ResidualTabularBlock(
            hidden_dim,
            hidden_dim // 2,
            dropout_rate=dropout_rate,
        )

        bottleneck_dim = hidden_dim // 2
        self.direct_head = nn.Linear(bottleneck_dim, num_classes)
        self.domain_latent_proj = nn.Linear(
            bottleneck_dim,
            latent_dim,
        )
        self.tracker_prototypes = nn.Parameter(
            torch.randn(num_classes, latent_dim)
            / math.sqrt(latent_dim)
        )
        self.head_blend = nn.Parameter(torch.tensor([0.5]))

        if self.has_tracker_priors:
            self.film_prior = nn.Linear(
                bottleneck_dim,
                num_prior_channels * 2,
            )
            with torch.no_grad():
                nn.init.zeros_(self.film_prior.weight)
                nn.init.zeros_(self.film_prior.bias)

            initial_weights = torch.tensor(
                [1.5, 1.0, 1.0, 1.5, 0.8],
                dtype=torch.float32,
            ).unsqueeze(0).repeat(num_classes, 1)

            self.tracker_prior_weights = nn.Parameter(initial_weights)
            self.tracker_prior_bias = nn.Parameter(
                torch.zeros(num_classes)
            )
            self.tracker_res_net = nn.Sequential(
                nn.Linear(num_prior_channels, 32),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, 1),
            )

            with torch.no_grad():
                nn.init.normal_(
                    self.tracker_res_net[0].weight,
                    std=0.01,
                )
                nn.init.zeros_(self.tracker_res_net[0].bias)
                nn.init.zeros_(self.tracker_res_net[3].weight)
                nn.init.zeros_(self.tracker_res_net[3].bias)

            self.res_scale = nn.Parameter(torch.tensor([0.5]))

        self.cooc_layer = nn.Linear(
            num_classes,
            num_classes,
            bias=False,
        )
        with torch.no_grad():
            if cooccurrence_matrix is not None:
                self.cooc_layer.weight.copy_(
                    torch.as_tensor(
                        cooccurrence_matrix.T,
                        dtype=torch.float32,
                    )
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
        hidden = self.input_norm(x)
        hidden = self.input_proj(hidden)
        hidden = self.res_block1(hidden)
        hidden = self.res_block2(hidden)

        direct_logits = self.direct_head(hidden)
        domain_latent = F.normalize(
            self.domain_latent_proj(hidden),
            p=2,
            dim=-1,
        )
        tracker_latent = F.normalize(
            self.tracker_prototypes,
            p=2,
            dim=-1,
        )
        prototype_logits = (
            torch.matmul(domain_latent, tracker_latent.t())
            * (math.sqrt(self.latent_dim) / 2.0)
        )

        alpha = torch.sigmoid(self.head_blend)
        base_logits = (
            alpha * direct_logits
            + (1.0 - alpha) * prototype_logits
        )

        if self.has_tracker_priors:
            prior_signals = x[
                :,
                -self.num_prior_channels * self.num_classes :,
            ].reshape(
                -1,
                self.num_prior_channels,
                self.num_classes,
            )
            prior_permuted = prior_signals.permute(0, 2, 1)

            film_parameters = self.film_prior(hidden)
            gamma, beta = film_parameters.chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            modulated = prior_permuted * gamma + beta

            direct_prior_logits = (
                modulated * self.tracker_prior_weights
            ).sum(dim=-1) + self.tracker_prior_bias
            mlp_prior_logits = self.tracker_res_net(
                modulated
            ).squeeze(-1)

            base_logits = base_logits + self.res_scale * (
                direct_prior_logits + mlp_prior_logits
            )

        probabilities = torch.sigmoid(base_logits)
        first_message = self.cooc_layer(probabilities)
        second_message = self.cooc_layer(first_message)
        relational_message = (
            first_message + self.rel_refine(second_message)
        )
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
            min=self.eps,
            max=1.0 - self.eps,
        )
        positive_loss = (
            -targets
            * (1.0 - positive_probabilities).pow(self.gamma_pos)
            * torch.log(positive_probabilities)
        )

        negative_probabilities = (
            probabilities - self.clip_margin
        ).clamp(
            min=self.eps,
            max=1.0 - self.eps,
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

        return (
            positive_loss + negative_loss
        ).sum(dim=-1).mean()


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
        self.asymmetric_focal = AsymmetricFocalRecallLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip_margin=clip_margin,
            eps=eps,
        )
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.rank_weight = rank_weight

    def forward(self, logits, targets):
        focal_loss = self.asymmetric_focal(logits, targets)

        negative_logits = torch.where(
            targets == 0,
            logits,
            torch.full_like(logits, -1e9),
        )
        hard_negative_logits, _ = torch.topk(
            negative_logits,
            k=self.top_k_neg,
            dim=-1,
        )

        margin_difference = self.margin - (
            logits.unsqueeze(-1)
            - hard_negative_logits.unsqueeze(1)
        )
        ranking_violation = F.relu(margin_difference)

        positive_mask = targets.unsqueeze(-1)
        positive_count = targets.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1.0)

        rank_loss = (
            (ranking_violation * positive_mask).sum(dim=(1, 2))
            / (positive_count.squeeze(-1) * self.top_k_neg)
        ).mean()

        return focal_loss + self.rank_weight * rank_loss


def recall_at_10(y_true, y_pred):
    targets = np.asarray(y_true)
    predictions = np.asarray(y_pred)

    top_k_indices = np.argpartition(
        predictions,
        -10,
        axis=1,
    )[:, -10:]
    row_indices = np.arange(len(predictions))[:, None]
    hits = targets[row_indices, top_k_indices].sum(axis=1)
    true_counts = targets.sum(axis=1)

    recalls = np.where(
        true_counts > 0,
        hits / true_counts,
        0.0,
    )
    return float(np.mean(recalls))


class TrackerRankNetEstimator(RegressorMixin, BaseEstimator):
    """Run the original custom PyTorch training loop per outer fit."""

    def __init__(
        self,
        num_classes=355,
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
        max_inner_validation_rows=30000,
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
        self.max_inner_validation_rows = max_inner_validation_rows

    def fit(
        self,
        X,
        y,
        global_tracker_priors=None,
        cooccurrence_matrix=None,
    ):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        feature_values = np.ascontiguousarray(
            np.nan_to_num(
                np.asarray(X, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        target_values = np.ascontiguousarray(
            np.asarray(y, dtype=np.float32)
        )

        # The original selected checkpoints on the same holdout it reported.
        # Honest outer CV cannot reproduce that leakage, so checkpoint selection
        # uses the same seeded, capped 20% split from this fold's training rows.
        permutation = np.arange(len(feature_values))
        random_state = np.random.RandomState(self.random_state)
        random_state.shuffle(permutation)
        n_validation = min(
            self.max_inner_validation_rows,
            int(len(permutation) * self.inner_validation_fraction),
        )

        validation_indices = permutation[:n_validation]
        training_indices = permutation[n_validation:]

        X_train = feature_values[training_indices]
        y_train = target_values[training_indices]
        X_validation = feature_values[validation_indices]
        y_validation = target_values[validation_indices]

        if global_tracker_priors is None:
            priors = target_values.mean(axis=0)
        else:
            priors = np.asarray(
                global_tracker_priors,
                dtype=np.float32,
            ).reshape(-1)

        clipped_priors = np.clip(priors, 1e-5, 1.0 - 1e-5)
        initial_bias = np.log(
            clipped_priors / (1.0 - clipped_priors)
        ).astype(np.float32)

        if cooccurrence_matrix is None:
            cooccurrence = normalize_cooccurrence(
                target_values.T.dot(target_values)
            )
        else:
            cooccurrence = np.asarray(
                cooccurrence_matrix,
                dtype=np.float32,
            )

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_ = TrackerRankNet(
            in_dim=feature_values.shape[1],
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

        decay_parameters = []
        no_decay_parameters = []
        for name, parameter in self.model_.named_parameters():
            if not parameter.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)

        optimizer = AdamW(
            [
                {
                    "params": decay_parameters,
                    "weight_decay": 1e-4,
                },
                {
                    "params": no_decay_parameters,
                    "weight_decay": 0.0,
                },
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

        batches_per_epoch = max(
            math.ceil(len(X_train_tensor) / self.batch_size),
            1,
        )
        best_validation_recall = -1.0
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model_.state_dict().items()
        }

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            random_order = torch.randperm(len(X_train_tensor))

            for batch_number, start in enumerate(
                range(0, len(X_train_tensor), self.batch_size)
            ):
                if epoch == 1:
                    warmup_step = batch_number + 1
                    learning_rate = self.min_lr + (
                        self.base_lr - self.min_lr
                    ) * (warmup_step / batches_per_epoch)
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = learning_rate

                batch_indices = random_order[
                    start : start + self.batch_size
                ]
                batch_X = X_train_tensor[batch_indices].to(
                    self.device_,
                    non_blocking=True,
                )
                batch_y = y_train_tensor[batch_indices].to(
                    self.device_,
                    non_blocking=True,
                )

                optimizer.zero_grad()
                logits = self.model_(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model_.parameters(),
                    max_norm=1.0,
                )
                optimizer.step()

            if epoch > 1:
                scheduler.step()

            validation_logits = self._predict_array(X_validation)
            validation_recall = recall_at_10(
                y_validation,
                validation_logits,
            )

            if validation_recall > best_validation_recall:
                best_validation_recall = validation_recall
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model_.state_dict().items()
                }

        self.model_.load_state_dict(
            {
                name: value.to(self.device_)
                for name, value in best_state.items()
            }
        )
        self.model_.eval()
        return self

    def _predict_array(self, X):
        feature_values = np.ascontiguousarray(
            np.nan_to_num(
                np.asarray(X, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )
        feature_tensor = torch.from_numpy(feature_values)

        outputs = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(feature_tensor), 4096):
                logits = self.model_(
                    feature_tensor[start : start + 4096].to(self.device_)
                )
                outputs.append(logits.cpu().numpy())

        return np.vstack(outputs)

    def predict(self, X):
        return self._predict_array(X)


recall_at_10_scorer = make_scorer(
    recall_at_10,
    greater_is_better=True,
    response_method="predict",
)

# This installed skrub version has no DataOp.with_scoring. Register the exact
# custom scorer under a scorer string used by make_grid_search.
_SCORERS["recall_at_10"] = recall_at_10_scorer


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — recorded reads. Intermediate JSON/NPZ/checkpoint files,
    # test inference, submission output, and chunking are omitted because they
    # do not contribute to the validation score.
    trackers_raw = (
        skrub.as_data_op("input/trackers.tsv")
        .skb.apply_func(pd.read_csv, sep="\t")
    )
    trackers = trackers_raw.assign(
        tracker_id=trackers_raw["tracker_id"].astype(int),
        tracking_domain_id=trackers_raw[
            "tracking_domain_id"
        ].astype(np.int64),
    )

    target_domains = (
        skrub.as_data_op("input/target.tsv")
        .skb.apply_func(pd.read_csv, sep="\t")
    )

    tracking_graph = (
        skrub.as_data_op("input/tracking_graph_train.parquet")
        .skb.apply_func(
            pd.read_parquet,
            columns=[
                "domain_id",
                "tracking_domain_id",
                "tracker_id",
            ],
        )
    )

    domains = (
        skrub.as_data_op("input/domains.parquet")
        .skb.apply_func(
            pd.read_parquet,
            columns=["domain", "domain_id"],
        )
    )

    links = (
        skrub.as_data_op("input/link-graph.parquet")
        .skb.apply_func(
            pd.read_parquet,
            columns=["source_domain_id", "target_domain_id"],
        )
    )

    press = (
        skrub.as_data_op("input/freedom-of-the-press.csv")
        .skb.apply_func(read_optional_press_freedom)
    )

    url_categories = (
        skrub.as_data_op("input/url-classification.csv")
        .skb.apply_func(read_optional_url_categories)
    )

    # 2. Prepare data — make the raw 355-column multilabel target through a
    # recorded long-to-wide reshape. Test domains are excluded before marking.
    target_edges = (
        tracking_graph[["domain_id", "tracker_id"]]
        .drop_duplicates()
        .assign(present=np.uint8(1))
        .set_index(["domain_id", "tracker_id"])["present"]
        .unstack(fill_value=np.uint8(0))
        .reindex(
            columns=range(NUM_TRACKERS),
            fill_value=np.uint8(0),
        )
        .set_axis(TRACKER_COLS, axis=1)
        .reset_index()
    )

    candidate_rows = target_edges[
        ~target_edges["domain_id"].isin(target_domains["domain_id"])
    ].reset_index(drop=True)

    y = candidate_rows[TRACKER_COLS].skb.mark_as_y()
    X = candidate_rows[["domain_id"]].skb.mark_as_X(
        cv=OriginalDomainHoldout(
            random_state=42,
            max_validation_rows=30000,
        ),
        split_kwargs={},
    )

    # 3. Recorded preprocessing and feature engineering.
    domain_features = X.merge(
        domains,
        on="domain_id",
        how="left",
    )
    domain_names = domain_features["domain"].fillna("")

    tld = domain_names.skb.apply_func(extract_tld_series)
    root = domain_names.skb.apply_func(extract_root_series)

    lexical = domain_features.assign(
        tld=tld,
        root=root,
        domain_length=domain_names.str.len()
        .clip(lower=1)
        .astype(np.float32),
        num_dots=domain_names.str.count(r"\.").astype(np.float32),
        num_hyphens=domain_names.str.count("-").astype(np.float32),
        num_digits=domain_names.str.count(r"\d").astype(np.float32),
        has_www=domain_names.str.lower()
        .str.startswith("www.")
        .astype(np.float32),
    )

    lexical = lexical.assign(
        digit_ratio=lexical["num_digits"] / lexical["domain_length"],
        has_multi_subdomain=(
            lexical["num_dots"] > 1
        ).astype(np.float32),
    )

    press_median = press["press_score"].median()
    lexical = lexical.merge(
        press,
        on="tld",
        how="left",
    )
    lexical = lexical.assign(
        has_press_score=lexical["press_score"]
        .notna()
        .astype(np.float32),
        press_score=lexical["press_score"]
        .fillna(press_median)
        .fillna(50.0)
        .astype(np.float32),
    )
    lexical = lexical.assign(
        is_authoritarian=(
            lexical["press_score"] > 60.0
        ).astype(np.float32),
        tld_category=lexical["tld"].where(
            lexical["tld"].isin(TOP_TLDS),
            "other",
        ),
    )

    lower_names = domain_names.str.lower()
    lexical = lexical.assign(
        kw_ecommerce=(
            lower_names.str.contains("shop", regex=False).astype(float)
            + lower_names.str.contains("store", regex=False).astype(float)
            + lower_names.str.contains("cart", regex=False).astype(float)
            + lower_names.str.contains("buy", regex=False).astype(float)
            + lower_names.str.contains("market", regex=False).astype(float)
            + lower_names.str.contains("mall", regex=False).astype(float)
            + lower_names.str.contains("deal", regex=False).astype(float)
            + lower_names.str.contains("pay", regex=False).astype(float)
        ),
        kw_media=(
            lower_names.str.contains("news", regex=False).astype(float)
            + lower_names.str.contains("press", regex=False).astype(float)
            + lower_names.str.contains("media", regex=False).astype(float)
            + lower_names.str.contains("times", regex=False).astype(float)
            + lower_names.str.contains("post", regex=False).astype(float)
            + lower_names.str.contains("daily", regex=False).astype(float)
            + lower_names.str.contains("gazette", regex=False).astype(float)
            + lower_names.str.contains("journal", regex=False).astype(float)
        ),
        kw_video=(
            lower_names.str.contains("video", regex=False).astype(float)
            + lower_names.str.contains("tv", regex=False).astype(float)
            + lower_names.str.contains("movie", regex=False).astype(float)
            + lower_names.str.contains("film", regex=False).astype(float)
            + lower_names.str.contains("tube", regex=False).astype(float)
            + lower_names.str.contains("stream", regex=False).astype(float)
        ),
        kw_adult=(
            lower_names.str.contains("adult", regex=False).astype(float)
            + lower_names.str.contains("sex", regex=False).astype(float)
            + lower_names.str.contains("porn", regex=False).astype(float)
            + lower_names.str.contains("xxx", regex=False).astype(float)
        ),
        kw_tech=(
            lower_names.str.contains("tech", regex=False).astype(float)
            + lower_names.str.contains("dev", regex=False).astype(float)
            + lower_names.str.contains("code", regex=False).astype(float)
            + lower_names.str.contains("cloud", regex=False).astype(float)
            + lower_names.str.contains("soft", regex=False).astype(float)
            + lower_names.str.contains("app", regex=False).astype(float)
            + lower_names.str.contains("web", regex=False).astype(float)
        ),
        kw_finance=(
            lower_names.str.contains("bank", regex=False).astype(float)
            + lower_names.str.contains("finance", regex=False).astype(float)
            + lower_names.str.contains("invest", regex=False).astype(float)
            + lower_names.str.contains("loan", regex=False).astype(float)
            + lower_names.str.contains("crypto", regex=False).astype(float)
            + lower_names.str.contains("coin", regex=False).astype(float)
        ),
    )

    lexical = lexical.assign(
        kw_total=(
            lexical["kw_ecommerce"]
            + lexical["kw_media"]
            + lexical["kw_video"]
            + lexical["kw_adult"]
            + lexical["kw_tech"]
            + lexical["kw_finance"]
        )
    )

    tld_flags = (
        lexical[["domain_id", "tld_category"]]
        .assign(present=np.float32(1.0))
        .set_index(["domain_id", "tld_category"])["present"]
        .unstack(fill_value=np.float32(0.0))
        .reindex(
            columns=TOP_TLDS + ["other"],
            fill_value=np.float32(0.0),
        )
        .set_axis(TLD_COLS, axis=1)
        .reset_index()
    )
    lexical = lexical.merge(
        tld_flags,
        on="domain_id",
        how="left",
    )

    url_host = (
        url_categories["url"]
        .astype("string")
        .str.replace(r"^[^:]+://", "", regex=True)
        .str.split("/", n=1)
        .str[0]
        .str.split(":", n=1)
        .str[0]
        .str.strip()
        .str.lower()
    )

    url_rows = url_categories.assign(host=url_host)
    exact_domains = domains.rename(
        columns={
            "domain": "host",
            "domain_id": "exact_domain_id",
        }
    )
    url_rows = url_rows.merge(
        exact_domains,
        on="host",
        how="left",
    )

    url_rows = url_rows.assign(
        host_without_www=url_rows["host"].str.replace(
            r"^www\.",
            "",
            regex=True,
        )
    )
    fallback_domains = domains.rename(
        columns={
            "domain": "host_without_www",
            "domain_id": "fallback_domain_id",
        }
    )
    url_rows = url_rows.merge(
        fallback_domains,
        on="host_without_www",
        how="left",
    )
    url_rows = url_rows.assign(
        matched_domain_id=url_rows["exact_domain_id"].fillna(
            url_rows["fallback_domain_id"]
        )
    )

    valid_url_rows = url_rows[
        url_rows["category"].isin(URL_CATEGORIES)
        & url_rows["matched_domain_id"].notna()
    ]

    url_counts = (
        valid_url_rows.groupby(
            ["matched_domain_id", "category"]
        )
        .size()
        .rename("category_count")
        .reset_index()
    )
    url_totals = (
        url_counts.groupby("matched_domain_id")["category_count"]
        .sum()
        .rename("category_total")
        .reset_index()
    )
    url_counts = url_counts.merge(
        url_totals,
        on="matched_domain_id",
        how="left",
    )
    url_counts = url_counts.assign(
        category_fraction=(
            url_counts["category_count"]
            / url_counts["category_total"]
        )
    )

    url_wide = (
        url_counts.set_index(
            ["matched_domain_id", "category"]
        )["category_fraction"]
        .unstack(fill_value=np.float32(0.0))
        .reindex(
            columns=URL_CATEGORIES,
            fill_value=np.float32(0.0),
        )
        .set_axis(URL_CAT_COLS, axis=1)
        .reset_index()
        .rename(columns={"matched_domain_id": "domain_id"})
        .assign(has_url_category=np.float32(1.0))
    )

    lexical = lexical.merge(
        url_wide,
        on="domain_id",
        how="left",
    )
    lexical_url_values = lexical[
        URL_CAT_COLS + ["has_url_category"]
    ].fillna(0.0).astype(np.float32)
    lexical = lexical.drop(
        columns=URL_CAT_COLS + ["has_url_category"]
    ).skb.concat(
        [lexical_url_values],
        axis=1,
    )

    out_degree = (
        links.groupby("source_domain_id")
        .size()
        .rename("out_degree")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )
    in_degree = (
        links.groupby("target_domain_id")
        .size()
        .rename("in_degree")
        .reset_index()
        .rename(columns={"target_domain_id": "domain_id"})
    )

    degree_table = out_degree.merge(
        in_degree,
        on="domain_id",
        how="outer",
    ).fillna(0.0)
    degree_table = degree_table.assign(
        total_degree=(
            degree_table["out_degree"]
            + degree_table["in_degree"]
        )
    )

    out_links_with_degree = links.merge(
        degree_table[
            ["domain_id", "total_degree"]
        ].rename(columns={"domain_id": "target_domain_id"}),
        on="target_domain_id",
        how="left",
    )
    in_links_with_degree = links.merge(
        degree_table[
            ["domain_id", "total_degree"]
        ].rename(columns={"domain_id": "source_domain_id"}),
        on="source_domain_id",
        how="left",
    )

    graph_features = X.merge(
        degree_table,
        on="domain_id",
        how="left",
    ).fillna(0.0)

    tracker_edges = links.merge(
        trackers[["tracker_id", "tracking_domain_id"]],
        left_on="target_domain_id",
        right_on="tracking_domain_id",
        how="inner",
    )

    tracker_link_counts = (
        tracker_edges.groupby("source_domain_id")
        .size()
        .rename("tracker_link_count")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )

    graph_features = graph_features.merge(
        tracker_link_counts,
        on="domain_id",
        how="left",
    )
    graph_features = graph_features.assign(
        tracker_link_count=graph_features[
            "tracker_link_count"
        ].fillna(0.0)
    )

    graph_features = graph_features.assign(
        log_out_degree=graph_features[
            "out_degree"
        ].skb.apply_func(np.log1p),
        log_in_degree=graph_features[
            "in_degree"
        ].skb.apply_func(np.log1p),
        in_out_ratio=(
            graph_features["in_degree"] + 1.0
        ) / (
            graph_features["out_degree"] + 1.0
        ),
        is_isolated=(
            graph_features["total_degree"] == 0
        ).astype(np.float32),
        log_tracker_link_count=graph_features[
            "tracker_link_count"
        ].skb.apply_func(np.log1p),
        tracker_link_ratio=(
            graph_features["tracker_link_count"]
            / graph_features["out_degree"].clip(lower=1.0)
        ),
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
        .set_axis(DIRECT_COLS, axis=1)
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )

    base = lexical.merge(
        graph_features[
            [
                "domain_id",
                "out_degree",
                "log_out_degree",
                "in_degree",
                "log_in_degree",
                "total_degree",
                "in_out_ratio",
                "is_isolated",
                "tracker_link_count",
                "log_tracker_link_count",
                "tracker_link_ratio",
            ]
        ],
        on="domain_id",
        how="left",
    )
    base = base.merge(
        direct_wide,
        on="domain_id",
        how="left",
    )

    direct_values = base[DIRECT_COLS].fillna(0.0).astype(np.float32)
    base = base.drop(columns=DIRECT_COLS).skb.concat(
        [direct_values],
        axis=1,
    )

    # Each fold-trained aggregation remains a separate estimator node.
    out_neighbors = base[["domain_id"]].skb.apply(
        NeighborTrackerFeatures(direction="out"),
        y=y,
        fit_kwargs={"links": out_links_with_degree},
        fit_transform_kwargs={"links": out_links_with_degree},
    )
    in_neighbors = base[["domain_id"]].skb.apply(
        NeighborTrackerFeatures(direction="in"),
        y=y,
        fit_kwargs={"links": in_links_with_degree},
        fit_transform_kwargs={"links": in_links_with_degree},
    )

    base = base.skb.concat(
        [out_neighbors, in_neighbors],
        axis=1,
    )
    base = base.assign(
        weighted_total_neighbor_degree=(
            base["weighted_out_neighbor_degree"]
            + base["weighted_in_neighbor_degree"]
        )
    )
    base = base.assign(
        log_weighted_total_neighbor_degree=base[
            "weighted_total_neighbor_degree"
        ].skb.apply_func(np.log1p)
    )

    tld_priors = base[["tld"]].skb.apply(
        TLDTrackerPrior(alpha=10.0),
        y=y,
    )

    root_input = base[["root"]].skb.concat(
        [tld_priors],
        axis=1,
    )
    root_priors = root_input.skb.apply(
        RootTrackerPrior(beta=2.0),
        y=y,
    )

    assembled = base.skb.concat(
        [root_priors, tld_priors],
        axis=1,
    )

    # Reproduce np.hstack order and np.nan_to_num as recorded matrix assembly.
    features = (
        assembled[FEATURE_COLS]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )

    # Fit-only statistics are independent recorded nodes evaluated from each
    # outer fold's training target.
    global_tracker_priors = y.mean(axis=0)
    cooccurrence_counts = y.T.dot(y)
    cooccurrence_matrix = cooccurrence_counts.skb.apply_func(
        normalize_cooccurrence
    )

    model = TrackerRankNetEstimator(
        num_classes=355,
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
        max_inner_validation_rows=30000,
    )

    pred = features.skb.apply(
        model,
        y=y,
        fit_kwargs={
            "global_tracker_priors": global_tracker_priors,
            "cooccurrence_matrix": cooccurrence_matrix,
        },
    )

    # 4. Score. No cv= here — OriginalDomainHoldout on mark_as_X drives.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring="recall_at_10",
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
