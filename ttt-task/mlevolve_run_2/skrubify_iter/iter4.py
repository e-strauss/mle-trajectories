import math

import numpy as np
import pandas as pd
import skrub
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import BaseCrossValidator
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# The original task has 355 tracker labels. This must be concrete because the
# estimators and the recorded indicator reshapes are constructed before data is read.
NUM_TRACKERS = 355
TRACKER_IDS = list(range(NUM_TRACKERS))
TARGET_COLS = [f"target_{i}" for i in TRACKER_IDS]
DIRECT_COLS = [f"direct_{i}" for i in TRACKER_IDS]
OUT_NBR_COLS = [f"out_nbr_{i}" for i in TRACKER_IDS]
IN_NBR_COLS = [f"in_nbr_{i}" for i in TRACKER_IDS]
ROOT_PRIOR_COLS = [f"root_prior_{i}" for i in TRACKER_IDS]
TLD_PRIOR_COLS = [f"tld_prior_{i}" for i in TRACKER_IDS]

TOP_TLDS = [
    "com", "ru", "org", "net", "de", "uk", "jp", "fr", "it", "pl",
    "br", "cn", "in", "nl", "es", "cz", "eu", "ua", "ca", "au",
    "ch", "se", "ro", "gr", "at", "tv", "io", "me", "co", "info",
]
TLD_ONEHOT_COLS = [f"tld_{t}" for t in TOP_TLDS] + ["tld_other"]

URL_CATEGORIES = [
    "Arts",
    "Business",
    "Computers",
    "Games",
    "Health",
    "Home",
    "Kids_and_Teens",
    "News",
    "Recreation",
    "Reference",
    "Regional",
    "Science",
    "Shopping",
    "Society",
    "Sports",
]
URL_CAT_COLS = [f"url_cat_{c}" for c in URL_CATEGORIES] + ["has_url_category"]

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

KEYWORD_GROUPS = {
    "kw_ecommerce": ["shop", "store", "cart", "buy", "market", "mall", "deal", "pay"],
    "kw_media": ["news", "press", "media", "times", "post", "daily", "gazette", "journal"],
    "kw_video": ["video", "tv", "movie", "film", "tube", "stream"],
    "kw_adult": ["adult", "sex", "porn", "xxx"],
    "kw_tech": ["tech", "dev", "code", "cloud", "soft", "app", "web"],
    "kw_finance": ["bank", "finance", "invest", "loan", "crypto", "coin"],
}

LEXICAL_COLS = [
    "domain_len",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "digit_ratio",
    "has_www",
    "has_multi_subdomain",
    "press_score",
    "has_press_score",
    "is_authoritarian",
    *TLD_ONEHOT_COLS,
    "kw_ecommerce",
    "kw_media",
    "kw_video",
    "kw_adult",
    "kw_tech",
    "kw_finance",
    "kw_total",
]

GRAPH_SCALAR_COLS = [
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
    "out_neighbor_weight",
    "log_out_neighbor_weight",
    "in_neighbor_weight",
    "log_in_neighbor_weight",
    "total_neighbor_weight",
    "log_total_neighbor_weight",
]

PRIOR_SCALAR_COLS = [
    "has_root_match",
    "root_count",
    "log_root_count",
    "root_prior_entropy",
    "root_prior_max",
]

SCALAR_COLS = LEXICAL_COLS + URL_CAT_COLS + GRAPH_SCALAR_COLS + PRIOR_SCALAR_COLS
FEATURE_COLS = (
    SCALAR_COLS
    + DIRECT_COLS
    + OUT_NBR_COLS
    + IN_NBR_COLS
    + ROOT_PRIOR_COLS
    + TLD_PRIOR_COLS
)


class OriginalDomainHoldout(BaseCrossValidator):
    """Reproduce np.random.seed(42), shuffle, then take the first 20% as val."""

    def __init__(self, random_state=42, max_validation_rows=30000):
        self.random_state = random_state
        self.max_validation_rows = max_validation_rows

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_rows = len(X)
        permutation = np.random.RandomState(self.random_state).permutation(n_rows)
        n_val = min(self.max_validation_rows, int(n_rows * 0.2))
        test_idx = permutation[:n_val]
        train_idx = permutation[n_val:]
        yield train_idx, test_idx


class NeighborTrackerFeatures(TransformerMixin, BaseEstimator):
    """Learn tracker adoption from this fold's training-domain neighbors."""

    def __init__(self, num_trackers=NUM_TRACKERS):
        self.num_trackers = num_trackers

    def fit(self, X, y, links=None):
        self.train_domain_ids_ = np.asarray(X["domain_id"], dtype=np.int64)
        self.train_targets_ = np.asarray(y, dtype=np.float32)
        return self

    def _transform(self, X, links):
        domain_ids = np.asarray(X["domain_id"], dtype=np.int64)
        src = np.asarray(links["source_domain_id"], dtype=np.int64)
        dst = np.asarray(links["target_domain_id"], dtype=np.int64)

        max_id = int(max(
            src.max(initial=0),
            dst.max(initial=0),
            domain_ids.max(initial=0),
            self.train_domain_ids_.max(initial=0),
        )) + 1

        out_degrees = np.bincount(src, minlength=max_id)
        in_degrees = np.bincount(dst, minlength=max_id)
        total_degrees = out_degrees + in_degrees

        selected_index = np.full(max_id, -1, dtype=np.int64)
        selected_index[domain_ids] = np.arange(len(domain_ids), dtype=np.int64)

        train_index = np.full(max_id, -1, dtype=np.int64)
        train_index[self.train_domain_ids_] = np.arange(
            len(self.train_domain_ids_), dtype=np.int64
        )

        src_selected = selected_index[src]
        dst_selected = selected_index[dst]
        src_train = train_index[src]
        dst_train = train_index[dst]

        out_mask = (src_selected >= 0) & (dst_train >= 0) & (src != dst)
        out_u = src_selected[out_mask]
        out_v = dst_train[out_mask]
        out_w = (
            1.0
            / np.log1p(np.maximum(total_degrees[dst[out_mask]], 1))
        ).astype(np.float32)

        out_adjacency = csr_matrix(
            (out_w, (out_u, out_v)),
            shape=(len(domain_ids), len(self.train_domain_ids_)),
        )
        out_weight = np.asarray(out_adjacency.sum(axis=1)).ravel().astype(np.float32)
        out_counts = np.asarray(
            out_adjacency @ csr_matrix(self.train_targets_, dtype=np.float32)
        )
        out_norm = np.where(
            out_weight[:, None] > 0,
            out_counts / np.maximum(out_weight[:, None], 1e-7),
            0.0,
        ).astype(np.float32)

        in_mask = (src_train >= 0) & (dst_selected >= 0) & (src != dst)
        in_u = dst_selected[in_mask]
        in_v = src_train[in_mask]
        in_w = (
            1.0
            / np.log1p(np.maximum(total_degrees[src[in_mask]], 1))
        ).astype(np.float32)

        in_adjacency = csr_matrix(
            (in_w, (in_u, in_v)),
            shape=(len(domain_ids), len(self.train_domain_ids_)),
        )
        in_weight = np.asarray(in_adjacency.sum(axis=1)).ravel().astype(np.float32)
        in_counts = np.asarray(
            in_adjacency @ csr_matrix(self.train_targets_, dtype=np.float32)
        )
        in_norm = np.where(
            in_weight[:, None] > 0,
            in_counts / np.maximum(in_weight[:, None], 1e-7),
            0.0,
        ).astype(np.float32)

        values = np.column_stack([out_weight, in_weight, out_norm, in_norm])
        columns = [
            "out_neighbor_weight",
            "in_neighbor_weight",
            *OUT_NBR_COLS,
            *IN_NBR_COLS,
        ]
        return pd.DataFrame(values, columns=columns, index=X.index)

    def fit_transform(self, X, y, links=None):
        self.fit(X, y, links=links)
        return self._transform(X, links)

    def transform(self, X, links=None):
        return self._transform(X, links)


class TldPriorFeatures(TransformerMixin, BaseEstimator):
    """Bayesian TLD tracker priors fitted on each outer training fold."""

    def __init__(self, alpha=10.0, num_trackers=NUM_TRACKERS):
        self.alpha = alpha
        self.num_trackers = num_trackers

    def fit(self, X, y):
        y_array = np.asarray(y, dtype=np.float32)
        tlds = X["tld"].fillna("unknown").astype(str).to_numpy()
        self.global_prior_ = y_array.mean(axis=0).astype(np.float32)
        self.priors_ = {}

        for tld in pd.unique(tlds):
            mask = tlds == tld
            counts = y_array[mask].sum(axis=0)
            total = int(mask.sum())
            self.priors_[tld] = (
                (counts + self.alpha * self.global_prior_)
                / (total + self.alpha)
            ).astype(np.float32)
        return self

    def transform(self, X):
        tlds = X["tld"].fillna("unknown").astype(str).to_numpy()
        values = np.vstack([
            self.priors_.get(tld, self.global_prior_) for tld in tlds
        ]).astype(np.float32)
        return pd.DataFrame(values, columns=TLD_PRIOR_COLS, index=X.index)


class RootPriorFeatures(TransformerMixin, BaseEstimator):
    """Root priors: leave-one-out on fit rows and ordinary lookup on scored rows."""

    def __init__(self, beta=2.0, num_trackers=NUM_TRACKERS):
        self.beta = beta
        self.num_trackers = num_trackers

    def fit(self, X, y):
        roots = X["root"].fillna("unknown").astype(str).to_numpy()
        targets = np.asarray(y, dtype=np.float32)
        self.root_counts_ = {}
        self.root_tracker_counts_ = {}

        for root in pd.unique(roots):
            mask = roots == root
            self.root_counts_[root] = int(mask.sum())
            self.root_tracker_counts_[root] = targets[mask].sum(axis=0)
        return self

    @staticmethod
    def _frame(root_prior, has_match, counts, index):
        entropy = -np.sum(
            root_prior * np.log(root_prior + 1e-12), axis=1
        ).astype(np.float32)
        maximum = root_prior.max(axis=1).astype(np.float32)
        values = np.column_stack([
            has_match,
            counts,
            np.log1p(counts),
            entropy,
            maximum,
            root_prior,
        ]).astype(np.float32)
        return pd.DataFrame(
            values,
            columns=PRIOR_SCALAR_COLS + ROOT_PRIOR_COLS,
            index=index,
        )

    def fit_transform(self, X, y, tld_prior=None):
        self.fit(X, y)
        roots = X["root"].fillna("unknown").astype(str).to_numpy()
        targets = np.asarray(y, dtype=np.float32)
        tld_base = np.asarray(tld_prior, dtype=np.float32)

        root_prior = tld_base.copy()
        has_match = np.zeros(len(X), dtype=np.float32)
        counts = np.zeros(len(X), dtype=np.float32)

        for i, root in enumerate(roots):
            total = self.root_counts_.get(root, 0)
            if total > 1:
                total_loo = total - 1
                count_loo = self.root_tracker_counts_[root] - targets[i]
                root_prior[i] = (
                    count_loo + self.beta * tld_base[i]
                ) / (total_loo + self.beta)
                has_match[i] = 1.0
                counts[i] = float(total_loo)

        return self._frame(root_prior, has_match, counts, X.index)

    def transform(self, X, tld_prior=None):
        roots = X["root"].fillna("unknown").astype(str).to_numpy()
        tld_base = np.asarray(tld_prior, dtype=np.float32)

        root_prior = tld_base.copy()
        has_match = np.zeros(len(X), dtype=np.float32)
        counts = np.zeros(len(X), dtype=np.float32)

        for i, root in enumerate(roots):
            if root in self.root_counts_:
                total = self.root_counts_[root]
                root_prior[i] = (
                    self.root_tracker_counts_[root] + self.beta * tld_base[i]
                ) / (total + self.beta)
                has_match[i] = 1.0
                counts[i] = float(total)

        return self._frame(root_prior, has_match, counts, X.index)


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
            self.film_prior = nn.Linear(
                bottleneck_dim, num_prior_channels * 2
            )
            with torch.no_grad():
                nn.init.zeros_(self.film_prior.weight)
                nn.init.zeros_(self.film_prior.bias)

            weights = torch.tensor(
                [1.5, 1.0, 1.0, 1.5, 0.8], dtype=torch.float32
            ).unsqueeze(0).repeat(num_classes, 1)
            self.tracker_prior_weights = nn.Parameter(weights)
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
            if cooccurrence_matrix is None:
                nn.init.normal_(self.cooc_layer.weight, std=0.01)
            else:
                self.cooc_layer.weight.copy_(
                    torch.from_numpy(cooccurrence_matrix.T).float()
                )

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
        domain_latent = F.normalize(
            self.domain_latent_proj(h), p=2, dim=-1
        )
        tracker_latent = F.normalize(
            self.tracker_prototypes, p=2, dim=-1
        )
        proto_logits = (
            torch.matmul(domain_latent, tracker_latent.t())
            * (math.sqrt(self.latent_dim) / 2.0)
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

        probs = torch.sigmoid(base_logits)
        msg1 = self.cooc_layer(probs)
        msg2 = self.cooc_layer(msg1)
        rel_msg = msg1 + self.rel_refine(msg2)
        gate = torch.sigmoid(self.cooc_gate(domain_latent))
        return base_logits + gate * rel_msg


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
        probs = torch.sigmoid(logits)
        pos_probs = probs.clamp(self.eps, 1.0 - self.eps)
        loss_pos = (
            -targets
            * (1.0 - pos_probs).pow(self.gamma_pos)
            * torch.log(pos_probs)
        )

        neg_probs = (probs - self.clip_margin).clamp(
            self.eps, 1.0 - self.eps
        )
        neg_probs = torch.where(
            probs <= self.clip_margin,
            torch.zeros_like(neg_probs),
            neg_probs,
        )
        loss_neg = (
            -(1.0 - targets)
            * neg_probs.pow(self.gamma_neg)
            * torch.log(1.0 - neg_probs)
        )
        return (loss_pos + loss_neg).sum(dim=-1).mean()


class HybridTopKRankingLoss(nn.Module):
    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=2.0,
        clip_margin=0.05,
        margin=1.0,
        top_k_neg=10,
        rank_weight=0.15,
    ):
        super().__init__()
        self.asym_focal = AsymmetricFocalRecallLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip_margin=clip_margin,
        )
        self.margin = margin
        self.top_k_neg = top_k_neg
        self.rank_weight = rank_weight

    def forward(self, logits, targets):
        focal_loss = self.asym_focal(logits, targets)
        neg_logits = torch.where(
            targets == 0,
            logits,
            torch.full_like(logits, -1e9),
        )
        hard_neg_logits, _ = torch.topk(
            neg_logits, k=self.top_k_neg, dim=-1
        )
        margin_diff = self.margin - (
            logits.unsqueeze(-1) - hard_neg_logits.unsqueeze(1)
        )
        ranking_violation = F.relu(margin_diff)
        pos_mask = targets.unsqueeze(-1)
        num_pos = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sample_loss = (
            (ranking_violation * pos_mask).sum(dim=(1, 2))
            / (num_pos.squeeze(-1) * self.top_k_neg)
        )
        return focal_loss + self.rank_weight * sample_loss.mean()


def recall_at_10(y_true, y_pred):
    targets = np.asarray(y_true)
    logits = np.asarray(y_pred)
    topk = np.argpartition(logits, -10, axis=1)[:, -10:]
    rows = np.arange(len(logits))[:, None]
    hits = targets[rows, topk].sum(axis=1)
    true_counts = targets.sum(axis=1)
    recalls = np.where(true_counts > 0, hits / true_counts, 0.0)
    return float(np.mean(recalls))


class TrackerRankNetEstimator(RegressorMixin, BaseEstimator):
    """The original PyTorch loop, including per-epoch checkpoint selection."""

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
        weight_decay=1e-4,
        random_state=42,
        validation_fraction=0.2,
        max_validation_rows=30000,
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
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.max_validation_rows = max_validation_rows

    @staticmethod
    def _recall(logits, targets):
        return recall_at_10(targets, logits)

    def _predict_array(self, X):
        values = np.ascontiguousarray(X, dtype=np.float32)
        tensor = torch.from_numpy(values)
        predictions = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(tensor), 4096):
                batch = tensor[start:start + 4096].to(self.device_)
                predictions.append(self.model_(batch).cpu().numpy())
        return np.vstack(predictions)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        values = np.ascontiguousarray(X, dtype=np.float32)
        targets = np.ascontiguousarray(y, dtype=np.float32)

        # The eager script selected checkpoints on the same validation rows it
        # reported, which leaks. Keep its 12-epoch checkpoint selection but carve
        # the checkpoint-validation rows from this outer fold's training rows.
        permutation = np.random.RandomState(self.random_state).permutation(
            len(values)
        )
        n_val = min(
            self.max_validation_rows,
            int(len(values) * self.validation_fraction),
        )
        val_idx = permutation[:n_val]
        train_idx = permutation[n_val:]

        X_train = torch.from_numpy(values[train_idx])
        y_train = torch.from_numpy(targets[train_idx])
        X_val = values[val_idx]
        y_val = targets[val_idx]

        global_priors = targets[train_idx].mean(axis=0).astype(np.float32)
        clipped = np.clip(global_priors, 1e-5, 1.0 - 1e-5)
        initial_bias = np.log(clipped / (1.0 - clipped)).astype(np.float32)

        cooc_counts = targets[train_idx].T.dot(targets[train_idx])
        tracker_freqs = np.diag(cooc_counts).copy()
        cond_cooc = cooc_counts / (tracker_freqs[:, None] + 10.0)
        cond_cooc = np.where(
            ~np.eye(self.num_classes, dtype=bool), cond_cooc, 0.0
        )
        row_max = cond_cooc.max(axis=1, keepdims=True)
        cond_cooc_norm = np.where(
            row_max > 0,
            cond_cooc / np.maximum(row_max, 1.0),
            0.0,
        ).astype(np.float32)

        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_ = TrackerRankNet(
            in_dim=values.shape[1],
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            initial_bias=initial_bias,
            num_prior_channels=self.num_prior_channels,
            cooccurrence_matrix=cond_cooc_norm,
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
        for name, parameter in self.model_.named_parameters():
            if not parameter.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

        optimizer = AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": no_decay_params,
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

        n_train = len(X_train)
        num_batches = math.ceil(n_train / self.batch_size)
        best_recall = -1.0
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model_.state_dict().items()
        }

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            order = torch.randperm(n_train)

            for batch_number, start in enumerate(
                range(0, n_train, self.batch_size)
            ):
                if epoch == 1:
                    warmup_step = batch_number + 1
                    warmup_lr = self.min_lr + (
                        self.base_lr - self.min_lr
                    ) * (warmup_step / num_batches)
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr

                indices = order[start:start + self.batch_size]
                batch_x = X_train[indices].to(
                    self.device_, non_blocking=True
                )
                batch_y = y_train[indices].to(
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

            if epoch > 1:
                scheduler.step()

            val_logits = self._predict_array(X_val)
            val_recall = self._recall(val_logits, y_val)
            if val_recall > best_recall:
                best_recall = val_recall
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }

        self.model_.load_state_dict({
            key: value.to(self.device_)
            for key, value in best_state.items()
        })
        self.model_.eval()
        return self

    def predict(self, X):
        return self._predict_array(X)


RECALL_AT_10_SCORER = make_scorer(
    recall_at_10,
    greater_is_better=True,
    response_method="predict",
)


with skrub.config_context(eager_data_ops=False):
    # 1. Load Data — every documented input is a recorded read. The original
    # optional-file existence guards are environment checks and are dropped.
    trackers = (
        skrub.as_data_op("input/trackers.tsv")
        .skb.apply_func(pd.read_csv, sep="\t")
        .assign(
            tracker_id=lambda frame: frame["tracker_id"].astype(int),
            tracking_domain_id=lambda frame: frame[
                "tracking_domain_id"
            ].astype(np.int64),
        )
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
        .assign(
            domain_id=lambda frame: frame["domain_id"].astype(np.int64),
            tracker_id=lambda frame: frame["tracker_id"].astype(int),
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
    press_tsv = (
        skrub.as_data_op("input/freedom-of-the-press.csv")
        .skb.apply_func(pd.read_csv, sep="\t")
    )
    press_auto = (
        skrub.as_data_op("input/freedom-of-the-press.csv")
        .skb.apply_func(pd.read_csv, sep=None, engine="python")
    )
    press_data = (press_tsv.shape[1] < 3).skb.if_else(
        press_auto, press_tsv
    )
    url_data = (
        skrub.as_data_op("input/url-classification.csv")
        .skb.apply_func(
            pd.read_csv,
            usecols=["url", "category"],
        )
    )

    # 2. Prepare Data — test domains are excluded before marking because this
    # changes which rows are scored. The raw target is the 355-column binary
    # indicator table. The custom one-split CV reproduces the original shuffled,
    # capped 20% holdout.
    candidate_domains = (
        tracking_graph.loc[
            ~tracking_graph["domain_id"].isin(target_domains["domain_id"]),
            ["domain_id"],
        ]
        .drop_duplicates()
        .sort_values("domain_id")
        .reset_index(drop=True)
    )

    target_flags = (
        tracking_graph.loc[
            tracking_graph["domain_id"].isin(candidate_domains["domain_id"]),
            ["domain_id", "tracker_id"],
        ]
        .drop_duplicates()
        .assign(present=np.uint8(1))
        .set_index(["domain_id", "tracker_id"])["present"]
        .unstack(fill_value=np.uint8(0))
        .reindex(
            index=candidate_domains["domain_id"],
            columns=TRACKER_IDS,
            fill_value=np.uint8(0),
        )
        .set_axis(TARGET_COLS, axis=1)
        .reset_index(drop=True)
    )

    # Dict lookup semantics keep the last duplicate key; de-duplicate before
    # every left merge so no lookup can fan out and change the row count.
    domain_lookup = domains.drop_duplicates(
        subset="domain_id", keep="last"
    )
    base = candidate_domains.merge(
        domain_lookup, on="domain_id", how="left"
    )
    base = base.assign(domain=base["domain"].fillna(""))

    y = target_flags.skb.mark_as_y()
    X = base.skb.mark_as_X(
        cv=OriginalDomainHoldout(
            random_state=42,
            max_validation_rows=30000,
        ),
        split_kwargs={},
    )

    # 3. Recorded stateless lexical feature engineering.
    domain_lower = X["domain"].astype(str).str.lower().str.strip()
    missing_domain = domain_lower.eq("")
    num_parts = domain_lower.str.count(r"\.") + 1
    last = domain_lower.str.extract(r"([^.]*)$", expand=False)
    second_last = domain_lower.str.extract(
        r"([^.]*)\.[^.]*$", expand=False
    )
    third_last = domain_lower.str.extract(
        r"([^.]*)\.[^.]*\.[^.]*$", expand=False
    )
    last_two = second_last + "." + last
    is_two_level = (num_parts >= 3) & last_two.isin(TWO_LEVEL_TLDS)

    tld = (
        last.where(~is_two_level, last_two)
        .where(num_parts > 1, last)
        .where(~missing_domain, "unknown")
    )
    root = (
        (second_last + "." + last)
        .where(~is_two_level, third_last + "." + last_two)
        .where(num_parts > 1, last)
        .where(~missing_domain, "unknown")
    )

    domain_len = domain_lower.str.len().clip(lower=1).astype(np.float32)
    num_dots = domain_lower.str.count(r"\.").astype(np.float32)
    num_hyphens = domain_lower.str.count("-").astype(np.float32)
    num_digits = domain_lower.str.count(r"\d").astype(np.float32)

    press = press_data.iloc[:, [0, 2]].set_axis(
        ["tld", "press_score"], axis=1
    )
    press = press.assign(
        tld=press["tld"].astype(str).str.strip().str.lower().str.lstrip("."),
        press_score=press["press_score"].skb.apply_func(
            pd.to_numeric, errors="coerce"
        ),
    ).dropna(subset=["press_score"])
    press = press.drop_duplicates(subset="tld", keep="last")
    median_press = (press.shape[0] == 0).skb.if_else(
        50.0, press["press_score"].median()
    )

    lexical_base = X.assign(tld=tld, root=root).merge(
        press, on="tld", how="left"
    )
    press_score = lexical_base["press_score"].fillna(median_press)

    keyword_values = {}
    for output_name, words in KEYWORD_GROUPS.items():
        value = domain_lower.str.contains(words[0], regex=False).astype(
            np.float32
        )
        for word in words[1:]:
            value = value + domain_lower.str.contains(
                word, regex=False
            ).astype(np.float32)
        keyword_values[output_name] = value

    lexical = lexical_base.assign(
        domain_len=domain_len,
        num_dots=num_dots,
        num_hyphens=num_hyphens,
        num_digits=num_digits,
        digit_ratio=num_digits / domain_len,
        has_www=domain_lower.str.startswith("www.").astype(np.float32),
        has_multi_subdomain=(num_dots > 1).astype(np.float32),
        press_score=press_score.astype(np.float32),
        has_press_score=lexical_base["press_score"].notna().astype(np.float32),
        is_authoritarian=(press_score > 60.0).astype(np.float32),
        **{
            f"tld_{value}": tld.eq(value).astype(np.float32)
            for value in TOP_TLDS
        },
        tld_other=(~tld.isin(TOP_TLDS)).astype(np.float32),
        **keyword_values,
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

    # URL-category features, expressed as parsing, lookup, aggregation and pivot
    # operations rather than the original per-row Python loop and scatter writes.
    name_lookup = domain_lookup.drop_duplicates(
        subset="domain", keep="last"
    )[["domain", "domain_id"]].rename(columns={"domain": "host"})
    fallback_lookup = name_lookup.rename(
        columns={
            "host": "host_without_www",
            "domain_id": "fallback_domain_id",
        }
    )

    host = (
        url_data["url"]
        .astype(str)
        .str.replace(r"^.*?://", "", regex=True)
        .str.replace(r"/.*$", "", regex=True)
        .str.replace(r":.*$", "", regex=True)
        .str.strip()
        .str.lower()
    )
    parsed_urls = url_data.assign(
        host=host,
        host_without_www=host.str.replace(
            r"^www\.", "", regex=True
        ),
    )
    parsed_urls = parsed_urls.merge(
        name_lookup, on="host", how="left"
    ).merge(
        fallback_lookup, on="host_without_www", how="left"
    )
    parsed_urls = parsed_urls.assign(
        matched_domain_id=parsed_urls["domain_id"].fillna(
            parsed_urls["fallback_domain_id"]
        )
    )
    url_counts = (
        parsed_urls.loc[
            parsed_urls["matched_domain_id"].notna()
            & parsed_urls["category"].isin(URL_CATEGORIES),
            ["matched_domain_id", "category"],
        ]
        .groupby(["matched_domain_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=URL_CATEGORIES, fill_value=0)
    )
    url_totals = url_counts.sum(axis=1)
    url_proportions = url_counts.div(
        url_totals.where(url_totals > 0, 1.0), axis=0
    )
    url_features = (
        url_proportions.assign(
            has_url_category=(url_totals > 0).astype(np.float32)
        )
        .set_axis(URL_CAT_COLS, axis=1)
        .reindex(index=X["domain_id"], fill_value=0.0)
        .reset_index(drop=True)
    )

    # Recorded graph degrees and direct tracker-link indicator reshape.
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

    tracker_lookup = trackers.drop_duplicates(
        subset="tracking_domain_id", keep="last"
    )[["tracking_domain_id", "tracker_id"]]
    tracker_edges = links.merge(
        tracker_lookup.rename(
            columns={"tracking_domain_id": "target_domain_id"}
        ),
        on="target_domain_id",
        how="left",
    )
    tracker_link_counts = (
        tracker_edges.loc[tracker_edges["tracker_id"].notna()]
        .groupby("source_domain_id")
        .size()
        .rename("tracker_link_count")
        .reset_index()
        .rename(columns={"source_domain_id": "domain_id"})
    )

    graph_base = (
        X.merge(out_degree_table, on="domain_id", how="left")
        .merge(in_degree_table, on="domain_id", how="left")
        .merge(tracker_link_counts, on="domain_id", how="left")
        .assign(
            out_degree=lambda frame: frame["out_degree"].fillna(0.0),
            in_degree=lambda frame: frame["in_degree"].fillna(0.0),
            tracker_link_count=lambda frame: frame[
                "tracker_link_count"
            ].fillna(0.0),
        )
    )

    direct_flags = (
        tracker_edges.loc[
            tracker_edges["source_domain_id"].isin(X["domain_id"])
            & tracker_edges["tracker_id"].notna(),
            ["source_domain_id", "tracker_id"],
        ]
        .drop_duplicates()
        .assign(present=np.float32(1.0))
        .set_index(["source_domain_id", "tracker_id"])["present"]
        .unstack(fill_value=np.float32(0.0))
        .reindex(
            index=X["domain_id"],
            columns=TRACKER_IDS,
            fill_value=np.float32(0.0),
        )
        .set_axis(DIRECT_COLS, axis=1)
        .reset_index(drop=True)
    )

    # Stateful neighbor adoption is fitted separately on each fold.
    neighbor = X[["domain_id"]].skb.apply(
        NeighborTrackerFeatures(num_trackers=NUM_TRACKERS),
        y=y,
        fit_transform_kwargs={"links": links},
        transform_kwargs={"links": links},
    )

    total_degree = graph_base["out_degree"] + graph_base["in_degree"]
    total_neighbor_weight = (
        neighbor["out_neighbor_weight"]
        + neighbor["in_neighbor_weight"]
    )
    graph_scalar = graph_base.assign(
        log_out_degree=graph_base["out_degree"].skb.apply_func(np.log1p),
        log_in_degree=graph_base["in_degree"].skb.apply_func(np.log1p),
        total_degree=total_degree,
        in_out_ratio=(
            (graph_base["in_degree"] + 1.0)
            / (graph_base["out_degree"] + 1.0)
        ),
        is_isolated=(total_degree == 0).astype(np.float32),
        log_tracker_link_count=graph_base[
            "tracker_link_count"
        ].skb.apply_func(np.log1p),
        tracker_link_ratio=(
            graph_base["tracker_link_count"]
            / graph_base["out_degree"].clip(lower=1.0)
        ),
        out_neighbor_weight=neighbor["out_neighbor_weight"],
        log_out_neighbor_weight=neighbor[
            "out_neighbor_weight"
        ].skb.apply_func(np.log1p),
        in_neighbor_weight=neighbor["in_neighbor_weight"],
        log_in_neighbor_weight=neighbor[
            "in_neighbor_weight"
        ].skb.apply_func(np.log1p),
        total_neighbor_weight=total_neighbor_weight,
        log_total_neighbor_weight=total_neighbor_weight.skb.apply_func(
            np.log1p
        ),
    )

    # TLD and root aggregations are independent fitted operators. Root uses
    # fit_transform for the original leave-one-out training-row branch.
    prior_keys = lexical[["tld", "root"]]
    tld_prior = prior_keys[["tld"]].skb.apply(
        TldPriorFeatures(alpha=10.0, num_trackers=NUM_TRACKERS),
        y=y,
    )
    root_prior = prior_keys[["root"]].skb.apply(
        RootPriorFeatures(beta=2.0, num_trackers=NUM_TRACKERS),
        y=y,
        fit_transform_kwargs={"tld_prior": tld_prior},
        transform_kwargs={"tld_prior": tld_prior},
    )

    # Assemble the exact positional order consumed by TrackerRankNet. The final
    # five 355-wide blocks are direct, outgoing, incoming, root and TLD priors.
    scalar_features = lexical[LEXICAL_COLS].skb.concat(
        [
            url_features[URL_CAT_COLS],
            graph_scalar[GRAPH_SCALAR_COLS],
            root_prior[PRIOR_SCALAR_COLS],
        ],
        axis=1,
    )
    features = scalar_features.skb.concat(
        [
            direct_flags[DIRECT_COLS],
            neighbor[OUT_NBR_COLS],
            neighbor[IN_NBR_COLS],
            root_prior[ROOT_PRIOR_COLS],
            tld_prior[TLD_PRIOR_COLS],
        ],
        axis=1,
    )
    features = (
        features[FEATURE_COLS]
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(np.float32)
    )

    model = TrackerRankNetEstimator(
        num_classes=NUM_TRACKERS,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        num_prior_channels=5,
        num_epochs=12,
        batch_size=4096,
        base_lr=3e-4,
        min_lr=1e-5,
        weight_decay=1e-4,
        random_state=42,
        validation_fraction=0.2,
        max_validation_rows=30000,
    )
    pred = features.skb.apply(model, y=y)

    # 4. Score. Submission generation, checkpoints and intermediate files are
    # omitted because they do not contribute to the cross-validated score.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring=RECALL_AT_10_SCORER,
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
