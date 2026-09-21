"""skrubified conversion of mlevolve_run_2 pipeline 0029 (TrackTheTrackers).

The original builds its own supervised table out of the "pile of data" in
``input/``: one row per domain that appears in the tracking graph, a 355-column
binary tracker indicator as the target, and features drawn from the domain name,
a press-freedom table, a URL content classifier, the hyperlink graph and
Bayesian tracker priors over root domains / TLDs. It then trains a torch
ranking network and reports Recall@10 on a single random hold-out.

What moved where, and why:

* The hold-out becomes a ``BaseCrossValidator`` on ``mark_as_X`` (guide section
  3) that reproduces the original's split exactly: the candidate domains sorted
  ascending, shuffled with seed 42, the first ``min(30000, 20%)`` held out.
* Row-local work (domain-name lexicon, press-freedom join, URL categories) is
  recorded pandas, so it re-runs per fold.
* Everything that depends on the fold's TRAINING LABELS -- the global tracker
  priors, the Adamic-Adar neighbour-adoption blocks, the Bayesian root/TLD
  priors, the tracker co-occurrence matrix -- lives in the wrapper estimator
  (guide pitfall 19). The original was already leak-free here (it derived all of
  them from ``train_domains`` only); the wrapper is simply the place where skrub
  can re-derive them per fold.
* Checkpoint selection changed. The original picked the epoch with the best
  Recall@10 *on the rows it then reported as its score*, which is selection on
  the reported metric. The wrapper carves an inner hold-out of the same size
  (``min(30000, 20%)`` of the fold's training rows) out of the training rows and
  selects on that instead, so the scored rows are never touched. Expect a score
  slightly below the original's optimistic number.
* Test-set feature construction and submission writing are dropped: they do not
  contribute to the validation score.
"""
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

INPUT_DIR = "./input"
NUM_TRACKERS = 355
TRACKER_COLS = [f"tracker_{i}" for i in range(NUM_TRACKERS)]

TOP_TLDS = [
    "com", "ru", "org", "net", "de", "uk", "jp", "fr", "it", "pl",
    "br", "cn", "in", "nl", "es", "cz", "eu", "ua", "ca", "au",
    "ch", "se", "ro", "gr", "at", "tv", "io", "me", "co", "info",
]
TLD_TO_COL_IDX = {t: i for i, t in enumerate(TOP_TLDS)}

KEYWORDS = {
    "kw_ecommerce": ["shop", "store", "cart", "buy", "market", "mall", "deal", "pay"],
    "kw_media": ["news", "press", "media", "times", "post", "daily", "gazette", "journal"],
    "kw_video": ["video", "tv", "movie", "film", "tube", "stream"],
    "kw_adult": ["adult", "sex", "porn", "xxx"],
    "kw_tech": ["tech", "dev", "code", "cloud", "soft", "app", "web"],
    "kw_finance": ["bank", "finance", "invest", "loan", "crypto", "coin"],
}

TWO_LEVEL_TLDS = {
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk", "sch.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "ed.jp", "go.jp", "gr.jp", "lg.jp",
    "com.br", "net.br", "org.br", "gov.br", "edu.br",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.ru", "net.ru", "org.ru", "pp.ru",
    "co.in", "net.in", "org.in", "gen.in", "firm.in", "ind.in",
    "com.mx", "net.mx", "org.mx", "edu.mx", "gob.mx",
    "com.tr", "net.tr", "org.tr", "edu.tr", "gov.tr",
    "com.pl", "net.pl", "org.pl", "info.pl",
    "co.nz", "net.nz", "org.nz", "govt.nz", "ac.nz",
    "co.za", "net.za", "org.za", "web.za",
    "co.kr", "ne.kr", "or.kr", "re.kr",
    "com.ar", "net.ar", "org.ar",
    "com.tw", "org.tw", "idv.tw",
    "com.ua", "net.ua", "org.ua", "kiev.ua",
    "co.il", "org.il", "net.il",
    "com.sg", "org.sg", "net.sg",
    "com.hk", "org.hk", "net.hk",
}

# The lexical block in the original's column order: X_lexical = column_stack([
#   d_lens, num_dots, num_hyphens, num_digits, num_digits / d_lens, has_www,
#   has_multi_sub, press_scores, has_press, is_auth, tld_onehot, kw_*, kw_total])
LEXICAL_COLS = (
    ["dom_len", "num_dots", "num_hyphens", "num_digits", "digit_ratio",
     "has_www", "has_multi_sub", "press_score", "has_press", "is_auth"]
    + [f"tld_{t}" for t in TOP_TLDS]
    + ["tld_other"]
    + list(KEYWORDS)
    + ["kw_total"]
)

URL_CATEGORIES = [
    "Arts", "Business", "Computers", "Games", "Health", "Home", "Kids_and_Teens",
    "News", "Recreation", "Reference", "Regional", "Science", "Shopping",
    "Society", "Sports",
]
URL_CAT_COLS = [f"url_cat_{c}" for c in URL_CATEGORIES] + ["url_cat_has"]

META_COLS = ["domain_id", "root", "tld"]


# =========================================================================
# Cross-validation: the original's single random hold-out, as a splitter
# =========================================================================
class CandidateDomainHoldout(BaseCrossValidator):
    """One hold-out fold reproducing the original's domain split.

    The original did ``np.random.seed(42); np.random.shuffle(candidates)`` on the
    ascending-sorted output of ``np.setdiff1d`` and kept the first
    ``min(30000, int(0.2 * n))`` shuffled domains as validation. ``X`` is marked
    in that same ascending domain order, so shuffling the row indices with
    ``RandomState(42)`` (identical to the legacy global RNG) yields exactly the
    original's train/validation membership *and* its training row order.
    """

    def __init__(self, max_val=30000, val_fraction=0.2, random_state=42):
        self.max_val = max_val
        self.val_fraction = val_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.RandomState(self.random_state).shuffle(indices)
        n_val = min(self.max_val, int(n_samples * self.val_fraction))
        yield indices[n_val:], indices[:n_val]


def recall_at_k(y_true, y_score, k=10):
    """Mean per-domain Recall@k, the task's metric (original: compute_recall_at_k)."""
    targets = np.asarray(y_true)
    scores = np.asarray(y_score)
    topk_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    row_indices = np.arange(scores.shape[0])[:, None]
    hits = targets[row_indices, topk_indices].sum(axis=1)
    true_counts = targets.sum(axis=1)
    recalls = np.where(true_counts > 0, hits / np.maximum(true_counts, 1), 0.0)
    return float(np.mean(recalls))


RECALL_AT_10 = make_scorer(recall_at_k, greater_is_better=True, response_method="predict")


# =========================================================================
# Row-local domain-name features (stateless transformers)
# =========================================================================
def extract_root_and_tld(domain_name):
    """Verbatim from the original."""
    if not domain_name or not isinstance(domain_name, str):
        return "unknown", "unknown"
    parts = domain_name.lower().strip().split(".")
    if len(parts) == 1:
        return parts[0], parts[0]
    tld = parts[-1]
    if len(parts) >= 3:
        last_two = f"{parts[-2]}.{parts[-1]}"
        if last_two in TWO_LEVEL_TLDS:
            root = f"{parts[-3]}.{last_two}"
            return root, last_two
    root = f"{parts[-2]}.{parts[-1]}"
    return root, tld


class DomainParts(TransformerMixin, BaseEstimator):
    """Split each hostname into its registrable root and its TLD.

    A Python loop is the faithful transcription of the original's
    ``extract_root_and_tld`` (two-level-TLD table, ``"unknown"`` fallback); it
    lives inside ``transform`` where a loop is legitimate (guide pitfall 20)
    rather than being approximated by ``.str`` ops.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        names = X["domain"].fillna("").astype(str).to_numpy()
        roots = np.empty(len(names), dtype=object)
        tlds = np.empty(len(names), dtype=object)
        for i, name in enumerate(names):
            roots[i], tlds[i] = extract_root_and_tld(name)
        return X.assign(root=roots, tld=tlds)


class DomainLexicalFeatures(TransformerMixin, BaseEstimator):
    """The original's ``X_lexical`` block, in its original column order.

    ``press_score`` / ``has_press`` / ``is_auth`` arrive as columns (joined from
    ``freedom-of-the-press.csv`` by a recorded merge) so the block can be emitted
    in one piece at positions 7-9, exactly where the original's ``column_stack``
    put them -- column order feeds the network's weight initialisation.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        names = X["domain"].fillna("").astype(str).to_numpy()
        lowered = np.array([n.lower() for n in names], dtype=object)

        dom_len = np.array([max(len(n), 1) for n in names], dtype=np.float32)
        num_dots = np.array([n.count(".") for n in names], dtype=np.float32)
        num_hyphens = np.array([n.count("-") for n in names], dtype=np.float32)
        num_digits = np.array(
            [sum(c.isdigit() for c in n) for n in names], dtype=np.float32
        )
        has_www = np.array(
            [1.0 if n.startswith("www.") else 0.0 for n in lowered], dtype=np.float32
        )
        has_multi_sub = np.where(num_dots > 1, 1.0, 0.0).astype(np.float32)

        out = {
            "dom_len": dom_len,
            "num_dots": num_dots,
            "num_hyphens": num_hyphens,
            "num_digits": num_digits,
            "digit_ratio": (num_digits / dom_len).astype(np.float32),
            "has_www": has_www,
            "has_multi_sub": has_multi_sub,
            "press_score": X["press_score"].to_numpy(np.float32),
            "has_press": X["has_press"].to_numpy(np.float32),
            "is_auth": X["is_auth"].to_numpy(np.float32),
        }

        tlds = X["tld"].astype(str).to_numpy()
        onehot = np.zeros((len(names), len(TOP_TLDS) + 1), dtype=np.float32)
        col_idx = np.array(
            [TLD_TO_COL_IDX.get(t, len(TOP_TLDS)) for t in tlds], dtype=np.int64
        )
        onehot[np.arange(len(names)), col_idx] = 1.0
        for j, tld in enumerate(TOP_TLDS):
            out[f"tld_{tld}"] = onehot[:, j]
        out["tld_other"] = onehot[:, len(TOP_TLDS)]

        kw_total = np.zeros(len(names), dtype=np.float32)
        for name, words in KEYWORDS.items():
            counts = np.array(
                [sum(1.0 for kw in words if kw in n) for n in lowered],
                dtype=np.float32,
            )
            out[name] = counts
            kw_total = kw_total + counts
        out["kw_total"] = kw_total

        features = pd.DataFrame(out, index=X.index)[LEXICAL_COLS]
        return pd.concat([X[META_COLS], features], axis=1)


@skrub.deferred
def build_tracker_matrix(edges, domain_ids):
    """Scatter the (domain, tracker) edge list into a dense 0/1 indicator frame.

    This is a matrix materialisation, not a dataframe transformation: the
    recorded equivalent (``pivot_table``/``crosstab``) would build an int64
    frame of ``n_domains x 355``, several times the ``uint8`` footprint the
    original relied on at 18.7M rows. Kept as one node for that reason (guide
    section 4 / pitfall 13).
    """
    positions = pd.Series(
        np.arange(len(domain_ids), dtype=np.int64), index=domain_ids.to_numpy()
    )
    rows = positions.reindex(edges["domain_id"].to_numpy()).to_numpy()
    matrix = np.zeros((len(domain_ids), NUM_TRACKERS), dtype=np.uint8)
    matrix[rows, edges["tracker_id"].to_numpy()] = 1
    return pd.DataFrame(matrix, columns=TRACKER_COLS)


# =========================================================================
# The network, verbatim from the original
# =========================================================================
class ResidualTabularBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout_rate: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        return self.act2(out + res)


class TrackerRankNet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 355,
        hidden_dim: int = 512,
        latent_dim: int = 128,
        dropout_rate: float = 0.2,
        initial_bias: np.ndarray = None,
        num_prior_channels: int = 5,
        cooccurrence_matrix: np.ndarray = None,
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
            hidden_dim, hidden_dim, dropout_rate=dropout_rate
        )
        self.res_block2 = ResidualTabularBlock(
            hidden_dim, hidden_dim // 2, dropout_rate=dropout_rate
        )

        bottleneck_dim = hidden_dim // 2
        self.direct_head = nn.Linear(bottleneck_dim, num_classes)
        self.domain_latent_proj = nn.Linear(bottleneck_dim, latent_dim)
        self.tracker_prototypes = nn.Parameter(
            torch.randn(num_classes, latent_dim) / math.sqrt(latent_dim)
        )
        self.head_blend = nn.Parameter(torch.tensor([0.5]))

        if self.has_tracker_priors:
            # Context-conditioned FiLM prior modulation from domain representation h
            self.film_prior = nn.Linear(bottleneck_dim, num_prior_channels * 2)
            with torch.no_grad():
                nn.init.zeros_(self.film_prior.weight)
                nn.init.zeros_(self.film_prior.bias)

            init_weights = torch.tensor(
                [1.5, 1.0, 1.0, 1.5, 0.8], dtype=torch.float32
            ).unsqueeze(0).repeat(num_classes, 1)
            self.tracker_prior_weights = nn.Parameter(init_weights)
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

        # 2-Hop Non-Linear Residual Relational Message Passing Module
        self.cooc_layer = nn.Linear(num_classes, num_classes, bias=False)
        if cooccurrence_matrix is not None:
            with torch.no_grad():
                self.cooc_layer.weight.copy_(
                    torch.from_numpy(cooccurrence_matrix.T).float()
                )
        else:
            with torch.no_grad():
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                :, -self.num_prior_channels * self.num_classes :
            ].reshape(-1, self.num_prior_channels, self.num_classes)
            prior_perm = prior_signals.permute(0, 2, 1)

            # Context-conditioned FiLM modulation: gamma(h) and beta(h) modulate prior channels
            film_params = self.film_prior(h)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma = 1.0 + torch.tanh(gamma).unsqueeze(1)
            beta = beta.unsqueeze(1)
            prior_perm_modulated = prior_perm * gamma + beta

            direct_prior_logits = (
                prior_perm_modulated * self.tracker_prior_weights
            ).sum(dim=-1) + self.tracker_prior_bias
            mlp_prior_logits = self.tracker_res_net(prior_perm_modulated).squeeze(-1)
            tracker_res_logits = direct_prior_logits + mlp_prior_logits
            base_logits = base_logits + self.res_scale * tracker_res_logits

        # 2-hop non-linear residual relational message passing without destructive LayerNorm
        probs = torch.sigmoid(base_logits)
        msg1 = self.cooc_layer(probs)
        msg2 = self.cooc_layer(msg1)
        rel_msg = msg1 + self.rel_refine(msg2)
        cooc_gate = torch.sigmoid(self.cooc_gate(domain_latent))
        relational_refinement = cooc_gate * rel_msg
        final_logits = base_logits + relational_refinement
        return final_logits


class AsymmetricFocalRecallLoss(nn.Module):

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip_margin: float = 0.05,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_probs = probs.clamp(min=self.eps, max=1.0 - self.eps)
        loss_pos = (
            -targets * (1.0 - pos_probs).pow(self.gamma_pos) * torch.log(pos_probs)
        )

        neg_probs = (probs - self.clip_margin).clamp(min=self.eps, max=1.0 - self.eps)
        neg_probs = torch.where(
            probs <= self.clip_margin, torch.zeros_like(neg_probs), neg_probs
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
        gamma_pos: float = 0.0,
        gamma_neg: float = 2.0,
        clip_margin: float = 0.05,
        margin: float = 1.0,
        top_k_neg: int = 10,
        rank_weight: float = 0.15,
        eps: float = 1e-7,
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

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.asym_focal(logits, targets)

        # Hard-negative mining top-K ranking margin loss
        neg_logits = torch.where(targets == 0, logits, torch.full_like(logits, -1e9))
        hard_neg_logits, _ = torch.topk(neg_logits, k=self.top_k_neg, dim=-1)

        margin_diff = self.margin - (
            logits.unsqueeze(-1) - hard_neg_logits.unsqueeze(1)
        )
        ranking_violation = F.relu(margin_diff)

        pos_mask = targets.unsqueeze(-1)
        num_pos = targets.sum(dim=-1, keepdim=True).clamp(min=1.0)

        sample_rank_loss = (ranking_violation * pos_mask).sum(dim=(1, 2)) / (
            num_pos.squeeze(-1) * self.top_k_neg
        )
        rank_loss = sample_rank_loss.mean()

        return focal_loss + self.rank_weight * rank_loss


# =========================================================================
# The wrapper: every label-dependent statistic plus the training loop
# =========================================================================
class TrackerRankNetEstimator(RegressorMixin, BaseEstimator):
    """Label-dependent feature construction + the original's torch training.

    ``fit`` receives one fold's training rows; the link graph and the tracker
    table come in through ``fit_kwargs`` as DataOps (guide section 7) and are
    kept on the estimator, because they are label-free reference tables that
    ``predict`` needs too. The fold's training domains, their targets, and the
    root/TLD statistics derived from them are what make the neighbour and
    Bayesian-prior blocks fold-specific: at ``predict`` time the scored rows get
    those blocks rebuilt from the TRAINING labels only, which is what the
    original achieved by hand with its ``train_domains`` / ``val_domains`` split.
    """

    def __init__(
        self,
        num_trackers=NUM_TRACKERS,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        num_epochs=12,
        batch_size=4096,
        base_lr=3e-4,
        min_lr=1e-5,
        weight_decay=1e-4,
        alpha_tld=10.0,
        beta_root=2.0,
        epsilon_cooc=10.0,
        max_checkpoint_rows=30000,
        checkpoint_fraction=0.2,
        random_state=42,
    ):
        self.num_trackers = num_trackers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.alpha_tld = alpha_tld
        self.beta_root = beta_root
        self.epsilon_cooc = epsilon_cooc
        self.max_checkpoint_rows = max_checkpoint_rows
        self.checkpoint_fraction = checkpoint_fraction
        self.random_state = random_state

    # --- small lookup helpers (replacing the original's dense id-indexed arrays,
    #     which allocate one slot per domain id up to max(domain_id)) ----------
    @staticmethod
    def _lookup(values, keys_sorted, vals_sorted, missing=-1):
        if len(keys_sorted) == 0:
            return np.full(len(values), missing, dtype=np.int64)
        pos = np.clip(np.searchsorted(keys_sorted, values), 0, len(keys_sorted) - 1)
        return np.where(keys_sorted[pos] == values, vals_sorted[pos], missing)

    @staticmethod
    def _position_map(ids):
        order = np.argsort(ids, kind="stable")
        return ids[order], order.astype(np.int64)

    def _prepare_graph(self, link_graph, trackers):
        self.src_ = link_graph["source_domain_id"].to_numpy(np.int64)
        self.dst_ = link_graph["target_domain_id"].to_numpy(np.int64)

        self.nodes_ = np.unique(np.concatenate([self.src_, self.dst_]))
        self.node_positions_ = np.arange(len(self.nodes_), dtype=np.int64)
        src_node = np.searchsorted(self.nodes_, self.src_)
        dst_node = np.searchsorted(self.nodes_, self.dst_)
        self.dst_node_ = dst_node
        self.src_node_ = src_node

        self.out_degrees_ = np.bincount(src_node, minlength=len(self.nodes_))
        self.in_degrees_ = np.bincount(dst_node, minlength=len(self.nodes_))
        self.tot_degrees_ = self.out_degrees_ + self.in_degrees_

        tracker_tdid = trackers["tracking_domain_id"].to_numpy(np.int64)
        tracker_id = trackers["tracker_id"].to_numpy(np.int64)
        order = np.argsort(tracker_tdid)
        self.dst_tracker_id_ = self._lookup(
            self.dst_, tracker_tdid[order], tracker_id[order]
        )
        is_tracker_edge = self.dst_tracker_id_ >= 0
        self.tracker_out_counts_ = np.bincount(
            src_node[is_tracker_edge], minlength=len(self.nodes_)
        )

    def _graph_blocks(self, domain_ids):
        """direct links, neighbour tracker adoption and the 16 graph scalars."""
        n_sel = len(domain_ids)
        sel_keys, sel_vals = self._position_map(domain_ids)
        src_sel_idx = self._lookup(self.src_, sel_keys, sel_vals)
        dst_sel_idx = self._lookup(self.dst_, sel_keys, sel_vals)
        src_train_idx = self._lookup(self.src_, self.train_keys_, self.train_vals_)
        dst_train_idx = self._lookup(self.dst_, self.train_keys_, self.train_vals_)
        n_train = self.train_targets_.shape[0]

        # Direct tracker link indicators (all 355 trackers)
        direct_mask = (src_sel_idx >= 0) & (self.dst_tracker_id_ >= 0)
        direct_links = np.zeros((n_sel, self.num_trackers), dtype=np.float32)
        direct_links[src_sel_idx[direct_mask], self.dst_tracker_id_[direct_mask]] = 1.0

        # Adamic-Adar weighted neighbour tracker adoption, outgoing then incoming
        out_mask = (src_sel_idx >= 0) & (dst_train_idx >= 0) & (self.src_ != self.dst_)
        w_out = (
            1.0 / np.log1p(np.maximum(self.tot_degrees_[self.dst_node_[out_mask]], 1))
        ).astype(np.float32)
        A_out = csr_matrix(
            (w_out, (src_sel_idx[out_mask], dst_train_idx[out_mask])),
            shape=(n_sel, n_train),
        )
        nbr_out_deg = np.asarray(A_out.sum(axis=1)).ravel()
        out_counts = (A_out @ self.train_targets_csr_).toarray()
        out_nbr_norm = np.where(
            nbr_out_deg[:, None] > 0,
            out_counts / np.maximum(nbr_out_deg[:, None], 1e-7),
            0.0,
        ).astype(np.float32)

        in_mask = (src_train_idx >= 0) & (dst_sel_idx >= 0) & (self.src_ != self.dst_)
        w_in = (
            1.0 / np.log1p(np.maximum(self.tot_degrees_[self.src_node_[in_mask]], 1))
        ).astype(np.float32)
        A_in = csr_matrix(
            (w_in, (dst_sel_idx[in_mask], src_train_idx[in_mask])),
            shape=(n_sel, n_train),
        )
        nbr_in_deg = np.asarray(A_in.sum(axis=1)).ravel()
        in_counts = (A_in @ self.train_targets_csr_).toarray()
        in_nbr_norm = np.where(
            nbr_in_deg[:, None] > 0,
            in_counts / np.maximum(nbr_in_deg[:, None], 1e-7),
            0.0,
        ).astype(np.float32)

        nbr_tot_deg = nbr_out_deg + nbr_in_deg

        sel_node = self._lookup(domain_ids, self.nodes_, self.node_positions_)
        safe = np.maximum(sel_node, 0)
        present = sel_node >= 0
        sel_out_deg = np.where(present, self.out_degrees_[safe], 0).astype(np.float32)
        sel_in_deg = np.where(present, self.in_degrees_[safe], 0).astype(np.float32)
        sel_tr_links = np.where(present, self.tracker_out_counts_[safe], 0).astype(
            np.float32
        )

        X_graph = np.column_stack([
            sel_out_deg,
            np.log1p(sel_out_deg),
            sel_in_deg,
            np.log1p(sel_in_deg),
            sel_out_deg + sel_in_deg,
            (sel_in_deg + 1.0) / (sel_out_deg + 1.0),
            (sel_out_deg + sel_in_deg == 0).astype(np.float32),
            sel_tr_links,
            np.log1p(sel_tr_links),
            sel_tr_links / np.maximum(sel_out_deg, 1.0),
            nbr_out_deg.astype(np.float32),
            np.log1p(nbr_out_deg).astype(np.float32),
            nbr_in_deg.astype(np.float32),
            np.log1p(nbr_in_deg).astype(np.float32),
            nbr_tot_deg.astype(np.float32),
            np.log1p(nbr_tot_deg).astype(np.float32),
        ]).astype(np.float32)

        return X_graph, direct_links, out_nbr_norm, in_nbr_norm

    def _fit_priors(self, roots, tlds):
        """Bayesian TLD / root-domain tracker priors from the training labels."""
        self.global_priors_ = self.train_targets_.mean(axis=0, dtype=np.float64).astype(
            np.float32
        )

        tld_keys, tld_inv = np.unique(tlds, return_inverse=True)
        tld_counts = np.bincount(tld_inv, minlength=len(tld_keys))
        grouper = csr_matrix(
            (
                np.ones(len(tld_inv), dtype=np.float32),
                (tld_inv, np.arange(len(tld_inv))),
            ),
            shape=(len(tld_keys), len(tld_inv)),
        )
        tld_tracker_counts = (grouper @ self.train_targets_csr_).toarray()
        self.tld_keys_ = pd.Index(tld_keys)
        # One extra row holding the global prior, used for unseen TLDs.
        self.tld_table_ = np.vstack([
            (tld_tracker_counts + self.alpha_tld * self.global_priors_)
            / (tld_counts[:, None] + self.alpha_tld),
            self.global_priors_[None, :],
        ]).astype(np.float32)

        root_keys, root_inv = np.unique(roots, return_inverse=True)
        self.root_counts_ = np.bincount(root_inv, minlength=len(root_keys))
        grouper = csr_matrix(
            (
                np.ones(len(root_inv), dtype=np.float32),
                (root_inv, np.arange(len(root_inv))),
            ),
            shape=(len(root_keys), len(root_inv)),
        )
        self.root_tracker_counts_ = (grouper @ self.train_targets_csr_).toarray()
        self.root_keys_ = pd.Index(root_keys)

    def _prior_blocks(self, roots, tlds, is_training_rows):
        """root/TLD prior matrices plus the 5 scalar prior features.

        Training rows use the original's leave-one-out root prior (the row's own
        labels removed from its root's counts); unseen rows use the plain root
        posterior. Rows whose root was never seen in training fall back to the
        TLD prior.
        """
        tld_idx = self.tld_keys_.get_indexer(pd.Index(tlds))
        tld_idx = np.where(tld_idx >= 0, tld_idx, len(self.tld_keys_))
        tld_prior = self.tld_table_[tld_idx]
        root_prior = tld_prior.copy()

        n_rows = len(roots)
        has_root_match = np.zeros(n_rows, dtype=np.float32)
        root_cnt = np.zeros(n_rows, dtype=np.float32)

        root_idx = self.root_keys_.get_indexer(pd.Index(roots))
        hit = root_idx >= 0
        rows = np.flatnonzero(hit)
        keys = root_idx[rows]
        totals = self.root_counts_[keys].astype(np.float32)

        if is_training_rows:
            keep = totals > 1
            rows, keys, totals = rows[keep], keys[keep], totals[keep] - 1.0
            counts = self.root_tracker_counts_[keys] - self.train_targets_[rows]
        else:
            counts = self.root_tracker_counts_[keys]

        root_prior[rows] = (
            (counts + self.beta_root * tld_prior[rows])
            / (totals[:, None] + self.beta_root)
        ).astype(np.float32)
        has_root_match[rows] = 1.0
        root_cnt[rows] = totals

        X_prior = np.column_stack([
            has_root_match,
            root_cnt,
            np.log1p(root_cnt),
            -np.sum(root_prior * np.log(root_prior + 1e-12), axis=1).astype(np.float32),
            root_prior.max(axis=1).astype(np.float32),
        ]).astype(np.float32)

        return X_prior, root_prior, tld_prior

    def _design_matrix(self, X, is_training_rows):
        domain_ids = X["domain_id"].to_numpy(np.int64)
        roots = X["root"].astype(str).to_numpy()
        tlds = X["tld"].astype(str).to_numpy()
        X_lexical_cat = X.drop(columns=META_COLS).to_numpy(np.float32)

        X_graph, direct, out_nbr, in_nbr = self._graph_blocks(domain_ids)
        X_prior, root_prior, tld_prior = self._prior_blocks(
            roots, tlds, is_training_rows
        )

        X_scalar = np.hstack([X_lexical_cat, X_graph, X_prior]).astype(np.float32)
        np.nan_to_num(X_scalar, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # The last 5 x 355 columns are the prior channels the network slices off.
        X_all = np.hstack(
            [X_scalar, direct, out_nbr, in_nbr, root_prior, tld_prior]
        ).astype(np.float32)
        np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X_all

    def _cooccurrence(self):
        Y = self.train_targets_
        cooc_counts = Y.T.dot(Y)
        tracker_freqs = np.diag(cooc_counts).copy()
        cond_cooc = cooc_counts / (tracker_freqs[:, None] + self.epsilon_cooc)
        np.fill_diagonal(cond_cooc, 0.0)
        row_max = cond_cooc.max(axis=1, keepdims=True)
        return np.where(
            row_max > 0, cond_cooc / np.maximum(row_max, 1.0), 0.0
        ).astype(np.float32)

    def _predict_logits(self, X_matrix):
        self.model_.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(X_matrix), 4096):
                batch = torch.from_numpy(X_matrix[start : start + 4096]).to(self.device_)
                chunks.append(self.model_(batch).cpu().numpy())
        return np.vstack(chunks)

    def fit(self, X, y, link_graph=None, trackers=None):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self.train_targets_ = np.asarray(y, dtype=np.float32)
        self.train_targets_csr_ = csr_matrix(self.train_targets_)
        self.train_keys_, self.train_vals_ = self._position_map(
            X["domain_id"].to_numpy(np.int64)
        )

        self._prepare_graph(link_graph, trackers)
        self._fit_priors(X["root"].astype(str).to_numpy(), X["tld"].astype(str).to_numpy())
        X_matrix = self._design_matrix(X, is_training_rows=True)
        cooccurrence = self._cooccurrence()

        # Inner hold-out for checkpoint selection, carved out of THIS fold's
        # training rows only (the original selected on the rows it reported).
        n_rows = len(X_matrix)
        n_hold = min(self.max_checkpoint_rows, int(n_rows * self.checkpoint_fraction))
        shuffled = np.random.RandomState(self.random_state).permutation(n_rows)
        hold_idx, fit_idx = shuffled[:n_hold], shuffled[n_hold:]

        prior_eps = 1e-5
        clipped_priors = np.clip(self.global_priors_, prior_eps, 1.0 - prior_eps)
        init_biases = np.log(clipped_priors / (1.0 - clipped_priors)).astype(np.float32)

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = TrackerRankNet(
            in_dim=X_matrix.shape[1],
            num_classes=self.num_trackers,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout_rate=self.dropout_rate,
            initial_bias=init_biases,
            num_prior_channels=5,
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

        decay_params, no_decay_params = [], []
        for name, param in self.model_.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.num_epochs - 1, eta_min=self.min_lr
        )

        X_fit_t = torch.from_numpy(np.ascontiguousarray(X_matrix[fit_idx]))
        Y_fit_t = torch.from_numpy(np.ascontiguousarray(self.train_targets_[fit_idx]))
        X_hold = np.ascontiguousarray(X_matrix[hold_idx])
        Y_hold = self.train_targets_[hold_idx]
        del X_matrix

        n_fit = X_fit_t.shape[0]
        num_batches_per_epoch = math.ceil(n_fit / self.batch_size)
        best_recall = -1.0
        best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}

        for epoch in range(1, self.num_epochs + 1):
            self.model_.train()
            total_loss, num_batches = 0.0, 0
            perm = torch.randperm(n_fit)

            for start in range(0, n_fit, self.batch_size):
                if epoch == 1:
                    warmup_lr = self.min_lr + (self.base_lr - self.min_lr) * (
                        (num_batches + 1) / num_batches_per_epoch
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = warmup_lr

                batch_indices = perm[start : start + self.batch_size]
                batch_x = X_fit_t[batch_indices].to(self.device_, non_blocking=True)
                batch_y = Y_fit_t[batch_indices].to(self.device_, non_blocking=True)

                optimizer.zero_grad()
                loss = criterion(self.model_(batch_x), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if epoch > 1:
                scheduler.step()

            hold_recall = recall_at_k(Y_hold, self._predict_logits(X_hold), k=10)
            print(
                f"Epoch {epoch:02d}/{self.num_epochs:02d} | "
                f"Train Loss: {total_loss / max(num_batches, 1):.4f} | "
                f"Inner Recall@10: {hold_recall:.5f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )
            if hold_recall > best_recall:
                best_recall = hold_recall
                best_state = {
                    k: v.cpu().clone() for k, v in self.model_.state_dict().items()
                }

        self.model_.load_state_dict(
            {k: v.to(self.device_) for k, v in best_state.items()}
        )
        self.n_features_in_ = X_fit_t.shape[1]
        return self

    def predict(self, X):
        return self._predict_logits(self._design_matrix(X, is_training_rows=False))


with skrub.config_context(eager_data_ops=False):
    # =====================================================================
    # 1. Recorded loads. Every table the original read, minus the parts that
    #    only serve submission writing.
    # =====================================================================
    trackers = skrub.as_data_op(f"{INPUT_DIR}/trackers.tsv").skb.apply_func(
        pd.read_csv, sep="\t"
    )
    target_domains = skrub.as_data_op(f"{INPUT_DIR}/target.tsv").skb.apply_func(
        pd.read_csv, sep="\t"
    )
    graph_train = skrub.as_data_op(
        f"{INPUT_DIR}/tracking_graph_train.parquet"
    ).skb.apply_func(pd.read_parquet, columns=["domain_id", "tracker_id"])
    domains = skrub.as_data_op(f"{INPUT_DIR}/domains.parquet").skb.apply_func(
        pd.read_parquet, columns=["domain", "domain_id"]
    )
    link_graph = skrub.as_data_op(f"{INPUT_DIR}/link-graph.parquet").skb.apply_func(
        pd.read_parquet, columns=["source_domain_id", "target_domain_id"]
    )
    # The original re-read this file with a sniffed separator when the tab read
    # produced fewer than 3 columns; the shipped file is tab separated, and a
    # parser fallback is not data-dependent logic.
    press = skrub.as_data_op(
        f"{INPUT_DIR}/freedom-of-the-press.csv"
    ).skb.apply_func(pd.read_csv, sep="\t")
    url_classification = skrub.as_data_op(
        f"{INPUT_DIR}/url-classification.csv"
    ).skb.apply_func(pd.read_csv, usecols=["url", "category"])

    # =====================================================================
    # 2. Build the supervised table: one row per candidate domain, in the
    #    ascending order np.setdiff1d gave the original, and the 355-column
    #    tracker indicator as the target.
    # =====================================================================
    candidate_edges = graph_train[
        ~graph_train["domain_id"].isin(target_domains["domain_id"])
    ]
    candidate_ids = (
        candidate_edges[["domain_id"]]
        .drop_duplicates()
        .sort_values("domain_id")
        .reset_index(drop=True)
    )
    domain_names = domains.drop_duplicates(subset="domain_id", keep="last")
    base = candidate_ids.merge(domain_names, on="domain_id", how="left")
    base = base.assign(domain=base["domain"].fillna(""))

    targets = build_tracker_matrix(candidate_edges, candidate_ids["domain_id"])

    y = targets.skb.mark_as_y()
    X = base.skb.mark_as_X(cv=CandidateDomainHoldout(), split_kwargs={})

    # =====================================================================
    # 3. Row-local features: hostname lexicon + press freedom + URL categories.
    # =====================================================================
    X = X.skb.apply(DomainParts())

    press_scores = (
        press.assign(
            tld=press["tld"].astype(str).str.strip().str.lower().str.lstrip("."),
            press_score=press["freedom_of_the_press"].astype("float64"),
        )[["tld", "press_score"]]
        .drop_duplicates(subset="tld", keep="last")
    )
    median_press_score = press_scores["press_score"].median()

    X = X.merge(press_scores, on="tld", how="left")
    X = X.assign(
        has_press=X["press_score"].notna().astype("float32"),
        press_score=X["press_score"].fillna(median_press_score),
    )
    X = X.assign(is_auth=(X["press_score"] > 60.0).astype("float32"))
    X = X.skb.apply(DomainLexicalFeatures())

    # Content-category mix per domain, matched by hostname with the original's
    # "www." fallback.
    url_rows = url_classification[url_classification["category"].isin(URL_CATEGORIES)]
    url_hosts = (
        url_rows["url"]
        .astype(str)
        .str.split("://", n=1)
        .str[-1]
        .str.split("/", n=1)
        .str[0]
        .str.split(":", n=1)
        .str[0]
        .str.strip()
        .str.lower()
    )
    url_rows = url_rows.assign(host=url_hosts, host_no_www=url_hosts.str.removeprefix("www."))
    name_lookup = domains[
        domains["domain_id"].isin(candidate_ids["domain_id"])
    ].drop_duplicates(subset="domain", keep="last")
    url_rows = url_rows.merge(
        name_lookup, left_on="host", right_on="domain", how="left"
    ).merge(
        name_lookup.rename(columns={"domain": "domain_www", "domain_id": "domain_id_www"}),
        left_on="host_no_www",
        right_on="domain_www",
        how="left",
    )
    url_rows = url_rows.assign(
        matched_id=url_rows["domain_id"].fillna(url_rows["domain_id_www"])
    )
    url_rows = url_rows[url_rows["matched_id"].notna()]

    category_counts = (
        url_rows.assign(matched_id=url_rows["matched_id"].astype("int64"))
        .groupby(["matched_id", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=URL_CATEGORIES, fill_value=0)
    )
    category_table = (
        category_counts.div(category_counts.sum(axis=1), axis=0)
        .rename(columns={c: f"url_cat_{c}" for c in URL_CATEGORIES})
        .assign(url_cat_has=1.0)
        .reset_index()
        .rename(columns={"matched_id": "domain_id"})
    )

    X = X.merge(category_table, on="domain_id", how="left")
    X = X.assign(**{col: X[col].fillna(0.0).astype("float32") for col in URL_CAT_COLS})

    # =====================================================================
    # 4. The model. The link graph and tracker table reach it through
    #    fit_kwargs; every label-derived block is rebuilt per fold inside fit.
    # =====================================================================
    pred = X.skb.apply(
        TrackerRankNetEstimator(),
        y=y,
        fit_kwargs={"link_graph": link_graph, "trackers": trackers},
    )

    # =====================================================================
    # 5. Score. No cv= here -- the hold-out declared on mark_as_X drives.
    # =====================================================================
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1,
            fitted=True,
            refit=False,
            scoring=RECALL_AT_10,
        )
        print(search.results_)
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(
            "Final Validation Performance: "
            f"{search.results_['mean_test_score'].iloc[0]}"
        )
