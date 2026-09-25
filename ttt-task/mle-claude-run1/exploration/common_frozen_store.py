"""Plan-building helpers for the TrackTheTrackers workspace.

THE TASK, restated as a supervised problem
------------------------------------------
`target.tsv` holds 50,000 domains that appear **nowhere** in
`tracking_graph_train` (verified in data_exploration_2/3: 0% overlap), and we
must name up to 10 trackers each, scored by **Recall@10**. So this is a pure
*cold-start multi-label* problem over a fixed label space of 355 trackers:

    one row      = one domain
    y            = 355 binary indicators (y_0 .. y_354)
    prediction   = a 355-wide score matrix; the top 10 per row are the guess
    metric       = recall_at_10 (mean over domains of |top10 ∩ true| / |true|)

Because the metric is a ranking metric over a matrix output, it cannot be an
sklearn scorer string -> it is declared **in the plan** via `attach_scoring`
(SCORER/SCORER_NAME below) and the harness locks it as `plan:recall_at_10`.
Models therefore expose a `predict` returning the (n, 355) score matrix.

THE ROW SAMPLE IS FROZEN
------------------------
The raw data (18.7M tracked domains, 623M link edges) cannot be
cross-validated directly, so `data_exploration_4.py` built ONE deterministic
working sample plus its precomputed, leakage-checked feature blocks under
`../features/` (train rows = tracked domains with `domain_id % 37 == 11`, i.e.
505,548 rows; a matching `*_target.parquet` exists for the 50k submission rows).
Every pipeline reads those same files, so all scores are comparable. **Never
rebuild or edit the feature store** — that would silently invalidate the board.
Adding a NEW block file is safe (rows and labels are untouched); a pipeline then
just opts into it via `load_xy(blocks=...)`.

Feature blocks (all computable for an unseen hostname, none reads the row's own
trackers -- see the leakage notes in data_exploration_4.py):
  nbr_out  no_0..no_354   # trackers of the domains this domain links TO
  nbr_in   ni_0..ni_354   # trackers of the domains that link TO this domain
  tld_pop  tp_0..tp_354   # leave-one-out P(tracker | TLD)
  meta                    # hostname, tld, reg_dom, degrees, press freedom, ...
"""
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

SEED = 42
N_FOLDS = 3
N_TRK = 355
K = 10                      # Recall@K -- the task's submission cap
WS_ROOT = Path(__file__).resolve().parent.parent
FEAT = WS_ROOT / "features"

YCOLS = [f"y_{i}" for i in range(N_TRK)]
ALL_BLOCKS = ("meta", "nbr_out", "nbr_in", "tld_pop")

# column-name prefixes of the three 355-wide score blocks
P_OUT, P_IN, P_TLD = "no_", "ni_", "tp_"


def make_cv():
    """The workspace's CV splitter -- the SINGLE place the split is defined.

    Rows are independent domains (one row per domain, no repeats), and target
    domains are held out at the domain level, so a plain shuffled KFold over the
    frozen sample is a faithful proxy for the real task: exploration_3 confirmed
    the sample matches target.tsv on tracked-neighbour coverage (91.7% vs 90.8%),
    TLD mix and label-count mix. 3 folds => ~168k test domains per fold, i.e. a
    standard error of ~0.0009 on the recall estimate.
    """
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def _read(name):
    """Recorded parquet read: a DataOp, not a frame."""
    return skrub.as_data_op(FEAT / name).skb.apply_func(pd.read_parquet)


def load_frame(blocks=ALL_BLOCKS, split="train", subsample=None):
    """Recorded read of the frozen store: labels (train only) + feature blocks.

    Rows come from `labels_train.parquet` (train) / `meta_target.parquet`
    (target) and every block is left-joined on `domain_id`, so row order is the
    frozen store order -- identical for every pipeline, hence identical folds.
    """
    if split == "train":
        data = _read("labels_train.parquet")
    else:
        data = _read(f"meta_{split}.parquet").loc[:, ["domain_id"]]
    if subsample:
        data = data.skb.subsample(n=subsample)
    for b in blocks:
        data = data.merge(_read(f"{b}_{split}.parquet"), on="domain_id", how="left")
    return data


def load_xy(blocks=ALL_BLOCKS, subsample=3_000):
    """The standard first line of every pipeline: (X, y) DataOps, marked.

    `y` is the RAW 355-column label matrix (uint8) -- never transformed before
    the mark, so every pipeline scores in the same domain. The CV splitter from
    make_cv() is attached here on mark_as_X (the only supported place).
    `subsample` only shrinks the eager *preview*, never the CV score.

    `blocks` selects which precomputed feature blocks to join; the row set and
    labels are unaffected, so pipelines using different blocks stay comparable.
    """
    data = load_frame(blocks, "train", subsample=subsample)
    y = data[YCOLS].skb.mark_as_y()
    X = data.drop(columns=YCOLS).skb.mark_as_X(cv=make_cv())
    return X, y


# --- The metric: Recall@10 over the 355-tracker label space ----------------
def recall_at_k(y_true, y_score, k=K):
    """Mean over rows of |top-k predicted ∩ true| / |true| -- the task metric.

    `y_score` is the model's (n, 355) score matrix (`response_method="predict"`).
    Ties are broken arbitrarily by argpartition, which mirrors submitting an
    arbitrary 10 among equally-scored trackers.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_score.ndim == 1:                     # degenerate 1-column output
        y_score = y_score.reshape(-1, 1)
    n_true = y_true.sum(axis=1)
    k = min(k, y_score.shape[1])
    top = np.argpartition(-y_score, k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top, axis=1).sum(axis=1)
    ok = n_true > 0
    return float((hits[ok] / n_true[ok]).mean()) if ok.any() else 0.0


SCORER = make_scorer(recall_at_k, response_method="predict", greater_is_better=True)
SCORER_NAME = "recall_at_10"


def attach_scoring(pred, sample_weight=None):
    """Attach the workspace's locked Recall@10 scorer to the final node.

    Recall@10 is a ranking metric over a matrix-valued prediction, so no sklearn
    scorer string can express it -- every pipeline must end with this call and
    ml-score must run WITHOUT `--scoring` (the harness then passes scoring=None
    so the plan's scorer drives, and locks `plan:recall_at_10`).
    """
    kwargs = {"sample_weight": sample_weight} if sample_weight is not None else None
    return pred.skb.with_scoring(SCORER, kwargs=kwargs, name=SCORER_NAME)
