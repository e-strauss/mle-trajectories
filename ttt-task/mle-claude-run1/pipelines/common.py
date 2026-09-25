"""Plan-building helpers for the TrackTheTrackers run (inlined feature version).

THE TASK, restated as a supervised problem
------------------------------------------
`target.tsv` holds 50,000 domains that appear **nowhere** in
`tracking_graph_train` (0/50,000 overlap, verified in
`../exploration/data_exploration_2.py`), and we must name up to 10 trackers
each, scored by **Recall@10**. So this is a pure *cold-start multi-label*
problem over a fixed label space of 355 trackers:

    one row      = one domain
    y            = 355 binary indicators (y_0 .. y_354)
    prediction   = a 355-wide score matrix; the top 10 per row are the guess
    metric       = recall_at_10 (mean over domains of |top10 ∩ true| / |true|)

Because the metric is a ranking metric over a matrix output, it cannot be an
sklearn scorer string -> it is declared **in the plan** via `attach_scoring`
(SCORER/SCORER_NAME below). Models therefore expose a `predict` returning the
(n, 355) score matrix.

WHAT CHANGED FROM THE ORIGINAL WORKSPACE (normalisation for this collection)
---------------------------------------------------------------------------
The original run materialised its features once into parquet and every pipeline
read them back. Here the computation is **inlined**: `load_xy` records the
builders in `features.py` as plan nodes, so a pipeline derives everything from
`input/` by itself and nothing is cached between runs. The features are
identical -- `../verify_inline_features.py` checks every block against the
original store, and the pipeline files themselves are untouched, because
`load_xy(blocks=...)` kept its signature.

The row sample is still frozen *by construction* rather than by file: tracked
domains with `domain_id % 37 == 11` (505,548 rows), sorted by domain_id. Row
order is what makes scores comparable across the run.

Cost: one build is a full scan of the 623M-edge link graph plus the per-block
work, about 3-6 minutes depending on which blocks a pipeline asks for, against
~20s for the old parquet read. It happens once per run (the blocks are built
before `mark_as_X`; see the placement discussion in `features.py`), not once
per fold.
"""
import os
from pathlib import Path

import numpy as np
import skrub
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold

import features

SEED = 42
N_FOLDS = 3
N_TRK = 355
K = 10                      # Recall@K -- the task's submission cap
YCOLS = [f"y_{i}" for i in range(N_TRK)]
ALL_BLOCKS = ("meta", "nbr_out", "nbr_in", "tld_pop")

# Collection convention: TTT_INPUT points at the task data. Defaults to the
# sibling `input/` symlink so the run works in place.
INPUT_DIR = Path(os.environ.get(
    "TTT_INPUT", Path(__file__).resolve().parent.parent / "input"))

# prefix carried by each 355-wide block, used to select it out of X
BLOCK_PREFIX = {"nbr_out": "no_", "nbr_in": "ni_", "tld_pop": "tp_",
                "nbr_hub": "nh_", "nbr_rec": "nr_", "nbr_2h": "n2_",
                "direct": "dl_", "nbr_frac": "nf_", "trk_cooc": "tc_",
                "tok_pop": "kp_"}


def make_cv():
    """The workspace's CV splitter -- the SINGLE place the split is defined.

    Rows are independent domains (one row each, no repeats) and the real target
    set is held out at the domain level, so a plain shuffled KFold over the
    frozen sample is a faithful proxy: `data_exploration_3.py` confirmed the
    sample matches `target.tsv` on tracked-neighbour coverage (91.7% vs 90.8%),
    TLD mix and label-count mix. 3 folds => ~168k test domains per fold, i.e. a
    standard error of ~0.0009 on the recall estimate.
    """
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# --- the recorded feature graph -------------------------------------------
def build_blocks(blocks, split="train", input_dir=None):
    """Record one node per feature block; returns {name: DataOp}.

    Only the requested blocks (and whatever they depend on) become nodes, so a
    pipeline that asks for `meta` alone does not pay for the 2-hop walk.
    """
    d = skrub.as_data_op(Path(input_dir or INPUT_DIR))
    ctx = d.skb.apply_func(features.load_context)
    need = set(blocks)

    # shared intermediates, created on demand
    cache = {}

    def edges():
        if "edges" not in cache:
            cache["edges"] = skrub.deferred(features.seed_edges)(ctx)
        return cache["edges"]

    def linkdeg():
        if "linkdeg" not in cache:
            cache["linkdeg"] = skrub.deferred(features.link_degrees)(ctx)
        return cache["linkdeg"]

    def hub():
        if "hub" not in cache:
            cache["hub"] = skrub.deferred(features.block_nbr_hub)(
                ctx, edges(), linkdeg())
        return cache["hub"]

    def two_hop():
        if "n2" not in cache:
            cache["n2"] = skrub.deferred(features.block_nbr_2h)(
                ctx, edges(), linkdeg())
        return cache["n2"]

    def labels_all():
        if "labels" not in cache:
            cache["labels"] = skrub.deferred(features.block_labels)(ctx)
        return cache["labels"]

    out = {}
    if "labels" in need:
        out["labels"] = labels_all()
    if "meta" in need:
        out["meta"] = skrub.deferred(features.block_meta)(ctx, edges())
    if "nbr_out" in need:
        out["nbr_out"] = skrub.deferred(features.block_nbr_out)(ctx, edges())
    if "nbr_in" in need:
        out["nbr_in"] = skrub.deferred(features.block_nbr_in)(ctx, edges())
    if "tld_pop" in need:
        out["tld_pop"] = skrub.deferred(features.block_tld_pop)(ctx, labels_all())
    if "nbr_hub" in need:
        out["nbr_hub"] = hub()
    if "nbr_rec" in need:
        out["nbr_rec"] = skrub.deferred(features.block_nbr_rec)(ctx, edges())
    if "nbr_2h" in need:
        out["nbr_2h"] = two_hop()
    if "nbr_frac" in need:
        out["nbr_frac"] = skrub.deferred(features.block_nbr_frac)(ctx, edges())
    if "direct" in need:
        out["direct"] = skrub.deferred(features.block_direct)(ctx)
    if "trk_cooc" in need:
        out["trk_cooc"] = skrub.deferred(features.block_trk_cooc)(ctx, hub())
    if "tok_pop" in need:
        out["tok_pop"] = skrub.deferred(features.block_tok_pop)(ctx)
    if "meta2" in need:
        out["meta2"] = skrub.deferred(features.block_meta2)(
            ctx, edges(), linkdeg(), hub(), two_hop())

    unknown = need - set(out)
    if unknown:
        raise KeyError(f"unknown block(s): {sorted(unknown)}")
    return {k: skrub.deferred(features.split_rows)(v, ctx, split)
            for k, v in out.items()}, ctx


def load_frame(blocks=ALL_BLOCKS, split="train", with_labels=True, input_dir=None):
    """Recorded build + join: labels (train only) + the requested feature blocks."""
    want = tuple(blocks) + (("labels",) if with_labels and split == "train" else ())
    built, _ = build_blocks(want, split, input_dir)
    if with_labels and split == "train":
        data = built["labels"]
    else:
        data = built[blocks[0]].loc[:, ["domain_id"]]
    for b in blocks:
        data = data.merge(built[b], on="domain_id", how="left")
    return data


def load_xy(blocks=ALL_BLOCKS, subsample=None):
    """The standard first line of every pipeline: (X, y) DataOps, marked.

    `y` is the RAW 355-column label matrix (uint8) -- never transformed before
    the mark, so every pipeline scores in the same domain. The CV splitter from
    make_cv() is attached here on mark_as_X (the only supported place).

    Built under `eager_data_ops=False`: the eager preview would run the whole
    623M-edge feature build at plan-construction time, and then again when the
    harness fits. `subsample` is accepted for signature compatibility and
    ignored -- previews are off.
    """
    with skrub.config_context(eager_data_ops=False):
        data = load_frame(blocks, "train")
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
    if y_score.ndim == 1:
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
    ml-score must run WITHOUT `--scoring`.
    """
    kwargs = {"sample_weight": sample_weight} if sample_weight is not None else None
    return pred.skb.with_scoring(SCORER, kwargs=kwargs, name=SCORER_NAME)
