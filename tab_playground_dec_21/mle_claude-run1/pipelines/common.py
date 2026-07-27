"""Plan-building helpers for skrub DataOps pipelines in this workspace.

Pipelines are built as skrub DataOps plans (see skrub_dataops_guide.md in the
workspace root). A pipeline file only DEFINES a plan: a module-level `pred`
(the final prediction node, raw-domain output), `DESCRIPTION`, and optionally
`PARENT`. Scoring and results.json bookkeeping are owned by the ml-score
harness (.claude/skills/ml-score/scripts/score_pipeline.py), which scores via
the plan's CV and the workspace's locked scorer — pipelines never score
themselves.

THIS FILE IS A TEMPLATE — adapt it to the task BEFORE writing the first
pipeline, then keep it stable so every pipeline is comparable:
  - load_csv / load_xy : change the filename and the reader to match the task's
    data — pd.read_csv, pd.read_parquet / pl.read_parquet, a partitioned parquet
    directory, or several tables read separately and joined for a multi-table
    plan. The default (single train.csv via pandas) is just the common case.
  - make_cv()          : the CV splitter — switch from KFold to a group/time
    splitter when rows are not independent (see its docstring).
  - SCORER/SCORER_NAME : only for a custom/weighted metric a scorer string can't
    express (see attach_scoring).
The harness reads the CV and scorer off the plan and LOCKS them on the first
scored pipeline (warning/refusing if they change mid-workspace), so settle these
up front.
"""
from pathlib import Path

import pandas as pd
import skrub
from sklearn.model_selection import KFold

SEED = 42
N_FOLDS = 3
WS_ROOT = Path(__file__).resolve().parent.parent


def make_cv():
    """The workspace's CV splitter -- the SINGLE place the split is defined.

    Returned by load_xy() and attached to the plan via mark_as_X(cv=...), so
    every pipeline in this workspace cross-validates the same way and scores stay
    comparable. The ml-score harness does NOT pass a cv of its own; it reads the
    one declared here off the plan (and records it in workspace.json, warning if
    it ever changes mid-workspace).

    To use a specialised split for this task, edit this ONE function (and pass
    `groups=` to load_xy if the splitter needs them), e.g.:

        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=N_FOLDS)            # + load_xy(..., groups="user_id")

    Keep it fixed for the rest of the workspace once the first pipeline is scored.
    """
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def load_csv(name="train.csv", **kwargs):
    """Recorded data read: returns a DataOp (the root of the plan), not a frame.

    ADAPT to the task's data format: swap `pd.read_csv` for `pd.read_parquet`
    (or `pl.read_parquet`) and the `name` for the right file/dir — e.g. a
    partitioned `train.parquet/` directory. Only needed directly for multi-table
    plans (read each table, join with the recorded pandas API before the marks);
    single-table pipelines should start from load_xy() instead.
    """
    return skrub.as_data_op(WS_ROOT / "input" / name).skb.apply_func(pd.read_csv, **kwargs)


def load_xy(target, name="train.csv", subsample=50_000, groups=None, **kwargs):
    """Recorded load + the X/y split: returns (X, y) DataOps, marked.

    The standard first line of every pipeline. y is the RAW target column --
    the CV score is computed against this node, so all pipelines score in the
    same domain regardless of any internal target transform (guide section 8:
    transform y after the mark, invert predictions at predict time gated on
    skrub.eval_mode()). `subsample` only affects previews, never the CV score.

    The CV splitter from make_cv() is attached to X here via mark_as_X -- this
    is the ONLY supported way to set the CV (the harness does not pass one, and
    split_kwargs such as `groups` can only be wired on mark_as_X). Pass
    `groups="col"` for a grouped/stratified-by-group splitter (e.g. GroupKFold);
    the column is read from the raw data, used as split groups, and dropped from
    X so it is not used as a feature.
    """
    data = load_csv(name, **kwargs)
    if subsample:
        data = data.skb.subsample(n=subsample)
    y = data[target].skb.mark_as_y()
    drop_cols = [target] + ([groups] if groups else [])
    split_kwargs = {"groups": data[groups]} if groups else None
    X = data.drop(columns=drop_cols).skb.mark_as_X(cv=make_cv(),
                                                   split_kwargs=split_kwargs)
    return X, y


# --- Custom / weighted scoring (optional) ----------------------------------
# Most tasks score with a plain sklearn scorer STRING passed to the ml-score
# harness (--scoring r2 / neg_root_mean_squared_error / ...). Use the hook below
# ONLY when the metric cannot be expressed as a string: a custom callable, or a
# sample-weighted metric. Define it ONCE here (the harness locks it and keeps it
# fixed for the workspace, like make_cv) and attach it to the final node.
#
# A scorer declared in the plan drives only when the harness passes scoring=None,
# which it does whenever it detects a with_scoring node -- so just build `pred`
# through attach_scoring() and run ml-score WITHOUT --scoring.
SCORER = None       # e.g. make_scorer(weighted_r2, response_method="predict")
SCORER_NAME = None  # e.g. "weighted_r2" -- the locked metric label on the board


def attach_scoring(pred, sample_weight=None):
    """Attach the workspace's custom scorer to the final prediction node.

    No-op when SCORER is None (the normal case: the harness uses --scoring).
    `sample_weight` must be a DataOp derived from the marked X so it is split per
    fold and stays row-aligned with the predictions -- typically a weight column
    kept in X and excluded from the model's features, e.g.:

        X, y = load_xy(target="resp")          # weight column stays in X
        feats = X.skb.drop("weight")           # model sees everything but weight
        pred = feats.skb.apply(model, y=y)
        pred = attach_scoring(pred, sample_weight=X["weight"])
    """
    if SCORER is None:
        return pred
    kwargs = {"sample_weight": sample_weight} if sample_weight is not None else None
    return pred.skb.with_scoring(SCORER, kwargs=kwargs, name=SCORER_NAME)
