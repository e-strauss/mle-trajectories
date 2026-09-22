"""Build a small, self-consistent sample of the playground-series-s6e7 input folder.

Writes playground-series-s6e7/sample/input/ (gitignored) from the full data in
./input/, so a skrubified pipeline and its original can be run side by side in
~1 minute instead of many:

    python -m skrubify <run>/pipelines/x.py --run-in playground-series-s6e7/sample

S6E7_OUT builds a different size into its own run-root:

    S6E7_N_TRAIN=60000 S6E7_N_TEST=25000 \
        S6E7_OUT=playground-series-s6e7/sample_60k/input \
        python playground-series-s6e7/make_sample.py

This is a single flat table, so a row sample is the right shape -- but it has to
be STRATIFIED. The metric is balanced accuracy over a 15:1 imbalance
(at-risk 592,561 / unhealthy 57,724 / fit 39,803), so an unstratified sample
moves the class proportions the pipelines engineer around, and every one of them
scores the rare classes as heavily as the common one. Proportions are kept rather
than balanced, because the imbalance IS the task: a sample that evens the classes
out makes balanced accuracy easy and stops measuring what the original measured.

`id` is a single sequential range that runs through train and on into test
(train 0..690,087, test from 690,088), so train and test are sampled separately
and sample_submission.csv is filtered to exactly the sampled test ids -- a
submission-writing pipeline fails on row count otherwise.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("S6E7_SRC", HERE / "input"))
OUT = Path(os.environ.get("S6E7_OUT", HERE / "sample" / "input"))

# Measured, not guessed: pipelines/0001 (LightGBM + XGBoost + CatBoost at
# n_estimators=1000 plus a torch net, 5 folds) runs in 57s wall at 10k train
# rows and was still going past 110s / 98 CPU-minutes at 60k. Rows are not the
# only cost driver here, so raise these only after re-timing.
N_TRAIN = int(os.environ.get("S6E7_N_TRAIN", 10_000))   # rows kept from train.csv
N_TEST = int(os.environ.get("S6E7_N_TEST", 5_000))      # rows kept from test.csv

TARGET = "health_condition"
N_SPLITS = 5   # StratifiedKFold(n_splits=5) in the run's pipelines

OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
counts = {}

print(f"reading {SRC}/train.csv ...", flush=True)
train = pd.read_csv(SRC / "train.csv")
counts["train.csv"] = {"src": len(train)}
print(f"  {len(train)} rows, classes {train[TARGET].value_counts().to_dict()}", flush=True)

# Proportional allocation per class, with a floor of N_SPLITS so no class can be
# reduced below what StratifiedKFold needs. The floor is what makes the sample
# safe at small N_TRAIN, not a correction to the proportions at the default size.
frac = min(1.0, N_TRAIN / len(train))
keep = []
for cls, grp in train.groupby(TARGET, sort=True):
    n = max(N_SPLITS, min(len(grp), int(round(len(grp) * frac))))
    keep.append(rng.choice(grp.index.to_numpy(), n, replace=False))
    print(f"  {cls}: {len(grp)} -> {n}", flush=True)
train_idx = np.sort(np.concatenate(keep))
train_sub = train.loc[train_idx].reset_index(drop=True)
counts["train.csv"]["out"] = len(train_sub)
train_sub.to_csv(OUT / "train.csv", index=False)
print(f"  wrote train.csv: {len(train_sub)} rows, "
      f"classes {train_sub[TARGET].value_counts().to_dict()}", flush=True)
del train

print(f"reading {SRC}/test.csv ...", flush=True)
test = pd.read_csv(SRC / "test.csv")
counts["test.csv"] = {"src": len(test)}
n_test = min(len(test), N_TEST)
test_idx = np.sort(rng.choice(test.index.to_numpy(), n_test, replace=False))
test_sub = test.loc[test_idx].reset_index(drop=True)
counts["test.csv"]["out"] = len(test_sub)
test_sub.to_csv(OUT / "test.csv", index=False)
print(f"  wrote test.csv: {len(test_sub)} rows", flush=True)

print(f"reading {SRC}/sample_submission.csv ...", flush=True)
sub = pd.read_csv(SRC / "sample_submission.csv")
counts["sample_submission.csv"] = {"src": len(sub)}
kept_ids = pd.Index(test_sub["id"])
sub_sub = sub[sub["id"].isin(kept_ids)].reset_index(drop=True)
assert len(sub_sub) == len(test_sub), (
    f"sample_submission has {len(sub_sub)} of the {len(test_sub)} sampled test ids")
counts["sample_submission.csv"]["out"] = len(sub_sub)
sub_sub.to_csv(OUT / "sample_submission.csv", index=False)
print(f"  wrote sample_submission.csv: {len(sub_sub)} rows", flush=True)

manifest = {
    "dataset": "playground-series-s6e7",
    "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "script": "make_sample.py",
    "strategy": "proportional stratified row sample on health_condition; "
                "test sampled independently, sample_submission filtered to the "
                "sampled test ids",
    "source": str(SRC),
    "out": str(OUT),
    "seed": 0,
    "knobs": {"S6E7_N_TRAIN": N_TRAIN, "S6E7_N_TEST": N_TEST, "N_SPLITS_FLOOR": N_SPLITS},
    "rows": counts,
    "class_counts": {
        "src": {str(k): int(v) for k, v in
                pd.read_csv(SRC / "train.csv", usecols=[TARGET])[TARGET]
                .value_counts().items()},
        "out": {str(k): int(v) for k, v in train_sub[TARGET].value_counts().items()},
    },
}
(OUT.parent / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"done -> {OUT}  (manifest: {OUT.parent / 'sample_manifest.json'})", flush=True)
