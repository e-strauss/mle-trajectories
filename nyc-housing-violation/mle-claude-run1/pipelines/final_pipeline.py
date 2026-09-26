"""FINAL refit/predict artifact for pipeline_29 -- NOT a scored candidate.

Do not run this through ml-score: it has no CV and no choices. It refits the
winning plan (pipeline_29: soft vote HistGB:LightGBM:logreg = 2:2:1 on
features_v28 = PLUTO + HPD violation/complaint history + non-HPD 311 + rodent +
DOB complaints + 5y/10y long history + ACRIS) on ALL labelled backtest rows
(cutoffs 2020, 2021, 2022) and predicts the test lots (PLUTO 22v3, unitsres >= 3,
cutoff 2023-01-01), writing <ws>/submission.parquet (bbl, score).

Unlike the scored plans (rooted on constant lake reads via load_xy), the ROWS
here are an overridable skrub.var("lots"), so the same fitted learner can be
applied to the test rows. The event tables stay recorded lake reads; every
feature uses only events before each row's own cutoff, so the test rows see
the lake up to 2023-01-01 (its end), exactly like the training rows see theirs.
"""
from pathlib import Path

import pandas as pd
import skrub

from common import (CUTOFFS, LAKE, STORAGE, TEST_CUTOFF, TEST_RELEASE, attach_label,
                    features_v28, load_events, make_ensemble, model_features, read_lots)

WS_ROOT = Path(__file__).resolve().parent.parent
TASK_ROOT = "gs://mle-nyc-lake/tasks/housing_violation_risk/v1"

# --- rows: labelled training lots (all cutoffs) and the test lots -----------------
lake = skrub.as_data_op(LAKE)
train_lots = skrub.deferred(attach_label)(lake.skb.apply_func(read_lots, CUTOFFS),
                                          load_events("hpd_violations", lake)).skb.eval()
test_lots = read_lots(LAKE, {TEST_CUTOFF: TEST_RELEASE})

# --- the pipeline_29 plan, rooted on an overridable variable ---------------------
data = skrub.var("lots", value=train_lots)
y = data["y"].skb.mark_as_y()
X = data.drop(columns=["y"], errors="ignore").skb.mark_as_X()
viol = load_events("hpd_violations")
feats = features_v28(X, viol)
pred = model_features(feats).skb.apply(make_ensemble(), y=y)

if __name__ == "__main__":
    learner = pred.skb.make_learner(fitted=True)
    proba = learner.predict_proba({"lots": test_lots})[:, 1]
    scored = pd.DataFrame({"bbl": test_lots["bbl"].to_numpy(), "score": proba})

    # align to the official test entities (order + exact set)
    entities = pd.read_parquet(f"{TASK_ROOT}/test_entities.parquet", storage_options=STORAGE)
    entities["bbl"] = entities["bbl"].astype(str)
    missing = set(entities["bbl"]) - set(scored["bbl"])
    extra = set(scored["bbl"]) - set(entities["bbl"])
    print(f"test lots built: {len(scored)}, entities: {len(entities)}, "
          f"missing: {len(missing)}, extra: {len(extra)}")
    submission = entities[["bbl"]].merge(scored, on="bbl", how="left")
    out = WS_ROOT / "submission.parquet"
    submission.to_parquet(out, index=False)
    print(f"wrote {out} ({len(submission)} rows)")
