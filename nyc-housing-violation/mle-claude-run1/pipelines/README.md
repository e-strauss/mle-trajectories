# housing_violation_risk v2: systematic lake exploration (15 pipelines)

## Setup (Phase 0/1)
- **Rows**: (lot, cutoff) for three backtest cutoffs, each built like the test set.
  Each cutoff uses the PLUTO release before it, restricted to lots with `unitsres >= 3`:
  2020-01-01 ← 19v2 (171,083 lots, 10.5% positive), 2021-01-01 ← 20v7 (170,641, 12.6%),
  2022-01-01 ← 21v4 (171,403, 13.5%). The test is 22v3 at 2023-01-01.
- **Label**: at least one Class C HPD violation inspected in [cutoff, cutoff+12 months).
- **CV**: `CutoffSplit`, an expanding window. Fold 1 trains on 2020 and scores 2021;
  fold 2 trains on 2020+2021 and scores 2022. Each fold scores one future year, like
  the test. Folds are shared by all pipelines, so compare them pairwise.
- **Metric**: `average_precision` (task metric).
- **Point-in-time rule**: features use only event rows dated before each row's own
  cutoff, and only immutable fields of those rows. No status/disposition columns and
  no "is on a list" flags from snapshot tables.
- Everything is read from the lake inside the plan. Nothing is cached. Exploration
  scripts are fine-grained skrub DataOps evaluated with `.skb.eval()`.
- `common.py`: `EventSpec` registry → a node chain per table
  (read → filter → dates → BBL key → events). Key resolution goes BBL, then
  borough+block+lot, then BIN→BBL via `building_footprints`. `event_features` is the
  generic per-table block: counts over 90d/1y/3y, top-5 categories chosen on
  pre-2020 events only, recency, optional value sum.

## Triage (Phase 2, `data_exploration_1.py` + `data_exploration_2.py`)
`data_exploration_1.py` profiles all 71 tables: key type, date columns, row counts.
`data_exploration_2.py` audits the 16 event tables with lot keys: key match rate,
coverage at each cutoff *including test*, and 1-year standalone AP.

- **Tax liens rejected before modelling**: coverage is 0% at the test cutoff.
- **Evictions**: coverage collapses to 0.05% in 2022 (COVID moratorium), then 1.4% at test.
- **Bedbug**: filings jump from 16–19% to 44% of lots from 2022 on.
- **311**: split across two tables (at 2020). Unioned; HPD-routed requests excluded as
  duplicates of hpd_complaints.
- **No lot-level key**: fire incidents (alarm boxes) and NYPD (coordinates only).

## Results
Current best: **pipeline_15, AP 0.7251**. Δ is fold by fold against the parent.

| pipeline | parent | change | AP | folds (2021 / 2022) | Δ vs parent | verdict |
|---|---|---|---|---|---|---|
| 01 | – | anchor: PLUTO + HPD violations + HPD complaints | 0.7237 | .7137 / .7338 | – | base |
| 02 | 01 | + litigations | 0.7240 | .7141 / .7338 | +.0005 / +.0000 | flat |
| 03 | 01 | + OMO charges | 0.7239 | .7137 / .7341 | +.0000 / +.0003 | flat |
| 04 | 01 | + HWO charges | 0.7236 | .7134 / .7338 | −.0003 / −.0000 | flat |
| 05 | 01 | + vacate orders + AEP | 0.7238 | .7137 / .7339 | +.0001 / +.0001 | flat |
| 06 | 01 | + bedbug reports | 0.7236 | .7138 / .7335 | +.0001 / −.0003 | flat |
| **07** | 01 | **+ 311 (non-HPD)** | **0.7248** | .7148 / .7347 | **+.0011 / +.0010** | **accepted** |
| 08 | 01 | + DOB violations + ECB | 0.7237 | .7139 / .7335 | +.0002 / −.0003 | flat |
| 09 | 01 | + DOB complaints | 0.7240 | .7139 / .7341 | +.0002 / +.0003 | tiny |
| 10 | 01 | + rodent inspections | 0.7242 | .7141 / .7342 | +.0004 / +.0004 | tiny |
| 11 | 01 | + evictions | 0.7242 | .7142 / .7342 | +.0005 / +.0004 | tiny, drift |
| 12 | 07 | + tax-block neighbourhood (own lot excluded) | 0.7245 | .7147 / .7344 | −.0001 / −.0004 | rejected |
| 13 | 07 | + 30-day windows (HPD, complaints, 311) | 0.7245 | .7145 / .7345 | −.0003 / −.0003 | rejected |
| 14 | 07 | HistGB sweep (12 settings) | 0.7245 | .7145 / .7346 | −.0004 / −.0001 | flat |
| 15 | 07 | + rodent + DOB complaints | **0.7251** | .7152 / .7349 | +.0004 / +.0001 | best, noise-level |

## Error analysis (Phase 6, `data_exploration_3.py`, fold 2022 rebuilt by hand, AP 0.7345)
| segment (history before cutoff) | lots | positive rate | share of positives | AP within |
|---|---|---|---|---|
| C in last 1y | 21.5k | 61.6% | 57.5% | 0.89 |
| C 1–3y ago only | 13.1k | 29.1% | 16.4% | 0.57 |
| violations, no C | 81.8k | 5.5% | 19.4% | 0.26 |
| no HPD violation in 3y | 55.0k | 2.8% | 6.7% | 0.19 |

Repeat offenders are already ranked almost perfectly. The remaining error is the 42%
of positives with no C in the last year, and no table tried so far helps with them.

## Takeaways
- The lot's own HPD history (violations and complaints) carries almost all the signal
  (about 0.72 of the 0.725). Of 13 extra tables/ideas, only 311 cleared +0.001 on both
  folds. The rest are within ±0.0005.
- Enforcement tables (OMO, HWO, litigations, vacate, AEP) are consequences of
  violations. They add no new information.
- The model is saturated on these features: 12 hyperparameter settings span only 0.003.
- Noise: the gap between folds is about 0.02 (year effect). Pairwise fold deltas below
  ±0.0005 should be treated as noise.
- The error analysis points to *inspection propensity* for lots with no recent C
  violation as the open problem. Candidates are ownership/management signals
  (registration contacts, ACRIS sales, owner portfolios; point-in-time handling
  needed) and longer history (more than 3 years).

## Model families (pipelines 16–23, all on the pipeline_15 feature set `features_v15`)
Current best: **pipeline_23, AP 0.7260**. Δ is fold by fold against pipeline_15
(.7152 / .7349).

| pipeline | parent | model | best setting | AP | folds (2021 / 2022) | Δ vs 15 |
|---|---|---|---|---|---|---|
| **23** | 22 | **soft vote HistGB : LightGBM : logreg = 2:2:1** (promoted) | – | **0.7260** | .7163 / .7357 | +.0011 / +.0008 |
| 22 | 21 | soft-vote ensemble, weight sweep | trees 2 : linear 1 | 0.7260 | .7162 / .7357 | +.0010 / +.0009 |
| 21 | 16 | LightGBM, simpler grid | 400 trees, 15 leaves | 0.7257 | .7159 / .7355 | +.0006 / +.0006 |
| 16 | 15 | LightGBM | 400 trees, 15 leaves | 0.7255 | .7157 / .7353 | +.0005 / +.0005 |
| 17 | 15 | XGBoost (GPU, native categoricals) | depth 4, 400 trees | 0.7248 | .7147 / .7349 | −.0005 / +.0000 |
| 18 | 15 | CatBoost (GPU) | depth 6, 1000 iterations | 0.7244 | .7147 / .7342 | −.0006 / −.0007 |
| 19 | 15 | logistic regression (dense, quantile-scaled) | C = 1 | 0.7186 | .7091 / .7282 | −.0061 / −.0067 |
| 20 | 15 | MLP (skorch, GPU) | 256 wide, 10 epochs | 0.7136 | .7022 / .7250 | −.0130 / −.0099 |

- **Tree families tie** at 0.724–0.726. In every sweep the simplest setting wins, and
  more capacity costs up to 0.007 (0.009 for XGBoost). That fits overfitting to
  year-specific patterns that don't carry over to the next year.
- **The linear model is only 0.007 behind.** The ranking is mostly monotone in "how much
  recent HPD trouble".
- **The MLP falls apart with capacity**: 0.643 at 512 wide and 30 epochs. Its std of
  0.0015 there comes with a *bad* score, so it is not a leakage tell. An NN isn't a fit
  for this feature set.
- **The ensemble gain comes from the linear model's diversity.** Voting over trees only
  gives 0.7256, the same as LightGBM alone.

## Incident: results.json truncated (2026-09-25)
The shared `/home` NFS filled up (ENOSPC) while pipeline_23 was being recorded. The
harness then wrote `results.json` in place and truncated it to 0 bytes.
- **Fix:** the harness (`.claude/skills/ml-score/scripts/score_pipeline.py`) now writes
  every file atomically: a temp file in the same directory, fsync, then `os.replace`.
  A failed write leaves the old file intact.
- **Rebuild:** by Elias's decision, `results.json` was rebuilt from the harness output
  logged in the session, not re-scored. Every entry carries `extra.rebuilt`. Grid rows
  lack the timing columns. Everything will be re-scored offline later.

## Ownership, violation types, long history (pipelines 24–32)
Screened on LightGBM (`make_lgbm`, the pipeline_21 config). Accepted blocks go into
`features_v28` and the ensemble (`make_ensemble`).
Current best: **pipeline_29, AP 0.7318**.

| pipeline | parent | change | AP | folds (2021 / 2022) | Δ vs parent | verdict |
|---|---|---|---|---|---|---|
| 24 | 21 | + ACRIS ownership/financing (deeds, mortgages, AL&R, satisfactions; sale recency) | 0.7262 | .7164 / .7360 | +.0005 / +.0005 | small, kept |
| 25 | 21 | + violation type mix (top-10 C and B order codes) | 0.7257 | .7159 / .7355 | +.0001 / +.0001 | flat |
| **26** | 21 | **+ long history: violations (all, C) + complaints over 5y/10y** | **0.7311** | .7215 / .7406 | **+.0056 / +.0052** | **accepted** |
| 27 | 26 | ablation: 5y windows only | 0.7284 | .7189 / .7380 | −.0026 / −.0026 | 10y window carries real signal |
| 28 | 26 | 26 + 24 (`features_v28`) | 0.7313 | .7219 / .7407 | +.0004 / +.0001 | kept |
| **29** | 28 | **ensemble 2:2:1 on `features_v28`** | **0.7318** | .7223 / .7412 | +.0004 / +.0005 | **best** |
| 30 | 28 | + owner portfolio (ACRIS grantee of the latest deed) | 0.7315 | .7220 / .7411 | +.0001 / +.0004 | rejected |
| 31 | 28 | + per-unit rates | 0.7314 | .7219 / .7409 | +.0000 / +.0002 | flat |
| 32 | 28 | + complaint type mix (top-10 categories) | 0.7316 | .7220 / .7412 | +.0001 / +.0005 | tiny |

### Point-in-time decisions in this round
- **HPD registrations / registration contacts: not used.** They only hold the current
  (2023) registration, which would be future owners for the 2020–2022 rows.
- **"Open violations at cutoff": not built.** The status fields show the 2023 state,
  and a status change after T is itself a sign of a re-inspection after T, which
  correlates with the label. That is a subtle route-2 leak.
- **ACRIS: dated documents only.** A document counts from its `recorded_datetime`.
  Owner = the grantee of the latest deed recorded before T.

### Audits
- **Long history (`data_exploration_5.py`).** No feature is near-perfect on its own
  (best 0.67), and fold stds are ordinary.
  - Coverage parity holds, including test: 10y Class C coverage is 29.4 / 30.3 /
    31.6 / 32.6% from 2020 through test.
  - **Caveat:** the HPD violations base is "open as of 2012-10-01". Violation counts
    ramp from 114k in 2008 to 380k in 2013 while complaints stay flat, so pre-2013
    years are an incomplete, open-only subset. The 10y window for the 2020/2021
    cutoffs reaches into that era. At the test cutoff the window (2013+) is fully
    complete.
  - Ablation 27 shows the 10y window still adds +0.0026 on both folds, so it is
    kept, and nothing reaches further back.
- **Owner portfolio (`data_exploration_6.py`).**
  - Only 44–50% of lots get an owner, because deeds are read from 2008 on.
  - Coverage drifts upward per cutoff: 43.9% → 50.2% at test.
  - The largest "portfolios" are condo-unit lots of single buildings (Parkchester
    6,380 unit lots, vacation-ownership plans).
  - Standalone AP is 0.16–0.21. Real landlord networks would need the snapshot-only
    registration contacts.

### Takeaways
- **Time depth was the missing dimension, not more tables.** The lot's own HPD history
  over 5–10 years added about +0.005, more than all 13 extra tables combined. Chronic
  buildings stay chronic.
- Ownership/financing (ACRIS) helps a little (+0.0005). Violation-type mix, per-unit
  normalization, complaint categories and owner portfolios are all at noise level.
- The model and ensemble findings from 16–23 hold on the richer features: the ensemble
  adds about +0.0005.

## Submission (2026-09-25)
- `pipelines/final_pipeline.py` refits **pipeline_29** (ensemble 2:2:1 on
  `features_v28`) on all labelled backtest rows: cutoffs 2020–2022, 513,127 lots.
- It predicts the 22v3 multi-dwelling lots at cutoff 2023-01-01. Their features use
  events up to the lake's end.
- The rows are an overridable `skrub.var("lots")`. It is not a scored candidate.
- Output: `submission.parquet` at the workspace root, with columns `bbl` (string) and
  `score` (float).
- Validation:
  - 171,587 rows in `test_entities` order; the set matches exactly.
  - No duplicates, all BBLs have 10 digits, no NaN.
  - Mean score is 0.134, in line with the 2022 base rate of 0.135.
- Expected AP is about 0.73–0.74. The CV fold for the latest year (2022) scored 0.741.
