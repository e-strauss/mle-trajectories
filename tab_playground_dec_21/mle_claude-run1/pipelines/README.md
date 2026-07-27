# Forest Cover Type (TPS Dec 2021) — pipeline exploration

Predict `Cover_Type` (7 forest cover classes) from 54 numeric features over a
**4,000,000-row** synthetic table. No test set / submission — the deliverable is
a cross-validated score per candidate pipeline.

**Headline result: a torch MLP reaches 0.96184 accuracy** (pipeline_24), beating
the best gradient-boosted tree (0.95646) by +0.005 and in a fraction of the
wall-clock. On this dataset, at 4M rows, the neural net wins.

## Metric & CV (locked)

- **Metric: `accuracy`.** Standard for this balanced-goal multiclass task and the
  original TPS-Dec-2021 metric. Majority-class floor (always predict class 2) is
  **0.5655** — the number every model must beat.
- **CV: `KFold(n_splits=3, shuffle=True, random_state=42)`** — the task specifies
  3-fold. *Not* `StratifiedKFold`: class 5 has a **single row** in all 4M
  (class 4 has 377), so stratified 3-fold is impossible. Plain shuffled KFold is
  used; the 1-row class is simply unlearnable and negligible for accuracy.
- The three folds are very stable and with 4M rows the noise floor on a 3-fold
  accuracy estimate is tiny (**~1e-4**), so the ~0.001–0.004 steps between top
  pipelines are real signal, not noise.

## Data notes (see `data_exploration_{1,2}.py`)

- 56 int columns: `Id`, 10 continuous, 44 binary indicators (4 wilderness + 40
  soil), `Cover_Type`. No missing values. `Id` is dropped everywhere.
- **Severe class imbalance**: class 2 = 56.6%, class 1 = 36.7%, class 3 = 4.9%,
  class 7 = 1.6%, class 6 = 0.29%, class 4 = 377 rows, class 5 = 1 row.
- The soil/wilderness indicators are **multi-hot, not one-hot** (soil active-count
  ranges 0–5, wilderness 0–2), so collapsing them to a single categorical code
  would lose information — the binaries are kept as-is, and their *counts* are
  added as features instead.
- `Soil_Type7` and `Soil_Type15` are all-zero (dead columns; harmless).

## Leaderboard (accuracy, 3-fold CV on full 4M rows)

Runs marked *(grid)* are single explorative `choose_from` runs; the score is the
**best** cell and the full grid lives in `results.json → extra.grid` (detailed
below). "Best variant" names that winning cell.

| Pipeline | Accuracy | Time | Model | What it is |
|---|---|---|---|---|
| **pipeline_24** | **0.96184** | 3665s | MLP | **Best. TUNE MLP (grid):** hidden{256,512}×layers{2,3}, 35 ep — best **512×3** |
| pipeline_23 | 0.96041 | 362s | MLP | torch MLP (skorch), 2×256, 15 ep, on champion features |
| pipeline_16 | 0.95646 | 2670s | LGBM | best GBDT: champion features (base14) at big_model (700/511/lr0.035) |
| pipeline_08 | 0.95589 | 2595s | LGBM | big_model (700/511/lr0.035) on pipeline_07 feature set |
| pipeline_21 | 0.95283 | 891s | LGBM | *(grid)* **soil ELU zones** — best variant `+elu_both` = **champion features** |
| pipeline_22 | 0.95283 | 708s | LGBM | *(grid)* target encoding — best variant is control (`champ`); TE **regressed** |
| pipeline_20 | 0.95249 | 1108s | LGBM | *(grid)* 400 trees/127 leaves, lr{0.05,0.075} — best lr=0.05 |
| pipeline_14 | 0.95192 | 276s | LGBM | **MERGE:** base_geo + soil descriptors + distance ratios (fast) = "base14" |
| pipeline_17 | 0.95192 | 1362s | LGBM | *(grid)* fast-config tune: num_leaves×min_child — nothing beat the default |
| pipeline_18 | 0.95192 | 884s | LGBM | *(grid)* ablation: soil_extra/ratios — best is base14 (soil_extra no help) |
| pipeline_07 | 0.95136 | 775s | LGBM | richer features (soil/wild counts, hillshade spread, \|VDH\|) @ 400/255 |
| pipeline_12 | 0.95077 | 252s | LGBM | + soil-descriptor features (stoniness, cryic, rock) |
| pipeline_15 | 0.95065 | 349s | LGBM | base14 + soil×geo crosses — **regressed** |
| pipeline_19 | 0.94909 | 831s | LGBM | medium config (400/lr0.05/**255 leaves**) — regressed (wrong leaf/budget) |
| pipeline_11 | 0.94907 | 248s | LGBM | + distance ratios & aggregates |
| pipeline_09 | 0.94791 | 228s | LGBM | fast-model **anchor**: base_geo only (200/127/lr0.1) |
| pipeline_06 | 0.94782 | 799s | LGBM | higher capacity: 400 trees, lr=0.05, 255 leaves |
| pipeline_10 | 0.94771 | 271s | LGBM | + multiplicative interactions — flat |
| pipeline_13 | 0.94495 | 251s | LGBM | + hillshade physics — regressed |
| pipeline_05 | 0.93554 | 69s | LGBM | + engineered geometric features |
| pipeline_03 | 0.92675 | 55s | LGBM | LightGBM, default params |
| pipeline_02 | 0.92262 | 252s | HGB | HistGradientBoosting, default params |
| pipeline_01 | 0.56552 | 6s | Dummy | majority-class floor |

**Lineage**
- GBDT capacity track: 01 → 02 → 03 → 05 → 06 → 07 → 08
- Feature track (fixed fast model): 09 anchor → {10,11,12,13} one theme each →
  **14** merges winners → 15 crosses (regressed); 18 soil_extra (no); **21** ELU
  (→ *champion features*); 22 target-encoding (no)
- Fast-config tuning on base14: 17, 19, 20 (none beat the fast default meaningfully)
- GBDT capacity on champion features: 14 → 16
- **Neural net (champion features): 21 → 23 → 24 (BEST)**

## Phase 1 — Baselines & the GBDT capacity track (01–08)

1. **Any gradient-boosted trees vs the floor** — the single biggest jump
   (0.566 → 0.923). Elevation + wilderness/soil separate the cover types well.
2. **LightGBM > sklearn HGB** — better *and* ~4× faster at defaults. LightGBM was
   the workhorse for every tree model after.
3. **Geometric feature engineering (+0.009, pipeline_05)** — Euclidean distance to
   hydrology, elevation adjusted by vertical hydrology distance, pairwise sums/diffs
   of the three distance features, aspect as sin/cos, mean hillshade.
4. **Model capacity was the strongest tree lever (+0.012 alone, pipeline_06)** —
   4M rows support far larger models than the defaults; more trees + deeper leaves
   + lower lr kept paying off through pipeline_08 (0.95589).

## Phase 2 — Feature exploration at fixed `fast_model` (09–15, 18, 21, 22)

To attribute score changes to *features* not capacity, these hold the model fixed
at `fast_model` (200 trees, lr=0.1, 127 leaves, ~4 min/run) and vary only features.
**09** re-establishes `base_geo` (= pipeline_07's features) as a same-config anchor
(0.94791). Ablations use the `choose_from` fused-run pattern (many variants, one
scored run — best to the board, full grid in `extra.grid`).

| Theme (vs anchor 09 = 0.94791) | Δ | Verdict |
|---|---|---|
| soil descriptors (stoniness, cryic, rock) — p12 | **+0.00286** | best single theme |
| distance ratios & aggregates — p11 | +0.00116 | helps |
| multiplicative interactions — p10 | −0.00020 | flat (drop) |
| hillshade physics — p13 | −0.00296 | hurts (drop) |
| **MERGE soil + ratios — p14 ("base14")** | **+0.00401** | **winners stack cleanly** |
| base14 + soil×geo crosses — p15 | −0.0013 vs 14 | crosses regressed |
| base14 + extra soil flags (warm/aquic/till) — p18 | −0.0013 vs 14 | no help |
| **base14 + soil ELU zones — p21 ("champion")** | **+0.00091 vs 14** | **helps** |
| champion + target encoding — p22 | −0.0070 vs champ | regressed hard |

Findings:
- **Positive themes stack additively** (14 = anchor + soil + ratios ≈ observed).
- **Explicit multiplicative crosses don't help a GBDT** — interactions (10) and
  soil×geo crosses (15) were flat-to-negative; the trees already split on the
  constituent columns, so products just add redundant, noisier features.
- **Soil ELU zones (21) are the one extra feature win.** Each soil type carries a
  USFS ELU code: 1st digit = climatic zone (2 lower-montane .. 8 alpine), 2nd =
  geologic zone. Interesting twist — **climatic zone alone regressed** (redundant
  with elevation + `cryic`), but **climatic + geologic together helped** (+0.0009):
  the geology is the axis that makes the climate band informative. Use both or
  neither. This defines the **champion feature set** (base14 + both ELU zones).
- **Target encoding hurt (−0.007).** Argmax-collapsing the multi-hot blocks is
  lossy, TargetEncoder's internal CV chokes on the 1-row class 5, and the soil→cover
  signal is already captured by descriptors + ELU zones.

## Phase 3 — Fast-config tuning (17, 19, 20): the tree config was already good

- **17** (num_leaves{127,255} × min_child_samples{20,100}): the *default* fast
  config (127 leaves, mcs 20) won; more leaves or more leaf-regularization both
  hurt at lr=0.1/200 trees.
- **19** medium config (400/lr0.05/**255 leaves**) *regressed* to 0.94909 —
  `num_leaves` must scale **with** the tree budget, it's not an independent lever.
  255 leaves is a bad middle ground; 511 leaves only pays with big_model's tree count.
- **20** the corrected trade (400 trees / **127** leaves / lr0.05) edged the fast
  default: 0.95249 (+0.0006), lr=0.05 > lr=0.075. Small, at ~4× cost.

Conclusion: the LightGBM fast config is near its budget optimum; tree accuracy is
capacity-bound, and capacity is expensive.

## Phase 4 — GBDT capacity on the champion features (16)

Best tree result: **pipeline_16 = 0.95646** — champion feature set at big_model
(700/511/lr0.035). Note the feature edge **shrinks with capacity**: the base14
features bought +0.0040 at fast (09→14) but only +0.00057 at big (08→16) — a
high-capacity GBDT rediscovers most of the hand-built structure on its own, so the
features mainly buy *efficiency* rather than a big accuracy gain at high capacity.

## Phase 5 — Neural net breakthrough (23, 24) ⭐

A torch MLP (wrapped as an sklearn estimator via **skorch**) on the champion
features **beats every tree**, cheaply:

- **pipeline_23** — modest first net (2×256, dropout 0.2, Adam 1e-3, 15 epochs,
  MPS): **0.96041** in **362s**. Already past the best GBDT (0.95646) and ~7× faster
  than the big_model run.
- **pipeline_24** — capacity grid (hidden{256,512}×layers{2,3}) at 35 epochs:
  **0.96184** (best cell 512×3). All four cells beat 0.96041.

What the grid says:
- **Epochs were the real lever** — even 256×2 at 35 epochs jumped +0.0009 over the
  15-epoch run; the first MLP was simply under-trained.
- **Capacity helps only marginally** — the whole width/depth spread is 0.0005
  (0.96132 → 0.96184); depth ≥ width. We're near this architecture's plateau on
  these features.

Why the MLP wins here (contra the prior "covertype is tree-friendly" expectation):
at 4M rows the continuous geometric features have smooth structure the net exploits,
and the data volume amply supports a neural model. The tree "ceiling" was a LightGBM
ceiling, not a task ceiling.

### Infrastructure notes for the NN (see `nn.py`)

- **skorch wrapper (`TorchMLP`)**: StandardScaler (NNs need scaling; trees didn't)
  → sanitize/clip z-scores → float32 → skorch net. **LabelEncoder** maps the raw
  1–7 labels to 0..C-1 for `CrossEntropyLoss` and back on predict, so the MLP scores
  in the **raw label domain** exactly like LightGBM (this is the wrapper XGBoost
  would also have needed — see below). `train_split=None` avoids skorch's internal
  stratified split (which would choke on the 1-row class 5).
- **OpenMP crash (macOS)**: LightGBM (`libomp`) and torch (its own OpenMP) collide
  on the second load and **abort the process with no Python traceback**. Fix: launch
  torch pipelines with **`KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`** (must be set
  in the environment *before* the process imports either lib — setting it in code is
  too late). Caveat: `OMP_NUM_THREADS=1` single-threads LightGBM, so an in-process
  LGBM+MLP blend needs a different approach (separate processes / out-of-fold preds).
- **Robust scaling**: full-4M folds occasionally produce a float32-overflowing
  z-score; `TorchMLP` sanitizes non-finite values and clips |z|>30.

## What didn't work / was abandoned

- **XGBoost (`pipeline_04.py`, not scored):** its sklearn wrapper rejects the
  non-contiguous `1..7` labels (a fold missing a rare class raises *"Invalid classes
  inferred…"*). Needs the same LabelEncoder-inverse wrapper `TorchMLP` uses; skipped
  at the time since LightGBM dominated. (In hindsight that wrapper is cheap.)
- **Collapsing soil/wilderness to a single categorical code** — lossy (multi-hot).
- **`HistGradientBoosting` default `early_stopping='auto'`** — its internal
  stratified split fails on the 1-row class 5; use `early_stopping=False`.
- **Feature ideas that regressed:** multiplicative interactions, soil×geo crosses,
  extra soil flags (warm/aquic/till/bouldery), target encoding, and (for GBDT) 255
  leaves at a small tree budget.

## Takeaways & current standing

- **Best pipeline: pipeline_24, MLP 512×3 @ 35 epochs = 0.96184** on the champion
  feature set (base_geo + soil descriptors + distance ratios + soil ELU zones).
- **The MLP is both more accurate and far cheaper than the GBDT here** — the tree
  line plateaued near 0.956 for a lot of compute; the MLP cleared 0.96 on its first
  honest run in 6 minutes.
- **Champion features = base14 + ELU zones.** Domain knowledge paid off (soil
  descriptors, ELU climatic+geologic zones); generic math crosses and target
  encoding did not.
- **Feature gains compress at high model capacity** — proven twice (GBDT fast→big,
  and features matter less to the high-capacity MLP than epochs/architecture).

### Not done yet (stopped here for the day) — natural next steps

1. **Blend LGBM + MLP** — decorrelated tree + net, soft-vote; the classic route to
   push past 0.962. Needs the OMP/threads workaround (separate processes or
   out-of-fold predictions), noted above.
2. **MLP convergence check** — 512×3 at ~60 epochs (~20 min) to confirm the net has
   plateaued and lock the best standalone config; possibly lr schedule / more dropout.
3. **Promote the winner** — pipeline_24's best cell (512×3) to its own single-scoring
   file for a clean lineage node.
4. **MLP on raw vs champion features** — untested; the net may not need the
   hand-built features, which would simplify the pipeline.
