# mle-claude-run1 — TrackTheTrackers

Agent: **Claude Opus 5** driving the `mle-claude` skill harness (`ml-workspace` /
`ml-score` / `ml-submit`), 2026-09-21 → 2026-09-22.
15 scored pipelines, 9 exploration rounds, one submission.

**Best CV Recall@10 = 0.90004** (pipeline_11), against a constant-popularity
baseline of 0.75477 and an attainable ceiling of 0.9992.

| | |
|---|---|
| metric | `plan:recall_at_10` — declared in the plan, locked by the harness |
| CV | `KFold(3, shuffle=True, random_state=42)` over 505,548 domains |
| rows | tracked domains with `domain_id % 37 == 11`, plus the 50,000 target domains |
| model | 5-seed ensemble of a 1024×512 MLP, softmax-CE on the row-normalised target |
| submission | `submission.tsv`, 500,000 rows (50,000 domains × 10) |

## Normalisation applied for this collection

The original workspace materialised its features **once into parquet** (a
"frozen feature store") and every pipeline read them back. The other
trajectories here do not cache features, so the computation has been **inlined**:

| | original workspace | here |
|---|---|---|
| feature computation | `data_exploration_4/_7/_9.py` → `features/*.parquet` | `pipelines/features.py`, recorded as plan nodes |
| what a pipeline reads | the parquet store | `input/` |
| `load_xy(blocks=...)` | reads parquet | records the builders |
| pipeline files | — | **unchanged** (the signature was kept) |

`pipelines/common.py::build_blocks` wires one node per block — context → link
degrees → seed edges → the 13 blocks → merge → marks — and builds only the
blocks a given pipeline asks for. The heavy lifting inside each node stays
numpy/polars: the inputs are a 623M-edge link graph and a 36.7M-edge tracking
graph, so recording it as pandas row operations is not an option at that size.
What the plan gets is *node granularity*, one recorded step per meaningful
operation, rather than a single opaque blob.

**The numbers did not move.** `verify_inline_features.py` rebuilds all 13 blocks
from `input/` and compares them against the original store, column by column,
for both halves. That is what licenses carrying `pipelines/results.json` over
unchanged:

```bash
python verify_inline_features.py --store <original workspace>/features
```

Result: **26/26 comparisons identical** (13 blocks x train/target, `labels`
train-only), across both halves and every column.

One difference had to be fixed to get there, and it is worth knowing about if
you touch `features.py`: `meta2.hub_weight_mass` initially drifted ~1e-6 because
pandas returns `DataFrame.to_numpy()` **F-contiguous**, and summing that along
axis 1 accumulates float32 in a different pairwise order than the C-contiguous
array the original summed. `np.ascontiguousarray` before the sum restores
bit-equality. Numerically the drift was irrelevant -- one scalar column that then
passes through `log1p` and standardisation -- but `results.json` was produced
against the original store, so "probably wouldn't matter" is not the claim this
should rest on.

End-to-end smoke test, harness -> `load_xy` -> recorded block nodes -> marks ->
CV -> scorer:

```
pipeline_01: best mean_test_score=0.75477 std=0.00101
pipeline_01: fold scores = [0.75357, 0.75604, 0.75469]
```

identical to the entry recorded during the original run.

### Cost of inlining

One build is a full scan of the link graph plus per-block work — about 3–6
minutes depending on which blocks are requested, against ~20s for the old
parquet read. It happens **once per run**, not once per fold: the blocks are
built before `mark_as_X`. `load_xy` builds under
`skrub.config_context(eager_data_ops=False)`, because the eager preview would
otherwise run the whole build at plan-construction time and again at fit time.

### Why the blocks sit before `mark_as_X` (and what that costs)

This is deliberate, not a shortcut. A domain's features are a function of the
**given** data — the link graph and the tracking graph over all 18.7M tracked
domains — and its own identity, never of its own label, which is always
excluded (self-loops dropped, `tld_pop` leave-one-out corrected,
`trk_cooc`/`tok_pop` estimated only on domains outside the modelled sample). At
real prediction time the 50,000 target domains are unseen but every *other*
domain's trackers are known, so recomputing the blocks per fold from
training-fold rows only would model a harder problem than the real one and make
the CV pessimistic.

The honest consequence: **cross-validation covers the model and the encoding,
but not feature construction.** A bug in `features.py` is invisible to the
score — it just makes every fold agree on the same wrong answer. That is exactly
how the `nbr_2h` self-leak below reached the leaderboard.
`exploration/data_exploration_8.py` is the guard for that layer.

## Results

| pipeline | Recall@10 | std | parent | what changed |
|---|---|---|---|---|
| 01 | 0.75477 | 0.00101 | — | constant global tracker popularity, no features |
| 02 | 0.84639 | 0.00105 | 01 | blend of link-neighbour / TLD / popularity blocks |
| 03 | 0.86894 | 0.00107 | 02 | multi-output Ridge |
| 04 | 0.87081 | 0.00065 | 03 | + degrees, hostname shape, one-hot TLD |
| 05 | 0.88953 | 0.00035 | 04 | GPU MLP + metric-aligned softmax-CE |
| 06 | 0.88954 | 0.00031 | 05 | hyperparameter sweep — saturated |
| 07 | 0.88998 | 0.00033 | 06 | + hub / reciprocal / 2-hop blocks (after leak fix) |
| 08 | 0.88673 | 0.00027 | 07 | + hostname char n-grams — regression |
| 09 | 0.89143 | 0.00028 | 08 | leave-one-block-out ablation |
| 10 | 0.89517 | 0.00034 | 09 | ablation-pruned features + 5-seed ensemble |
| **11** | **0.90004** | 0.00038 | 10 | **+ `direct`: hyperlinks to tracker domains** |
| 12 | 0.89540 | 0.00037 | 10 | + `nbr_frac` |
| 13 | 0.89533 | 0.00038 | 10 | + `trk_cooc` |
| 14 | 0.89364 | 0.00054 | 10 | + `tok_pop` — regression |
| 15 | 0.90019 | 0.00041 | 11 | fused choice over combinations — all within noise |

Three findings carried most of the run:

1. **The task is pure cold start.** 0/50,000 target domains appear in the
   tracking graph, so nothing about a target domain's own trackers is ever
   observable. 90.8% do have a *tracked link neighbour*, which is what makes it
   tractable at all.
2. **Aligning the objective with the metric** was the largest modelling win
   (+0.019). Recall@10 of a set *S* is `Σ_{t∈S} y_t / n_true`, so the optimal ten
   are the largest `E[y_t/n_true | x]`, not the largest `P(y_t=1 | x)` — a
   softmax over the 355 trackers fitted against the row-normalised label vector
   estimates exactly that.
3. **`direct` was the largest feature win** (+0.0049): a hyperlink between a
   domain and a tracker's own hostname coincides with a true tracking edge 50.4%
   of the time against a 0.55% base rate.

Full analysis, the leakage post-mortem, the noise-floor discussion and a
step-by-step chronology are in **`pipelines/README.md`**.

## The leak, kept in the record

pipeline_07 first scored **0.91276 with a fold std of 0.00005**. The tell was the
*uniformity*, not the size: real evidence moves the mean, it does not make three
data folds agree to five decimals. `nbr_2h` walked `seed → mid → mid's tracked
neighbours`, and since `mid` was discovered *via* the seed, the seed came back as
its own neighbour, writing every training row's own label vector into its own
feature (own-tracker coverage measured exactly 1.0000). Target domains cannot
leak that way, so the feature was rich in training and empty at submission time.
Fixed (`keep2` in `features.block_nbr_2h`), re-scored honestly at 0.88998 — the
whole +0.023 was the leak.

## Layout

```
input/                        symlink to the task data
pipelines/
  features.py                 INLINED feature construction (the normalisation)
  common.py                   row sample, CV, recall_at_10 scorer, load_xy
  models.py                   PopularityRanker, BlendRanker, BlockTransform, TorchMLPRanker
  featureset.py               shared feature graph for pipelines 11-15
  pipeline_01..15.py          the scored candidates
  final_pipeline.py           refit-on-all + submission writer (never scored)
  results.json                the leaderboard (written only by the ml-score harness)
  README.md                   full analysis + chronology
exploration/
  data_exploration_1..3.py    data shape, cold-start diagnosis, sample representativeness
  data_exploration_4,7,9.py   the ORIGINAL store builders — kept for provenance;
                              their logic now lives in pipelines/features.py
  data_exploration_5.py       loss attribution
  data_exploration_6.py       offline GPU-MLP architecture/objective sweep
  data_exploration_8.py       leakage audit (the guard for the feature layer)
  common_frozen_store.py      the original parquet-reading common.py, for reference
verify_inline_features.py     inlined build == original store
```

## Running

```bash
# score one pipeline (needs the ml-score harness from the mle-claude repo)
TTT_INPUT=./input python <mle-claude>/.claude/skills/ml-score/scripts/score_pipeline.py \
    pipelines/pipeline_11.py

# refit on everything and write submission.tsv
TTT_INPUT=./input PYTHONPATH=pipelines python pipelines/final_pipeline.py
```

Needs `skrub>=0.9`, torch with CUDA, polars, pyarrow, scipy, scikit-learn.
The exploration scripts under `exploration/` reference the original workspace's
`features/` directory and are kept as a record of how the blocks were derived,
not as runnable entry points here.
