# Moving work out of `TrackerRankNetEstimator`, one step at a time

Each version is a complete, runnable pipeline. Consecutive versions differ by
exactly one block moved out of the estimator.

The goal is **node granularity**, not fewer lines: stratum reports per-operator
statistics, and anything inside one estimator is a single opaque entry. The rule
the moves follow is

> stateful computation goes in a transformer; everything stateless moves out of
> the transformer/estimator and into recorded DataOps.

`diff -u versions/v1_graph_degrees.py versions/v2_direct_links.py` shows one move.

| version | what moved out of the estimator | estimator |
|---|---|---|
| `v0_udf_baseline.py`   | — (baseline: first version with no UDF left in the plan) | 434 lines |
| `v1_graph_degrees.py`  | 10 link-graph degree features (`deg_out`/`deg_in`/`tracker_links` + 7 derived) | 417 |
| `v2_direct_links.py`   | `direct_links` — 355 columns, the first of the five prior channels | 406 |
| `v3_cooccurrence.py`   | the 355x355 tracker co-occurrence matrix | 393 |
| `v4_global_priors.py`  | empirical tracker rates + the output layer's initial bias logits | 394 |
| `v5_priors_transformer.py` | Bayesian root/TLD priors into their own `BayesianTrackerPriors` transformer (124 lines), with two of its three scalar summaries hoisted into the plan | 300 |
| `v6_neighbour_transformers.py` | neighbour adoption into two `NeighbourTrackerAdoption` transformers (80 lines, one per direction); Adamic-Adar edge weights and all six graph scalars became recorded ops | 204 |
| `v7_design_matrix.py` | the design matrix itself -- a recorded column selection plus the nan/inf scrub | 186 |

`0029_e141d0f7f68a492faa18ce40dd7b5286.py` (the live file) is identical to the
newest version.

## What each move needed

* **v1** — pure graph aggregates (`groupby().size()` + merge). Label-free, so
  ordinary recorded features. `_prepare_graph` keeps only the TOTAL degree, which
  is still needed as the Adamic-Adar edge weight.
* **v2** — the same long-to-wide `unstack` as the target matrix, over link-graph
  edges whose destination is a tracker. This also removed the `trackers`
  fit_kwarg: nothing else in the estimator used that table.
* **v3** — `y.T.dot(y)` and friends. Consumed only while fitting, so it travels
  as a fit_kwarg derived from `y` and needs no `freeze_after_fit`.
* **v4** — `y.to_numpy(np.float32).mean(axis=0, dtype=np.float64)`, plus the
  `log(p / (1 - p))` that becomes the output layer's initial bias. Also fit-time
  only, so again a fit_kwarg rather than a frozen node.
* **v5** — a transformer, not `freeze_after_fit`. The original's
  `is_training_rows` flag is exactly the `fit_transform`/`transform` split:
  training rows get the leave-one-out posterior, scored rows the plain one.
  `root_match_count_log` and `root_prior_max` are stateless row arithmetic over
  the block it emits, so they are recorded ops in the plan.

  `root_prior_entropy` had to stay inside the transformer. `np.log(frame + eps)`
  promotes float32 to float64 in pandas while the original accumulated in
  float32, so the recorded version drifts ~1e-6. Worth knowing that the SCORE
  check did not catch this -- Recall@10 was unchanged -- only the design-matrix
  comparison did.

## Verification

Every version is checked two ways, and all four pass identically:

* the per-fold design matrix is **bit-identical** to the one the original
  pipeline builds (train 16000x1860 and validation 4000x1860, same row
  membership and order, same targets);
* the end-to-end score on the sample is `0.7805021056023915`.

The score check is not redundant: v3 moves a matrix that only initialises the
relational layer's weights, which the design-matrix check cannot see.

Reproduce with the 20k-domain sample, which now lives in
`ttt-task/sample/input/` (`ttt-task/input/` holds the full data). `TTT_INPUT` is
read relative to the working directory, so run from the sample's run-root:

```bash
cd ttt-task/sample
TTT_INPUT=./input python ../mlevolve_run_2/skrubify_manual/versions/v2_direct_links.py
TTT_INPUT=./input TTT_EPOCHS=4 python ../mlevolve_run_2/skrubify_manual/versions/v3_cooccurrence.py   # fast loop
```

Rebuild the sample itself with `python ttt-task/make_input_sample.py`.

## Note on provenance

v0-v2 were reconstructed by reversing the applied patches rather than snapshotted
at the time; they were then verified by running them, which is what the table
above reports. v3 onwards are snapshots.

## What is still in the estimator

Everything left is label- or fold-dependent and would need
`freeze_after_fit` to move:

* neighbour tracker adoption (`out_nbr` / `in_nbr` blocks + 6 degree columns) --
  counts over neighbours that are TRAINING domains;
Nothing, apart from the torch model and its training loop, which stays by
design. `TrackerRankNetEstimator` is down to `__init__`, the training loop, a
two-line `predict`, and a three-line `to_numpy`.

Three estimators remain, each doing one stateful thing:

| estimator | why it has to be one |
|---|---|
| `BayesianTrackerPriors` | count tables over the fold's training labels |
| `NeighbourTrackerAdoption` (x2) | the neighbours that count are training domains, and the signal is their labels |
| `TrackerRankNetEstimator` | the model |

## Two things that could not be moved out

* `root_prior_entropy` (v5). `np.log(frame + eps)` promotes float32 to float64 in
  pandas while the original accumulated in float32, so every recorded
  formulation drifted 2e-6 to 1e-5. It stays inside the priors transformer.
* The `nan_to_num` scrub is recorded (v7) but costs about 1.3x the memory of the
  original's in-place numpy call -- roughly +100 GB at full scale, once per fold.
  Reverting it to the estimator is a one-line change if a full run gets tight.
