# mle-trajectories

Development trajectories of ML engineering agents (MLE-STAR, mlevolve, Claude
Code), collected per dataset, plus tooling to compare the pipelines they produce.

The experiment: take every script an agent executed during a run, translate it
into a skrub DataOps plan, extract the stratum operator DAG, and diff each step
against its parent. That shows what the agent actually changed from iteration to
iteration, and what it kept.

## Layout

```
<dataset>/
    get_data.sh       fetch the data into input/ (Kaggle datasets only)
    DATA.md           where the data comes from, when a script cannot fetch it
    make_sample.py    build sample/input/ from input/ (per dataset)
    input/            the full data, gitignored
    sample/input/     a small self-consistent sample, gitignored
    <agent>_run_<n>/
        pipelines/        the agent's original scripts, untouched
        final_state.json  its state dump / journal (format depends on the agent)
        skrubify*/        the skrub DataOps rewrite of each step
```

## Data

Pipelines read `./input/...`, and both `skrubify --run-in` and
`pipeline_analyzer.runtime --run-in` take the directory *holding* that `input/`.
So `--run-in <dataset>` runs against full data and `--run-in <dataset>/sample`
against the sample, with no edit to any pipeline.

Four datasets are Kaggle competitions and have a `get_data.sh` (it formalises
what `tools/getcomp.sh` did by hand, minus the assumption that every file is a
flat CSV):

| dataset | competition |
| --- | --- |
| `aptos2019-blindness-detection` | `aptos2019-blindness-detection` |
| `playground-series-s6e7` | `playground-series-s6e7` |
| `tab_playground_dec_21` | `tabular-playground-series-dec-2021` |
| `nyc_taxi_fare` | `new-york-city-taxi-fare-prediction` |

The scripts are idempotent (re-running is a no-op unless passed `--force`) and
need the competition rules accepted in the browser once, or the download returns
403 rather than the data.

The other four are not on Kaggle and carry a `DATA.md` instead, recording the
upstream, the derivation needed to reach the `input/` layout, and the local path
on this machine where one exists: `house_price` (HM Land Registry price-paid plus
an out-of-time split), `cover_type_multi_table` (a Zenodo autofeat tar rebuilt
into 17 tables), `ttt-task` (TrackTheTrackers), `beaver_enroll` (the BEAVER
benchmark, not present here).

Full data is the wrong size for skrubify's repair loop, which runs the candidate
*and* the original once per round. The `dataset-sample` skill writes a
per-dataset `make_sample.py` producing `sample/input/`; `ttt-task/make_input_sample.py`
is the worked example. The data is gitignored, so `sample/sample_manifest.json`
— sizes, knobs, seed, row counts before and after — is the record of what any
sampled number was measured on.

## Corpus

8 datasets, 19 agent runs, **1039 pipelines** in total. One pipeline = one script
the agent actually executed. Skrubified rewrites (`skrubify*/`) are the same
pipeline expressed as a skrub DataOps plan, so they are not counted again.

| dataset | run | agent | init | train / improve | ablation | ensemble | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aptos2019-blindness-detection | mle-star-run-1 | MLE-STAR | 2 | 35 | 7 | – | **44** |
| aptos2019-blindness-detection | mle-star-run-2 | MLE-STAR | 2 | 33 | 7 | – | **42** |
| beaver_enroll | mle_star_flash_run_1 | MLE-STAR | 5 | 114 | 36 | 3 | **158** |
| beaver_enroll | mle_star_flash_run_2 | MLE-STAR | 5 | 114 | 36 | 3 | **158** |
| beaver_enroll | mle_star_flash_run_3 | MLE-STAR | 5 | 114 | 36 | 3 | **158** |
| beaver_enroll | mle_star_flash_run_4 | MLE-STAR | 5 | 114 | 36 | 3 | **158** |
| cover_type_multi_table | mle_star_run_1 | MLE-STAR | 2 | 9 | 2 | – | **13** |
| cover_type_multi_table | mle_star_run_2 | MLE-STAR | 2 | 28 | 5 | 3 | **38** |
| house_price | mlevolve_run_1 | mlevolve | – | 13 | – | – | **13** |
| house_price | mlevolve_run_2 | mlevolve | – | 46 | – | – | **46** |
| nyc_taxi_fare | mlevolve_run_1 | mlevolve | – | 21 | – | – | **21** |
| nyc_taxi_fare | mlevolve_run_2 | mlevolve | – | 17 | – | – | **17** |
| nyc_taxi_fare | mlevolve_run_3 | mlevolve | – | 18 | – | – | **18** |
| playground-series-s6e7 | mlevolve_run_1 | mlevolve | – | 24 | – | – | **24** |
| tab_playground_dec_21 | mle_star | MLE-STAR | 2 | 53 | 10 | 3 | **68** |
| tab_playground_dec_21 | mle_claude_run_1 | Claude Code | – | 24 | – | – | **24** |
| ttt-task | mlevolve_run_1 | mlevolve | – | 12 | – | – | **12** |
| ttt-task | mlevolve_run_2 | mlevolve | – | 19 | – | – | **19** |
| ttt-task | mlevolve_run_3 | mlevolve | – | 8 | – | – | **8** |
| | | | **30** | **816** | **175** | **18** | **1039** |

Runs per dataset: beaver_enroll 4, aptos2019-blindness-detection 2,
cover_type_multi_table 2, house_price 2, nyc_taxi_fare 3,
playground-series-s6e7 1,
tab_playground_dec_21 2 (one MLE-STAR, one Claude Code), ttt-task 3.

Notes on the counts:

- The ablation column is MLE-STAR's ablation scripts — runnable variants of the
  current solution, but probes rather than candidate solutions. Drop them and
  the corpus is 864 pipelines.
- mlevolve keeps a script only for nodes that ran; its journals hold 31 nodes
  against 21 saved scripts (nyc_taxi_fare), 31 against 24
  (playground-series-s6e7), 16 against 12 (ttt-task run 1), 35 against 19
  (ttt-task run 2), 11 against 8 (ttt-task run 3), 16 against 13
  (house_price run 1), 51 against 46 (house_price run 2), and 21 against 17
  and 21 against 18 (nyc_taxi_fare runs 2 and 3).
- `nyc_taxi_fare` run 1 saw an anonymised description of the task
  (`od_cost_regression`, with `record_id`, `cost`, `origin_x/y`); runs 2 and 3
  saw the un-anonymised one naming New York City taxi fares. Same data and
  metric, so the scores compare, but in runs 2-3 the agent reconstructs domain
  knowledge that is in no data file - the September 2012 fare hike, the JFK flat
  rate, airport and river-crossing flags, a rotated Manhattan street grid.
  Runs 2 and 3 differ only in how the task description frames an optional
  library (TabFM): "optional, second-best to gradient boosting" in run 2,
  a peer of the GBDT libraries in run 3. Neither run used it in any node.
- `mle_claude_run_1` writes skrub DataOps plans directly, so it has no
  `skrubify*/` folder. Its `common.py`, `features.py`, `nn.py` (shared modules)
  and `data_exploration_*.py` are not pipelines and are excluded.
- `tab_playground_dec_21/mle_star` is one run; `skrubify_gemini` and
  `skrubify_openai` are two translations of it by different skrubify providers.
- Both aptos runs ended in agent error, and the beaver_enroll runs were never
  validated against real data (see `mle_star_flash_run_3/result.txt`) — they are
  kept as trajectories, not as working solutions.

## Tools

- [`tools/skrubify`](tools/skrubify) — convert a pandas/sklearn script into a
  skrub DataOps pipeline (LLM does the translation, a validator gates it).
- [`tools/pipeline_analyzer`](tools/pipeline_analyzer) — build the lineage of a
  run and emit an HTML report: tree, per-step operator DAG, per-step diff.
  `pipeline_analyzer.runtime` executes the pipelines under stratum and caches
  per-pipeline runtime stats in a json the report picks up.
- [`tools/trajectory.py`](tools/trajectory.py) — tabular overview of one run
  (steps, scores, timings, parents).
- [`tools/getcomp.sh`](tools/getcomp.sh) — one-off Kaggle competition fetch into
  `<dir>/input/`; the per-dataset `get_data.sh` scripts are the wired-up version.
- [`.claude/skills/dataset-sample`](.claude/skills/dataset-sample) — skill: given a
  dataset folder, write its `make_sample.py` and build `sample/input/` so a run
  in the repair loop takes about a minute. (Claude Code only discovers skills
  under `.claude/skills/`, which is why it does not live in `tools/`.)
