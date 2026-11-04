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
    dataset.toml      dataset facts, incl. the [sample] recipe
    make_sample.py    custom sampler, only where the [sample] recipe is not enough
    input/            the full data, gitignored
    sample_<size>/
        input/                a self-consistent sample, same file names, gitignored
        sample_manifest.json  how it was built (tracked)
    <agent>_run_<n>/
        pipelines/        the agent's original scripts, untouched
        final_state.json  its state dump / journal (format depends on the agent)
        skrubify*/        the skrub DataOps rewrite of each step
```

## Data

The new agent-visible GCS bucket is `gs://mle-trajectories-data`, with prefixes
matching the dataset folders. Four Kaggle datasets keep their original train/test splits; UK housing has a
versioned temporal test with private labels. UK housing, NYC taxi, S6E7,
December 2021 and APTOS are verified. TrackTheTrackers inputs are also verified;
its scorer and held-out labels are in the private bucket.
See [GCS setup, upload, and direct reads](infra/README.md).
NYC housing keeps its existing GCS lake; BEAVER is skipped.

Pipelines read `./input/...`, and both `skrubify --run-in` and
`pipeline_analyzer.runtime --run-in` take the directory *holding* that `input/`.
So `--run-in <dataset>` runs against full data and
`--run-in <dataset>/sample_<size>` against a sample, with no edit to any pipeline.

Six datasets are Kaggle competitions and have a `get_data.sh` (it formalises
what `tools/getcomp.sh` did by hand, minus the assumption that every file is a
flat CSV):

| dataset | competition |
| --- | --- |
| `aptos2019-blindness-detection` | `aptos2019-blindness-detection` |
| `playground-series-s6e7` | `playground-series-s6e7` |
| `tab_playground_dec_21` | `tabular-playground-series-dec-2021` |
| `nyc_taxi_fare` | `new-york-city-taxi-fare-prediction` |
| `spaceship-titanic` | `spaceship-titanic` |
| `titanic` | `titanic` |

`titanic`'s script then splits the competition's labeled `train.csv` (stratified
20%, seed 42) so the test set can be scored. The competition's own test file is
not what the run saw; `DATA.md` records the split.

The scripts are idempotent (re-running is a no-op unless passed `--force`) and
need the competition rules accepted in the browser once, or the download returns
403 rather than the data.

Three of the others are not on Kaggle and carry a `DATA.md` instead, recording
the upstream, the derivation needed to reach the `input/` layout, and the local
path on this machine where one exists: `house_price` (HM Land Registry
price-paid plus an out-of-time split), `ttt-task` (TrackTheTrackers),
`beaver_enroll` (the BEAVER benchmark, not present here). `nyc-housing-violation`
reads its lake straight from GCS, and `voice-intent` is built by its
`prepare.py` from AmazonScience/MASSIVE.

Full data is the wrong size for skrubify's repair loop, which runs the candidate
*and* the original once per round. `python -m dataset_sample <dataset> --size 100k`
builds `sample_100k/input/` from the `[sample]` recipe in `dataset.toml` (see
[`tools/dataset_sample`](tools/dataset_sample)); the `dataset-sample` skill says
how to choose a recipe, or write a `make_sample.py` where one is not enough. The
data is gitignored, so `sample_<size>/sample_manifest.json`
— sizes, knobs, seed, row counts before and after — is the record of what any
sampled number was measured on.

## Corpus

11 dataset folders, 52 agent runs, **1933 pipelines** in total (as of
2026-11-04). One pipeline = one script the agent actually executed. Skrubified
rewrites (`skrubify*/`) are the same pipeline expressed as a skrub DataOps plan,
so they are not counted again. `voice-intent` is prepared (data on GCS, task
description) but has no runs yet. The numbers below come from the website's
registry and code analysis (`/api/corpus`, `/api/runs/<dataset>/<run>`,
`/api/dashboard`), so re-query those rather than recount by hand.

### Datasets

| dataset | modality | task | metric | size (train / test) | runs | pipelines | held-out test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aptos2019-blindness-detection | image (retina PNGs + label CSV) | 5-grade ordinal classification | quadratic weighted kappa | 3,662 / 1,928 images | 2 | 86 | – |
| beaver_enroll | tabular, multi-table (warehouse dump, data discovery) | binary classification | F1 | dozens of tables; not on this machine | 4 | 626 | – |
| house_price | tabular, single table, temporal | regression, out-of-time split | RMSE on log10(price) | 22.1M / 375k rows | 2 | 59 | – (private labels exist) |
| nyc-housing-violation | tabular, multi-table (71-table data lake) | binary classification | average precision | lake on GCS | 4 | 155 | – |
| nyc_taxi_fare | tabular, single table, geo + time | regression | RMSE | 55.4M / 9,914 rows | 14 | 338 | Kaggle public LB |
| playground-series-s6e7 | tabular, single table (synthetic student health) | 3-class classification | balanced accuracy | 690k / 296k rows | 1 | 24 | – |
| spaceship-titanic | tabular, single table, small | binary classification | accuracy | 8,693 / 4,277 rows | 6 | 208 | Kaggle public LB |
| tab_playground_dec_21 | tabular, single table (synthetic forest cover) | 7-class classification | accuracy | 4M / 1M rows | 7 | 141 | Kaggle private LB |
| titanic | tabular, single table, small | binary classification | accuracy | 712 / 179 rows (split of train.csv) | 1 | 81 | local split (not scored) |
| ttt-task | multi-modal: web link graph + tabular metadata + hostnames | multi-label ranking (≤10 trackers per domain) | Recall@10 | 50k target domains; 623M link edges, 46M domains | 11 | 215 | scoring server |
| voice-intent | text (multilingual utterances, 52 locales) | 60-class intent classification | accuracy | 704k / 155k rows | 0 | – | scoring server |

### Agents

| agent | runs | pipelines | datasets |
| --- | --- | --- | --- |
| MLE-STAR | 8 | 861 | 4 |
| meta-mle | 22 | 491 | 4 |
| mlevolve | 15 | 453 | 6 |
| Claude Code | 5 | 102 | 3 |
| AIDE | 2 | 26 | 2 |

meta-mle runs are labelled by the search profile they used (greedy, fm-agent /
islands, mlevolve, aide): meta-mle's own version of that search strategy, not
the original agent's code.

### Runs

Best CV is the best validation score in the run's lineage, in the dataset's
metric (lower is better for RMSE). Test is the held-out score of the submitted
pipeline, from `test_scores.toml`. Models are the ones used in at least 15% of
the run's scripts, most frequent first. Counts in parentheses are derived
copies, not agent runs, and are not in the totals.

| dataset | run | agent | LLM | budget | pipelines | lineage nodes | best CV | test | main models |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aptos2019-blindness-detection | mle-star-run-1 | MLE-STAR | – | – | **44** | – | – | – | custom NN, timm CNN |
| aptos2019-blindness-detection | mle-star-run-2 | MLE-STAR | – | – | **42** | – | – | – | custom NN, timm CNN |
| beaver_enroll | mle_star_flash_run_1 | MLE-STAR | Gemini 2.5 Flash | – | **157** | 157 | 0.8506 | – | random forest |
| beaver_enroll | mle_star_flash_run_2 | MLE-STAR | Gemini 2.5 Flash | – | **158** | 158 | 1.0000 | – | random forest |
| beaver_enroll | mle_star_flash_run_3 | MLE-STAR | Gemini 2.5 Flash | – | **155** | 155 | 1.0000 | – | random forest, TabNet |
| beaver_enroll | mle_star_flash_run_4 | MLE-STAR | Gemini 2.5 Pro | – | **156** | 156 | 0.0000 | – | random forest, LightGBM, TabPFN |
| house_price | mlevolve_run_1 | mlevolve | Gemini 3.8 Flash | 15 steps / 12 h | **13** | 16 | 0.2203 | – | custom NN, LightGBM |
| house_price | mlevolve_run_2 | mlevolve | Gemini 3.8 Flash | 50 steps / 16 h | **46** | 51 | 0.2065 | – | LightGBM, custom NN, XGBoost |
| nyc-housing-violation | mle-claude-run1 | Claude Code | Claude | – | **32** | 32 | 0.7318 | – | HistGB, LightGBM |
| nyc-housing-violation | mle-claude-run1_v2 | Claude Code | Claude | – | (32) | 32 | 0.7318 | – | HistGB, LightGBM |
| nyc-housing-violation | mlevolve_run_0 | mlevolve | Gemini 3.8 Flash | 20 steps / 12 h | **18** | 21 | 0.7195 | – | custom NN, LightGBM, CatBoost |
| nyc-housing-violation | mlevolve_run_1 | mlevolve | Gemini 3.8 Flash | 20 steps / 12 h | **17** | 21 | 0.7447 | – | LightGBM, XGBoost, custom NN |
| nyc-housing-violation | mlevolve_run_2 | mlevolve | Gemini 3.8 Flash | 100 steps / 24 h | **88** | 101 | 0.7594 | – | LightGBM, XGBoost, custom NN |
| nyc_taxi_fare | meta_mle_fm_agent_gemini_run_1 | meta-mle (fm-agent / islands) | Gemini 3.8 Flash | 5 h | **39** | 39 | 3.9579 | 3.1700 | LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_fm_agent_gemini_run_2 | meta-mle (fm-agent / islands) | Gemini 3.8 Flash | 5 h | **36** | 36 | 3.9097 | 3.1442 | LightGBM, custom NN, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_1 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 5 h | **18** | 18 | 3.9101 | 3.1002 | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_2 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 5 h | **17** | 17 | 3.7184 | 3.0396 | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_3 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 8 h | **34** | 34 | 3.7033 | 3.0207 | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_4 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 8 h | **46** | 46 | 3.9189 | 3.0873 | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_4_upscaled | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 8 h | (46) | 46 | 3.9189 | – | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_4_upscaled_tuned | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 8 h | (46) | 46 | 3.9189 | – | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_5 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 8 h | **26** | 26 | 3.8947 | 3.0886 | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_mlevolve_gemini_run_6 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 8 h | **28** | 28 | 3.7086 | 3.0136 | custom NN, LightGBM, XGBoost |
| nyc_taxi_fare | meta_mle_sol_run_1 | meta-mle (greedy) | GPT-6.1-sol | 8 h | **12** | 12 | 4.0213 | 2.9611 | XGBoost, CatBoost |
| nyc_taxi_fare | meta_mle_sol_run_2 | meta-mle (greedy) | GPT-6.1-sol | 8 h | **10** | 10 | 3.7451 | 2.9597 | XGBoost, LightGBM, custom NN |
| nyc_taxi_fare | meta_mle_sol_run_3 | meta-mle (greedy) | GPT-6.1-sol | 24 h | **16** | 16 | 3.9879 | 2.9832 | custom NN, XGBRF |
| nyc_taxi_fare | mlevolve_run_1 | mlevolve | Gemini 3.6 Flash | 30 steps / 3 h | **21** | 31 | 3.9418 | 3.0721 | LightGBM, XGBoost, CatBoost |
| nyc_taxi_fare | mlevolve_run_2 | mlevolve | Gemini 3.8 Flash | 20 steps / 20 h | **17** | 21 | 2.6838 | 3.2408 | LightGBM, XGBoost, custom NN |
| nyc_taxi_fare | mlevolve_run_3 | mlevolve | Gemini 3.8 Flash | 20 steps / 20 h | **18** | 21 | 2.4933 | 3.1769 | LightGBM, XGBoost, CatBoost |
| playground-series-s6e7 | mlevolve_run_1 | mlevolve | Gemini 3.6 Flash | 30 steps / 3 h | **24** | 31 | 0.9504 | – | custom NN, CatBoost, XGBoost |
| spaceship-titanic | meta_mle_aide_gemini_run_1 | meta-mle (aide profile) | Gemini 3.8 Flash | 2 h | **24** | 24 | 0.8215 | 0.8078 | CatBoost |
| spaceship-titanic | meta_mle_fm_agent_gemini_run_1 | meta-mle (fm-agent / islands) | Gemini 3.8 Flash | 2 h | **52** | 52 | 0.8210 | 0.8073 | CatBoost, LightGBM, XGBoost |
| spaceship-titanic | meta_mle_gemini_run_1 | meta-mle (greedy) | Gemini 3.8 Flash | 2 h | **34** | 34 | 0.8199 | 0.8059 | LightGBM, CatBoost, XGBoost |
| spaceship-titanic | meta_mle_mlevolve_gemini_run_1 | meta-mle (mlevolve profile) | Gemini 3.8 Flash | 2 h | **27** | 27 | 0.8156 | 0.8034 | LightGBM, XGBoost, logistic regression |
| spaceship-titanic | mlevolve_run_1 | mlevolve | Gemini 3.8 Flash | 50 steps / 2 h | **30** | 35 | 0.8219 | 0.8069 | XGBoost, LightGBM, CatBoost |
| spaceship-titanic | mlevolve_run_2 | mlevolve | Gemini 3.8 Flash | 50 steps / 2 h | **41** | 45 | 0.8315 | 0.8078 | custom NN, LightGBM, XGBoost |
| tab_playground_dec_21 | aide_run_1 | AIDE | Gemini 3.8 Flash | 4 h | **9** | 9 | 0.9624 | 0.9559 | custom NN, XGBoost, LightGBM |
| tab_playground_dec_21 | meta_mle_gemini_run_1 | meta-mle (greedy) | Gemini 3.8 Flash | 4 h | **10** | 10 | 0.9625 | 0.9554 | LightGBM |
| tab_playground_dec_21 | meta_mle_gemini_run_2 | meta-mle (greedy) | Gemini 3.8 Flash | 4 h | **9** | 9 | 0.9624 | 0.9557 | LightGBM |
| tab_playground_dec_21 | meta_mle_gemini_run_3 | meta-mle (greedy) | Gemini 3.8 Flash | 4 h | **10** | 10 | 0.9626 | 0.9561 | LightGBM |
| tab_playground_dec_21 | meta_mle_sol_run_1 | meta-mle (greedy) | GPT-6.1-sol | – | **12** | 12 | 0.9606 | – | LightGBM |
| tab_playground_dec_21 | mle_claude_run_1 | Claude Code | Claude | – | **23** | 23 | 0.9618 | 0.9557 | LightGBM |
| tab_playground_dec_21 | mle_star | MLE-STAR | Gemini 2.5 Flash | – | **68** | 68 | 0.9589 | – | random forest |
| titanic | mle_star_flash_run_1 | MLE-STAR | Gemini 3.7 Flash | – | **81** | 81 | 0.8601 | – | CatBoost, XGBoost, custom NN |
| ttt-task | aide_run_1 | AIDE | Gemini 3.8 Flash | 8 h | **17** | 17 | 0.8883 | 0.8422 | no ML library |
| ttt-task | meta_mle_gemini_run_1 | meta-mle (greedy) | Gemini 3.8 Flash | 9 h | **8** | 8 | 0.8476 | 0.8508 | no ML library |
| ttt-task | meta_mle_sol_run_1 | meta-mle (greedy) | GPT-6.1-sol | 16 h | **11** | 11 | 0.8839 | 0.8867 | custom NN |
| ttt-task | meta_mle_sol_run_2 | meta-mle (greedy) | GPT-6.1-sol | 24 h | **12** | 12 | 0.8576 | 0.8598 | custom NN, XGBoost |
| ttt-task | mle-claude-run1 | Claude Code | Claude | – | **15** | 15 | 0.9002 | 0.8659 | no ML library |
| ttt-task | mle-claude-run2 | Claude Code | Claude | – | **16** | 16 | 0.8808 | 0.8453 | LightGBM, custom NN |
| ttt-task | mle-claude-run3 | Claude Code | Claude | – | **16** | 16 | 0.8920 | 0.8963 | custom NN |
| ttt-task | mlevolve_run_1 | mlevolve | Gemini 3.8 Flash | 15 steps / 2 h | **12** | 16 | 0.8464 | 0.7952 | custom NN |
| ttt-task | mlevolve_run_2 | mlevolve | Gemini 3.8 Flash | 50 steps / 12 h | **19** | 35 | 0.9161 | 0.8862 | custom NN |
| ttt-task | mlevolve_run_3 | mlevolve | Gemini 3.8 Flash | 10 steps / 16 h | **8** | 11 | 0.8895 | 0.8525 | custom NN |
| ttt-task | mlevolve_run_4 | mlevolve | Gemini 3.8 Flash | 100 steps / 36 h | **81** | 101 | 0.9091 | 0.8690 | custom NN, SGD |
| | | | | | **1933** | | | | |

### Models

Share of a dataset's pipelines that use each model (a script using several
counts once for each). "custom NN" means a network the script defines itself,
mostly in PyTorch. "no ML library" means a hand-written ranker or heuristic
with no model class, which is what most ttt-task scripts outside the NNs are.

| dataset | runs | pipelines | most used models |
| --- | --- | --- | --- |
| aptos2019-blindness-detection | 2 | 86 | custom NN 100%, timm CNN 100% |
| beaver_enroll | 4 | 626 | random forest 96%, LightGBM 23%, TabNet 23%, TabPFN 22%, logistic regression 2% |
| house_price | 2 | 59 | LightGBM 83%, custom NN 66%, XGBoost 36%, CatBoost 24%, Ridge 8% |
| nyc-housing-violation | 4 | 155 | LightGBM 83%, custom NN 63%, XGBoost 62%, CatBoost 45%, HistGB 12% |
| nyc_taxi_fare | 14 | 338 | LightGBM 73%, custom NN 59%, XGBoost 57%, CatBoost 6%, Ridge 4% |
| playground-series-s6e7 | 1 | 24 | custom NN 100%, CatBoost 100%, XGBoost 100%, LightGBM 100%, ExtraTrees 4% |
| spaceship-titanic | 6 | 208 | LightGBM 63%, CatBoost 60%, XGBoost 43%, custom NN 22%, logistic regression 20% |
| tab_playground_dec_21 | 7 | 141 | LightGBM 46%, random forest 46%, custom NN 4%, XGBoost 2%, HistGB 1% |
| titanic | 1 | 81 | CatBoost 93%, XGBoost 52%, custom NN 49%, LightGBM 16%, logistic regression 11% |
| ttt-task | 11 | 215 | custom NN 74%, no ML library 19%, LightGBM 7%, SGD 7%, XGBoost 4% |

Across the corpus: LightGBM 42%, custom NN 36%, random forest 35%, XGBoost 25%,
CatBoost 17%, TabNet 8%, TabPFN 7%, timm CNN 4%, logistic regression 4%, no ML
library 2%, HistGB 2%, Ridge 1%.

- Gradient boosting (LightGBM, XGBoost, CatBoost) is the default on every
  tabular dataset. A custom NN usually sits next to it, as an ensemble member
  rather than a replacement: 59-66% of the pipelines on nyc_taxi_fare,
  house_price and nyc-housing-violation, and every s6e7 pipeline.
- On aptos, every script builds its own network around a `timm` backbone, so
  both names count for all of them.
- ttt-task is the one dataset where boosting is rare: NNs and hand-written
  rankers do the work.
- The random-forest share comes almost entirely from MLE-STAR (beaver_enroll,
  tab_playground_dec_21).

Notes on the counts:

- Pipelines are the scripts the website's code analysis covers. For runs with a
  lineage, that is the scripts that appear in it. A few saved scripts are not in
  the lineage and are not counted: 1-3 per beaver_enroll run, and `pipeline_04`
  in `tab_playground_dec_21/mle_claude_run_1`.
- Lineage nodes are the steps the run's trajectory records. For mlevolve that is
  more than the scripts, because it keeps a script only for nodes that ran. aptos has no state file, so it has no lineage.
- 181 of the pipelines are ablations: MLE-STAR's `ablation_*.py` (runnable
  variants of the current solution, probes rather than candidates) and 5 in
  ttt-task Claude Code runs (`ablation_NN.py` in run 2; `pipeline_05.py` and
  `pipeline_10.py` in run 3, fused-choice grids of 8 and 16 variants). Without
  them, the corpus is 1752 pipelines.
- Degenerate scores: beaver_enroll's best CV values of 1.0 and 0.0 are not
  real (the runs were never validated against real data, see
  `mle_star_flash_run_3/result.txt`). Both aptos runs ended in agent error.
- `nyc_taxi_fare` mlevolve run 1 saw an anonymised description of the task
  (`od_cost_regression`, with `record_id`, `cost`, `origin_x/y`); runs 2 and 3
  saw the un-anonymised one naming New York City taxi fares. Same data and
  metric, so the scores compare, but in runs 2-3 the agent reconstructs domain
  knowledge that is in no data file - the September 2012 fare hike, the JFK flat
  rate, airport and river-crossing flags, a rotated Manhattan street grid.
  Runs 2 and 3 differ only in how the task description frames an optional
  library (TabFM): "optional, second-best to gradient boosting" in run 2,
  a peer of the GBDT libraries in run 3. Neither run used it in any node.
  Their best CV (2.49-2.68) is on the agent's own split and is not comparable
  to the meta-mle runs' locked CV; the Kaggle test column is.
- meta-mle nyc_taxi_fare runs chose their own training sample (0.5M-26M rows;
  see each `run.toml`). `meta_mle_mlevolve_gemini_run_4_upscaled{,_tuned}`
  re-run run 4's code on 10M rows (the tuned copy also hand-tunes
  `pipeline_02`). They are not agent runs.
- Claude Code runs (`mle-claude`, `mle_claude_run_1`) and meta-mle runs write
  skrub DataOps plans directly, so they have no `skrubify*/` folder. Their shared
  modules (`common.py`, `features.py`, `nn.py`, `rankers.py`, ...), exploration
  and analysis scripts, meta-mle's `evaluation_setup.py`, and `final_pipeline.py` (a refit for submission, "not a
  scored candidate") are not pipelines. Which Claude model ran them is not
  recorded in the run folders.
- `nyc-housing-violation/mle-claude-run1_v2` is not a separate run. It has the
  same workspace id and a byte-identical `results.json` as `mle-claude-run1`,
  with the pipelines rewritten onto a fine-grained DataOps `common.py`.
- `tab_playground_dec_21/mle_star` is one run; `skrubify_gemini` and
  `skrubify_openai` are two translations of it by different skrubify providers.
- `titanic/mle_star_flash_run_1` ran two solutions: `pipelines/1/`,
  `pipelines/2/` (solution 2's files are named `*_sol2.py`) and
  `pipelines/ensemble/`.

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
  dataset folder, choose its `[sample]` recipe and build `sample_<size>/input/` so a run
  in the repair loop takes about a minute. (Claude Code only discovers skills
  under `.claude/skills/`, which is why it does not live in `tools/`.)
