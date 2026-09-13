# mle-trajectories

Development trajectories of ML engineering agents (MLE-STAR, mlevolve, Claude
Code), collected per dataset, plus tooling to compare the pipelines they produce.

The experiment: take every script an agent executed during a run, translate it
into a skrub DataOps plan, extract the stratum operator DAG, and diff each step
against its parent. That shows what the agent actually changed from iteration to
iteration, and what it kept.

## Layout

```
<dataset>/<agent>_run_<n>/
    pipelines/        the agent's original scripts, untouched
    final_state.json  its state dump / journal (format depends on the agent)
    skrubify*/        the skrub DataOps rewrite of each step
```

## Corpus

7 datasets, 15 agent runs, **945 pipelines** in total. One pipeline = one script
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
| nyc_taxi_fare | mlevolve_run_1 | mlevolve | – | 21 | – | – | **21** |
| playground-series-s6e7 | mlevolve_run_1 | mlevolve | – | 24 | – | – | **24** |
| tab_playground_dec_21 | mle_star | MLE-STAR | 2 | 53 | 10 | 3 | **68** |
| tab_playground_dec_21 | mle_claude_run_1 | Claude Code | – | 24 | – | – | **24** |
| ttt-task | mlevolve_run_1 | mlevolve | – | 12 | – | – | **12** |
| ttt-task | mlevolve_run_2 | mlevolve | – | 19 | – | – | **19** |
| ttt-task | mlevolve_run_3 | mlevolve | – | 8 | – | – | **8** |
| | | | **30** | **722** | **175** | **18** | **945** |

Runs per dataset: beaver_enroll 4, aptos2019-blindness-detection 2,
cover_type_multi_table 2, nyc_taxi_fare 1, playground-series-s6e7 1,
tab_playground_dec_21 2 (one MLE-STAR, one Claude Code), ttt-task 3.

Notes on the counts:

- The ablation column is MLE-STAR's ablation scripts — runnable variants of the
  current solution, but probes rather than candidate solutions. Drop them and
  the corpus is 770 pipelines.
- mlevolve keeps a script only for nodes that ran; its journals hold 31 nodes
  against 21 saved scripts (nyc_taxi_fare), 31 against 24
  (playground-series-s6e7), 16 against 12 (ttt-task run 1), 35 against 19
  (ttt-task run 2) and 11 against 8 (ttt-task run 3).
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
