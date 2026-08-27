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

## Tools

- [`tools/skrubify`](tools/skrubify) — convert a pandas/sklearn script into a
  skrub DataOps pipeline (LLM does the translation, a validator gates it).
- [`tools/pipeline_analyzer`](tools/pipeline_analyzer) — build the lineage of a
  run and emit an HTML report: tree, per-step operator DAG, per-step diff.
- [`tools/trajectory.py`](tools/trajectory.py) — tabular overview of one run
  (steps, scores, timings, parents).
