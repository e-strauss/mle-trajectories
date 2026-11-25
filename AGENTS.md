# AGENTS.md

## Read this first: skrub DataOps

**Before any work on pipelines in this repo, read `tools/skrub_dataops_summary.md`
in full — every section, top to bottom, not a skim or a grep.** Skrub DataOps is
the format every pipeline in this project is represented in, and the guide
records verified behaviour and pitfalls (CV on `mark_as_X`, `split_kwargs={}`,
raw target marking, eval-mode gating, `choose_from` only, …) that are not
obvious from the skrub docs and fail silently when missed. Re-read it in a new
session; do not rely on memory of it.

## What this project is

Machine Learning Engineering (MLE) agents are given a dataset (or a data lake of
several tables) and a task, and must build a well-performing ML pipeline. This is
a code-optimization problem: the agent searches the space of possible pipelines,
using a cross-validation score as the optimization signal.

This repo collects the **search trajectories** of different agents (MLE-STAR,
mlevolve, Codex / "mle-Codex") on different datasets, and analyzes them.
The central idea: translate every pipeline an agent executed into a **skrub
DataOps plan** — a computation graph of the classical, messy Python script — so
pipelines can be compared structurally and each step diffed against its parent
to see what the agent actually changed.

## Layout

```
<dataset>/
    input/                 full data (gitignored); pipelines read ./input/...
    sample_<size>/input/   self-consistent samples, same file names (gitignored)
    sample_<size>/sample_manifest.json   how each was built (tracked)
    get_data.sh | DATA.md  how to obtain the data
    make_sample.py         per-dataset sampler, where the [sample] recipe needs one
    <agent>_run_<N>/       one agent run (naming varies: mle-star-run-1, mle_star_run_1, mle-Codex-run1, ...)
        pipelines/         the agent's original scripts — never modify
        final_state.json   run metadata / state dump (or journal_slim.json, workspace.json, ...; format depends on the agent)
        skrubify*/         skrub DataOps rewrites of each pipeline
tools/
    skrub_dataops_summary.md   the DataOps guide (read in full, see above)
    skrubify/                  LLM-driven script -> DataOps converter + validator
    pipeline_analyzer/         lineage, operator DAGs, per-step diffs -> HTML report; runtime measurement
    dataset_sample/            builds sample_<size>/ from the [sample] recipe in dataset.toml
    code_stats/                static features of a script (size, components, hyperparameters, data handling)
    trajectory.py              tabular overview of one run's final_state.json
    getcomp.sh                 one-off Kaggle download
website/                       corpus explorer: Python registry/API + Next.js frontend (see website/README.md)
```

Each dataset folder has a `dataset.toml` and each run folder a `run.toml`: the
few facts the folders cannot say (agent, metric, main skrubify source, notes).
`website/backend/registry.py` derives everything else and reports inconsistencies.

`README.md` has the full corpus table, per-dataset notes and data sources.

## Tools

At the start of a session, list `tools/` (`ls tools`) so you know which tools
exist by name. Do not investigate them further by default. Look into a tool
only when:

- the user mentions it, or
- you think it could help with the current task — in that case **ask the user
  first** whether to use it before reading or running it.

When you do use a tool, read its README first (`tools/skrubify/README.md`,
`tools/pipeline_analyzer/README.md`).

## Working conventions

- Agent originals under `pipelines/` are the raw data of the study: never edit them.
  Rewrites go into `skrubify*/` folders.
- Agents that write DataOps plans themselves (mle-Codex runs) have no `skrubify*/`
  folder; their shared modules (`common.py`, `features.py`, ...) and
  `data_exploration_*.py` are not pipelines.
- `--run-in <dataset>` runs against full data, `--run-in <dataset>/sample_<size>`
  against a sample. Full data is slow; build a sample first
  (`python -m dataset_sample <dataset> --size 100k`, or "Build sample" on the
  website) before iterating with skrubify's repair loop or the runtime tool. A
  dataset without a `[sample]` recipe needs one: the `dataset-sample` skill says
  how to choose it. Never cap rows at read time instead: there is no row cap
  any more, and a sample must keep joins and classes intact.
- `stratum` is a drop-in for skrub (`import stratum as skrub`) with a much faster
  evaluator; large fine-grained plans are only practically runnable under it.
- Environment: `uv` project (`pyproject.toml`, Python >= 3.14).
