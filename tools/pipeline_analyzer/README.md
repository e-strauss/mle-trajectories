# pipeline_analyzer

Analyze how a series of **skrub DataOps pipelines** evolves across iterations, by
extracting each pipeline's **stratum logical operator DAG** and diffing every
pipeline against its `PARENT`. Emits a self-contained, theme-aware HTML report:
a lineage tree plus, per pipeline, a diff-colored operator DAG and a summary of
what stayed the same and what changed (structurally *and* in estimator
hyperparameters).

## Run

Two ways to get the lineage, depending on what the agent left behind.

**A. The pipelines annotate themselves** (`PARENT`/`DESCRIPTION` in each file,
scores in `results.json`) — the mle-claude style:

```bash
# from tab_playground_dec_21/mle_claude-run1/
python -m pipeline_analyzer --pipelines pipelines --out pipeline_evolution.html --text
```

**B. A trajectory supplies the lineage** — the run's own state dump holds every
executed step with its score, and the steps were (hand-)skrubified into a folder
of plain `.py` rewrites with no annotations. The format is detected from the file:

```bash
# MLE-STAR — from tab_playground_dec_21/mle_star-run4/
python -m pipeline_analyzer --trajectory final_state.json \
    --pipelines skrubify_openai --pipelines ensemble/skrubify_openai \
    --fold-identical-code --out pipeline_evolution_openai.html --text

# mlevolve — from nyc_taxi_fare/mlevolve-run1/
python -m pipeline_analyzer --trajectory journal_slim.json \
    --pipelines skrubify_5_6_sol --out pipeline_evolution.html --text
```

Options:
- `--pipelines DIR`  folder of pipeline files (default `./pipelines`); **repeatable**
  — a step is matched to the first folder holding `<module>.py`
- `--trajectory FILE` run state dump supplying parents, scores, timings and the
  agent's rationale, instead of `PARENT`/`results.json`
- `--trajectory-type` trajectory format (default: detected — `mle-star`, `mlevolve`)
- `--fold-identical-code` with `--trajectory`: steps whose *original* code is
  byte-identical share one skrubified DAG (see "Skrubification noise" below)
- `--results FILE`   `results.json` for scores (default `<pipelines>/results.json`);
  ignored with `--trajectory`
- `--out FILE`       output HTML (default `pipeline_evolution.html`)
- `--unroll-choices` unroll `choose_from` into separate branches (default: folded)
- `--text`           also print a one-line-per-pipeline summary to stdout

No dataset is required — see below.

## Trajectory mode

`pipeline_analyzer/trajectory.py` reads a run's state dump into ordered `Step`s
(`tools/trajectory.py` prints the same steps as a table; `--modules` there shows
the module/parent mapping). Two formats, detected by shape:

### `mlevolve` (`journal_slim.json`)

An explicit search tree, so nothing is inferred: `nodes` carry the score, stage
(draft / debug / improve / evolution), timing, exception type and the agent's
plan, `node2parent` carries the edges, and each node names the file its code went
to (`code_file` → the module). Nodes whose code was not kept (buggy attempts) are
still reported as steps but hold no module, so a child's parent is lifted to the
nearest ancestor that does have one.

### `mle-star` (`final_state.json`)

Per step it recovers:

- **the exported module** — the file stem the run's code was written to, which is
  what links a trajectory step to a skrubified file:
  `init_code_{cand}`, `train0_{merge_round}`, `train{step}`, `ablation_{step}`,
  `train{step}_improve{plan}`, `ensemble{round}`, `final_solution`.
- **the parent** — MLE-STAR records no parent pointers, so they are reconstructed
  from the search structure: candidates are independent roots, the merger chain
  starts at the winning candidate, each refine step's base is the previous step's
  *accepted* variant, and ablations/plan variants branch off their step's base.
  The winner is identified by **code identity** (an accepted variant is copied
  verbatim into the next base, so a string match is exact), falling back to the
  best score.
- **score / time / returncode / rationale** — the refine plan text for an improve
  step, the ablation summary for an ablation, the ensemble plan for an ensemble.

Steps that were never skrubified are listed and skipped; a skipped *parent* is
replaced by its nearest skrubified ancestor, so a partial folder (e.g.
`skrubify_gemini`, 7 of 68 steps) still yields one connected tree whose diffs
span the gap.

### Skrubification noise

When each step is skrubified independently, two rewrites of the *same* original
differ in ways the original does not — a row filter written as `isin` in one file
and `map(...) >= n` in the next re-signatures every node above it (Merkle diffs
bubble). On run4 the resulting floor is **~14% shared nodes between rewrites of
byte-identical original code** (median over the 12 such pairs), i.e. a step that
changed nothing reads as a near-total rewrite.

`--fold-identical-code` removes exactly that noise: the trajectory knows which
steps are code-identical, so the first one's DAG stands for the group and those
steps diff as "no change" (on run4: all 11 unchanged `train{s}` bases plus
`train0_0`). It cannot help *across* different originals — those diffs remain
noise-dominated, so read structural diffs between genuinely different steps with
that in mind, or skrubify once and reuse the file for repeated code.

## How it works

1. **Load** (`loader.py`) — import each pipeline module under
   `skrub.config_context(eager_data_ops=False)` so building the plan touches **no
   data** (no CSV read, no previews), grab the module-level `pred`, and run
   stratum's `logical_optimize` to get the logical Op DAG.
2. **Model** (`dag.py`) — walk the DAG into signature-keyed nodes. Each node gets
   a **recursive content signature** (Merkle hash: `op type + config + child
   signatures`). This is what lets nodes align *across* separately-built
   pipelines — stratum's own `Op.structure_key()` keys inputs by `id()`, which
   only works within one DAG. Constants (numpy/pandas) are hashed **by value** so
   identical feature blocks match even though each pipeline re-imports
   `features.py` and mints fresh objects.
3. **Diff** (`diff.py`) — set-difference over signatures → shared / added /
   removed. The **change frontier** is the added nodes whose inputs all already
   existed in the parent (the operations *genuinely* introduced); every other
   added node is an ancestor whose signature shifted because a descendant changed
   (Merkle diffs bubble up). Estimator **swaps** and **hyperparameter** deltas are
   reported separately, aligned by logical family.
4. **Lineage** (`lineage.py`) — build the `PARENT` tree, annotate with scores from
   `results.json`.
5. **Render** (`render.py`, `html.py`) — Graphviz → inline SVG for DAGs (diff
   coloring) and the lineage tree; assembled into one self-contained HTML file.

## Notes / limitations

- **Logical IR only.** The full physical `optimize()` currently crashes on these
  plans (`PosixPath has no len()` — `common.load_csv` wraps a `Path`, not a
  `str`, and read-op lowering assumes a string). Logical IR is the right altitude
  for cross-pipeline structural diffing anyway.
- **Frontier subtlety worth knowing.** skrub records column access on the
  *current* frame, so re-accessing a column after an intermediate `.assign(...)`
  produces a *new* projection node. Feature builders that re-project columns
  therefore show a small, real change frontier (the new projections) with the
  downstream math bubbling — this is accurate, not noise.
- **Opaque nodes** (`ImplOp`, a raw `lambda` inside `apply_func`) are keyed by a
  repr with hex ids scrubbed; two distinct lambdas can look equal. Only affects
  `add_target_encoding`'s lambda (pipeline_22); harmless here.
- **Dependencies:** `stratum` (imported as `stratum.optimizer.*`), `skrub`,
  `graphviz` (python binding + the `dot` binary), and whatever the pipelines
  themselves import (lightgbm, torch/skorch, …). A pipeline whose import fails is
  reported and skipped, not fatal.
- Designed to move: it only imports `stratum.optimizer.*`, so it works both in
  this repo and later with stratum installed as a dependency. Point `--pipelines`
  at any workspace of `pipeline_*.py` files following the same `pred` / `PARENT` /
  `DESCRIPTION` convention.
