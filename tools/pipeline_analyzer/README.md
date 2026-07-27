# pipeline_analyzer

Analyze how a series of **skrub DataOps pipelines** evolves across iterations, by
extracting each pipeline's **stratum logical operator DAG** and diffing every
pipeline against its `PARENT`. Emits a self-contained, theme-aware HTML report:
a lineage tree plus, per pipeline, a diff-colored operator DAG and a summary of
what stayed the same and what changed (structurally *and* in estimator
hyperparameters).

## Run

```bash
# from tab_playground_dec_21/mle_claude-run1/ (the dir containing this package)
python -m pipeline_analyzer --pipelines pipelines --out pipeline_evolution.html --text
```

Options:
- `--pipelines DIR`  folder of `pipeline_*.py` files (default `./pipelines`)
- `--results FILE`   `results.json` for scores (default `<pipelines>/results.json`)
- `--out FILE`       output HTML (default `pipeline_evolution.html`)
- `--unroll-choices` unroll `choose_from` into separate branches (default: folded)
- `--text`           also print a one-line-per-pipeline summary to stdout

No dataset is required — see below.

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
