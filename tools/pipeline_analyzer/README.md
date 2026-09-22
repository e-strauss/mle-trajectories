# pipeline_analyzer

Analyze how a series of **skrub DataOps pipelines** evolves across iterations, by
extracting each pipeline's **stratum logical operator DAG** and diffing every
pipeline against its `PARENT`. Emits a self-contained, theme-aware HTML report
with three parts:

1. **the search tree** — every step the agent ran, linked to the step it came
   from, colored by score-vs-parent or by search phase,
2. **the operator explorer** — *one* graph for the whole run: every pipeline's
   DAG overlaid, an operation shared by several pipelines collapsed onto a single
   node. Tick pipelines to add them; click an operation to see which pipelines
   carry it,
3. **per-pipeline detail** — what each step changed against its parent,
   structurally *and* in estimator hyperparameters.

## Run

A run directory is laid out `<dataset>/<agent>_run_<n>/`, holding

- `pipelines/` — the agent's **original** scripts, exactly as it wrote them
  (sub-folders such as `ensemble/` kept as-is),
- the state dump / journal (`final_state.json`, `journal_slim.json`, …) *beside*
  `pipelines/`, not inside it,
- one `skrubify*/` folder per conversion run, mirroring `pipelines/`' sub-paths,
- the generated report(s).

`--pipelines` therefore defaults to `./pipelines` and points at whichever folder
holds the *skrub* pipelines: the agent's own output for an agent that already
writes DataOps plans, a `skrubify*/` folder otherwise.

Two ways to get the lineage, depending on what the agent left behind.

**A. The pipelines annotate themselves** (`PARENT`/`DESCRIPTION` in each file,
scores in `results.json`) — the mle-claude style:

```bash
# from tab_playground_dec_21/mle_claude_run_1/
python -m pipeline_analyzer --pipelines pipelines --out pipeline_evolution.html --text
```

**B. A trajectory supplies the lineage** — the run's own state dump holds every
executed step with its score, and the steps were (hand-)skrubified into a folder
of plain `.py` rewrites with no annotations. The format is detected from the file:

```bash
# MLE-STAR — from tab_playground_dec_21/mle_star/
python -m pipeline_analyzer --trajectory final_state.json \
    --pipelines skrubify_openai --pipelines skrubify_openai/ensemble \
    --fold-identical-code --out pipeline_evolution_openai.html --text

# mlevolve — from nyc_taxi_fare/mlevolve_run_1/
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
- `--runtime-stats FILE` measured runtimes to fold into the report (default:
  `runtime_stats_<folder>.json` beside the pipelines folder, if it exists);
  `--no-runtime-stats` ignores it
- `--per-pipeline-dags` also render each pipeline's own DAG as a static graphviz
  SVG under its section (default off — the explorer covers this interactively,
  and on a 68-pipeline run those SVGs were ~70% of the file)
- `--text`           also print a one-line-per-pipeline summary to stdout

No dataset is required — see below.

## The report

### Search tree

One node per step, `score (Δ vs parent)`, edges pointing from the step that was
derived from which. Two colorings, switchable in the page: **score vs parent**
(green improved / amber flat / red regressed — oriented by the trajectory's own
metric direction, so a lower-is-better run reads the right way round) and
**search phase** (Init / Ablation / Improve / … in the same colors the explorer's
pipeline list uses).

Clicking a node ticks that pipeline in the explorer; shift-clicking takes its
whole subtree. Ticked steps carry a thick outline, so the tree doubles as the
map of what the graph below is showing.

### Operator explorer

The union of every pipeline's DAG. Nodes are keyed by the **content signature**
of the whole sub-computation below them (`dag.py`), so two pipelines share a node
only when they really do compute the same thing — this is the same key the diffs
use, and it is what makes overlaying meaningful rather than merely name-matched.

- **which pipelines** — tick them in the left-hand list (grouped by phase, with
  score, Δ and operator count; `all` / `none` / `invert` / `best path` /
  `roots`, plus a name filter). The default selection is the **best path**: root
  → the best-scoring pipeline. Hovering a row outlines that pipeline's
  operations in the graph.
- **colors** — *how widely shared*: an operation every ticked pipeline has is
  neutral gray, and the fewer pipelines carry it the hotter it reads (so what
  the run actually varied is what lights up); *by pipeline*: an operation unique
  to one ticked pipeline takes that pipeline's color (the dots in the list are
  the key), shared ones stay gray — the useful mode for comparing two or three;
  *diff vs parent*: with exactly one pipeline ticked, the same green/amber/red
  diff coloring the per-pipeline sections describe, parent-only operations drawn
  dashed.
- **which pipelines share an operation** — click it. The inspector names the
  operation and lists every pipeline containing it, collapsed by default when
  the list is long. Each entry is a tick (to add/remove it from the graph) plus
  a link to its section.
- scroll to zoom, drag to pan, `+` / `−` / `reset view`, and **full screen** —
  which is the answer to a large union DAG: the graph takes the whole window
  with the picker still beside it (native Fullscreen API, falling back to a
  fixed overlay; `Esc` leaves).

Layout is done in the browser (`explorer.js`: layered/Sugiyama — longest-path
layering tightened toward the sinks, dummy nodes for long edges, barycenter
ordering keeping the crossing-minimal pass, then order-preserving coordinate
relaxation), because a graphviz layout fixed at generation time would scatter a
three-pipeline selection across a canvas sized for all 68. It lays out 654 nodes
(the 68-pipeline run, everything ticked) in ~100 ms.

### Operator statistics

Three tables, all aggregated over the run rather than per pipeline. Two are
static — operator counts at the **logical** and at the **physical** altitude
(total, how many pipelines contain it, and the per-pipeline distribution). The
third, **measured time**, appears when a runtime store is loaded and is the time
counterpart: per operator class, total time, share, calls, **per call**,
how many pipelines it ran in, and the pipeline it was heaviest in (linked).

The two rankings are nothing alike, which is the point of having both: on the
dec21 run the most *numerous* operator is 1506 elementwise `NumericOp` maps
worth 0.1% of the time, while `PredictorOp` — 630 calls — is 95.8%. `per call`
is what separates an operator that is expensive from one that is merely
frequent. A collapsed row-per-operator-*instance* breakdown sits under the
table for the 113-row detail (which assign-map, which projection).

Operator classes are matched to the physical table's spelling: stratum's
readable label drops the `Op` suffix on a few ops (`Predictor`, `Split`,
`Transformer`, `Choice`), restored here against the classes the physical DAGs
actually contain.

### Per-pipeline detail

Operator counts vs parent, the added/removed operations, estimator swaps and
hyperparameter deltas, the measured runtime block — and *show in explorer*,
which ticks that pipeline alone and switches the graph to diff coloring.

## Runtime statistics

The report above is static analysis: it never executes a pipeline and needs no
dataset. Actually *running* the pipelines is a separate command, because it needs
the real data and is expensive:

```bash
# from tab_playground_dec_21/mle_star/
python -m pipeline_analyzer.runtime \
    --pipelines skrubify_openai --pipelines skrubify_openai/ensemble \
    --run-in .. --sample-rows 100000
```

This writes `runtime_stats_skrubify_openai.json` beside the report, which
`python -m pipeline_analyzer` then picks up automatically and renders as a
per-pipeline runtime block plus a ranked table.

Each pipeline runs in its own process, with `--run-in` as the working directory
(the folder holding `input/`, which is what `./input/train.csv` in a pipeline
resolves against). Scoring goes through stratum's scheduler
(`stratum._api.grid_search`) with `stats=True`, so every operator is timed.

Collected per pipeline: wall time of the scored grid search, time inside
operator bodies, buffer-pool overhead, per-operator-type call counts and times,
buffer-pool hit/spill counters, resident memory over time, the CV scores, and the
splitter used.

### Memory

`memory_tracker.MemoryTracker` runs a side-car process that polls RSS every
100 ms, so the store keeps the *shape* of a run and not just its high-water mark:
peak (and when it happened), mean, start/end, and a downsampled curve the report
draws as an area chart with the scored grid search shaded. The full sample series
goes to `runtime_stats_<folder>.mem/<pipeline>.csv` (`time_sec,rss_mb`), written
incrementally so a killed or OOM-ed pipeline still leaves its trace.

Two peaks are stored and they mean different things:

- `memory.peak_mb` — the sampled curve's maximum, for the pipeline's own
  interpreter. Misses a spike that fits between two polls.
- `max_rss_mb` — `getrusage`, covering this process *and* any worker processes an
  estimator forked. This is the number to trust for "how much did it need".

Over the 68 pipelines of `mle_star/skrubify_openai` the two agree exactly for
half of them (median difference 0 MB, mean 5.8 MB), but 15 differ by more than
10 MB and the worst, `train0_improve0`, by 90 MB (1361 vs 1452) — a peak that
fell between two 100 ms polls. So: read the curve for shape, `max_rss_mb` for
the high-water mark, and drop `--mem-interval` if you need the curve to catch
short spikes.

**`--sample-rows` changes one operator, not just the data size.** It caps rows by
rebinding `pd.read_csv` before the pipeline builds its plan, and the plan captures
that wrapper (`apply_func(pd.read_csv)`). stratum's lowering recognises the real
`pandas.read_csv` and turns it into a native `PandasReadCSV` op; it does not
recognise the wrapper, so under sampling the read stays an opaque
`CallOp(read_csv)` over a `ValueOp(path)`. Everything else measures the same plan
— but the read row of a sampled store is not comparable with a full-data one, and
the report says so where it shows.

- `--mem-mode process|system|off` — `system` measures total memory in use, for a
  workload that fans out over processes; `off` skips sampling
- `--mem-interval SEC`, `--no-mem-csv`

Turning tracking on backfills a store measured without it: an entry with no
memory series is not considered fresh, so no `--force` is needed.

**An upgraded stratum invalidates a store.** The logical optimizer decides the
DAG, so a new stratum build changes both the op mix a pipeline is credited with
and how long it takes — on one run the same 68 pipelines came out as 769
`NumericOp` against 782, and 468 `CallOp` against 419. `stratum.__version__`
cannot see this: it is a static `0.0.0.dev2` across builds, so the store records
the installed *commit* (from the wheel's `direct_url.json`) per entry and an entry
from a different commit is re-measured. A store from before the commit was
recorded has none, and is stale as soon as the executor has one. The sweep says
which build it found:

```
! store was measured under stratum 834dc0298037, this interpreter has
  8b7f7ba3c632 -- those entries are stale and will be re-measured
```

`--list` applies the same rule, so it is the cheap way to see whether an upgrade
orphaned a store before starting a sweep that may take hours.

Options:
- `--run-in DIR`     working directory for the pipelines (default: nearest
  ancestor of `--pipelines` holding `input/`)
- `--sample-rows N`  cap `read_csv` at N rows. A full sweep on a large table can
  take days; a sample makes it minutes. Stored per entry, and an entry measured
  at a different sample size is re-measured rather than silently mixed in
- `--only NAME …`, `--limit N`  measure a subset
- `--timeout S`      per pipeline, killing the whole process group (default 3600)
- `--force`, `--retry-failed`  re-measure cached / previously failed entries
- `--no-stats`       skip stratum's per-operator timing (wall clock only, no
  instrumentation overhead)
- `--list`           show what the store holds and what a sweep would run
- `--out FILE`       store path

**The store is a cache.** A pipeline already measured with the same code
(`code_sha1`), the same sample size and the same stratum commit is skipped, and
the store is rewritten after every pipeline, so a sweep can be interrupted and
resumed, or filled in one pipeline at a time. The commit is kept per entry
rather than only per store, because an interrupted sweep across an upgrade
leaves entries from two builds and each has to be judged on the one that
produced it.

### What the runner does and does not touch

The pipeline files run unmodified. `make_grid_search` is intercepted so the
scoring call goes to `stratum._api.grid_search` (what stratum's own patch does
under `scheduler=True`) and the scheduler can be timed and queried for stats; the
call returns a shim whose `results_` looks like skrub's pandas frame, because
stratum's is a polars frame keyed `id`/`scores`, so each file's own reporting
block still prints its score.

`cv` is left to stratum: `grid_search._resolve_cv` prioritises an explicit `cv`
and otherwise uses the splitter declared on the plan via
`mark_as_X(cv=..., split_kwargs=...)`, resolving it through skrub when it is
itself a DataOp. Earlier versions dropped the declared splitter and crashed on
stratified ones ([#199](https://github.com/deem-data/stratum/issues/199), fixed
in `834dc029`); the runner carried a workaround for both, now removed. Removing
it reproduced scores bit-identically, so measurements taken through it remain
comparable.

**Known gap** ([#200](https://github.com/deem-data/stratum/issues/200)):
stratum's scheduler only honours a bare string `scoring`. A plain callable, or
`None`, is silently replaced by `mean_squared_error`; a `make_scorer(...)`
scorer loses its kwargs; and `neg_*` scorers report with the opposite sign to
sklearn (the ranking stays right, the value does not). Entries carry
`scoring_honoured: false` where this applies — one pipeline here
(`train9_improve0`) passes a callable, so its `best_score` is an MSE, not the
accuracy it asked for. Its *runtime* numbers are unaffected.

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
5. **Merge** (`merged.py`) — union every pipeline's DAG by signature into one
   node list (topologically ordered, each node carrying the pipelines that
   contain it) and serialize it as the explorer's JSON payload; ~100 KB for the
   68-pipeline run.
6. **Render** (`render.py`, `html.py`) — graphviz → inline SVG for the search
   tree (both colorings, toggled in the page); the explorer's markup, CSS
   (`explorer.css`) and layout/interaction code (`explorer.js`) inlined into one
   self-contained HTML file. The merged DAG itself is laid out client-side.

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
- **Runtime numbers are a measurement, not a property of the plan.** They depend
  on the machine, the sample size, and `stats=True`'s own instrumentation
  overhead (every operator is timed, so a plan with many cheap operators pays
  more of it than one with a single expensive fit). Compare pipelines within one
  store, not across stores.
- **The explorer needs JavaScript.** It is one inlined script and one inlined
  JSON payload, no network access, but with scripting off the report is the
  search tree plus the text sections (`--per-pipeline-dags` brings the static
  per-pipeline DAGs back).
- **Dependencies:** `stratum` (imported as `stratum.optimizer.*`), `skrub`,
  `graphviz` (python binding + the `dot` binary), `psutil` (runtime memory
  sampling only), and whatever the pipelines themselves import (lightgbm,
  torch/skorch, …). A pipeline whose import fails is
  reported and skipped, not fatal.
- Designed to move: it only imports `stratum.optimizer.*`, so it works both in
  this repo and later with stratum installed as a dependency. Point `--pipelines`
  at any workspace of `pipeline_*.py` files following the same `pred` / `PARENT` /
  `DESCRIPTION` convention.
