---
name: dataset-sample
description: Build a small, self-consistent sample of a dataset folder so skrubify's --run-in repair loop and pipeline_analyzer runs finish in about a minute instead of hours. Use when pointed at a dataset directory in mle-trajectories, or when asked to downsample / subsample / make a sample of task data, or when a pipeline run against full data is too slow to iterate on.
---

# Downsampling a dataset for the run loop

## Why this exists

skrubify's validation **runs** the pipeline (`tools/skrubify/validate.py:179`).
With `--compare-source` it runs the original too, once per repair round. Against
nyc_taxi's 5.7 GB `train.csv` or ttt-task's 623M-edge link graph, one repair
round is hours, so the run layer — the only layer that catches scoring-time
failures and the only one that checks a conversion is *faithful* rather than
merely valid — is unusable.

The fix is a small sample that keeps the task's structure. Not a row cap:
`pipeline_analyzer` already has `--sample-rows`, and its own README warns
(`tools/pipeline_analyzer/README.md:207-213`) that it
"changes one operator, not just the data size… the read row of a sampled store
is not comparable with a full-data one". On a joined or graph task a row cap is
worse than slow — it produces a dataset that runs and measures nothing.

Your output is a **per-dataset `make_sample.py`**, because the reasoning is
per-dataset. `ttt-task/make_input_sample.py` is the worked example; read it
before writing a new one.

## Layout you are producing

```
<dataset>/
    input/                  full data (get_data.sh or DATA.md)
    make_sample.py          what you write
    sample/
        input/              SAME filenames as ../input/
        sample_manifest.json
```

`sample/` is a second run-root, so pipelines that hardcode `./input/train.csv`
need no edit at all:

```bash
python -m skrubify <run>/pipelines/x.py --run-in <dataset>          # full
python -m skrubify <run>/pipelines/x.py --run-in <dataset>/sample   # sample
```

Keep **filenames, extensions and formats identical** to `input/`. A sample that
renames `train.csv` to `train_small.csv` is useless here.

## Step 1 — read the task before touching the data

`<dataset>/DATA.md` if present, then the run's own record: `config.yaml`
(mlevolve: `data_dir`, `desc_file`, inline `goal`), `final_state.json`
(MLE-STAR: `task_description`, `task_type`, `lower`), `workspace.json`
(Claude Code: `input_source`, `scoring`, `cv`), or `task_description.txt`.

You need four things from it: **the target**, **the metric**, **the CV/split**,
and **which files are joined to which**. Do not infer the split from the data —
`house_price` looks like a plain table and is an out-of-time split.

Then profile `input/`: per file, size, row count, columns, dtypes, and the
cardinality of every candidate key. Stream; never load a multi-GB file to count
its rows.

## Step 2 — pick the strategy from the task shape

Get this wrong and the sample runs but measures nothing.

| shape | example | strategy |
|---|---|---|
| single flat table | `playground-series-s6e7`, `tab_playground_dec_21` | seeded row sample; **stratify on the target** for classification, so rare classes survive. Sample `test.csv` and `sample_submission.csv` to the *same* ids |
| out-of-time split | `house_price` | sample **within** each period, never across the boundary. Keep the last training period well represented — that is where the extrapolation signal is |
| star / snowflake | `cover_type_multi_table`, `beaver_enroll` | sample the fact/label table, then semi-join every dimension down to surviving keys. Follow the key chain (base key → intermediate keys → leaf tables). Keep small lookups whole. Keep **all** rows of a multi-row-per-key table for every surviving key |
| graph | `ttt-task` | neighbourhood expansion from seeds, then keep an edge only if **both** endpoints survive. Random node sampling leaves a graph with no edges, because two random nodes are almost never connected |
| file-per-row (images) | `aptos2019-blindness-detection` | stratify the index CSV on the label, then **hardlink** only the referenced files (`os.link`) rather than copying gigabytes |

State the choice and its justification in the script's docstring, including what
a naive row sample would have destroyed. `make_input_sample.py:1-11` is the model:

> Random domain sampling would destroy the link graph (623M edges over 46M
> nodes: two random domains are almost never connected), so the sample is a
> NEIGHBOURHOOD: seed domains plus their link-graph neighbours that are
> themselves tracked, then every edge whose endpoints both survive. That keeps
> the graph-derived features (degrees, direct tracker links, neighbour tracker
> adoption) non-trivial.

## Step 3 — write `<dataset>/make_sample.py`

Contract, all of it taken from `make_input_sample.py`:

- **Docstring** naming the strategy, the reason, and what breaks without it.
- **Paths as env-overridable defaults**, relative to the script:
  ```python
  HERE = Path(__file__).resolve().parent
  SRC = Path(os.environ.get("<DS>_SRC", HERE / "input"))
  OUT = Path(os.environ.get("<DS>_OUT", HERE / "sample" / "input"))
  ```
  The override is how several sizes come out of one script — it is how
  `ttt-task`'s `input_200k` / `input_1m` / `input_5m` were built.
- **Size knobs as env-overridable ints** with defaults tuned so one pipeline runs
  in roughly a minute: `N_ROWS = int(os.environ.get("<DS>_N_ROWS", 100_000))`.
- **`rng = np.random.default_rng(0)`** — one fixed seed, so the sample is
  reproducible and two runs compare.
- **Streaming** for anything large: `pd.read_csv(..., chunksize=500_000)` with a
  per-chunk filter and one `concat` at the end, as
  `make_input_sample.py:93-102` does.
- **Dtypes preserved** on write (downcast only deliberately, and say so).
- **A progress print per step** with row counts, `flush=True`. These are what
  tell you the sample is not degenerate before you run anything.
- **`sample/sample_manifest.json`** written at the end: source dir, every knob's
  value, the seed, per-file row counts before and after, and a timestamp. It is
  the only record of how the sample was built, since the data itself is
  gitignored — this is why `.gitignore` has an explicit `!**/sample_manifest.json`
  negation.

## Step 4 — verify before believing it

Run every one of these; a sample that fails any of them is not a sample.

- **Same file set** as `SRC`, same names and extensions.
- **Same schema** per table: identical column names *in the same order*, and
  identical dtypes.
- **Referential integrity**: for every join edge, no key value in the child
  table is missing from the parent. For graphs, no edge endpoint missing from the
  node table.
- **Label coverage**: classification — every class still present, and rare
  classes not reduced below the CV fold count (a class with fewer rows than
  `n_splits` breaks `StratifiedKFold`; several pipelines in this repo have
  explicit rare-class handling that a sample can accidentally trigger or
  silence). Regression — target range and median still plausible.
- **Hardcoded constants still hold.** Grep the pipelines for magic numbers
  before sampling. `make_input_sample.py:83` copies all 355 tracker rows
  unfiltered precisely so `NUM_TRACKERS = 355` stays true. Where a constant
  *cannot* hold — `cover_type_multi_table`'s `assert len(X) == 423680` — record
  it in `DATA.md` and in the manifest, so the resulting assertion failure is
  read as "the sample works", not as a conversion defect.
- **Not degenerate.** This is the check that catches a wrong strategy: graph
  degrees non-zero for a real share of nodes; group sizes > 1 wherever the task
  aggregates per key; join fan-out per key unchanged in distribution.
- **Fast enough**: time one pipeline, and pick the size knob from that
  measurement rather than from a row count that looks small. Rows are not the
  only cost driver — `playground-series-s6e7`'s pipelines stack LightGBM,
  XGBoost and CatBoost at `n_estimators=1000` plus a torch net over 5 folds, so
  60,000 rows (8.7% of the data) still ran past 110s wall and 98 CPU-minutes on
  64 cores. Target ~1 minute, and note the size you actually measured in the
  manifest. Much *faster* than expected usually means something is empty.

## Step 5 — prove it on a real pipeline

Cheap first — no LLM, no API key:

```bash
python -m pipeline_analyzer.runtime \
    --pipelines <dataset>/<run>/pipelines \
    --run-in <dataset>/sample --limit 2
```

A missing or empty `input/` is reported rather than discovered mid-sweep
(`tools/pipeline_analyzer/runtime.py:386-404`), so a clean start here means the
layout is right.

Then the real target:

```bash
python -m skrubify <dataset>/<run>/pipelines/x.py --engine stratum \
    --provider openai --model gpt-5.6-sol \
    --run-in <dataset>/sample --compare-source
```

**Read the result honestly.** A sample score is not comparable to a full-data
score, and nothing about the sample is evidence that the conversion reproduces
the full-data number. What must match is **candidate vs original on the same
sample** — the standard `mlevolve_run_2/skrubify_manual/versions/README.md`
used: bit-identical per-fold design matrix and an end-to-end score stable at
`0.7805021056023915`. Before believing a small delta, check the original's own
run-to-run spread; `tools/skrubify/README.md:287-291` documents LightGBM with
`n_jobs=-1` spanning 4e-3 between repeated runs of the *same* script.

## Pitfalls

- **Never** `nrows=` or a random row cap on a joined, grouped or graph task.
- Do not sample `test.csv` and `sample_submission.csv` out of sync with each
  other, or a submission-writing pipeline fails on row count rather than on
  anything real.
- Keep small lookup tables whole. Filtering a 355-row table saves nothing and
  breaks a hardcoded width.
- Hardlink images and other per-row files; do not copy.
- `--run-in` executes the **original** too, so the sample dir must tolerate two
  scripts running in it and they must not collide on written files
  (`tools/skrubify/README.md:293-298`).
- Give each concurrent worker its own working directory. Four pipelines each
  fitting with `n_jobs=-1` oversubscribed 24 cores badly enough to turn a 12s run
  into a >1100s timeout.
- Sizing is per *pipeline*, not per dataset. If the heaviest pipeline in a run is
  a big ensemble, either sample harder for it or raise `--run-timeout` (default
  1800s), rather than assuming a size that worked for the cheapest one.
- `**/input/**` is gitignored, so `sample/input/` is ignored automatically. The
  manifest is not — that is deliberate.

## Reference

- `ttt-task/make_input_sample.py` — the worked graph case, and the shape every
  generated script should follow.
- `playground-series-s6e7/make_sample.py` — the worked flat-table case:
  proportional stratification with an `n_splits` floor, test and
  sample_submission kept on the same ids, manifest written.
- `ttt-task/mlevolve_run_2/skrubify_manual/versions/README.md` — how a 20k sample
  was used to verify a conversion step by step.
- `<dataset>/DATA.md` / `<dataset>/get_data.sh` — how `input/` gets populated in
  the first place.
- `tools/pipeline_analyzer/README.md` — why `--sample-rows` is not this.
