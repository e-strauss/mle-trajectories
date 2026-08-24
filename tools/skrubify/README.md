# skrubify

Turn a plain pandas/scikit-learn ML script into a **valid skrub DataOps
pipeline**, with an LLM doing the translation and a validator gating the result.

The conversion target is the hand-written style already in this repo
(`tab_playground_dec_21/mle_star-run1/skrubify/init1.py`): one self-contained
file, taking no command-line arguments, that *builds* a plan when imported and
cross-validates it when run directly (the scoring block sits behind
`if __name__ == "__main__":`).

## Run

```bash
cd tools                       # the dir containing the skrubify/ package

# convert one script -> tab_playground_dec_21/mle_star-run1/skrubify/train0.py
python -m skrubify ../tab_playground_dec_21/mle_star-run1/train0.py \
       --provider openai --model gpt-5

# batch, explicit output dir
python -m skrubify ../nyc_taxi_fare/mlevolve-run1/pipelines/*.py \
       -o ../nyc_taxi_fare/mlevolve-run1/skrubify \
       --provider gemini --model gemini-2.5-pro --json-report run.json

# validate pipelines (no LLM, no API key, no dataset needed)
python -m skrubify --check ../tab_playground_dec_21/mle_star-run1/skrubify/init1.py

# convert, then RUN it and compare its score against the original's
python -m skrubify ../tab_playground_dec_21/mle_star-run4/train0.py \
       --provider openai --model gpt-5.6-sol \
       --run-in ../tab_playground_dec_21 --compare-source

# inspect exactly what would be sent, without calling anything
python -m skrubify ../path/to/script.py --print-prompt | less
```

Default output path is `<source_dir>/skrubify/<source_name>`, matching the
existing layout. `-o` takes a file (single source) or a directory.

## Install

```bash
uv pip install litellm          # only needed for the LLM call
```

API keys come from a dot file — `.env`, `.env.local` or `.skrubify.env`,
searched from the current directory up to the repo root, then inside the package
(`--env-file FILE` forces one, `$SKRUBIFY_ENV_FILE` also works). See
`.env.example`. Real environment variables always win over the dot file.

Any litellm provider works: `--provider {openai,anthropic,gemini,vertex_ai,azure,
openrouter,deepseek,mistral,groq,together_ai,xai,ollama}` combined with
`--model`, or a single fully-qualified `--model openrouter/qwen/qwen3-coder`.
`--temperature` is *not* sent unless you pass it (some models reject anything but
their default).

## How it works

1. **Prompt** (`prompt.py`) — one call, three knowledge sources: a system message
   holding the output contract and translation rules; the full skrub DataOps
   guide (`tools/skrub_dataops_summary.md`, override with `--guide`); and
   few-shot pairs from `examples/` (`NN_source.py` + `NN_skrub.py`, override with
   `--examples-dir` / `--n-examples`). ~16k tokens by default.
2. **Extract** — the longest fenced ```python block of the reply becomes the file.
3. **Validate** (`checks.py`, `validate.py`) — two layers, neither of which needs
   the dataset:
   - *static*: ~25 rules encoding the output contract and the guide's pitfalls
     (recorded read, `mark_as_X(cv=...)`, no `cv=` at scoring time, `.skb.apply`
     only for estimators, no leftover fold loop or `train_test_split`, no
     submission writing, `.skb.concat([...])`, …). Calls with nested parentheses
     are parsed brace-balanced, not by regex, so `make_grid_search(cv=KFold(3),
     fitted=True)` is read correctly.

     Each rule has a **scope**, which is what keeps it from deleting working
     code. Comments and docstrings are stripped first (prose about the conversion
     never trips a rule), and then:
     - `plan` — an error only inside the `config_context` block: `.copy()`,
       in-place `df["c"] = ...`, `for col in X.columns`.
     - `toplevel` — an error only at module/plan level, outside every def/class:
       a fold loop, `train_test_split`, `.iloc[train_idx]`, a hand-computed
       metric, `eval_set=`/`early_stopping_rounds=`. Inside a wrapper estimator
       all of these are legitimate (an inner validation split for early stopping
       is per-fit, not the outer CV), so there they only warn.
     - `all` — everywhere: the contract rules, including "no argparse".

     Outside its scope a forbid rule still reports, as a *warning* that never
     forces a repair round.

     One check is an AST pass rather than a pattern: fitting on a target derived
     from the `mark_as_y` node (`y - 1`, `np.log1p`, `astype`, …) without an
     `eval_mode()`-gated inverse on the predictions. That scores predictions in
     the wrong domain — a plan that builds perfectly and returns a plausible
     number (accuracy 0.0 for a label shift). Nothing else catches it.
   - *plan build*: the file is imported in a **subprocess** under
     `skrub.config_context(eager_data_ops=False)`, so the recorded read is never
     executed — the graph is built, `pred` is located, and the marks/CV/param-grid
     are read back off the graph. `--python` builds with a different interpreter;
     `--build-timeout` bounds a candidate that tries to train something.
   A `ModuleNotFoundError` for a third-party package (e.g. `catboost`) is
   reported as an *environment gap*, not a candidate defect: the build is left
   unjudged and the traceback is kept out of the repair feedback, so the model is
   never pushed into dropping the estimator or hiding the import inside `fit()`.
   Install the package (or point `--python` at an interpreter that has it).
4. **Run** (optional, `--run-in DIR`) — execute the pipeline with `DIR` as the
   working directory its relative paths resolve against (the one holding
   `./input`) and read back the score it prints; `--compare-source` runs the
   ORIGINAL script the same way and reports both scores and the delta. This is
   the only layer that catches **scoring-time** failures — a plan can build
   perfectly and still die inside `make_grid_search` — and the only one that
   verifies a conversion is faithful rather than merely valid.
5. **Repair** (`core.py`) — on failure the candidate plus the validator's report
   (static errors, the real traceback, structural problems) go back to the model,
   up to `--max-repairs` rounds (default 2). Exit code is non-zero if the final
   candidate still fails; `--json-report` records every attempt and
   `--keep-attempts` writes each one to `<stem>.attemptN.py`.

## Verified against real data

`mle_star-run4/train0.py` (RandomForest, 3-fold stratified CV, with the
original's exclusion of classes holding fewer than `n_splits` samples) converted
in one shot and scored **bit-identically** to the original on a 100k-row sample
of the Dec-2021 playground data — `0.943480001188684` both ways, including the
rare-class exclusion path (class 5 has a single row).

Getting there surfaced a **skrub 0.8.0 bug** worth knowing about:
`mark_as_X(cv=...)` without `split_kwargs` stores `None`, and the splitter
wrapper later evaluates `**None`, so every such plan builds cleanly and then dies
at scoring time with `TypeError: ... argument after ** must be a mapping, not
NoneType`. Fixed upstream in skrub 0.10 (`split_kwargs or {}`). Since this
workspace pins 0.8.0 through stratum, the contract and the checks now require an
explicit `split_kwargs={}`, which is valid on every version.

### Ablation scripts become one fused-choice plan

A script that scores SEVERAL variants in one run (an ablation study, a model
comparison) is not converted into several pipelines: the variants become one
named `skrub.choose_from` fused with `.as_data_op()`, so a single grid search
scores all of them and `search.results_` has one row per variant. The contract
requires exactly the variants the original ran — two independent choices over 2
estimator sizes and 2 feature sets would score 4 cells where the original scored
3 — and discrete `choose_from([...])` only (`choose_float`/`choose_int`/
`make_randomized_search` are rejected by the checks).

Verified on `mle_star-run4/ablation_0.py` (baseline / `n_estimators=50` /
no-`Soil_Type`), one shot, all three variants matching the original to 12
decimals:

```
                                                   variant  mean_test_score
0                             Baseline Solution (Original)   0.943480001189
1  Ablation 1: RandomForestClassifier with n_estimators=50   0.942739992289
2               Ablation 2: Exclude all Soil_Type features   0.928539999686
```

The feature ablation came out as `X.skb.drop(s.glob("*Soil_Type*"))` rather than
the original's list comprehension over `X.columns` (which a plan cannot express),
and the rare-class filter as
`data[raw_target.groupby(raw_target).transform("size") >= 3]` — a recorded
reformulation of `value_counts()` + `isin` + index-based `drop`.

## Observed behaviour (gpt-5.5 / gpt-5.6-sol, real runs)

- `mle_star-run1/train0_1.py` (LightGBM+XGBoost soft-vote with per-fold
  dummy-class augmentation) — one shot, 0 errors, ~20k tokens. The augmentation
  and the probability averaging moved into a `ClassifierMixin, BaseEstimator`
  wrapper (the only leakage-free place for per-fold logic), the raw 1-7 target
  marked and shifted inside the plan, predictions shifted back gated on
  `skrub.eval_mode()`.
- `mlevolve_run1/pipelines/0001_*.py` (553 lines: ordinal maps, group z-scores,
  target encoding, LightGBM/XGBoost/CatBoost + a torch ResNet with focal loss,
  weight optimisation) — a ~100-node plan that builds, in one shot once the rule
  scoping above was in place, in two with an early over-strict version.
- `mle_star-run4/train0_1.py` (two forests, different per-fold training sets,
  pooled out-of-fold soft vote) — one shot. It solved the "model 2 also trains on
  the rare-class rows" requirement with a custom `BaseCrossValidator` that
  appends those rows to every fold's train partition and never to validation,
  plus a marker column in `X` that masks them out of model 1's `fit`. Score
  0.943729998 vs the original's 0.94373: the residual is structural, since skrub
  averages per-fold accuracies where the original pools all OOF rows into one.
- Cost per conversion is roughly $0.13-0.55 at these prompt sizes.

## Adding examples

Drop a `NN_source.py` / `NN_skrub.py` pair into `examples/` (or a separate
directory passed with `--examples-dir`); they are picked up in filename order.
Keep them validated:

```bash
python -m skrubify --check skrubify/examples/*_skrub.py
```

- `01_*` — a plain `StratifiedKFold` fold loop (LightGBM) → the minimal plan.
- `02_*` — a feature-engineering-heavy script (chunked read, in-place column
  mutation, submission writing, prediction clipping) → recorded `.assign` ops, a
  `SimpleImputer` replacing a leaky full-table median, and `skrub.eval_mode()`
  gating on the prediction post-processing.

## Notes

- Conversion is deliberately **faithful, not creative**: same model, same
  hyperparameters, same features, same metric, same folds. It does not add
  `choose_from` exploration — that is the next step *after* a pipeline is
  skrubified.
- Two things a conversion cannot always keep literal, and the prompt says so
  explicitly: constants the original computed at runtime (`num_class=len(
  y.unique())`) must become literals, and a leaky full-table statistic becomes an
  in-plan estimator (it changes the score slightly — for the better).
- `--check` doubles as a linter for hand-written pipelines. It flags e.g.
  `pred.skb.draw_graph().open()` in `init1.py`, which blocks on a browser wait.
- Target is skrub (`skrub >= 0.8`). The stratum variant
  (`import stratum as st`) is out of scope for now; the prompt, examples and
  checks would need a second variant.
