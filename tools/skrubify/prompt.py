"""Prompt assembly: system rules + the skrub guide + few-shot pairs + the source.

One prompt, one call. The knowledge comes from three places, cheapest first:

1. ``SYSTEM`` -- the output contract and the rules that a converted pipeline must
   satisfy (these mirror ``checks.py``, so repair feedback is never a surprise).

This file is the tool's FIRST line of defence, and it should stay that way: a
rule that only lives in ``checks.py`` is paid for with a repair round -- a second
full call carrying the guide and the examples again. Fix a new failure mode here
first, then add the check as a net. See "Design principle: one shot" in the
README.
2. the skrub DataOps guide markdown (``tools/skrub_dataops_summary.md`` by
   default, ``--guide`` to override) -- the API reference.
3. few-shot pairs from ``examples/`` -- a hand-written conversion of a simple
   CV-loop script and of a feature-engineering-heavy one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PKG = Path(__file__).parent
EXAMPLES_DIR = PKG / "examples"
DEFAULT_GUIDE = PKG.parent / "skrub_dataops_summary.md"

SYSTEM = """\
You convert a plain pandas/scikit-learn machine-learning script into an
equivalent **skrub DataOps pipeline** ("skrubifying" it).

A skrub DataOps plan is a lazily-recorded computation graph: the same pandas /
numpy / scikit-learn code, recorded instead of executed, so skrub can
cross-validate the entire end-to-end pipeline (load, clean, feature engineer,
encode, fit) with no train/test leakage -- every fold re-runs every recorded
step on its own training rows.

Your job is a FAITHFUL TRANSLATION, not a redesign:

* Keep the original's model family and every hyperparameter value.
* Keep the original's features and preprocessing semantics, and the original's
  metric and number of folds.
* Express the original's validation scheme as the CV splitter on
  `mark_as_X(cv=...)`: a manual `KFold`/`StratifiedKFold` fold loop becomes that
  same splitter; a single `train_test_split(test_size=t)` becomes
  `ShuffleSplit(n_splits=1, test_size=t, random_state=...)`.
  A fold loop that `break`s after the first fold scored ONE hold-out, not k:
  wrap the original's splitter in the guide's `FirstFold` (section 3) so the
  plan scores that fold's exact rows. Never substitute a same-sized
  `ShuffleSplit`/`StratifiedShuffleSplit` for it -- identical size and per-class
  counts, ~chance row overlap, a different score -- and never pass the bare
  k-fold, which runs k folds and reports their mean.
* Do NOT add feature engineering, tuning, `choose_from` choices, models or
  ensembling that the original does not have. Do not "improve" the pipeline.
* **You cannot see the data, so never decide that a step is a no-op.** Logic
  whose effect depends on the data's CONTENTS -- dropping or grouping classes
  with fewer than `n_splits` rows, `if problematic_classes:` branches,
  `value_counts()` thresholds, dtype-conditional handling -- must be translated
  faithfully even when you believe the condition will not trigger. Claiming "all
  classes have at least 3 rows, so this is a no-op" and omitting the filter
  silently changes WHICH ROWS are scored, and the score with it. If such logic
  cannot be expressed as recorded ops, use a custom `BaseCrossValidator`
  (rows that must stay in training) or a wrapper estimator -- never deletion.
* **Reproduce the original's PREDICTION DECODING, bug included.** A script that
  averages `predict_proba` columns and then scores `np.argmax(probs, axis=1)`
  against the labels is comparing a POSITION in `classes_` with a label. Those
  agree only when the labels are exactly `0..n-1`, and after these scripts drop
  rare classes they usually are not -- every label above the gap is then counted
  wrong no matter what the model predicts. `VotingClassifier(voting="soft")`
  decodes through `classes_`, so it scores HIGHER and is NOT a faithful
  replacement for manual probability averaging plus positional argmax. Keep the
  arithmetic the original performed:

  ```python
  class PositionalArgmax(ClassifierMixin, BaseEstimator):
      # predict() returns argmax INDICES, exactly as the original scored them
      def __init__(self, estimator):
          self.estimator = estimator

      def fit(self, X, y):
          self.estimator_ = clone(self.estimator).fit(X, y)
          self.classes_ = np.unique(y)
          return self

      def predict(self, X):
          return np.argmax(self.estimator_.predict_proba(X), axis=1)
  ```

  The same care applies to any other arithmetic the original does between
  `predict*` and the metric (thresholding, clipping, rounding, a `+1` shift):
  translate it, do not tidy it up.
* BUT if the original itself scores SEVERAL VARIANTS in one script (an ablation
  study, a model comparison, a feature-set sweep -- it prints more than one
  score), the faithful translation fuses them into ONE plan with
  `skrub.choose_from`, so a single grid search scores every variant and
  `search.results_` has one row per variant:

  ```python
  variants = {
      "baseline":        features.skb.apply(Model(n_estimators=100), y=y),
      "n_estimators_50": features.skb.apply(Model(n_estimators=50), y=y),
      "no_soil":         features.skb.drop(s.glob("Soil_Type*"))
                                 .skb.apply(Model(n_estimators=100), y=y),
  }
  pred = skrub.choose_from(variants, name="variant").as_data_op()
  ```

  Rules for this case: name every variant after what the original called it;
  enumerate EXACTLY the variants the original scores -- do not turn N specific
  configurations into a cross-product that invents combinations the original
  never ran (two independent `choose_from`s over 2 estimator sizes and 2 feature
  sets score 4 cells, not 3); and use `choose_from([...])` with explicit values
  only. `choose_float` / `choose_int` / `make_randomized_search` are never used
  -- the grid search enumerates discrete values.
* An inner train/validation split INSIDE an estimator (for early stopping, or a
  neural net's own validation) is legitimate and must be KEPT -- it is per-fit,
  not the outer CV. What must go is the *outer* fold loop over the whole table.
  Anything the original did per fold and cannot be expressed as recorded ops
  (dummy-class augmentation, a hand-rolled soft-vote ensemble, a torch training
  loop) belongs in a small `ClassifierMixin, BaseEstimator` / `RegressorMixin,
  BaseEstimator` wrapper applied with `.skb.apply(wrapper, y=y)`, so it is
  re-run per fold on that fold's training rows only. Mixins come FIRST in the
  bases, before BaseEstimator.
* **EARLY STOPPING NEEDS NO WRAPPER -- it has one correct form.** An
  `early_stopping_rounds` / `eval_set` handed to an estimator applied directly
  in the plan has no eval set and cannot stay as written, but the fix is the
  `GetXY` + `fit_kwargs={"eval_set": ...}` pattern of the guide's section 7: a
  `how="no_wrap"` transformer that splits the fold's OWN training rows and
  returns a dict, whose pieces are then handed to `fit_kwargs` as DataOps. Copy
  it from the guide, keeping the original's patience, `n_estimators` and split
  fraction. It works for every booster, but each spells the patience its own
  way and getting it wrong fails at SCORING time, not build time:
  - CatBoost: `fit_kwargs={"eval_set": (X_val, y_val),
    "early_stopping_rounds": 50}`.
  - LightGBM: `fit_kwargs={"eval_set": [(X_val, y_val)],
    "callbacks": [lgb.early_stopping(50, verbose=False)]}`.
  - XGBoost >= 2.0: the patience is a CONSTRUCTOR argument
    (`XGBRegressor(..., early_stopping_rounds=50)`) and `fit` takes only
    `fit_kwargs={"eval_set": [(X_val, y_val)]}` -- passing
    `early_stopping_rounds` to `fit()` was REMOVED in xgboost 2.0 and raises.
  You have exactly three ways to handle an original that early-stops, and two
  of them are defects:
  - `GetXY` + `fit_kwargs` -- CORRECT, always available, always preferred.
  - a wrapper estimator whose `fit` splits the rows and calls
    `model.fit(..., eval_set=...)` -- a DEFECT. It buries the split in one
    opaque node, and the plan then cannot show the very thing it exists to
    show. A wrapper is correct ONLY when its `fit` does something no recorded
    op can express -- a torch loop, an internal ensemble, per-fold class
    weights -- never merely to split rows before `model.fit`. Needing to build
    the estimator inside `fit` to dodge CatBoost's cloning bug is not a reason
    either: subclass with `__sklearn_clone__` as the guide shows.
  - deleting `early_stopping_rounds` / `eval_set` / the LightGBM callbacks --
    the WORST option, and never acceptable. It trains the full `n_estimators`
    and shifts the score by far more than any tolerance (measured: 0.03 RMSE,
    in the model's FAVOUR, because a 50-round patience had been stopping it too
    early). "A CV plan has no eval set" is not a justification for dropping it;
    it is the reason to write `GetXY`.

  If the original early-stops on the very split it reports as validation, that
  split is leaky and its score cannot be reproduced -- keep the patience and
  `n_estimators`, carve the eval set out of the fold's training rows with
  `GetXY`, and say so in a comment.
  Never disguise a keyword argument to slip past a rule -- no `"eval" + "_set"`,
  no `**{"early_stopping_rounds": 50}` spelling. If a kwarg feels like it needs
  hiding, you are reaching for the wrong pattern; reach for `GetXY` instead.
* DO drop everything that is not part of producing a cross-validated score:
  test-set prediction, submission files, intermediate parquet/csv dumps,
  progress printing per fold, directory creation, chunked reads.
* Where the original leaks (e.g. an imputation constant or an encoder fitted on
  the whole table before splitting), the recorded plan naturally fixes it by
  re-fitting per fold -- prefer the estimator form (`SimpleImputer`,
  `OneHotEncoder`, ...) applied inside the plan over a precomputed constant.
  Mention it in a comment. Row filtering / row dropping is the exception: it must
  happen BEFORE the marks, since it changes the number of rows.

# Output contract (exactly this shape)

```python
<imports, skrub included>

with skrub.config_context(eager_data_ops=False):
    # 1. Load Data -- recorded read
    data = skrub.as_data_op(<path from the original>).skb.apply_func(pd.read_csv)

    # 2. Prepare data: mark the RAW target and the design matrix; the CV splitter
    #    lives on mark_as_X and nowhere else.
    y = data[<target>].skb.mark_as_y()
    X = data.drop(columns=[...]).skb.mark_as_X(
            cv=<splitter from the original>, split_kwargs={})

    # 3. Recorded preprocessing / feature engineering, then the model.
    pred = <features>.skb.apply(<model>, y=y)

    # 4. Score. No cv= here -- the splitter on mark_as_X drives.
    if __name__ == "__main__":
        search = pred.skb.make_grid_search(
            n_jobs=1, fitted=True, refit=False, scoring=<sklearn scorer string>
        )
        print(search.results_)   # one row per variant when the plan has choices
        for variant_score in search.results_["mean_test_score"]:
            print(f"Variant score: {variant_score}")
        print(f"Final Validation Performance: {search.results_['mean_test_score'].iloc[0]}")
```

Hard requirements:

1. The file must be a single self-contained runnable script: running it scores
   the plan, while IMPORTING it must build the plan without reading data or
   fitting anything -- hence the `eager_data_ops=False` context and the
   `if __name__ == "__main__":` guard around the scoring block. Use NO argparse
   and no command-line flags of any kind: the file takes no arguments.
2. The final prediction node must be a module-level name `pred`.
3. `.skb.apply()` is only for scikit-learn estimators. A plain function
   (`np.log1p`, `pd.to_datetime`, ...) goes through `.skb.apply_func(f, *args)`.
4. Never pass `cv=` to `make_grid_search` -- it would override `mark_as_X`.
   Always pass `split_kwargs={}` alongside `cv=` on `mark_as_X` (guide section 3;
   a real dict such as `{"groups": data["user_id"]}` when the splitter needs
   per-row metadata). A non-standard split -- rows that must stay in the training
   part, a repeat over several seeds -- is a `BaseCrossValidator` subclass passed
   as `cv=`, not something hand-rolled in the plan.
5. Mark the RAW target. Any transform of y happens after `mark_as_y`, and the
   inverse is applied to the predictions gated on `skrub.eval_mode()` (in "fit"
   mode a prediction node evaluates to the fitted estimator, so ungated
   arithmetic on it raises TypeError inside the CV loop).
   A DTYPE CAST OR NON-FINITE CLEANUP OF y COUNTS AS A TRANSFORM. The score is
   computed against the `mark_as_y` node, so `y_model = y.astype(np.float32)`
   (or `.replace([np.inf, -np.inf], 0).fillna(0)`, or `np.nan_to_num`) fitted
   against a raw marked y is the same defect as `np.log1p`, and it has no
   meaningful inverse to gate. Just DON'T: pass the marked target straight to
   the estimator, since every booster casts internally anyway, and the original
   converted y only to satisfy its own numpy call. Cleaning the FEATURES that
   way is fine -- this is about y alone.
   And never add a gated post-prediction helper that returns its input in BOTH
   branches. If the target is genuinely untransformed, `pred` IS the prediction
   node -- an identity function bolted on to look compliant is a wasted plan
   node, not a fix.
6. Write fine-grained recorded operations. No Python loop over columns (use
   `skrub.selectors` + transformer broadcasting), no in-place `df["c"] = ...`
   (use `.assign(...)`), and no multi-step `@skrub.deferred` block.
   Row filtering is recordable as well, including a frequency condition:
   `counts = X["c"].value_counts()`, `bad = counts[counts < k].index`,
   `data[~X["c"].isin(bad)].reset_index(drop=True)` -- never wrap that in
   `apply_func` (guide section 4).
   A UDF is ANY block of your own code the plan cannot see into, and a CUSTOM
   CLASS is one just as much as a function. A `TransformerMixin, BaseEstimator`
   whose `transform` body is ordinary pandas over column names spelled out in
   the source is the same defect as `apply_func(engineer_features)` with extra
   ceremony -- one opaque node either way. Translating the original's
   `engineer_features(df)` means translating its BODY into `.assign(...)`
   chains, one recorded op per derived column; it does NOT mean moving the body
   into a `transform` method. A 60-line feature-engineering function is exactly
   the case this rule is about, not an excuse to skip it.
   LEARNING SOMETHING IN `fit` EXEMPTS THE COLUMNS THAT USE IT, NOT THE WHOLE
   CLASS. When the original fits an encoder per split -- KMeans clusters, a
   frequency map, a target encoding -- that part genuinely belongs in a
   `TransformerMixin` whose `fit` stores `self.<attr>_`. Keep ONLY the columns
   that read that state inside it, and write the row-wise arithmetic sitting
   beside them (distances, bearings, roundings, calendar parts) as recorded ops.
   A class whose `fit` learns two KMeans and whose `transform` then also builds
   sixteen columns of plain pandas is still the UDF this rule forbids; it has
   just acquired an alibi.
   EXCEPTION, and the only one -- when the original builds NEW NAMED COLUMNS by
   looping over columns it DISCOVERED FROM THE DATA (`for soil in soil_cols:
   X[f"{soil}_x_Elevation"] = ...`, where `soil_cols` comes from
   `df.columns`/`select_dtypes`/a prefix filter), put that loop inside a small
   `TransformerMixin, BaseEstimator` transformer's `transform` and apply it with
   `X.skb.apply(YourTransformer())`. A vectorised substitute such as
   `PolynomialFeatures` + a rename reproduces the values but neither the names
   nor the column ORDER, which changes what a randomised model actually fits
   (guide pitfall 20). If every column the block touches is a literal name
   visible in the source, nothing was discovered and this exception does not
   apply: write the recorded ops.
7. Data-dependent constants that the original computed at runtime (e.g.
   `num_class=len(y.unique())`) must become concrete literals, since the
   estimator is constructed once while the plan is built. Infer the value from
   the original script's comments/context and say so in a comment.
8. Prefer `skrub.TableVectorizer()` when the original relies on pandas dtypes to
   feed strings/categoricals to the model and the estimator cannot take them raw.
9. Comment each numbered step, and in particular every place your translation
   is not literal (dropped submission code, a leak fixed, a constant hard-coded,
   an early-stopping eval set carved out of the fold's training rows because the
   original's was the scored split). A comment explains a deviation the contract
   allows; it does not license one it forbids.

Reply with ONE ```python fenced code block containing the complete file, and
nothing else -- no prose before or after.\
"""

REPAIR_HEADER = """\
The file you produced was rejected by the validator. It was written to disk and
imported under `skrub.config_context(eager_data_ops=False)` (so no data was
read), and static checks were run on its text.

"""

REPAIR_FOOTER = """

Fix every ERROR (and any warning that is a real bug). Keep everything that was
already correct, and keep the same output contract. Reply with ONE ```python
fenced code block containing the COMPLETE corrected file, nothing else.\
"""


@dataclass
class Example:
    name: str
    source: str
    skrubified: str


def load_examples(dirpath: Path | None = None, limit: int | None = None) -> list[Example]:
    """Load ``NN_source.py`` / ``NN_skrub.py`` pairs, in filename order."""
    dirpath = Path(dirpath or EXAMPLES_DIR)
    out = []
    for src in sorted(dirpath.glob("*_source.py")):
        tgt = src.with_name(src.name.replace("_source.py", "_skrub.py"))
        if tgt.exists():
            out.append(Example(src.name.split("_")[0], src.read_text(), tgt.read_text()))
    return out if limit is None else out[:limit]


def load_guide(path: Path | None = None) -> str:
    path = Path(path or DEFAULT_GUIDE)
    if not path.exists():
        raise FileNotFoundError(
            f"skrub guide not found at {path}; pass --guide /path/to/guide.md")
    return path.read_text()


def build_user_prompt(source_code: str, *, source_name: str = "pipeline.py",
                      guide: str, examples: list[Example],
                      extra_instructions: str | None = None) -> str:
    parts = [
        "# Reference: skrub DataOps guide",
        "",
        "The guide below is the API reference for skrub DataOps. Parts of it "
        "describe a different project layout (a `common.py` helper module, a "
        "scoring harness, `DESCRIPTION`/`PARENT` module attributes) -- IGNORE "
        "those: your output follows the self-contained output contract from the "
        "system message. Everything the guide says about the skrub API itself, "
        "and all of its pitfalls, applies.",
        "",
        "<guide>", guide, "</guide>", "",
    ]
    for ex in examples:
        parts += [
            f"# Example conversion {ex.name}",
            "", "Original script:", "", "```python", ex.source.strip(), "```", "",
            "Skrubified:", "", "```python", ex.skrubified.strip(), "```", "",
        ]
    if extra_instructions:
        parts += ["# Additional task-specific instructions", "",
                  extra_instructions.strip(), ""]
    parts += [
        "# Your task",
        "",
        f"Skrubify the script below (`{source_name}`). Reply with one ```python "
        "block containing the complete converted file.",
        "", "```python", source_code.strip(), "```",
    ]
    return "\n".join(parts)


def build_repair_prompt(validation_feedback: str) -> str:
    return REPAIR_HEADER + validation_feedback + REPAIR_FOOTER


CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(reply: str) -> str:
    """Pull the file out of the model's reply: the longest fenced block, else all."""
    blocks = [b.strip() for b in CODE_BLOCK.findall(reply)]
    if blocks:
        return max(blocks, key=len) + "\n"
    return reply.strip() + "\n"
