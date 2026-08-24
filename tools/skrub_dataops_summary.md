# Skrub DataOps — Guide for Building & Scoring ML Pipelines from Scratch

This document gives you everything you need to take a raw tabular dataset (e.g. a
`train.csv`) and produce a **valid skrub DataOps pipeline that can be scored by
cross-validation**.

> **Your goal as the agent using this guide:** given a training table, build an
> end-to-end skrub DataOps plan and obtain a cross-validated score for it. You
> will be asked to try many different candidate pipelines; each one only needs to
> be **scored**. You do NOT predict on a test set, you do NOT write a submission,
> and you do NOT need a fitted model artifact. The single deliverable is a score.

The scoring is always done the same way:

```python
# the CV splitter is declared on mark_as_X (see section 3), NOT passed here
search = pred.skb.make_grid_search(n_jobs=1, fitted=True, refit=False,
                                   scoring="r2")   # any sklearn scorer string
print(search.results_)        # a DataFrame; row 0 is the best score
```
- `fitted=True` runs the cross-validation immediately on the data captured in the
  variables.
- `refit=False` skips the (expensive, unneeded) final refit on all data — we only
  want the CV score, not a model.
- `search.results_` is a pandas DataFrame **sorted best-first**. Read the score
  from column `mean_test_score`.
- **No `cv=` is passed**: the splitter comes from `mark_as_X(cv=...)` in the plan
  (section 3). A `cv=` passed here would *override* it, which is never what you
  want in this project.

> **In this project the `make_grid_search` call is executed by the ml-score
> harness, not by your pipeline file.** A pipeline file only *defines* the plan
> (module-level `pred`, `DESCRIPTION`, optional `PARENT`); the harness owns the
> locked scorer and results.json. The **CV splitter is part of your plan**
> (declared on `mark_as_X`, centrally via `common.py`); the harness passes no
> `cv=` of its own so your plan's splitter drives, falling back to a default
> `KFold` only if the plan declares none. Sections 9–10 still matter: they
> explain what the harness runs and why choices must stay discrete.

Target version: **skrub >= 0.9.0**.

---

## 1. The mental model

A **DataOps plan** is a lazily-built computation graph. You write ordinary
pandas / numpy / scikit-learn code, but each operation is *recorded* into a graph
instead of being executed once. From that single graph skrub can cross-validate
the entire end-to-end pipeline (loading, cleaning, feature engineering, encoding,
model) without any train/test leakage — every fold re-runs all the recorded
preprocessing on its own training rows.

- `skrub.var("name", value)` introduces an **input node**; operations on it are
  recorded. `skrub.as_data_op(value)` wraps a constant (e.g. a file path) as a
  node — applying `pd.read_csv` to it records the file read itself in the plan.
- `X.skb.mark_as_X()` / `y.skb.mark_as_y()` mark the **design matrix** and the
  **target**. These are the points at which cross-validation splits the data.
- `.skb.apply(estimator)` records a scikit-learn estimator/transformer step.
- The **last DataOp** (the prediction node) is the handle you call
  `.skb.make_grid_search(...)` on to get the score.

The `.skb` namespace holds all skrub-specific methods. Anything *not* under `.skb`
(e.g. `X.drop(...)`, `X["col"]`, `X.assign(...)`, `df.merge(...)`) is just the
recorded version of the underlying pandas API.

**Previews:** every DataOp carries a preview computed on the data you passed to
`skrub.var`, so you can inspect intermediate results while building.

---

## 2. The canonical skeleton (build → score)

```python
import skrub
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

# 1. Record the CSV read as the first step of the plan: wrap the *path* as a
#    constant DataOp, then apply pd.read_csv to it.
data = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)

# 2. (optional) subsample for fast previews while developing.
#    IGNORED during scoring unless keep_subsampling=True, so it is safe.
data = data.skb.subsample(n=1000)

# 3. Separate target and features, and MARK them. Always mark the RAW target —
#    never transform y before mark_as_y (see section 8). The CV splitter is set
#    HERE on mark_as_X (section 3), the only place it can live.
y = data["target"].skb.mark_as_y()
X = data.drop(columns=["target", "Id"]).skb.mark_as_X(
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        split_kwargs={})            # REQUIRED on skrub < 0.10 -- see section 3

# 4. Preprocess / feature engineer using recorded operations (sections 4-6).

# 5. Encode everything to a clean numeric matrix (section 5).
X_vec = X.skb.apply(skrub.TableVectorizer())

# 6. Apply the final model. y= makes it supervised.
pred = X_vec.skb.apply(HistGradientBoostingRegressor(), y=y)

# 7. Score the whole plan by cross-validation. THIS IS THE DELIVERABLE.
#    No cv= here — the splitter from mark_as_X drives.
search = pred.skb.make_grid_search(n_jobs=1, fitted=True, refit=False,
                                   scoring="r2")
print(search.results_)
print("CV score:", search.results_["mean_test_score"].iloc[0])
```

### Why the order matters
- **Mark `X` as early as possible**, right after separating it from the target.
  Everything *downstream* of `mark_as_X()` becomes part of the pipeline and is
  re-applied automatically to each CV fold. Do all real preprocessing *after* the
  mark.
- `mark_as_y()` is applied to the target only.

---

## 3. Variables and marking

```python
skrub.var("name", value)   # named input node, with a preview value
skrub.X(value)             # shorthand: a var already marked as X
skrub.y(value)             # shorthand: a var already marked as y
skrub.as_data_op(value)    # wrap a constant value as a DataOp
```

**Reading files:** start the plan from the file *path* wrapped as a constant
DataOp and record the read itself, instead of an eager `pd.read_csv` outside
the plan:

```python
data = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)
```

```python
X = something.skb.mark_as_X()
y = something.skb.mark_as_y()
```
### Cross-validation lives on `mark_as_X` — this is the only way to set it

`mark_as_X` carries the cross-validation splitter for the whole plan:
```python
from sklearn.model_selection import GroupKFold
X = data.drop(columns="target").skb.mark_as_X(cv=GroupKFold(5),
                                              split_kwargs={"groups": data["user_id"]})
```
**In this project you always set the CV here, never by passing `cv=` to
`make_grid_search`.** Two facts make this the only correct approach:

1. **Priority is the reverse of what you might expect.** When a `cv=` is passed
   to `make_grid_search`, skrub uses *that* and ignores
   `mark_as_X(cv=...)`. The `mark_as_X` splitter is honoured *only when no `cv=`
   is passed*. The ml-score harness therefore passes **no `cv=`** when your plan
   declares one (so it drives), and falls back to a default `KFold` only when
   your plan declares none.
2. **`split_kwargs` has no other home.** Per-row split metadata — e.g. the
   `groups` for `GroupKFold`, or a `date_id` array for a custom time splitter —
   can be supplied **only** through `mark_as_X(split_kwargs=...)`. There is no
   channel to deliver `groups` to a `cv=` passed at scoring time (skrub calls
   `search.fit(X, y)` with no `groups`), so a `GroupKFold` passed that way raises
   `ValueError: The 'groups' parameter should not be None`. Because
   `split_kwargs` can reference a data column (`data["user_id"]`), the groups are
   recomputed per fold from the recorded plan — no leakage.

The CV is set **centrally in `common.py`** (`make_cv()` + `load_xy`, which calls
`mark_as_X(cv=make_cv(), split_kwargs=...)`), so every pipeline in the workspace
splits identically and scores stay comparable. Keep it fixed once the first
pipeline is scored — the harness records the effective CV in `workspace.json`
and warns loudly if a later pipeline's CV differs. For a grouped/time split,
edit `make_cv()` once and pass `groups="col"` to `load_xy`.

> Splitters that need **no** per-row metadata (`KFold`, `StratifiedKFold`,
> `TimeSeriesSplit` on row order) are *static* — they could in principle be
> passed at scoring time. We still set them on `mark_as_X` so there is one
> uniform, harness-compatible convention for every task.

### Always pass `split_kwargs` together with `cv` (skrub < 0.10)

**On skrub 0.8/0.9, `mark_as_X(cv=...)` without `split_kwargs` builds a plan that
crashes when it is scored** (verified on 0.8.0):

```
TypeError: sklearn.model_selection._split._BaseKFold.get_n_splits()
           argument after ** must be a mapping, not NoneType
```

`mark_as_X` stores the missing `split_kwargs` as `None`, and skrub's internal
`_Splitter` then evaluates `self.splitter.split(X, y, **self.split_kwargs)`.
Fixed upstream in skrub 0.10 (`self.split_kwargs = split_kwargs or {}`). Pass an
explicit empty dict when the splitter needs no per-row metadata — it is valid on
every version, so this is the pattern to use unconditionally:

```python
X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        split_kwargs={})
```

Note *when* this fails: the plan builds, `describe_steps()` works, previews work
— it only raises inside `make_grid_search` / any scoring entry point that uses
the plan's splitter. Building a plan is therefore **not** evidence that it can be
scored (section 11).

### Custom splitters: any `BaseCrossValidator` works

`cv=` is not limited to scikit-learn's splitters. A plain
`sklearn.model_selection.BaseCrossValidator` subclass is accepted and driven per
fold like any other (verified), which is how a non-standard split gets into a
plan. Useful shapes:

- **rows that must always be in TRAIN, never in test** — e.g. a class with too
  few samples for `StratifiedKFold`, which the eager version of the script
  dropped or hand-appended per fold:
  ```python
  class TrainOnlyRareClasses(BaseCrossValidator):
      def __init__(self, n_splits=3, random_state=42):
          self.n_splits, self.random_state = n_splits, random_state

      def get_n_splits(self, X=None, y=None, groups=None):
          return self.n_splits

      def split(self, X, y, groups=None):
          y = np.asarray(y).reshape(-1)
          classes, counts = np.unique(y, return_counts=True)
          rare = np.isin(y, classes[counts < self.n_splits])
          rare_idx, ok_idx = np.flatnonzero(rare), np.flatnonzero(~rare)
          inner = StratifiedKFold(self.n_splits, shuffle=True,
                                  random_state=self.random_state)
          for tr, te in inner.split(np.zeros((len(ok_idx), 1)), y[ok_idx]):
              yield np.concatenate([ok_idx[tr], rare_idx]), ok_idx[te]
  ```
  Keep those rows **in** `X` and let the splitter place them, rather than
  filtering them out before `mark_as_X`.
- **averaging over several seeds** — a script that repeats its CV with 3 seeds
  and averages becomes one splitter yielding all 3×k folds (`mean_test_score` is
  then the mean over every fold, which is what the script computed).
- **a fixed hold-out** — `ShuffleSplit(n_splits=1, test_size=t, random_state=r)`
  reproduces a single `train_test_split(test_size=t, random_state=r)`.

**Multiple input tables:** give each its own variable and record the join with the
pandas API.
```python
data   = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)
labels = skrub.as_data_op("labels.csv").skb.apply_func(pd.read_csv)
df = data.merge(labels, on="ID", how="inner")
y = df["target"].skb.mark_as_y()
X = df.drop(columns=["target"]).skb.mark_as_X()
```

---

## 4. Recording ordinary dataframe operations

Inside the plan you use the normal pandas API and skrub records it.

- **Column access / slicing / arithmetic** is recorded transparently:
  ```python
  y  = data["price"].skb.mark_as_y()
  X2 = X.assign(area=X["w"] * X["h"])     # use assign, not X["a"] = ...
  X3 = X2.drop(columns=["w", "h"])
  ```
- **Prefer `.assign(...)`** to create new columns rather than in-place `X["new"] = ...`.
- **`.skb.drop` / `.skb.select` vs pandas `.drop` / `[]`:** the `.skb` versions
  *freeze the exact column list at fit time* and reapply it on every fold, which
  is safer. Prefer them for feature-subset selection.

### Applying a plain function — `apply_func` / `deferred`
For a generic function like `np.log1p`, `pd.to_datetime`, `np.sin` use
`apply_func` (NOT `.skb.apply`, which is only for scikit-learn estimators):

```python
import numpy as np
y_log = y.skb.apply_func(np.log1p)   # transform the target (AFTER mark_as_y;
                                     # invert at predict time — section 8)
X_dt  = X.assign(dt=X["datetime"].skb.apply_func(pd.to_datetime))
X_sin = X.assign(s=(X["f3"].skb.apply_func(np.sin)) * 2.0)
```
`X.skb.apply_func(f, *a, **k)` == `skrub.deferred(f)(X, *a, **k)`. Use
`skrub.deferred` to record any custom multi-arg function:
```python
@skrub.deferred
def combine(a, b):
    return a.merge(b, on="id")
merged = combine(table_a, table_b)
```

### Prefer fine-grained recorded steps over one opaque `deferred` block

A `skrub.deferred` function becomes a **single node** in the DAG: a multi-step
feature-engineering function shows up in `describe_steps()` / `full_report()`
as one opaque `Call 'add_feats'`, with no per-operation previews and nothing to
inspect inside it. The same logic written as recorded operations exposes every
step in the graph. Reserve `skrub.deferred` / `apply_func` for a *single*
operation that genuinely cannot be expressed as recorded ops — e.g. the
function call itself (`pd.to_datetime`) or the eval-mode-gated prediction
post-processing of section 8.

```python
# AVOID: ~10 operations hidden inside one opaque DAG node
@skrub.deferred
def add_feats(df):
    d = pd.to_datetime(df["Date of Transfer"])
    return df.assign(year=d.dt.year, dow=d.dt.dayofweek,
                     dist_pt=df["District"] + "|" + df["Property Type"])
X = add_feats(X)

# PREFER: the same logic as fine-grained recorded steps
date = X["Date of Transfer"].skb.apply_func(pd.to_datetime)
X = X.assign(
    year=date.dt.year,            # attribute access (.dt, .str, ...) is recorded
    dow=date.dt.dayofweek,
    day_idx=(date - pd.Timestamp("1995-01-01")).dt.days,
    dist_pt=X["District"] + "|" + X["Property Type"],
)
```

A derived DataOp referenced several times (`date` above) is computed **once**
and shared between the downstream nodes, so there is no performance penalty for
the fine-grained version.

### Row filtering is recordable too — including frequency conditions

A filter that depends on a *computed statistic* of a column still needs no
`deferred`/`apply_func`. `value_counts()`, comparison, `.index`, `.isin(...)`,
`~`, boolean-mask `df[mask]` and `reset_index` are all recorded, so the common
"drop classes with fewer than `n_splits` rows before cross-validating" reads
almost exactly like the eager code (verified equal to the loop-and-drop version
row for row):

```python
raw_target         = data["Cover_Type"]
class_counts       = raw_target.value_counts()
problematic_classes = class_counts[class_counts < 3].index
filtered = data[~raw_target.isin(problematic_classes)].reset_index(drop=True)
```

or, without materialising the class list, in one pass:

```python
sizes    = raw_target.groupby(raw_target).transform("size")
filtered = data[sizes >= 3].reset_index(drop=True)
```

Both appear in `describe_steps()` as their individual operations
(`CallMethod 'value_counts'`, `BinOp: lt`, `GetAttr 'index'`,
`CallMethod 'isin'`, `UnaryOp: invert`, …) instead of one opaque
`Call 'drop_rare_classes'`. Row filtering must sit **before** `mark_as_X` /
`mark_as_y`, since it changes the number of rows (section 2).

> **`np.log1p` caution:** only valid for a strictly-positive target (it produces
> NaN for values <= -1, which then crashes model fitting). Skip the log transform
> if the target can be zero/negative or is already well-scaled.

A plain pandas `.apply` on a selected sub-frame broadcasts an elementwise function
across its columns (also recorded):
```python
X_skewed_log = X_skewed.apply(np.log1p)
```

---

## 5. Encoding: turning raw columns into a numeric matrix

A model needs an all-numeric, clean matrix. Two strategies.

### Strategy A (recommended default): `TableVectorizer`
`skrub.TableVectorizer()` inspects each column and applies a sensible encoder
(numbers passed through, low-cardinality categoricals one-hot encoded,
high-cardinality strings encoded, dates expanded). It handles mixed real-world
data with almost no configuration:

```python
from skrub import TableVectorizer
X_vec = X.skb.apply(TableVectorizer())
pred  = X_vec.skb.apply(HistGradientBoostingRegressor(), y=y)
```

> **Important — TableVectorizer does NOT impute missing numeric values.** It
> leaves NaN in numeric columns. NaN-tolerant models
> (`HistGradientBoostingRegressor/Classifier`) handle that natively, but linear
> models, SVMs, KNN, etc. will crash on NaN. For those, add an imputer **after**
> the vectorizer:
> ```python
> from sklearn.impute import SimpleImputer
> from sklearn.linear_model import Ridge
> X_vec = X.skb.apply(TableVectorizer()).skb.apply(SimpleImputer())
> pred  = X_vec.skb.apply(Ridge(), y=y)
> ```

Customise per column kind:
```python
from skrub import TableVectorizer, StringEncoder
from sklearn.preprocessing import OneHotEncoder
tv = TableVectorizer(
    high_cardinality=StringEncoder(),
    low_cardinality=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)
```
Useful skrub encoders: `TableVectorizer`, `Cleaner`, `DatetimeEncoder`,
`StringEncoder`, `TextEncoder`, `GapEncoder`, `MinHashEncoder`,
`SimilarityEncoder`, `ToDatetime`, `ToCategorical`, `SquashingScaler`. A ready
baseline: `skrub.tabular_pipeline("regressor")` / `tabular_pipeline("classifier")`.

### Strategy B: explicit per-group encoding with selectors + concat
For precise control (mirroring a sklearn `ColumnTransformer`): split by
**selectors**, encode each part, then `concat` horizontally.

```python
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

num = X.skb.select(s.numeric())
cat = X.skb.select(s.string() | s.categorical())

num_v = num.skb.apply(SimpleImputer(strategy="median"))
cat_v = (cat.skb.apply(SimpleImputer(strategy="most_frequent"))
            .skb.apply(OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

X_vec = num_v.skb.concat([cat_v], axis=1)   # note: others is a LIST
```

> skrub works on **dense** arrays — always pass `sparse_output=False` to `OneHotEncoder`.

---

## 6. Selectors — choosing columns declaratively

You **cannot** iterate over `X.columns` inside the plan. Use `skrub.selectors`
(import `from skrub import selectors as s`). Apply with `X.skb.select(sel)` (keep)
or `X.skb.drop(sel)` (remove).

| Selector | Matches |
|---|---|
| `s.all()` | every column |
| `s.numeric()` | numeric columns |
| `s.integer()` / `s.float()` | integer / float columns |
| `s.string()` | string columns |
| `s.categorical()` | categorical-dtype columns |
| `s.boolean()` | boolean columns |
| `s.any_date()` | datetime columns |
| `s.has_dtype(dtype, ...)` | columns of given dtype(s) |
| `s.cardinality_below(k)` | columns with fewer than `k` unique values |
| `s.has_nulls(p=0.0)` | columns whose null proportion exceeds `p` |
| `s.glob("pat*")` | column names matching a shell glob |
| `s.regex("pat")` | column names matching a regex |
| `s.cols("a", "b")` | the named columns |

Predicate selectors (most flexible):
```python
low_card = s.filter(lambda col: col.dtype != "object" and col.nunique() < 10)  # by column
ordinal  = s.filter_names(lambda name: "ordinal" in name or "cat" in name)     # by name
```
Combine with set algebra: `|` union, `&` intersection, `-` difference,
`~`/`s.inv(sel)` complement.
```python
cats = s.string() | s.categorical()
non_id_numeric = s.numeric() - s.cols("Id")
```

**Broadcasting:** `X.skb.apply(SomeTransformer())` over a multi-column frame
broadcasts the transformer (one clone per column for single-column transformers),
eliminating manual `for col in cols:` loops. A custom per-column transformer must
subclass `BaseEstimator, TransformerMixin` and operate on a single-column frame:

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class RankEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        col = X.columns[0]
        self.vocab_ = X[col].value_counts().rank(method="dense", ascending=False)
        return self
    def transform(self, X):
        col = X.columns[0]
        return pd.DataFrame({col: X[col].map(self.vocab_)})

high_card = X.skb.select(s.filter(lambda c: c.nunique() >= 30))
high_card_v = high_card.skb.apply(RankEncoder())   # auto-broadcast to each col
```

---

## 7. Applying estimators — `.skb.apply`

```python
node.skb.apply(estimator, *, y=None, cols=..., exclude_cols=None,
               allow_reject=False, unsupervised=False)
```
- **Transformer:** `X_v = X.skb.apply(StandardScaler())`. Restrict to a subset with
  `cols=` (name, list, or selector) or `exclude_cols=`.
- **Predictor:** pass the target → `pred = X_v.skb.apply(model, y=y)`. The result is
  the prediction node.
- `allow_reject=True` lets a transformer skip columns it can't handle (e.g.
  `ToDatetime()` on all columns converts parseable strings, passes the rest through).
- `unsupervised=True` for estimators that need no `y`.

Multiple models on the same features, merged:
```python
p1 = X_vec.skb.apply(Model(), y=labels["label1"])
p2 = X_vec.skb.apply(Model(), y=labels["label2"])
pred = p1.skb.concat([p2], axis=1)
```

---

## 8. Target transforms — mark raw y, invert at predict time

The CV score is computed against the node marked with `mark_as_y` (verified).
Two rules follow:

1. **Always mark the RAW target.** Never transform y before `mark_as_y` — that
   silently changes the scoring domain, and pipelines with and without the
   transform stop being comparable (a log-space RMSE and a raw-space RMSE end
   up sorted together in the same leaderboard).
2. If you fit on a transformed target (e.g. `log1p` for a skewed price), apply
   the **inverse to the predictions** so the plan's output is back in the raw
   domain — but **gated on the eval mode**: during fit, a predictor node
   evaluates to the *fitted estimator*, not predictions, so an ungated
   `np.expm1` (or any arithmetic on prediction nodes) raises
   `TypeError: unsupported operand ...` inside the CV loop.

```python
import numpy as np

y     = data["Price"].skb.mark_as_y()        # RAW target
X     = data.drop(columns=["Price"]).skb.mark_as_X()
y_log = y.skb.apply_func(np.log1p)           # transform INSIDE the plan
pred_log = X_vec.skb.apply(model, y=y_log)

def expm1_predictions(pred, mode):
    if mode == "fit":     # the node's value is the fitted estimator here
        return None
    return np.expm1(pred)

pred = pred_log.skb.apply_func(expm1_predictions, skrub.eval_mode())
```

The same gating applies to **any** computation on prediction nodes (clipping,
weighted averaging of two models' predictions, ...).

Metric note: with predictions back in the raw domain you can still score by
relative error — `scoring="neg_root_mean_squared_log_error"` (RMSLE) is
numerically identical to RMSE on the log-space predictions, while every
pipeline (log target or not) stays comparable. RMSLE rejects negative
predictions, so add `np.clip(pred, 0, None)` in the non-fit branch when the
model was fit on a raw target and could extrapolate below zero.

`eval_mode()` evaluates to `"preview"`, `"fit"`, `"transform"`, `"predict"`, etc.
`.skb.match(mapping, default=...)` picks by mode; `cond.skb.if_else(a, b)` branches
on a boolean DataOp.

---

## 9. Scoring with `make_grid_search` (THE main task)

Always score with a grid search run on the captured data and read `results_`.
Pass **no `cv=`** — the splitter set on `mark_as_X` (section 3) drives:

```python
search = pred.skb.make_grid_search(n_jobs=1, fitted=True, refit=False,
                                   scoring="r2")
print(search.results_)
best_score = search.results_["mean_test_score"].iloc[0]   # row 0 = best
```

### `results_` shape (verified)
- It is a **pandas DataFrame, sorted best-first** (row 0 has the highest score).
- `mean_test_score` is the **unweighted mean of the per-fold test scores**
  (sklearn's `cv_results_`), *not* a metric pooled over all out-of-fold rows. The
  two differ slightly whenever folds have unequal sizes — a script that
  concatenates OOF predictions and scores them once lands ~1e-9 away from the
  same pipeline scored here. That gap is structural: it cannot be closed by
  `make_grid_search`, and it is not a sign of a broken pipeline.
- **No choices in the plan** → one row, columns `['mean_test_score']`.
- **With choices** → one row per parameter combination; each named choice becomes
  its own column plus `mean_test_score`. Example with a choice named `"alpha"`:
  ```
     alpha  mean_test_score
  0    1.0         0.921894
  1    0.1         0.921886
  2   10.0         0.919758
  ```

### `scoring` strings (scikit-learn)
Regression: `"r2"`, `"neg_root_mean_squared_error"`,
`"neg_mean_absolute_error"`, `"neg_mean_squared_error"`,
`"neg_root_mean_squared_log_error"` (RMSLE — relative error for skewed,
strictly-positive targets; predictions must be >= 0).
Classification: `"accuracy"`, `"roc_auc"`, `"f1"`, `"neg_log_loss"`,
`"average_precision"`. (`neg_*` scorers are negated so that higher = better.)

### Custom / weighted metrics — declare them in the plan with `with_scoring`

A scorer **string** is all most tasks need (passed to the harness as
`--scoring`). When the metric is a custom callable, or needs **per-row
`sample_weight`**, a string can't express it — declare the scorer **in the
plan** with `.skb.with_scoring(scorer, kwargs=..., name=...)`. This is the
scoring analog of setting the CV on `mark_as_X` (section 3), and the priority is
the same: skrub uses the plan's scorer **only when `make_grid_search` is called
with `scoring=None`** — a passed `scoring=` overrides it. The harness detects a
`with_scoring` node and passes `scoring=None` so your scorer drives; run
ml-score **without `--scoring`** and it locks the scorer's `name`.

```python
from sklearn.metrics import make_scorer

def weighted_r2(y_true, y_pred, sample_weight=None):
    ...

scorer = make_scorer(weighted_r2, response_method="predict")
pred = X_feat.skb.apply(model, y=y).skb.with_scoring(
    scorer, kwargs={"sample_weight": X["weight"]}, name="weighted_r2")
```

- **`kwargs` may be DataOps.** `kwargs={"sample_weight": X["weight"]}` is
  evaluated **per fold** on that fold's rows, so the scorer receives the test
  fold's weights, correctly aligned with the predictions (verified). Always pass
  `name=` — it becomes the locked metric label on the leaderboard.
- **Keep the weight column in X, exclude it from the model.** `sample_weight`
  must derive from the marked `X` so it is split per fold; drop it only from the
  *model's* features, not from `X`:
  ```python
  X, y = load_xy(target="resp", groups="date_id")  # weight stays in X
  feats = X.skb.drop("weight")                      # model sees all but weight
  pred  = feats.skb.apply(model, y=y).skb.with_scoring(
              scorer, kwargs={"sample_weight": X["weight"]}, name="weighted_r2")
  ```
- In this project the scorer is set once **centrally in `common.py`**
  (`SCORER`/`SCORER_NAME` + `attach_scoring`), so every pipeline scores
  identically — the harness locks it and refuses a different metric without
  `--change-metric`, exactly as for string scorers. Use a **single** scorer per
  workspace: the leaderboard tracks one metric (`mean_test_score`).

### The choice tool in this project: `choose_from` (discrete only)
The harness always scores with `make_grid_search`, which enumerates a **discrete**
grid — so **`skrub.choose_from([...])` with explicit values is the only choice
constructor we use.** It expresses everything this project needs: hyperparameter
search, estimator swaps, and feature-set ablations (section 10). Continuous-range
constructors (`choose_float` / `choose_int`) and `make_randomized_search` are
**not used here** — grid search rejects continuous ranges, and randomized search
is out of scope.

A plan with **no choices at all** is equally valid — the grid search just
cross-validates that single pipeline and `results_` has one row. That is a
*single-scoring* candidate; a plan **with** choices is an *explorative
fused-choice* run that scores many variants at once (section 10).

---

## 10. Exploration with `choose_from` — hyperparameters, estimators, ablations

`skrub.choose_from([...], name=...)` is the single workhorse of exploration.
Replace any hard-coded value — a hyperparameter, a whole estimator, or an entire
feature-engineering sub-graph — with a choice; skrub collects every choice and the
harness enumerates the full grid in **one scored run**, recording the best
combination to the leaderboard and the whole grid to `results.json → extra.grid`.
Each named choice becomes its own column in `results_` (section 9).

**(a) hyperparameters and estimator swaps** — tune within one candidate:
```python
import skrub
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

model = skrub.choose_from(
    {
        "ridge": Ridge(alpha=skrub.choose_from([0.1, 1.0, 10.0], name="alpha")),
        "hgb":   HistGradientBoostingRegressor(
                    learning_rate=skrub.choose_from([0.03, 0.05, 0.1], name="lr")),
    },
    name="model",
)
pred = X_vec.skb.apply(model, y=y)   # harness scores; cv on mark_as_X
```

**(b) feature-set ablation** — fuse alternative feature sub-graphs and let the grid
pick the winner, so one run answers "does this feature help?":
```python
base = add_ratios(add_soil(base_geo(X)))
variants = {
    "base":     base.skb.apply(model, y=y),
    "base+elu": add_elu(base).skb.apply(model, y=y),
    "base+te":  add_target_enc(base, y).skb.apply(model, y=y),
}
pred = skrub.choose_from(variants, name="featureset").as_data_op()
```
`.as_data_op()` turns a choice over prediction sub-graphs back into a single
prediction node the harness can score. This is the **explorative fused-choice**
pattern: every variant is one value of a named choice, scored under one CV, with
the per-variant scores captured in `extra.grid`.

> **Discrete only.** Every choice enumerates explicit values via `choose_from([...])`;
> grid search takes no continuous ranges (section 9).

> **Fused choice = discovery, not lineage.** A fused-choice run records as ONE
> leaderboard row (the best variant) plus the full grid in `extra.grid`. Use it to
> *discover* what works; once a variant wins, promote it to its own single-scoring
> pipeline file so it becomes a first-class, comparable leaderboard/lineage node.

Introspection: `pred.skb.describe_param_grid()`, `pred.skb.describe_defaults()`.

---

## 11. Inspection & debugging

- `pred.skb.preview()` — value of the node on the (sub)sample data.
- `pred.skb.eval({"data": some_df})` — run the plan on a given environment.
- `pred.skb.describe_steps()` — text view of the graph.
- `pred.skb.full_report()` — rich HTML report of every step.
- `skrub.TableReport(df)` — exploratory report of a dataframe.

**Performance while building:** every recorded op eagerly computes a preview on
the var data. On a large table, either keep a `.skb.subsample(n=...)` right
after the load (previews stay cheap, CV score unaffected) or wrap the plan
construction in `with skrub.config_context(eager_data_ops=False):` to skip
eager previews entirely.

**Two levels of "does it work", and only the second one is proof.** Importing a
plan under `config_context(eager_data_ops=False)` builds the whole graph without
reading a byte of data — cheap, needs no dataset, and catches every structural
mistake (a typo'd column op, `.skb.apply` on a plain function, a missing mark).
It does *not* run the splitter, fit any estimator, or call the scorer, so it
cannot catch pitfall 17's `split_kwargs=None`, an estimator that rejects a
column's dtype, a metric that refuses negative predictions, or a target left in
the wrong domain. Build first, then score.

### Exploration scripts (data_exploration_N.py)

For exploration scripts, **use plain `pd.read_csv` directly** — do NOT call
`load_csv()` or `load_xy()`. Those helpers return DataOps (lazy recorded nodes
for pipeline plans), not DataFrames. Calling `.skb.get_data()` on a DataOp
returns an internal dict, not a DataFrame.

```python
# Correct pattern for exploration scripts
import pandas as pd
from common import WS_ROOT   # absolute path to the workspace root

df = pd.read_csv(WS_ROOT / "input" / "train.csv", nrows=50_000)
print(df.shape)
print(df.dtypes)
print(df.describe())
```

Only switch to `load_csv()` / `load_xy()` when you are building an actual
pipeline plan (i.e. the file defines `pred`, `DESCRIPTION`, `PARENT`).

---

## 12. Common pitfalls (read before you ship)

1. **`.skb.apply` is for scikit-learn estimators only.** Plain functions →
   `.skb.apply_func` / `skrub.deferred`. Elementwise column math → pandas `.apply`.
2. **Mark `X` early.** Real preprocessing must come *after* `mark_as_X()`.
3. **`concat` takes a list:** `a.skb.concat([b, c], axis=1)`, never `a.skb.concat(b)`.
4. **No Python loops over columns** in the plan — use selectors + broadcasting.
5. **Dense only:** set `sparse_output=False` on `OneHotEncoder`.
6. **`TableVectorizer` doesn't impute numeric NaNs** — add a `SimpleImputer` for
   models that can't handle NaN (linear/SVM/KNN); `HistGradientBoosting*` is fine.
7. **`choose_from` is the only choice tool** — grid search enumerates discrete
   values, so use `choose_from([...])` for hyperparameters, estimator swaps, and
   feature ablations (no continuous ranges or randomized search).
8. **Score with `fitted=True, refit=False`** — fitted runs the CV, refit=False skips
   the unnecessary full-data refit. Read `results_["mean_test_score"].iloc[0]`.
9. **`np.log1p` only for strictly-positive targets** (NaN otherwise → fit crash).
10. **`subsample` only affects previews** unless `keep_subsampling=True`, so it does
    not change your CV score — safe to leave in for fast previews.
11. **Mark the RAW target.** The CV score is computed against the `mark_as_y`
    node — transforming y before the mark silently changes the scoring domain
    and breaks comparability across pipelines. Transform after the mark and
    invert the predictions at predict time (section 8).
12. **In fit mode a predictor node evaluates to the fitted estimator**, not
    predictions. Gate any post-prediction computation (inverse transforms,
    clipping, blending) on `skrub.eval_mode()` (section 8).
13. **Don't wrap multi-step feature engineering in one `@skrub.deferred`
    function** — it becomes a single opaque DAG node. Write fine-grained
    recorded ops instead (section 4); `deferred` is only for a single operation
    that can't be recorded (e.g. eval-mode-gated post-prediction code).
14. **Set the CV on `mark_as_X`, never via `cv=` at scoring time** (section 3).
    A passed `cv=` overrides `mark_as_X` (the reverse of what you'd guess), and
    `split_kwargs` (e.g. `GroupKFold` `groups`) can *only* be wired on
    `mark_as_X`. In this project the CV is set centrally in `common.py`
    (`make_cv()` + `load_xy`); keep it fixed across the workspace — the harness
    records it and warns if a later pipeline's split differs.
15. **CatBoost's sklearn wrapper has real `clone()`/param-substitution bugs.**
    `sklearn.base.clone()` fails outright when `cat_features` or a bare
    `learning_rate` is passed to the constructor (`RuntimeError: Cannot clone
    object CatBoostClassifier(...), as the constructor either does not set or
    modifies parameter <name>`) — reproduces even on a toy example; other
    params (`depth`, `thread_count`, `random_state`, `verbose`,
    `auto_class_weights`) clone fine. Fix with a `__sklearn_clone__` override
    (honored by sklearn>=1.3), which bypasses the buggy default check:
    ```python
    class CatBoostClassifierCloneable(CatBoostClassifier):
        def __sklearn_clone__(self):
            return CatBoostClassifierCloneable(**self.get_params(deep=False))
    ```
    Separately, embedding `skrub.choose_from([...], name=...)` directly as a
    single CatBoost constructor kwarg (the normal pattern for HGB/sklearn
    estimators) fails differently — `TypeError: Object of type Choice is not
    JSON serializable`, raised during skrub's eager preview because its
    per-param substitution also relies on CatBoost's non-standard
    get/set_params. The `__sklearn_clone__` fix does **not** fix this second
    issue. Instead, never embed `choose_from` as a single CatBoost param —
    build fully-specified whole-estimator instances for every combination and
    choose over the complete objects:
    ```python
    variants = {f"d{d}_lr{lr}": CatBoostClassifierCloneable(depth=d, learning_rate=lr, ...)
                for d in (6, 8) for lr in (0.05, 0.15)}
    model = skrub.choose_from(variants, name="model")
    ```
16. **A custom classifier wrapper (e.g. label-encoding a model that rejects
    string targets, or post-processing predictions) must inherit
    `ClassifierMixin` BEFORE `BaseEstimator`** —
    `class Foo(ClassifierMixin, BaseEstimator)`, not
    `class Foo(BaseEstimator, ClassifierMixin)`. sklearn's tag system
    (`__sklearn_tags__`, used by `is_classifier()`) resolves via MRO: with
    `BaseEstimator` first, `BaseEstimator.__sklearn_tags__` wins and
    `ClassifierMixin`'s `estimator_type = "classifier"` override never runs —
    silently, no error at construction or `.fit()`/`.predict()` time. It only
    surfaces when something explicitly checks `is_classifier()`, e.g.
    `VotingClassifier`/`StackingClassifier`'s `_validate_estimators()`, as a
    confusing `ValueError: The estimator Foo should be a classifier.` raised
    deep inside the meta-estimator's `fit()` — nowhere near the actual bug.
17. **On skrub < 0.10, always pass `split_kwargs={}` next to `cv=`** on
    `mark_as_X` (section 3). Without it the plan builds and previews fine, then
    dies inside `make_grid_search` with `TypeError: ... argument after ** must be
    a mapping, not NoneType`.
18. **A plan that builds is not a plan that scores.** Building under
    `config_context(eager_data_ops=False)` proves the graph is well-formed and
    touches no data; it does not exercise the splitter, the estimators' `fit`, or
    the scorer. Pitfall 17, an estimator rejecting a dtype, and a metric that
    refuses negative predictions all pass the build and fail the run. Score the
    plan before believing it.
19. **Per-fold logic that cannot be recorded belongs in a wrapper estimator, not
    in the plan.** Recorded ops cover dataframe transformations; anything the
    eager script did *inside its fold loop* that is not a dataframe
    transformation — augmenting the training rows, fitting several models and
    averaging their `predict_proba`, a torch training loop, computing
    `class_weight` from that fold's `y` — goes into a small
    `ClassifierMixin, BaseEstimator` / `RegressorMixin, BaseEstimator` wrapper
    applied with `.skb.apply(wrapper, y=y)`. Inside its `fit` you are in ordinary
    Python: loops, `.copy()`, an **inner** `train_test_split` for early stopping
    (legitimate — it is per-fit, not the outer CV). skrub re-fits the wrapper per
    fold, so the logic stays leakage-free. Mixin first in the bases (pitfall 16).
20. **A vectorised substitute for a column loop reproduces values, not column
    names or order — and order changes randomised models.** Replacing
    `for c in soil_cols: X[f"{c}_x_Elevation"] = X[c] * X["Elevation"]` with
    `PolynomialFeatures(interaction_only=True)` plus a rename gives numerically
    identical columns named `Elevation_x_Soil_Type3` instead of
    `Soil_Type3_x_Elevation`, in a different column order. Names are cosmetic,
    but column ORDER feeds the per-split feature subsampling of
    `RandomForest`/boosters, so the CV score moves (measured: 5e-4 on a forest).
    When exact names/order matter, put the loop inside a
    `TransformerMixin, BaseEstimator` transformer's `transform` (pitfall 19) —
    there a Python loop over `X.columns` is fine, because it runs at fit time on
    a real dataframe rather than being recorded.

---

## 13. End-to-end worked example (score a candidate on a generic `train.csv`)

```python
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

# --- Recorded load ---
data = (skrub.as_data_op("train.csv")
             .skb.apply_func(pd.read_csv)
             .skb.subsample(n=2000))                 # subsample: fast previews only

# --- Target and features (mark early; CV splitter set on mark_as_X) ---
y = data["SalePrice"].skb.mark_as_y()
X = data.drop(columns=["SalePrice", "Id"]).skb.mark_as_X(
        cv=KFold(n_splits=5, shuffle=True, random_state=42), split_kwargs={})

# --- Light feature engineering (recorded) ---
X = X.assign(TotalSF=X["1stFlrSF"] + X["2ndFlrSF"] + X["TotalBsmtSF"])

# --- Encode all mixed-type columns in one shot ---
X_vec = X.skb.apply(skrub.TableVectorizer())

# --- Model with one discrete tunable hyperparameter ---
model = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_from([0.03, 0.05, 0.1, 0.3], name="lr"),
    random_state=42,
)
pred = X_vec.skb.apply(model, y=y)

# --- Score the plan by cross-validation. This is the deliverable. ---
#     No cv= — the KFold set on mark_as_X above drives.
search = pred.skb.make_grid_search(n_jobs=1, fitted=True, refit=False,
                                   scoring="r2")
print(search.results_)
print("Best CV R2:", search.results_["mean_test_score"].iloc[0])
```

---

## 14. PyTorch models via skorch

A PyTorch `nn.Module` can be applied inside a plan exactly like any other
estimator (`.skb.apply(model, y=y)`), once wrapped with
[skorch](https://skorch.readthedocs.io)'s `NeuralNetClassifier`/
`NeuralNetRegressor`, which gives it a scikit-learn-compatible
`fit`/`predict`/`predict_proba` API. See also skrub's own worked example:
<https://skrub-data.org/stable/auto_examples/02_data_ops/1160_pytorch.html>.

**Not part of the default shared venv** (skrub_dataops guide + CLAUDE.md's
package list) — install on a machine with no GPU using the CPU-only wheel
(the default PyPI `torch` wheel pulls in CUDA dependencies you don't need):
```bash
uv pip install -p <venv>/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p <venv>/bin/python skorch
```

### The three things a plain `.skb.apply(NeuralNetClassifier(...), y=y)` won't handle for you

1. **`nn.CrossEntropyLoss` needs integer class labels (`int64`), not the raw
   string target.** But section 8's rule still applies: mark the **raw**
   (string) `y` at `mark_as_y`, never encode before the mark. The encode/decode
   has to happen **inside the estimator**, the same way a target-transform is
   inverted at predict time (section 8) — label-encode in `fit()`, and
   inverse-transform the argmax back to string labels in `predict()`.
2. **Class-imbalance handling has no `class_weight="balanced"` shortcut.**
   The equivalent for a neural net's loss is
   `nn.CrossEntropyLoss(weight=<tensor>)` — a per-class weight tensor, which
   for balanced weighting is exactly `sklearn.utils.class_weight
   .compute_class_weight("balanced", classes=..., y=...)` cast to a tensor.
   Compute it **inside `fit()`** from that call's own `y`, so it's recomputed
   correctly per CV fold (leakage-free), the same pattern used for a model
   family with no native `class_weight` param (guide pitfall pattern used for
   XGBoost elsewhere in this project).
3. **The input feature count (`n_features` for the first `nn.Linear`) isn't
   known until the vectorized data arrives** — `TableVectorizer`'s output
   width depends on the data (categorical cardinalities, etc.), not something
   fixed at plan-construction time. Read `X.shape[1]` inside `fit()` and pass
   it to the module constructor there, rather than hard-coding it.

Because all three are per-fit, per-fold concerns, the cleanest fix is one
small `ClassifierMixin, BaseEstimator` wrapper that does the label
encoding, weight computation, and dynamic module sizing, then delegates to an
internally-constructed `NeuralNetClassifier` — the plan itself just calls
`.skb.apply(YourWrapper(...), y=y)` like any other classifier:

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from skorch import NeuralNetClassifier


class SimpleMLP(nn.Module):
    def __init__(self, n_features, hidden_units=64, n_classes=3, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_units, n_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.out(x)


class BalancedSkorchMLP(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_units=64, dropout=0.2, lr=1e-3, max_epochs=15,
                 batch_size=1024, random_state=0):
        self.hidden_units, self.dropout = hidden_units, dropout
        self.lr, self.max_epochs = lr, max_epochs
        self.batch_size, self.random_state = batch_size, random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.array(X, dtype=np.float32)          # np.array COPIES -- np.asarray
        y = np.asarray(y)                           # can hand torch a non-writable
        self.label_encoder_ = LabelEncoder().fit(y)  # array and trigger a UserWarning
        y_enc = self.label_encoder_.transform(y).astype(np.int64)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        weight = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_enc)
        self.net_ = NeuralNetClassifier(
            module=SimpleMLP,
            module__n_features=X.shape[1],          # read from the actual data
            module__hidden_units=self.hidden_units,
            module__n_classes=n_classes,
            module__dropout=self.dropout,
            max_epochs=self.max_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            optimizer=torch.optim.Adam,
            criterion=nn.CrossEntropyLoss,
            criterion__weight=torch.tensor(weight, dtype=torch.float32),
            device="cpu",       # no GPU on this machine; set "cuda"/"mps" if available
            train_split=None,   # let the harness's own CV drive validation, not skorch's
            verbose=0,
        )
        self.net_.fit(X, y_enc)
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(np.array(X, dtype=np.float32))

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(idx)
```

Used in a plan like any other estimator, with hyperparameters swept via the
usual `choose_from` (constructor kwargs of a plain `BaseEstimator` subclass
substitute the same way as any sklearn estimator's — no special handling
needed):

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

X, y = load_xy(target="health_condition")

# MLPs need a clean float32 numeric matrix and benefit a lot from scaled
# inputs (unlike trees) -- TableVectorizer alone leaves NaN in numeric
# columns (section 5) and does not scale.
X_vec = (X.skb.apply(skrub.TableVectorizer())
           .skb.apply(SimpleImputer(strategy="median"))
           .skb.apply(StandardScaler()))

model = BalancedSkorchMLP(
    hidden_units=skrub.choose_from([32, 64], name="hidden_units"),
    max_epochs=skrub.choose_from([10, 20], name="max_epochs"),
    random_state=0,
)
pred = X_vec.skb.apply(model, y=y)
```

**Cost is modest for a small MLP on CPU**: on a 690k-row, ~30-feature table,
a full-data 10-epoch fit took ~27s on a 24-core CPU box — cheap enough to
run through the normal `choose_from` grid-search harness like any other
model, no special-casing needed. More epochs cost roughly linearly more; size
the grid accordingly.

> **Not (yet) a `class_weight`-style native lever, and not obviously better
> than boosting here.** In one workspace's test on an imbalanced 3-class
> tabular task, a 2-hidden-layer, 64-unit MLP (20 epochs) reached a
> balanced_accuracy a few points below a tuned LightGBM/HistGradientBoosting
> model — consistent with the general pattern that gradient-boosted trees
> tend to win on tabular data without a lot of extra architecture/tuning
> effort. Useful as a genuinely different model family for an ensemble or a
> sanity check, not a default first choice for tabular classification/regression.