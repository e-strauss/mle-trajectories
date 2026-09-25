# TrackTheTrackers — 15-pipeline exploration

> **Note for this collection.** This is the analysis written during the original
> run, when the feature blocks lived in a precomputed parquet store under
> `../features/`. In this copy that computation has been **inlined** into
> `features.py` and recorded as plan nodes, so references below to "the frozen
> store", `../features/` and `data_exploration_4/_7/_9` describe how the blocks
> were *derived*; the blocks themselves are now built from `input/` at run time.
> `../verify_inline_features.py` confirms the two produce identical values,
> which is what lets the scores below stand unchanged. See `../README.md` for
> the normalisation, and `features.py` for where each block now lives.

Predict up to 10 third-party trackers for each of the 50,000 domains in
`input/target.tsv`, scored by **Recall@10**.

## How the task was framed

`target.tsv` domains appear **nowhere** in `tracking_graph_train`
(0/50,000 overlap, verified in `data_exploration_2.py`). So this is not a
partially-observed recommendation problem — it is **pure cold start** over a
fixed label space of 355 trackers:

| | |
|---|---|
| one row | one domain |
| `y` | 355 binary indicators (`y_0..y_354`) |
| prediction | a (n, 355) score matrix; the top 10 per row is the submission |
| metric | `recall_at_10` = mean over domains of \|top10 ∩ true\| / \|true\| |

**Metric choice.** Recall@10 is a ranking metric over a matrix-valued
prediction, so no sklearn scorer string can express it. It is declared in the
plan via `common.attach_scoring` and locked by the harness as
`plan:recall_at_10`. It is the task's own metric, so there was nothing to guess.

**CV choice.** `KFold(3, shuffle=True, random_state=42)` over the frozen row
sample, set once in `common.make_cv()`. Rows are independent domains (one row
each, no repeats) and the real target set is held out at the domain level, so a
shuffled KFold is a faithful proxy. `data_exploration_3.py` checked that the
sample matches `target.tsv` on the things that matter — tracked-neighbour
coverage 91.7% vs 90.8%, TLD mix, label-count mix — so the CV is not
optimistic about connectivity. 3 folds ⇒ ~168k test domains per fold.

## Building training data from a "pile of data"

There is no given feature table. `data_exploration_4.py` and `_7.py` build a
**frozen feature store** under `../features/` (train + matching target
variants), rooted on one deterministic row sample — tracked domains with
`domain_id % 37 == 11`, i.e. 505,548 rows, no randomness — because the raw data
(18.7M tracked domains, 623M link edges) cannot be cross-validated directly.
Every pipeline reads those same files, so all scores are comparable.

| block | columns | what it is |
|---|---|---|
| `labels` | `y_0..y_354` | the target |
| `nbr_out` / `nbr_in` | `no_*` / `ni_*` | trackers of the domains this domain links to / from |
| `tld_pop` | `tp_*` | **leave-one-out** P(tracker \| TLD) |
| `nbr_hub` | `nh_*` | neighbourhood weighted `1/log2(2+linkdeg(nbr))` — hubs discounted |
| `nbr_rec` | `nr_*` | reciprocal neighbours only (`d→n` **and** `n→d`) |
| `nbr_2h` | `n2_*` | 2-hop reach through untracked, low-degree intermediaries |
| `meta`, `meta2` | — | hostname, TLD, degrees, press freedom, url category, graph scalars |

Leakage discipline: self-loops dropped everywhere, TLD profiles
leave-one-out corrected, and no feature ever reads its own row's trackers.

## How the code is organised: two layers (and what that costs)

**Almost no pipeline reads `input/` directly.** The work is split in two:

```
input/  (raw: 46M domains, 623M link edges, 36.7M tracking edges)
   |
   |   data_exploration_4 / _7 / _9        <- read raw input, run ONCE
   v
features/  (the frozen store: 505,548 train rows + 50,000 target rows)
   |
   |   pipeline_01..15  via common.load_xy  <- read ONLY the store
   v
results.json
```

This is forced by scale. A skrub plan re-runs every recorded step for each CV
fold and each grid variant; pipeline_09 alone was 27 fits. Re-deriving
neighbourhood histograms from a 623M-edge link graph 27 times is not an option,
and the CLAUDE.md guidance for datasets too large to CV directly says to build
one deterministic working sample and give every pipeline a shared recorded read
of it. Comparability then depends on those files never changing, which is why
they are written once and treated as frozen.

**The cost is real and worth stating plainly: the cross-validation does not
cover feature construction.** Each fold re-runs the l1 normalisation, the
TableVectorizer, the imputer, the scaler and the model on that fold's rows — but
the blocks themselves were computed once, over all rows, outside the CV. So a
mistake made *inside the store* is invisible to the score: it simply makes every
fold agree on the same wrong answer.

That is exactly how the `nbr_2h` leak happened, and exactly why it showed up as
a **fold std of 0.00005** rather than as a bad number. Nothing in the harness
could have caught it. `data_exploration_8.py` exists to fill that gap and is the
only real guard on this layer — it is the equivalent of a CV for the feature
store, and any new block must pass it before being scored.

Two consequences for anyone continuing this workspace:

- **Adding a block file is safe; editing one is not.** New files leave the rows
  and labels untouched, so pipelines just opt in via `load_xy(blocks=...)` and
  old scores stand. Editing a block silently invalidates every pipeline that
  read it — when the `nbr_2h` fix rewrote that block, pipeline_07 had to be
  re-scored (pipelines 01-06 never read it and were unaffected).
- `final_pipeline.py` is the only file that is end-to-end in the deployment
  sense: it roots on an overridable `skrub.var` and predicts on rows the plan
  was not built with. It still consumes the store's `*_target.parquet` half
  rather than deriving features from raw input.


## Results

| pipeline | Recall@10 | std | parent | what changed |
|---|---|---|---|---|
| 01 | 0.75477 | 0.00101 | — | constant global tracker popularity, **no features** |
| 02 | 0.84639 | 0.00105 | 01 | fused-choice blend of neighbour / TLD / popularity blocks |
| 03 | 0.86894 | 0.00107 | 02 | multi-output **Ridge** (learns cross-tracker weights) |
| 04 | 0.87081 | 0.00065 | 03 | + degrees, hostname shape, one-hot TLD / url-category |
| 05 | 0.88953 | 0.00035 | 04 | **GPU MLP + metric-aligned softmax-CE objective** |
| 06 | 0.88954 | 0.00031 | 05 | MLP hyperparameter sweep — **saturated** |
| 07 | 0.88998 | 0.00033 | 06 | + hub / reciprocal / 2-hop blocks (**after fixing a leak**) |
| 08 | 0.88673 | 0.00027 | 07 | + hostname char n-grams — **regression** |
| 09 | 0.89143 | 0.00028 | 08 | leave-one-block-out ablation (best variant: `-nbr_rec`) |
| 10 | 0.89517 | 0.00034 | 09 | ablation-pruned features + 5-seed ensemble |
| **11** | **0.90004** | 0.00038 | 10 | **+ `direct`: hyperlinks to the tracker domains** |
| 12 | 0.89540 | 0.00037 | 10 | + `nbr_frac`: per-neighbour-normalised neighbourhood |
| 13 | 0.89533 | 0.00038 | 10 | + `trk_cooc`: out-of-sample tracker co-occurrence |
| 14 | 0.89364 | 0.00054 | 10 | + `tok_pop`: P(tracker \| hostname token) — **regression** |
| 15 | 0.90019 | 0.00041 | 11 | fused choice over combinations — all within noise |

Reference points: constant top-10 popularity = 0.7548; the attainable ceiling
(`mean min(10, n_true)/n_true`) = 0.9992, since 99.9% of domains have ≤10
trackers.

## What actually mattered

Of the +0.140 over the popularity baseline:

- **+0.092 — the evidence blocks themselves** (pipeline_02). Link-graph
  neighbours plus leave-one-out TLD profiles. 90.8% of target domains have at
  least one *tracked* link neighbour, which is what makes cold start tractable
  at all. Notably `tld_only` alone scores 0.795 — better than global popularity.
- **+0.023 — learning cross-tracker weights** (pipeline_03). A 1065→355 Ridge
  beats any hand-weighted blend because trackers come in bundles.
- **+0.019 — aligning the objective with the metric** (pipeline_05), the single
  largest modelling win and the one worth remembering. Recall@10 of a set *S* is
  `Σ_{t∈S} y_t / n_true`, so its expectation is maximised by the 10 largest
  **`E[y_t / n_true | x]`**, *not* the 10 largest `P(y_t = 1 | x)`. The two
  rankings differ because `n_true` correlates with *which* trackers appear: a
  tracker that mostly sits on 20-tracker portals buys less recall per slot than
  one that is a small site's only tracker. Fitting a softmax over the 355
  trackers against the row-normalised label vector estimates the right quantity
  directly. Swapping BCE → softmax-CE alone was worth ~+0.015 on fold 0
  (`data_exploration_6.py`) — more than pipelines 03 and 04 combined.
- **+0.005 — pruning harmful features and seed-averaging** (pipeline_10).

## What did not work (measured, not assumed)

- **Hyperparameters are exhausted.** Six configurations spanning width
  512→2048, depth 1→3, dropout 0.1→0.3, lr, and 30 vs 60 epochs land within
  0.0085 of each other, and the *smallest, shortest* one wins. 60 epochs costs
  ~0.003 — the model overfits, it does not underfit.
- **Hostname character n-grams hurt: −0.0033** on every fold. 256 SVD
  dimensions of surface text dilute the evidence blocks at fixed capacity.
- **Two blocks I built are net-negative.** Dropping `nbr_rec` is worth
  **+0.00145** and `nbr_2h` **+0.00135**; both are re-weightings of evidence
  `nbr_hub` already carries. `nbr_hub` is the most valuable single block
  (−0.00115 to remove).
- **`tld_pop` has strong standalone but near-zero marginal value**: 0.795 alone,
  but −0.00013 to drop from the full set. Its information is absorbed by the
  hub-discounted neighbourhood plus the one-hot TLD.

## The leak I caught (read this one)

pipeline_07 first scored **0.91276 with a fold std of 0.00005** — an order of
magnitude tighter than every other pipeline (0.0003–0.0011). The tell was not
the size of the gain but its *uniformity*: real evidence moves the mean, it does
not make three different data folds agree to five decimals.

`nbr_2h` walks `seed → mid → mid's tracked neighbours`. But `mid` was
*discovered* as a neighbour of the seed, so the seed–mid edge is still in
`mid`'s edge list — walking it returns the seed itself. Every train row is in
the tracking graph, so **its own tracker set was written into its own feature**.
`data_exploration_8.py` measured it: own-tracker coverage of the block was
exactly **1.0000** (vs 0.906 for the leak-free 1-hop block) and 200/200 probe
seeds were reachable as their own 2-hop neighbour.

This was worse than a mis-scored CV. Target domains are absent from the tracking
graph, so they are never "tracked" neighbours and **cannot** leak — the feature
was systematically informative in training and systematically empty at
submission time. After the fix (`keep2` mask in `data_exploration_7.py`),
pipeline_07's true score is **0.88998**: the entire +0.023 was the leak, and the
three new blocks are worth **+0.00044** together.

`data_exploration_8.py` is kept as a reusable two-part guard for any new block:
own-label coverage against a leak-free reference, plus a train-vs-target
structural comparison (all blocks now match within 0.003–0.03 on
any-evidence rate, with targets slightly *richer* — consistent with
`data_exploration_3`).

## Noise floor

Fold-to-fold std is 0.0003–0.0011, so the standard error of a 3-fold mean is
~0.0002–0.0006. Differences below ~0.001 are only trustworthy via **paired fold
diffs**, which is how the small results here were judged:

- pipeline_04 vs 03: +0.0019, positive on 3/3 folds (+0.00174/+0.00128/+0.00257) → real.
- pipeline_07 vs 06: +0.00044, positive on 3/3 with near-identical magnitude
  (+0.00041/+0.00048/+0.00044) → real but negligible.
- pipeline_10 vs 07: +0.0052 on 3/3 (+0.00529/+0.00521/+0.00508) → real.
- pipeline_06 vs 05: +0.00001 → indistinguishable from zero, as intended (it was
  a score-neutrality check on moving standardisation into the model).

## Where the remaining 0.10 sits

`data_exploration_5.py` attributed the loss under the Ridge model:

- by evidence: 0 tracked neighbours (8.4% of rows) → recall 0.832 and 10.7% of
  loss; >100 neighbours (5.6%) → 0.798 and 8.6% of loss. Big hub-heavy sites are
  as hard as disconnected ones.
- by label count: single-tracker domains (53.7% of rows) hold 40% of the loss;
  domains with >10 trackers score 0.42 but are only 0.3% of rows.
- only **79.7%** of a domain's true trackers appear anywhere in its 1-hop
  histogram, which bounds what neighbourhood evidence alone can do — the model
  already exceeds that via the TLD and popularity priors.

The honest next step is *new information*, not more modelling: page-level content
of the target hostnames, or a denser view of the link graph around them.
Pipelines 06–09 show that tuning, extra graph re-weightings and text features on
top of this evidence are worth ≤0.0005 each or are negative.

## Round 2 (pipelines 11-15): feature engineering only

After pipeline_09 the selection rule became explicit: **a block only earns its
keep if the model cannot already derive it.** `nbr_rec`/`nbr_2h` were
re-weightings of `nbr_hub` and came out net-negative; `tld_pop` scores 0.795
standalone but has -0.00013 marginal value. So each of 11-14 adds exactly one
new block and is scored as a **sibling** of pipeline_10 (same net, same 5 seeds,
all `PARENT=pipeline_10`) — an additive chain is what let two harmful blocks hide
in the first place.

| block | what it is | why it isn't redundant | Δ vs pipeline_10 |
|---|---|---|---|
| `direct` | hyperlinks to/from the 355 tracker hostnames | **new data** — no other block reads edges pointing at trackers | **+0.00487** |
| `nbr_frac` | P(tracker \| random *distinct* neighbour) | **non-linear** in the mention-count blocks | +0.00023 |
| `trk_cooc` | neighbourhood × out-of-sample P(b\|a) | linear, but fitted on 18.18M domains vs 337k rows | +0.00016 |
| `tok_pop` | IDF-weighted P(tracker \| hostname token), smoothed | new data, in the form that worked for TLDs | **−0.00153** |

**`direct` is the largest feature win in the workspace** (+0.0049, and
+0.00487/+0.00491/+0.00483 across folds). A hyperlink between a domain and a
tracker's own hostname coincides with a true tracking edge **50.4% of the time
against a 0.55% base rate** — a ~90× lift. It is sparse (10.9% of train, 16.9%
of target domains; 3.05% of true edges) but near-decisive where it fires, and
structurally invisible to every neighbourhood block. Its coverage being *higher*
on target rows is the safe asymmetry — more available at submission time, not
less, the opposite of the `nbr_2h` leak.

**`nbr_frac` and `trk_cooc` are noise.** Both were predicted redundant before
scoring: they sit on the same 91.7% support as `nbr_hub` and score 0.7505 /
0.7503 standalone against its 0.7641. pipeline_15 fused the four combinations and
the whole grid spans 0.89990–0.90019 against a fold std of 0.00041, with the
nominal winner beating `direct` alone by +0.00031/+0.00015/**−0.00002** —
mixed signs, no effect. `trk_cooc`'s null result is informative on its own: the
MLP already has enough data to learn tracker co-occurrence unaided, so
importing a better-estimated version of it adds nothing.

**`tok_pop` is the most interesting negative.** It has the *highest standalone
Recall@10 of any block* on its covered rows (0.8083, above `tld_pop`'s 0.7951)
and still costs 0.0015 in combination. That is now twice that hostname-derived
information has hurt despite being individually predictive — raw char n-grams in
pipeline_08, and here a properly smoothed, out-of-sample-estimated profile. At
22.9% coverage the block is mostly a constant prior vector, and the net appears
to spend capacity on it rather than gain from it. **Standalone strength and
marginal value are different quantities**, which is the entire case for scoring
blocks as siblings instead of stacking them.

A note on the `tok_pop` build: raising coverage required dropping the token
document-frequency floor from 100 to 20, and IDF weighting hands the *rarest*
tokens the *largest* weight — exactly where the profiles are worst estimated. So
the profiles are empirical-Bayes shrunk, `(count + 50·prior)/(df + 50)`, letting a
thin token decay to the global prior rather than inject noise. Without that the
block would have been far worse than −0.0015.

## Leakage guard, generalised

`data_exploration_8.py` gained a third check for this round: **standalone
Recall@10 per block**. Ranking a row's 355 trackers by one block alone and
scoring it detects a leak sharply (a leaking block scores ~1.0 on its covered
rows — the buggy `nbr_2h` would have) while doubling as a price list. All nine
blocks passed, with the repaired `nbr_2h` now at 0.7925 instead of ~1.0:

| block | coverage | standalone Recall@10 (covered rows) |
|---|---|---|
| `tok_pop` | 0.229 | 0.8083 |
| `tld_pop` | 1.000 | 0.7951 |
| `nbr_2h` (fixed) | 0.405 | 0.7925 |
| `nbr_hub` | 0.917 | 0.7641 |
| `nbr_frac` | 0.917 | 0.7505 |
| `trk_cooc` | 0.917 | 0.7503 |
| `direct` | 0.109 | 0.2244 |

Both label-derived blocks (`trk_cooc`, `tok_pop`) are estimated **only on the
18.18M tracked domains outside the frozen sample** — stronger than
leave-one-out, since the statistic is independent of every scored row's label by
construction and identical in kind for target rows.

## Chronological order — what ran when

Two sessions, 2026-09-21 and 2026-09-22. Pipeline times are the CV runs recorded
in `results.json` (89 min of scoring in total); exploration times are wall clock.
Reading top to bottom gives the actual decision path, including the two detours.

### Session 1 — establishing the problem and a baseline (2026-09-21)

| # | what ran | time | outcome |
|---|---|---|---|
| 1 | `data_exploration_1` | 19s | shapes: 18.7M tracked domains, 623M link edges, 355 trackers. Global top-10 = **0.755** |
| 2 | `data_exploration_2` | 19s | **the key finding: 0/50,000 target domains appear in the tracking graph** → pure cold start. 90.8% have a tracked link neighbour; url-classification covers only 2.3% |
| 3 | `data_exploration_3` | 9s | a uniform sample of tracked domains matches `target.tsv` (91.7% vs 90.8% neighbour coverage, same TLD/label mix) → plain KFold is a fair proxy |
| 4 | `data_exploration_4` | 46s | **builds store v1**: labels, `nbr_out`, `nbr_in`, `tld_pop`, `meta` |
| 5 | `common.py`, `models.py` | — | row sample, CV and the `recall_at_10` scorer fixed; rankers written |
| 6 | **pipeline_01** | 5s | 0.75477 — reproduces the offline baseline exactly |
| 7 | **pipeline_02** | 46s | 0.84639 — blend; `tld_only` alone is 0.795 |
| 8 | **pipeline_03** | 122s | 0.86894 — Ridge; alpha essentially flat |
| 9 | `data_exploration_5` | 46s | loss attribution: worst buckets are 0 neighbours (0.832) and >100 neighbours (0.798); only **79.7%** of true trackers appear in the neighbourhood at all |
| 10 | **pipeline_04** | 229s | 0.87081 — meta context, +0.0019 on 3/3 folds |
| 11 | `data_exploration_6` | 189s | offline fold-0 sweep: **softmax-CE beats BCE by +0.015**; depth/width nearly flat |
| 12 | **pipeline_05** | 253s | 0.88953 — the metric-aligned objective lands |
| 13 | `data_exploration_7` | 143s | **builds store v2**: `nbr_hub`, `nbr_rec`, `nbr_2h`, `meta2` (ran alongside pipeline_05) |
| 14 | **pipeline_06** | 748s | 0.88954 — hyperparameters saturated; also confirms the GPU-standardisation refactor is score-neutral |
| 15 | pipeline_07 (1st run) | 124s | **0.91276 — too good, fold std 0.00005. Not accepted.** |
| 16 | `data_exploration_8` | 46s | **leak confirmed**: own-tracker coverage of `nbr_2h` is exactly 1.0000; 200/200 probe seeds are their own 2-hop neighbour |
| 17 | `data_exploration_7` (re-run) | 142s | `keep2` fix → `nbr_2h` and `meta2` rewritten |
| 18 | **pipeline_07** (re-scored) | 124s | **0.88998** — the true value; the entire +0.023 was the leak |
| 19 | **pipeline_08** | 317s | 0.88673 — hostname char n-grams, negative on 3/3 folds |
| 20 | **pipeline_09** | 1240s | 0.89143 — ablation; `nbr_rec` and `nbr_2h` are *harmful*, `nbr_hub` is the most valuable block |
| 21 | **pipeline_10** | 246s | 0.89517 — pruned features + 5-seed ensemble |
| 22 | `final_pipeline.py` | 271s | first `submission.tsv`, from pipeline_10 |

### Session 2 — feature engineering only (2026-09-22)

| # | what ran | time | outcome |
|---|---|---|---|
| 23 | inline probe | 25s | tests whether the link graph contains edges to the tracker hostnames: **50.4% precision vs a 0.55% base rate**. Logic and numbers are restated in `pipeline_11`'s docstring; the coverage figure is reproduced in `data_exploration_8`'s standalone table |
| 24 | `data_exploration_9` | 82s | **builds store v3**: `direct`, `nbr_frac`, `trk_cooc`, `tok_pop` (run 3x — a polars API fix, then empirical-Bayes smoothing added to `tok_pop`) |
| 25 | `data_exploration_8` (extended) | 60s | new **standalone Recall@10 per block** check; all 9 blocks pass, repaired `nbr_2h` reads 0.7925 not ~1.0 |
| 26 | **pipeline_11** | 240s | **0.90004** — `direct`, +0.0049 on 3/3 folds, the largest feature gain in the workspace |
| 27 | **pipeline_12** | 246s | 0.89540 — `nbr_frac`, +0.0002 (noise) |
| 28 | **pipeline_13** | 252s | 0.89533 — `trk_cooc`, +0.0002 (noise) |
| 29 | **pipeline_14** | 246s | 0.89364 — `tok_pop`, negative on 3/3 folds |
| 30 | **pipeline_15** | 1034s | 0.90019 — fused choice; whole grid inside 0.0003, so nothing combines with `direct` |
| 31 | `final_pipeline.py` | 289s | `submission.tsv` regenerated from **pipeline_11**'s configuration |

Pipelines 11-14 are **siblings** (all `PARENT=pipeline_10`), not a chain — the
additive chain in session 1 is what let two harmful blocks hide until the
ablation. Everything else follows the lineage recorded in `results.json`.


## Files

- `data_exploration_1..3` — data shape, cold-start diagnosis, sample representativeness
- `data_exploration_4`, `_7`, `_9` — the frozen feature store (run once; **do not re-run**)
- `data_exploration_5` — loss attribution under the Ridge model
- `data_exploration_6` — offline GPU-MLP architecture/objective sweep on fold 0
- `data_exploration_8` — **leakage audit** (own-label coverage, train/target structure, standalone Recall@10)
- `featureset.py` — the shared pipeline_10 feature graph + opt-in new blocks (11-15)
- `common.py` — frozen row sample, CV, the `recall_at_10` scorer
- `models.py` — `PopularityRanker`, `BlendRanker`, `BlockTransform`, `TorchMLPRanker`
- `pipeline_01..15` — the scored candidates; `final_pipeline.py` — the refit/submission artifact (**never** ml-score it)
- `results.json` — the leaderboard, written only by ml-score
- `../features/` — the frozen store (19 parquet files); `../submission.tsv` — the deliverable

Read `## Chronological order` above for the sequence these were produced in, and
`## How the code is organised` for why the pipelines read `../features/` rather
than `../input/`.

## Submission

`final_pipeline.py` refits **pipeline_11** on all 505,548 training rows (5 nets
× 30 epochs, ~4.5 min) and writes `../submission.tsv`.

Not the leaderboard's nominal top row (pipeline_15, 0.90019): that was a fused
choice whose entire grid fits inside 0.0003 against a fold std of 0.00041, and
paired fold-by-fold against pipeline_11 it is +0.00031/+0.00015/−0.00002. The
two extra 355-column blocks buy nothing measurable, so the simpler model ships.

Format notes, since this task has no `test.csv`/`sample_submission.csv` for
ml-submit to copy: the prediction set is `input/target.tsv`, features come from
the `features/*_target.parquet` half of the frozen store, and the output is the
task's documented long format — `domain_id`, `tracking_domain_id`, 10 rows per
domain, tab separated. **`tracking_domain_id` is the tracker's own domain id,
not the 0-354 `tracker_id` that indexes the label columns**; the two are mapped
through `input/trackers.tsv`. Getting that backwards yields a well-formed file
that scores ~0.

Validated: 500,000 rows, 50,000 distinct domains (all of `target.tsv`), exactly
10 rows each, no duplicate pairs, all tracker ids valid.

Predictions are genuinely personalised rather than a dressed-up popularity list:
20,595 distinct predicted 10-sets, 316 distinct top-1 trackers, all 355
trackers used somewhere, and only a handful of domains out of 50,000 receive the
global popularity top-10. Regional structure shows up as expected (yadro.ru and
yandex.ru for `.ru`, cnzz.com for Chinese sites).

Expected leaderboard score ≈ the CV estimate of **0.900**, with two caveats
pushing in opposite directions: the refit sees 50% more data than any CV fold
(slightly optimistic CV), while target domains are marginally better connected
than the sample (median `nbr_out` mass 14 vs 8 — slightly pessimistic CV).
