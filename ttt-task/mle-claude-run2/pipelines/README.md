# TrackTheTrackers — workspace summary

Predict, for each of 50,000 held-out domains, which of 355 third-party trackers
it uses. Scored by **recall@10** averaged over domains.

This is an end-to-end task: `input/` contains no design matrix, only a 36.7M-row
tracking graph over 18.68M domains, a 623M-edge link graph, 46.3M hostnames and
three small metadata files. The rows, the target and every feature are built
inside the plan.

---

## 1. The setup, and why it is what it is

### Row design — one row per (domain, candidate tracker), all 355 candidates

```
SAMPLE : domain_id % 1301 == 7   ->  14,335 domains x 355 trackers = 5,088,925 rows
label  : 1 if that domain uses that tracker
```

Carrying the **full 355-tracker candidate set** rather than generating candidates
first means there is no candidate-generation recall ceiling baked into every
score: recall@10 measured here is exactly the metric the leaderboard computes.
The alternative sized to the same 5M rows — top-120 candidates over 42k domains
— would have capped every pipeline at 0.958 and made candidate generation a
second thing to tune.

`domain_id` is alphabetical rank over registrable hostnames, so `% 1301` is an
effectively random deterministic sample, reproducible from code with no sample
file on disk.

**Is the sample representative of target.tsv?** Exploration round 2 checked, and
found a real but small shift: target domains carry more link edges (mean outdeg
36.6 vs 21.2), though every quantile up to the 90th nearly matches and the
fraction with zero out-links is identical (0.272 vs 0.269) — the gap lives in the
extreme tail. Reweighting the sample's per-stratum recall to the target's degree
profile moves the baseline from 0.7548 to ~0.750. That is a ~0.005 shift in the
absolute number and does not reorder anything, so the uniform sample was kept.
Expect the real score to land slightly below the CV estimate for this reason.

### CV — `GroupKFold(3, shuffle, seed 42)` on `domain_id`

A domain contributes 355 rows. Splitting them across folds would make recall@10
uncomputable for that domain and would leak. Set once in `common.make_cv()` and
wired through `mark_as_X(split_kwargs=...)`.

### Metric — a custom plan scorer, locked as `plan:recall@10`

recall@10 is not an sklearn scorer string: it needs the per-domain grouping key.
It is declared in the plan as a bare callable `(estimator, X, y)`, because
sklearn hands such a scorer the test fold's **marked X** — which still carries
`domain_id` and `n_true` even though the model never sees them (they are dropped
from the features, the same trick the guide uses for `sample_weight`).

The denominator is the domain's **full** true tracker count, so any candidate
that a future pipeline drops is penalised honestly. Under the all-355 design that
happens to equal the positives present in the row set, but the scorer does not
rely on it.

### Label provenance — the one invariant

Every feature here is built from *other domains' labels*, which is precisely the
leakage route the marks do not close: the tracking graph is a constant to skrub,
so a feature that reaches a modelled row's own label leaks identically in every
fold and the CV looks perfectly clean.

The invariant, enforced in `load_context` and asserted in `data_exploration_4.py`:

```
POOL = tracking-graph rows whose domain_id is NOT in the modelled sample
```

POOL is provably disjoint from every modelled row, so no walk — one hop, two
hops, TLD aggregate, co-occurrence — can reach a modelled domain's own label, and
no `d -> n -> d` return path exists (the focused link graph also has zero
self-loops). Every label-derived statistic reads POOL and never the full graph.
This also mirrors reality: target.tsv domains were removed from the tracking
graph entirely, so POOL-style statistics are exactly what is available at
submission time.

`data_exploration_4.py` is the reusable audit and was run on every block before
it was used: standalone recall@10 per column, coverage/mass parity against
target.tsv, and the provenance assertion. No block came back near-perfect
standalone (the strongest single column is `sp_p` at 0.8344), and no `_p` column
showed a coverage mismatch. The `_c` count columns do flag one (0.267 vs 0.325
non-zero) — that is the known target degree shift, not a leak, and the
normalised columns match to three decimals.

---

## 2. Leaderboard

Metric `plan:recall@10`, CV `GroupKFold(3, shuffle, seed 42)` on `domain_id`.
Because the split is fixed workspace-wide, the fold scores are pairwise comparable.
`ablation_*` entries are fused-choice exploration runs, not lineage nodes.

| pipeline | recall@10 | std | fold scores | parent | what it added |
|---|---|---|---|---|---|
| pipeline_13 | 0.88078 | 0.00123 | 0.8790, 0.8816, 0.8817 | pipeline_12 | 8-seed GPU ensemble (4x A100) + LightGBM — **no gain** |
| **pipeline_12** | **0.88068** | **0.00061** | 0.8802, 0.8803, 0.8815 | pipeline_11 | **RECOMMENDED**: rank-blend of GPU ranker (2/3) + LightGBM (1/3) |
| pipeline_11 | 0.87882 | 0.00123 | 0.8789, 0.8773, 0.8803 | pipeline_09 | GPU listwise ranker: embeddings + 355-way softmax |
| _ablation_03_ | _0.87882_ | 0.00123 | 0.8789, 0.8773, 0.8803 | pipeline_09 | _GPU ranker architecture sweep (5 variants)_ |
| pipeline_10 | 0.87658 | 0.00079 | 0.8769, 0.8755, 0.8774 | pipeline_09 | rank-blend of log-loss + lambdarank |
| pipeline_09 | 0.87634 | 0.00228 | 0.8735, 0.8764, 0.8791 | pipeline_08 | 800 trees x 63 leaves (capacity) |
| _ablation_02_ | _0.87604_ | 0.00133 | 0.8742, 0.8767, 0.8772 | pipeline_08 | _capacity sweep (4 variants)_ |
| _ablation_01_ | _0.87549_ | 0.00182 | 0.8749, 0.8736, 0.8780 | pipeline_07 | _feature-block subsets (6 variants)_ |
| pipeline_08 | 0.87503 | 0.00193 | 0.8735, 0.8738, 0.8778 | pipeline_07 | + hostname tokens, url-category, press freedom |
| pipeline_07 | 0.87489 | 0.00151 | 0.8742, 0.8735, 0.8770 | pipeline_04 | + tracker company/brand/category/country pooling |
| pipeline_04 | 0.87290 | 0.00098 | 0.8718, 0.8727, 0.8742 | pipeline_03 | + weighted 1-hop views (AA / specificity / reciprocal / same-TLD) |
| pipeline_06 | 0.87253 | 0.00099 | 0.8739, 0.8720, 0.8717 | pipeline_04 | lambdarank@10 instead of log-loss |
| pipeline_05 | 0.87202 | 0.00193 | 0.8698, 0.8718, 0.8745 | pipeline_04 | + within-domain rank / share features |
| pipeline_03 | 0.85900 | 0.00190 | 0.8566, 0.8591, 0.8612 | pipeline_02 | + out/in link-neighbour tracker profiles |
| pipeline_02 | 0.79487 | 0.00145 | 0.7928, 0.7961, 0.7957 | pipeline_01 | + P(tracker \| TLD) |
| pipeline_01 | 0.75478 | 0.00310 | 0.7504, 0.7572, 0.7568 | - | baseline: global tracker prior only |

**The gain comes in two bursts, with a long plateau between.** Features carry
0.755 -> 0.795 -> 0.859 -> 0.873 (+0.118) in four steps. Then six pipelines and
two ablations of feature and hyperparameter work add +0.004 in total. The
plateau breaks only with a change of model family: the GPU listwise ranker and
its blend with LightGBM add another +0.004 between them (0.87658 -> 0.88068),
matching in two pipelines everything the preceding six achieved.

### The noise floor is ~0.001, and it is not the CV

Two independent runs of the *same* configuration on the *same* folds disagree by
about 0.001:

| configuration | run A | run B | gap |
|---|---|---|---|
| base + meta | pipeline_07 0.87489 | ablation_01 `meta` 0.87401 | 0.00088 |
| full features, 300x63 | pipeline_08 0.87503 | ablation_02 `n300_l63` 0.87483 | 0.00020 |

That is LightGBM's multithreaded histogram building (`deterministic=False`), not
fold-to-fold variance -- the folds are identical in both runs. It means no
difference below ~0.001 in this table is real, which covers pipeline_07 vs
pipeline_08 vs pipeline_09 vs pipeline_10 as a group. Rerunning with
`deterministic=True, force_row_wise=True` would remove it at some cost in fit
time, and would be worth doing before trusting any further small comparison.


---

## 3. What was learned

### What worked

1. **TLD conditioning (+0.040).** Trackers are regional. P(tracker | TLD) shrunk
   toward the global prior is the single cheapest large gain and needs no graph.
2. **1-hop link neighbourhood (+0.064).** The largest gain by far. Sites that
   link to each other share owners, CMSs and ad networks. Both directions
   matter; in-links cover more domains (83% have a labelled in-neighbour vs 70%
   for out-links) but out-links are not redundant.
3. **Weighting the same neighbours (+0.014).** Discounting hubs (Adamic-Adar,
   1/log(2+degree)), discounting cluttered portals (1/log(2+tracker count)),
   restricting to reciprocal links and to same-TLD neighbours. Re-reading the
   same edges more carefully was worth more than any new edge.
4. **Pooling trackers through trackers.tsv (+0.002).** The last idea that beat
   noise, and the only one aimed at a diagnosed problem rather than guessed --
   see below.
5. **More, smaller trees (+0.001).** 300 -> 800 trees at 63 leaves. Wider (255
   leaves) or deeper-and-slower (1500 x 127) both did worse AND had 2-3x the
   fold spread.

### What did not work, and what that tells you

- **Within-domain rank features (-0.001).** Handing the trees explicit per-domain
  ranks changed nothing.
- **The lambdarank objective (-0.000).** Optimising the actual ranking metric
  instead of log-loss also changed nothing. Taken with the previous point: the
  model is not the bottleneck.
- **Every extra graph hop.** 2-hop neighbours, co-citation siblings,
  co-occurrence smoothing and neighbour-extremes priced at between -0.0007 and
  +0.0011 on fold 0, and `meta_graph` came *last but one* in the 3-fold ablation.
  Exploration round 5 had already bounded them: of the pairs pipeline_03 misses,
  2-hop can reach only 10.7% and co-citation 6.1%, while **66.5% are already
  inside the 1-hop neighbourhood** -- present in the features, just not ranked
  top-10. The link graph is exhausted at one hop.
- **The blend (+0.000 mean, but std 0.0023 -> 0.0008).** The pointwise and
  lambdarank models make largely the *same* mistakes. That is a statement about
  the task, not a failed pipeline: the residual is in the data, not the fit.
  Worth keeping anyway for the variance reduction alone.

### The two diagnoses that mattered

**Where the misses are.** On fold 0, true pairs whose tracker appears on >=1% of
domains are recovered **94.6%** of the time; rare trackers only **38.5%**, and
they carry 76.5% of all misses. With ~28k positives spread over 355 trackers a
rare tracker has too few examples to be learned on its own terms. That diagnosis
is what produced the `meta` block (pool evidence across a tracker's company,
brand, category and country) instead of another graph feature -- and it was the
last thing that beat noise.

**Is it a sample-size problem?** The obvious response to "rare trackers have too
few examples" is more rows, so exploration round 7 measured it directly:
hold the fold-0 test domains fixed and grow the training set from other
`domain_id % 1301` classes.

| training domains | rows | positives | recall@10 |
|---|---|---|---|
| 9,556 | 3.4M | 19k | 0.8718 |
| 24,056 | 8.5M | 47k | 0.8690 |
| 52,659 | 18.7M | 104k | 0.8690 |
| 110,059 | 39.1M | 216k | 0.8665 |

> **RETRACTED — see section 6.** This experiment was wrong, and its conclusion
> ("more rows will not help") was the wrong call. Two flaws in the design, both
> pushing the curve down: the extra training domains were left inside POOL (so
> their TLD and hostname-token features carried a trace of their own labels,
> which the test rows did not), and — far worse — they were outside FOCUS, so
> the link graph had been filtered to exclude their edges and **every neighbour
> feature on every added row was zero**. The curve was measuring the effect of
> padding the training set with feature-poor rows. Redone correctly in section
> 6, both models improve steadily out to 8x and are still climbing.

### Two bugs worth carrying forward

**Newton divergence, not a data bug (cost: 0.43 recall).** The first pipeline_01
used a default `HistGradientBoostingClassifier` and scored **0.329** with fold
scores [0.225, 0.004, 0.757] -- impossible for a plan whose every feature is a
function of `tracker_id`, since it can only express one global ranking. At a
0.56% base rate the initial raw prediction is logit(0.0056) ~ -5.2, and for the
most popular tracker (present on 59% of domains) the leaf's Newton step -G/H has
H = sum p(1-p) ~ 0: the step overshoots to p~1, the hessian collapses the other
way, and the leaf oscillates out to raw = **-2091**, i.e. probability exactly
zero for the single most useful tracker. Missing it alone costs ~0.5 recall.
HistGB and LightGBM diverge identically at their defaults -- it reproduces in
twenty lines of numpy (`data_exploration_3.py`). `min_child_weight=100`, a floor
on the per-leaf hessian sum, fixes it at the cause and is documented in
`common.make_model` as not-a-tuning-knob.

**`n_jobs=-1` was 48x slower than `n_jobs=16`.** LightGBM parallelises histogram
building over *features*, and these plans have a few dozen; with 64 threads the
rest busy-wait. Measured on pipeline_01's 4-feature matrix: 343s vs 7.1s per fit,
same score to four decimals.

**A determinism trap that was caught before it scored.** `block_meta` first
encoded tracker country/category with `pl.Categorical.to_physical()`, which
assigns codes in order of first appearance -- so a train fold and a test fold
with different row orders would encode the same country as different numbers.
Replaced with a rank over the sorted distinct values of `trackers.tsv`, and
`data_exploration_4.py` now asserts row-order independence.

### Where this pointed next

Sections 1-3 cover pipelines 01-10, which end on a plateau at ~0.877. The
conclusions that survived that plateau, and the one that pointed out of it:

- Rerun the top LightGBM pipelines with `deterministic=True` before trusting
  their order (pipelines 11-12 are already reproducible).
- The ceiling is not obviously data-limited: 66.5% of misses are *present* in the
  1-hop features. What is missing is a way to trust a single rare-tracker witness,
  which is a per-tracker calibration problem, not a feature-coverage problem —
  so the promising direction is a model that shares parameters ACROSS trackers
  rather than another feature. **Section 4 acts on exactly this** and is where
  the plateau finally breaks.
- Absolute expectation on the real target: slightly **below** the CV estimate,
  ~0.005, because target.tsv is mildly shifted toward high-degree domains and
  recall falls with degree (0.778 at outdeg 0 down to 0.594 at outdeg >= 300).

---

## 4. Breaking the plateau on GPU (pipelines 11-12)

Pipelines 05-10 established that the tree model was not the bottleneck:
within-domain rank features, the lambdarank objective, every extra graph hop and
a full capacity sweep together moved the score by +0.004. What finally moved it
was changing the model family, on hardware that had been sitting idle.

### The model

`TorchListRanker` (in `common.py`), one A100, ~20s per fold — **faster than the
LightGBM it beats**, since a training fold is only ~0.7 GB of float32 and stays
resident on the GPU. Two departures from everything before it, both aimed at the
diagnosed weakness rather than at general capacity:

**Learned embeddings instead of hand-built pooling.** Every candidate's logit is
built from embeddings of the tracker's identity *and* of its company, brand,
category and country (`block_trkcode` supplies fold-independent codes). A rare
tracker's score is therefore partly its company's embedding, trained on every
domain that uses *any* of that company's trackers. This is pipeline_07's `meta`
block — which bought +0.002 by hand-computing company/brand/category masses —
with the weights fitted instead of specified.

**A listwise softmax over all 355 candidates.** Each domain is exactly one list,
so the loss is a softmax across the whole list with soft targets `y / n_true`:
put the domain's probability mass on its true trackers, in competition with the
other 354. pipeline_06 showed lambdarank ties with log-loss; a full-list softmax
is a stronger form of the same idea, and it is only cheap because the 355-wide
list fits naturally on a GPU as a `(domains, 355, features)` tensor.

`ablation_03` swept the schedule and found less is more — the model's value is in
its embeddings, not its capacity:

| variant | recall@10 |
|---|---|
| **e40** | **0.87882** |
| e80_drop02 | 0.87772 |
| e80 | 0.87702 |
| e80_emb64 | 0.87458 |
| e120 | 0.87242 |

A wider 512x3 MLP also scored below the 256x2 default in the fold-0 probe
(0.8762 vs 0.8789).

### The blend is where the rest of the gain is

pipeline_10 blended two objectives and gained nothing, because both were the
same trees on the same features. The NN-plus-GBDT blend is across *families*, and
the difference is visible before any score: their within-domain rankings
correlate at only **0.754**. On fold 0:

| w_nn | 0.0 (GBDT) | 0.3 | 0.5 | 0.6 | 0.7 | 1.0 (NN) |
|---|---|---|---|---|---|---|
| recall@10 | 0.8740 | 0.8779 | 0.8791 | 0.8794 | 0.8801 | 0.8789 |

The weight is fixed at **2/3**, not the grid's argmax of 0.7 — the curve is flat
from 0.5 to 0.7, so "weight the stronger model twice as much" is a principled
choice while reading off one fold's maximum would be fitting noise.

pipeline_12 lands at **0.88068**, ahead of pipeline_11 on all three folds
(+0.0013, +0.0030, +0.0013) and with the smallest fold spread in the workspace
(std 0.00061).

### Two notes on reading these numbers

- **The neural pipelines are reproducible; the tree ones are not.** pipeline_11
  reproduces `ablation_03`'s winner to five decimals (0.87882 both times), where
  identical LightGBM configurations differed by up to 0.00088 between runs. Torch
  with a fixed seed is deterministic here; LightGBM's threaded histogram building
  is not.
- **This does not overturn "GBDTs win on tabular data."** It is a ranking problem
  with a 355-way structured output and a severe long tail, which is exactly the
  shape where embeddings earn their keep — and even so, the blend beats the
  neural net alone, so the trees are still contributing.

### Where to go next with the GPU

- Only ONE of the four A100s was used, and never above a fraction of it. The
  obvious spend is not a bigger net but **seed-averaging**: 4-8 TorchListRanker
  fits with different seeds, rank-averaged, in parallel across the four cards
  for roughly the wall-clock of one.
- The learning curve said more *rows* do not help a tree (flat to 8x). That was
  measured on the GBDT and should be re-measured for the embedding model, whose
  per-tracker parameters have a much better reason to want more data.
- A two-tower factorisation (domain representation dot tracker embedding) would
  push the embedding idea further and would let tracker representations be
  pretrained on the full 18.7M-domain POOL, not just the 14k modelled domains.


---

## 5. Seed-averaging across four GPUs: a confirmed dead end

With three A100s idle, the obvious next spend was not a bigger network but more
of the same one. The premise: a listwise softmax fitted on 9.5k training lists,
with an embedding table whose gradients come from a long-tailed label
distribution, should be a high-variance estimator worth averaging.

**The premise is false**, and one table settles it — the within-domain rank
correlation between independently fitted members:

| member vs. the reference net | rank correlation |
|---|---|
| different seed (B) | 0.9943 |
| different seed (C) | 0.9943 |
| 60 epochs, dropout 0.25 | 0.9924 |
| 25 epochs | 0.9850 |
| emb 16 / hidden 128 | 0.9794 |
| emb 64 / hidden 384 / 3 layers | 0.9785 |
| **LightGBM** | **0.7539** |

Changing the seed, the width, the depth *and* the schedule all leave the model
ranking candidates essentially identically. There is nothing to average:

| ensemble (fold 0) | recall@10 |
|---|---|
| 1 / 2 / 4 / 8 / 16 seeds | 0.8789 / 0.8784 / 0.8780 / 0.8790 / 0.8781 |
| 3 architectures | 0.8787 |
| all 7 NN variants | 0.8790 |
| single net | 0.8789 |

Confirmed under the full 3-fold CV as pipeline_13: **0.88078 vs pipeline_12's
0.88068, i.e. +0.0001** — below the noise floor, not consistent across folds
(one fold is worse), and with double the fold spread (std 0.00123 vs 0.00061)
for 8x the compute. **pipeline_12 remains the recommended model.**

The multi-GPU machinery itself works fine — `SeedAveragedListRanker` drives all
four cards from one process with threads (the training loop is CUDA kernel
launches, which release the GIL), 8 seeds in ~29s wall-clock against ~20s for
one. It is the averaging that is worthless, not the parallelism.

### The useful thing this bought

That correlation table is the real explanation for why pipeline_12's blend
worked, and it is not "ensembling helps". It is that the NN and the GBDT
genuinely disagree, at 0.754, while two neural nets agree at 0.994. Ensembling
pays exactly in proportion to disagreement, and this workspace now has both ends
of that scale measured on the same data.

It also closes off a whole direction: "more of the same model" is done. Anything
further has to change what the model *sees*, not how many copies of it vote.

### Where the GPUs could still earn their keep

Both remaining ideas share one property the ensembles lacked — they give the
model information it does not currently have:

- **Two-tower with POOL-pretrained tracker embeddings.** Right now the tracker
  embeddings are learned from 9.5k training domains. The co-occurrence structure
  of all **18.7M** POOL domains is sitting unused and is exactly the data that
  would pin down a rare tracker's representation. Pretrain the 355 embeddings on
  POOL co-occurrence, then fine-tune. This is the one idea with a clear
  mechanism against the diagnosed bottleneck.
- ~~Re-measure the learning curve for the embedding model.~~ **Done — section
  6.** It reversed round 7's result outright: more rows help both models, a lot.


---

## 6. The learning curve, redone — and round 7 was wrong

Section 3 reported that more training rows do not help. That was the most
consequential claim in this write-up and it was **wrong**. Redone with a correct
design, holding fold 0's 4,779 test domains fixed and growing the training set
from other `domain_id % 1301` classes:

| training domains | rows | positives | **NN** | **GBDT** | round 7 (flawed) |
|---|---|---|---|---|---|
| 9,556 (1x) | 3.4M | 19k | 0.8789 | 0.8741 | 0.8718 |
| 24,056 (2x) | 8.5M | 47k | 0.8811 | 0.8763 | 0.8690 |
| 52,659 (4x) | 18.7M | 104k | 0.8853 | 0.8777 | 0.8690 |
| 110,059 (8x) | 39.1M | 216k | **0.8875** | **0.8805** | 0.8665 |

**+0.0086 for the neural net and +0.0064 for the GBDT from 1x to 8x, still
climbing at the last point** — roughly +0.002 to +0.004 per doubling with no
sign of saturation. Round 7 had it not merely flat but *inverted*.

### What round 7 got wrong

Two flaws, both pushing the curve down, and both growing with the number of
extra domains — which is exactly what manufactured a declining curve:

1. **The added rows had no neighbour features at all.** `load_context` filters
   the 623M-edge link graph to FOCUS = the modelled sample plus target.tsv. The
   extra training domains were never in FOCUS, so *every* neighbour column on
   *every* added row was zero. Since the neighbourhood blocks are the single
   largest source of signal in this workspace (+0.064 at pipeline_03), round 7
   was padding the training set with feature-poor rows and measuring the damage.
2. **The added rows had slightly leaky features.** They were left inside POOL,
   so the TLD prior and hostname-token prior each included a trace of their own
   labels — up to ~2% of a minimum-size token's statistic. The test rows, being
   outside POOL, had no such trace, so the model learned to over-trust exactly
   the columns that would not hold up at prediction time.

`data_exploration_8.py` rebuilds POOL, the focused edge set and all features
from scratch at every curve point, so train and test rows are constructed
identically and POOL excludes every modelled domain — the workspace invariant,
which round 7 quietly broke. The 1x point reproduces the workspace's own fold-0
numbers to four decimals (NN 0.8789, GBDT 0.8741), which is the check that the
rebuilt context is faithful to `load_context`.

### What this means for the row design

The locked 14,335-domain / 5M-row design is **not** past the point of returns;
it is the binding constraint on this workspace. At 8x the neural net reaches
0.8875 on fold 0 against 0.8789 at 1x — larger than every feature and model
improvement after pipeline_04 put together.

Scores are only comparable within a row design, so acting on this means a fresh
workspace, not a 14th pipeline here. What that workspace should change:

- **Model ~110k+ domains** rather than 14k. Cost is mild and the GPU absorbs it:
  the 8x fit took 182s against 36s at 1x, and the GBDT 420s against 60s — the
  neural net's advantage over the trees actually *widens* with scale (+0.0048 at
  1x, +0.0070 at 8x).
- **Reconsider all-355 candidates at that size.** Rows = domains x 355, so 1M
  domains is 355M rows and ~74 GB of float32 — at the edge of one A100. Beyond
  roughly 200k domains, candidate generation stops being the needless
  self-handicap it would have been at 14k domains and becomes the thing that
  buys another order of magnitude of domains. The 0.958 ceiling of a top-120
  candidate set is far above where this task currently sits.
- **Re-run the curve further out** before settling on a size; 16x and 32x cost
  minutes, and nothing here suggests 8x is the end of it.

### The general lesson

Both flaws were invisible in the result — the curve looked smooth, plausible and
consistent with the well-known prior that gradient-boosted trees saturate on
tabular data. It agreed with expectations, so it was not checked. The tell was
available and ignored: a *declining* curve is not a normal saturation shape, and
should have prompted a look at what the added rows actually contained.


---

### Submission

None was requested, so none was written. The **ml-submit** skill is the route,
but note it will need adapting: its template roots `final_pipeline.py` on a
single overridable `skrub.var("data", value=train_df)`, whereas these plans root
on `skrub.as_data_op(INPUT)` and build rows for a *deterministic id sample*. To
predict on target.tsv the row builder has to take the domain id list as the
variable -- `build_rows` and `load_context` are already split so that only
`sample_ids` needs to become an input.

