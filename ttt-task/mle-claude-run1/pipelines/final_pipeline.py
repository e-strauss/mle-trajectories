"""FINAL REFIT + SUBMISSION ARTIFACT -- not a scored candidate. Do NOT ml-score this.

Reproduces pipeline_11 (Recall@10 = 0.90004), refits it on ALL 505,548 training
rows instead of 2/3 per fold, and writes the submission for the 50,000 domains
of `input/target.tsv`.

Why pipeline_11 and not the leaderboard's nominal top row (pipeline_15,
0.90019): pipeline_15 was a fused choice over combinations, and its whole grid
spans 0.89990-0.90019 against a fold std of 0.00041. Paired against pipeline_11
fold-by-fold the nominal winner gives +0.00031 / +0.00015 / -0.00002 -- mixed
signs, no effect. So the two extra 355-column blocks (nbr_frac, trk_cooc) buy
nothing measurable, and the simpler model ships.

Two deviations from the ml-submit template, both forced by this task:

1. There is no `input/test.csv` / `sample_submission.csv`. The prediction set is
   `input/target.tsv` (domain ids only), and its features are the target half
   of the same inline build that produces the training half, with identical
   columns.

2. Features are built inline from `input/` by `features.py` (this collection
   does not cache precomputed features), rather than read from the original
   run's parquet store. `../verify_inline_features.py` checks the two agree.

3. The submission is not one value per row. The task asks for a TSV of
   `domain_id, tracking_domain_id` pairs, up to 10 rows per domain, so the
   (50000, 355) score matrix is turned into the top-10 tracker ids per domain
   and exploded to long format. `tracking_domain_id` is the tracker's own DOMAIN
   id, which is NOT the 0-354 `tracker_id` that indexes the label columns -- the
   two are mapped via `input/trackers.tsv`. Written as `submission.tsv` (tab
   separated) because the task specifies a TSV.

As the template requires, the plan roots on an overridable `skrub.var("data")`
rather than `common.load_csv`'s `skrub.as_data_op(path)` constant, so it can
predict on rows it was not built with; the target column is dropped with
`errors="ignore"` because the target frame has no label columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

import features as F
from common import INPUT_DIR
from models import BlockTransform, TorchMLPRanker

WS_ROOT = Path(__file__).resolve().parent.parent
IN = INPUT_DIR
N_TRK, K = 355, 10
YCOLS = [f"y_{i}" for i in range(N_TRK)]
# nbr_rec / nbr_2h dropped on pipeline_09's ablation; `direct` added on
# pipeline_11's +0.00487 (the largest feature gain in the workspace)
BLOCK_FILES = ("meta", "nbr_out", "nbr_in", "tld_pop", "nbr_hub", "meta2",
               "direct")


def build_all():
    """Build every block this configuration needs, once, for both halves.

    Eager rather than through `common.load_xy`'s recorded plan: this file fits
    on ALL rows and then predicts on the target half, so it needs both halves of
    the same build. The block functions are the identical ones the plan records.
    """
    ctx = F.load_context(IN)
    edges = F.seed_edges(ctx)
    linkdeg = F.link_degrees(ctx)
    labels = F.block_labels(ctx)
    blocks = {
        "meta": F.block_meta(ctx, edges),
        "nbr_out": F.block_nbr_out(ctx, edges),
        "nbr_in": F.block_nbr_in(ctx, edges),
        "tld_pop": F.block_tld_pop(ctx, labels),
        "nbr_hub": F.block_nbr_hub(ctx, edges, linkdeg),
        "direct": F.block_direct(ctx),
    }
    # meta2 needs the 2-hop mass even though the nbr_2h BLOCK is not a feature here
    blocks["meta2"] = F.block_meta2(ctx, edges, linkdeg, blocks["nbr_hub"],
                                    F.block_nbr_2h(ctx, edges, linkdeg))

    def half(split):
        df = (F.split_rows(labels, ctx, split) if split == "train"
              else F.split_rows(blocks["meta"], ctx, split)[["domain_id"]])
        for b in BLOCK_FILES:
            df = df.merge(F.split_rows(blocks[b], ctx, split),
                          on="domain_id", how="left")
        return df

    return half("train"), half("target")


print("building features from input/ (one full link-graph scan)...")
train_df, target_df = build_all()
print(f"train {train_df.shape}  target {target_df.shape}")
assert [c for c in train_df.columns if c not in YCOLS] == list(target_df.columns), \
    "train/target feature columns must match exactly"

# --- the pipeline_10 plan, rooted on an OVERRIDABLE var -------------------
data = skrub.var("data", value=train_df)
y = data[YCOLS].skb.mark_as_y()
X = data.drop(columns=YCOLS, errors="ignore").skb.mark_as_X()

# dl_* is a sparse hyperlink count, deliberately NOT l1-renormalised: the
# BlockTransform below rescales only the prefixes it is given.
BLOCKS = (s.glob("no_*") | s.glob("ni_*") | s.glob("tp_*") | s.glob("nh_*")
          | s.glob("dl_*"))
COUNTS = s.cols("outdeg", "indeg", "n_trk_nbr_out", "n_trk_nbr_in",
                "n_trk_nbr_tot", "n_rec_nbr", "n_untracked_lowdeg_nbr",
                "hub_weight_mass", "two_hop_mass", "mean_nbr_linkdeg",
                "max_nbr_linkdeg", "own_linkdeg")
CONTEXT = s.cols("n_labels", "host_len", "n_digits", "n_hyphens",
                 "freedom_of_the_press", "tld", "uc_category")

blocks = X.skb.select(BLOCKS).skb.apply(
    BlockTransform(mode="l1", prefixes=("no_", "ni_", "nh_")))
counts = X.skb.select(COUNTS).skb.apply(
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
context = (X.skb.select(CONTEXT)
           .skb.apply(skrub.TableVectorizer(cardinality_threshold=1000))
           .skb.apply(SimpleImputer(strategy="median")))
feats = blocks.skb.concat([counts, context], axis=1)

model = TorchMLPRanker(hidden=(1024, 512), dropout=0.2, lr=1e-3, epochs=30,
                       batch_size=4096, loss="softmax_ce", standardize=True,
                       n_seeds=5, seed=0)
pred = feats.skb.apply(model, y=y)

print("fitting on all training rows (5 nets x 30 epochs)...")
learner = pred.skb.make_learner(fitted=True)

# --- predict + write the submission ---------------------------------------
scores = np.asarray(learner.predict({"data": target_df}))
print("score matrix:", scores.shape)
assert scores.shape == (len(target_df), N_TRK)

top = np.argpartition(-scores, K - 1, axis=1)[:, :K]
# order each row's 10 by descending score (irrelevant to Recall@10, but makes
# the file readable and puts the most confident guess first)
row = np.arange(len(top))[:, None]
top = np.take_along_axis(top, np.argsort(-scores[row, top], axis=1), axis=1)

trk = pd.read_csv(IN / "trackers.tsv", sep="\t")
trk_to_dom = trk.set_index("tracker_id")["tracking_domain_id"]
assert trk_to_dom.index.is_unique and len(trk_to_dom) == N_TRK

sub = pd.DataFrame({
    "domain_id": np.repeat(target_df["domain_id"].to_numpy(), K),
    "tracking_domain_id": trk_to_dom.reindex(top.ravel()).to_numpy(),
})
assert sub["tracking_domain_id"].notna().all()
sub["tracking_domain_id"] = sub["tracking_domain_id"].astype("int64")

out = WS_ROOT / "submission.tsv"
sub.to_csv(out, sep="\t", index=False)

# --- sanity checks (ml-submit step 4, adapted to this task's format) -------
tgt_ids = pd.read_csv(IN / "target.tsv", sep="\t")["domain_id"]
print(f"\nwrote {out}  ({len(sub)} rows)")
print(f"  distinct domains          : {sub['domain_id'].nunique()} "
      f"(target.tsv has {tgt_ids.nunique()})")
print(f"  rows per domain           : {sub.groupby('domain_id').size().unique()}")
print(f"  every target domain covered: "
      f"{set(tgt_ids) == set(sub['domain_id'].unique())}")
print(f"  tracker ids valid         : "
      f"{sub['tracking_domain_id'].isin(trk['tracking_domain_id']).all()}")
print(f"  duplicate (domain,tracker): {int(sub.duplicated().sum())}")
print("\nhead:")
print(sub.head(12).to_string(index=False))
