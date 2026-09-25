"""Final candidate: the best feature set + the best net, seed-ensembled.

Nine pipelines of evidence say the same thing about where the remaining room is:

  * hyperparameters are exhausted -- pipeline_06's six configurations spanned
    only 0.881-0.8895 and the smallest/shortest won;
  * extra graph re-weightings are nearly exhausted -- once the 2-hop label leak
    was fixed, the hub-discounted + reciprocal + 2-hop blocks were worth
    +0.00044 together (pipeline_07);
  * hostname text actively hurts -- -0.0033 on every fold (pipeline_08).

pipeline_09's leave-one-block-out ablation then found that two of the blocks I
built are not merely useless but HARMFUL -- dropping nbr_rec is worth +0.00145
and dropping nbr_2h +0.00135, while nbr_hub is the most valuable single block
(-0.00115 to remove). Both are re-weightings of evidence nbr_hub already carries,
so they only add dilution at fixed capacity. They are dropped here.

So this reaches for no new feature and no bigger network. It takes the ablation's
winning feature set and removes the one remaining source of avoidable error:
the single-draw variance of one randomly-initialised MLP.
`n_seeds=5` trains five independent networks per fold and averages their
predicted distributions. Averaging is the right operation here because the
objective is a softmax over the 355 trackers, so each net outputs a
distribution, and a mean of distributions is itself one -- it cannot help a
single net's ranking by luck, only remove its jitter.

This is the pipeline to submit from.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from common import attach_scoring, load_xy
from models import BlockTransform, TorchMLPRanker

# nbr_rec / nbr_2h dropped on pipeline_09's evidence. Their SCALAR summaries stay
# in COUNTS below, exactly as in the measured "-nbr_rec" / "-nbr_2h" variants.
X, y = load_xy(blocks=("meta", "nbr_out", "nbr_in", "tld_pop",
                       "nbr_hub", "meta2"))

BLOCKS = (s.glob("no_*") | s.glob("ni_*") | s.glob("tp_*") | s.glob("nh_*"))
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
pred = attach_scoring(pred)

DESCRIPTION = ("FINAL: ablation-pruned features (nbr_rec/nbr_2h dropped) + "
               "5-seed ensemble of the 1024x512 softmax-CE MLP")
PARENT = "pipeline_09"
