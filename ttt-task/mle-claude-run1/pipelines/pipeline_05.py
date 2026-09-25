"""Swap Ridge -> GPU multi-label MLP, trained on the METRIC-ALIGNED objective.

Two changes from pipeline_04, both aimed at the same place:

1. Non-linear model. The evidence blocks interact with the degree features
   (a 1-neighbour histogram deserves less trust than a 200-neighbour one) and
   trackers co-occur in bundles; a linear map cannot express either.

2. `softmax_ce` instead of independent per-tracker sigmoids. Recall@10 of a set
   S is sum_{t in S} y_t / n_true, so the optimal 10 are the largest
   E[y_t / n_true | x] -- a softmax over the 355 trackers fitted against the
   row-normalised label vector estimates exactly that, while BCE estimates
   P(y_t = 1 | x) and over-ranks trackers that live on tracker-heavy portals.
   data_exploration_6 measured this objective swap alone as worth ~+0.015 on
   fold 0, more than everything after pipeline_02 put together.

Architecture/epochs come from the data_exploration_6 sweep (fold 0 only), where
1024x512 and 2048x1024 were within 0.0002 of each other, so the smaller wins.
Same features as pipeline_04.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import ALL_BLOCKS, attach_scoring, load_xy
from models import BlockTransform, TorchMLPRanker

X, y = load_xy(blocks=ALL_BLOCKS)

BLOCKS = s.glob("no_*") | s.glob("ni_*") | s.glob("tp_*")
DEGREE = s.cols("outdeg", "indeg", "n_trk_nbr_out", "n_trk_nbr_in", "n_trk_nbr_tot")
CONTEXT = s.cols("n_labels", "host_len", "n_digits", "n_hyphens",
                 "freedom_of_the_press", "tld", "uc_category")

blocks = X.skb.select(BLOCKS).skb.apply(BlockTransform(mode="l1"))
degree = X.skb.select(DEGREE).skb.apply(
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
context = X.skb.select(CONTEXT).skb.apply(
    skrub.TableVectorizer(cardinality_threshold=1000))

feats = (blocks.skb.concat([degree, context], axis=1)
         .skb.apply(SimpleImputer(strategy="median"))
         .skb.apply(StandardScaler()))

model = TorchMLPRanker(hidden=(1024, 512), dropout=0.2, lr=1e-3, epochs=30,
                       batch_size=4096, loss="softmax_ce", seed=0)
pred = feats.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("GPU MLP 1024x512, softmax-CE on the row-normalised target "
               "(metric-aligned E[y/n]) -- same features as pipeline_04")
PARENT = "pipeline_04"
