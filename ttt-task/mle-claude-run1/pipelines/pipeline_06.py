"""Explorative fused-choice run: how much is left in the MLP's hyperparameters?

Same features as pipeline_05; only the network configuration varies. The grid
deliberately contains a variant equal to pipeline_05's configuration
("e30_1024x512_d20"), because this pipeline also moves standardisation off the
plan and into the model:

  pipeline_05 spent ~60s per fold in a SimpleImputer + StandardScaler over the
  ~1320-column float64 frame -- affordable once, but it is re-run for every
  (variant, fold) pair of a grid, and would have dominated this run. The model
  now z-scores on the GPU from the training fold's own statistics (identical
  computation, still refit per fold, still leakage-free), and the imputer runs
  only on the 7 context columns that can actually be null.

So the reference variant doubles as the check that the refactor is
score-neutral: it should land on pipeline_05's 0.88953 +- noise.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

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
context = (X.skb.select(CONTEXT)
           .skb.apply(skrub.TableVectorizer(cardinality_threshold=1000))
           .skb.apply(SimpleImputer(strategy="median")))
feats = blocks.skb.concat([degree, context], axis=1)

CONFIGS = {
    "e30_1024x512_d20":  dict(hidden=(1024, 512), epochs=30, dropout=0.2),
    "e60_1024x512_d20":  dict(hidden=(1024, 512), epochs=60, dropout=0.2),
    "e30_1024x512_d10":  dict(hidden=(1024, 512), epochs=30, dropout=0.1),
    "e60_2048x1024_d30": dict(hidden=(2048, 1024), epochs=60, dropout=0.3),
    "e30_2048x1024_d30": dict(hidden=(2048, 1024), epochs=30, dropout=0.3),
    "e60_2048x1024_d30_lr3e3": dict(hidden=(2048, 1024), epochs=60, dropout=0.3,
                                    lr=3e-3),
}
model = skrub.choose_from(
    {name: TorchMLPRanker(loss="softmax_ce", standardize=True, seed=0, **cfg)
     for name, cfg in CONFIGS.items()},
    name="mlp")

pred = feats.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("Fused-choice MLP hyperparameter sweep (width/depth/dropout/lr/"
               "epochs) on pipeline_05's features, GPU-side standardisation")
PARENT = "pipeline_05"
