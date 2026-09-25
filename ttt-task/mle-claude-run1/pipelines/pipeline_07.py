"""Add the three new graph evidence blocks built in data_exploration_7.

data_exploration_5 diagnosed the loss: only 79.7% of a domain's true trackers
appear anywhere in its 1-hop histogram, and recall is worst both where evidence
is absent (0 tracked neighbours: 0.832) and where it is drowned in hubs (>100
neighbours: 0.798). The three blocks target exactly that:

  nh_*  hub-discounted neighbourhood -- each link weighted 1/log2(2+linkdeg(nbr)),
        so a 50k-outlink directory stops outvoting a handful of real links.
  nr_*  reciprocal neighbours only (d->n AND n->d, 1.79M such pairs) -- mutual
        links are a far stronger similarity signal than one-way links.
  n2_*  2-hop reach through UNTRACKED, low-degree (<=32) intermediaries, i.e.
        strictly new evidence for the domains whose direct neighbours are all
        untracked.
  meta2 the scalars that let the model calibrate them (reciprocal count,
        untracked-neighbour count, hub-weight mass, 2-hop mass, neighbour
        link-degree summaries, own link-degree).

Model: the pipeline_06 winner (1024x512, dropout 0.2, 30 epochs) -- that sweep
found hyperparameters saturated (whole grid within 0.881-0.8895, and 60 epochs
strictly worse than 30), so nothing here is tuned and the delta is the features
alone.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from common import attach_scoring, load_xy
from models import BlockTransform, TorchMLPRanker

X, y = load_xy(blocks=("meta", "nbr_out", "nbr_in", "tld_pop",
                       "nbr_hub", "nbr_rec", "nbr_2h", "meta2"))

BLOCKS = (s.glob("no_*") | s.glob("ni_*") | s.glob("tp_*")
          | s.glob("nh_*") | s.glob("nr_*") | s.glob("n2_*"))
COUNTS = s.cols("outdeg", "indeg", "n_trk_nbr_out", "n_trk_nbr_in",
                "n_trk_nbr_tot", "n_rec_nbr", "n_untracked_lowdeg_nbr",
                "hub_weight_mass", "two_hop_mass", "mean_nbr_linkdeg",
                "max_nbr_linkdeg", "own_linkdeg")
CONTEXT = s.cols("n_labels", "host_len", "n_digits", "n_hyphens",
                 "freedom_of_the_press", "tld", "uc_category")

blocks = X.skb.select(BLOCKS).skb.apply(
    BlockTransform(mode="l1", prefixes=("no_", "ni_", "nh_", "nr_", "n2_")))
counts = X.skb.select(COUNTS).skb.apply(
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
context = (X.skb.select(CONTEXT)
           .skb.apply(skrub.TableVectorizer(cardinality_threshold=1000))
           .skb.apply(SimpleImputer(strategy="median")))
feats = blocks.skb.concat([counts, context], axis=1)

model = TorchMLPRanker(hidden=(1024, 512), dropout=0.2, lr=1e-3, epochs=30,
                       batch_size=4096, loss="softmax_ce", standardize=True,
                       seed=0)
pred = feats.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("MLP + new graph blocks: hub-discounted, reciprocal-only and "
               "2-hop-via-untracked neighbour histograms + their scalars")
PARENT = "pipeline_06"
