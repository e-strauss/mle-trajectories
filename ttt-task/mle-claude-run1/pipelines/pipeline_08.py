"""Add the hostname STRING itself, as character n-grams.

Everything so far describes a domain by its graph neighbourhood and its TLD.
The hostname text is the one signal left that is available for a completely
unconnected domain, and it is far from empty: character n-grams pick up the
language ("nachrichten", "noticias"), the platform ("blogspot", "wordpress"),
and the genre ("shop", "forum", "news", "porn") -- all of which correlate with
which ad/analytics stack a site runs. It is also the only feature that can help
the 8.4% of domains with no tracked neighbour at all beyond their TLD prior.

`skrub.StringEncoder` = char_wb tf-idf n-grams + truncated SVD, so it lands as
a compact dense block (fitted per fold on the training hostnames only). The
registrable domain is encoded separately from the full hostname: for a
subdomain they differ, and reg_dom groups blog.x.com with x.com.

Model: unchanged from pipeline_07, so this isolates the text features.
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
text = X.skb.select(s.cols("domain", "reg_dom")).skb.apply(
    skrub.StringEncoder(n_components=128, analyzer="char_wb",
                        ngram_range=(2, 4), random_state=0))

feats = blocks.skb.concat([counts, context, text], axis=1)

model = TorchMLPRanker(hidden=(1024, 512), dropout=0.2, lr=1e-3, epochs=30,
                       batch_size=4096, loss="softmax_ce", standardize=True,
                       seed=0)
pred = feats.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("pipeline_07 + hostname/reg_dom character n-gram embeddings "
               "(StringEncoder 128d each)")
PARENT = "pipeline_07"
