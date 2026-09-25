"""pipeline_03 + the meta block: does domain *context* help beyond the blocks?

New inputs on top of pipeline_03's 1065 block features:
  * log1p degrees (outdeg / indeg / #tracked neighbours) -- how much evidence
    the neighbour histogram is actually based on. l1 normalisation deliberately
    threw this away, so the model can now learn to discount a 1-neighbour
    histogram and trust a 200-neighbour one.
  * hostname shape (n_labels, host_len, n_digits, n_hyphens)
  * one-hot TLD and url-classification category, plus the press-freedom score
    of the TLD's country (both are sparse: 68% of rows have no press-freedom
    value, 98% no url category).

Everything is still leakage-free and available for an unseen hostname.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from common import ALL_BLOCKS, attach_scoring, load_xy
from models import BlockTransform

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

model = Ridge(alpha=skrub.choose_from([1.0, 30.0], name="alpha"), solver="cholesky")
pred = feats.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("Ridge on blocks + meta context (log degrees, hostname shape, "
               "one-hot TLD/url-category, press freedom)")
PARENT = "pipeline_03"
