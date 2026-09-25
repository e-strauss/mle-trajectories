"""The pipeline_10 feature graph, with opt-in extra blocks (pipelines 11-15).

Pipelines 11-14 each add exactly ONE of data_exploration_9's new blocks to the
pipeline_10 baseline and are scored as SIBLINGS of it (all PARENT=pipeline_10),
not as a chain. After pipeline_09 showed that two blocks were individually
harmful yet survived unnoticed inside an additive chain, measuring each new
block's marginal value against one fixed reference is the only way to read the
result. Keeping the shared graph here means the pipelines differ by one argument.

The four new blocks are all already on a probability-like or small-count scale
(`nbr_frac`, `trk_cooc`, `tok_pop` are probabilities; `direct` is a sparse count
of hyperlinks), so none of them is l1-renormalised -- `BlockTransform` only
rescales the prefixes it is given and passes everything else through untouched.
The model standardises whatever it receives on the GPU.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

from models import BlockTransform, TorchMLPRanker

# pipeline_10's set: nbr_rec / nbr_2h stay dropped on pipeline_09's evidence
BASE_FILES = ("meta", "nbr_out", "nbr_in", "tld_pop", "nbr_hub", "meta2")
BASE_GLOBS = ("no_*", "ni_*", "tp_*", "nh_*")
L1_PREFIXES = ("no_", "ni_", "nh_")          # the raw-count blocks

# block file -> its column prefix, for the new blocks
EXTRA = {"direct": "dl_", "nbr_frac": "nf_", "trk_cooc": "tc_", "tok_pop": "kp_"}

COUNTS = s.cols("outdeg", "indeg", "n_trk_nbr_out", "n_trk_nbr_in",
                "n_trk_nbr_tot", "n_rec_nbr", "n_untracked_lowdeg_nbr",
                "hub_weight_mass", "two_hop_mass", "mean_nbr_linkdeg",
                "max_nbr_linkdeg", "own_linkdeg")
CONTEXT = s.cols("n_labels", "host_len", "n_digits", "n_hyphens",
                 "freedom_of_the_press", "tld", "uc_category")


def files(*extra):
    """`blocks=` argument for common.load_xy, base + the named new blocks."""
    for e in extra:
        if e not in EXTRA:
            raise KeyError(f"unknown block {e!r}; known: {sorted(EXTRA)}")
    return BASE_FILES + tuple(extra)


def build_pred(X, y, *extra, n_seeds=5, seed=0, epochs=30, hidden=(1024, 512)):
    """pipeline_10's plan, optionally widened by the named new blocks."""
    sel = None
    for g in BASE_GLOBS + tuple(f"{EXTRA[e]}*" for e in extra):
        sel = s.glob(g) if sel is None else sel | s.glob(g)

    blocks = X.skb.select(sel).skb.apply(
        BlockTransform(mode="l1", prefixes=L1_PREFIXES))
    counts = X.skb.select(COUNTS).skb.apply(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
    context = (X.skb.select(CONTEXT)
               .skb.apply(skrub.TableVectorizer(cardinality_threshold=1000))
               .skb.apply(SimpleImputer(strategy="median")))
    feats = blocks.skb.concat([counts, context], axis=1)

    model = TorchMLPRanker(hidden=hidden, dropout=0.2, lr=1e-3, epochs=epochs,
                           batch_size=4096, loss="softmax_ce", standardize=True,
                           n_seeds=n_seeds, seed=seed)
    return feats.skb.apply(model, y=y)
