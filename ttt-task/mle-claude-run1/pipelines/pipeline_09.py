"""Explorative fused-choice ABLATION: which evidence blocks actually earn their keep?

Twelve pipelines of additive changes leave an obvious question unanswered: the
score went up, but did every block contribute, or is one of them dead weight
that a bigger model merely tolerates? Each variant here is a complete
prediction sub-graph with exactly ONE block group removed, fused via
`choose_from(...).as_data_op()`, so a single CV run under the workspace's fixed
folds prices every block against the same reference.

The network is exactly pipeline_07's (1024x512 / 30 epochs -- pipeline_06 found
that configuration optimal anyway), and the "all" variant is pipeline_07's exact
feature set, so the "all" row should land on pipeline_07's 0.88998 and the whole
run is directly comparable to the leaderboard.

This run matters more than a routine confirmation: after the pipeline_07 leak was
fixed, the three new graph blocks were worth only +0.00044 TOGETHER, so at least
one of them is plausibly dead weight. Pricing them individually is the only way
to know, and a block that costs nothing to drop is one less thing to compute for
the submission.
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

ALL_PREF = ("no_", "ni_", "tp_", "nh_", "nr_", "n2_")
COUNTS = s.cols("outdeg", "indeg", "n_trk_nbr_out", "n_trk_nbr_in",
                "n_trk_nbr_tot", "n_rec_nbr", "n_untracked_lowdeg_nbr",
                "hub_weight_mass", "two_hop_mass", "mean_nbr_linkdeg",
                "max_nbr_linkdeg", "own_linkdeg")
CONTEXT = s.cols("n_labels", "host_len", "n_digits", "n_hyphens",
                 "freedom_of_the_press", "tld", "uc_category")

counts = X.skb.select(COUNTS).skb.apply(
    FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
context = (X.skb.select(CONTEXT)
           .skb.apply(skrub.TableVectorizer(cardinality_threshold=1000))
           .skb.apply(SimpleImputer(strategy="median")))
text = X.skb.select(s.cols("domain", "reg_dom")).skb.apply(
    skrub.StringEncoder(n_components=128, analyzer="char_wb",
                        ngram_range=(2, 4), random_state=0))


def build(prefixes, with_counts=True, with_context=True, with_text=False):
    """One complete prediction sub-graph over the chosen feature groups."""
    sel = None
    for p in prefixes:
        sel = s.glob(f"{p}*") if sel is None else sel | s.glob(f"{p}*")
    parts = []
    if with_counts:
        parts.append(counts)
    if with_context:
        parts.append(context)
    if with_text:
        parts.append(text)
    blocks = X.skb.select(sel).skb.apply(
        BlockTransform(mode="l1",
                       prefixes=tuple(p for p in prefixes if p != "tp_")))
    feats = blocks.skb.concat(parts, axis=1) if parts else blocks
    model = TorchMLPRanker(hidden=(1024, 512), dropout=0.2, lr=1e-3, epochs=30,
                           batch_size=4096, loss="softmax_ce",
                           standardize=True, seed=0)
    return feats.skb.apply(model, y=y)


# Reference = pipeline_07's feature set (the actual best: pipeline_08 showed the
# hostname text block is a -0.0033 regression, so it is OFF here and appears
# only as the "+text" variant, which should reproduce that finding).
VARIANTS = {
    "all":         build(ALL_PREF),
    "+text":       build(ALL_PREF, with_text=True),
    "-nbr_2h":     build(tuple(p for p in ALL_PREF if p != "n2_")),
    "-nbr_hub":    build(tuple(p for p in ALL_PREF if p != "nh_")),
    "-nbr_rec":    build(tuple(p for p in ALL_PREF if p != "nr_")),
    "-tld_pop":    build(tuple(p for p in ALL_PREF if p != "tp_")),
    "-nbr_out_in": build(tuple(p for p in ALL_PREF if p not in ("no_", "ni_"))),
    "-context":    build(ALL_PREF, with_context=False),
    "-counts":     build(ALL_PREF, with_counts=False),
}
pred = skrub.choose_from(VARIANTS, name="ablation").as_data_op()
pred = attach_scoring(pred)

DESCRIPTION = ("Fused-choice leave-one-block-out ablation over all evidence "
               "groups (fixed 1024x512/e30 net for comparability)")
PARENT = "pipeline_08"
