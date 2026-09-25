"""Explorative fused-choice run: how much signal is in each evidence block?

No learning beyond the global prior -- just a weighted blend of the three
precomputed 355-wide score blocks, so the grid directly measures the standalone
and combined value of:
    no_*  trackers of the domains this domain links TO
    ni_*  trackers of the domains that link TO this domain
    tp_*  leave-one-out P(tracker | TLD)
    prior global tracker frequency (= pipeline_01)
`l1` turns each block into a distribution over the 355 trackers before blending
(so the blocks are on a comparable scale regardless of a domain's degree);
`log1p` is the alternative squashing of raw counts. The winner tells later
pipelines which blocks are worth feeding a real model.
"""
import skrub

from common import attach_scoring, load_xy
from models import BlendRanker

X, y = load_xy(blocks=("nbr_out", "nbr_in", "tld_pop"))

# (w_out, w_in, w_tld, w_pop, transform) -- hand-picked, informative variants
CONFIGS = {
    "out_only":       (1.0, 0.0, 0.0, 0.0, "l1"),
    "in_only":        (0.0, 1.0, 0.0, 0.0, "l1"),
    "tld_only":       (0.0, 0.0, 1.0, 0.0, "l1"),
    "pop_only":       (0.0, 0.0, 0.0, 1.0, "l1"),
    "out+in":         (1.0, 1.0, 0.0, 0.0, "l1"),
    "out+in+pop":     (1.0, 1.0, 0.0, 0.5, "l1"),
    "out+in+tld":     (1.0, 1.0, 1.0, 0.0, "l1"),
    "all_even":       (1.0, 1.0, 1.0, 0.5, "l1"),
    "all_tldheavy":   (1.0, 1.0, 2.0, 0.5, "l1"),
    "all_popheavy":   (1.0, 1.0, 1.0, 2.0, "l1"),
    "all_nbrheavy":   (3.0, 3.0, 1.0, 0.5, "l1"),
    "log_out+in":     (1.0, 1.0, 0.0, 0.0, "log1p"),
    "log_all":        (1.0, 1.0, 3.0, 3.0, "log1p"),
}
model = skrub.choose_from(
    {name: BlendRanker(w_out=a, w_in=b, w_tld=c, w_pop=d, transform=t)
     for name, (a, b, c, d, t) in CONFIGS.items()},
    name="blend")

pred = X.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("Fused-choice blend of link-neighbour / TLD / popularity score "
               "blocks (no learned weights) -- measures each block's value")
PARENT = "pipeline_01"
