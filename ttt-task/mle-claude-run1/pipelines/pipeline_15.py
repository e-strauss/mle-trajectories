"""FINAL: does anything combine with `direct`, or does it dilute? (fused choice)

The sibling runs priced each new block against pipeline_10 (0.89517):

    direct    +0.00487   (0.90004)  clear, +~0.0049 on all three folds
    nbr_frac  +0.00023   (0.89540)  positive on 3/3 folds but ~noise-sized
    trk_cooc  +0.00016   (0.89533)  positive on 3/3, very consistent, tiny
    tok_pop   -0.00153   (0.89364)  negative on 3/3 -- dropped

`direct` is in. The open question is the two marginal blocks: each is worth
about one standard deviation, which is exactly the regime where pipeline_09
caught me out -- `nbr_rec` and `nbr_2h` also looked harmless individually and
turned out to COST 0.0014 apiece once stacked, because near-redundant blocks
dilute a fixed-capacity net. Their standalone Recall@10 on covered rows
(nbr_frac 0.7505, trk_cooc 0.7503, both on the same 91.7% support as nbr_hub's
0.7641) says they are largely restating evidence already present.

So rather than guess, this fuses the four combinations into one CV run and lets
the grid answer it. Full 5-seed ensembles so every variant is directly
comparable to the leaderboard, and the winner is the configuration the
submission is regenerated from.
"""
import skrub

from common import attach_scoring, load_xy
from featureset import EXTRA, build_pred, files

COMBOS = {
    "direct":            ("direct",),
    "direct+frac":       ("direct", "nbr_frac"),
    "direct+cooc":       ("direct", "trk_cooc"),
    "direct+frac+cooc":  ("direct", "nbr_frac", "trk_cooc"),
}
# every block any variant needs is joined once; each variant SELECTS its own
X, y = load_xy(blocks=files(*sorted(EXTRA)))

pred = skrub.choose_from(
    {name: build_pred(X, y, *blocks, n_seeds=5) for name, blocks in COMBOS.items()},
    name="combo").as_data_op()
pred = attach_scoring(pred)

DESCRIPTION = ("FINAL fused choice: direct-link block alone vs combined with "
               "the marginal nbr_frac / trk_cooc blocks (5-seed ensembles)")
PARENT = "pipeline_11"
