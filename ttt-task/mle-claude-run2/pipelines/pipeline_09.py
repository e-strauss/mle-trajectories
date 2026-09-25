"""Promote the capacity-sweep winner: 800 trees at 63 leaves.

Model capacity was the one axis never varied -- every pipeline up to here used
300 trees / 63 leaves / lr 0.05. ablation_02 swept it on pipeline_08's features:

    n800_l63          0.87604   <- this pipeline
    n300_l63          0.87483   what every earlier pipeline used
    n1500_l127_lr03   0.87477
    n800_l255         0.87436

The shape of that result is more informative than the winner. Going from 300 to
800 trees at the same width helps; making the trees WIDER (255 leaves) or much
deeper-and-slower (1500 x 127) hurts, and both have visibly larger fold spread
(0.0023 / 0.0029 against 0.0013). With ~50 features, ~28k positives and 0.56%
base rate, the extra width is spent memorising individual domains. So the useful
direction is more, small trees -- and the gain from even that is +0.0012, which
is barely above the ~0.001 nondeterminism floor.

min_child_weight stays at 100 and was deliberately kept out of the grid: it is
not a tuning knob here but the fix for the Newton divergence documented in
common.make_model.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in",
                                     "nbr_w", "meta", "host", "content"))
    pred = feats.skb.apply(make_model(n_estimators=800), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "ablation_02 winner: full feature set, LightGBM 800 trees x 63 leaves"
PARENT = "pipeline_08"
