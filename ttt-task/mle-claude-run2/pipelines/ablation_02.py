"""Ablation (fused choice): LightGBM capacity on the winning feature set.

Everything so far has used one model configuration (300 trees, 63 leaves, lr
0.05, min_child_weight 100). With ~50 features and 3.4M training rows per fold
that may well be underfit, and capacity is the one axis never varied. The
min_child_weight floor is NOT in the grid: it is not a tuning knob here but the
fix for the Newton divergence documented in common.make_model, and dropping it
reintroduces probability-0 predictions for the most popular tracker.

Whole-estimator variants rather than one choose_from per keyword, so the grid
stays a short, readable list of real configurations instead of a cross product
of 16 mostly-pointless ones.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

BLOCKS = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w", "meta", "host", "content")

MODELS = {
    "n300_l63":        dict(n_estimators=300, num_leaves=63, learning_rate=0.05),
    "n800_l63":        dict(n_estimators=800, num_leaves=63, learning_rate=0.05),
    "n800_l255":       dict(n_estimators=800, num_leaves=255, learning_rate=0.05),
    "n1500_l127_lr03": dict(n_estimators=1500, num_leaves=127, learning_rate=0.03),
}

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=BLOCKS)
    preds = {name: feats.skb.apply(make_model(**kw), y=y) for name, kw in MODELS.items()}
    pred = skrub.choose_from(preds, name="model").as_data_op()
    pred = attach_scoring(pred)

DESCRIPTION = "ABLATION: LightGBM capacity sweep on pipeline_08's features"
PARENT = "pipeline_08"
