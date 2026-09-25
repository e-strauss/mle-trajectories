"""Baseline: rank candidate trackers by global popularity alone.

The floor every later pipeline has to beat. Features are the tracker's identity
and its POOL frequency, with no information about the domain at all -- so the
model can only learn one global ranking of the 355 trackers and hand it to every
domain. Exploration round 1 measured that ranking directly at recall@10 = 0.7548,
so this pipeline should land there, and all three folds should agree closely:
that is what makes it a usable check on the row design, the GroupKFold split and
the recall@10 scorer all at once.

The first version of this file used a default HistGradientBoostingClassifier and
scored 0.329 with folds [0.225, 0.004, 0.757] -- see `make_model` in common.py
and data_exploration_3.py for why (Newton-step divergence on a 0.56%-positive
matrix, not a data bug).
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior",))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "Baseline: global tracker prior only (tracker_id + POOL frequency)"
PARENT = None
