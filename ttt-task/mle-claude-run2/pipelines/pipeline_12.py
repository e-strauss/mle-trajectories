"""FINAL: rank-blend of the GPU listwise ranker and the LightGBM classifier.

pipeline_10 blended two objectives and gained nothing, because both were the
same trees on the same features -- they made the same mistakes. This blend is
across model FAMILIES, and the difference shows up before any score is computed:
the two models' within-domain rankings correlate at only 0.754, so they
genuinely disagree about which candidates belong in a domain's top 10.

On fold 0 that disagreement is worth something in both directions -- GBDT alone
0.8740, NN alone 0.8789, blended 0.8791-0.8801 across w_nn in [0.5, 0.7]. The
weight is fixed at 2/3, not the grid's argmax, since the curve is flat over that
range and reading off a single fold's maximum would be fitting noise.

Cost: ~20s for the neural net and ~60s for LightGBM per fold, on top of ~60s of
feature building.
"""
import skrub

from common import HybridRanker, attach_scoring, features, load_xy

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in", "nbr_w",
                                     "meta", "host", "content", "trkcode"))
    feats = feats.skb.concat([X[["domain_id"]]], axis=1)
    pred = feats.skb.apply(HybridRanker(nn_weight=2 / 3), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "FINAL: rank-blend of GPU listwise ranker (2/3) + LightGBM (1/3)"
PARENT = "pipeline_11"
