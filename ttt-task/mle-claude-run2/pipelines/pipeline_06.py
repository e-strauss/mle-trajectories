"""Same features as pipeline_04, ranking objective instead of log-loss.

pipeline_05 showed that handing the trees explicit within-domain ranks as
FEATURES does nothing (-0.0009, inside the fold noise). That is a hint about
where the mismatch actually is: the problem is not that the model cannot see the
per-domain comparison, it is that the pointwise log-loss objective never asks
for one. It spends capacity making probabilities comparable ACROSS domains,
which recall@10 never looks at.

lambdarank asks the right question directly -- it only compares candidates
within a domain, and truncating it at 10 concentrates the gradient on exactly
the cut the metric takes. Feature set is held identical to pipeline_04 so the
difference measured here is the objective and nothing else.
"""
import skrub

from common import LambdaRanker, attach_scoring, features, load_xy

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in", "nbr_w"))
    # the ranker needs its grouping key in X and drops it from the features itself
    feats = feats.skb.concat([X[["domain_id"]]], axis=1)
    pred = feats.skb.apply(LambdaRanker(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "pipeline_04 features, LGBMRanker lambdarank@10 instead of log-loss"
PARENT = "pipeline_04"
