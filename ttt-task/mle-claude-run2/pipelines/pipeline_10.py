"""Final: full feature set, 800x63 capacity, pointwise + lambdarank rank-blend.

Everything the workspace established, in one plan:

  features  the eight blocks ablation_01 selected (global prior, TLD prior, both
            raw 1-hop neighbour profiles, the four weighted 1-hop views, tracker
            company/brand/category/country pooling, hostname-token priors, and
            url-category / press-freedom)
  capacity  800 trees x 63 leaves, the ablation_02 winner
  model     rank-average of the log-loss classifier and the lambdarank ranker

The blend is the one idea left with a reason to work. pipeline_04 and
pipeline_06 tied at 0.8729 / 0.8725 on identical features, which says the two
objectives are equally good, not that they are the same model -- one fits
probabilities across domains, the other orders candidates within one. Averaging
their WITHIN-DOMAIN RANKS is the only way to combine them that recall@10 can
see, since the metric reads nothing but the per-domain order.

If this does not beat pipeline_09 the honest conclusion is that the two models
make the same mistakes, which would be a real finding about the task rather than
a failed pipeline: it would mean the residual is in the data, not the fit.
"""
import skrub

from common import BlendRanker, attach_scoring, features, load_xy

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in",
                                     "nbr_w", "meta", "host", "content"))
    feats = feats.skb.concat([X[["domain_id"]]], axis=1)
    pred = feats.skb.apply(BlendRanker(n_estimators=800), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "FINAL: full features, 800x63, rank-blend of log-loss + lambdarank"
PARENT = "pipeline_09"
