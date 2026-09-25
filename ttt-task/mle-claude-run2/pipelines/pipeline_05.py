"""+ within-domain relative features (rank, share of mass, share of max).

recall@10 only cares about the ORDER of a domain's 355 candidates, but a GBDT
sees absolute feature values: it has to reconstruct "high FOR THIS DOMAIN" out
of interactions between a score and the domain's degree. A domain with three
labelled neighbours and one with three hundred produce out_p values on totally
different scales, and the same absolute out_p means opposite things in the two.

This block hands that comparison over directly -- for each of the four main
scores (tld, out, in, and their prior-corrected product) it adds the candidate's
rank within its own domain, its share of the domain's total mass, and its share
of the domain's maximum.

Audit: the rank columns score ~0.000 standalone, which is expected and correct
-- a rank is meaningless as an absolute score across domains, it only has
meaning inside one. The share/max columns reproduce their parent scores exactly
(0.7959 / 0.8183 / 0.8130), confirming they are monotone re-expressions per
domain and add no new information on their own. The value, if any, is purely in
making the per-domain comparison cheap for the trees.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in",
                                     "nbr_w", "rank"))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "+ within-domain rank / share-of-mass / share-of-max features"
PARENT = "pipeline_04"
