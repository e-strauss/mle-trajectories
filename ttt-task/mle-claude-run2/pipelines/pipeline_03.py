"""+ link-graph neighbourhood: what trackers the domains around this one use.

The first feature that actually looks at the individual domain rather than the
class it belongs to. Sites that link to each other share owners, CMSs, ad
networks and agencies, so a domain's out-links (who it points at) and in-links
(who points at it) each carry a tracker profile.

Both directions are added at once rather than one per pipeline: exploration
round 1 measured them at 0.5164 and 0.5773 standing alone and 0.8145 / 0.8068
blended with the prior, i.e. they are two views of the same neighbourhood and
splitting them across two pipelines would mostly measure their overlap. The
ablation in pipeline_04 separates their individual contributions properly.

Audited in data_exploration_4.py: the strongest column is out_p at 0.8183
standalone -- strong but nowhere near the ceiling -- and coverage parity against
target.tsv holds (0.698 vs 0.703 of rows with a labelled out-neighbour). The
neighbour labels come from POOL only, which excludes every modelled domain, so
no d -> n -> d walk can return a domain its own label.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in"))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "prior + tld + out/in link-neighbour tracker profiles"
PARENT = "pipeline_02"
