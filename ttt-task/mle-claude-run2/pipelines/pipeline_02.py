"""+ TLD conditioning: P(tracker | top-level domain), shrunk toward the prior.

The cheapest real conditioning available, and the only one that needs no graph.
Trackers are regional: a .ru site and a .de site do not buy analytics from the
same vendors, so the TLD alone reorders the candidate list a long way. Ranking by
the shrunk P(tracker | tld) with no model at all already measured 0.7959 in
exploration round 1 against the 0.7548 prior baseline, and the block's audit
(data_exploration_4.py) is clean: no column near-perfect on its own, and coverage
matches between the modelled sample and target.tsv.

Both blocks are kept, not just the TLD one: the model needs the global prior as
the reference level the TLD lift is measured against.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld"))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "prior + P(tracker | tld) shrunk toward the global prior"
PARENT = "pipeline_01"
