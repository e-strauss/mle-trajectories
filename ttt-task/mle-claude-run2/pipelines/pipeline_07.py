"""+ tracker metadata: pool evidence across trackers that belong together.

Three pipelines in a row came back flat (rank features, lambdarank, every extra
graph hop), so the diagnosis had to change. The residual is concentrated
somewhere specific: on fold 0, true pairs whose tracker appears on >=1% of
domains are recovered 94.6% of the time, and rare ones only 38.5%. With ~28k
positives spread over 355 trackers, a rare tracker simply has too few examples
to be learned on its own terms.

But trackers are not independent. trackers.tsv groups the 355 into companies,
brands, categories and countries, and a domain whose neighbours all run Google
products is a good bet for an unseen Google tracker even though that tracker's
own frequency says otherwise. This block gives each candidate the
neighbourhood's mass on its company / brand / category / country, always with
the candidate's own contribution removed from both the mass and the reference
prior -- without that subtraction a single-tracker company would just restate
u_p and the block would look informative while adding nothing.

Priced at +0.0035 on fold 0 in data_exploration_7.py, the largest single-block
gain since pipeline_04.

Not included: `host` and `content` (+0.0009 together, and they DILUTE meta --
meta+host+content scored below meta alone). They get their own ablation rather
than a free ride here.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in",
                                     "nbr_w", "meta"))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "+ tracker company/brand/category/country pooling (trackers.tsv)"
PARENT = "pipeline_04"
