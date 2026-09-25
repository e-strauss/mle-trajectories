"""Promote the ablation winner: meta + hostname tokens + content metadata.

ablation_01 re-measured every surviving block under the real 3-fold CV instead
of the single fold used to price them, and reversed the fold-0 verdict: content
blocks did NOT dilute meta.

    meta_all    0.87549     meta + host + content   <- this pipeline
    meta_host   0.87544
    meta_cont   0.87509
    meta_graph  0.87465     meta + 2-hop + co-occurrence
    meta        0.87401     pipeline_07
    base        0.87290     pipeline_04

Two things that grid settles. First, the extra graph hops are not just flat but
actively worse than the content blocks -- the link graph really is exhausted at
one hop, which is what the reachability analysis in exploration round 5 implied
and what three flat pipelines in a row had already hinted at.

Second, and worth stating plainly: `meta` scores 0.87401 here against 0.87489
for pipeline_07, which is the SAME feature set and the SAME folds. That 0.0009
gap is LightGBM's own run-to-run nondeterminism (multithreaded histogram
building, `deterministic=False`), and it sets the real floor for reading this
leaderboard. Everything between meta_all, meta_host and meta_cont is inside that
floor and should be treated as a tie; only the step up from `meta` (+0.0015) and
from `base` (+0.0026) is worth anything, and even that is marginal. The full set
is taken because it wins on the largest number of independent comparisons, not
because 0.87549 is meaningfully above 0.87544.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in",
                                     "nbr_w", "meta", "host", "content"))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "ablation winner: + hostname-token and url-category/press-freedom priors"
PARENT = "pipeline_07"
