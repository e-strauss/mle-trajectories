"""Ablation (fused choice): settle the final feature-block set in one scored run.

Every block priced so far was measured on fold 0 only, where the spread between
repeated fits is ~0.002 -- the same size as most of the deltas being compared.
This re-measures the surviving candidates under the workspace's real 3-fold CV,
all against the same folds, so the per-variant grid in results.json -> extra.grid
is directly comparable.

Variants, from the current best (pipeline_07) outward:
  base        pipeline_04's features (no meta)      -- the reference
  meta        pipeline_07                           -- current best
  meta_host   + hostname-token priors
  meta_cont   + url-category + press freedom
  meta_all    + both content blocks
  meta_graph  + 2-hop and co-occurrence smoothing
Fold-0 pricing said host/content help alone but dilute meta, and the graph
blocks do nothing; this is where that gets confirmed or overturned.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

BASE = ("prior", "tld", "nbr_out", "nbr_in", "nbr_w")
VARIANTS = {
    "base": BASE,
    "meta": BASE + ("meta",),
    "meta_host": BASE + ("meta", "host"),
    "meta_cont": BASE + ("meta", "content"),
    "meta_all": BASE + ("meta", "host", "content"),
    "meta_graph": BASE + ("meta", "hop2", "cooc"),
}

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    preds = {name: features(ctx, X, blocks=blocks).skb.apply(make_model(), y=y)
             for name, blocks in VARIANTS.items()}
    pred = skrub.choose_from(preds, name="featureset").as_data_op()
    pred = attach_scoring(pred)

DESCRIPTION = "ABLATION: feature-block subsets around pipeline_07 (fused choice)"
PARENT = "pipeline_07"
