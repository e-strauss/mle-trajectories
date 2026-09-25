"""+ weighted views of the same 1-hop neighbourhood.

Residual analysis (data_exploration_5.py) found the payoff is here, not further
out in the graph: 66.5% of the pairs pipeline_03 misses are ALREADY inside the
1-hop neighbourhood -- the right tracker is among the neighbours' trackers, it
just does not reach the top 10. By comparison a 2-hop walk reaches 10.7% of the
misses and co-citation 6.1%. So this pipeline re-reads the same neighbours with
weights that discount the uninformative ones:

  * Adamic-Adar (1/log(2+global degree)) -- a 50k-outlink hub is not evidence
    about any single domain;
  * specificity (1/log(2+tracker count))  -- a portal carrying 30 trackers is
    weak evidence for any one of them;
  * reciprocity -- mutual links (11% of the focused edges) mean a real
    relationship, not a directory listing;
  * the undirected union, which covers more domains than either direction;
  * same-TLD neighbours only, since vendors are regional.

Audit (data_exploration_4.py): best standalone column is sp_p at 0.8344, against
out_p's 0.8183 -- an improvement, nowhere near the ceiling. The `_c` count
columns flag a coverage difference vs target.tsv (0.267 vs 0.325 nonzero); that
is the known mild degree shift of the target sample measured in exploration
round 2, and the normalised `_p` columns match to three decimals.
"""
import skrub

from common import attach_scoring, features, load_xy, make_model

with skrub.config_context(eager_data_ops=False):
    ctx, X, y = load_xy()
    feats = features(ctx, X, blocks=("prior", "tld", "nbr_out", "nbr_in", "nbr_w"))
    pred = feats.skb.apply(make_model(), y=y)
    pred = attach_scoring(pred)

DESCRIPTION = "+ degree/specificity/reciprocity/same-tld weighted 1-hop profiles"
PARENT = "pipeline_03"
