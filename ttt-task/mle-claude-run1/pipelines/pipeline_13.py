"""+ `trk_cooc`: neighbour evidence propagated through tracker co-occurrence.

Aimed squarely at the hardest ceiling in this task: only **79.7%** of a domain's
true trackers appear anywhere in its neighbourhood histogram
(data_exploration_5), so a fifth of the answer is invisible to neighbourhood
evidence no matter how it is weighted. Co-occurrence can surface those: if the
neighbours run googlesyndication, doubleclick is likely even if no neighbour
shows it.

`tc = nbr_hub_l1 @ P(b|a)`. Unlike the other three blocks this IS a linear map
of features the model already has, so the MLP could in principle learn it -- the
bet is purely about estimation quality. P(b|a) is measured on **18.18M
out-of-sample domains**, roughly 54x the 337k rows an MLP fold trains on, which
is the same reason target encoding beats making a model learn a high-cardinality
category unaided. If it fails, that is a clean result too: it says the MLP
already has enough data to learn tracker co-occurrence by itself.

Leakage: P(b|a) uses ONLY tracked domains outside the frozen sample, so it is
independent of every scored row's label by construction -- stronger than
leave-one-out, and identical in kind for target rows.

Scored as a SIBLING of pipeline_10.
"""
from common import attach_scoring, load_xy
from featureset import build_pred, files

X, y = load_xy(blocks=files("trk_cooc"))
pred = attach_scoring(build_pred(X, y, "trk_cooc"))

DESCRIPTION = ("pipeline_10 + neighbourhood propagated through the tracker "
               "co-occurrence matrix P(b|a) fitted out-of-sample")
PARENT = "pipeline_10"
