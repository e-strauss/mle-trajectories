"""Trivial baseline: predict the 10 globally most popular trackers for everyone.

The floor every later pipeline must beat. Uses no features at all -- only the
training fold's label frequencies -- so it measures how much of Recall@10 comes
for free from the extreme popularity skew (google-analytics.com alone is on
61% of all tracked domains).
"""
from skrub import selectors as s

from common import attach_scoring, load_xy
from models import PopularityRanker

X, y = load_xy(blocks=("meta",))

# the ranker ignores its input; keep one cheap column so the node is well-formed
pred = X.skb.select(s.cols("domain_id")).skb.apply(PopularityRanker(), y=y)
pred = attach_scoring(pred)

DESCRIPTION = "Baseline: constant global tracker-popularity ranking (no features)"
PARENT = None
