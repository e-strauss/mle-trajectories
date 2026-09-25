"""First LEARNED model: multi-output ridge on the same three blocks as pipeline_02.

pipeline_02 blended the blocks with hand-set weights that are shared by all 355
trackers. A multi-output Ridge instead learns a full 1065 -> 355 weight matrix,
so it can pick up *cross-tracker* structure the blend cannot express, e.g.
"a neighbour using yandex.ru also implies rambler.ru", and per-tracker
calibration (a rare tracker needs stronger evidence than google-analytics).

Same features as pipeline_02 (no meta block yet -- one change at a time), l1
squashing on the neighbour blocks because that variant won pipeline_02.
The alpha sweep is a fused-choice exploration.
"""
import skrub
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import attach_scoring, load_xy
from models import BlockTransform

X, y = load_xy(blocks=("nbr_out", "nbr_in", "tld_pop"))

feats = (X.skb.drop("domain_id")
          .skb.apply(BlockTransform(mode="l1"))
          .skb.apply(StandardScaler()))

model = Ridge(alpha=skrub.choose_from([1.0, 30.0, 300.0, 3000.0], name="alpha"),
              solver="cholesky")
pred = feats.skb.apply(model, y=y)
pred = attach_scoring(pred)

DESCRIPTION = ("Multi-output Ridge (alpha sweep) on l1-normalised neighbour "
               "blocks + TLD profile -- learns cross-tracker weights")
PARENT = "pipeline_02"
