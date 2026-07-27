"""pipeline_20 -- corrected capacity test: more trees at the PROVEN leaf count.

pipeline_19 showed 255 leaves is a bad middle ground: at the same learning budget
(n_estimators x lr ~= 20) the fast model's 127 leaves (0.95192) beat 255 leaves
(0.94909). num_leaves must scale WITH the tree budget, not independently. So hold
leaves at the proven 127 and add trees (200 -> 400), bracketing the learning
budget with an lr grid:
  lr=0.05  -> budget 20 (same as fast, but lower lr / more rounds)
  lr=0.075 -> budget 30 (more total learning at proven leaves)
One run via choose_from; harness records the best + full grid in extra.grid.
Question: can a cheap-ish config beat the fast model's 0.95192 without going big?
"""
import skrub
from lightgbm import LGBMClassifier

from common import load_xy, SEED
from features import base_geo, add_soil_descriptors, add_ratios

model = LGBMClassifier(
    n_estimators=400, num_leaves=127,
    learning_rate=skrub.choose_from([0.05, 0.075], name="lr"),
    random_state=SEED, n_jobs=-1, verbose=-1,
)

X, y = load_xy(target="Cover_Type")
X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
pred = X.skb.apply(model, y=y)

DESCRIPTION = "CAPACITY: 400 trees/127 leaves, lr{0.05,0.075} on base14 features"
PARENT = "pipeline_14"
