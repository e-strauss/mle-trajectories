"""pipeline_17 -- TUNE the fast model: num_leaves x min_child_samples grid.

On the best feature set (pipeline_14: base_geo + soil + ratios), sweep the two
cheapest high-leverage LightGBM knobs at the fast budget (200 trees, lr=0.1):
  - num_leaves        127 -> 255  (per-tree capacity; capacity has been the top lever)
  - min_child_samples 20  -> 100  (leaf regularization; default 20 is tiny at 4M rows)

skrub collects both choices; the ml-score harness runs make_grid_search over all
4 combinations, records the BEST to the leaderboard, and saves the full grid in
results.json under extra.grid. Goal: a stronger-but-still-cheap fast config, which
also transfers to the big model more faithfully than the current weak fast config.
"""
import skrub
from lightgbm import LGBMClassifier

from common import load_xy, SEED
from features import base_geo, add_soil_descriptors, add_ratios

model = LGBMClassifier(
    n_estimators=200, learning_rate=0.1,
    num_leaves=skrub.choose_from([127, 255], name="num_leaves"),
    min_child_samples=skrub.choose_from([20, 100], name="min_child_samples"),
    random_state=SEED, n_jobs=-1, verbose=-1,
)

X, y = load_xy(target="Cover_Type")
X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
pred = X.skb.apply(model, y=y)

DESCRIPTION = "TUNE fast: num_leaves{127,255} x min_child_samples{20,100} on pipeline_14 features"
PARENT = "pipeline_14"
