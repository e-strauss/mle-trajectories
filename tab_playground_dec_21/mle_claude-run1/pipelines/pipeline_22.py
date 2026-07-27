"""pipeline_22 -- FEATURE: fold-safe target encoding of dominant soil & wilderness.

Class-conditional target statistics are the textbook strong lever for covertype and
give the tree information no geometric/count feature can: P(cover_type | dominant
soil) and P(cover_type | wilderness area). Derived argmax categoricals + sklearn
multiclass TargetEncoder, applied with y= inside the plan so skrub refits per fold
(no leakage; smoke-tested). Ablated on the current champion (best_features =
base14 + ELU zones, 0.95283) at the fast config, one run via choose_from:

  champ    : best_features                          (control = pipeline_21 winner)
  champ+TE : best_features + target encoding         (do target stats add?)
"""
import skrub

from common import load_xy
from features import best_features, add_target_encoding, fast_model

X, y = load_xy(target="Cover_Type")
Xd = X.skb.drop(cols="Id")
champ = best_features(Xd)

variants = {
    "champ": champ.skb.apply(fast_model(), y=y),
    "champ+TE": add_target_encoding(champ, y).skb.apply(fast_model(), y=y),
}
pred = skrub.choose_from(variants, name="featureset").as_data_op()

DESCRIPTION = "FEATURE: target encoding (dominant soil/wilderness) on champion (fast, ablation)"
PARENT = "pipeline_21"
