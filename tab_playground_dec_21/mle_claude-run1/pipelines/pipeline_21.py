"""pipeline_21 -- FEATURE: soil ELU climatic/geologic zones (meaningful domain info).

Each soil type carries a USFS ELU code whose 1st digit is a climatic zone
(2 lower montane .. 8 alpine) and 2nd is a geologic zone. The climatic band is an
elevation/temperature ordering that should map almost directly onto cover type --
real external structure the raw soil binaries and the coarse `cryic` flag don't
fully expose. Ablate it on the best feature set (base14) at the fast config, in one
run via choose_from:

  base14        : base_geo + soil descriptors + ratios     (control = pipeline_14)
  +elu_climatic : + soil_climatic_zone                      (the strong hypothesis)
  +elu_both     : + soil_climatic_zone + soil_geologic_zone (does geology add?)
"""
import skrub

from common import load_xy
from features import (
    base_geo, add_soil_descriptors, add_ratios, add_soil_elu, fast_model,
)

X, y = load_xy(target="Cover_Type")
Xd = X.skb.drop(cols="Id")
base14 = add_ratios(add_soil_descriptors(base_geo(Xd)))

variants = {
    "base14": base14.skb.apply(fast_model(), y=y),
    "+elu_climatic": add_soil_elu(base14, geologic=False).skb.apply(fast_model(), y=y),
    "+elu_both": add_soil_elu(base14, geologic=True).skb.apply(fast_model(), y=y),
}
pred = skrub.choose_from(variants, name="featureset").as_data_op()

DESCRIPTION = "FEATURE: soil ELU climatic/geologic zones on base14 (fast config, ablation)"
PARENT = "pipeline_14"
