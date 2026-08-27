"""Composable feature-engineering builders + shared model configs.

Each add_* function takes a DataOp X (with the raw columns present) and returns X
with extra recorded columns; raw columns are always kept so builders compose in
any order. Importable directly (the harness puts pipelines/ on sys.path).

The exploratory pipelines (09-13) hold the model FIXED at fast_model() and vary
only the feature set, so score deltas attribute cleanly to features. The merge
pipelines (14-15) combine the builders that helped.
"""
import numpy as np
import pandas as pd
from skrub import selectors as s
from lightgbm import LGBMClassifier

from common import SEED

# --- Shared model configs ---------------------------------------------------

def fast_model():
    """Compute-friendly config for feature exploration (~a few min/run)."""
    return LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=127,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def medium_model():
    """Mid-capacity config: the cheap-model ceiling needs lr-down + trees-up.

    pipeline_17 showed the fast budget (200 trees, lr=0.1) can't use more leaves --
    capacity per tree only pays with a lower learning rate and more rounds. This is
    that trade at ~3x fast cost (~800s), still ~3x cheaper than big_model: a
    stronger exploration base and a more faithful big_model proxy.
    """
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=255,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def big_model():
    """High-capacity config (matches pipeline_08) for the final merged run."""
    return LGBMClassifier(
        n_estimators=700, learning_rate=0.035, num_leaves=511,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


# --- Soil descriptor flags, parsed from the data dictionary ------------------
# The 40 soil-type descriptions encode stoniness and (via the "cry-" prefix) a
# cryic/cold temperature regime that tracks high elevation -> high-elevation
# cover types. Parse them once into per-soil-type weight vectors.
_SOIL_DESC = {
    1: "cathedral family - rock outcrop complex, extremely stony.",
    2: "vanet - ratake families complex, very stony.",
    3: "haploborolis - rock outcrop complex, rubbly.",
    4: "ratake family - rock outcrop complex, rubbly.",
    5: "vanet family - rock outcrop complex complex, rubbly.",
    6: "vanet - wetmore families - rock outcrop complex, stony.",
    7: "gothic family.",
    8: "supervisor - limber families complex.",
    9: "troutville family, very stony.",
    10: "bullwark - catamount families - rock outcrop complex, rubbly.",
    11: "bullwark - catamount families - rock land complex, rubbly.",
    12: "legault family - rock land complex, stony.",
    13: "catamount family - rock land - bullwark family complex, rubbly.",
    14: "pachic argiborolis - aquolis complex.",
    15: "unspecified in the usfs soil and elu survey.",
    16: "cryaquolis - cryoborolis complex.",
    17: "gateview family - cryaquolis complex.",
    18: "rogert family, very stony.",
    19: "typic cryaquolis - borohemists complex.",
    20: "typic cryaquepts - typic cryaquolls complex.",
    21: "typic cryaquolls - leighcan family, till substratum complex.",
    22: "leighcan family, till substratum, extremely bouldery.",
    23: "leighcan family, till substratum - typic cryaquolls complex.",
    24: "leighcan family, extremely stony.",
    25: "leighcan family, warm, extremely stony.",
    26: "granile - catamount families complex, very stony.",
    27: "leighcan family, warm - rock outcrop complex, extremely stony.",
    28: "leighcan family - rock outcrop complex, extremely stony.",
    29: "como - legault families complex, extremely stony.",
    30: "como family - rock land - legault family complex, extremely stony.",
    31: "leighcan - catamount families complex, extremely stony.",
    32: "catamount family - rock outcrop - leighcan family complex, extremely stony.",
    33: "leighcan - catamount families - rock outcrop complex, extremely stony.",
    34: "cryorthents - rock land complex, extremely stony.",
    35: "cryumbrepts - rock outcrop - cryaquepts complex.",
    36: "bross family - rock land - cryumbrepts complex, extremely stony.",
    37: "rock outcrop - cryumbrepts - cryorthents complex, extremely stony.",
    38: "leighcan - moran families - cryaquolls complex, extremely stony.",
    39: "moran family - cryorthents - leighcan family complex, extremely stony.",
    40: "moran family - cryorthents - rock land complex, extremely stony.",
}


def _stoniness(desc):
    if "extremely stony" in desc or "extremely bouldery" in desc:
        return 3
    if "very stony" in desc:
        return 2
    if "stony" in desc:
        return 1
    return 0


def _soil_flag_series():
    """dict[flag_name -> pd.Series indexed by 'Soil_TypeN' with the weight]."""
    idx = [f"Soil_Type{i}" for i in range(1, 41)]
    flags = {
        "stoniness": [_stoniness(_SOIL_DESC[i]) for i in range(1, 41)],
        "cryic": [int("cry" in _SOIL_DESC[i]) for i in range(1, 41)],
        "rock_outcrop": [int("rock outcrop" in _SOIL_DESC[i]) for i in range(1, 41)],
        "rock_land": [int("rock land" in _SOIL_DESC[i]) for i in range(1, 41)],
        "rubbly": [int("rubbly" in _SOIL_DESC[i]) for i in range(1, 41)],
        "leighcan": [int("leighcan" in _SOIL_DESC[i]) for i in range(1, 41)],
    }
    return {k: pd.Series(v, index=idx, dtype="float64") for k, v in flags.items()}


_SOIL_FLAGS = _soil_flag_series()


# --- Soil ELU (Ecological Land Unit) codes, from the UCI covertype docs --------
# Each soil type carries a 4-digit USFS ELU code: the 1st digit is the CLIMATIC
# zone (2 lower montane .. 8 alpine -- a temperature/elevation band that tracks
# cover type directly) and the 2nd is the GEOLOGIC zone (1 alluvium, 2 glacial,
# 5 mixed sedimentary, 7 igneous/metamorphic, 8 volcanic). This is external domain
# structure the raw binaries don't expose. Sanity-checked vs _SOIL_DESC: soil 22
# ("till substratum") -> 7201 -> geologic 2 = glacial/till; soil 38 (moran/cryaquolls,
# alpine) -> 8771 -> climatic 8 = alpine.
_SOIL_ELU = {
    1: 2702, 2: 2703, 3: 2704, 4: 2705, 5: 2706, 6: 2717, 7: 3501, 8: 3502,
    9: 4201, 10: 4703, 11: 4704, 12: 4744, 13: 4758, 14: 5101, 15: 5151, 16: 6101,
    17: 6102, 18: 6731, 19: 7101, 20: 7102, 21: 7103, 22: 7201, 23: 7202, 24: 7700,
    25: 7701, 26: 7702, 27: 7709, 28: 7710, 29: 7745, 30: 7746, 31: 7755, 32: 7756,
    33: 7757, 34: 7790, 35: 8703, 36: 8707, 37: 8708, 38: 8771, 39: 8772, 40: 8776,
}


def _elu_zone_series(kind):
    """pd.Series (indexed by Soil_TypeN) of the climatic or geologic zone digit."""
    idx = [f"Soil_Type{i}" for i in range(1, 41)]
    if kind == "climatic":
        vals = [_SOIL_ELU[i] // 1000 for i in range(1, 41)]        # thousands digit
    else:
        vals = [(_SOIL_ELU[i] // 100) % 10 for i in range(1, 41)]  # hundreds digit
    return pd.Series(vals, index=idx, dtype="float64")


_ELU_CLIMATIC = _elu_zone_series("climatic")
_ELU_GEOLOGIC = _elu_zone_series("geologic")


# --- Feature builders --------------------------------------------------------

def base_geo(X):
    """pipeline_07 feature set: geometric interactions + indicator counts."""
    hdh = X["Horizontal_Distance_To_Hydrology"]
    vdh = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)
    return X.assign(
        Euclidean_Hydro=(hdh**2 + vdh**2).skb.apply_func(np.sqrt),
        Elev_minus_VDH=elev - vdh,
        Elev_minus_HDH=elev - hdh * 0.2,
        Hydro_plus_Fire=hdh + fire,
        Hydro_minus_Fire=hdh - fire,
        Hydro_plus_Road=hdh + road,
        Hydro_minus_Road=hdh - road,
        Fire_plus_Road=fire + road,
        Fire_minus_Road=fire - road,
        Aspect_sin=aspect_rad.skb.apply_func(np.sin),
        Aspect_cos=aspect_rad.skb.apply_func(np.cos),
        Hillshade_mean=(X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]) / 3.0,
        soil_count=X.skb.select(s.glob("Soil_Type*")).sum(axis=1),
        wild_count=X.skb.select(s.glob("Wilderness_Area*")).sum(axis=1),
        Hillshade_9am_minus_3pm=X["Hillshade_9am"] - X["Hillshade_3pm"],
        abs_VDH=vdh.skb.apply_func(np.abs),
    )


def add_interactions(X):
    """Products trees can't form directly (elevation x slope/aspect, etc.)."""
    elev = X["Elevation"]
    slope = X["Slope"]
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)
    return X.assign(
        Elev_x_Slope=elev * slope,
        Elev_x_AspectSin=elev * aspect_rad.skb.apply_func(np.sin),
        Elev_x_AspectCos=elev * aspect_rad.skb.apply_func(np.cos),
        Slope_x_HillNoon=slope * X["Hillshade_Noon"],
        Road_x_Fire=X["Horizontal_Distance_To_Roadways"] * X["Horizontal_Distance_To_Fire_Points"],
        Elev_x_HydroH=elev * X["Horizontal_Distance_To_Hydrology"],
    )


def add_ratios(X):
    """Scale-relative distance features: aggregates + pairwise ratios."""
    hdh = X["Horizontal_Distance_To_Hydrology"]
    vdh = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]
    euclid = (hdh**2 + vdh**2).skb.apply_func(np.sqrt)
    return X.assign(
        Dist_total=hdh + road + fire,
        Dist_mean=(hdh + road + fire) / 3.0,
        Hydro_over_Road=hdh / (road.skb.apply_func(np.abs) + 1.0),
        Fire_over_Road=fire / (road.skb.apply_func(np.abs) + 1.0),
        Hydro_over_Fire=hdh / (fire.skb.apply_func(np.abs) + 1.0),
        VDH_over_HDH=vdh / (hdh + 1.0),
        Elev_over_Hydro=elev / (euclid + 1.0),
    )


def add_soil_descriptors(X):
    """Row aggregates of soil-type descriptor flags (dot with the multi-hot block)."""
    soil = X.skb.select(s.glob("Soil_Type*"))
    new = {}
    for name, wser in _SOIL_FLAGS.items():
        new[f"soil_{name}"] = (soil * wser).sum(axis=1)
    return X.assign(**new)


def add_soil_elu(X, geologic=True):
    """Soil ELU climatic (+ optional geologic) zone per row.

    Dots the multi-hot soil block with the per-soil-type ELU zone digit, so each
    row gets its soil's climatic band (2..8) and geologic zone. For the rare
    multi-soil rows the zones sum (soil_count is already a feature, so the tree can
    normalise). The climatic zone is the strong hypothesis -- an elevation/temperature
    ordering that maps almost directly onto cover type.
    """
    soil = X.skb.select(s.glob("Soil_Type*"))
    new = {"soil_climatic_zone": (soil * _ELU_CLIMATIC).sum(axis=1)}
    if geologic:
        new["soil_geologic_zone"] = (soil * _ELU_GEOLOGIC).sum(axis=1)
    return X.assign(**new)


def add_soil_extra(X):
    """More domain flags parsed from the soil dictionary (extends add_soil_descriptors).

    add_soil_descriptors captured stoniness/cryic/rock/leighcan; this adds the
    remaining regime signals the descriptions encode, each a genuinely new piece of
    external knowledge (not a product of existing columns, which have consistently
    not helped this GBDT):
      - warm  : the "warm" leighcan variants (temperature regime distinct from cryic)
      - aquic : cryaqu*/aquolis/borohemists -> wet/poorly-drained soils (moisture regime)
      - till  : glacial "till substratum" parent material
      - bouldery : "extremely bouldery" coarse fragments (distinct from stoniness)
    Aggregated over the multi-hot soil block, same as add_soil_descriptors.
    """
    soil = X.skb.select(s.glob("Soil_Type*"))
    extra_flags = {
        "warm": pd.Series([int("warm" in _SOIL_DESC[i]) for i in range(1, 41)],
                          index=[f"Soil_Type{i}" for i in range(1, 41)], dtype="float64"),
        "aquic": pd.Series([int("aqu" in _SOIL_DESC[i]) for i in range(1, 41)],
                           index=[f"Soil_Type{i}" for i in range(1, 41)], dtype="float64"),
        "till": pd.Series([int("till" in _SOIL_DESC[i]) for i in range(1, 41)],
                          index=[f"Soil_Type{i}" for i in range(1, 41)], dtype="float64"),
        "bouldery": pd.Series([int("bouldery" in _SOIL_DESC[i]) for i in range(1, 41)],
                              index=[f"Soil_Type{i}" for i in range(1, 41)], dtype="float64"),
    }
    new = {f"soil_{name}": (soil * wser).sum(axis=1) for name, wser in extra_flags.items()}
    return X.assign(**new)


def add_target_encoding(X, y):
    """Fold-safe class-conditional target stats for dominant soil & wilderness.

    Derives the dominant soil type and wilderness area (argmax of each multi-hot
    block) and target-encodes them against the 7-class target with sklearn's
    multiclass TargetEncoder -> a P(cover_type | category) column per class. Applied
    with y= INSIDE the plan, so skrub refits the encoder on each training fold only
    (no leakage). Gives the tree class-probability structure that no geometric or
    count feature can express.
    """
    from sklearn.preprocessing import TargetEncoder
    cats = X.skb.apply_func(lambda d: pd.DataFrame({
        "soil_dom": d.filter(like="Soil_Type").idxmax(axis=1),
        "wild_dom": d.filter(like="Wilderness_Area").idxmax(axis=1),
    }))
    te = cats.skb.apply(TargetEncoder(target_type="multiclass"), y=y)
    return X.skb.concat([te], axis=1)


def best_features(X):
    """Current champion feature set (pipeline_21): base14 + BOTH ELU zones.

    = base_geo + soil descriptors + distance ratios + soil ELU climatic & geologic
    zones. Beat the base14 merge by +0.00091 at the fast config (0.95283). NB the
    ELU zones must go in together -- climatic alone regressed. X must already have
    Id dropped (same contract as base_geo).
    """
    return add_soil_elu(add_ratios(add_soil_descriptors(base_geo(X))), geologic=True)


def add_soil_geo_cross(X):
    """Cross the winning soil descriptors with the strongest geo signals.

    Cryic soils encode a cold/high-elevation regime, and stoniness / rock outcrop
    covary with steep, high terrain -- multiplying the soil aggregates by
    Elevation / Slope hands the tree ready-made splits that separate the
    high-elevation cover types. REQUIRES add_soil_descriptors to have run first
    (reads the soil_* aggregate columns it creates).
    """
    elev = X["Elevation"]
    slope = X["Slope"]
    return X.assign(
        Cryic_x_Elev=X["soil_cryic"] * elev,
        Stoniness_x_Elev=X["soil_stoniness"] * elev,
        Stoniness_x_Slope=X["soil_stoniness"] * slope,
        RockOutcrop_x_Elev=X["soil_rock_outcrop"] * elev,
    )


def add_hillshade_physics(X):
    """Shade dynamics + slope-adjusted illumination."""
    shade = X.skb.select(s.cols("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"))
    slope_rad = X["Slope"].skb.apply_func(np.deg2rad)
    slope_cos = slope_rad.skb.apply_func(np.cos)
    return X.assign(
        Hillshade_min=shade.min(axis=1),
        Hillshade_max=shade.max(axis=1),
        Hillshade_range=shade.max(axis=1) - shade.min(axis=1),
        Hillshade_std=shade.std(axis=1),
        Slope_cos=slope_cos,
        Illum_noon=X["Hillshade_Noon"] * slope_cos,
    )
