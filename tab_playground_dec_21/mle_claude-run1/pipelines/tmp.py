
"""All pipelines (except pipeline_04, which is buggy) as pipeNN(X, y) functions.

Each function mirrors the corresponding pipeline_NN.py plan body, taking the
marked (X, y) and returning `pred`. The loop at the bottom builds every plan and
opens its graph.
"""
import numpy as np
import skrub
from skrub import selectors as s
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

from common import load_xy, SEED
from features import (
    base_geo,
    add_interactions,
    add_ratios,
    add_soil_descriptors,
    add_hillshade_physics,
    add_soil_geo_cross,
    add_soil_extra,
    add_soil_elu,
    add_target_encoding,
    best_features,
    fast_model,
    medium_model,
    big_model,
)
from nn import TorchMLP


def pipe01(X, y):
    # pipeline_01 -- majority-class DummyClassifier (accuracy floor)
    X = X.skb.drop(cols="Id")
    pred = X.skb.apply(DummyClassifier(strategy="most_frequent"), y=y)
    return pred


def pipe02(X, y):
    # pipeline_02 -- HistGradientBoostingClassifier, default params, raw numeric
    X = X.skb.drop(cols="Id")
    pred = X.skb.apply(
        HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
        y=y,
    )
    return pred


def pipe03(X, y):
    # pipeline_03 -- LightGBM classifier, default params, raw numeric
    X = X.skb.drop(cols="Id")
    pred = X.skb.apply(
        LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        y=y,
    )
    return pred


def pipe05(X, y):
    # pipeline_05 -- LightGBM + classic Forest-Cover engineered features
    X = X.skb.drop(cols="Id")

    hdh = X["Horizontal_Distance_To_Hydrology"]
    vdh = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)

    X = X.assign(
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
    )

    pred = X.skb.apply(
        LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        y=y,
    )
    return pred


def pipe06(X, y):
    # pipeline_06 -- higher-capacity LightGBM on the engineered features
    X = X.skb.drop(cols="Id")

    hdh = X["Horizontal_Distance_To_Hydrology"]
    vdh = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)

    X = X.assign(
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
    )

    pred = X.skb.apply(
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=255,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        y=y,
    )
    return pred


def pipe07(X, y):
    # pipeline_07 -- richer feature set, same high-capacity LightGBM as pipeline_06
    X = X.skb.drop(cols="Id")

    hdh = X["Horizontal_Distance_To_Hydrology"]
    vdh = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)

    X = X.assign(
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
        # --- new in pipeline_07 ---
        soil_count=X.skb.select(s.glob("Soil_Type*")).sum(axis=1),
        wild_count=X.skb.select(s.glob("Wilderness_Area*")).sum(axis=1),
        Hillshade_9am_minus_3pm=X["Hillshade_9am"] - X["Hillshade_3pm"],
        abs_VDH=vdh.skb.apply_func(np.abs),
    )

    pred = X.skb.apply(
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=255,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        y=y,
    )
    return pred


def pipe08(X, y):
    # pipeline_08 -- final capacity push: more trees, deeper, slower learning
    X = X.skb.drop(cols="Id")

    hdh = X["Horizontal_Distance_To_Hydrology"]
    vdh = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]
    aspect_rad = X["Aspect"].skb.apply_func(np.deg2rad)

    X = X.assign(
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

    pred = X.skb.apply(
        LGBMClassifier(
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=511,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        ),
        y=y,
    )
    return pred


def pipe09(X, y):
    # pipeline_09 -- ANCHOR: base_geo features, fast model
    X = base_geo(X.skb.drop(cols="Id"))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe10(X, y):
    # pipeline_10 -- direction A: multiplicative interaction features
    X = add_interactions(base_geo(X.skb.drop(cols="Id")))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe11(X, y):
    # pipeline_11 -- direction B: distance ratios & aggregates
    X = add_ratios(base_geo(X.skb.drop(cols="Id")))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe12(X, y):
    # pipeline_12 -- direction C: soil-descriptor domain features
    X = add_soil_descriptors(base_geo(X.skb.drop(cols="Id")))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe13(X, y):
    # pipeline_13 -- direction D: hillshade dynamics & slope-adjusted illumination
    X = add_hillshade_physics(base_geo(X.skb.drop(cols="Id")))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe14(X, y):
    # pipeline_14 -- MERGE: base_geo + soil descriptors + distance ratios
    X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe15(X, y):
    # pipeline_15 -- IMPROVE: pipeline_14 merge + soil x geo crosses
    X = add_soil_geo_cross(add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id")))))
    pred = X.skb.apply(fast_model(), y=y)
    return pred


def pipe16(X, y):
    # pipeline_16 -- FINAL: pipeline_14 feature set at big_model capacity
    X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
    pred = X.skb.apply(big_model(), y=y)
    return pred


def pipe17(X, y):
    # pipeline_17 -- TUNE fast: num_leaves x min_child_samples grid on base14
    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.1,
        num_leaves=skrub.choose_from([127, 255], name="num_leaves"),
        min_child_samples=skrub.choose_from([20, 100], name="min_child_samples"),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
    pred = X.skb.apply(model, y=y)
    return pred


def pipe18(X, y):
    # pipeline_18 -- ABLATION: soil_extra on/off x ratios on/off (fast config)
    Xd = X.skb.drop(cols="Id")
    base14 = add_ratios(add_soil_descriptors(base_geo(Xd)))
    variants = {
        "base14": base14.skb.apply(fast_model(), y=y),
        "base14+soilx": add_soil_extra(base14).skb.apply(fast_model(), y=y),
        "soilx_no_ratios": add_soil_extra(add_soil_descriptors(base_geo(Xd))).skb.apply(fast_model(), y=y),
    }
    pred = skrub.choose_from(variants, name="featureset").as_data_op()
    return pred


def pipe19(X, y):
    # pipeline_19 -- MEDIUM config on base14 features
    X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
    pred = X.skb.apply(medium_model(), y=y)
    return pred


def pipe20(X, y):
    # pipeline_20 -- CAPACITY: 400 trees/127 leaves, lr grid on base14
    model = LGBMClassifier(
        n_estimators=400, num_leaves=127,
        learning_rate=skrub.choose_from([0.05, 0.075], name="lr"),
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    X = add_ratios(add_soil_descriptors(base_geo(X.skb.drop(cols="Id"))))
    pred = X.skb.apply(model, y=y)
    return pred


def pipe21(X, y):
    # pipeline_21 -- FEATURE: soil ELU climatic/geologic zones on base14 (ablation)
    Xd = X.skb.drop(cols="Id")
    base14 = add_ratios(add_soil_descriptors(base_geo(Xd)))
    variants = {
        "base14": base14.skb.apply(fast_model(), y=y),
        "+elu_climatic": add_soil_elu(base14, geologic=False).skb.apply(fast_model(), y=y),
        "+elu_both": add_soil_elu(base14, geologic=True).skb.apply(fast_model(), y=y),
    }
    pred = skrub.choose_from(variants, name="featureset").as_data_op()
    return pred


def pipe22(X, y):
    # pipeline_22 -- FEATURE: fold-safe target encoding on champion (ablation)
    Xd = X.skb.drop(cols="Id")
    champ = best_features(Xd)
    variants = {
        "champ": champ.skb.apply(fast_model(), y=y),
        "champ+TE": add_target_encoding(champ, y).skb.apply(fast_model(), y=y),
    }
    pred = skrub.choose_from(variants, name="featureset").as_data_op()
    return pred


def pipe23(X, y):
    # pipeline_23 -- MODEL: torch MLP (skorch) on champion features
    X = best_features(X.skb.drop(cols="Id"))
    pred = X.skb.apply(
        TorchMLP(hidden=256, n_layers=2, dropout=0.2, lr=1e-3,
                 max_epochs=15, batch_size=8192, device="mps"),
        y=y,
    )
    return pred


PIPES = [
    pipe01, pipe02, pipe03,
    pipe05, pipe06, pipe07, pipe08, pipe09, pipe10,
    pipe11, pipe12, pipe13, pipe14, pipe15, pipe16,
    pipe17, pipe18, pipe19, pipe20, pipe21, pipe22, pipe23,
]

with skrub.config_context(eager_data_ops=False):

    X, y = load_xy(target="Cover_Type")

    for func in PIPES:
        pred = func(X, y)
        pred.skb.draw_graph().open()
