import os

# Prevent BLAS/OpenMP thread contention inside multiprocessing workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import gc
import importlib.util
import inspect
import math
import pickle
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Sampler
from xgboost import XGBClassifier

cv2.setNumThreads(0)

# ============================================================================
# Global Constants, Paths & Pretrained Backbone Provenance
# ============================================================================
SEED = 42
INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
SUBMISSION_DIR = Path("./submission")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

PATCH_SIZE = 256
FALLBACK_H_VEC = np.array([0.650, 0.704, 0.286], dtype=np.float32)
FALLBACK_E_VEC = np.array([0.072, 0.990, 0.105], dtype=np.float32)
FALLBACK_WHITE = np.array([248.0, 248.0, 248.0], dtype=np.float32)

# Precomputed 16x16 GLCM index matrices
GLCM_LEVELS = 16
_II, _JJ = np.meshgrid(
    np.arange(GLCM_LEVELS, dtype=np.float32),
    np.arange(GLCM_LEVELS, dtype=np.float32),
    indexing="ij",
)
_GLCM_DIFF_SQ = (_II - _JJ) ** 2
_GLCM_IDM_W = 1.0 / (1.0 + _GLCM_DIFF_SQ)

# Pretrained Vision Backbone Identity & Provenance (per competition rules):
# - Primary ID: "vit_small_patch14_reg4_dinov2.lvd142m" (timm library)
# - Fallback ID: "vit_small_patch14_dinov2.lvd142m"
# - Architecture: Meta AI DINOv2 ViT-S/14 with 4 learnable register tokens
#   (Oquab et al., TMLR 2024; Darcet et al., ICLR 2024).
# - Pretraining Corpus: LVD-142M curated natural web images via self-supervised
#   discriminative distillation + iBOT masked image modeling (zero supervision,
#   zero Camelyon/histopathology labeled dataset overlap).
DEFAULT_DINOV2_BACKBONE = "vit_small_patch14_reg4_dinov2.lvd142m"
FALLBACK_DINOV2_BACKBONE = "vit_small_patch14_dinov2.lvd142m"
DINO_INPUT_SIZE = 252  # Exact 18 x 18 grid of 14x14 patches (98.4% of 256x256)
DINO_GRID_SIZE = 18

# Default 7-family index partition matching Step 1's 148-feature schema
DEFAULT_FAMILY_INDICES: Dict[str, List[int]] = {
    "topology": list(range(0, 33)),
    "structure_tensor": list(range(33, 51)),
    "multizone": list(range(51, 75)),
    "chromatin_glcm": list(range(75, 91)),
    "cytomorphometry": list(range(91, 111)),
    "color_physics": list(range(111, 126)),
    "interactions": list(range(126, 148)),
}

# Training & Evaluation Configuration
NUM_NEURAL_FOLDS = 3
NEURAL_EPOCHS = 12
EARLY_STOP_MIN_EPOCHS = 6
EARLY_STOP_PATIENCE = 4
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 128
TABULAR_TRAIN_NOISE_STD = 0.025
TTA_VIEWS = (0, 1, 2, 3, 4, 5, 6, 7)
MAX_PIPELINE_SECONDS = 28800.0  # Safety guard within 9-hour limit


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unit_vec(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return v.astype(np.float32)
    return (v / norm).astype(np.float32)


FALLBACK_H_VEC = _unit_vec(FALLBACK_H_VEC)
FALLBACK_E_VEC = _unit_vec(FALLBACK_E_VEC)


# ============================================================================
# Step 1.1: Data Loading, Schema Audit & Multi-Stage Split Construction
# ============================================================================
def load_and_prepare_splits(input_dir: Path = INPUT_DIR):
    train_df = pd.read_csv(input_dir / "train.csv")
    val_df = pd.read_csv(input_dir / "validation.csv")
    test_df = pd.read_csv(input_dir / "test.csv")
    sub_df = pd.read_csv(input_dir / "sample_submission.csv")

    required_dev_cols = {
        "sample_id",
        "image_path",
        "label",
        "slide_id",
        "group_id",
        "center_id",
    }
    assert required_dev_cols.issubset(train_df.columns), "Missing columns in train.csv"
    assert required_dev_cols.issubset(
        val_df.columns
    ), "Missing columns in validation.csv"
    assert list(sub_df.columns) == [
        "sample_id",
        "tumor_probability",
    ], "Unexpected submission format"
    assert (
        test_df["sample_id"].values == sub_df["sample_id"].values
    ).all(), "test.csv and sample_submission.csv sample_id order mismatch"

    # Strict leak check between fixed train and fixed validation sets
    assert set(train_df["sample_id"]).isdisjoint(
        set(val_df["sample_id"])
    ), "sample_id overlap!"
    assert set(train_df["slide_id"]).isdisjoint(
        set(val_df["slide_id"])
    ), "slide_id overlap!"
    assert set(train_df["group_id"]).isdisjoint(
        set(val_df["group_id"])
    ), "group_id overlap!"
    assert set(train_df["sample_id"]).isdisjoint(
        set(test_df["sample_id"])
    ), "train/test overlap!"
    assert set(val_df["sample_id"]).isdisjoint(
        set(test_df["sample_id"])
    ), "val/test overlap!"

    # Verify labels in train and validation
    assert set(train_df["label"].unique()) == {
        0,
        1,
    }, "train labels must be binary {0, 1}"
    assert set(val_df["label"].unique()) == {
        0,
        1,
    }, "validation labels must be binary {0, 1}"

    # Verify image files exist on disk
    for part_name, df_part in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        missing = [
            p for p in df_part["image_path"].tolist() if not (input_dir / p).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} images in {part_name}, e.g. {missing[0]}"
            )

    # Map center_id deterministically based ONLY on train_df centers for internal splits
    unique_centers = sorted(train_df["center_id"].unique().tolist())
    center_to_idx = {c: idx for idx, c in enumerate(unique_centers)}

    train_s1 = train_df.copy().reset_index(drop=True)
    val_s1 = val_df.copy().reset_index(drop=True)
    test_meta = test_df.copy().reset_index(drop=True)

    # 1) Leave-One-Center-Out (LOCO) folds strictly inside train_df
    train_s1["center_idx"] = train_s1["center_id"].map(center_to_idx).astype(int)
    val_s1["center_idx"] = (
        val_s1["center_id"].map(lambda c: center_to_idx.get(c, -1)).astype(int)
    )
    train_s1["loco_fold"] = train_s1["center_idx"].astype(int)

    # 2) 5-Fold StratifiedGroupKFold strictly inside train_df (grouped by group_id)
    sgkf_s1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    strat_target_s1 = (
        train_s1["center_id"].astype(str) + "_" + train_s1["label"].astype(str)
    )
    train_s1["sgkf_fold"] = -1
    for fold_idx, (_, va_idx) in enumerate(
        sgkf_s1.split(train_s1, strat_target_s1, groups=train_s1["group_id"])
    ):
        train_s1.loc[va_idx, "sgkf_fold"] = fold_idx
    assert (train_s1["sgkf_fold"] >= 0).all()

    # Verify group_id and slide_id disjointness across sgkf_fold
    for f in range(5):
        tr_mask = train_s1["sgkf_fold"] != f
        va_mask = train_s1["sgkf_fold"] == f
        assert set(train_s1.loc[tr_mask, "group_id"]).isdisjoint(
            set(train_s1.loc[va_mask, "group_id"])
        )
        assert set(train_s1.loc[tr_mask, "slide_id"]).isdisjoint(
            set(train_s1.loc[va_mask, "slide_id"])
        )

    # 3) Post-Validation Stage 2 Full-Development Merged Dataset (train.csv + validation.csv)
    # Confirmed: validation.csv contains full ground-truth labels and is merged here ONLY for
    # Stage 2 post-validation refit splits, keeping Stage 1 official validation strictly isolated.
    t_part = train_df.copy()
    t_part["source_partition"] = "train"
    v_part = val_df.copy()
    v_part["source_partition"] = "validation"
    full_dev_s2 = pd.concat([t_part, v_part], axis=0, ignore_index=True)
    full_dev_s2["center_idx"] = (
        full_dev_s2["center_id"]
        .map(lambda c: center_to_idx.get(c, len(center_to_idx)))
        .astype(int)
    )

    sgkf_s2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 101)
    strat_target_s2 = (
        full_dev_s2["center_id"].astype(str) + "_" + full_dev_s2["label"].astype(str)
    )
    full_dev_s2["refit_fold"] = -1
    for fold_idx, (_, va_idx) in enumerate(
        sgkf_s2.split(full_dev_s2, strat_target_s2, groups=full_dev_s2["group_id"])
    ):
        full_dev_s2.loc[va_idx, "refit_fold"] = fold_idx
    assert (full_dev_s2["refit_fold"] >= 0).all()

    for f in range(5):
        tr_mask = full_dev_s2["refit_fold"] != f
        va_mask = full_dev_s2["refit_fold"] == f
        assert set(full_dev_s2.loc[tr_mask, "group_id"]).isdisjoint(
            set(full_dev_s2.loc[va_mask, "group_id"])
        )
        assert set(full_dev_s2.loc[tr_mask, "slide_id"]).isdisjoint(
            set(full_dev_s2.loc[va_mask, "slide_id"])
        )

    return train_s1, val_s1, full_dev_s2, test_meta


# ============================================================================
# Step 1.2: Shrinkage-Regularized Adaptive Macenko (SRAM) Physics Engine
# ============================================================================
def read_rgb_patch(image_full_path: str) -> np.ndarray:
    bgr = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to decode image: {image_full_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
        rgb = cv2.resize(rgb, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
    return rgb


def estimate_patch_whitepoint(rgb_f32: np.ndarray, ref_white: np.ndarray) -> tuple:
    min_c = np.min(rgb_f32, axis=-1)
    max_c = np.max(rgb_f32, axis=-1)
    glass_mask = (min_c > 210.0) & ((max_c - min_c) < 35.0)
    n_glass = int(np.count_nonzero(glass_mask))
    if n_glass >= 64:
        wp_emp = np.percentile(rgb_f32[glass_mask], 90.0, axis=0).astype(np.float32)
        w = float(n_glass) / float(n_glass + 256.0)
        wp = w * wp_emp + (1.0 - w) * ref_white
        wp = np.clip(wp, 215.0, 255.0).astype(np.float32)
    else:
        wp = ref_white.copy()
    return wp, n_glass


def svd_macenko_vectors_from_od(
    od_pixels: np.ndarray, h_ref: np.ndarray, e_ref: np.ndarray
) -> tuple:
    """Estimate (h_cand, e_cand, angular_spread, valid_flag) from foreground OD pixels."""
    if od_pixels.shape[0] < 64:
        return h_ref.copy(), e_ref.copy(), 0.0, False

    cov = (od_pixels.T @ od_pixels) / float(od_pixels.shape[0])
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    if evals[2] < 1e-5 or evals[1] < 1e-6:
        return h_ref.copy(), e_ref.copy(), 0.0, False

    v1 = evecs[:, 2].astype(np.float32)
    v2 = evecs[:, 1].astype(np.float32)
    if float(v1.sum()) < 0.0:
        v1 = -v1

    proj1 = od_pixels @ v1
    proj2 = od_pixels @ v2
    phi = np.arctan2(proj2, proj1)
    phi_lo, phi_hi = np.percentile(phi, [1.0, 99.0])
    spread = float(phi_hi - phi_lo)

    u_a = v1 * np.cos(phi_lo) + v2 * np.sin(phi_lo)
    u_b = v1 * np.cos(phi_hi) + v2 * np.sin(phi_hi)
    if float(u_a.sum()) < 0.0:
        u_a = -u_a
    if float(u_b.sum()) < 0.0:
        u_b = -u_b
    u_a = _unit_vec(np.maximum(u_a, 0.0))
    u_b = _unit_vec(np.maximum(u_b, 0.0))

    # Align candidate vectors with reference Hematoxylin and Eosin
    score_ab = float(np.dot(u_a, h_ref) + np.dot(u_b, e_ref))
    score_ba = float(np.dot(u_b, h_ref) + np.dot(u_a, e_ref))
    if score_ab >= score_ba:
        h_cand, e_cand = u_a, u_b
    else:
        h_cand, e_cand = u_b, u_a

    return h_cand, e_cand, spread, True


def fit_stain_prior_from_df(
    df: pd.DataFrame,
    input_dir: Path = INPUT_DIR,
    patches_per_slide: int = 24,
    seed: int = SEED,
) -> dict:
    """
    Fit center-balanced reference Macenko stain prior and slide-level stain basis bank
    STRICTLY on the supplied DataFrame (e.g., train_df ONLY for Stage 1).
    """
    rng = np.random.default_rng(seed)
    slide_h_list, slide_e_list, slide_center_list = [], [], []
    slide_matrices, slide_whites, slide_c99 = [], [], []

    grouped = df.groupby("slide_id", sort=True)
    for _, s_df in grouped:
        s_df_sorted = s_df.sort_values("sample_id")
        if len(s_df_sorted) > patches_per_slide:
            chosen_idx = rng.choice(
                len(s_df_sorted), size=patches_per_slide, replace=False
            )
            sampled_paths = s_df_sorted["image_path"].iloc[chosen_idx].tolist()
        else:
            sampled_paths = s_df_sorted["image_path"].tolist()

        od_chunks = []
        wp_chunks = []
        for rel_p in sampled_paths:
            rgb = read_rgb_patch(str(input_dir / rel_p)).astype(np.float32)
            wp, n_glass = estimate_patch_whitepoint(rgb, FALLBACK_WHITE)
            if n_glass >= 64:
                wp_chunks.append(wp)
            od = -0.4342944819 * np.log(np.clip((rgb + 1.0) / (wp + 1.0), 1e-4, 1.0))
            od_sub = od[::2, ::2].reshape(-1, 3)
            od_norm_sq = np.sum(od_sub * od_sub, axis=1)
            valid = (od_norm_sq > 0.0225) & (np.max(od_sub, axis=1) < 2.6)
            if np.count_nonzero(valid) > 16:
                od_chunks.append(od_sub[valid])

        slide_wp = (
            np.median(np.stack(wp_chunks, axis=0), axis=0).astype(np.float32)
            if wp_chunks
            else FALLBACK_WHITE.copy()
        )
        if not od_chunks:
            continue

        od_all = np.concatenate(od_chunks, axis=0)
        if od_all.shape[0] > 24000:
            keep = rng.choice(od_all.shape[0], size=24000, replace=False)
            od_all = od_all[keep]

        h_s, e_s, spread, ok = svd_macenko_vectors_from_od(
            od_all, FALLBACK_H_VEC, FALLBACK_E_VEC
        )
        cos_he = float(np.dot(h_s, e_s))
        if (not ok) or spread < 0.08 or cos_he > 0.985:
            continue

        r_s = _unit_vec(np.cross(h_s, e_s))
        M_s = np.stack([h_s, e_s, r_s], axis=1).astype(np.float32)
        inv_M_s = np.linalg.inv(M_s.astype(np.float64)).astype(np.float32)
        conc = od_all @ inv_M_s.T
        ch99 = float(np.percentile(np.maximum(conc[:, 0], 0.0), 99.0))
        ce99 = float(np.percentile(np.maximum(conc[:, 1], 0.0), 99.0))

        slide_h_list.append(h_s)
        slide_e_list.append(e_s)
        slide_center_list.append(s_df_sorted["center_id"].iloc[0])
        slide_matrices.append(M_s)
        slide_whites.append(slide_wp)
        slide_c99.append(np.array([max(ch99, 0.25), max(ce99, 0.20)], dtype=np.float32))

    # Center-balanced aggregation so no single training center dominates canonical prior
    slide_centers = np.array(slide_center_list)
    unique_c = np.unique(slide_centers)
    center_h_means, center_e_means, center_c99_means, center_wp_means = [], [], [], []
    slide_h_arr = np.stack(slide_h_list, axis=0)
    slide_e_arr = np.stack(slide_e_list, axis=0)
    slide_c99_arr = np.stack(slide_c99, axis=0)
    slide_wp_arr = np.stack(slide_whites, axis=0)

    for c in unique_c:
        m = slide_centers == c
        center_h_means.append(_unit_vec(np.mean(slide_h_arr[m], axis=0)))
        center_e_means.append(_unit_vec(np.mean(slide_e_arr[m], axis=0)))
        center_c99_means.append(np.median(slide_c99_arr[m], axis=0))
        center_wp_means.append(np.median(slide_wp_arr[m], axis=0))

    h_ref = _unit_vec(np.mean(np.stack(center_h_means, axis=0), axis=0))
    e_ref = _unit_vec(np.mean(np.stack(center_e_means, axis=0), axis=0))
    r_ref = _unit_vec(np.cross(h_ref, e_ref))
    M_ref = np.stack([h_ref, e_ref, r_ref], axis=1).astype(np.float32)
    inv_M_ref = np.linalg.inv(M_ref.astype(np.float64)).astype(np.float32)
    c99_ref = np.mean(np.stack(center_c99_means, axis=0), axis=0).astype(np.float32)
    white_ref = np.mean(np.stack(center_wp_means, axis=0), axis=0).astype(np.float32)

    return {
        "h_ref": h_ref,
        "e_ref": e_ref,
        "r_ref": r_ref,
        "M_ref": M_ref,
        "inv_M_ref": inv_M_ref,
        "c_h99_ref": np.float32(c99_ref[0]),
        "c_e99_ref": np.float32(c99_ref[1]),
        "white_ref": white_ref,
        "slide_stain_matrices": np.stack(slide_matrices, axis=0).astype(np.float32),
        "slide_white_points": slide_wp_arr.astype(np.float32),
        "slide_c99": slide_c99_arr.astype(np.float32),
    }


def sram_deconvolve_patch(
    rgb_u8: np.ndarray, stain_prior: dict, return_canon_rgb: bool = False
):
    """
    Deterministic per-patch Shrinkage-Regularized Adaptive Macenko (SRAM) deconvolution.
    Identical execution path across train, validation, and unseen-center test samples.
    """
    rgb_f32 = rgb_u8.astype(np.float32)
    h_ref = stain_prior["h_ref"]
    e_ref = stain_prior["e_ref"]
    c_h99_ref = float(stain_prior["c_h99_ref"])
    c_e99_ref = float(stain_prior["c_e99_ref"])
    white_ref = stain_prior["white_ref"]

    wp, _ = estimate_patch_whitepoint(rgb_f32, white_ref)
    od = -0.4342944819 * np.log(np.clip((rgb_f32 + 1.0) / (wp + 1.0), 1e-4, 1.0))

    # Subsample 2x2 grid for fast, robust 3x3 SVD wedge angle estimation
    od_sub = od[::2, ::2].reshape(-1, 3)
    od_sub_sq = np.sum(od_sub * od_sub, axis=1)
    valid_sub = (od_sub_sq > 0.0225) & (np.max(od_sub, axis=1) < 2.7)
    fg_sub = od_sub[valid_sub]
    n_fg_est = int(fg_sub.shape[0]) * 4

    gamma = 0.0
    if fg_sub.shape[0] >= 64:
        h_cand, e_cand, spread, ok = svd_macenko_vectors_from_od(fg_sub, h_ref, e_ref)
        if ok:
            cos_he = float(np.dot(h_cand, e_cand))
            min_align = min(float(np.dot(h_cand, h_ref)), float(np.dot(e_cand, e_ref)))
            w_fg = float(n_fg_est) / float(n_fg_est + 2048.0)
            w_spread = float(np.clip(spread / 0.32, 0.0, 1.0))
            w_sep = float(np.clip((0.985 - cos_he) / 0.12, 0.0, 1.0))
            w_plaus = float(np.clip((min_align - 0.72) / 0.22, 0.0, 1.0))
            gamma = 0.85 * w_fg * w_spread * w_sep * w_plaus
            h_vec = _unit_vec(gamma * h_cand + (1.0 - gamma) * h_ref)
            e_vec = _unit_vec(gamma * e_cand + (1.0 - gamma) * e_ref)
        else:
            h_vec, e_vec = h_ref, e_ref
    else:
        h_vec, e_vec = h_ref, e_ref

    r_raw = np.cross(h_vec, e_vec)
    if float(np.linalg.norm(r_raw)) < 1e-4:
        inv_M = stain_prior["inv_M_ref"]
    else:
        r_vec = _unit_vec(r_raw)
        M_adapt = np.stack([h_vec, e_vec, r_vec], axis=1).astype(np.float64)
        inv_M = np.linalg.inv(M_adapt).astype(np.float32)

    conc = (od.reshape(-1, 3) @ inv_M.T).reshape(PATCH_SIZE, PATCH_SIZE, 3)
    c_h_raw = np.maximum(conc[:, :, 0], 0.0)
    c_e_raw = np.maximum(conc[:, :, 1], 0.0)
    c_r_abs = np.abs(conc[:, :, 2])

    fg_mask = np.sum(od * od, axis=-1) > 0.0225
    fg_count = int(np.count_nonzero(fg_mask))
    if fg_count >= 512:
        h99_emp = float(np.percentile(c_h_raw[fg_mask], 99.0))
        e99_emp = float(np.percentile(c_e_raw[fg_mask], 99.0))
        w_sc = 0.65 * (float(fg_count) / float(fg_count + 4096.0))
        scale_h = (
            w_sc * np.clip(h99_emp, 0.35 * c_h99_ref, 2.8 * c_h99_ref)
            + (1.0 - w_sc) * c_h99_ref
        )
        scale_e = (
            w_sc * np.clip(e99_emp, 0.35 * c_e99_ref, 2.8 * c_e99_ref)
            + (1.0 - w_sc) * c_e99_ref
        )
    else:
        scale_h, scale_e = c_h99_ref, c_e99_ref

    c_h = (c_h_raw * (c_h99_ref / (scale_h + 1e-6))).astype(np.float32)
    c_e = (c_e_raw * (c_e99_ref / (scale_e + 1e-6))).astype(np.float32)

    canon_rgb = None
    if return_canon_rgb:
        M_ref_he = stain_prior["M_ref"][:, :2]  # (3, 2)
        he_flat = np.stack([c_h.ravel(), c_e.ravel()], axis=1)
        od_canon = (he_flat @ M_ref_he.T).reshape(PATCH_SIZE, PATCH_SIZE, 3)
        canon_rgb = (
            np.clip(white_ref * np.exp(-2.302585093 * od_canon), 0.0, 255.0).astype(
                np.float32
            )
            / 255.0
        )

    return c_h, c_e, c_r_abs, od, fg_mask, float(gamma), canon_rgb


# ============================================================================
# Step 1.3: Six-Family Histopathology Domain Feature Extraction
# ============================================================================
def _cubical_betti_euler(binary_mask: np.ndarray) -> tuple:
    """
    Exact 2D 4-connected cubical topology:
    chi = V - E_h - E_v + F_2x2, beta_0 = connectedComponents, beta_1 = beta_0 - chi.
    """
    b = binary_mask.astype(np.uint8)
    v = int(b.sum())
    if v == 0:
        return 0.0, 0.0, 0.0
    eh = int((b[:, :-1] & b[:, 1:]).sum())
    ev = int((b[:-1, :] & b[1:, :]).sum())
    f2 = int((b[:-1, :-1] & b[:-1, 1:] & b[1:, :-1] & b[1:, 1:]).sum())
    chi = v - eh - ev + f2
    n_labels = int(cv2.connectedComponents(b, connectivity=4)[0]) - 1
    beta0 = max(0, n_labels)
    beta1 = max(0, beta0 - chi)
    norm_area = float(b.size) / 1000.0  # density per 1,000 pixels
    return float(beta0) / norm_area, float(beta1) / norm_area, float(chi) / norm_area


def _extract_topology_features(c_h: np.ndarray, c_e: np.ndarray) -> list:
    feats = []
    abs_taus = [0.15, 0.30, 0.50, 0.75, 1.05]
    b0_curve, b1_curve = [], []
    for tau in abs_taus:
        b0, b1, chi = _cubical_betti_euler(c_h >= tau)
        feats.extend([b0, b1, chi])
        b0_curve.append(b0)
        b1_curve.append(b1)

    # Scale-free patch-quantile filtration (invariant to monotonic contrast shifts)
    q_levels = np.percentile(c_h, [70.0, 82.0, 90.0, 95.0])
    for qt in q_levels:
        if qt > 1e-4:
            b0, b1, chi = _cubical_betti_euler(c_h >= qt)
        else:
            b0, b1, chi = 0.0, 0.0, 0.0
        feats.extend([b0, b1, chi])

    # Summary persistence functionals + Eosin stromal/sinus topology
    b0_arr = np.array(b0_curve, dtype=np.float32)
    b1_arr = np.array(b1_curve, dtype=np.float32)
    b0_auc = float(np.mean(b0_arr))
    b1_auc = float(np.mean(b1_arr))
    peak_hole_ratio = float(np.max(b1_arr / (b0_arr + 0.05)))
    e_b0, e_b1, e_chi = _cubical_betti_euler(c_e >= 0.25)
    feats.extend([b0_auc, b1_auc, peak_hole_ratio, e_b0, e_b1, e_chi])
    return feats  # 33 features


def _structure_tensor_stats(
    field: np.ndarray, sigma_grad: float, sigma_win: float
) -> list:
    sm = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma_grad, sigmaY=sigma_grad)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigmaX=sigma_win, sigmaY=sigma_win)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigmaX=sigma_win, sigmaY=sigma_win)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigmaX=sigma_win, sigmaY=sigma_win)

    tr = jxx + jyy
    diff = jxx - jyy
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * jxy * jxy, 0.0))
    lam1 = 0.5 * (tr + disc)
    lam2 = np.maximum(0.5 * (tr - disc), 0.0)

    coherency = (lam1 - lam2) / (tr + 1e-5)
    curvature = (4.0 * lam1 * lam2) / (tr + 1e-5)
    log_energy = np.log1p(tr)

    return [
        float(np.mean(coherency)),
        float(np.percentile(coherency, 90.0)),
        float(np.mean(curvature)),
        float(np.percentile(curvature, 90.0)),
        float(np.mean(log_energy)),
    ]


def _extract_structure_tensor_features(c_h: np.ndarray, c_e: np.ndarray) -> list:
    feats = []
    # Micro (nuclear membrane), Meso (cell cluster), Macro (epithelial nest/follicle)
    for sg, sw in [(1.0, 2.5), (2.2, 5.5), (4.5, 11.0)]:
        feats.extend(_structure_tensor_stats(c_h, sg, sw))
    # Stromal / Eosin fiber orientation scale
    feats.extend(_structure_tensor_stats(c_e, 2.5, 7.5)[:3])
    return feats  # 18 features


def _zone_summary(
    ch_z: np.ndarray, ce_z: np.ndarray, grad_z: np.ndarray, mask_z: np.ndarray = None
) -> list:
    if mask_z is not None:
        ch_vals = ch_z[mask_z]
        ce_vals = ce_z[mask_z]
        gr_vals = grad_z[mask_z]
        b_in = (ch_z >= 0.45) & mask_z
        area_scale = float(np.count_nonzero(mask_z)) / 1000.0
    else:
        ch_vals = ch_z.ravel()
        ce_vals = ce_z.ravel()
        gr_vals = grad_z.ravel()
        b_in = ch_z >= 0.45
        area_scale = float(ch_z.size) / 1000.0

    ch_mean = float(np.mean(ch_vals))
    ch_p90 = float(np.percentile(ch_vals, 90.0))
    ch_std = float(np.std(ch_vals))
    ce_mean = float(np.mean(ce_vals))
    he_ratio = ch_mean / (ce_mean + 0.05)
    gr_mean = float(np.mean(gr_vals))

    b0, b1, _ = _cubical_betti_euler(b_in)
    full_scale = float(ch_z.size) / 1000.0
    b0 = b0 * (full_scale / max(area_scale, 1e-3))
    b1 = b1 * (full_scale / max(area_scale, 1e-3))
    return [ch_mean, ch_p90, ch_std, ce_mean, he_ratio, b0, b1, gr_mean]


def _extract_multizone_features(c_h: np.ndarray, c_e: np.ndarray) -> list:
    gx = cv2.Sobel(c_h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(c_h, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    # Core 96x96 [80:176], Mid 160x160 [48:208] ring, Outer peripheral ring
    core_feats = _zone_summary(
        c_h[80:176, 80:176], c_e[80:176, 80:176], grad[80:176, 80:176]
    )

    mid_mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    mid_mask[48:208, 48:208] = True
    mid_mask[80:176, 80:176] = False
    mid_feats = _zone_summary(c_h, c_e, grad, mask_z=mid_mask)

    outer_mask = np.ones((PATCH_SIZE, PATCH_SIZE), dtype=bool)
    outer_mask[48:208, 48:208] = False
    outer_feats = _zone_summary(c_h, c_e, grad, mask_z=outer_mask)

    return core_feats + mid_feats + outer_feats  # 24 features


def _glcm_stats_from_quantized(
    q_img: np.ndarray,
    dist: int,
    valid_mask: np.ndarray = None,
    include_shade: bool = False,
) -> list:
    counts = np.zeros(GLCM_LEVELS * GLCM_LEVELS, dtype=np.int64)
    offsets = [(0, dist), (dist, 0), (dist, dist), (dist, -dist)]
    h, w = q_img.shape
    for dy, dx in offsets:
        r0_a = max(0, -dy)
        r1_a = min(h, h - dy)
        c0_a = max(0, -dx)
        c1_a = min(w, w - dx)
        a = q_img[r0_a:r1_a, c0_a:c1_a]
        b = q_img[r0_a + dy : r1_a + dy, c0_a + dx : c1_a + dx]
        if valid_mask is not None:
            ma = valid_mask[r0_a:r1_a, c0_a:c1_a]
            mb = valid_mask[r0_a + dy : r1_a + dy, c0_a + dx : c1_a + dx]
            m = ma & mb
            if np.any(m):
                idx = a[m].astype(np.int32) * GLCM_LEVELS + b[m].astype(np.int32)
                counts += np.bincount(idx, minlength=GLCM_LEVELS * GLCM_LEVELS)
        else:
            idx = a.ravel().astype(np.int32) * GLCM_LEVELS + b.ravel().astype(np.int32)
            counts += np.bincount(idx, minlength=GLCM_LEVELS * GLCM_LEVELS)

    mat = counts.reshape(GLCM_LEVELS, GLCM_LEVELS).astype(np.float64)
    mat = mat + mat.T
    total = float(mat.sum())
    if total < 16.0:
        return [0.0] * (6 if include_shade else 5)

    P = (mat / total).astype(np.float32)
    contrast = float(np.sum(P * _GLCM_DIFF_SQ))
    homog = float(np.sum(P * _GLCM_IDM_W))
    energy = float(np.sum(P * P))
    entropy = float(-np.sum(P * np.log(P + 1e-12)))

    marg = np.sum(P, axis=1)
    levels = np.arange(GLCM_LEVELS, dtype=np.float32)
    mu = float(np.sum(levels * marg))
    var = float(np.sum(((levels - mu) ** 2) * marg))
    if var > 1e-6:
        corr = float(np.sum(P * (_II - mu) * (_JJ - mu)) / var)
        shade = float(np.sum(P * ((_II + _JJ - 2.0 * mu) ** 3)) / ((var**1.5) + 1e-6))
    else:
        corr, shade = 0.0, 0.0

    if include_shade:
        return [contrast, homog, energy, entropy, corr, shade]
    return [contrast, homog, energy, entropy, corr]


def _extract_glcm_features(c_h: np.ndarray, fg_mask: np.ndarray) -> list:
    # Standardized physical chromatin quantization in [0, 1.6]
    q_phys = np.clip((c_h * (15.999 / 1.6)).astype(np.int32), 0, GLCM_LEVELS - 1)
    f_d1 = _glcm_stats_from_quantized(
        q_phys, dist=1, valid_mask=None, include_shade=False
    )
    f_d4 = _glcm_stats_from_quantized(
        q_phys, dist=4, valid_mask=None, include_shade=False
    )

    # Rank-equalized foreground quantization (100% invariant to monotonic intensity scaling)
    fg_count = int(np.count_nonzero(fg_mask))
    if fg_count >= 256:
        q_edges = np.percentile(
            c_h[fg_mask], np.linspace(0.0, 100.0, GLCM_LEVELS + 1)[1:-1]
        )
        q_rank = np.searchsorted(q_edges, c_h).astype(np.int32)
        f_rank = _glcm_stats_from_quantized(
            q_rank, dist=2, valid_mask=fg_mask, include_shade=True
        )
    else:
        f_rank = [0.0] * 6

    return f_d1 + f_d4 + f_rank  # 16 features


def _extract_cytomorphometry_features(c_h: np.ndarray) -> list:
    ch_sm = cv2.GaussianBlur(c_h, (3, 3), 0.7)

    # 1) Cohesive chromatin mask for rim-to-interior margination & syncytial sheet fraction
    mask_mod = (ch_sm > 0.38).astype(np.uint8)
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    interior = cv2.erode(mask_mod, kernel3, iterations=1)
    rim = mask_mod & (1 - interior)
    n_int = int(np.count_nonzero(interior))
    n_rim = int(np.count_nonzero(rim))
    if n_int >= 32 and n_rim >= 32:
        rim_mean = float(np.mean(c_h[rim > 0]))
        int_mean = float(np.mean(c_h[interior > 0]))
        rim_to_int_ratio = rim_mean / (int_mean + 0.05)
    else:
        rim_to_int_ratio = 1.0

    n_mod, _, stats_mod, _ = cv2.connectedComponentsWithStats(mask_mod, connectivity=8)
    if n_mod > 1:
        areas_mod = stats_mod[1:, cv2.CC_STAT_AREA]
        syncytial_area_frac = float(np.sum(areas_mod[areas_mod > 450])) / float(
            PATCH_SIZE * PATCH_SIZE
        )
    else:
        syncytial_area_frac = 0.0

    # 2) DoG-separated individual nuclear objects for pleomorphism & spatial point process
    ch_bg = cv2.GaussianBlur(c_h, (13, 13), 3.0)
    nuc_cores = ((ch_sm > 0.30) & ((ch_sm - ch_bg) > 0.025)).astype(np.uint8)
    n_lbl, lbl_map, stats, centroids = cv2.connectedComponentsWithStats(
        nuc_cores, connectivity=8
    )

    if n_lbl <= 1:
        return [
            0.0,
            0.0,
            0.0,
            syncytial_area_frac,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            rim_to_int_ratio,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    pts = centroids[1:].astype(np.float32)

    # Compute mean Hematoxylin per segmented nucleus via vectorized bincount
    lbl_flat = lbl_map.ravel()
    sum_ch = np.bincount(lbl_flat, weights=c_h.ravel(), minlength=n_lbl)[1:]
    mean_ch_per_nuc = sum_ch / np.maximum(areas, 1.0)

    valid_nuc = (areas >= 6.0) & (areas <= 1200.0)
    if not np.any(valid_nuc):
        return [
            0.0,
            0.0,
            0.0,
            syncytial_area_frac,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            rim_to_int_ratio,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

    areas_v = areas[valid_nuc]
    w_v = widths[valid_nuc]
    h_v = heights[valid_nuc]
    ch_v = mean_ch_per_nuc[valid_nuc]
    pts_v = pts[valid_nuc]

    bbox_solidity = areas_v / np.maximum(w_v * h_v, 1.0)
    aspect = np.maximum(w_v, h_v) / np.maximum(np.minimum(w_v, h_v), 1.0)

    n_total = float(areas_v.size)
    # Small condensed hyperchromatic lymphocytes vs enlarged pleomorphic nuclei
    n_lymph = float(
        np.sum((areas_v >= 8.0) & (areas_v <= 42.0) & (ch_v >= 0.55) & (aspect <= 1.8))
    )
    n_atypical = float(np.sum((areas_v >= 48.0) & (areas_v <= 500.0)))

    area_mean = float(np.mean(areas_v))
    area_p90 = float(np.percentile(areas_v, 90.0))
    area_p99 = float(np.percentile(areas_v, 99.0))
    q25, q50, q75 = np.percentile(areas_v, [25.0, 50.0, 75.0])
    area_iqr_ratio = float((q75 - q25) / (q50 + 1.0))

    sol_mean = float(np.mean(bbox_solidity))
    sol_p10 = float(np.percentile(bbox_solidity, 10.0))
    asp_p90 = float(np.percentile(aspect, 90.0))

    # Nearest-neighbor & k=5 crowding dispersion via cKDTree
    n_pts = pts_v.shape[0]
    if n_pts >= 3:
        k_query = min(6, n_pts)
        tree = cKDTree(pts_v)
        dists, _ = tree.query(pts_v, k=k_query)
        nn1 = dists[:, 1].astype(np.float32)
        nn1_mean = float(np.mean(nn1))
        nn1_std = float(np.std(nn1))
        nn1_cv = nn1_std / (nn1_mean + 1e-4)
        nn1_p10 = float(np.percentile(nn1, 10.0))
        nn1_p90 = float(np.percentile(nn1, 90.0))

        nnk = dists[:, -1].astype(np.float32)
        nn5_mean = float(np.mean(nnk))
        nn5_cv = float(np.std(nnk)) / (nn5_mean + 1e-4)

        density = float(n_pts) / float(PATCH_SIZE * PATCH_SIZE)
        clark_evans = 2.0 * nn1_mean * float(np.sqrt(density))
    elif n_pts == 2:
        d12 = float(np.linalg.norm(pts_v[0] - pts_v[1]))
        nn1_mean, nn1_std, nn1_cv, nn1_p10, nn1_p90 = d12, 0.0, 0.0, d12, d12
        nn5_mean, nn5_cv, clark_evans = d12, 0.0, 2.0 * d12 * np.sqrt(2.0 / 65536.0)
    else:
        nn1_mean = nn1_std = nn1_cv = nn1_p10 = nn1_p90 = 0.0
        nn5_mean = nn5_cv = clark_evans = 0.0

    return [
        n_total,
        n_lymph,
        n_atypical,
        syncytial_area_frac,
        area_mean,
        area_p90,
        area_p99,
        area_iqr_ratio,
        sol_mean,
        sol_p10,
        asp_p90,
        rim_to_int_ratio,
        nn1_mean,
        nn1_std,
        nn1_cv,
        nn1_p10,
        nn1_p90,
        nn5_mean,
        nn5_cv,
        clark_evans,
    ]  # 20 features


def _extract_color_physics_features(
    c_h: np.ndarray,
    c_e: np.ndarray,
    c_r_abs: np.ndarray,
    od: np.ndarray,
    fg_mask: np.ndarray,
    gamma: float,
) -> list:
    fg_frac = float(np.mean(fg_mask))

    # Clear adipose / vacuole globules (smooth circular clear spaces bounded by tissue)
    clear_mask = (~fg_mask).astype(np.uint8)
    n_cl, _, stats_cl, _ = cv2.connectedComponentsWithStats(clear_mask, connectivity=4)
    if n_cl > 1:
        a_cl = stats_cl[1:, cv2.CC_STAT_AREA]
        w_cl = stats_cl[1:, cv2.CC_STAT_WIDTH]
        h_cl = stats_cl[1:, cv2.CC_STAT_HEIGHT]
        fill_cl = a_cl / np.maximum(w_cl * h_cl, 1)
        globules = (a_cl >= 64) & (a_cl <= 8000) & (fill_cl >= 0.45)
        adipose_count = float(np.sum(globules))
        adipose_frac = float(np.sum(a_cl[globules])) / float(PATCH_SIZE * PATCH_SIZE)
    else:
        adipose_count, adipose_frac = 0.0, 0.0

    ch_p50, ch_p90, ch_p99 = np.percentile(c_h, [50.0, 90.0, 99.0])
    ce_p50, ce_p90, ce_p99 = np.percentile(c_e, [50.0, 90.0, 99.0])
    cr_mean = float(np.mean(c_r_abs))
    cr_p95, cr_p99 = np.percentile(c_r_abs, [95.0, 99.0])

    # Multi-scale scanner MTF focus/sharpness log-variance on total OD intensity
    od_gray = np.mean(od, axis=-1).astype(np.float32)
    lap_fine = cv2.Laplacian(od_gray, cv2.CV_32F, ksize=3)
    od_half = cv2.resize(od_gray, (128, 128), interpolation=cv2.INTER_AREA)
    lap_coarse = cv2.Laplacian(od_half, cv2.CV_32F, ksize=3)
    sharp_fine = float(np.log1p(np.var(lap_fine) * 1e4))
    sharp_coarse = float(np.log1p(np.var(lap_coarse) * 1e4))

    return [
        fg_frac,
        adipose_count,
        adipose_frac,
        float(ch_p50),
        float(ch_p90),
        float(ch_p99),
        float(ce_p50),
        float(ce_p90),
        float(ce_p99),
        cr_mean,
        float(cr_p95),
        float(cr_p99),
        float(gamma),
        sharp_fine,
        sharp_coarse,
    ]  # 15 features


def get_base_feature_schema() -> tuple:
    names = []
    families = {}

    start = len(names)
    for t in ["015", "030", "050", "075", "105"]:
        names.extend([f"topo_ch_t{t}_b0", f"topo_ch_t{t}_b1", f"topo_ch_t{t}_chi"])
    for q in ["70", "82", "90", "95"]:
        names.extend([f"topo_ch_q{q}_b0", f"topo_ch_q{q}_b1", f"topo_ch_q{q}_chi"])
    names.extend(
        [
            "topo_ch_b0_auc",
            "topo_ch_b1_auc",
            "topo_ch_peak_hole_ratio",
            "topo_ce_t025_b0",
            "topo_ce_t025_b1",
            "topo_ce_t025_chi",
        ]
    )
    families["topology"] = list(range(start, len(names)))

    start = len(names)
    for sc in ["micro", "meso", "macro"]:
        names.extend(
            [
                f"st_ch_{sc}_coh_mean",
                f"st_ch_{sc}_coh_p90",
                f"st_ch_{sc}_curv_mean",
                f"st_ch_{sc}_curv_p90",
                f"st_ch_{sc}_log_energy",
            ]
        )
    names.extend(
        ["st_ce_stroma_coh_mean", "st_ce_stroma_coh_p90", "st_ce_stroma_curv_mean"]
    )
    families["structure_tensor"] = list(range(start, len(names)))

    start = len(names)
    for zn in ["core", "mid", "outer"]:
        names.extend(
            [
                f"zone_{zn}_ch_mean",
                f"zone_{zn}_ch_p90",
                f"zone_{zn}_ch_std",
                f"zone_{zn}_ce_mean",
                f"zone_{zn}_he_ratio",
                f"zone_{zn}_b0",
                f"zone_{zn}_b1",
                f"zone_{zn}_grad_mean",
            ]
        )
    families["multizone"] = list(range(start, len(names)))

    start = len(names)
    for tag in ["phys_d1", "phys_d4"]:
        names.extend(
            [
                f"glcm_{tag}_contrast",
                f"glcm_{tag}_homog",
                f"glcm_{tag}_energy",
                f"glcm_{tag}_entropy",
                f"glcm_{tag}_corr",
            ]
        )
    names.extend(
        [
            "glcm_rank_d2_contrast",
            "glcm_rank_d2_homog",
            "glcm_rank_d2_energy",
            "glcm_rank_d2_entropy",
            "glcm_rank_d2_corr",
            "glcm_rank_d2_shade",
        ]
    )
    families["chromatin_glcm"] = list(range(start, len(names)))

    start = len(names)
    names.extend(
        [
            "cyto_n_total",
            "cyto_n_lymph",
            "cyto_n_atypical",
            "cyto_syncytial_frac",
            "cyto_area_mean",
            "cyto_area_p90",
            "cyto_area_p99",
            "cyto_area_iqr_ratio",
            "cyto_sol_mean",
            "cyto_sol_p10",
            "cyto_aspect_p90",
            "cyto_rim_to_int_ratio",
            "cyto_nn1_mean",
            "cyto_nn1_std",
            "cyto_nn1_cv",
            "cyto_nn1_p10",
            "cyto_nn1_p90",
            "cyto_nn5_mean",
            "cyto_nn5_cv",
            "cyto_clark_evans",
        ]
    )
    families["cytomorphometry"] = list(range(start, len(names)))

    start = len(names)
    names.extend(
        [
            "phys_fg_frac",
            "phys_adipose_count",
            "phys_adipose_frac",
            "phys_ch_p50",
            "phys_ch_p90",
            "phys_ch_p99",
            "phys_ce_p50",
            "phys_ce_p90",
            "phys_ce_p99",
            "phys_cr_mean",
            "phys_cr_p95",
            "phys_cr_p99",
            "phys_sram_gamma",
            "phys_sharp_fine",
            "phys_sharp_coarse",
        ]
    )
    families["color_physics"] = list(range(start, len(names)))

    return names, families  # 126 base features


def extract_single_patch_features(
    image_full_path: str, stain_prior: dict
) -> np.ndarray:
    rgb = read_rgb_patch(image_full_path)
    c_h, c_e, c_r_abs, od, fg_mask, gamma, _ = sram_deconvolve_patch(
        rgb, stain_prior, return_canon_rgb=False
    )
    feats = []
    feats.extend(_extract_topology_features(c_h, c_e))
    feats.extend(_extract_structure_tensor_features(c_h, c_e))
    feats.extend(_extract_multizone_features(c_h, c_e))
    feats.extend(_extract_glcm_features(c_h, fg_mask))
    feats.extend(_extract_cytomorphometry_features(c_h))
    feats.extend(_extract_color_physics_features(c_h, c_e, c_r_abs, od, fg_mask, gamma))
    return np.asarray(feats, dtype=np.float32)


_WORKER_STAIN_PRIOR = None
_WORKER_INPUT_DIR = None


def _init_feature_worker(stain_prior: dict, input_dir_str: str) -> None:
    global _WORKER_STAIN_PRIOR, _WORKER_INPUT_DIR
    cv2.setNumThreads(0)
    _WORKER_STAIN_PRIOR = stain_prior
    _WORKER_INPUT_DIR = Path(input_dir_str)


def _worker_extract_row(rel_image_path: str) -> np.ndarray:
    return extract_single_patch_features(
        str(_WORKER_INPUT_DIR / rel_image_path), _WORKER_STAIN_PRIOR
    )


def extract_features_parallel(
    image_paths: list, stain_prior: dict, input_dir: Path = INPUT_DIR
) -> np.ndarray:
    n_workers = max(2, min(os.cpu_count() or 4, 12))
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_feature_worker,
        initargs=(stain_prior, str(input_dir)),
    ) as pool:
        results = list(pool.map(_worker_extract_row, image_paths, chunksize=128))
    return np.stack(results, axis=0).astype(np.float32)


# ============================================================================
# Step 1.4: Tabular Domain Interactions & Leak-Free Multi-Modal Scaling
# ============================================================================
def compute_domain_interactions(X_base: np.ndarray, base_names: list) -> tuple:
    idx = {name: i for i, name in enumerate(base_names)}
    inter_cols = []
    inter_names = []

    def _add(name: str, arr: np.ndarray):
        inter_names.append(name)
        inter_cols.append(arr.astype(np.float32))

    # 1) Multi-zone focal core vs outer/mid context contrasts & ratios
    _add(
        "inter_core_minus_outer_ch_mean",
        X_base[:, idx["zone_core_ch_mean"]] - X_base[:, idx["zone_outer_ch_mean"]],
    )
    _add(
        "inter_core_minus_outer_ch_p90",
        X_base[:, idx["zone_core_ch_p90"]] - X_base[:, idx["zone_outer_ch_p90"]],
    )
    _add(
        "inter_core_minus_outer_b1",
        X_base[:, idx["zone_core_b1"]] - X_base[:, idx["zone_outer_b1"]],
    )
    _add(
        "inter_core_minus_outer_b0",
        X_base[:, idx["zone_core_b0"]] - X_base[:, idx["zone_outer_b0"]],
    )
    _add(
        "inter_core_minus_mid_grad",
        X_base[:, idx["zone_core_grad_mean"]] - X_base[:, idx["zone_mid_grad_mean"]],
    )
    _add(
        "inter_core_to_outer_he_ratio",
        (X_base[:, idx["zone_core_he_ratio"]] + 0.05)
        / (X_base[:, idx["zone_outer_he_ratio"]] + 0.05),
    )

    # 2) Cytological pleomorphism, lymphocyte depletion & vesicular clearing interactions
    n_tot = X_base[:, idx["cyto_n_total"]]
    n_lym = X_base[:, idx["cyto_n_lymph"]]
    n_atyp = X_base[:, idx["cyto_n_atypical"]]
    _add("inter_atypical_frac", n_atyp / (n_tot + 1.0))
    _add("inter_lymph_frac", n_lym / (n_tot + 1.0))
    _add("inter_atypical_to_lymph_ratio", (n_atyp + 0.5) / (n_lym + 1.0))
    _add(
        "inter_vesicular_b1_per_nucleus",
        X_base[:, idx["topo_ch_t050_b1"]] / (n_tot / 65.536 + 0.25),
    )
    _add(
        "inter_vesicular_pleomorphism_product",
        X_base[:, idx["cyto_rim_to_int_ratio"]]
        * np.log1p(np.maximum(X_base[:, idx["cyto_area_p90"]], 0.0)),
    )
    _add(
        "inter_syncytial_x_macro_coherency",
        X_base[:, idx["cyto_syncytial_frac"]] * X_base[:, idx["st_ch_macro_coh_mean"]],
    )

    # 3) Cross-scale structure tensor & chromatin GLCM ratios
    _add(
        "inter_st_macro_to_micro_coh_ratio",
        (X_base[:, idx["st_ch_macro_coh_mean"]] + 0.02)
        / (X_base[:, idx["st_ch_micro_coh_mean"]] + 0.02),
    )
    _add(
        "inter_st_micro_to_macro_curv_ratio",
        (X_base[:, idx["st_ch_micro_curv_mean"]] + 0.01)
        / (X_base[:, idx["st_ch_macro_curv_mean"]] + 0.01),
    )
    _add(
        "inter_glcm_d4_to_d1_contrast_ratio",
        (X_base[:, idx["glcm_phys_d4_contrast"]] + 0.05)
        / (X_base[:, idx["glcm_phys_d1_contrast"]] + 0.05),
    )
    _add(
        "inter_glcm_d4_minus_d1_entropy",
        X_base[:, idx["glcm_phys_d4_entropy"]] - X_base[:, idx["glcm_phys_d1_entropy"]],
    )

    # 4) Topological Betti-1 vesicular shift & anthracotic pigment/MTF corrections
    _add(
        "inter_topo_q82_b1_to_b0_ratio",
        X_base[:, idx["topo_ch_q82_b1"]] / (X_base[:, idx["topo_ch_q82_b0"]] + 0.1),
    )
    _add(
        "inter_topo_t030_minus_t075_chi",
        X_base[:, idx["topo_ch_t030_chi"]] - X_base[:, idx["topo_ch_t075_chi"]],
    )
    _add(
        "inter_pigment_adjusted_ch_p99",
        X_base[:, idx["phys_ch_p99"]] - 1.5 * X_base[:, idx["phys_cr_p99"]],
    )
    _add(
        "inter_mtf_fine_to_coarse_sharp_ratio",
        (X_base[:, idx["phys_sharp_fine"]] + 0.1)
        / (X_base[:, idx["phys_sharp_coarse"]] + 0.1),
    )
    _add(
        "inter_cellular_non_adipose_fg",
        np.maximum(
            X_base[:, idx["phys_fg_frac"]] - X_base[:, idx["phys_adipose_frac"]], 0.0
        ),
    )
    _add(
        "inter_ch_to_ce_p90_ratio",
        (X_base[:, idx["phys_ch_p90"]] + 0.05) / (X_base[:, idx["phys_ce_p90"]] + 0.05),
    )

    X_inter = np.column_stack(inter_cols).astype(np.float32)
    return X_inter, inter_names  # 22 interaction features -> 148 total features


class TabularFeaturePipeline:
    """
    Stateful leak-free tabular preprocessor fitted ONLY on training rows:
    1. Finite sanitization + training-median imputation
    2. Training [0.1%, 99.9%] quantile Winsorization (prevents unseen-center artifact spikes)
    3. Dual representation output:
       - X_tree: Winsorized physical/topological/interaction features for GBDTs
       - X_neural: Skew-compressed + RobustScaler + smooth arcsinh-bounded features for neural cross-attention
    """

    def __init__(self, clip_quantiles: tuple = (0.1, 99.9)):
        self.clip_quantiles = clip_quantiles
        self.medians_ = None
        self.lo_bounds_ = None
        self.hi_bounds_ = None
        self.skew_mask_ = None
        self.scaler_ = None
        self.scale_fix_ = None
        self.feature_names_ = None
        self.family_indices_ = None

    def fit(self, X_base_train: np.ndarray, base_names: list, base_families: dict):
        X_inter, inter_names = compute_domain_interactions(X_base_train, base_names)
        X_full = np.hstack([X_base_train, X_inter]).astype(np.float64)

        self.feature_names_ = list(base_names) + list(inter_names)
        self.family_indices_ = {k: list(v) for k, v in base_families.items()}
        self.family_indices_["interactions"] = list(
            range(len(base_names), len(base_names) + len(inter_names))
        )

        # Ensure zero metadata columns exist in feature space
        forbidden = {"sample_id", "slide_id", "group_id", "center_id", "label"}
        assert forbidden.isdisjoint(
            set(self.feature_names_)
        ), "Metadata leakage in feature names!"

        X_clean = np.where(np.isfinite(X_full), X_full, np.nan)
        self.medians_ = np.nanmedian(X_clean, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        nan_rows, nan_cols = np.where(np.isnan(X_clean))
        if nan_rows.size > 0:
            X_clean[nan_rows, nan_cols] = self.medians_[nan_cols]

        self.lo_bounds_ = np.percentile(X_clean, self.clip_quantiles[0], axis=0)
        self.hi_bounds_ = np.percentile(X_clean, self.clip_quantiles[1], axis=0)
        X_win = np.clip(X_clean, self.lo_bounds_, self.hi_bounds_)

        # Detect heavy-tailed features strictly on training set for signed-log1p compression
        centered = X_win - np.mean(X_win, axis=0)
        stds = np.std(X_win, axis=0)
        skews = np.where(
            stds > 1e-6, np.mean((centered / (stds + 1e-8)) ** 3, axis=0), 0.0
        )
        self.skew_mask_ = (np.abs(skews) > 1.75) & (self.lo_bounds_ >= -1.0)

        X_neu = X_win.copy()
        X_neu[:, self.skew_mask_] = np.sign(X_neu[:, self.skew_mask_]) * np.log1p(
            np.abs(X_neu[:, self.skew_mask_])
        )

        self.scaler_ = RobustScaler(quantile_range=(25.0, 75.0))
        self.scaler_.fit(X_neu)
        iqr_scale = self.scaler_.scale_.copy()
        fallback_std = np.std(X_neu, axis=0)
        self.scale_fix_ = np.where(
            iqr_scale > 1e-5,
            iqr_scale,
            np.where(fallback_std > 1e-5, fallback_std, 1.0),
        )
        return self

    def transform(self, X_base: np.ndarray, base_names: list) -> tuple:
        X_inter, _ = compute_domain_interactions(X_base, base_names)
        X_full = np.hstack([X_base, X_inter]).astype(np.float64)

        X_clean = np.where(np.isfinite(X_full), X_full, np.nan)
        nan_rows, nan_cols = np.where(np.isnan(X_clean))
        if nan_rows.size > 0:
            X_clean[nan_rows, nan_cols] = self.medians_[nan_cols]

        X_tree = np.clip(X_clean, self.lo_bounds_, self.hi_bounds_).astype(np.float32)

        X_neu = X_tree.astype(np.float64).copy()
        X_neu[:, self.skew_mask_] = np.sign(X_neu[:, self.skew_mask_]) * np.log1p(
            np.abs(X_neu[:, self.skew_mask_])
        )
        z = (X_neu - self.scaler_.center_) / self.scale_fix_
        # Smooth, strictly monotonic arcsinh compression prevents extreme unseen-center tail blowup
        X_neural = (1.5 * np.arcsinh(z / 1.5)).astype(np.float32)
        return X_tree, X_neural


# ============================================================================
# Step 1.5: Physics-Grounded Unseen-Center Stain Augmenter & Multi-Modal Dataset
# ============================================================================
def apply_d4_symmetry(arr: np.ndarray, view_idx: int) -> np.ndarray:
    """Exact 8-element dihedral group D4 transform on (H, W, C) array."""
    k = view_idx % 4
    out = np.rot90(arr, k=k, axes=(0, 1)) if k > 0 else arr
    if view_idx >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


class DirichletMacenkoStainAugmenter:
    """
    Simulates unseen pathology centers during training by perturbing a patch's optical density
    concentrations, projecting onto an extrapolated Dirichlet mixture of real training slide stain
    matrices, shifting lamp color temperature, and applying scanner MTF blur/sharpen jitter—before
    passing through the exact deterministic SRAM canonicalizer used at validation/test time.
    """

    def __init__(self, stain_prior: dict, p_stain: float = 0.85, p_mtf: float = 0.40):
        self.stain_prior = stain_prior
        self.slide_matrices = stain_prior["slide_stain_matrices"]  # (N_slides, 3, 3)
        self.slide_whites = stain_prior["slide_white_points"]  # (N_slides, 3)
        self.p_stain = p_stain
        self.p_mtf = p_mtf

    def __call__(self, rgb_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out_f32 = rgb_u8.astype(np.float32)

        if rng.random() < self.p_stain and len(self.slide_matrices) >= 2:
            c_h, c_e, c_r_abs, _, _, _, _ = sram_deconvolve_patch(
                rgb_u8, self.stain_prior, return_canon_rgb=False
            )
            n_bank = len(self.slide_matrices)
            idx3 = rng.choice(n_bank, size=min(3, n_bank), replace=False)
            weights = rng.dirichlet(np.ones(len(idx3), dtype=np.float64)).astype(
                np.float32
            )
            M_mix = np.tensordot(weights, self.slide_matrices[idx3], axes=([0], [0]))

            # Add slight random angular perturbation to H and E vectors
            h_syn = _unit_vec(
                np.maximum(
                    M_mix[:, 0] + rng.normal(0.0, 0.045, size=3).astype(np.float32),
                    0.01,
                )
            )
            e_syn = _unit_vec(
                np.maximum(
                    M_mix[:, 1] + rng.normal(0.0, 0.045, size=3).astype(np.float32),
                    0.01,
                )
            )

            # Independent section-thickness & dye concentration log-normal scaling + additive offset
            alpha_h = float(np.exp(rng.normal(0.0, 0.20)))
            alpha_e = float(np.exp(rng.normal(0.0, 0.20)))
            beta_h = float(rng.uniform(-0.02, 0.02))
            beta_e = float(rng.uniform(-0.02, 0.02))

            ch_aug = np.maximum(c_h * alpha_h + beta_h, 0.0)
            ce_aug = np.maximum(c_e * alpha_e + beta_e, 0.0)

            # Preserve minor off-plane pigment component while synthesizing unseen-center H&E OD
            od_syn = (
                ch_aug[:, :, None] * h_syn[None, None, :]
                + ce_aug[:, :, None] * e_syn[None, None, :]
                + (0.5 * c_r_abs[:, :, None]) * self.stain_prior["r_ref"][None, None, :]
            )

            # Random lamp color-temperature shift around empirical white-point bank
            wp_base = self.slide_whites[int(rng.integers(0, len(self.slide_whites)))]
            wp_jitter = wp_base * rng.uniform(0.95, 1.03, size=3).astype(np.float32)
            wp_jitter = np.clip(wp_jitter, 215.0, 255.0)
            out_f32 = np.clip(
                wp_jitter[None, None, :] * np.exp(-2.302585093 * od_syn), 0.0, 255.0
            )

        # Simulate scanner objective numerical-aperture defocus or unsharp post-processing
        if rng.random() < self.p_mtf:
            if rng.random() < 0.55:
                sigma = float(rng.uniform(0.45, 1.05))
                out_f32 = cv2.GaussianBlur(out_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)
            else:
                blur = cv2.GaussianBlur(out_f32, (0, 0), sigmaX=0.9, sigmaY=0.9)
                amount = float(rng.uniform(0.20, 0.55))
                out_f32 = np.clip(out_f32 + amount * (out_f32 - blur), 0.0, 255.0)

        return np.clip(np.rint(out_f32), 0.0, 255.0).astype(np.uint8)


class HistopathologyMultiModalDataset(Dataset):
    """
    Unified multi-modal PyTorch Dataset providing:
    - canon_rgb: (3, 256, 256) SRAM-canonicalized H&E RGB in [0, 1]
    - stain_channels: (3, 256, 256) quantitative [C_H, C_E, |C_R|] optical density maps
    - tabular_neural: (D,) outlier-bounded continuous domain feature vector
    Validation and Test modes share the exact same deterministic preprocessing path.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        X_neural: np.ndarray,
        stain_prior: dict,
        input_dir: Path = INPUT_DIR,
        is_train: bool = False,
        tta_view_idx: int = 0,
        seed: int = SEED,
    ):
        self.sample_ids = meta_df["sample_id"].tolist()
        self.image_paths = meta_df["image_path"].tolist()
        self.has_label = "label" in meta_df.columns
        self.labels = (
            meta_df["label"].values.astype(np.float32)
            if self.has_label
            else np.full(len(meta_df), -1.0, dtype=np.float32)
        )
        self.center_idx = (
            meta_df["center_idx"].values.astype(np.int64)
            if "center_idx" in meta_df.columns
            else np.full(len(meta_df), -1, dtype=np.int64)
        )
        self.X_neural = X_neural.astype(np.float32)
        self.stain_prior = stain_prior
        self.input_dir = Path(input_dir)
        self.is_train = is_train
        self.tta_view_idx = int(tta_view_idx)
        self.base_seed = int(seed)
        self.augmenter = (
            DirichletMacenkoStainAugmenter(stain_prior) if is_train else None
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        rgb = read_rgb_patch(str(self.input_dir / self.image_paths[idx]))

        if self.is_train:
            worker_info = torch.utils.data.get_worker_info()
            w_seed = (
                self.base_seed + idx * 1009 + int(time.time_ns() % 1000003)
                if worker_info is None
                else worker_info.seed + idx * 1009
            )
            rng = np.random.default_rng(w_seed)
            d4_idx = int(rng.integers(0, 8))
            rgb = apply_d4_symmetry(rgb, d4_idx)
            rgb = self.augmenter(rgb, rng)
        elif self.tta_view_idx > 0:
            rgb = apply_d4_symmetry(rgb, self.tta_view_idx)

        c_h, c_e, c_r_abs, _, _, _, canon_rgb = sram_deconvolve_patch(
            rgb, self.stain_prior, return_canon_rgb=True
        )
        stain_ch = np.stack(
            [
                np.clip(c_h / 1.5, 0.0, 2.5),
                np.clip(c_e / 1.5, 0.0, 2.5),
                np.clip(c_r_abs / 0.5, 0.0, 2.5),
            ],
            axis=0,
        ).astype(np.float32)
        canon_chw = np.transpose(canon_rgb, (2, 0, 1)).astype(np.float32)

        return {
            "canon_rgb": torch.from_numpy(canon_chw),
            "stain_channels": torch.from_numpy(stain_ch),
            "tabular_neural": torch.from_numpy(self.X_neural[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "center_idx": torch.tensor(self.center_idx[idx], dtype=torch.long),
            "sample_id": self.sample_ids[idx],
        }


def run_step1_data_processing_and_feature_engineering() -> None:
    t0 = time.time()
    seed_everything(SEED)

    print(
        "[Step 1.1] Auditing dataset schemas, image files, and constructing leak-free splits..."
    )
    train_s1, val_s1, full_dev_s2, test_meta = load_and_prepare_splits(INPUT_DIR)
    print(
        f"  Stage 1 Splits -> train: {len(train_s1)} (slides={train_s1['slide_id'].nunique()}, "
        f"centers={train_s1['center_id'].nunique()}, pos_rate={train_s1['label'].mean():.4f}) | "
        f"official_val: {len(val_s1)} (slides={val_s1['slide_id'].nunique()}, "
        f"pos_rate={val_s1['label'].mean():.4f})"
    )
    print(
        f"  Stage 2 Merged Full-Dev Split (train + labeled validation confirmed & merged for post-val refit) -> "
        f"full_dev: {len(full_dev_s2)} (slides={full_dev_s2['slide_id'].nunique()}) | test: {len(test_meta)}"
    )

    print(
        "[Step 1.2] Fitting Stage 1 SRAM Stain Prior STRICTLY on train.csv (zero validation/test access)..."
    )
    stage1_stain_prior = fit_stain_prior_from_df(
        train_s1, INPUT_DIR, patches_per_slide=24, seed=SEED
    )
    print(
        f"  Stage 1 Reference H={np.round(stage1_stain_prior['h_ref'], 4)}, "
        f"E={np.round(stage1_stain_prior['e_ref'], 4)}, "
        f"Slide Basis Bank={len(stage1_stain_prior['slide_stain_matrices'])} slides"
    )

    print(
        "[Step 1.2b] Fitting Stage 2 Refit Slide Stain Bank on full_dev (isolated for post-val Stage 2 only)..."
    )
    stage2_stain_prior = fit_stain_prior_from_df(
        full_dev_s2, INPUT_DIR, patches_per_slide=24, seed=SEED + 7
    )
    print(
        f"  Stage 2 Refit Slide Basis Bank={len(stage2_stain_prior['slide_stain_matrices'])} slides"
    )

    base_names, base_families = get_base_feature_schema()
    print(
        f"[Step 1.3] Extracting {len(base_names)} physics/topological/cytomorphometric features across "
        f"{len(train_s1) + len(val_s1) + len(test_meta)} patches..."
    )
    t_ext = time.time()
    X_train_base = extract_features_parallel(
        train_s1["image_path"].tolist(), stage1_stain_prior, INPUT_DIR
    )
    X_val_base = extract_features_parallel(
        val_s1["image_path"].tolist(), stage1_stain_prior, INPUT_DIR
    )
    X_test_base = extract_features_parallel(
        test_meta["image_path"].tolist(), stage1_stain_prior, INPUT_DIR
    )
    print(
        f"  Parallel extraction finished in {time.time() - t_ext:.1f}s -> "
        f"shapes: train={X_train_base.shape}, val={X_val_base.shape}, test={X_test_base.shape}"
    )

    print(
        "[Step 1.4] Fitting leak-free TabularFeaturePipeline (interactions + Winsorization + RobustScaler/arcsinh)..."
    )
    # Stage 1 pipeline: fitted STRICTLY on X_train_base ONLY
    stage1_tab_pipe = TabularFeaturePipeline(clip_quantiles=(0.1, 99.9)).fit(
        X_train_base, base_names, base_families
    )
    X_train_tree_s1, X_train_neural_s1 = stage1_tab_pipe.transform(
        X_train_base, base_names
    )
    X_val_tree_s1, X_val_neural_s1 = stage1_tab_pipe.transform(X_val_base, base_names)
    X_test_tree_s1, X_test_neural_s1 = stage1_tab_pipe.transform(
        X_test_base, base_names
    )

    # Stage 2 refit pipeline: fitted on merged full_dev (strictly isolated from Stage 1 validation)
    X_fulldev_base = np.vstack([X_train_base, X_val_base])
    stage2_tab_pipe = TabularFeaturePipeline(clip_quantiles=(0.1, 99.9)).fit(
        X_fulldev_base, base_names, base_families
    )
    X_fulldev_tree_s2, X_fulldev_neural_s2 = stage2_tab_pipe.transform(
        X_fulldev_base, base_names
    )
    X_test_tree_s2, X_test_neural_s2 = stage2_tab_pipe.transform(
        X_test_base, base_names
    )

    for arr_name, arr in [
        ("X_train_tree_s1", X_train_tree_s1),
        ("X_val_tree_s1", X_val_tree_s1),
        ("X_test_tree_s1", X_test_tree_s1),
        ("X_train_neural_s1", X_train_neural_s1),
        ("X_val_neural_s1", X_val_neural_s1),
        ("X_test_neural_s1", X_test_neural_s1),
        ("X_fulldev_tree_s2", X_fulldev_tree_s2),
        ("X_fulldev_neural_s2", X_fulldev_neural_s2),
    ]:
        assert np.isfinite(arr).all(), f"Non-finite values detected in {arr_name}!"

    print(
        f"  Engineered feature dimensionality: D={len(stage1_tab_pipe.feature_names_)} across "
        f"{len(stage1_tab_pipe.family_indices_)} semantic feature families "
        f"({{k: len(v) for k, v in stage1_tab_pipe.family_indices_.items()}})"
    )

    print(
        "[Step 1.5] Verifying Multi-Modal Dataset & Dirichlet-Macenko DataLoader pipeline..."
    )
    ds_train_check = HistopathologyMultiModalDataset(
        train_s1.iloc[:8].reset_index(drop=True),
        X_train_neural_s1[:8],
        stage1_stain_prior,
        input_dir=INPUT_DIR,
        is_train=True,
    )
    ds_val_check = HistopathologyMultiModalDataset(
        val_s1.iloc[:8].reset_index(drop=True),
        X_val_neural_s1[:8],
        stage1_stain_prior,
        input_dir=INPUT_DIR,
        is_train=False,
        tta_view_idx=3,
    )
    dl_check = DataLoader(ds_train_check, batch_size=4, shuffle=False, num_workers=2)
    batch_sample = next(iter(dl_check))
    val_item = ds_val_check[0]
    assert batch_sample["canon_rgb"].shape == (4, 3, PATCH_SIZE, PATCH_SIZE)
    assert batch_sample["stain_channels"].shape == (4, 3, PATCH_SIZE, PATCH_SIZE)
    assert batch_sample["tabular_neural"].shape == (
        4,
        len(stage1_tab_pipe.feature_names_),
    )
    assert (
        torch.isfinite(batch_sample["canon_rgb"]).all()
        and torch.isfinite(val_item["canon_rgb"]).all()
    )

    print(
        "[Step 1.6] Saving splits, stain priors, feature matrices, and preprocessors to ./working/..."
    )
    train_s1.to_csv(WORKING_DIR / "train_s1_meta.csv", index=False)
    val_s1.to_csv(WORKING_DIR / "val_s1_meta.csv", index=False)
    full_dev_s2.to_csv(WORKING_DIR / "full_dev_s2_meta.csv", index=False)
    test_meta.to_csv(WORKING_DIR / "test_meta.csv", index=False)

    np.savez_compressed(WORKING_DIR / "stage1_stain_prior.npz", **stage1_stain_prior)
    np.savez_compressed(WORKING_DIR / "stage2_stain_prior.npz", **stage2_stain_prior)

    np.savez_compressed(
        WORKING_DIR / "features_stage1.npz",
        X_train_base=X_train_base,
        X_val_base=X_val_base,
        X_test_base=X_test_base,
        X_train_tree=X_train_tree_s1,
        X_val_tree=X_val_tree_s1,
        X_test_tree=X_test_tree_s1,
        X_train_neural=X_train_neural_s1,
        X_val_neural=X_val_neural_s1,
        X_test_neural=X_test_neural_s1,
        y_train=train_s1["label"].values.astype(np.int64),
        y_val=val_s1["label"].values.astype(np.int64),
        feature_names=np.array(stage1_tab_pipe.feature_names_),
    )

    np.savez_compressed(
        WORKING_DIR / "features_stage2_refit.npz",
        X_fulldev_tree=X_fulldev_tree_s2,
        X_test_tree=X_test_tree_s2,
        X_fulldev_neural=X_fulldev_neural_s2,
        X_test_neural=X_test_neural_s2,
        y_fulldev=full_dev_s2["label"].values.astype(np.int64),
        feature_names=np.array(stage2_tab_pipe.feature_names_),
    )

    with open(WORKING_DIR / "feature_metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "base_names": base_names,
                "base_families": base_families,
                "feature_names": stage1_tab_pipe.feature_names_,
                "family_indices": stage1_tab_pipe.family_indices_,
                "stage1_tabular_pipeline": stage1_tab_pipe,
                "stage2_tabular_pipeline": stage2_tab_pipe,
            },
            f,
        )

    print(f"[Step 1 Complete] All artifacts saved cleanly in {time.time() - t0:.1f}s.")


# ============================================================================
# Step 2: Model Design — BatchNorm-Free Building Blocks & Stain-Physics Pyramid
# ============================================================================
class DropPath(nn.Module):
    """Stochastic depth per sample for residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """
    Pure channels-first LayerNorm over (B, C, H, W) computed in float32.
    Eliminates BatchNorm running-statistic drift on unseen pathology centers.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f32 = x.float()
        mean = x_f32.mean(dim=1, keepdim=True)
        var = (x_f32 - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x_f32 - mean) * torch.rsqrt(var + self.eps)
        out = x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out.to(orig_dtype)


class DoGStainFilterBank(nn.Module):
    """
    Non-trainable physics filter bank operating at native 256x256 resolution:
    extracts multi-scale Difference-of-Gaussians (DoG) chromatin rim/clearing bands
    and off-plane pigment-corrected Hematoxylin purity maps from [C_H, C_E, |C_R|].
    """

    def __init__(self):
        super().__init__()
        k1 = self._make_gaussian_kernel(kernel_size=5, sigma=1.0)
        k2 = self._make_gaussian_kernel(kernel_size=9, sigma=2.4)
        self.register_buffer("kernel_fine", k1.view(1, 1, 5, 5), persistent=False)
        self.register_buffer("kernel_meso", k2.view(1, 1, 9, 9), persistent=False)
        self.register_buffer(
            "rgb_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = (
            torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        )
        g1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
        g2d = torch.outer(g1d, g1d)
        return g2d / g2d.sum()

    def forward(
        self, canon_rgb: torch.Tensor, stain_channels: torch.Tensor
    ) -> torch.Tensor:
        rgb_f32 = (canon_rgb.float() - self.rgb_mean) / self.rgb_std
        stain_f32 = stain_channels.float()

        c_h = stain_f32[:, 0:1, :, :]
        c_r = stain_f32[:, 2:3, :, :]

        ch_pad_fine = F.pad(c_h, (2, 2, 2, 2), mode="reflect")
        ch_fine = F.conv2d(ch_pad_fine, self.kernel_fine)

        ch_pad_meso = F.pad(c_h, (4, 4, 4, 4), mode="reflect")
        ch_meso = F.conv2d(ch_pad_meso, self.kernel_meso)

        dog_rim = c_h - ch_fine  # Sharp 1-2 px nuclear membrane margination
        dog_vesicular = ch_fine - ch_meso  # Intranuclear vesicular clearing band
        pure_chromatin = F.relu(c_h - 0.75 * c_r)  # Suppresses off-plane carbon pigment

        out = torch.cat(
            [rgb_f32, stain_f32, dog_rim, dog_vesicular, pure_chromatin], dim=1
        )
        return out.to(canon_rgb.dtype)  # (B, 9, 256, 256)


class MultiDilationStainBlock(nn.Module):
    """
    BatchNorm-free multi-scale depthwise-separable block pairing local 7x7 nuclear
    context with dilated 3x3 cellular-neighborhood context and LayerScale.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-4,
    ):
        super().__init__()
        self.dw_local = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.dw_dilated = nn.Conv2d(
            dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.pw1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(hidden_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        feat = 0.5 * (self.dw_local(x) + self.dw_dilated(x))
        feat = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        feat = self.norm(feat)
        feat = self.pw2(self.act(self.pw1(feat)))
        feat = self.gamma * feat
        feat = feat.permute(0, 3, 1, 2)  # (B, C, H, W)
        return residual + self.drop_path(feat)


class StainPhysicsDifferentialPyramid(nn.Module):
    """
    Full-resolution (256x256 -> 128 -> 64 -> 32) BatchNorm-free Stain-Physics Pyramid.
    Preserves sub-token chromatin margination and pigment physics before projecting
    onto both the 18x18 DINOv2 token lattice and a concentric core-vs-context vector.
    """

    def __init__(
        self,
        d_model: int = 256,
        dims: Tuple[int, int, int] = (64, 128, 192),
        drop_path_rate: float = 0.10,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.filter_bank = DoGStainFilterBank()
        c1, c2, c3 = dims
        dp = [x.item() for x in torch.linspace(0.0, drop_path_rate, 6)]

        self.stem = nn.Sequential(
            nn.Conv2d(9, c1, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(c1),
            nn.GELU(),
        )  # 128x128
        self.stage1 = nn.Sequential(
            MultiDilationStainBlock(c1, drop_path=dp[0]),
            MultiDilationStainBlock(c1, drop_path=dp[1]),
        )
        self.down2 = nn.Sequential(
            LayerNorm2d(c1),
            nn.Conv2d(c1, c2, kernel_size=2, stride=2),
        )  # 64x64
        self.stage2 = nn.Sequential(
            MultiDilationStainBlock(c2, drop_path=dp[2]),
            MultiDilationStainBlock(c2, drop_path=dp[3]),
        )
        self.down3 = nn.Sequential(
            LayerNorm2d(c2),
            nn.Conv2d(c2, c3, kernel_size=2, stride=2),
        )  # 32x32
        self.stage3 = nn.Sequential(
            MultiDilationStainBlock(c3, drop_path=dp[4]),
            MultiDilationStainBlock(c3, drop_path=dp[5]),
        )

        # LayerNorm Squeeze-and-Excitation channel recalibration
        self.channel_gate = nn.Sequential(
            nn.LayerNorm(c3),
            nn.Linear(c3, c3 // 4),
            nn.GELU(),
            nn.Linear(c3 // 4, c3),
            nn.Sigmoid(),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(c3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Concentric 32x32 readout: core [10:22, 10:22] (exact 96x96 px center),
        # core-minus-full differential, and full spatial peak
        self.global_proj = nn.Sequential(
            nn.Linear(c3 * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self, canon_rgb: torch.Tensor, stain_channels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.filter_bank(canon_rgb, stain_channels)
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down2(x))
        feat32 = self.stage3(self.down3(x))  # (B, c3, 32, 32)

        se = self.channel_gate(feat32.mean(dim=(2, 3))).unsqueeze(-1).unsqueeze(-1)
        feat32 = feat32 * se

        # Align onto 18x18 DINOv2 grid combining mean density and peak hyperchromasia
        grid_avg = F.adaptive_avg_pool2d(feat32, (DINO_GRID_SIZE, DINO_GRID_SIZE))
        grid_max = F.adaptive_max_pool2d(feat32, (DINO_GRID_SIZE, DINO_GRID_SIZE))
        grid_18 = 0.5 * (grid_avg + grid_max)
        phys_tokens = grid_18.flatten(2).transpose(1, 2)  # (B, 324, c3)
        phys_tokens = self.token_proj(phys_tokens)  # (B, 324, d_model)

        # Concentric zone-aware global physics summary
        core_mean = feat32[:, :, 10:22, 10:22].mean(dim=(2, 3))
        full_mean = feat32.mean(dim=(2, 3))
        full_max = feat32.amax(dim=(2, 3))
        z_phys = self.global_proj(
            torch.cat([core_mean, core_mean - full_mean, full_max], dim=-1)
        )
        return phys_tokens, z_phys


# ============================================================================
# Self-Supervised DINOv2 Multi-Depth Stream & Concentric Gated-MIL Readout
# ============================================================================
def _build_dinov2_vit(
    backbone_name: str = DEFAULT_DINOV2_BACKBONE,
    pretrained: bool = True,
    drop_path_rate: float = 0.15,
) -> nn.Module:
    available = set(timm.list_models("*dinov2*"))
    chosen = backbone_name
    if chosen not in available and FALLBACK_DINOV2_BACKBONE in available:
        chosen = FALLBACK_DINOV2_BACKBONE

    create_sig = inspect.signature(timm.create_model)
    kwargs = dict(
        pretrained=pretrained,
        num_classes=0,
        img_size=DINO_INPUT_SIZE,
        drop_path_rate=drop_path_rate,
    )
    if "dynamic_img_size" in create_sig.parameters:
        kwargs["dynamic_img_size"] = False
    return timm.create_model(chosen, **kwargs)


class DINOv2MultiDepthBackbone(nn.Module):
    """
    Extracts both mid-level cytological texture (block 7) and high-level tissue
    architecture (final block) from DINOv2 ViT-S/14 with register-token separation.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_DINOV2_BACKBONE,
        pretrained: bool = True,
        d_model: int = 256,
        drop_path_rate: float = 0.15,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.backbone = _build_dinov2_vit(
            backbone_name=backbone_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )
        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        num_blocks = len(self.backbone.blocks)
        self.mid_block_idx = max(0, (num_blocks * 2) // 3 - 1)  # Block 7 of 12

        self.mid_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.embed_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cls_proj = nn.Sequential(
            nn.Linear(self.embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, canon_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = (canon_rgb.float() - self.mean) / self.std
        if x.shape[-2:] != (DINO_INPUT_SIZE, DINO_INPUT_SIZE):
            x = F.interpolate(
                x,
                size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        x = x.to(canon_rgb.dtype)

        # Stateless transient hook avoids persistent graph state during EMA deepcopy
        captured: Dict[str, torch.Tensor] = {}
        hook = self.backbone.blocks[self.mid_block_idx].register_forward_hook(
            lambda _m, _inp, out: captured.setdefault("mid", out)
        )
        try:
            tokens_final = self.backbone.forward_features(x)
        finally:
            hook.remove()

        tokens_mid = self.mid_norm(captured["mid"])

        cls_tok = self.cls_proj(tokens_final[:, 0, :])
        spatial_final = tokens_final[:, self.num_prefix_tokens :, :]
        spatial_mid = tokens_mid[:, self.num_prefix_tokens :, :]

        spatial_fused = self.spatial_proj(
            torch.cat([spatial_mid, spatial_final], dim=-1)
        )  # (B, 324, d_model)
        return spatial_fused, cls_tok


class ConcentricZoneGatedMIL(nn.Module):
    """
    Fuses 18x18 DINOv2 tokens with co-located Stain-Physics Pyramid tokens and reads
    out focal micro-metastases via:
    1. Ilse Gated-Attention MIL over all 324 tokens
    2. Top-k (k=24, ~7.5% patch area) focal instance pooling to protect positive recall
    3. Concentric Core [6:12, 6:12] vs. Outer Ring differential pooling to cancel
       residual additive slide-level domain shifts.
    """

    def __init__(
        self,
        d_model: int = 256,
        top_k_tokens: int = 24,
        attn_dropout: float = 0.15,
        proj_dropout: float = 0.25,
    ):
        super().__init__()
        self.top_k = int(top_k_tokens)
        self.attn_dropout = float(attn_dropout)

        zone_map = torch.full((DINO_GRID_SIZE, DINO_GRID_SIZE), 2, dtype=torch.long)
        zone_map[3:15, 3:15] = 1  # Mid cellular context ring
        zone_map[6:12, 6:12] = 0  # Central diagnostic core (~84-96 px)
        zone_flat = zone_map.view(-1)

        self.register_buffer("zone_ids", zone_flat, persistent=False)
        self.register_buffer(
            "core_mask", (zone_flat == 0).float().view(1, -1, 1), persistent=False
        )
        self.register_buffer(
            "outer_mask", (zone_flat == 2).float().view(1, -1, 1), persistent=False
        )
        self.zone_embed = nn.Embedding(3, d_model)

        self.phys_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.token_norm = nn.LayerNorm(d_model)

        # Ilse Gated Attention MIL
        self.mil_v = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh())
        self.mil_u = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Sigmoid())
        self.mil_w = nn.Linear(d_model // 2, 1)

        self.out_proj = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )

    def forward(
        self,
        dino_tokens: torch.Tensor,
        phys_tokens: torch.Tensor,
        cls_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gate = self.phys_gate(torch.cat([dino_tokens, phys_tokens], dim=-1))
        z_emb = self.zone_embed(self.zone_ids).unsqueeze(0)
        tokens = self.token_norm(dino_tokens + gate * phys_tokens + z_emb)

        # Gated MIL scores in float32 for stable softmax
        mil_hidden = self.mil_v(tokens) * self.mil_u(tokens)
        mil_scores = self.mil_w(mil_hidden).squeeze(-1).float()  # (B, 324)

        if self.training and self.attn_dropout > 0.0:
            drop_mask = torch.empty_like(mil_scores).bernoulli_(1.0 - self.attn_dropout)
            mil_scores_reg = mil_scores.masked_fill(drop_mask == 0.0, -1e4)
        else:
            mil_scores_reg = mil_scores

        attn_weights = F.softmax(mil_scores_reg, dim=-1).to(tokens.dtype)
        v_mil = torch.bmm(attn_weights.unsqueeze(1), tokens).squeeze(1)

        # Soft Top-k focal pooling for small micro-metastatic clusters
        k = min(self.top_k, tokens.shape[1])
        topk_vals, topk_idx = torch.topk(mil_scores, k=k, dim=-1)
        topk_w = F.softmax(topk_vals, dim=-1).to(tokens.dtype)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        topk_tokens = torch.gather(tokens, dim=1, index=gather_idx)
        v_topk = (topk_tokens * topk_w.unsqueeze(-1)).sum(dim=1)

        # Concentric Core & Slide-Offset-Canceling Core-Minus-Outer Differential
        core_sum = (tokens * self.core_mask).sum(dim=1)
        v_core = core_sum / self.core_mask.sum().clamp_min(1.0)
        outer_sum = (tokens * self.outer_mask).sum(dim=1)
        v_outer = outer_sum / self.outer_mask.sum().clamp_min(1.0)
        v_contrast = v_core - v_outer

        z_vis = self.out_proj(
            torch.cat([cls_token, v_mil, v_core, v_contrast, v_topk], dim=-1)
        )
        return z_vis, tokens


# ============================================================================
# 7-Family Periodic-Linear Set-Transformer & Bidirectional Cross-Modal Fusion
# ============================================================================
class PeriodicLinearFamilyEncoder(nn.Module):
    """
    Periodic-Linear-ReLU/GELU numerical embedding (Gorishniy et al., NeurIPS 2022)
    mapping continuous topological, GLCM, and cytomorphometric scalars into rich
    threshold-sensitive domain tokens.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int,
        n_freq_bins: int = 8,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(in_features, n_freq_bins) * 0.35)
        expanded_dim = in_features * (1 + 2 * n_freq_bins)
        self.proj = nn.Sequential(
            nn.Linear(expanded_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * x.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        periodic = torch.cat(
            [
                x.float(),
                torch.cos(angles).flatten(1),
                torch.sin(angles).flatten(1),
            ],
            dim=-1,
        ).to(x.dtype)
        return self.proj(periodic)


class FamilyPeriodicSetTransformer(nn.Module):
    """
    Encodes Step 1's 148 domain features into 7 semantic family tokens, applies
    intra-tabular self-attention, and performs bidirectional cross-attention with
    the 18x18 spatial visual-physics tokens.
    """

    def __init__(
        self,
        tabular_dim: int = 148,
        family_indices: Optional[Dict[str, List[int]]] = None,
        d_model: int = 256,
        num_heads: int = 8,
        dropout: float = 0.20,
    ):
        super().__init__()
        if family_indices is None:
            family_indices = DEFAULT_FAMILY_INDICES

        total_mapped = sum(len(v) for v in family_indices.values())
        if total_mapped != tabular_dim:
            chunk_splits = np.array_split(np.arange(tabular_dim), 7)
            family_indices = {
                f"family_{i}": idx.tolist() for i, idx in enumerate(chunk_splits)
            }

        self.family_names = list(family_indices.keys())
        self.num_families = len(self.family_names)

        self.family_encoders = nn.ModuleList()
        for idx_, name in enumerate(self.family_names):
            cols = family_indices[name]
            self.register_buffer(
                f"fam_cols_{idx_}",
                torch.tensor(cols, dtype=torch.long),
                persistent=False,
            )
            self.family_encoders.append(
                PeriodicLinearFamilyEncoder(
                    in_features=len(cols), d_model=d_model, dropout=dropout
                )
            )

        self.family_pos = nn.Parameter(
            torch.randn(1, self.num_families, d_model) * 0.02
        )
        self.self_attn_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Cross-attention: 7 pathology family tokens query 324 visual spatial tokens
        self.tab_q_norm = nn.LayerNorm(d_model)
        self.vis_kv_norm = nn.LayerNorm(d_model)
        self.tab_to_vis_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.tab_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Reverse cross-attention: visual summary queries enriched 7 family tokens
        self.vis_q_norm = nn.LayerNorm(d_model)
        self.tab_kv_norm = nn.LayerNorm(d_model)
        self.vis_to_tab_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.family_pool_score = nn.Linear(d_model, 1)
        self.tab_out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tabular_neural: torch.Tensor,
        spatial_tokens: torch.Tensor,
        z_vis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fam_tokens = []
        for i, encoder in enumerate(self.family_encoders):
            cols = getattr(self, f"fam_cols_{i}")
            x_fam = torch.index_select(tabular_neural, dim=-1, index=cols)
            fam_tokens.append(encoder(x_fam))

        fam_stack = torch.stack(fam_tokens, dim=1)
        fam_seq = fam_stack + self.family_pos.to(fam_stack.dtype)  # (B, 7, d_model)
        fam_seq = self.self_attn_block(fam_seq)

        # Ground tabular domain tokens in spatial visual-physics patches (AMP-safe dtype match)
        q_tv = self.tab_q_norm(fam_seq)
        kv_tv = self.vis_kv_norm(spatial_tokens).to(q_tv.dtype)
        tab_cross, _ = self.tab_to_vis_attn(
            query=q_tv,
            key=kv_tv,
            value=kv_tv,
            need_weights=False,
        )
        fam_enriched = fam_seq + tab_cross
        fam_enriched = fam_enriched + self.tab_ffn(fam_enriched)

        # Enrich visual summary with pathology family tokens (AMP-safe dtype match)
        q_vt = self.vis_q_norm(z_vis.unsqueeze(1))
        kv_vt = self.tab_kv_norm(fam_enriched).to(q_vt.dtype)
        vis_cross, _ = self.vis_to_tab_attn(
            query=q_vt,
            key=kv_vt,
            value=kv_vt,
            need_weights=False,
        )
        z_vis_refined = z_vis + vis_cross.squeeze(1)

        pool_w = F.softmax(
            self.family_pool_score(fam_enriched).squeeze(-1).float(), dim=-1
        ).to(fam_enriched.dtype)
        z_tab = self.tab_out_norm((fam_enriched * pool_w.unsqueeze(-1)).sum(dim=1))
        return z_tab, z_vis_refined


# ============================================================================
# Zero-Bias Bipolar von Mises-Fisher (vMF) Hyperspherical Classification Head
# ============================================================================
class BipolarVMFHypersphericalHead(nn.Module):
    """
    Projects embeddings onto the unit hypersphere S^{D-1} and computes the exact
    symmetric log-likelihood ratio between unit-norm Tumor and Normal prototypes:
        logit(x) = scale * 0.5 * (<z_hat, u_tumor> - <z_hat, u_normal>)
    With zero unconstrained scalar bias and L2-normalized features, multiplicative
    feature norm shifts on unseen medical centers cancel identically, structurally
    anchoring the Bayes-optimal Balanced Accuracy boundary at logit = 0 (p = 0.5).
    """

    def __init__(
        self,
        in_dim: int = 256,
        sphere_dim: int = 128,
        init_scale: float = 12.0,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, sphere_dim),
            nn.LayerNorm(sphere_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sphere_dim, sphere_dim),
        )
        # Initialize antipodal prototypes u_pos = -u_neg so step-0 probabilities sit at 0.5
        init_dir = F.normalize(torch.randn(sphere_dim), dim=0)
        self.proto_pos = nn.Parameter(init_dir.clone())
        self.proto_neg = nn.Parameter(-init_dir.clone())
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(init_scale), dtype=torch.float32)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_raw = self.projector(x).float()
        z_unit = F.normalize(z_raw, p=2, dim=-1, eps=1e-6)
        u_pos = F.normalize(self.proto_pos.float(), p=2, dim=-1, eps=1e-6)
        u_neg = F.normalize(self.proto_neg.float(), p=2, dim=-1, eps=1e-6)

        scale = self.log_scale.float().exp().clamp(min=4.0, max=30.0)
        cos_pos = F.linear(z_unit, u_pos.unsqueeze(0)).squeeze(-1)
        cos_neg = F.linear(z_unit, u_neg.unsqueeze(0)).squeeze(-1)
        logits = scale * 0.5 * (cos_pos - cos_neg)
        proto_cos = torch.dot(u_pos, u_neg)
        return logits, z_unit, proto_cos


# ============================================================================
# Unified Architecture: TopoStainDINOv2PyramidNet
# ============================================================================
class TopoStainDINOv2PyramidNet(nn.Module):
    """
    End-to-end 100% BatchNorm-free Trimodal Histopathology Classifier:
      1. DINOv2 ViT-S/14 (with registers) multi-depth semantic/cytological tokens
         + Concentric Core-vs-Outer Gated-MIL & Top-k focal aggregation
      2. Native 256x256 Stain-Physics & DoG Differential Pyramid ([RGB, C_H, C_E, |C_R|])
      3. 7-Family Periodic-Linear Set-Transformer with bidirectional spatial cross-attention
      4. Bipolar vMF Hyperspherical Head self-anchored at threshold p = 0.5
    """

    def __init__(
        self,
        tabular_dim: int = 148,
        family_indices: Optional[Dict[str, List[int]]] = None,
        backbone_name: str = DEFAULT_DINOV2_BACKBONE,
        pretrained: bool = True,
        d_model: int = 256,
        sphere_dim: int = 128,
        drop_path_rate: float = 0.15,
        dropout: float = 0.20,
        modality_drop_prob: float = 0.10,
    ):
        super().__init__()
        self.modality_drop_prob = float(modality_drop_prob)

        self.vision_stream = DINOv2MultiDepthBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            d_model=d_model,
            drop_path_rate=drop_path_rate,
            dropout=dropout * 0.75,
        )
        self.physics_pyramid = StainPhysicsDifferentialPyramid(
            d_model=d_model,
            drop_path_rate=drop_path_rate * 0.75,
            dropout=dropout,
        )
        self.concentric_mil = ConcentricZoneGatedMIL(
            d_model=d_model,
            top_k_tokens=24,
            attn_dropout=0.15,
            proj_dropout=dropout,
        )
        self.tabular_transformer = FamilyPeriodicSetTransformer(
            tabular_dim=tabular_dim,
            family_indices=family_indices,
            d_model=d_model,
            num_heads=8,
            dropout=dropout,
        )

        # Trimodal adaptive gating network
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Main and auxiliary hyperspherical / zero-bias heads
        self.main_head = BipolarVMFHypersphericalHead(
            in_dim=d_model, sphere_dim=sphere_dim, init_scale=12.0, dropout=dropout
        )
        self.aux_vis_head = BipolarVMFHypersphericalHead(
            in_dim=d_model, sphere_dim=sphere_dim // 2, init_scale=10.0, dropout=dropout
        )
        self.aux_phys_head = BipolarVMFHypersphericalHead(
            in_dim=d_model, sphere_dim=sphere_dim // 2, init_scale=10.0, dropout=dropout
        )
        self.aux_tab_head = BipolarVMFHypersphericalHead(
            in_dim=d_model, sphere_dim=sphere_dim // 2, init_scale=10.0, dropout=dropout
        )

    def _apply_modality_dropout(
        self, z_vis: torch.Tensor, z_phys: torch.Tensor, z_tab: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_drop_prob <= 0.0:
            return z_vis, z_phys, z_tab

        bsz = z_vis.shape[0]
        rnd = torch.rand(bsz, 1, device=z_vis.device)
        keep_phys = (rnd >= (0.5 * self.modality_drop_prob)).to(z_phys.dtype)
        keep_tab = (
            (rnd < (0.5 * self.modality_drop_prob)) | (rnd >= self.modality_drop_prob)
        ).to(z_tab.dtype)
        return z_vis, z_phys * keep_phys, z_tab * keep_tab

    def forward(
        self,
        canon_rgb: torch.Tensor,
        stain_channels: torch.Tensor,
        tabular_neural: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dino_tokens, cls_tok = self.vision_stream(canon_rgb)
        phys_tokens, z_phys = self.physics_pyramid(canon_rgb, stain_channels)
        z_vis_raw, fused_spatial_tokens = self.concentric_mil(
            dino_tokens, phys_tokens, cls_tok
        )
        z_tab, z_vis = self.tabular_transformer(
            tabular_neural, fused_spatial_tokens, z_vis_raw
        )

        z_vis_m, z_phys_m, z_tab_m = self._apply_modality_dropout(z_vis, z_phys, z_tab)
        concat_all = torch.cat([z_vis_m, z_phys_m, z_tab_m], dim=-1)

        mod_weights = F.softmax(self.fusion_gate(concat_all).float(), dim=-1).to(
            z_vis.dtype
        )
        z_weighted = (
            mod_weights[:, 0:1] * z_vis_m
            + mod_weights[:, 1:2] * z_phys_m
            + mod_weights[:, 2:3] * z_tab_m
        )
        z_fused = z_weighted + self.fusion_mlp(concat_all)

        logits, unit_features, proto_cos = self.main_head(z_fused)
        aux_vis_logits, _, _ = self.aux_vis_head(z_vis)
        aux_phys_logits, _, _ = self.aux_phys_head(z_phys)
        aux_tab_logits, _, _ = self.aux_tab_head(z_tab)

        probabilities = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "unit_features": unit_features,
            "proto_cos": proto_cos,
            "aux_vis_logits": aux_vis_logits,
            "aux_phys_logits": aux_phys_logits,
            "aux_tab_logits": aux_tab_logits,
        }

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.forward(
            canon_rgb=batch["canon_rgb"],
            stain_channels=batch["stain_channels"],
            tabular_neural=batch["tabular_neural"],
        )


# ============================================================================
# Task-Aligned Loss: CenterInvariantBalancedAccuracyLoss
# ============================================================================
class CenterInvariantBalancedAccuracyLoss(nn.Module):
    """
    Directly aligns training optimization with unseen-center Balanced Accuracy
    at the fixed p = 0.5 (logit = 0.0) decision threshold:
      1. Class-Balanced Symmetric BCE (0.5 * L_pos + 0.5 * L_neg) with symmetric
         label smoothing (preserving the Bayes-optimal fixed point at p = 0.5)
      2. Smooth Surrogate Balanced-Error Loss around logit = 0:
         1 - 0.5 * (TPR_soft + TNR_soft)
      3. Multi-Center Variance-Risk Extrapolation (V-REx) equalizing per-center
         balanced risk + Class-Conditional Hyperspherical Feature Alignment across
         training centers (uses center_idx ONLY during training; safely 0 if <2 centers)
      4. Auxiliary trimodal supervision + antipodal prototype orthogonality
    """

    def __init__(
        self,
        label_smoothing: float = 0.015,
        w_soft_bal: float = 0.35,
        soft_sharpness: float = 4.0,
        w_vrex: float = 0.20,
        w_cc_align: float = 0.15,
        w_aux: float = 0.25,
        w_proto: float = 0.05,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.w_soft_bal = float(w_soft_bal)
        self.soft_sharpness = float(soft_sharpness)
        self.w_vrex = float(w_vrex)
        self.w_cc_align = float(w_cc_align)
        self.w_aux = float(w_aux)
        self.w_proto = float(w_proto)

    def _class_balanced_bce(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits_f = logits.float().view(-1)
        y_raw = labels.float().view(-1)
        y_smooth = y_raw * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        per_sample = F.binary_cross_entropy_with_logits(
            logits_f, y_smooth, reduction="none"
        )
        pos_mask = y_raw > 0.5
        neg_mask = ~pos_mask

        if pos_mask.any() and neg_mask.any():
            return 0.5 * per_sample[pos_mask].mean() + 0.5 * per_sample[neg_mask].mean()
        return per_sample.mean()

    def _soft_balanced_error_at_half(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits_f = logits.float().view(-1)
        y_raw = labels.float().view(-1)
        pos_mask = y_raw > 0.5
        neg_mask = ~pos_mask

        if pos_mask.any() and neg_mask.any():
            tpr_soft = torch.sigmoid(self.soft_sharpness * logits_f[pos_mask]).mean()
            tnr_soft = torch.sigmoid(-self.soft_sharpness * logits_f[neg_mask]).mean()
            return 1.0 - 0.5 * (tpr_soft + tnr_soft)
        signed = torch.where(pos_mask, logits_f, -logits_f)
        return 1.0 - torch.sigmoid(self.soft_sharpness * signed).mean()

    def _center_invariance_penalties(
        self,
        logits: torch.Tensor,
        unit_features: torch.Tensor,
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = logits.new_zeros((), dtype=torch.float32)
        if center_idx is None:
            return zero, zero

        c_idx = center_idx.view(-1)
        valid = c_idx >= 0
        if int(valid.sum()) < 8:
            return zero, zero

        logits_v = logits.float().view(-1)[valid]
        feats_v = unit_features.float()[valid]
        y_v = labels.float().view(-1)[valid]
        c_v = c_idx[valid]

        unique_centers = torch.unique(c_v)
        if unique_centers.numel() < 2:
            return zero, zero

        # 1) V-REx: penalize variance of balanced classification risk across centers
        center_risks = []
        for c in unique_centers:
            m_c = c_v == c
            if int(m_c.sum()) >= 4:
                center_risks.append(self._class_balanced_bce(logits_v[m_c], y_v[m_c]))
        if len(center_risks) >= 2:
            risk_stack = torch.stack(center_risks)
            loss_vrex = torch.var(risk_stack, unbiased=False)
        else:
            loss_vrex = zero

        # 2) Class-Conditional Hyperspherical Centroid & Covariance Alignment
        align_terms = []
        for cls_val in (0.0, 1.0):
            m_cls = (y_v > 0.5) if cls_val > 0.5 else (y_v <= 0.5)
            means_cls = []
            covs_cls = []
            for c in unique_centers:
                m_cc = m_cls & (c_v == c)
                n_cc = int(m_cc.sum())
                if n_cc >= 3:
                    f_cc = feats_v[m_cc]
                    mu_cc = f_cc.mean(dim=0)
                    centered = f_cc - mu_cc.unsqueeze(0)
                    cov_cc = (centered.T @ centered) / float(n_cc - 1)
                    means_cls.append(mu_cc)
                    covs_cls.append(cov_cc)
            if len(means_cls) >= 2:
                for i in range(len(means_cls)):
                    for j in range(i + 1, len(means_cls)):
                        mean_diff = (means_cls[i] - means_cls[j]).pow(2).sum()
                        cov_diff = (covs_cls[i] - covs_cls[j]).pow(2).mean() * float(
                            feats_v.shape[-1]
                        )
                        align_terms.append(mean_diff + 0.5 * cov_diff)

        loss_cc_align = torch.stack(align_terms).mean() if align_terms else zero
        return loss_vrex, loss_cc_align

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = outputs["logits"].float()
        y = labels.float()

        loss_main_bce = self._class_balanced_bce(logits, y)
        loss_soft_bal = self._soft_balanced_error_at_half(logits, y)

        loss_aux = (
            self._class_balanced_bce(outputs["aux_vis_logits"], y)
            + self._class_balanced_bce(outputs["aux_phys_logits"], y)
            + self._class_balanced_bce(outputs["aux_tab_logits"], y)
        ) / 3.0

        proto_cos = outputs["proto_cos"].float()
        loss_proto = F.relu(proto_cos + 0.5).pow(2)

        loss_vrex, loss_cc_align = self._center_invariance_penalties(
            logits=logits,
            unit_features=outputs["unit_features"],
            labels=y,
            center_idx=center_idx,
        )

        total_loss = (
            loss_main_bce
            + self.w_soft_bal * loss_soft_bal
            + self.w_aux * loss_aux
            + self.w_proto * loss_proto
            + self.w_vrex * loss_vrex
            + self.w_cc_align * loss_cc_align
        )

        metrics_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_bce": float(loss_main_bce.detach().item()),
            "loss_soft_bal": float(loss_soft_bal.detach().item()),
            "loss_vrex": float(loss_vrex.detach().item()),
            "loss_cc_align": float(loss_cc_align.detach().item()),
        }
        return total_loss, metrics_dict


# ============================================================================
# Weight-Space Exponential Moving Average (EMA) & Differential LLRD Optimizer
# ============================================================================
class ModelEMA:
    """
    Float32 weight-space Exponential Moving Average wrapper.
    Because TopoStainDINOv2PyramidNet is 100% LayerNorm-based (zero BatchNorm),
    EMA weights never suffer from stale batch-norm running buffer mismatch.
    """

    def __init__(self, model: nn.Module, decay: float = 0.997):
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, decay_override: Optional[float] = None) -> None:
        d = self.decay if decay_override is None else float(decay_override)
        ema_params = dict(self.ema_model.named_parameters())
        for name, param in model.named_parameters():
            ema_p = ema_params[name]
            ema_p.data.mul_(d).add_(param.detach().data, alpha=1.0 - d)


def _is_no_weight_decay_param(name: str, param: nn.Parameter) -> bool:
    if param.ndim <= 1:
        return True
    no_wd_tokens = (
        "bias",
        "norm",
        "pos_embed",
        "cls_token",
        "reg_token",
        "gamma",
        "ls1",
        "ls2",
        "freqs",
        "family_pos",
        "zone_embed",
        "proto_pos",
        "proto_neg",
        "log_scale",
    )
    return any(tok in name for tok in no_wd_tokens)


def build_differential_optimizer_and_scheduler(
    model: TopoStainDINOv2PyramidNet,
    total_steps: int,
    backbone_lr: float = 7.5e-5,
    head_lr: float = 3.0e-4,
    llrd_decay: float = 0.84,
    weight_decay: float = 0.03,
    warmup_fraction: float = 0.10,
    min_lr_ratio: float = 0.02,
) -> Tuple[AdamW, LambdaLR]:
    """
    Constructs AdamW with Layer-wise Learning Rate Decay (LLRD) across DINOv2
    transformer blocks and higher learning rate on newly initialized stain-physics,
    tabular cross-attention, and vMF head modules, paired with linear-warmup cosine decay.
    """
    num_dino_blocks = len(model.vision_stream.backbone.blocks)
    grouped: Dict[Tuple[float, float], List[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("vision_stream.backbone."):
            sub = name[len("vision_stream.backbone.") :]
            if sub.startswith("blocks."):
                block_idx = int(sub.split(".")[1])
                exponent = num_dino_blocks - 1 - block_idx
                lr = backbone_lr * (llrd_decay**exponent)
            elif sub.startswith(("patch_embed", "cls_token", "reg_token", "pos_embed")):
                lr = backbone_lr * (llrd_decay**num_dino_blocks)
            else:
                lr = backbone_lr
        else:
            lr = head_lr

        wd = 0.0 if _is_no_weight_decay_param(name, param) else weight_decay
        key = (round(lr, 10), wd)
        grouped.setdefault(key, []).append(param)

    param_groups = [
        {"params": params, "lr": lr_val, "weight_decay": wd_val}
        for (lr_val, wd_val), params in grouped.items()
    ]
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def build_pathology_gbdt_models(
    seed: int = 42, early_stopping_rounds: int = 75
) -> Dict[str, object]:
    """
    Factory functions creating domain-regularized LightGBM and XGBoost classifiers
    for Step 1's 148 physical/topological/cytomorphometric tree features (X_*_tree),
    audited via Leave-One-Center-Out validation before multimodal blending.
    """
    lgbm_model = LGBMClassifier(
        n_estimators=1400,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=6,
        min_child_samples=45,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=1.5,
        reg_lambda=5.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )

    xgb_model = XGBClassifier(
        n_estimators=1400,
        learning_rate=0.025,
        max_depth=5,
        min_child_weight=8.0,
        subsample=0.80,
        colsample_bytree=0.55,
        reg_alpha=1.5,
        reg_lambda=6.0,
        gamma=0.20,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
    )

    return {"lightgbm": lgbm_model, "xgboost": xgb_model}


def create_model_criterion_optimizer(
    working_dir: Path = WORKING_DIR,
    total_steps: int = 412 * 12,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Instantiates the complete model design suite (TopoStainDINOv2PyramidNet,
    ModelEMA, CenterInvariantBalancedAccuracyLoss, LLRD AdamW + Cosine Scheduler,
    and GBDT builders) aligned with Step 1's saved feature metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tabular_dim = 148
    family_indices = DEFAULT_FAMILY_INDICES
    meta_path = Path(working_dir) / "feature_metadata.pkl"
    if meta_path.is_file():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        tabular_dim = len(meta["feature_names"])
        family_indices = meta["family_indices"]

    model = TopoStainDINOv2PyramidNet(
        tabular_dim=tabular_dim,
        family_indices=family_indices,
        backbone_name=DEFAULT_DINOV2_BACKBONE,
        pretrained=pretrained,
        d_model=256,
        sphere_dim=128,
        drop_path_rate=0.15,
        dropout=0.20,
        modality_drop_prob=0.10,
    ).to(device)

    # Strict architectural verification: zero BatchNorm layers anywhere
    has_bn = any(
        isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()
    )
    assert (
        not has_bn
    ), "BatchNorm detected; architecture must remain 100% BatchNorm-free!"

    ema = ModelEMA(model, decay=0.997)
    criterion = CenterInvariantBalancedAccuracyLoss().to(device)
    optimizer, scheduler = build_differential_optimizer_and_scheduler(
        model=model,
        total_steps=total_steps,
    )
    gbdt_models = build_pathology_gbdt_models(seed=seed)

    return {
        "model": model,
        "ema": ema,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "gbdt_models": gbdt_models,
    }


# ============================================================================
# Step 3: Official Metric Loader, Samplers, Training & Unified Inference
# ============================================================================
def load_official_metric_fn(input_dir: Path = INPUT_DIR):
    metric_path = Path(input_dir) / "metric.py"
    spec = importlib.util.spec_from_file_location("official_metric", str(metric_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.binary_metrics


official_binary_metrics = load_official_metric_fn(INPUT_DIR)


def compute_detailed_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    official = official_binary_metrics(y_true, y_prob)
    preds = (y_prob >= 0.5).astype(np.int64)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = float(np.mean(preds[pos_mask] == 1)) if np.any(pos_mask) else 0.0
    tnr = float(np.mean(preds[neg_mask] == 0)) if np.any(neg_mask) else 0.0
    official["tpr"] = tpr
    official["tnr"] = tnr
    return official


def load_npz_dict(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def safe_torch_load_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


class CenterClassSlideBalancedBatchSampler(Sampler[List[int]]):
    """
    Constructs mini-batches with equal representation across every (center_idx, label)
    stratum and round-robin slide interleaving within each stratum.
    Guarantees 50/50 class balance within every center per batch (preventing center
    prevalence shortcuts) and >= 12-16 samples per (center, class) cell for stable
    V-REx and class-conditional hyperspherical covariance alignment.
    """

    def __init__(
        self,
        meta_df: pd.DataFrame,
        batch_size: int = TRAIN_BATCH_SIZE,
        seed: int = SEED,
    ):
        super().__init__()
        self.batch_size = int(batch_size)
        self.base_seed = int(seed)
        self.epoch = 0

        centers = meta_df["center_idx"].values.astype(int)
        labels = meta_df["label"].values.astype(int)
        slides = meta_df["slide_id"].astype(str).values

        strata_slide_map: Dict[Tuple[int, int], Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row_idx, (c_id, lbl, s_id) in enumerate(zip(centers, labels, slides)):
            strata_slide_map[(c_id, lbl)][s_id].append(row_idx)

        self.strata_keys = sorted(strata_slide_map.keys())
        self.strata_slides: List[List[np.ndarray]] = []
        for key in self.strata_keys:
            slide_dict = strata_slide_map[key]
            slide_arrays = [
                np.asarray(idxs, dtype=np.int64)
                for _, idxs in sorted(slide_dict.items())
            ]
            self.strata_slides.append(slide_arrays)

        n_strata = len(self.strata_keys)
        base_quota = self.batch_size // n_strata
        rem = self.batch_size % n_strata
        self.quotas = [base_quota + (1 if i < rem else 0) for i in range(n_strata)]
        self.num_batches = max(1, len(meta_df) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_slide_interleaved_stream(
        self, slide_arrays: List[np.ndarray], needed: int, rng: np.random.Generator
    ) -> np.ndarray:
        chunks: List[int] = []
        while len(chunks) < needed:
            slide_order = rng.permutation(len(slide_arrays))
            permuted_slides = [
                rng.permutation(slide_arrays[s_i]).tolist() for s_i in slide_order
            ]
            max_len = max(len(arr) for arr in permuted_slides)
            for pos in range(max_len):
                for arr in permuted_slides:
                    if pos < len(arr):
                        chunks.append(arr[pos])
        return np.asarray(chunks[:needed], dtype=np.int64)

    def __iter__(self):
        rng = np.random.default_rng(self.base_seed + self.epoch * 10007)
        streams = [
            self._build_slide_interleaved_stream(
                slide_arrays=s_list,
                needed=self.num_batches * quota,
                rng=rng,
            )
            for s_list, quota in zip(self.strata_slides, self.quotas)
        ]

        for b_idx in range(self.num_batches):
            batch_parts = [
                stream[b_idx * quota : (b_idx + 1) * quota]
                for stream, quota in zip(streams, self.quotas)
            ]
            batch = np.concatenate(batch_parts, axis=0)
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.num_batches


def _compute_center_class_gbdt_weights(
    y: np.ndarray, center_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes sample weights equalizing every (center, class) cell so center identity
    carries zero label prior bias and the Bayes-optimal decision boundary sits at 0.5.
    """
    y_int = y.astype(np.int64)
    c_int = center_idx.astype(np.int64)
    n_total = float(len(y_int))
    unique_c = np.unique(c_int)
    n_centers = float(max(1, len(unique_c)))

    class_counts = {
        0: float(max(1, np.sum(y_int == 0))),
        1: float(max(1, np.sum(y_int == 1))),
    }
    w_xgb = np.ones(len(y_int), dtype=np.float32)
    w_lgb = np.ones(len(y_int), dtype=np.float32)

    for c in unique_c:
        for cls_val in (0, 1):
            mask = (c_int == c) & (y_int == cls_val)
            n_cell = float(np.sum(mask))
            if n_cell > 0:
                w_xgb[mask] = n_total / (2.0 * n_centers * n_cell)
                w_lgb[mask] = class_counts[cls_val] / (n_centers * n_cell)

    w_xgb /= float(np.mean(w_xgb))
    w_lgb /= float(np.mean(w_lgb))
    return w_lgb, w_xgb


def fit_gbdt_pair_on_split(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    c_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int = SEED,
) -> Dict[str, object]:
    models = build_pathology_gbdt_models(seed=seed, early_stopping_rounds=75)
    w_lgb, w_xgb = _compute_center_class_gbdt_weights(y_tr, c_tr)

    lgbm_clf = models["lightgbm"]
    lgbm_clf.fit(
        X_tr,
        y_tr,
        sample_weight=w_lgb,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    xgb_clf = models["xgboost"]
    xgb_clf.fit(
        X_tr,
        y_tr,
        sample_weight=w_xgb,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    return {"lightgbm": lgbm_clf, "xgboost": xgb_clf}


def predict_gbdt_pair_logits(pair: Dict[str, object], X: np.ndarray) -> np.ndarray:
    p_lgb = np.clip(
        pair["lightgbm"].predict_proba(X)[:, 1].astype(np.float64), 1e-5, 1.0 - 1e-5
    )
    p_xgb = np.clip(
        pair["xgboost"].predict_proba(X)[:, 1].astype(np.float64), 1e-5, 1.0 - 1e-5
    )
    return 0.5 * (logit(p_lgb) + logit(p_xgb))


def run_gbdt_loco_and_kfold(
    meta_df: pd.DataFrame,
    X_tree: np.ndarray,
    fold_col: str,
    seed: int = SEED,
    run_loco: bool = True,
) -> Tuple[List[Dict[str, object]], np.ndarray, Optional[np.ndarray]]:
    y_all = meta_df["label"].values.astype(np.int64)
    c_all = meta_df["center_idx"].values.astype(np.int64)
    folds = meta_df[fold_col].values.astype(np.int64)

    oof_loco_logits = None
    if run_loco and "loco_fold" in meta_df.columns:
        loco_vals = meta_df["loco_fold"].values.astype(np.int64)
        unique_loco = sorted(np.unique(loco_vals).tolist())
        if len(unique_loco) >= 2:
            oof_loco_logits = np.zeros(len(meta_df), dtype=np.float64)
            for l_idx, held_out_c in enumerate(unique_loco):
                va_loco_mask = loco_vals == held_out_c
                tr_pool_idx = np.where(~va_loco_mask)[0]
                es_mask_local = folds[tr_pool_idx] == 0
                if np.sum(es_mask_local) < 32 or np.sum(~es_mask_local) < 64:
                    cut = max(32, len(tr_pool_idx) // 5)
                    es_idx = tr_pool_idx[:cut]
                    fit_idx = tr_pool_idx[cut:]
                else:
                    es_idx = tr_pool_idx[es_mask_local]
                    fit_idx = tr_pool_idx[~es_mask_local]

                loco_pair = fit_gbdt_pair_on_split(
                    X_tr=X_tree[fit_idx],
                    y_tr=y_all[fit_idx],
                    c_tr=c_all[fit_idx],
                    X_va=X_tree[es_idx],
                    y_va=y_all[es_idx],
                    seed=seed + 300 + l_idx,
                )
                oof_loco_logits[va_loco_mask] = predict_gbdt_pair_logits(
                    loco_pair, X_tree[va_loco_mask]
                )

    oof_kfold_logits = np.zeros(len(meta_df), dtype=np.float64)
    fold_models: List[Dict[str, object]] = []
    for f_id in sorted(np.unique(folds).tolist()):
        va_mask = folds == f_id
        tr_mask = ~va_mask
        assert set(meta_df.loc[tr_mask, "group_id"]).isdisjoint(
            set(meta_df.loc[va_mask, "group_id"])
        ), f"Group leakage in GBDT fold {f_id}!"
        pair = fit_gbdt_pair_on_split(
            X_tr=X_tree[tr_mask],
            y_tr=y_all[tr_mask],
            c_tr=c_all[tr_mask],
            X_va=X_tree[va_mask],
            y_va=y_all[va_mask],
            seed=seed + f_id * 19,
        )
        oof_kfold_logits[va_mask] = predict_gbdt_pair_logits(pair, X_tree[va_mask])
        fold_models.append(pair)

    return fold_models, oof_kfold_logits, oof_loco_logits


@torch.no_grad()
def predict_single_model_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            canon_rgb = batch["canon_rgb"].to(device, non_blocking=True)
            stain_ch = batch["stain_channels"].to(device, non_blocking=True)
            tab_neu = batch["tabular_neural"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                out = model(canon_rgb, stain_ch, tab_neu)
            chunks.append(out["logits"].detach().float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def train_single_neural_fold(
    fold_idx: int,
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    X_tr_neural: np.ndarray,
    X_va_neural: np.ndarray,
    stain_prior: Dict[str, np.ndarray],
    ckpt_path: Path,
    device: torch.device,
    seed: int,
    stage_tag: str,
    pipeline_t0: float,
    stage_deadline_sec: float,
) -> Tuple[Path, np.ndarray, float]:
    assert set(tr_df["group_id"]).isdisjoint(
        set(va_df["group_id"])
    ), f"{stage_tag} fold {fold_idx}: group_id overlap!"
    assert set(tr_df["slide_id"]).isdisjoint(
        set(va_df["slide_id"])
    ), f"{stage_tag} fold {fold_idx}: slide_id overlap!"

    seed_everything(seed)
    n_workers = max(2, min(6, (os.cpu_count() or 4) // 2))

    train_ds = HistopathologyMultiModalDataset(
        meta_df=tr_df.reset_index(drop=True),
        X_neural=X_tr_neural,
        stain_prior=stain_prior,
        input_dir=INPUT_DIR,
        is_train=True,
        seed=seed,
    )
    val_ds = HistopathologyMultiModalDataset(
        meta_df=va_df.reset_index(drop=True),
        X_neural=X_va_neural,
        stain_prior=stain_prior,
        input_dir=INPUT_DIR,
        is_train=False,
        tta_view_idx=0,
        seed=seed,
    )

    batch_sampler = CenterClassSlideBalancedBatchSampler(
        meta_df=tr_df.reset_index(drop=True),
        batch_size=TRAIN_BATCH_SIZE,
        seed=seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=INFER_BATCH_SIZE,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
    )

    total_steps = len(train_loader) * NEURAL_EPOCHS
    suite = create_model_criterion_optimizer(
        working_dir=WORKING_DIR,
        total_steps=total_steps,
        pretrained=True,
        device=device,
        seed=seed,
    )
    model: TopoStainDINOv2PyramidNet = suite["model"]
    ema: ModelEMA = suite["ema"]
    criterion: CenterInvariantBalancedAccuracyLoss = suite["criterion"]
    optimizer = suite["optimizer"]
    scheduler = suite["scheduler"]

    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_fp16_scaler = use_cuda and (not use_bf16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)

    y_va = va_df["label"].values.astype(np.int64)
    best_score_tuple = (-1.0, -1.0)
    best_oof_logits = np.zeros(len(va_df), dtype=np.float64)
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, NEURAL_EPOCHS + 1):
        if (time.time() - pipeline_t0) > stage_deadline_sec and epoch > 1:
            break

        ep_t0 = time.time()
        batch_sampler.set_epoch(epoch)
        model.train()

        running_loss = 0.0
        running_vrex = 0.0
        n_batches = 0

        for batch in train_loader:
            canon_rgb = batch["canon_rgb"].to(device, non_blocking=True)
            stain_ch = batch["stain_channels"].to(device, non_blocking=True)
            tab_neu = batch["tabular_neural"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            center_idx = batch["center_idx"].to(device, non_blocking=True)

            if TABULAR_TRAIN_NOISE_STD > 0.0:
                tab_neu = tab_neu + torch.randn_like(tab_neu) * TABULAR_TRAIN_NOISE_STD

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_cuda
            ):
                outputs = model(canon_rgb, stain_ch, tab_neu)
                loss, loss_info = criterion(outputs, labels, center_idx=center_idx)

            if use_fp16_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            global_step += 1
            warmup_ema_decay = min(
                ema.decay, (1.0 + float(global_step)) / (10.0 + float(global_step))
            )
            ema.update(model, decay_override=warmup_ema_decay)

            running_loss += loss_info["loss_total"]
            running_vrex += loss_info["loss_vrex"]
            n_batches += 1

        raw_logits = predict_single_model_logits(
            model, val_loader, device, amp_dtype, use_cuda
        )
        ema_logits = predict_single_model_logits(
            ema.ema_model, val_loader, device, amp_dtype, use_cuda
        )
        raw_metrics = compute_detailed_metrics(y_va, expit(raw_logits))
        ema_metrics = compute_detailed_metrics(y_va, expit(ema_logits))

        ema_tuple = (
            round(ema_metrics["balanced_accuracy"], 6),
            round(ema_metrics["auroc"], 6),
        )
        raw_tuple = (
            round(raw_metrics["balanced_accuracy"], 6),
            round(raw_metrics["auroc"], 6),
        )
        if epoch >= 2 and ema_tuple >= raw_tuple:
            cand_tuple = ema_tuple
            cand_logits = ema_logits
            cand_model = ema.ema_model
            cand_src = "EMA"
            cand_metrics = ema_metrics
        else:
            cand_tuple = raw_tuple
            cand_logits = raw_logits
            cand_model = model
            cand_src = "RAW"
            cand_metrics = raw_metrics

        improved = cand_tuple > best_score_tuple
        if improved:
            best_score_tuple = cand_tuple
            best_oof_logits = cand_logits.copy()
            state_cpu = {
                k: v.detach().cpu() for k, v in cand_model.state_dict().items()
            }
            torch.save(state_cpu, ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Strictly 1 single summary line per epoch
        print(
            f"[{stage_tag} Fold {fold_idx} | Ep {epoch:02d}/{NEURAL_EPOCHS:02d}] "
            f"loss={running_loss / max(1, n_batches):.4f} (vrex={running_vrex / max(1, n_batches):.4f}) | "
            f"raw_bal={raw_metrics['balanced_accuracy']:.4f} ema_bal={ema_metrics['balanced_accuracy']:.4f} "
            f"auroc={cand_metrics['auroc']:.4f} ({cand_src}{'*' if improved else ''}) | "
            f"{time.time() - ep_t0:.1f}s"
        )

        if epoch >= EARLY_STOP_MIN_EPOCHS and epochs_no_improve >= EARLY_STOP_PATIENCE:
            break

    del (
        train_loader,
        val_loader,
        train_ds,
        val_ds,
        suite,
        model,
        ema,
        criterion,
        optimizer,
        scheduler,
    )
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    return ckpt_path, best_oof_logits, float(best_score_tuple[0])


def select_leakfree_gbdt_blend_weight(
    meta_df: pd.DataFrame,
    nn_oof_logits: np.ndarray,
    nn_oof_mask: np.ndarray,
    gbdt_loco_logits: Optional[np.ndarray],
    gbdt_kfold_logits: np.ndarray,
) -> float:
    """
    Selects log-odds blend weight w_gbdt strictly on train.csv out-of-fold predictions
    using unseen-center LOCO GBDT logits (never touching validation.csv).
    """
    ref_gbdt_logits = (
        gbdt_loco_logits if gbdt_loco_logits is not None else gbdt_kfold_logits
    )
    y_eval = meta_df["label"].values[nn_oof_mask].astype(np.int64)
    nn_eval = nn_oof_logits[nn_oof_mask]
    gbdt_eval = ref_gbdt_logits[nn_oof_mask]

    gbdt_only_bal = official_binary_metrics(y_eval, expit(gbdt_eval))[
        "balanced_accuracy"
    ]
    if gbdt_only_bal < 0.78:
        return 0.0

    best_w = 0.0
    base_metrics = official_binary_metrics(y_eval, expit(nn_eval))
    best_bal = base_metrics["balanced_accuracy"]
    best_auc = base_metrics["auroc"]

    for w_cand in (0.00, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15):
        blend_logits = (1.0 - w_cand) * nn_eval + w_cand * np.clip(gbdt_eval, -8.0, 8.0)
        m = official_binary_metrics(y_eval, expit(blend_logits))
        bal, auc = m["balanced_accuracy"], m["auroc"]
        if (bal > best_bal + 1e-6) or (
            abs(bal - best_bal) <= 1e-6 and auc > best_auc + 1e-5
        ):
            best_bal = bal
            best_auc = auc
            best_w = float(w_cand)

    return best_w


@torch.no_grad()
def run_unified_ensemble_inference(
    meta_df: pd.DataFrame,
    X_neural: np.ndarray,
    X_tree: np.ndarray,
    stain_prior: Dict[str, np.ndarray],
    neural_ckpt_paths: Sequence[Path],
    gbdt_fold_models: Sequence[Dict[str, object]],
    w_gbdt: float,
    tabular_dim: int,
    family_indices: Dict[str, List[int]],
    device: torch.device,
    tta_views: Sequence[int] = TTA_VIEWS,
) -> np.ndarray:
    """
    Single deterministic inference pipeline shared identically by official validation
    and final test prediction:
      - Uses ONLY image_path + sample_id + derived image/tabular features (zero metadata features)
      - Evaluates all K neural checkpoints simultaneously per D4 symmetry view so CPU SRAM
        deconvolution runs once per view
      - Combines symmetric zero-bias vMF neural logits with 5-fold GBDT log-odds around 0.0 (p = 0.5)
    """
    infer_meta = meta_df[["sample_id", "image_path"]].copy().reset_index(drop=True)
    n_samples = len(infer_meta)
    assert len(X_neural) == n_samples and len(X_tree) == n_samples

    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    n_workers = max(2, min(6, (os.cpu_count() or 4) // 2))

    loaded_models: List[TopoStainDINOv2PyramidNet] = []
    for ckpt_path in neural_ckpt_paths:
        net = TopoStainDINOv2PyramidNet(
            tabular_dim=tabular_dim,
            family_indices=family_indices,
            backbone_name=DEFAULT_DINOV2_BACKBONE,
            pretrained=False,
            d_model=256,
            sphere_dim=128,
            drop_path_rate=0.15,
            dropout=0.20,
            modality_drop_prob=0.10,
        ).to(device)
        net.load_state_dict(safe_torch_load_state_dict(ckpt_path, device), strict=True)
        net.eval()
        loaded_models.append(net)

    n_models = len(loaded_models)
    n_views = len(tta_views)
    accum_nn_logits = np.zeros(n_samples, dtype=np.float64)

    with torch.inference_mode():
        for view_idx in tta_views:
            ds_view = HistopathologyMultiModalDataset(
                meta_df=infer_meta,
                X_neural=X_neural,
                stain_prior=stain_prior,
                input_dir=INPUT_DIR,
                is_train=False,
                tta_view_idx=int(view_idx),
                seed=SEED,
            )
            loader_view = DataLoader(
                ds_view,
                batch_size=INFER_BATCH_SIZE,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=use_cuda,
            )
            offset = 0
            for batch in loader_view:
                bsz = batch["canon_rgb"].shape[0]
                canon_rgb = batch["canon_rgb"].to(device, non_blocking=True)
                stain_ch = batch["stain_channels"].to(device, non_blocking=True)
                tab_neu = batch["tabular_neural"].to(device, non_blocking=True)

                batch_logit_sum = np.zeros(bsz, dtype=np.float64)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_cuda
                ):
                    for net in loaded_models:
                        out = net(canon_rgb, stain_ch, tab_neu)
                        batch_logit_sum += (
                            out["logits"]
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                accum_nn_logits[offset : offset + bsz] += batch_logit_sum
                offset += bsz
            assert offset == n_samples

    mean_nn_logits = accum_nn_logits / float(max(1, n_models * n_views))

    if w_gbdt > 0.0 and len(gbdt_fold_models) > 0:
        gbdt_logits_accum = np.zeros(n_samples, dtype=np.float64)
        for pair in gbdt_fold_models:
            gbdt_logits_accum += predict_gbdt_pair_logits(pair, X_tree)
        mean_gbdt_logits = np.clip(
            gbdt_logits_accum / float(len(gbdt_fold_models)), -8.0, 8.0
        )
        final_logits = (1.0 - w_gbdt) * mean_nn_logits + w_gbdt * mean_gbdt_logits
    else:
        final_logits = mean_nn_logits

    probabilities = np.clip(expit(final_logits), 1e-6, 1.0 - 1e-6).astype(np.float64)
    assert np.isfinite(probabilities).all(), "Non-finite inference probabilities!"

    del loaded_models
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    return probabilities


def run_step3_training_and_evaluation(pipeline_t0: float) -> float:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_s1 = pd.read_csv(WORKING_DIR / "train_s1_meta.csv")
    val_s1 = pd.read_csv(WORKING_DIR / "val_s1_meta.csv")
    full_dev_s2 = pd.read_csv(WORKING_DIR / "full_dev_s2_meta.csv")
    test_meta = pd.read_csv(WORKING_DIR / "test_meta.csv")
    sample_sub = pd.read_csv(INPUT_DIR / "sample_submission.csv")

    stage1_stain_prior = load_npz_dict(WORKING_DIR / "stage1_stain_prior.npz")
    stage2_stain_prior = load_npz_dict(WORKING_DIR / "stage2_stain_prior.npz")
    feats_s1 = load_npz_dict(WORKING_DIR / "features_stage1.npz")
    feats_s2 = load_npz_dict(WORKING_DIR / "features_stage2_refit.npz")

    with open(WORKING_DIR / "feature_metadata.pkl", "rb") as f:
        feat_meta = pickle.load(f)
    tabular_dim = len(feat_meta["feature_names"])
    family_indices = feat_meta["family_indices"]

    # Strict pre-flight leak isolation assertions between train_s1 and official val_s1
    assert set(train_s1["sample_id"]).isdisjoint(set(val_s1["sample_id"]))
    assert set(train_s1["slide_id"]).isdisjoint(set(val_s1["slide_id"]))
    assert set(train_s1["group_id"]).isdisjoint(set(val_s1["group_id"]))

    # ------------------------------------------------------------------------
    # Stage 0: Leave-One-Center-Out (LOCO) Unseen-Center Audit & Stage 1 GBDTs
    # ------------------------------------------------------------------------
    print(
        "[Stage 0] Running Leave-One-Center-Out (LOCO) & 5-Fold GBDT training on train.csv..."
    )
    s1_gbdt_models, oof_gbdt_sgkf, oof_gbdt_loco = run_gbdt_loco_and_kfold(
        meta_df=train_s1,
        X_tree=feats_s1["X_train_tree"],
        fold_col="sgkf_fold",
        seed=SEED,
        run_loco=True,
    )
    sgkf_gbdt_m = compute_detailed_metrics(feats_s1["y_train"], expit(oof_gbdt_sgkf))
    if oof_gbdt_loco is not None:
        loco_gbdt_m = compute_detailed_metrics(
            feats_s1["y_train"], expit(oof_gbdt_loco)
        )
        print(
            f"  GBDT OOF Audit -> LOCO Unseen-Center BalAcc: {loco_gbdt_m['balanced_accuracy']:.4f} "
            f"(AUROC={loco_gbdt_m['auroc']:.4f}) | 5-Fold SGKF BalAcc: {sgkf_gbdt_m['balanced_accuracy']:.4f}"
        )

    # ------------------------------------------------------------------------
    # Stage 1: Group-Disjoint Neural Ensemble Training STRICTLY on train.csv
    # ------------------------------------------------------------------------
    print(
        f"[Stage 1] Training {NUM_NEURAL_FOLDS}-fold TopoStainDINOv2PyramidNet ensemble strictly on train.csv..."
    )
    s1_ckpts: List[Path] = []
    nn_oof_logits = np.zeros(len(train_s1), dtype=np.float64)
    nn_oof_mask = np.zeros(len(train_s1), dtype=bool)
    stage1_deadline = MAX_PIPELINE_SECONDS * 0.48

    for k in range(NUM_NEURAL_FOLDS):
        if k > 0 and (time.time() - pipeline_t0) > stage1_deadline:
            break
        va_mask = train_s1["sgkf_fold"].values == k
        tr_mask = ~va_mask
        ckpt_k = WORKING_DIR / f"stage1_dinov2_fold{k}.pt"
        ckpt_saved, fold_oof_logits, _ = train_single_neural_fold(
            fold_idx=k,
            tr_df=train_s1.loc[tr_mask],
            va_df=train_s1.loc[va_mask],
            X_tr_neural=feats_s1["X_train_neural"][tr_mask],
            X_va_neural=feats_s1["X_train_neural"][va_mask],
            stain_prior=stage1_stain_prior,
            ckpt_path=ckpt_k,
            device=device,
            seed=SEED + k * 41,
            stage_tag="Stage1",
            pipeline_t0=pipeline_t0,
            stage_deadline_sec=stage1_deadline,
        )
        s1_ckpts.append(ckpt_saved)
        nn_oof_logits[va_mask] = fold_oof_logits
        nn_oof_mask[va_mask] = True

    nn_oof_m = compute_detailed_metrics(
        feats_s1["y_train"][nn_oof_mask], expit(nn_oof_logits[nn_oof_mask])
    )
    selected_w_gbdt = select_leakfree_gbdt_blend_weight(
        meta_df=train_s1,
        nn_oof_logits=nn_oof_logits,
        nn_oof_mask=nn_oof_mask,
        gbdt_loco_logits=oof_gbdt_loco,
        gbdt_kfold_logits=oof_gbdt_sgkf,
    )
    print(
        f"  Stage 1 Internal OOF Summary (n={int(nn_oof_mask.sum())}) -> Neural OOF BalAcc: "
        f"{nn_oof_m['balanced_accuracy']:.4f} (AUROC={nn_oof_m['auroc']:.4f}) | "
        f"Locked LOCO-Audited GBDT Blend Weight: {selected_w_gbdt:.2f}"
    )

    # ------------------------------------------------------------------------
    # Official Pre-Refit Validation Evaluation on Untouched validation.csv
    # ------------------------------------------------------------------------
    print(
        f"[Validation] Running deterministic 8-view D4-TTA ensemble inference on untouched validation.csv (n={len(val_s1)})..."
    )
    val_probabilities = run_unified_ensemble_inference(
        meta_df=val_s1,
        X_neural=feats_s1["X_val_neural"],
        X_tree=feats_s1["X_val_tree"],
        stain_prior=stage1_stain_prior,
        neural_ckpt_paths=s1_ckpts,
        gbdt_fold_models=s1_gbdt_models,
        w_gbdt=selected_w_gbdt,
        tabular_dim=tabular_dim,
        family_indices=family_indices,
        device=device,
        tta_views=TTA_VIEWS,
    )
    val_metrics = compute_detailed_metrics(feats_s1["y_val"], val_probabilities)
    validation_balanced_accuracy = float(val_metrics["balanced_accuracy"])
    validation_auroc = float(val_metrics["auroc"])

    per_center_logs = []
    for c_name, grp in val_s1.groupby("center_id", sort=True):
        c_idx_rows = grp.index.values
        c_m = compute_detailed_metrics(
            feats_s1["y_val"][c_idx_rows], val_probabilities[c_idx_rows]
        )
        per_center_logs.append(
            f"{c_name}: bal_acc={c_m['balanced_accuracy']:.4f} (tpr={c_m['tpr']:.4f}, tnr={c_m['tnr']:.4f})"
        )

    print(
        f"  Official Validation Results (threshold=0.5) -> BalAcc: {validation_balanced_accuracy:.6f} | "
        f"AUROC: {validation_auroc:.6f} | TPR: {val_metrics['tpr']:.4f} | TNR: {val_metrics['tnr']:.4f}"
    )
    print(f"  Per-Center Validation Breakdown -> {' | '.join(per_center_logs)}")

    val_out_df = pd.DataFrame(
        {
            "sample_id": val_s1["sample_id"].values,
            "label": feats_s1["y_val"],
            "tumor_probability": val_probabilities,
        }
    )
    val_out_df.to_csv(WORKING_DIR / "stage1_validation_predictions.csv", index=False)

    # ------------------------------------------------------------------------
    # Stage 2: Post-Validation Refit on Merged Full-Dev (train + validation)
    # ------------------------------------------------------------------------
    target_s2_folds = len(s1_ckpts)
    print(
        f"[Stage 2 Refit] Refitting identical {target_s2_folds}-fold neural + 5-fold GBDT pipeline on "
        f"merged full_dev (train + validation = {len(full_dev_s2)} labeled patches)..."
    )
    s2_gbdt_models, oof_s2_gbdt, _ = run_gbdt_loco_and_kfold(
        meta_df=full_dev_s2,
        X_tree=feats_s2["X_fulldev_tree"],
        fold_col="refit_fold",
        seed=SEED + 500,
        run_loco=False,
    )
    s2_gbdt_m = compute_detailed_metrics(feats_s2["y_fulldev"], expit(oof_s2_gbdt))
    print(
        f"  Stage 2 Full-Dev 5-Fold GBDT OOF BalAcc: {s2_gbdt_m['balanced_accuracy']:.4f} "
        f"(AUROC={s2_gbdt_m['auroc']:.4f})"
    )

    s2_ckpts: List[Path] = []
    stage2_deadline = (
        MAX_PIPELINE_SECONDS - 240.0
    )  # Reserve >= 4 min for 8-view test TTA
    for k in range(target_s2_folds):
        if k > 0 and (time.time() - pipeline_t0) > stage2_deadline:
            break
        va_mask = full_dev_s2["refit_fold"].values == k
        tr_mask = ~va_mask
        ckpt_k = WORKING_DIR / f"stage2_dinov2_fold{k}.pt"
        ckpt_saved, _, _ = train_single_neural_fold(
            fold_idx=k,
            tr_df=full_dev_s2.loc[tr_mask],
            va_df=full_dev_s2.loc[va_mask],
            X_tr_neural=feats_s2["X_fulldev_neural"][tr_mask],
            X_va_neural=feats_s2["X_fulldev_neural"][va_mask],
            stain_prior=stage2_stain_prior,
            ckpt_path=ckpt_k,
            device=device,
            seed=SEED + 700 + k * 41,
            stage_tag="Stage2Refit",
            pipeline_t0=pipeline_t0,
            stage_deadline_sec=stage2_deadline,
        )
        s2_ckpts.append(ckpt_saved)

    # ------------------------------------------------------------------------
    # Final Test Inference & Submission Verification
    # ------------------------------------------------------------------------
    print(
        f"[Test Inference] Running identical 8-view D4-TTA ensemble inference on test.csv "
        f"(n={len(test_meta)}, neural_folds={len(s2_ckpts)}, w_gbdt={selected_w_gbdt:.2f})..."
    )
    test_probabilities = run_unified_ensemble_inference(
        meta_df=test_meta,
        X_neural=feats_s2["X_test_neural"],
        X_tree=feats_s2["X_test_tree"],
        stain_prior=stage2_stain_prior,
        neural_ckpt_paths=s2_ckpts,
        gbdt_fold_models=s2_gbdt_models,
        w_gbdt=selected_w_gbdt,
        tabular_dim=tabular_dim,
        family_indices=family_indices,
        device=device,
        tta_views=TTA_VIEWS,
    )

    submission_df = pd.DataFrame(
        {
            "sample_id": test_meta["sample_id"].values,
            "tumor_probability": test_probabilities,
        }
    )
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    # Strict submission format & content checks against sample_submission.csv
    assert SUBMISSION_PATH.is_file() and SUBMISSION_PATH.stat().st_size > 0
    sub_check = pd.read_csv(SUBMISSION_PATH)
    assert list(sub_check.columns) == ["sample_id", "tumor_probability"]
    assert len(sub_check) == len(sample_sub) == len(test_meta)
    assert (sub_check["sample_id"].values == sample_sub["sample_id"].values).all()
    probs_arr = sub_check["tumor_probability"].values
    assert np.isfinite(probs_arr).all()
    assert ((probs_arr >= 0.0) & (probs_arr <= 1.0)).all()
    assert float(np.std(probs_arr)) > 0.05, "Degenerate near-constant test predictions!"

    print(
        f"[Complete] Saved verified ./submission/submission.csv "
        f"(rows={len(sub_check)}, mean={probs_arr.mean():.4f}, std={probs_arr.std():.4f}, "
        f"elapsed={time.time() - pipeline_t0:.1f}s)"
    )
    return validation_balanced_accuracy


def main() -> None:
    pipeline_t0 = time.time()
    run_step1_data_processing_and_feature_engineering()
    validation_balanced_accuracy = run_step3_training_and_evaluation(pipeline_t0)
    print(f"Final Validation Score: {validation_balanced_accuracy}")


if __name__ == "__main__":
    main()