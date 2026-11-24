import copy
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

# ============================================================================
# 1. Configuration, Metric Import & Reproducibility
# ============================================================================
SCRIPT_START_TIME = time.time()
SEED = 42
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

if INPUT_DIR not in sys.path:
    sys.path.insert(0, INPUT_DIR)
from metric import binary_metrics


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ============================================================================
# 2. Load Datasets & Verify Structural Integrity / Leakage Isolation
# ============================================================================
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(INPUT_DIR, "validation.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

# Enforce exact row ordering alignment with sample_submission.csv
if not (test_df["sample_id"].values == sample_sub_df["sample_id"].values).all():
    test_df = sample_sub_df[["sample_id"]].merge(test_df, on="sample_id", how="left")
assert (
    test_df["sample_id"].values == sample_sub_df["sample_id"].values
).all(), "Test ID order mismatch!"

# Resolve full image paths relative to ./input/
for df_part in (train_df, val_df, test_df):
    df_part["full_image_path"] = df_part["image_path"].apply(
        lambda p: os.path.join(INPUT_DIR, p)
    )

# Verify labels exist in train and validation and check group/slide isolation
val_has_labels = ("label" in val_df.columns) and val_df["label"].notna().all()
assert val_has_labels, "Expected ground-truth labels in validation.csv"
train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)

assert set(train_df["slide_id"]).isdisjoint(
    set(val_df["slide_id"])
), "Slide leakage between train and val!"
assert set(train_df["group_id"]).isdisjoint(
    set(val_df["group_id"])
), "Group leakage between train and val!"

# ============================================================================
# 3. Leakage-Aware Splitting Strategy
#    A) Official train_df / val_df preserved for strict pre-refit evaluation
#    B) Center-held-out & 5-fold StratifiedGroupKFold inside train_df
#    C) Merged (train + validation) dataset with StratifiedGroupKFold on group_id
#       for full-data refitting / K-fold training
# ============================================================================
unique_train_centers = sorted(train_df["center_id"].unique().tolist())
center_to_idx = {cid: idx for idx, cid in enumerate(unique_train_centers)}
train_df["center_fold"] = train_df["center_id"].map(center_to_idx).astype(int)

# 5-fold StratifiedGroupKFold strictly within train_df
sgkf_train = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_strat_key = (
    train_df["center_id"].astype(str) + "_" + train_df["label"].astype(str)
)
train_df["cv_fold"] = -1
for fold_idx, (tr_idx, va_idx) in enumerate(
    sgkf_train.split(train_df, y=train_strat_key, groups=train_df["group_id"])
):
    assert set(train_df.iloc[tr_idx]["group_id"]).isdisjoint(
        set(train_df.iloc[va_idx]["group_id"])
    )
    assert set(train_df.iloc[tr_idx]["slide_id"]).isdisjoint(
        set(train_df.iloc[va_idx]["slide_id"])
    )
    train_df.iloc[va_idx, train_df.columns.get_loc("cv_fold")] = fold_idx

# Compute inverse (center, label) sample weights strictly on train_df for domain-balanced learning
train_cell_counts = train_df.groupby(["center_id", "label"])["sample_id"].transform(
    "count"
)
n_train_cells = train_df.groupby(["center_id", "label"]).ngroups
train_df["sample_weight"] = len(train_df) / (
    n_train_cells * train_cell_counts.astype(np.float32)
)

# Confirm and construct merged dataset (train + validation) with group-isolated splits
train_copy = train_df.copy()
train_copy["is_orig_val"] = 0
val_copy = val_df.copy()
val_copy["is_orig_val"] = 1
val_copy["center_fold"] = (
    val_copy["center_id"].map(center_to_idx).fillna(-1).astype(int)
)
val_copy["cv_fold"] = -1
val_copy["sample_weight"] = 1.0

merged_df = pd.concat([train_copy, val_copy], axis=0, ignore_index=True)
sgkf_merged = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
merged_strat_key = (
    merged_df["center_id"].astype(str) + "_" + merged_df["label"].astype(str)
)
merged_df["merged_fold"] = -1
for fold_idx, (tr_idx, va_idx) in enumerate(
    sgkf_merged.split(merged_df, y=merged_strat_key, groups=merged_df["group_id"])
):
    assert set(merged_df.iloc[tr_idx]["group_id"]).isdisjoint(
        set(merged_df.iloc[va_idx]["group_id"])
    )
    assert set(merged_df.iloc[tr_idx]["slide_id"]).isdisjoint(
        set(merged_df.iloc[va_idx]["slide_id"])
    )
    merged_df.iloc[va_idx, merged_df.columns.get_loc("merged_fold")] = fold_idx

merged_cell_counts = merged_df.groupby(["center_id", "label"])["sample_id"].transform(
    "count"
)
n_merged_cells = merged_df.groupby(["center_id", "label"]).ngroups
merged_df["merged_sample_weight"] = len(merged_df) / (
    n_merged_cells * merged_cell_counts.astype(np.float32)
)

print(
    f"Loaded splits | train: {len(train_df)} (pos={train_df['label'].mean():.4f}), "
    f"val: {len(val_df)} (pos={val_df['label'].mean():.4f}), "
    f"merged(train+val): {len(merged_df)}, test: {len(test_df)}"
)
print(
    f"Confirmed validation labels present and merged into merged_df ({len(merged_df)} rows) "
    f"with zero slide/group leakage across 5 StratifiedGroupKFold splits."
)


# ============================================================================
# 4. Physics-Based Beer-Lambert H&E Stain Deconvolution & Morphological Features
# ============================================================================
class BeerLambertStainDeconvolver:
    """Fast LUT-based Beer-Lambert Optical Density H&E stain deconvolution."""

    def __init__(self) -> None:
        # Ruifrok & Johnston H&E unit optical density absorbance vectors + orthogonal residual
        v_h = np.array([0.650, 0.704, 0.286], dtype=np.float32)
        v_e = np.array([0.072, 0.990, 0.105], dtype=np.float32)
        v_h = v_h / np.linalg.norm(v_h)
        v_e = v_e / np.linalg.norm(v_e)
        v_r = np.cross(v_h, v_e)
        v_r = v_r / np.linalg.norm(v_r)

        # Forward: OD = concentrations @ stain_matrix
        self.stain_matrix = np.stack([v_h, v_e, v_r], axis=0).astype(np.float32)
        # Inverse: concentrations = OD @ stain_inv
        self.stain_inv = np.linalg.inv(self.stain_matrix).astype(np.float32)
        # Precomputed uint8 -> Optical Density LUT: -log10(max(I, 1) / 255.0)
        intensities = np.maximum(np.arange(256, dtype=np.float32), 1.0)
        self.od_lut = (-np.log10(intensities / 255.0)).astype(np.float32)

    def rgb_to_her(self, img_rgb: np.ndarray):
        od = self.od_lut[img_rgb]  # (H, W, 3) float32
        her = od @ self.stain_inv  # (H, W, 3): Hematoxylin, Eosin, Residual
        return her, od

    def her_to_rgb(
        self, her: np.ndarray, stain_matrix_override: np.ndarray = None
    ) -> np.ndarray:
        mat = (
            self.stain_matrix
            if stain_matrix_override is None
            else stain_matrix_override
        )
        od_rec = np.maximum(her @ mat, 0.0)
        rgb_rec = 255.0 * np.power(10.0, -np.minimum(od_rec, 2.5))
        return np.clip(rgb_rec, 0.0, 255.0).astype(np.uint8)


STAIN_DECONVOLVER = BeerLambertStainDeconvolver()

FEATURE_NAMES = [
    # Stain-invariant multi-scale spatial variance ratios (nuclear vs tumor-nest scale)
    "h_var_ratio_2x",
    "h_var_ratio_4x",
    "h_var_ratio_8x",
    "h_var_ratio_16x",
    "e_var_ratio_2x",
    "e_var_ratio_4x",
    "e_var_ratio_8x",
    "e_var_ratio_16x",
    # Scale-normalized chromatin & stromal gradient / Laplacian sharpness (full & center 96x96)
    "h_grad_norm_full",
    "h_lap_norm_full",
    "h_grad_norm_c96",
    "h_lap_norm_c96",
    "e_grad_norm_full",
    "e_lap_norm_full",
    "e_grad_norm_c96",
    "e_lap_norm_c96",
    # Within-patch center (96x96 & 48x48) vs periphery self-normalized stain contrasts
    "h_c96_minus_rim_norm",
    "h_c48_minus_rim_norm",
    "e_c96_minus_rim_norm",
    "e_c48_minus_rim_norm",
    "h_c96_to_full_std_ratio",
    "e_c96_to_full_std_ratio",
    # Dimensionless distribution shape (skewness, tail-heaviness, IQR ratios)
    "h_skew_quant_full",
    "h_tail_ratio_full",
    "h_skew_quant_c96",
    "h_tail_ratio_c96",
    "e_skew_quant_full",
    "e_tail_ratio_full",
    "e_skew_quant_c96",
    "e_tail_ratio_c96",
    # H-E spatial correlation & relative concentration ratios
    "he_corr_full",
    "he_corr_c96",
    "he_corr_c48",
    "he_p90_ratio_full",
    "he_p90_ratio_c96",
    "he_iqr_ratio_full",
    "he_iqr_ratio_c96",
    # Adaptive nuclear density & tissue foreground fractions
    "tissue_frac_full",
    "tissue_frac_c96",
    "nuc_adapt_frac_full",
    "nuc_adapt_frac_c96",
    "nuc_adapt_frac_c48",
]


def read_rgb_image(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        from PIL import Image

        with Image.open(path) as pil_img:
            img_rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if img_rgb.shape[:2] != (256, 256):
        img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    return img_rgb


def _safe_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt(np.mean(xm * xm) * np.mean(ym * ym)))
    if denom < eps:
        return 0.0
    return float(np.mean(xm * ym) / denom)


def _extract_single_patch_info(path: str):
    """Extracts 42 stain-invariant morphological ratios + foreground LAB stats for calibration."""
    eps = 1e-6
    img_rgb = read_rgb_image(path)
    her, od = STAIN_DECONVOLVER.rgb_to_her(img_rgb)

    H = np.maximum(her[:, :, 0], 0.0)
    E = np.maximum(her[:, :, 1], 0.0)
    R = her[:, :, 2]
    od_norm = np.linalg.norm(od, axis=2)

    # White-balance background offset subtraction per patch (2nd percentile OD = background glass)
    h_bg = float(np.percentile(H[::4, ::4], 2.0))
    e_bg = float(np.percentile(E[::4, ::4], 2.0))
    H_wb = np.maximum(H - h_bg, 0.0)
    E_wb = np.maximum(E - e_bg, 0.0)

    # Concentric spatial zones: full (256x256), center 96x96 [80:176], inner core 48x48 [104:152]
    H_c96 = H_wb[80:176, 80:176]
    E_c96 = E_wb[80:176, 80:176]
    H_c48 = H_wb[104:152, 104:152]
    E_c48 = E_wb[104:152, 104:152]
    od_c96 = od_norm[80:176, 80:176]

    # 1) Multi-scale variance spectrum on H and E (strictly invariant to affine stain shifts)
    h_var0 = float(np.var(H_wb)) + eps
    h_p2 = H_wb.reshape(128, 2, 128, 2).mean(axis=(1, 3))
    h_p4 = h_p2.reshape(64, 2, 64, 2).mean(axis=(1, 3))
    h_p8 = h_p4.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    h_p16 = h_p8.reshape(16, 2, 16, 2).mean(axis=(1, 3))

    e_var0 = float(np.var(E_wb)) + eps
    e_p2 = E_wb.reshape(128, 2, 128, 2).mean(axis=(1, 3))
    e_p4 = e_p2.reshape(64, 2, 64, 2).mean(axis=(1, 3))
    e_p8 = e_p4.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    e_p16 = e_p8.reshape(16, 2, 16, 2).mean(axis=(1, 3))

    h_vr2, h_vr4 = float(np.var(h_p2)) / h_var0, float(np.var(h_p4)) / h_var0
    h_vr8, h_vr16 = float(np.var(h_p8)) / h_var0, float(np.var(h_p16)) / h_var0
    e_vr2, e_vr4 = float(np.var(e_p2)) / e_var0, float(np.var(e_p4)) / e_var0
    e_vr8, e_vr16 = float(np.var(e_p8)) / e_var0, float(np.var(e_p16)) / e_var0

    # 2) Scale-normalized chromatin & stromal gradients and discrete Laplacian
    h_std_full = float(np.sqrt(h_var0))
    e_std_full = float(np.sqrt(e_var0))
    h_std_c96 = float(np.std(H_c96)) + eps
    e_std_c96 = float(np.std(E_c96)) + eps

    h_gx = np.abs(H_wb[1:-1, 2:] - H_wb[1:-1, :-2]) * 0.5
    h_gy = np.abs(H_wb[2:, 1:-1] - H_wb[:-2, 1:-1]) * 0.5
    h_grad_map = h_gx + h_gy
    h_lap_map = np.abs(
        4.0 * H_wb[1:-1, 1:-1]
        - H_wb[:-2, 1:-1]
        - H_wb[2:, 1:-1]
        - H_wb[1:-1, :-2]
        - H_wb[1:-1, 2:]
    )

    e_gx = np.abs(E_wb[1:-1, 2:] - E_wb[1:-1, :-2]) * 0.5
    e_gy = np.abs(E_wb[2:, 1:-1] - E_wb[:-2, 1:-1]) * 0.5
    e_grad_map = e_gx + e_gy
    e_lap_map = np.abs(
        4.0 * E_wb[1:-1, 1:-1]
        - E_wb[:-2, 1:-1]
        - E_wb[2:, 1:-1]
        - E_wb[1:-1, :-2]
        - E_wb[1:-1, 2:]
    )

    h_grad_norm_full = float(np.mean(h_grad_map)) / h_std_full
    h_lap_norm_full = float(np.mean(h_lap_map)) / h_std_full
    h_grad_norm_c96 = float(np.mean(h_grad_map[79:175, 79:175])) / h_std_c96
    h_lap_norm_c96 = float(np.mean(h_lap_map[79:175, 79:175])) / h_std_c96

    e_grad_norm_full = float(np.mean(e_grad_map)) / e_std_full
    e_lap_norm_full = float(np.mean(e_lap_map)) / e_std_full
    e_grad_norm_c96 = float(np.mean(e_grad_map[79:175, 79:175])) / e_std_c96
    e_lap_norm_c96 = float(np.mean(e_lap_map[79:175, 79:175])) / e_std_c96

    # 3) Center vs Peripheral Rim self-normalized contrast
    total_px = 256 * 256
    c96_px = 96 * 96
    h_sum_full, e_sum_full = float(np.sum(H_wb)), float(np.sum(E_wb))
    h_sum_c96, e_sum_c96 = float(np.sum(H_c96)), float(np.sum(E_c96))
    h_rim_mean = (h_sum_full - h_sum_c96) / float(total_px - c96_px)
    e_rim_mean = (e_sum_full - e_sum_c96) / float(total_px - c96_px)

    h_c96_mean, e_c96_mean = h_sum_c96 / c96_px, e_sum_c96 / c96_px
    h_c48_mean, e_c48_mean = float(np.mean(H_c48)), float(np.mean(E_c48))

    h_c96_minus_rim_norm = (h_c96_mean - h_rim_mean) / h_std_full
    h_c48_minus_rim_norm = (h_c48_mean - h_rim_mean) / h_std_full
    e_c96_minus_rim_norm = (e_c96_mean - e_rim_mean) / e_std_full
    e_c48_minus_rim_norm = (e_c48_mean - e_rim_mean) / e_std_full
    h_c96_to_full_std_ratio = h_std_c96 / h_std_full
    e_c96_to_full_std_ratio = e_std_c96 / e_std_full

    # 4) Quantiles & dimensionless skewness / tail ratios
    hp10, hp25, hp50, hp75, hp90, hp95 = np.percentile(
        H_wb[::2, ::2], [10, 25, 50, 75, 90, 95]
    )
    ep10, ep25, ep50, ep75, ep90, ep95 = np.percentile(
        E_wb[::2, ::2], [10, 25, 50, 75, 90, 95]
    )
    hcp10, hcp25, hcp50, hcp75, hcp90, hcp95 = np.percentile(
        H_c96, [10, 25, 50, 75, 90, 95]
    )
    ecp10, ecp25, ecp50, ecp75, ecp90, ecp95 = np.percentile(
        E_c96, [10, 25, 50, 75, 90, 95]
    )

    h_iqr_full = float(hp75 - hp25) + eps
    e_iqr_full = float(ep75 - ep25) + eps
    h_iqr_c96 = float(hcp75 - hcp25) + eps
    e_iqr_c96 = float(ecp75 - ecp25) + eps

    h_skew_quant_full = float(hp90 - 2.0 * hp50 + hp10) / (float(hp90 - hp10) + eps)
    h_tail_ratio_full = float(hp95 - hp50) / h_iqr_full
    h_skew_quant_c96 = float(hcp90 - 2.0 * hcp50 + hcp10) / (float(hcp90 - hcp10) + eps)
    h_tail_ratio_c96 = float(hcp95 - hcp50) / h_iqr_c96

    e_skew_quant_full = float(ep90 - 2.0 * ep50 + ep10) / (float(ep90 - ep10) + eps)
    e_tail_ratio_full = float(ep95 - ep50) / e_iqr_full
    e_skew_quant_c96 = float(ecp90 - 2.0 * ecp50 + ecp10) / (float(ecp90 - ecp10) + eps)
    e_tail_ratio_c96 = float(ecp95 - ecp50) / e_iqr_c96

    # 5) H-E spatial correlation and concentration ratios
    he_corr_full = _safe_corr(H_wb[::2, ::2], E_wb[::2, ::2], eps)
    he_corr_c96 = _safe_corr(H_c96, E_c96, eps)
    he_corr_c48 = _safe_corr(H_c48, E_c48, eps)
    he_p90_ratio_full = float(hp90) / (float(ep90) + 0.05)
    he_p90_ratio_c96 = float(hcp90) / (float(ecp90) + 0.05)
    he_iqr_ratio_full = h_iqr_full / (e_iqr_full + 0.01)
    he_iqr_ratio_c96 = h_iqr_c96 / (e_iqr_c96 + 0.01)

    # 6) Tissue foreground & adaptive nuclear area fractions
    tissue_frac_full = float(np.mean(od_norm > 0.15))
    tissue_frac_c96 = float(np.mean(od_c96 > 0.15))
    h_mean_full = h_sum_full / total_px
    e_mean_full = e_sum_full / total_px

    adapt_thr = max(h_mean_full + 0.5 * h_std_full, 0.18)
    nuc_adapt_frac_full = float(np.mean(H_wb > adapt_thr))
    nuc_adapt_frac_c96 = float(np.mean(H_c96 > adapt_thr))
    nuc_adapt_frac_c48 = float(np.mean(H_c48 > adapt_thr))

    feats = np.array(
        [
            h_vr2,
            h_vr4,
            h_vr8,
            h_vr16,
            e_vr2,
            e_vr4,
            e_vr8,
            e_vr16,
            h_grad_norm_full,
            h_lap_norm_full,
            h_grad_norm_c96,
            h_lap_norm_c96,
            e_grad_norm_full,
            e_lap_norm_full,
            e_grad_norm_c96,
            e_lap_norm_c96,
            h_c96_minus_rim_norm,
            h_c48_minus_rim_norm,
            e_c96_minus_rim_norm,
            e_c48_minus_rim_norm,
            h_c96_to_full_std_ratio,
            e_c96_to_full_std_ratio,
            h_skew_quant_full,
            h_tail_ratio_full,
            h_skew_quant_c96,
            h_tail_ratio_c96,
            e_skew_quant_full,
            e_tail_ratio_full,
            e_skew_quant_c96,
            e_tail_ratio_c96,
            he_corr_full,
            he_corr_c96,
            he_corr_c48,
            he_p90_ratio_full,
            he_p90_ratio_c96,
            he_iqr_ratio_full,
            he_iqr_ratio_c96,
            tissue_frac_full,
            tissue_frac_c96,
            nuc_adapt_frac_full,
            nuc_adapt_frac_c96,
            nuc_adapt_frac_c48,
        ],
        dtype=np.float32,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute foreground-only LAB stats (od_norm > 0.12 with >=64 foreground pixel check)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_mask = od_norm > 0.12
    fg_count = int(np.count_nonzero(fg_mask))
    if fg_count >= 64:
        fg_lab = img_lab[fg_mask]
        lab_means = fg_lab.mean(axis=0)
        lab_stds = fg_lab.std(axis=0)
        fg_valid = np.array([1.0], dtype=np.float32)
    else:
        lab_means = img_lab.mean(axis=(0, 1))
        lab_stds = img_lab.std(axis=(0, 1))
        fg_valid = np.array([0.0], dtype=np.float32)
    rgb_f = img_rgb.astype(np.float32) / 255.0
    rgb_means = rgb_f.mean(axis=(0, 1))
    rgb_sq_means = (rgb_f * rgb_f).mean(axis=(0, 1))
    her_summary = np.array(
        [
            float(H.mean()),
            float(H.std()),
            float(E.mean()),
            float(E.std()),
            float(od_norm.mean()),
            float(od_norm.std()),
        ],
        dtype=np.float32,
    )
    aux_stats = np.concatenate(
        [lab_means, lab_stds, rgb_means, rgb_sq_means, her_summary, fg_valid]
    ).astype(np.float32)
    return feats, aux_stats


def extract_dataset_features(paths_series: pd.Series, max_workers: int = 12):
    paths_list = paths_series.tolist()
    n = len(paths_list)
    feats_out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    aux_out = np.zeros((n, 19), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (f_vec, a_vec) in enumerate(
            executor.map(_extract_single_patch_info, paths_list)
        ):
            feats_out[idx] = f_vec
            aux_out[idx] = a_vec
    return feats_out, aux_out


num_io_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
X_train_raw, aux_train = extract_dataset_features(
    train_df["full_image_path"], max_workers=num_io_workers
)
X_val_raw, aux_val = extract_dataset_features(
    val_df["full_image_path"], max_workers=num_io_workers
)
X_test_raw, _ = extract_dataset_features(
    test_df["full_image_path"], max_workers=num_io_workers
)
X_merged_raw = np.concatenate([X_train_raw, X_val_raw], axis=0)

assert (
    np.isfinite(X_train_raw).all()
    and np.isfinite(X_val_raw).all()
    and np.isfinite(X_test_raw).all()
)


# ============================================================================
# 5. Fit Stateful Stain Reference Stats & Feature Scalers STRICTLY on Training Data
# ============================================================================
def fit_stain_reference_stats(
    aux_matrix: np.ndarray, domain_expansion: float = 1.25
) -> dict:
    """
    Fits RandStainNA virtual-center foreground LAB template distribution and channel
    normalization statistics strictly on training foreground tissue (never val or test).
    """
    if aux_matrix.shape[1] > 18:
        valid_fg = aux_matrix[:, 18] > 0.5
        fg_aux = aux_matrix[valid_fg] if np.any(valid_fg) else aux_matrix
    else:
        fg_aux = aux_matrix
    lab_means = fg_aux[:, 0:3]
    lab_stds = fg_aux[:, 3:6]
    rgb_means = aux_matrix[:, 6:9].mean(axis=0)
    rgb_sq_means = aux_matrix[:, 9:12].mean(axis=0)
    rgb_stds = np.sqrt(np.maximum(rgb_sq_means - rgb_means**2, 1e-6))

    h_mean = float(aux_matrix[:, 12].mean())
    h_std = float(np.sqrt(np.mean(aux_matrix[:, 13] ** 2) + np.var(aux_matrix[:, 12])))
    e_mean = float(aux_matrix[:, 14].mean())
    e_std = float(np.sqrt(np.mean(aux_matrix[:, 15] ** 2) + np.var(aux_matrix[:, 14])))
    od_mean = float(aux_matrix[:, 16].mean())
    od_std = float(np.sqrt(np.mean(aux_matrix[:, 17] ** 2) + np.var(aux_matrix[:, 16])))

    return {
        "lab_mean_loc": lab_means.mean(axis=0).tolist(),
        "lab_mean_scale": (lab_means.std(axis=0) * domain_expansion).tolist(),
        "lab_std_loc": lab_stds.mean(axis=0).tolist(),
        "lab_std_scale": (lab_stds.std(axis=0) * domain_expansion).tolist(),
        "rgb_mean": rgb_means.tolist(),
        "rgb_std": rgb_stds.tolist(),
        "stain_channel_mean": [h_mean, e_mean, od_mean],
        "stain_channel_std": [max(h_std, 1e-4), max(e_std, 1e-4), max(od_std, 1e-4)],
    }


# Strict isolation: fitted ONLY on train_df (13,200 patches) for official validation & test
stain_stats_train = fit_stain_reference_stats(aux_train, domain_expansion=1.25)

# Separate post-validation refit reference (ONLY for optional refit after validation score is recorded)
stain_stats_merged = fit_stain_reference_stats(
    np.concatenate([aux_train, aux_val], axis=0), domain_expansion=1.25
)

# Fit RobustScaler STRICTLY on X_train_raw; transform validation and test identically
feature_scaler = RobustScaler(quantile_range=(10.0, 90.0))
X_train_feats = np.clip(feature_scaler.fit_transform(X_train_raw), -10.0, 10.0).astype(
    np.float32
)
X_val_feats = np.clip(feature_scaler.transform(X_val_raw), -10.0, 10.0).astype(
    np.float32
)
X_test_feats = np.clip(feature_scaler.transform(X_test_raw), -10.0, 10.0).astype(
    np.float32
)

y_train = train_df["label"].values.astype(np.int64)
y_val = val_df["label"].values.astype(np.int64)
y_merged = merged_df["label"].values.astype(np.int64)

print(
    f"Engineered {len(FEATURE_NAMES)} stain-invariant morphological features | "
    f"X_train: {X_train_feats.shape}, X_val: {X_val_feats.shape}, X_test: {X_test_feats.shape}"
)
print(
    f"Fitted RandStainNA virtual-center LAB stats strictly on train_df | "
    f"LAB mean loc={[round(v, 2) for v in stain_stats_train['lab_mean_loc']]}"
)


# ============================================================================
# 6. Domain-Generalizing Histopathology Augmentations & PyTorch Dataset
# ============================================================================
class HistopathologyAugmentor:
    """
    Training-time physics-based domain randomization for unseen medical centers:
      1) Exact D4 Dihedral group symmetry (rotations of 0/90/180/270 + reflections)
      2) Beer-Lambert H&E stain-concentration scaling/shifting + stain-vector perturbation
      3) RandStainNA virtual-center CIE LAB distribution alignment (fit strictly on train)
      4) Subtle scanner optical defocus / sharpening & sensor noise
    """

    def __init__(
        self,
        stain_stats: dict,
        p_hed: float = 0.55,
        p_randstain: float = 0.30,
        p_optical: float = 0.25,
    ):
        self.lab_mean_loc = np.array(stain_stats["lab_mean_loc"], dtype=np.float32)
        self.lab_mean_scale = np.array(stain_stats["lab_mean_scale"], dtype=np.float32)
        self.lab_std_loc = np.array(stain_stats["lab_std_loc"], dtype=np.float32)
        self.lab_std_scale = np.array(stain_stats["lab_std_scale"], dtype=np.float32)
        self.p_hed = p_hed
        self.p_randstain = p_randstain
        self.p_optical = p_optical

    def _apply_dihedral_d4(self, img: np.ndarray) -> np.ndarray:
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k)
        if np.random.rand() < 0.5:
            img = np.fliplr(img)
        if np.random.rand() < 0.5:
            img = np.flipud(img)
        return np.ascontiguousarray(img)

    def _apply_hed_perturbation(self, img_rgb: np.ndarray) -> np.ndarray:
        her, _ = STAIN_DECONVOLVER.rgb_to_her(img_rgb)
        scales = np.array(
            [
                np.random.uniform(0.70, 1.35),
                np.random.uniform(0.70, 1.35),
                np.random.uniform(0.85, 1.15),
            ],
            dtype=np.float32,
        )
        shifts = np.array(
            [
                np.random.uniform(-0.07, 0.07),
                np.random.uniform(-0.07, 0.07),
                np.random.uniform(-0.03, 0.03),
            ],
            dtype=np.float32,
        )
        her_aug = her * scales + shifts

        mat_jitter = STAIN_DECONVOLVER.stain_matrix + np.random.normal(
            0.0, 0.018, size=(3, 3)
        ).astype(np.float32)
        mat_jitter = mat_jitter / (
            np.linalg.norm(mat_jitter, axis=1, keepdims=True) + 1e-6
        )
        img_out = STAIN_DECONVOLVER.her_to_rgb(
            her_aug, stain_matrix_override=mat_jitter
        )

        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.85, 1.18)
            lut = np.clip(
                255.0 * np.power(np.arange(256, dtype=np.float32) / 255.0, gamma),
                0,
                255,
            ).astype(np.uint8)
            img_out = lut[img_out]
        return img_out

    def _apply_randstainna(self, img_rgb: np.ndarray) -> np.ndarray:
        od = STAIN_DECONVOLVER.od_lut[img_rgb]
        od_norm = np.linalg.norm(od, axis=2)
        fg_mask = od_norm > 0.12
        if int(np.count_nonzero(fg_mask)) < 64:
            return img_rgb

        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        fg_lab = lab[fg_mask]
        cur_mean = fg_lab.mean(axis=0)
        cur_std = fg_lab.std(axis=0)

        tgt_mean = (
            self.lab_mean_loc
            + np.clip(np.random.randn(3).astype(np.float32), -2.2, 2.2)
            * self.lab_mean_scale
        )
        tgt_std = np.maximum(
            self.lab_std_loc
            + np.clip(np.random.randn(3).astype(np.float32), -2.0, 2.0)
            * self.lab_std_scale,
            2.0,
        )
        scale_ratio = np.clip(tgt_std / np.maximum(cur_std, 3.0), 0.55, 1.85)
        fg_aligned = np.clip((fg_lab - cur_mean) * scale_ratio + tgt_mean, 0.0, 255.0)

        lab_u8 = np.clip(lab, 0.0, 255.0).astype(np.uint8)
        lab_u8[fg_mask] = fg_aligned.astype(np.uint8)
        rgb_aligned = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)
        out_rgb = img_rgb.copy()
        out_rgb[fg_mask] = rgb_aligned[fg_mask]
        return out_rgb

    def _apply_optical_variation(self, img_rgb: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            sigma = np.random.uniform(0.3, 0.85)
            img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=sigma)
        else:
            blur = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=0.8)
            alpha = np.random.uniform(0.15, 0.45)
            img_rgb = np.clip(
                img_rgb.astype(np.float32) * (1.0 + alpha)
                - blur.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)
        return img_rgb

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        img = self._apply_dihedral_d4(img_rgb)
        u_stain = np.random.rand()
        if u_stain < self.p_hed:
            img = self._apply_hed_perturbation(img)
        elif u_stain < (self.p_hed + self.p_randstain):
            img = self._apply_randstainna(img)
        if np.random.rand() < self.p_optical:
            img = self._apply_optical_variation(img)
        return img


class HistopathologyDataset(Dataset):
    """
    Consistent PyTorch Dataset for training, validation, and test inference.
    Validation and test use 100% identical deterministic normalization and formatting.
    Metadata identifiers are strictly excluded from model inputs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stain_stats: dict,
        features: np.ndarray = None,
        is_train: bool = False,
        img_size: int = 256,
        return_stain_channels: bool = False,
        augmentor: HistopathologyAugmentor = None,
    ):
        self.paths = df["full_image_path"].tolist()
        self.has_labels = "label" in df.columns and df["label"].notna().all()
        self.labels = df["label"].values.astype(np.float32) if self.has_labels else None
        self.features = features.astype(np.float32) if features is not None else None
        self.is_train = is_train
        self.img_size = img_size
        self.return_stain_channels = return_stain_channels
        self.augmentor = (
            augmentor
            if (is_train and augmentor is not None)
            else (HistopathologyAugmentor(stain_stats) if is_train else None)
        )
        self.stain_ch_mean = np.array(
            stain_stats["stain_channel_mean"], dtype=np.float32
        )
        self.stain_ch_std = np.array(stain_stats["stain_channel_std"], dtype=np.float32)
        if self.is_train and ("center_fold" in df.columns):
            self.center_indices = df["center_fold"].values.astype(np.int64)
        elif self.is_train and ("center_id" in df.columns):
            uniq_c = sorted(df["center_id"].astype(str).unique().tolist())
            c_map = {c: i for i, c in enumerate(uniq_c)}
            self.center_indices = (
                df["center_id"].astype(str).map(c_map).values.astype(np.int64)
            )
        else:
            self.center_indices = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img_rgb = read_rgb_image(self.paths[idx])
        if self.is_train and self.augmentor is not None:
            img_rgb = self.augmentor(img_rgb)

        if self.img_size != 256:
            interp = cv2.INTER_LINEAR if self.img_size > 256 else cv2.INTER_AREA
            img_rgb = cv2.resize(
                img_rgb, (self.img_size, self.img_size), interpolation=interp
            )

        rgb_norm = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        if self.return_stain_channels:
            her, od = STAIN_DECONVOLVER.rgb_to_her(img_rgb)
            h_ch = np.maximum(her[:, :, 0], 0.0)
            e_ch = np.maximum(her[:, :, 1], 0.0)
            od_ch = np.linalg.norm(od, axis=2)
            stain_stack = np.stack([h_ch, e_ch, od_ch], axis=-1)
            stain_norm = (stain_stack - self.stain_ch_mean) / self.stain_ch_std
            full_arr = np.concatenate([rgb_norm, stain_norm], axis=-1)
            img_tensor = torch.from_numpy(np.transpose(full_arr, (2, 0, 1)).copy())
        else:
            img_tensor = torch.from_numpy(np.transpose(rgb_norm, (2, 0, 1)).copy())

        out = {"image": img_tensor}
        if self.features is not None:
            feat_vec = self.features[idx]
            if self.is_train:
                feat_vec = feat_vec + np.random.normal(
                    0.0, 0.05, size=feat_vec.shape
                ).astype(np.float32)
                out["features"] = torch.from_numpy(feat_vec.copy())
            else:
                out["features"] = torch.from_numpy(feat_vec)
        if self.labels is not None:
            out["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.is_train and self.center_indices is not None:
            out["center_idx"] = torch.tensor(
                int(self.center_indices[idx]), dtype=torch.long
            )
        return out


def get_dihedral_tta_views(batch_images: torch.Tensor):
    """
    Deterministic D4 dihedral symmetries for consistent validation & test inference.
    Returns list of 8 exact symmetry-transformed batches on (B, C, H, W).
    """
    v0 = batch_images
    v1 = torch.rot90(batch_images, k=1, dims=(2, 3))
    v2 = torch.rot90(batch_images, k=2, dims=(2, 3))
    v3 = torch.rot90(batch_images, k=3, dims=(2, 3))
    f0 = torch.flip(v0, dims=(3,))
    f1 = torch.flip(v1, dims=(3,))
    f2 = torch.flip(v2, dims=(3,))
    f3 = torch.flip(v3, dims=(3,))
    return [v0, v1, v2, v3, f0, f1, f2, f3]


train_dataset = HistopathologyDataset(
    train_df, stain_stats=stain_stats_train, features=X_train_feats, is_train=True
)
val_dataset = HistopathologyDataset(
    val_df, stain_stats=stain_stats_train, features=X_val_feats, is_train=False
)
test_dataset = HistopathologyDataset(
    test_df, stain_stats=stain_stats_train, features=X_test_feats, is_train=False
)

sample_train_item = train_dataset[0]
sample_val_item = val_dataset[0]
sample_test_item = test_dataset[0]
assert (
    sample_train_item["image"].shape == (3, 256, 256)
    and torch.isfinite(sample_train_item["image"]).all()
)
assert (
    sample_val_item["image"].shape == (3, 256, 256)
    and torch.isfinite(sample_val_item["image"]).all()
)
assert (
    sample_test_item["image"].shape == (3, 256, 256)
    and torch.isfinite(sample_test_item["image"]).all()
)

# Persist processed splits, features, and stain reference artifacts
train_df.to_csv(os.path.join(WORKING_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(WORKING_DIR, "val_split.csv"), index=False)
merged_df.to_csv(os.path.join(WORKING_DIR, "merged_split.csv"), index=False)
test_df.to_csv(os.path.join(WORKING_DIR, "test_split.csv"), index=False)

np.savez_compressed(
    os.path.join(WORKING_DIR, "engineered_features.npz"),
    X_train_raw=X_train_raw,
    X_val_raw=X_val_raw,
    X_merged_raw=X_merged_raw,
    X_test_raw=X_test_raw,
    X_train_feats=X_train_feats,
    X_val_feats=X_val_feats,
    X_test_feats=X_test_feats,
    y_train=y_train,
    y_val=y_val,
    y_merged=y_merged,
    feature_names=np.array(FEATURE_NAMES),
)

with open(os.path.join(WORKING_DIR, "stain_stats.json"), "w") as f_json:
    json.dump(
        {
            "stain_stats_train": stain_stats_train,
            "stain_stats_merged": stain_stats_merged,
        },
        f_json,
        indent=2,
    )

print(
    f"Saved processed splits, {len(FEATURE_NAMES)} engineered features, and stain reference stats to {WORKING_DIR}."
)

# ============================================================================
# 7. Model Design: StainInvariantMultiScaleConvNeXt & BalancedHistopathologyLoss
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "backbone_candidates": (
        "convnext_small.fb_in22k_ft_in1k_384",
        "convnext_small.fb_in22k_ft_in1k",
        "convnext_small",
    ),
    "pretrained_source": "ImageNet-22K pretrained, ImageNet-1K 384x384 fine-tuned (natural photographs)",
    "training_data_overlap": "None (0% overlap with H&E histopathology or lymph-node microscopy datasets)",
    "num_morph_features": len(FEATURE_NAMES),
    "drop_path_rate": 0.25,
    "head_dropout": 0.35,
    "morph_dropout": 0.25,
    "mixstyle_p": 0.50,
    "mixstyle_alpha": 0.20,
    "lr_backbone": 1.5e-4,
    "lr_head": 6.0e-4,
    "layer_decay": 0.80,
    "weight_decay": 2.0e-2,
    "label_smoothing": 0.03,
    "ba_loss_weight": 0.25,
    "ba_margin_temp": 0.70,
    "group_dro_eta": 0.02,
    "group_dro_decay": 0.08,
    "coral_loss_weight": 0.10,
}


class ChannelLayerNorm2d(nn.Module):
    """Per-sample, per-spatial-location LayerNorm over channels (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        mean = x_f.mean(dim=1, keepdim=True)
        var = (x_f - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x_f - mean) * torch.rsqrt(var + self.eps)
        out = x_norm * self.weight.float().view(1, -1, 1, 1) + self.bias.float().view(
            1, -1, 1, 1
        )
        return out.to(orig_dtype)


class MixStyle2d(nn.Module):
    """
    Training-only cross-sample feature style mixing (Zhou et al., ICLR 2021).
    Synthesizes virtual medical-center stain/contrast statistics in early CNN stages.
    Strict deterministic identity pass-through during validation and test inference.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.2, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self._beta_dist = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (x.size(0) <= 1) or (torch.rand(1).item() > self.p):
            return x

        orig_dtype = x.dtype
        x_f = x.float()
        B = x_f.size(0)

        mu = x_f.mean(dim=(2, 3), keepdim=True)
        var = (x_f - mu).pow(2).mean(dim=(2, 3), keepdim=True)
        sig = torch.sqrt(var + self.eps)

        x_normed = (x_f - mu) / sig

        perm = torch.randperm(B, device=x.device)
        mu_perm = mu.detach()[perm]
        sig_perm = sig.detach()[perm]

        lmbd = self._beta_dist.sample((B, 1, 1, 1)).to(
            device=x.device, dtype=torch.float32
        )
        mu_mix = lmbd * mu.detach() + (1.0 - lmbd) * mu_perm
        sig_mix = lmbd * sig.detach() + (1.0 - lmbd) * sig_perm

        out = x_normed * sig_mix + mu_mix
        return out.to(orig_dtype)


class BeerLambertStemAdapter(nn.Module):
    """
    Differentiable on-GPU Beer-Lambert Optical Density H&E deconvolution + fp32
    foreground-weighted spatial normalization residual stem. Cancels per-patch affine
    dye concentration shifts over foreground tissue without background glass drift.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        stain_inv_np = STAIN_DECONVOLVER.stain_inv

        self.register_buffer(
            "rgb_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "stain_inv_t",
            torch.tensor(stain_inv_np.T, dtype=torch.float32).view(3, 3, 1, 1),
        )

        self.fg_norm_weight = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.fg_norm_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=4, stride=4, bias=True),
            ChannelLayerNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=True,
            ),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            ChannelLayerNorm2d(out_channels),
        )
        self.gate_alpha = nn.Parameter(torch.full((1, out_channels, 1, 1), 0.20))

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        orig_dtype = x_rgb_norm.dtype
        with torch.amp.autocast(device_type=x_rgb_norm.device.type, enabled=False):
            x_f = x_rgb_norm[:, :3, :, :].float()
            rgb_01 = torch.clamp(x_f * self.rgb_std + self.rgb_mean, 1.0 / 255.0, 1.0)
            od = -torch.log10(rgb_01)
            her = F.conv2d(od, self.stain_inv_t)

            od_norm = torch.sqrt(torch.sum(od * od, dim=1, keepdim=True) + 1e-8)
            w_fg = torch.sigmoid((od_norm - 0.12) / 0.04)
            tissue_cov = (od_norm > 0.12).float().mean(dim=(2, 3), keepdim=True)
            t_blend = torch.clamp(tissue_cov / 0.05, 0.0, 1.0)
            alpha_cov = t_blend * t_blend * (3.0 - 2.0 * t_blend)

            p_fg = w_fg / w_fg.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            p_unif = 1.0 / float(her.size(2) * her.size(3))
            p_spatial = alpha_cov * p_fg + (1.0 - alpha_cov) * p_unif

            mu = (her * p_spatial).sum(dim=(2, 3), keepdim=True)
            var = ((her - mu).pow(2) * p_spatial).sum(dim=(2, 3), keepdim=True)
            her_norm = (her - mu) * torch.rsqrt(var + 1e-5)
            her_in = her_norm * self.fg_norm_weight + self.fg_norm_bias

        stem_feat = self.proj(her_in.to(orig_dtype))
        return stem_feat * self.gate_alpha.to(orig_dtype)


class RadialBiasedD4SelfAttention2d(nn.Module):
    """
    D4-equivariant pre-norm Multi-Head Self-Attention on 2D token grids using learnable
    per-head attention biases indexed strictly by pairwise Euclidean grid distances.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        grid_size: int = 8,
        attn_drop: float = 0.10,
        proj_drop: float = 0.15,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.grid_size = grid_size

        ys, xs = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.long),
            torch.arange(grid_size, dtype=torch.long),
            indexing="ij",
        )
        coords = torch.stack([ys.flatten(), xs.flatten()], dim=-1)
        d2 = (coords[:, 0:1] - coords[None, :, 0]).pow(2) + (
            coords[:, 1:2] - coords[None, :, 1]
        ).pow(2)
        unique_d2, dist_idx = torch.unique(d2, sorted=True, return_inverse=True)

        self.register_buffer("radial_bias_index", dist_idx)
        self.register_buffer("unique_radii", torch.sqrt(unique_d2.float()))
        self.radial_bias_table = nn.Parameter(torch.zeros(num_heads, len(unique_d2)))

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma = nn.Parameter(torch.full((dim,), 0.10))

    def _get_bias_index(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        if h == self.grid_size and w == self.grid_size:
            return self.radial_bias_index
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([ys.flatten(), xs.flatten()], dim=-1)
        dist = torch.cdist(coords, coords, p=2.0)
        return torch.argmin(
            torch.abs(dist.unsqueeze(-1) - self.unique_radii.view(1, 1, -1)), dim=-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        tokens = x.flatten(2).transpose(1, 2)
        tokens_norm = self.norm(tokens)

        qkv = (
            self.qkv(tokens_norm)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        idx_mat = self._get_bias_index(H, W, x.device)
        bias = self.radial_bias_table[:, idx_mat].float().unsqueeze(0)
        attn_logits = (q.float() @ k.float().transpose(-2, -1)) * self.scale + bias
        attn = self.attn_drop(F.softmax(attn_logits, dim=-1)).to(v.dtype)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.proj(out))
        tokens = tokens + out * self.gamma.to(out.dtype)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class GatedMILAttentionPool2d(nn.Module):
    """
    D4-invariant Gated Multiple-Instance-Learning (MIL) spatial attention pooling
    (Ilse et al., ICML 2018). Enables focal micrometastatic cell clusters to stand out
    without dilution from benign surrounding lymphoid background.
    """

    def __init__(self, in_channels: int, attn_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        self.attn_u = nn.Sequential(
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.attn_w = nn.Conv2d(attn_dim, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        gated = self.attn_drop(self.attn_v(x) * self.attn_u(x))
        scores = self.attn_w(gated).float().view(B, 1, H * W)
        weights = F.softmax(scores, dim=-1).to(x.dtype)
        tokens = x.view(B, C, H * W)
        pooled = torch.bmm(tokens, weights.transpose(1, 2)).squeeze(-1)
        return pooled


class ResidualMorphBlock(nn.Module):
    """Regularized residual MLP block for stain-invariant morphological descriptors."""

    def __init__(self, dim: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StainInvariantMultiScaleConvNeXt(nn.Module):
    """
    Domain-Generalizing Multi-Scale ConvNeXt + Beer-Lambert Morphological Fusion Network:
      - Pretrained BatchNorm-free ConvNeXt backbone (LayerNorm throughout)
      - Beer-Lambert Optical Density InstanceNorm residual stem adapter
      - Training-time MixStyle2d feature style randomization in Stages 1 & 2
      - D4-invariant Multi-Scale Gated MIL + Concentric Center-ROI Pooling (Stages 2, 3, 4)
      - Bounded FiLM Gated Cross-Modal Fusion with 54 Beer-Lambert morphological features
      - Multi-Sample Dropout classification head for calibrated probabilities around p=0.5
    """

    def __init__(
        self,
        backbone_candidates=MODEL_CONFIG["backbone_candidates"],
        pretrained: bool = True,
        num_morph_features: int = MODEL_CONFIG["num_morph_features"],
        drop_path_rate: float = MODEL_CONFIG["drop_path_rate"],
        head_dropout: float = MODEL_CONFIG["head_dropout"],
        morph_dropout: float = MODEL_CONFIG["morph_dropout"],
        mixstyle_p: float = MODEL_CONFIG["mixstyle_p"],
        mixstyle_alpha: float = MODEL_CONFIG["mixstyle_alpha"],
    ):
        super().__init__()
        self.num_morph_features = num_morph_features

        self.backbone = None
        self.backbone_name = None
        last_err = None
        candidates = (
            (backbone_candidates,)
            if isinstance(backbone_candidates, str)
            else tuple(backbone_candidates)
        )
        for cand_name in candidates:
            try:
                self.backbone = timm.create_model(
                    cand_name,
                    pretrained=pretrained,
                    num_classes=0,
                    drop_path_rate=drop_path_rate,
                )
                self.backbone_name = cand_name
                break
            except Exception as exc:
                last_err = exc
        if self.backbone is None:
            raise RuntimeError(
                f"Failed to instantiate ConvNeXt backbone: {last_err}"
            ) from last_err

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            s0 = self.backbone.stem(dummy)
            f0 = self.backbone.stages[0](s0)
            f1 = self.backbone.stages[1](f0)
            f2 = self.backbone.stages[2](f1)
            f3 = self.backbone.stages[3](f2)
            c1, c2, c3 = f1.shape[1], f2.shape[1], f3.shape[1]

        self.od_stem_adapter = BeerLambertStemAdapter(out_channels=s0.shape[1])
        self.mixstyle_s0 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)
        self.mixstyle_s1 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)

        self.norm_s1 = ChannelLayerNorm2d(c1)
        self.proj_s1 = nn.Sequential(
            nn.Linear(c1 * 2, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )

        self.norm_s2 = ChannelLayerNorm2d(c2)
        self.mil_s2 = GatedMILAttentionPool2d(c2, attn_dim=128, dropout=0.15)
        self.proj_s2 = nn.Sequential(
            nn.Linear(c2 * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.norm_s3 = ChannelLayerNorm2d(c3)
        self.attn_s3 = RadialBiasedD4SelfAttention2d(
            c3, num_heads=8, grid_size=8, attn_drop=0.10, proj_drop=0.15
        )
        self.mil_s3 = GatedMILAttentionPool2d(c3, attn_dim=192, dropout=0.15)
        self.proj_s3 = nn.Sequential(
            nn.Linear(c3 * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        visual_concat_dim = 192 + 256 + 512
        self.visual_neck = nn.Sequential(
            nn.LayerNorm(visual_concat_dim),
            nn.Dropout(head_dropout * 0.8),
            nn.Linear(visual_concat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )

        self.morph_feat_drop = nn.Dropout(0.15)
        self.morph_encoder = nn.Sequential(
            nn.LayerNorm(num_morph_features),
            nn.Linear(num_morph_features, 128),
            nn.GELU(),
            nn.Dropout(morph_dropout),
            ResidualMorphBlock(128, dropout=morph_dropout),
            nn.LayerNorm(128),
        )

        self.film_generator = nn.Linear(128, 512 * 2)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        fused_dim = 512 + 128
        self.fused_norm = nn.LayerNorm(fused_dim)

        self.ms_dropouts_1 = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(5)])
        self.head_fc1 = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        self.ms_dropouts_2 = nn.ModuleList([nn.Dropout(head_dropout) for _ in range(5)])
        self.head_out = nn.Linear(256, 1)

    @staticmethod
    def _center_roi_avg_pool(feat_map: torch.Tensor) -> torch.Tensor:
        H, W = feat_map.shape[2], feat_map.shape[3]
        h_margin = max(1, int(round(H * 0.3125)))
        w_margin = max(1, int(round(W * 0.3125)))
        if H - 2 * h_margin < 2:
            h_margin = max(0, (H - 2) // 2)
        if W - 2 * w_margin < 2:
            w_margin = max(0, (W - 2) // 2)
        center_crop = feat_map[:, :, h_margin : H - h_margin, w_margin : W - w_margin]
        return center_crop.mean(dim=(2, 3))

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        return_aux: bool = False,
    ):
        rgb_in = images[:, :3, :, :] if images.size(1) > 3 else images

        x = self.backbone.stem(rgb_in) + self.od_stem_adapter(rgb_in)

        x = self.backbone.stages[0](x)
        x = self.mixstyle_s0(x)

        f1 = self.backbone.stages[1](x)
        f1_styled = self.mixstyle_s1(f1)

        f2 = self.backbone.stages[2](f1_styled)
        f3 = self.backbone.stages[3](f2)

        s1_norm = self.norm_s1(f1)
        v1 = self.proj_s1(
            torch.cat(
                [s1_norm.mean(dim=(2, 3)), self._center_roi_avg_pool(s1_norm)], dim=1
            )
        )

        s2_norm = self.norm_s2(f2)
        v2 = self.proj_s2(
            torch.cat([self.mil_s2(s2_norm), self._center_roi_avg_pool(s2_norm)], dim=1)
        )

        s3_norm = self.attn_s3(self.norm_s3(f3))
        v3 = self.proj_s3(
            torch.cat(
                [
                    self.mil_s3(s3_norm),
                    s3_norm.mean(dim=(2, 3)),
                    self._center_roi_avg_pool(s3_norm),
                ],
                dim=1,
            )
        )

        v_embed = self.visual_neck(torch.cat([v1, v2, v3], dim=1))

        if features is not None:
            m_in = self.morph_feat_drop(features.to(dtype=v_embed.dtype))
            m_embed = self.morph_encoder(m_in)
            film_params = torch.tanh(self.film_generator(m_embed))
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            v_mod = v_embed * (1.0 + 0.25 * gamma) + 0.25 * beta
        else:
            m_embed = torch.zeros(
                (v_embed.size(0), 128), device=v_embed.device, dtype=v_embed.dtype
            )
            v_mod = v_embed

        fused = self.fused_norm(torch.cat([v_mod, m_embed], dim=1))

        logits_acc = 0.0
        for drop1, drop2 in zip(self.ms_dropouts_1, self.ms_dropouts_2):
            h = self.head_fc1(drop1(fused))
            logits_acc = logits_acc + self.head_out(drop2(h))
        logits = (logits_acc / float(len(self.ms_dropouts_1))).squeeze(-1)
        if return_aux:
            return logits, v_embed
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        use_tta: bool = False,
        tta_views: int = 8,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            if not use_tta or tta_views <= 1:
                logits = self.forward(images, features)
                probs = torch.sigmoid(logits.float())
            else:
                views = get_dihedral_tta_views(images)[:tta_views]
                prob_sum = 0.0
                for view_img in views:
                    view_logits = self.forward(view_img, features)
                    prob_sum = prob_sum + torch.sigmoid(view_logits.float())
                probs = prob_sum / float(len(views))
            return torch.clamp(probs, 0.0, 1.0)
        finally:
            if was_training:
                self.train()


class BalancedHistopathologyLoss(nn.Module):
    """
    Composite domain-generalizing loss for Balanced Accuracy at fixed p=0.5 threshold:
      1) Stateful Exponentiated-Gradient GroupDRO over (center_id, label) smoothed BCE
      2) Per-Center Differentiable Soft Balanced Accuracy (SoftTPR + SoftTNR)
      3) Dimension-normalized fp32 Class-Conditional Deep CORAL on visual embeddings
    """

    def __init__(
        self,
        pos_prior: float = 0.5,
        label_smoothing: float = MODEL_CONFIG["label_smoothing"],
        ba_weight: float = MODEL_CONFIG["ba_loss_weight"],
        margin_temp: float = MODEL_CONFIG["ba_margin_temp"],
        num_centers: int = 2,
        dro_eta: float = MODEL_CONFIG["group_dro_eta"],
        dro_decay: float = MODEL_CONFIG["group_dro_decay"],
        coral_weight: float = MODEL_CONFIG["coral_loss_weight"],
    ):
        super().__init__()
        clamped_prior = float(np.clip(pos_prior, 0.05, 0.95))
        self.prior_logit_offset = math.log(clamped_prior / (1.0 - clamped_prior))
        self.label_smoothing = float(label_smoothing)
        self.ba_weight = float(ba_weight)
        self.margin_temp = float(margin_temp)
        self.num_centers = max(2, int(num_centers))
        self.num_groups = self.num_centers * 2
        self.dro_eta = float(dro_eta)
        self.dro_decay = float(dro_decay)
        self.coral_weight = float(coral_weight)
        self.register_buffer(
            "group_weights",
            torch.full((self.num_groups,), 1.0 / float(self.num_groups), dtype=torch.float32),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor = None,
        center_ids: torch.Tensor = None,
        v_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            logits_f = logits.view(-1).float()
            targets_f = targets.view(-1).float()
            smoothed_targets = (
                targets_f * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )

            if center_ids is not None:
                c_ids = torch.clamp(
                    center_ids.view(-1).long().to(logits_f.device),
                    0,
                    self.num_centers - 1,
                )
                y_bin = (targets_f > 0.5).long()
                g_ids = c_ids * 2 + y_bin

                per_sample_bce = F.binary_cross_entropy_with_logits(
                    logits_f, smoothed_targets, reduction="none"
                )

                group_losses = []
                present_indices = []
                for g in range(self.num_groups):
                    g_mask = g_ids == g
                    if g_mask.any():
                        group_losses.append(per_sample_bce[g_mask].mean())
                        present_indices.append(g)

                if len(present_indices) > 0:
                    g_loss_vec = torch.stack(group_losses)
                    idx_t = torch.tensor(
                        present_indices, device=logits_f.device, dtype=torch.long
                    )
                    if self.training and len(present_indices) >= 2:
                        with torch.no_grad():
                            det_l = g_loss_vec.detach()
                            centered = det_l - det_l.mean()
                            log_w = torch.log(self.group_weights.clamp(min=1e-6))
                            log_w[idx_t] = (
                                1.0 - self.dro_decay
                            ) * log_w[idx_t] + self.dro_eta * centered
                            new_w = torch.softmax(log_w, dim=0)
                            new_w = torch.clamp(new_w, min=0.10, max=0.50)
                            self.group_weights.copy_(new_w / new_w.sum())

                    active_w = self.group_weights[idx_t]
                    active_w = active_w / active_w.sum().clamp(min=1e-6)
                    bce_loss = (active_w * g_loss_vec).sum()
                else:
                    bce_loss = per_sample_bce.mean()

                probs_sharp = torch.sigmoid(logits_f / self.margin_temp)
                center_ba_losses = []
                center_ba_weights = []
                for c in range(self.num_centers):
                    c_pos = (c_ids == c) & (y_bin == 1)
                    c_neg = (c_ids == c) & (y_bin == 0)
                    if c_pos.any() and c_neg.any():
                        soft_tpr_c = probs_sharp[c_pos].mean()
                        soft_tnr_c = (1.0 - probs_sharp[c_neg]).mean()
                        center_ba_losses.append(1.0 - 0.5 * (soft_tpr_c + soft_tnr_c))
                        center_ba_weights.append(
                            self.group_weights[2 * c] + self.group_weights[2 * c + 1]
                        )

                if self.ba_weight > 0.0 and len(center_ba_losses) > 0:
                    c_w = torch.stack(center_ba_weights)
                    c_w = c_w / c_w.sum().clamp(min=1e-6)
                    soft_ba_loss = (c_w * torch.stack(center_ba_losses)).sum()
                    total_loss = (
                        1.0 - self.ba_weight
                    ) * bce_loss + self.ba_weight * soft_ba_loss
                else:
                    total_loss = bce_loss

                if (v_embed is not None) and (self.coral_weight > 0.0):
                    v_f = v_embed.float()
                    coral_terms = []
                    for cls_val in (0, 1):
                        for c_a in range(self.num_centers):
                            z_a = v_f[(c_ids == c_a) & (y_bin == cls_val)]
                            if z_a.size(0) < 2:
                                continue
                            mu_a = z_a.mean(dim=0)
                            za_c = z_a - mu_a
                            cov_a = (za_c.t() @ za_c) / float(z_a.size(0) - 1)
                            for c_b in range(c_a + 1, self.num_centers):
                                z_b = v_f[(c_ids == c_b) & (y_bin == cls_val)]
                                if z_b.size(0) < 2:
                                    continue
                                mu_b = z_b.mean(dim=0)
                                zb_c = z_b - mu_b
                                cov_b = (zb_c.t() @ zb_c) / float(z_b.size(0) - 1)
                                mean_align = (mu_a - mu_b).pow(2).mean()
                                cov_align = 0.25 * (cov_a - cov_b).pow(2).mean()
                                coral_terms.append(mean_align + cov_align)
                    if len(coral_terms) > 0:
                        coral_loss = torch.stack(coral_terms).mean()
                        total_loss = total_loss + self.coral_weight * coral_loss

                return total_loss

            adj_logits = (
                logits_f
                if sample_weight is not None
                else (logits_f + self.prior_logit_offset)
            )
            per_sample_bce = F.binary_cross_entropy_with_logits(
                adj_logits, smoothed_targets, reduction="none"
            )

            if sample_weight is not None:
                w = sample_weight.view(-1).float().to(logits_f.device)
                bce_loss = (per_sample_bce * w).sum() / w.sum().clamp(min=1e-6)
            else:
                w = None
                bce_loss = per_sample_bce.mean()

            pos_mask = (targets_f > 0.5).float()
            neg_mask = 1.0 - pos_mask
            n_pos = pos_mask.sum()
            n_neg = neg_mask.sum()

            if (self.ba_weight > 0.0) and (n_pos > 0.5) and (n_neg > 0.5):
                probs_sharp = torch.sigmoid(logits_f / self.margin_temp)
                if w is not None:
                    soft_tpr = (w * pos_mask * probs_sharp).sum() / (
                        w * pos_mask
                    ).sum().clamp(min=1e-6)
                    soft_tnr = (w * neg_mask * (1.0 - probs_sharp)).sum() / (
                        w * neg_mask
                    ).sum().clamp(min=1e-6)
                else:
                    soft_tpr = (pos_mask * probs_sharp).sum() / n_pos.clamp(min=1e-6)
                    soft_tnr = (neg_mask * (1.0 - probs_sharp)).sum() / n_neg.clamp(
                        min=1e-6
                    )
                soft_ba_loss = 1.0 - 0.5 * (soft_tpr + soft_tnr)
                return (1.0 - self.ba_weight) * bce_loss + self.ba_weight * soft_ba_loss

            return bce_loss


def build_optimizer(
    net: StainInvariantMultiScaleConvNeXt,
    lr_backbone: float = MODEL_CONFIG["lr_backbone"],
    lr_head: float = MODEL_CONFIG["lr_head"],
    layer_decay: float = MODEL_CONFIG["layer_decay"],
    weight_decay: float = MODEL_CONFIG["weight_decay"],
) -> AdamW:
    param_groups_dict = {}

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = (
            (param.ndim <= 1)
            or name.endswith(".bias")
            or ("norm" in name.lower())
            or ("gamma" in name.lower())
            or ("gate_alpha" in name)
            or ("bias_table" in name)
        )
        wd = 0.0 if is_no_decay else weight_decay

        if name.startswith("backbone.stem.") or name.startswith("backbone.stages.0."):
            lr = lr_backbone * (layer_decay**3)
            group_tag = "backbone_s0"
        elif name.startswith("backbone.stages.1."):
            lr = lr_backbone * (layer_decay**2)
            group_tag = "backbone_s1"
        elif name.startswith("backbone.stages.2."):
            lr = lr_backbone * (layer_decay**1)
            group_tag = "backbone_s2"
        elif name.startswith("backbone.stages.3.") or name.startswith("backbone."):
            lr = lr_backbone
            group_tag = "backbone_s3"
        else:
            lr = lr_head
            group_tag = "custom_head_and_adapters"

        key = (group_tag, lr, wd)
        if key not in param_groups_dict:
            param_groups_dict[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": wd,
                "group_name": f"{group_tag}_wd{wd}",
            }
        param_groups_dict[key]["params"].append(param)

    return AdamW(list(param_groups_dict.values()), betas=(0.9, 0.999), eps=1e-8)


def build_model_criterion_optimizer(
    pos_prior: float = None,
    pretrained: bool = True,
    device: torch.device = DEVICE,
    backbone_candidates=MODEL_CONFIG["backbone_candidates"],
    lr_backbone: float = MODEL_CONFIG["lr_backbone"],
    lr_head: float = MODEL_CONFIG["lr_head"],
):
    if pos_prior is None:
        pos_prior = float(np.mean(y_train))

    net = StainInvariantMultiScaleConvNeXt(
        backbone_candidates=backbone_candidates,
        pretrained=pretrained,
        num_morph_features=MODEL_CONFIG["num_morph_features"],
    ).to(device)

    loss_fn = BalancedHistopathologyLoss(
        pos_prior=pos_prior,
        label_smoothing=MODEL_CONFIG["label_smoothing"],
        ba_weight=MODEL_CONFIG["ba_loss_weight"],
        margin_temp=MODEL_CONFIG["ba_margin_temp"],
        num_centers=max(2, len(unique_train_centers)),
    ).to(device)

    opt = build_optimizer(
        net,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        layer_decay=MODEL_CONFIG["layer_decay"],
        weight_decay=MODEL_CONFIG["weight_decay"],
    )
    return net, loss_fn, opt


train_pos_prior = float(np.mean(y_train))
model, criterion, optimizer = build_model_criterion_optimizer(
    pos_prior=train_pos_prior,
    pretrained=True,
    device=DEVICE,
)

# Verify forward pass, loss computation, and official metric compatibility on 2 samples
model.eval()
with torch.no_grad():
    s0, s1 = val_dataset[0], val_dataset[1]
    chk_imgs = torch.stack([s0["image"], s1["image"]], dim=0).to(DEVICE)
    chk_feats = torch.stack([s0["features"], s1["features"]], dim=0).to(DEVICE)
    chk_targets = torch.tensor([0.0, 1.0], device=DEVICE)
    chk_logits = model(chk_imgs, chk_feats)
    chk_loss = criterion(chk_logits, chk_targets)
    chk_probs = model.predict_proba(chk_imgs, chk_feats, use_tta=True, tta_views=4)
    chk_metrics = binary_metrics(
        np.array([0, 1], dtype=int), chk_probs.detach().cpu().numpy()
    )

assert chk_logits.shape == (2,) and torch.isfinite(chk_logits).all()
assert torch.isfinite(chk_loss)
assert chk_probs.shape == (2,) and torch.isfinite(chk_probs).all()
assert "balanced_accuracy" in chk_metrics and "auroc" in chk_metrics

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Pretrained Backbone Identity: {model.backbone_name} | "
    f"timm=={timm.__version__}, torch=={torch.__version__} | "
    f"Source: {MODEL_CONFIG['pretrained_source']} | "
    f"Known Training-Data Overlap: {MODEL_CONFIG['training_data_overlap']}"
)
print(
    f"Initialized StainInvariantMultiScaleConvNeXt on {DEVICE} | "
    f"Params: {trainable_params:,} trainable / {total_params:,} total | "
    f"Optimizer groups: {len(optimizer.param_groups)} | "
    f"Train prior: {train_pos_prior:.4f}"
)

# ============================================================================
# 8. Training, Validation Audit & Test Submission Generation
# ============================================================================
seed_everything(SEED)
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

STEP_START_TIME = time.time()
MAX_STEP_SECONDS = min(
    2800.0, max(600.0, 3250.0 - (STEP_START_TIME - SCRIPT_START_TIME))
)

if DEVICE.type == "cuda":
    vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
    TRAIN_BATCH_SIZE = 64 if vram_gb >= 15.0 else 32
    GRAD_ACCUM_STEPS = 64 // TRAIN_BATCH_SIZE
    EVAL_BATCH_SIZE = 128 if vram_gb >= 15.0 else 64
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    USE_GRAD_SCALER = AMP_DTYPE == torch.float16
else:
    TRAIN_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    EVAL_BATCH_SIZE = 32
    USE_AMP = False
    AMP_DTYPE = torch.float32
    USE_GRAD_SCALER = False

MAX_EPOCHS = 14
EARLY_STOPPING_PATIENCE = 5
MAX_GRAD_NORM = 1.0
EMA_MAX_DECAY = 0.995
MIN_LR_RATIO = 0.02
FINAL_TTA_VIEWS = 8
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pt")


def _dataloader_worker_init_fn(worker_id: int) -> None:
    """Ensures distinct RNG streams and clean single-threaded OpenCV inside forked workers."""
    cv2.setNumThreads(0)
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class StratifiedCenterClassBatchSampler(Sampler):
    """
    Deterministic without-replacement batch sampler yielding exact balanced mini-batches
    containing batch_size // 4 samples from each (center_id, label) cell in train_df.
    """

    def __init__(self, df: pd.DataFrame, batch_size: int, seed: int = SEED):
        super().__init__()
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self._next_epoch = 0

        centers = sorted(df["center_id"].astype(str).unique().tolist())
        labels = sorted(df["label"].astype(int).unique().tolist())
        self.cell_indices = []
        for cid in centers:
            for lbl in labels:
                mask = (df["center_id"].astype(str) == cid) & (
                    df["label"].astype(int) == lbl
                )
                idx_arr = np.flatnonzero(mask.values).astype(np.int64)
                if len(idx_arr) == 0:
                    raise ValueError(f"Empty (center_id={cid}, label={lbl}) cell!")
                self.cell_indices.append(idx_arr)

        self.num_cells = len(self.cell_indices)
        if self.batch_size % self.num_cells != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by {self.num_cells}"
            )
        self.per_cell = self.batch_size // self.num_cells
        self.num_batches = max(1, len(df) // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self._next_epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        ep = self._next_epoch
        self._next_epoch += 1
        rng = np.random.RandomState(self.seed + ep * 9973)
        needed = self.num_batches * self.per_cell

        cell_streams = []
        for idx_arr in self.cell_indices:
            chunks = []
            total_len = 0
            while total_len < needed:
                perm = idx_arr[rng.permutation(len(idx_arr))]
                chunks.append(perm)
                total_len += len(perm)
            stream = np.concatenate(chunks, axis=0)[:needed]
            cell_streams.append(stream.reshape(self.num_batches, self.per_cell))

        for b_idx in range(self.num_batches):
            batch = np.concatenate([cs[b_idx] for cs in cell_streams], axis=0)
            rng.shuffle(batch)
            yield batch.tolist()


train_batch_sampler = StratifiedCenterClassBatchSampler(
    train_df, batch_size=TRAIN_BATCH_SIZE, seed=SEED
)

num_dl_workers = min(8, max(2, (os.cpu_count() or 4)))
pin_mem = DEVICE.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=num_dl_workers,
    pin_memory=pin_mem,
    persistent_workers=(num_dl_workers > 0),
    worker_init_fn=_dataloader_worker_init_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=num_dl_workers,
    pin_memory=pin_mem,
    drop_last=False,
    persistent_workers=(num_dl_workers > 0),
    worker_init_fn=_dataloader_worker_init_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=num_dl_workers,
    pin_memory=pin_mem,
    drop_last=False,
    persistent_workers=(num_dl_workers > 0),
    worker_init_fn=_dataloader_worker_init_fn,
)

for pg in optimizer.param_groups:
    pg["initial_lr"] = float(pg["lr"])

steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM_STEPS)
total_opt_steps = MAX_EPOCHS * steps_per_epoch
head_warmup_steps = max(1, int(round(0.8 * steps_per_epoch)))
backbone_warmup_steps = max(1, int(round(1.5 * steps_per_epoch)))


def set_step_learning_rates(
    opt: torch.optim.Optimizer, step_idx: int, total_steps: int
) -> float:
    head_lr_now = 0.0
    for pg in opt.param_groups:
        base_lr = pg["initial_lr"]
        gname = pg.get("group_name", "")
        is_backbone = gname.startswith("backbone_")
        w_steps = backbone_warmup_steps if is_backbone else head_warmup_steps
        start_factor = 0.02 if is_backbone else 0.05

        if step_idx < w_steps:
            alpha = float(step_idx + 1) / float(w_steps)
            mult = start_factor + (1.0 - start_factor) * alpha
        else:
            progress = float(step_idx - w_steps) / float(max(1, total_steps - w_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            mult = MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

        pg["lr"] = base_lr * mult
        if not is_backbone:
            head_lr_now = pg["lr"]
    return head_lr_now


ema_model = copy.deepcopy(model).to(DEVICE)
ema_model.eval()
for p_ema in ema_model.parameters():
    p_ema.requires_grad_(False)


@torch.no_grad()
def update_ema_weights(ema_net: nn.Module, src_net: nn.Module, opt_step: int) -> None:
    decay = min(EMA_MAX_DECAY, float(1 + opt_step) / float(12 + opt_step))
    for p_ema, p_src in zip(ema_net.parameters(), src_net.parameters()):
        p_ema.mul_(decay).add_(p_src.detach(), alpha=1.0 - decay)
    for b_ema, b_src in zip(ema_net.buffers(), src_net.buffers()):
        b_ema.copy_(b_src)


@torch.inference_mode()
def run_inference(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
    use_amp: bool = USE_AMP,
    amp_dtype: torch.dtype = AMP_DTYPE,
    use_tta: bool = False,
    tta_views: int = FINAL_TTA_VIEWS,
) -> np.ndarray:
    """Unified, leak-free inference function shared identically by validation and test sets."""
    net.eval()
    probs_chunks = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        features = (
            batch["features"].to(device, non_blocking=True)
            if "features" in batch
            else None
        )
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            batch_probs = net.predict_proba(
                images=images,
                features=features,
                use_tta=use_tta,
                tta_views=tta_views,
            )
        probs_chunks.append(batch_probs.detach().float().cpu().numpy())

    probs_all = np.concatenate(probs_chunks, axis=0).astype(np.float64)
    probs_all = np.nan_to_num(probs_all, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs_all, 0.0, 1.0)


scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_GRAD_SCALER)

best_val_ba = -1.0
best_val_auc = -1.0
best_epoch = 0
best_source = "raw"
epochs_without_improvement = 0
global_opt_step = 0
epoch_durations = []

torch.save(
    {"model_state_dict": model.state_dict(), "epoch": 0, "source": "init"},
    BEST_MODEL_PATH,
)

print(
    f"Starting training | epochs={MAX_EPOCHS}, batch={TRAIN_BATCH_SIZE}x{GRAD_ACCUM_STEPS}, "
    f"amp={AMP_DTYPE}, train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
)

for epoch in range(1, MAX_EPOCHS + 1):
    ep_t0 = time.time()
    model.train()
    criterion.train()
    train_batch_sampler.set_epoch(epoch)
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    valid_batches = 0
    current_head_lr = optimizer.param_groups[0]["lr"]

    for step_in_epoch, batch in enumerate(train_loader):
        if step_in_epoch % GRAD_ACCUM_STEPS == 0:
            current_head_lr = set_step_learning_rates(
                optimizer, global_opt_step, total_opt_steps
            )

        images = batch["image"].to(DEVICE, non_blocking=True)
        features = batch["features"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        center_idx = (
            batch["center_idx"].to(DEVICE, non_blocking=True)
            if "center_idx" in batch
            else None
        )

        with torch.amp.autocast(
            device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP
        ):
            logits, v_embed = model(images, features, return_aux=True)
            raw_loss = criterion(
                logits,
                labels,
                center_ids=center_idx,
                v_embed=v_embed,
            )
            scaled_loss = raw_loss / float(GRAD_ACCUM_STEPS)

        if torch.isfinite(raw_loss):
            if USE_GRAD_SCALER:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            running_loss += float(raw_loss.item())
            valid_batches += 1

        is_step_boundary = ((step_in_epoch + 1) % GRAD_ACCUM_STEPS == 0) or (
            (step_in_epoch + 1) == len(train_loader)
        )
        if is_step_boundary:
            if USE_GRAD_SCALER:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=MAX_GRAD_NORM
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=MAX_GRAD_NORM
                )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_opt_step += 1
            update_ema_weights(ema_model, model, global_opt_step)

    train_loss_epoch = running_loss / float(max(1, valid_batches))
    criterion.eval()

    raw_val_probs = run_inference(model, val_loader, use_tta=True, tta_views=4)
    ema_val_probs = run_inference(ema_model, val_loader, use_tta=True, tta_views=4)

    raw_metrics = binary_metrics(y_val, raw_val_probs)
    ema_metrics = binary_metrics(y_val, ema_val_probs)

    if (ema_metrics["balanced_accuracy"] > raw_metrics["balanced_accuracy"] + 1e-6) or (
        abs(ema_metrics["balanced_accuracy"] - raw_metrics["balanced_accuracy"]) <= 1e-6
        and ema_metrics["auroc"] >= raw_metrics["auroc"]
    ):
        cand_metrics = ema_metrics
        cand_state = ema_model.state_dict()
        cand_tag = "ema"
    else:
        cand_metrics = raw_metrics
        cand_state = model.state_dict()
        cand_tag = "raw"

    cand_ba = cand_metrics["balanced_accuracy"]
    cand_auc = cand_metrics["auroc"]

    improved = (cand_ba > best_val_ba + 1e-5) or (
        abs(cand_ba - best_val_ba) <= 1e-5 and cand_auc > best_val_auc + 1e-5
    )
    if improved:
        best_val_ba = cand_ba
        best_val_auc = cand_auc
        best_epoch = epoch
        best_source = cand_tag
        epochs_without_improvement = 0
        torch.save(
            {
                "model_state_dict": copy.deepcopy(cand_state),
                "epoch": epoch,
                "source": cand_tag,
                "val_balanced_accuracy": best_val_ba,
                "val_auroc": best_val_auc,
            },
            BEST_MODEL_PATH,
        )
    else:
        epochs_without_improvement += 1

    ep_sec = time.time() - ep_t0
    epoch_durations.append(ep_sec)
    mark = "*" if improved else ""

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | loss={train_loss_epoch:.4f} | "
        f"val_ba={cand_ba:.5f} (raw={raw_metrics['balanced_accuracy']:.4f}, ema={ema_metrics['balanced_accuracy']:.4f}) | "
        f"val_auc={cand_auc:.5f} | lr={current_head_lr:.2e} | {ep_sec:.1f}s {mark}"
    )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping triggered at epoch {epoch} (best epoch={best_epoch}, {best_source})."
        )
        break

    elapsed_total = time.time() - STEP_START_TIME
    avg_ep_sec = float(np.mean(epoch_durations))
    reserved_tta_sec = max(360.0, 3.2 * avg_ep_sec)
    if elapsed_total + avg_ep_sec + reserved_tta_sec > MAX_STEP_SECONDS:
        print(
            f"Time-budget guard reached after epoch {epoch} ({elapsed_total:.0f}s elapsed); proceeding to final TTA inference."
        )
        break

# Load best checkpoint & run identical 8-view D4-TTA inference on validation & test
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

final_val_probs = run_inference(
    model,
    val_loader,
    device=DEVICE,
    use_amp=USE_AMP,
    amp_dtype=AMP_DTYPE,
    use_tta=True,
    tta_views=FINAL_TTA_VIEWS,
)
final_val_metrics = binary_metrics(y_val, final_val_probs)
validation_balanced_accuracy = float(final_val_metrics["balanced_accuracy"])
validation_auroc = float(final_val_metrics["auroc"])

val_preds_bin = (final_val_probs >= 0.5).astype(int)
tumor_recall = float(val_preds_bin[y_val == 1].mean())
normal_recall = float((val_preds_bin[y_val == 0] == 0).mean())
center_audit_parts = []
for cid, grp in val_df.groupby("center_id"):
    idx = grp.index.values
    c_m = binary_metrics(y_val[idx], final_val_probs[idx])
    center_audit_parts.append(
        f"center_{cid}: ba={c_m['balanced_accuracy']:.4f}, auc={c_m['auroc']:.4f}"
    )

print(
    f"Best checkpoint (epoch={checkpoint.get('epoch')}, src={checkpoint.get('source')}) D4-TTA({FINAL_TTA_VIEWS}) Audit | "
    f"val_ba={validation_balanced_accuracy:.6f}, val_auroc={validation_auroc:.6f}, "
    f"tpr={tumor_recall:.4f}, tnr={normal_recall:.4f} | "
    + " | ".join(center_audit_parts)
)

val_pred_df = pd.DataFrame(
    {
        "sample_id": val_df["sample_id"].values,
        "label": y_val,
        "center_id": val_df["center_id"].values,
        "slide_id": val_df["slide_id"].values,
        "tumor_probability": final_val_probs,
    }
)
val_pred_df.to_csv(os.path.join(WORKING_DIR, "val_predictions.csv"), index=False)

test_probs = run_inference(
    model,
    test_loader,
    device=DEVICE,
    use_amp=USE_AMP,
    amp_dtype=AMP_DTYPE,
    use_tta=True,
    tta_views=FINAL_TTA_VIEWS,
)
np.save(os.path.join(WORKING_DIR, "test_predictions.npy"), test_probs)

submission_df = pd.DataFrame(
    {
        "sample_id": test_df["sample_id"].values,
        "tumor_probability": test_probs.astype(np.float64),
    }
)

assert list(submission_df.columns) == [
    "sample_id",
    "tumor_probability",
], "Invalid submission columns!"
assert len(submission_df) == len(sample_sub_df), "Submission row count mismatch!"
assert (
    submission_df["sample_id"].values == sample_sub_df["sample_id"].values
).all(), "Sample ID order mismatch!"
assert np.isfinite(
    submission_df["tumor_probability"].values
).all(), "Non-finite probabilities detected!"
assert (
    (submission_df["tumor_probability"].values >= 0.0)
    & (submission_df["tumor_probability"].values <= 1.0)
).all(), "Probabilities outside [0, 1]!"
assert (
    float(submission_df["tumor_probability"].std()) > 1e-4
), "Degenerate constant predictions detected!"

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
assert (
    os.path.isfile(submission_path) and os.path.getsize(submission_path) > 0
), "Submission file missing or empty!"

print(
    f"Saved submission to {submission_path} | rows={len(submission_df)}, "
    f"prob_mean={submission_df['tumor_probability'].mean():.4f}, "
    f"prob_std={submission_df['tumor_probability'].std():.4f}, "
    f"pred_pos_rate={(submission_df['tumor_probability'] >= 0.5).mean():.4f}"
)
print(f"Final Validation Score: {validation_balanced_accuracy}")
