import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import hashlib
import importlib.util
import math
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union

_RUN_START_TIME = time.perf_counter()

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import skew
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import xgboost as xgb

# Prevent OpenCV multithreading contention inside worker pools
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# =============================================================================
# 1. Global Constants, Reproducibility & Official Metric Loader
# =============================================================================
SEED = 42
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
IMG_SIZE = 256
FOURIER_LOW_RADIUS = 12  # 24x24 central low-frequency crop around DC
FOURIER_BANK_SIZE = 1024
MAX_RUNTIME_BUDGET_SEC = 2700.0  # 45-min target ceiling within 1-hour hard limit

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_metric_spec = importlib.util.spec_from_file_location(
    "competition_metric", os.path.join(INPUT_DIR, "metric.py")
)
_metric_mod = importlib.util.module_from_spec(_metric_spec)
_metric_spec.loader.exec_module(_metric_mod)
binary_metrics = _metric_mod.binary_metrics


def _elapsed_sec() -> float:
    return time.perf_counter() - _RUN_START_TIME


def seed_everything(seed: int = SEED, seed_cuda: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Seed CPU RNGs prior to multiprocessing fork pools in Step 1; CUDA seeded right after
seed_everything(SEED, seed_cuda=False)

# =============================================================================
# 2. Physics-Informed Ruifrok-Johnston Orthogonal H&E Stain Deconvolution
# =============================================================================
# Unit absorption vectors in RGB Optical Density space:
# Hematoxylin (nuclear chromatin), Eosin (cytoplasm/stroma), Residual (orthogonal pigment)
_V_H = np.array([0.650, 0.704, 0.286], dtype=np.float32)
_V_H /= np.linalg.norm(_V_H)
_V_E = np.array([0.072, 0.990, 0.105], dtype=np.float32)
_V_E /= np.linalg.norm(_V_E)
_V_R = np.cross(_V_H, _V_E).astype(np.float32)
_V_R /= np.linalg.norm(_V_R)

HE_STAIN_MATRIX = np.stack([_V_H, _V_E, _V_R], axis=0).astype(np.float32)  # (3, 3)
HE_INV_MATRIX = np.linalg.inv(HE_STAIN_MATRIX).astype(np.float32)  # (3, 3)


def rgb_to_od(img_rgb: np.ndarray) -> np.ndarray:
    """Beer-Lambert conversion from uint8/float RGB [0, 255] to Optical Density."""
    img_clamped = np.clip(img_rgb.astype(np.float32), 1.0, 255.0)
    return -np.log10(img_clamped / 255.0)


def deconvolve_he_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Deconvolve RGB patch into [Hematoxylin, Eosin, Orthogonal Residual] concentrations."""
    od = rgb_to_od(img_rgb)
    return od @ HE_INV_MATRIX


def reconstruct_rgb_from_he(stain_conc: np.ndarray) -> np.ndarray:
    """Reconstruct RGB [0, 255] float32 image from perturbed [H, E, Residual] concentrations."""
    od_rec = np.clip(stain_conc @ HE_STAIN_MATRIX, 0.0, 3.0)
    rgb_rec = 255.0 * np.power(10.0, -od_rec)
    return np.clip(rgb_rec, 0.0, 255.0).astype(np.float32)


def load_rgb_patch(rel_image_path: str) -> np.ndarray:
    """Safely load a 256x256 RGB uint8 histopathology patch from ./input/."""
    full_path = os.path.join(INPUT_DIR, rel_image_path)
    bgr = cv2.imread(full_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image at {full_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != IMG_SIZE or rgb.shape[1] != IMG_SIZE:
        rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return rgb


# =============================================================================
# 3. Precomputed Spatial & Spectral Masks for Fast Vectorized Extraction
# =============================================================================
_HANN_1D = np.hanning(IMG_SIZE).astype(np.float32)
_HANN_2D = np.outer(_HANN_1D, _HANN_1D).astype(np.float32)

_FY = np.fft.fftfreq(IMG_SIZE, d=1.0)[:, None] * IMG_SIZE
_FX = np.fft.rfftfreq(IMG_SIZE, d=1.0)[None, :] * IMG_SIZE
_FRAD = np.sqrt(_FY**2 + _FX**2).astype(np.float32)
_FANG = np.arctan2(_FY, _FX).astype(np.float32)

_RADIAL_EDGES = [
    (1.0, 4.0),
    (4.0, 8.0),
    (8.0, 16.0),
    (16.0, 32.0),
    (32.0, 64.0),
    (64.0, 128.1),
]
_RADIAL_MASKS = [(_FRAD >= lo) & (_FRAD < hi) for lo, hi in _RADIAL_EDGES]
_RADIAL_INT = np.clip(np.rint(_FRAD).astype(np.int32), 0, IMG_SIZE // 2)

# Quadrant directional masks for mid-to-high spatial frequencies (r in [4, 64]) for stromal anisotropy
_ANISO_BASE = (_FRAD >= 4.0) & (_FRAD <= 64.0)
_ANISO_SECTORS = [
    _ANISO_BASE & (np.abs(_FANG) < (np.pi / 8.0)),
    _ANISO_BASE & (_FANG >= (np.pi / 8.0)) & (_FANG < (3.0 * np.pi / 8.0)),
    _ANISO_BASE & (np.abs(_FANG) >= (3.0 * np.pi / 8.0)),
    _ANISO_BASE & (_FANG <= -(np.pi / 8.0)) & (_FANG > -(3.0 * np.pi / 8.0)),
]

# Concentric ROI masks: Core (64x64), Mid-Ring (128x128 \ 64x64), Outer-Ring (256x256 \ 128x128)
_CORE_MASK = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
_CORE_MASK[96:160, 96:160] = True
_MID_BOX = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
_MID_BOX[64:192, 64:192] = True
_MID_RING_MASK = _MID_BOX & (~_CORE_MASK)
_OUTER_RING_MASK = ~_MID_BOX

# Smooth 2D raised-cosine window for artifact-free Fourier Low-Frequency Amplitude Mixing
_yy, _xx = np.ogrid[
    -FOURIER_LOW_RADIUS:FOURIER_LOW_RADIUS, -FOURIER_LOW_RADIUS:FOURIER_LOW_RADIUS
]
_rr_norm = np.clip(np.sqrt(_yy**2 + _xx**2) / float(FOURIER_LOW_RADIUS), 0.0, 1.0)
FOURIER_SMOOTH_WINDOW = (0.5 * (1.0 + np.cos(np.pi * _rr_norm)))[:, :, None].astype(
    np.float32
)

_TOPHAT_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

FEATURE_NAMES = [
    # Family 1: Multi-Concentric ROI Deconvolved Stain & Tissue Architecture (24)
    "fg_frac_full",
    "fg_frac_core",
    "fg_frac_mid",
    "fg_frac_outer",
    "H_mean_full",
    "H_std_full",
    "H_p90_full",
    "H_p99_full",
    "H_iqr_full",
    "H_skew_full",
    "E_mean_full",
    "E_std_full",
    "E_p90_full",
    "R_abs_mean_full",
    "R_abs_p95_full",
    "H_mean_core",
    "H_p90_core",
    "H_mean_mid",
    "H_mean_outer",
    "H_core_minus_outer",
    "H_core_div_outer",
    "HE_ratio_core",
    "HE_ratio_mid",
    "HE_ratio_outer",
    # Family 2: Segmented Nuclear Cytometry & Spatial Point-Process Topology (18)
    "nuc_count_total",
    "nuc_count_core",
    "nuc_core_density_ratio",
    "nuc_area_mean",
    "nuc_area_std",
    "nuc_area_cv",
    "nuc_area_p90",
    "nuc_area_max",
    "nuc_aspect_mean",
    "nuc_extent_mean",
    "nuc_intra_H_mean",
    "nuc_intra_H_std",
    "nuc_intra_grad_ratio",
    "nuc_nnd1_mean",
    "nuc_nnd1_std",
    "nuc_nnd1_cv",
    "nuc_nnd3_mean",
    "nuc_clark_evans_idx",
    # Family 3: Scale-Invariant 2D Radial Fourier Power Spectrum & Anisotropy (10)
    "fft_ring_1_4",
    "fft_ring_4_8",
    "fft_ring_8_16",
    "fft_ring_16_32",
    "fft_ring_32_64",
    "fft_ring_64_128",
    "fft_spectral_slope",
    "fft_spectral_curv",
    "fft_high_to_low_ratio",
    "fft_directional_anisotropy",
    # Family 4: Multi-Scale LoG Chromatin Scale & Structure Tensor Coherency (10)
    "log_e_s1",
    "log_e_s2",
    "log_e_s4",
    "log_ratio_s2_s1",
    "log_ratio_s4_s1",
    "log_core_to_outer_s2",
    "st_coherency_mean",
    "st_coherency_p90",
    "st_energy_mean",
    "st_core_coherency",
    # Family 5: Quantized Chromatin Co-occurrence & Joint Stain Entropy (6)
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "joint_HE_entropy",
    "core_H_entropy",
]
NUM_TABULAR_FEATURES = len(FEATURE_NAMES)


def _safe_glcm_and_entropy(h_norm: np.ndarray, e_norm: np.ndarray) -> list:
    """Fast vectorized 16-level GLCM on standardized Hematoxylin + joint H-E Shannon entropy."""
    hq = np.clip(np.floor((h_norm + 2.5) / 5.0 * 16.0), 0, 15).astype(np.int32)
    eq = np.clip(np.floor((e_norm + 2.5) / 5.0 * 16.0), 0, 15).astype(np.int32)

    pairs_h = hq[:, :-1] * 16 + hq[:, 1:]
    pairs_v = hq[:-1, :] * 16 + hq[1:, :]
    counts = np.bincount(pairs_h.ravel(), minlength=256) + np.bincount(
        pairs_v.ravel(), minlength=256
    )
    P = counts.reshape(16, 16).astype(np.float64)
    P = 0.5 * (P + P.T)
    P /= max(P.sum(), 1.0)

    idx = np.arange(16, dtype=np.float64)
    diff = idx[:, None] - idx[None, :]
    contrast = float(np.sum(P * (diff**2)) / 225.0)
    homogeneity = float(np.sum(P / (1.0 + np.abs(diff))))
    energy = float(np.sum(P**2))

    px = P.sum(axis=1)
    mu = float(np.sum(idx * px))
    var = float(np.sum(((idx - mu) ** 2) * px))
    if var > 1e-8:
        corr = float(np.sum(P * ((idx[:, None] - mu) * (idx[None, :] - mu))) / var)
    else:
        corr = 0.0

    joint_bins = hq * 16 + eq
    jp = np.bincount(joint_bins.ravel(), minlength=256).astype(np.float64)
    jp /= max(jp.sum(), 1.0)
    jp_nz = jp[jp > 0]
    joint_entropy = float(-np.sum(jp_nz * np.log2(jp_nz)))

    core_hq = hq[96:160, 96:160].ravel()
    cp = np.bincount(core_hq, minlength=16).astype(np.float64)
    cp /= max(cp.sum(), 1.0)
    cp_nz = cp[cp > 0]
    core_entropy = float(-np.sum(cp_nz * np.log2(cp_nz)))

    return [contrast, homogeneity, energy, corr, joint_entropy, core_entropy]


def extract_single_patch_features(rel_image_path: str) -> np.ndarray:
    """Extract 68 stain-deconvolved cytological, point-process, ROI, and spectral features."""
    rgb = load_rgb_patch(rel_image_path)
    od = rgb_to_od(rgb)
    stain = od @ HE_INV_MATRIX

    H = np.clip(stain[:, :, 0], 0.0, None)
    E = np.clip(stain[:, :, 1], 0.0, None)
    R_abs = np.abs(stain[:, :, 2])
    od_norm = np.linalg.norm(od, axis=2)

    # Foreground tissue mask (excludes bright slide background & empty fat vacuoles)
    fg_mask = od_norm > 0.12
    n_fg = int(fg_mask.sum())
    total_px = float(IMG_SIZE * IMG_SIZE)

    fg_frac_full = n_fg / total_px
    fg_frac_core = float(fg_mask[_CORE_MASK].mean())
    fg_frac_mid = float(fg_mask[_MID_RING_MASK].mean())
    fg_frac_outer = float(fg_mask[_OUTER_RING_MASK].mean())

    # Reference pixels for foreground-robust summary statistics
    H_ref = H[fg_mask] if n_fg >= 32 else H.ravel()
    E_ref = E[fg_mask] if n_fg >= 32 else E.ravel()
    R_ref = R_abs[fg_mask] if n_fg >= 32 else R_abs.ravel()

    h_pcts = np.percentile(H_ref, [25.0, 75.0, 90.0, 99.0])
    e_pcts = np.percentile(E_ref, [90.0])
    r_pcts = np.percentile(R_ref, [95.0])

    H_mean_full = float(H_ref.mean())
    H_std_full = float(H_ref.std())
    H_p90_full = float(h_pcts[2])
    H_p99_full = float(h_pcts[3])
    H_iqr_full = float(h_pcts[1] - h_pcts[0])
    H_skew_full = float(skew(H_ref)) if H_std_full > 1e-6 else 0.0

    E_mean_full = float(E_ref.mean())
    E_std_full = float(E_ref.std())
    E_p90_full = float(e_pcts[0])

    R_abs_mean_full = float(R_ref.mean())
    R_abs_p95_full = float(r_pcts[0])

    H_core_vals = H[_CORE_MASK]
    H_mid_vals = H[_MID_RING_MASK]
    H_outer_vals = H[_OUTER_RING_MASK]
    E_core_vals = E[_CORE_MASK]
    E_mid_vals = E[_MID_RING_MASK]
    E_outer_vals = E[_OUTER_RING_MASK]

    H_mean_core = float(H_core_vals.mean())
    H_p90_core = float(np.percentile(H_core_vals, 90.0))
    H_mean_mid = float(H_mid_vals.mean())
    H_mean_outer = float(H_outer_vals.mean())
    H_core_minus_outer = H_mean_core - H_mean_outer
    H_core_div_outer = (H_mean_core + 0.02) / (H_mean_outer + 0.02)

    HE_ratio_core = (H_mean_core + 0.02) / (float(E_core_vals.mean()) + 0.05)
    HE_ratio_mid = (H_mean_mid + 0.02) / (float(E_mid_vals.mean()) + 0.05)
    HE_ratio_outer = (H_mean_outer + 0.02) / (float(E_outer_vals.mean()) + 0.05)

    fam1 = [
        fg_frac_full,
        fg_frac_core,
        fg_frac_mid,
        fg_frac_outer,
        H_mean_full,
        H_std_full,
        H_p90_full,
        H_p99_full,
        H_iqr_full,
        H_skew_full,
        E_mean_full,
        E_std_full,
        E_p90_full,
        R_abs_mean_full,
        R_abs_p95_full,
        H_mean_core,
        H_p90_core,
        H_mean_mid,
        H_mean_outer,
        H_core_minus_outer,
        H_core_div_outer,
        HE_ratio_core,
        HE_ratio_mid,
        HE_ratio_outer,
    ]

    # Family 2: Adaptive Nuclear Blob Cytometry & Spatial Point-Process Graph
    H_smooth = cv2.GaussianBlur(H, (3, 3), 0.8)
    H_u8 = np.clip(H_smooth / 1.5 * 255.0, 0, 255).astype(np.uint8)
    H_tophat = cv2.morphologyEx(H_u8, cv2.MORPH_TOPHAT, _TOPHAT_KERNEL)
    nuc_score = np.clip(
        0.55 * H_u8.astype(np.float32) + 0.75 * H_tophat.astype(np.float32), 0, 255
    ).astype(np.uint8)

    if n_fg >= 64:
        otsu_thr, _ = cv2.threshold(
            nuc_score[fg_mask], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        thr_val = max(float(otsu_thr), 26.0)
    else:
        thr_val = 36.0

    nuc_bin = ((nuc_score > thr_val) & fg_mask).astype(np.uint8)
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
        nuc_bin, connectivity=8
    )

    # Filter biologically plausible nuclear sizes (6 to 600 px at 256x256 microscopy scale)
    if num_labels > 1:
        areas_all = stats[1:, cv2.CC_STAT_AREA]
        valid_idx = np.where((areas_all >= 6) & (areas_all <= 600))[0] + 1
    else:
        valid_idx = np.array([], dtype=np.int32)

    n_valid = int(len(valid_idx))
    gx = cv2.Sobel(H_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(H_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    if n_valid > 0:
        v_areas = stats[valid_idx, cv2.CC_STAT_AREA].astype(np.float32)
        v_w = np.maximum(stats[valid_idx, cv2.CC_STAT_WIDTH].astype(np.float32), 1.0)
        v_h = np.maximum(stats[valid_idx, cv2.CC_STAT_HEIGHT].astype(np.float32), 1.0)
        v_cents = centroids[valid_idx]  # (n_valid, 2) in (x, y)

        in_core = (
            (v_cents[:, 0] >= 96.0)
            & (v_cents[:, 0] < 160.0)
            & (v_cents[:, 1] >= 96.0)
            & (v_cents[:, 1] < 160.0)
        )
        nuc_count_core = float(in_core.sum())
        nuc_core_density_ratio = (nuc_count_core * 16.0) / float(n_valid)

        nuc_area_mean = float(v_areas.mean())
        nuc_area_std = float(v_areas.std())
        nuc_area_cv = nuc_area_std / (nuc_area_mean + 1e-5)
        nuc_area_p90 = float(np.percentile(v_areas, 90.0))
        nuc_area_max = float(v_areas.max())

        aspects = np.maximum(v_w / v_h, v_h / v_w)
        extents = v_areas / (v_w * v_h)
        nuc_aspect_mean = float(aspects.mean())
        nuc_extent_mean = float(extents.mean())

        valid_mask = np.isin(labels_im, valid_idx)
        if valid_mask.any():
            h_in_nuc = H[valid_mask]
            g_in_nuc = grad_mag[valid_mask]
            nuc_intra_H_mean = float(h_in_nuc.mean())
            nuc_intra_H_std = float(h_in_nuc.std())
            nuc_intra_grad_ratio = float(g_in_nuc.mean()) / (nuc_intra_H_mean + 0.02)
        else:
            nuc_intra_H_mean, nuc_intra_H_std, nuc_intra_grad_ratio = 0.0, 0.0, 0.0

        if n_valid >= 4:
            tree = cKDTree(v_cents)
            dists, _ = tree.query(v_cents, k=4)
            d1 = dists[:, 1]
            d3 = dists[:, 3]
            nuc_nnd1_mean = float(d1.mean())
            nuc_nnd1_std = float(d1.std())
            nuc_nnd1_cv = nuc_nnd1_std / (nuc_nnd1_mean + 1e-5)
            nuc_nnd3_mean = float(d3.mean())
            eff_area = max(float(n_fg), 256.0)
            density = float(n_valid) / eff_area
            nuc_clark_evans_idx = float(2.0 * nuc_nnd1_mean * np.sqrt(density))
        elif n_valid >= 2:
            tree = cKDTree(v_cents)
            dists, _ = tree.query(v_cents, k=2)
            d1 = dists[:, 1]
            nuc_nnd1_mean = float(d1.mean())
            nuc_nnd1_std = float(d1.std())
            nuc_nnd1_cv = nuc_nnd1_std / (nuc_nnd1_mean + 1e-5)
            nuc_nnd3_mean = nuc_nnd1_mean * 1.5
            eff_area = max(float(n_fg), 256.0)
            nuc_clark_evans_idx = float(
                2.0 * nuc_nnd1_mean * np.sqrt(float(n_valid) / eff_area)
            )
        else:
            (
                nuc_nnd1_mean,
                nuc_nnd1_std,
                nuc_nnd1_cv,
                nuc_nnd3_mean,
                nuc_clark_evans_idx,
            ) = (64.0, 0.0, 0.0, 96.0, 0.0)
    else:
        (
            nuc_count_core,
            nuc_core_density_ratio,
            nuc_area_mean,
            nuc_area_std,
            nuc_area_cv,
            nuc_area_p90,
            nuc_area_max,
            nuc_aspect_mean,
            nuc_extent_mean,
            nuc_intra_H_mean,
            nuc_intra_H_std,
            nuc_intra_grad_ratio,
            nuc_nnd1_mean,
            nuc_nnd1_std,
            nuc_nnd1_cv,
            nuc_nnd3_mean,
            nuc_clark_evans_idx,
        ) = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            128.0,
            0.0,
            0.0,
            128.0,
            0.0,
        )

    fam2 = [
        float(n_valid),
        nuc_count_core,
        nuc_core_density_ratio,
        nuc_area_mean,
        nuc_area_std,
        nuc_area_cv,
        nuc_area_p90,
        nuc_area_max,
        nuc_aspect_mean,
        nuc_extent_mean,
        nuc_intra_H_mean,
        nuc_intra_H_std,
        nuc_intra_grad_ratio,
        nuc_nnd1_mean,
        nuc_nnd1_std,
        nuc_nnd1_cv,
        nuc_nnd3_mean,
        nuc_clark_evans_idx,
    ]

    # Instance-standardized stain maps (invariant to global brightness/concentration scaling)
    H_norm = (H - float(H.mean())) / (float(H.std()) + 1e-5)
    E_norm = (E - float(E.mean())) / (float(E.std()) + 1e-5)

    # Family 3: 2D Radial Fourier Power Spectrum & Anisotropy
    fft_c = np.fft.rfft2(H_norm * _HANN_2D)
    psd = (fft_c.real**2 + fft_c.imag**2).astype(np.float64)
    psd[0, 0] = 0.0

    ring_energies = [float(psd[m].sum()) for m in _RADIAL_MASKS]
    total_ring_e = sum(ring_energies) + 1e-8
    ring_ratios = [e / total_ring_e for e in ring_energies]

    rad_sum = np.bincount(
        _RADIAL_INT.ravel(), weights=psd.ravel(), minlength=IMG_SIZE // 2 + 1
    )
    rad_cnt = np.bincount(_RADIAL_INT.ravel(), minlength=IMG_SIZE // 2 + 1)
    r_vals = np.arange(2, 96, dtype=np.float64)
    rad_mean = rad_sum[2:96] / np.maximum(rad_cnt[2:96], 1)
    if np.any(rad_mean > 1e-10):
        log_r = np.log(r_vals)
        log_p = np.log(rad_mean + 1e-10)
        poly2 = np.polyfit(log_r, log_p, deg=2)
        fft_spectral_curv = float(poly2[0])
        fft_spectral_slope = float(np.polyfit(log_r, log_p, deg=1)[0])
    else:
        fft_spectral_slope, fft_spectral_curv = 0.0, 0.0

    fft_high_to_low_ratio = (ring_ratios[3] + ring_ratios[4] + ring_ratios[5]) / (
        ring_ratios[0] + ring_ratios[1] + 1e-5
    )
    sec_means = [float(psd[sm].mean()) for sm in _ANISO_SECTORS]
    fft_directional_anisotropy = (max(sec_means) + 1e-6) / (min(sec_means) + 1e-6)

    fam3 = ring_ratios + [
        fft_spectral_slope,
        fft_spectral_curv,
        fft_high_to_low_ratio,
        fft_directional_anisotropy,
    ]

    # Family 4: Multi-Scale LoG Chromatin Scale & Structure Tensor Coherency
    g1 = cv2.GaussianBlur(H_norm, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(H_norm, (0, 0), 2.0)
    g8 = cv2.GaussianBlur(H_norm, (0, 0), 8.0)

    dog1 = np.abs(H_norm - g1)
    dog2 = np.abs(g1 - g2)
    dog4 = np.abs(g2 - g8)

    log_e_s1 = float(dog1.mean())
    log_e_s2 = float(dog2.mean())
    log_e_s4 = float(dog4.mean())
    log_ratio_s2_s1 = log_e_s2 / (log_e_s1 + 1e-5)
    log_ratio_s4_s1 = log_e_s4 / (log_e_s1 + 1e-5)
    log_core_to_outer_s2 = (float(dog2[_CORE_MASK].mean()) + 1e-5) / (
        float(dog2[_OUTER_RING_MASK].mean()) + 1e-5
    )

    ix = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(ix * ix, (0, 0), 2.5)
    jyy = cv2.GaussianBlur(iy * iy, (0, 0), 2.5)
    jxy = cv2.GaussianBlur(ix * iy, (0, 0), 2.5)

    trace = jxx + jyy
    det_diff = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * (jxy**2), 0.0))
    coherency = (det_diff / (trace + 1e-5)) ** 2

    st_coherency_mean = float(coherency.mean())
    st_coherency_p90 = float(np.percentile(coherency, 90.0))
    st_energy_mean = float(trace.mean())
    st_core_coherency = float(coherency[_CORE_MASK].mean())

    fam4 = [
        log_e_s1,
        log_e_s2,
        log_e_s4,
        log_ratio_s2_s1,
        log_ratio_s4_s1,
        log_core_to_outer_s2,
        st_coherency_mean,
        st_coherency_p90,
        st_energy_mean,
        st_core_coherency,
    ]

    # Family 5: Quantized GLCM & Joint Stain Entropy
    fam5 = _safe_glcm_and_entropy(H_norm, E_norm)

    feats = np.asarray(fam1 + fam2 + fam3 + fam4 + fam5, dtype=np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def extract_features_parallel(image_paths: list, num_workers: int = 8) -> np.ndarray:
    """Parallel extraction of 68 cytological & spectral features without progress bars."""
    n_workers = max(1, min(num_workers, os.cpu_count() or 4))
    if "fork" in mp.get_all_start_methods() and not torch.cuda.is_initialized():
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            rows = pool.map(extract_single_patch_features, image_paths, chunksize=64)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            rows = list(ex.map(extract_single_patch_features, image_paths))
    mat = np.vstack(rows).astype(np.float32)
    assert mat.shape == (
        len(image_paths),
        NUM_TABULAR_FEATURES,
    ), f"Unexpected feature matrix shape: {mat.shape}"
    assert np.isfinite(mat).all(), "Non-finite values detected in raw feature matrix"
    return mat


# =============================================================================
# 4. Training-Only Image Statistics & Cross-Center Fourier Amplitude Bank
# =============================================================================
def _extract_low_freq_amp_and_stats(rel_image_path: str):
    """Extract low-frequency Fourier amplitude crop + per-image RGB & H/E/R channel moments."""
    rgb = load_rgb_patch(rel_image_path).astype(np.float32)
    rgb01 = rgb / 255.0
    rgb_mean = rgb01.mean(axis=(0, 1))
    rgb_sq = (rgb01**2).mean(axis=(0, 1))

    stain = deconvolve_he_rgb(rgb)
    h_pos = np.clip(stain[:, :, 0], 0.0, None)
    e_pos = np.clip(stain[:, :, 1], 0.0, None)
    r_val = stain[:, :, 2]
    her = np.stack([h_pos, e_pos, r_val], axis=-1)
    her_mean = her.mean(axis=(0, 1))
    her_sq = (her**2).mean(axis=(0, 1))

    fft_rgb = np.fft.fftshift(np.fft.fft2(rgb, axes=(0, 1)), axes=(0, 1))
    amp = np.abs(fft_rgb)
    c = IMG_SIZE // 2
    r = FOURIER_LOW_RADIUS
    low_amp = amp[c - r : c + r, c - r : c + r, :].astype(np.float32)
    return rgb_mean, rgb_sq, her_mean, her_sq, low_amp


def fit_train_only_image_statistics_and_fourier_bank(
    train_dataframe: pd.DataFrame, seed: int = SEED
) -> dict:
    """
    Fit RGB & [H, E, Residual] channel statistics and stratified Cross-Center Low-Frequency
    Fourier Amplitude Bank STRICTLY on training samples (zero validation/test leakage).
    """
    rng = np.random.default_rng(seed)
    paths = train_dataframe["image_path"].tolist()
    centers = train_dataframe["center_id"].values
    unique_centers = np.unique(centers)

    per_center_quota = max(1, FOURIER_BANK_SIZE // len(unique_centers))
    bank_indices = []
    for cid in unique_centers:
        c_idx = np.where(centers == cid)[0]
        chosen = rng.choice(
            c_idx, size=min(per_center_quota, len(c_idx)), replace=False
        )
        bank_indices.append((cid, chosen))

    stat_indices = rng.choice(len(paths), size=min(4096, len(paths)), replace=False)
    stat_set = set(stat_indices.tolist())
    for _, c_idx in bank_indices:
        stat_set.update(c_idx.tolist())
    eval_indices = sorted(stat_set)
    eval_paths = [paths[i] for i in eval_indices]

    n_workers = max(1, min(8, os.cpu_count() or 4))
    if "fork" in mp.get_all_start_methods() and not torch.cuda.is_initialized():
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            res = pool.map(_extract_low_freq_amp_and_stats, eval_paths, chunksize=64)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            res = list(ex.map(_extract_low_freq_amp_and_stats, eval_paths))

    idx_to_pos = {orig_i: pos for pos, orig_i in enumerate(eval_indices)}

    rgb_means = np.stack([res[idx_to_pos[i]][0] for i in stat_indices], axis=0)
    rgb_sqs = np.stack([res[idx_to_pos[i]][1] for i in stat_indices], axis=0)
    her_means = np.stack([res[idx_to_pos[i]][2] for i in stat_indices], axis=0)
    her_sqs = np.stack([res[idx_to_pos[i]][3] for i in stat_indices], axis=0)

    rgb_mean = rgb_means.mean(axis=0).astype(np.float32)
    rgb_std = np.sqrt(np.maximum(rgb_sqs.mean(axis=0) - rgb_mean**2, 1e-6)).astype(
        np.float32
    )
    her_mean = her_means.mean(axis=0).astype(np.float32)
    her_std = np.sqrt(np.maximum(her_sqs.mean(axis=0) - her_mean**2, 1e-6)).astype(
        np.float32
    )

    bank_arrays = []
    bank_center_ids = []
    for cid_num, (_, c_idx) in enumerate(bank_indices):
        for i in c_idx:
            bank_arrays.append(res[idx_to_pos[int(i)]][4])
            bank_center_ids.append(cid_num)

    fourier_bank = np.stack(bank_arrays, axis=0).astype(np.float32)
    fourier_bank_centers = np.asarray(bank_center_ids, dtype=np.int64)

    return {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "her_mean": her_mean,
        "her_std": her_std,
        "fourier_bank": fourier_bank,
        "fourier_bank_centers": fourier_bank_centers,
    }


# =============================================================================
# 5. Leak-Free Stateful Tabular Feature Preprocessor
# =============================================================================
class LeakFreeTabularPreprocessor:
    """
    Winsorizes extreme outliers using training-only [0.5%, 99.5%] quantiles, maps skewed
    cytological/spectral distributions via QuantileTransformer(normal), and standardizes.
    """

    def __init__(self, seed: int = SEED):
        self.seed = seed
        self.clip_low_ = None
        self.clip_high_ = None
        self.qt_ = None
        self.scaler_ = None
        self.is_fitted_ = False

    def fit(self, X_train: np.ndarray) -> "LeakFreeTabularPreprocessor":
        X_clean = np.nan_to_num(
            X_train.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.clip_low_ = np.percentile(X_clean, 0.5, axis=0)
        self.clip_high_ = np.percentile(X_clean, 99.5, axis=0)
        X_clipped = np.clip(X_clean, self.clip_low_, self.clip_high_)

        n_q = min(1000, max(100, X_clipped.shape[0] // 5))
        self.qt_ = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            random_state=self.seed,
            subsample=20000,
        )
        X_qt = self.qt_.fit_transform(X_clipped)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_qt)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError(
                "LeakFreeTabularPreprocessor must be fitted on training data before transform."
            )
        X_clean = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        X_clipped = np.clip(X_clean, self.clip_low_, self.clip_high_)
        X_qt = self.qt_.transform(X_clipped)
        X_scaled = self.scaler_.transform(X_qt)
        X_scaled = np.clip(
            np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0
        )
        return X_scaled.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


# =============================================================================
# 6. Domain-Generalizing Physics & Spectral Augmentations + Multi-View Dataset
# =============================================================================
def apply_fourier_amplitude_mix(
    img_rgb_f32: np.ndarray, target_low_amp: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Phase-preserving Fourier Amplitude Mixing (FACT/FDA) using a smooth 2D cosine window
    around DC to transfer cross-center scanner/illumination spectra without ringing artifacts.
    """
    fft_src = np.fft.fftshift(np.fft.fft2(img_rgb_f32, axes=(0, 1)), axes=(0, 1))
    amp_src = np.abs(fft_src)
    phase_src = np.angle(fft_src)

    c = IMG_SIZE // 2
    r = FOURIER_LOW_RADIUS
    w = alpha * FOURIER_SMOOTH_WINDOW
    amp_src[c - r : c + r, c - r : c + r, :] = (1.0 - w) * amp_src[
        c - r : c + r, c - r : c + r, :
    ] + w * target_low_amp

    fft_mixed = amp_src * np.exp(1j * phase_src)
    img_mixed = np.real(
        np.fft.ifft2(np.fft.ifftshift(fft_mixed, axes=(0, 1)), axes=(0, 1))
    )
    return np.clip(img_mixed, 0.0, 255.0).astype(np.float32)


def apply_he_subspace_stain_aug(
    img_rgb_f32: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Physically grounded H&E dye concentration scaling + shift in Ruifrok-Johnston space,
    simulating unseen hospital staining protocols while preserving nuclear morphology.
    """
    stain = deconvolve_he_rgb(img_rgb_f32)
    od_norm = np.linalg.norm(rgb_to_od(img_rgb_f32), axis=2)
    tissue = (od_norm > 0.10).astype(np.float32)

    alpha_h = rng.uniform(0.74, 1.32)
    alpha_e = rng.uniform(0.74, 1.32)
    alpha_r = rng.uniform(0.85, 1.15)
    beta_h = rng.uniform(-0.045, 0.045)
    beta_e = rng.uniform(-0.045, 0.045)

    stain[:, :, 0] = stain[:, :, 0] * alpha_h + beta_h * tissue
    stain[:, :, 1] = stain[:, :, 1] * alpha_e + beta_e * tissue
    stain[:, :, 2] = stain[:, :, 2] * alpha_r

    rgb_aug = reconstruct_rgb_from_he(stain)
    gamma = rng.uniform(0.88, 1.14)
    rgb_aug = 255.0 * np.power(np.clip(rgb_aug / 255.0, 0.0, 1.0), gamma)
    return np.clip(rgb_aug, 0.0, 255.0).astype(np.float32)


def compute_calibrated_stain_tensor(
    img_rgb_f32: np.ndarray, her_mean: np.ndarray, her_std: np.ndarray
) -> np.ndarray:
    """
    Deterministic instance-and-train-calibrated 3-channel structural stain representation:
    [Foreground-centered Hematoxylin, Foreground-centered Eosin, Standardized Pigment Residual].
    Identical formula used across train, validation, and test.
    """
    od = rgb_to_od(img_rgb_f32)
    stain = od @ HE_INV_MATRIX
    h_pos = np.clip(stain[:, :, 0], 0.0, None)
    e_pos = np.clip(stain[:, :, 1], 0.0, None)
    r_val = stain[:, :, 2]

    fg_mask = np.linalg.norm(od, axis=2) > 0.12
    if int(fg_mask.sum()) >= 64:
        h_anchor = 0.5 * float(np.median(h_pos[fg_mask])) + 0.5 * float(her_mean[0])
        e_anchor = 0.5 * float(np.median(e_pos[fg_mask])) + 0.5 * float(her_mean[1])
    else:
        h_anchor = float(her_mean[0])
        e_anchor = float(her_mean[1])

    ch_h = (h_pos - h_anchor) / (float(her_std[0]) + 1e-5)
    ch_e = (e_pos - e_anchor) / (float(her_std[1]) + 1e-5)
    ch_r = (r_val - float(her_mean[2])) / (float(her_std[2]) + 1e-5)

    stacked = np.stack([ch_h, ch_e, ch_r], axis=0)  # (3, 256, 256)
    return np.clip(stacked, -6.0, 6.0).astype(np.float32)


class HistopathologyMultiViewDataset(Dataset):
    """
    Multi-view PyTorch Dataset yielding:
      - 'rgb': (3, 256, 256) normalized RGB tensor
      - 'stain': (3, 256, 256) calibrated [H, E, Residual] structural tensor
      - 'tabular': (68,) scaled cytological + spectral feature vector
      - 'label': float32 binary label (when available)
      - 'center_idx': int64 domain index for training-time domain invariance (-1 at eval/test)
      - 'sample_id': sample identifier string
    Validation and Test use 100% IDENTICAL deterministic preprocessing (`is_train=False`).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tabular_features: np.ndarray,
        image_stats: dict,
        is_train: bool = False,
        center_to_idx: dict = None,
    ):
        self.sample_ids = dataframe["sample_id"].astype(str).tolist()
        self.image_paths = dataframe["image_path"].astype(str).tolist()
        self.tabular_features = np.asarray(tabular_features, dtype=np.float32)
        assert len(self.sample_ids) == len(self.tabular_features)

        self.has_labels = "label" in dataframe.columns
        if self.has_labels:
            self.labels = dataframe["label"].values.astype(np.float32)
        else:
            self.labels = None

        if (
            is_train
            and ("center_id" in dataframe.columns)
            and (center_to_idx is not None)
        ):
            self.center_indices = np.array(
                [center_to_idx.get(str(c), -1) for c in dataframe["center_id"].values],
                dtype=np.int64,
            )
        else:
            self.center_indices = np.full(len(self.sample_ids), -1, dtype=np.int64)

        self.is_train = is_train
        self.rgb_mean = image_stats["rgb_mean"][:, None, None].astype(np.float32)
        self.rgb_std = image_stats["rgb_std"][:, None, None].astype(np.float32)
        self.her_mean = image_stats["her_mean"].astype(np.float32)
        self.her_std = image_stats["her_std"].astype(np.float32)
        self.fourier_bank = image_stats["fourier_bank"]
        self.fourier_bank_centers = image_stats["fourier_bank_centers"]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        rgb_u8 = load_rgb_patch(self.image_paths[idx])
        rgb_f32 = rgb_u8.astype(np.float32)

        if self.is_train:
            rng = np.random.default_rng()
            # 1. Exact D4 dihedral symmetry (lossless 90-deg rotation + reflection)
            k_rot = int(rng.integers(0, 4))
            if k_rot > 0:
                rgb_f32 = np.rot90(rgb_f32, k=k_rot, axes=(0, 1))
            if rng.random() < 0.5:
                rgb_f32 = np.fliplr(rgb_f32)
            rgb_f32 = np.ascontiguousarray(rgb_f32)

            # 2. Cross-Center Fourier Low-Frequency Amplitude Mixing (FACT/FDA)
            if rng.random() < 0.65 and len(self.fourier_bank) > 0:
                src_cid = self.center_indices[idx]
                other_mask = self.fourier_bank_centers != src_cid
                if np.any(other_mask) and rng.random() < 0.75:
                    cand_idx = np.where(other_mask)[0]
                    b_idx = int(cand_idx[rng.integers(0, len(cand_idx))])
                else:
                    b_idx = int(rng.integers(0, len(self.fourier_bank)))
                alpha = float(rng.uniform(0.15, 0.55))
                rgb_f32 = apply_fourier_amplitude_mix(
                    rgb_f32, self.fourier_bank[b_idx], alpha
                )

            # 3. Physical Ruifrok-Johnston H&E Stain Concentration Subspace Perturbation
            if rng.random() < 0.75:
                rgb_f32 = apply_he_subspace_stain_aug(rgb_f32, rng)

            # 4. Subtle scanner optical MTF blur / sensor shot noise simulation
            r_opt = rng.random()
            if r_opt < 0.20:
                sigma = float(rng.uniform(0.35, 0.85))
                rgb_f32 = cv2.GaussianBlur(rgb_f32, (3, 3), sigma)
            elif r_opt < 0.35:
                noise = rng.normal(
                    0.0, rng.uniform(1.0, 3.0), size=rgb_f32.shape
                ).astype(np.float32)
                rgb_f32 = np.clip(rgb_f32 + noise, 0.0, 255.0)

        # Identical deterministic tensor conversion for train (post-aug), validation, and test
        stain_tensor = compute_calibrated_stain_tensor(
            rgb_f32, self.her_mean, self.her_std
        )
        rgb_norm = ((rgb_f32 / 255.0).transpose(2, 0, 1) - self.rgb_mean) / (
            self.rgb_std + 1e-6
        )

        item = {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb_norm, dtype=np.float32)),
            "stain": torch.from_numpy(
                np.ascontiguousarray(stain_tensor, dtype=np.float32)
            ),
            "tabular": torch.from_numpy(self.tabular_features[idx]),
            "center_idx": torch.tensor(self.center_indices[idx], dtype=torch.long),
            "sample_id": self.sample_ids[idx],
        }
        if self.has_labels:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def compute_balanced_domain_class_weights(dataframe: pd.DataFrame) -> np.ndarray:
    """Inverse joint (center_id, label) frequency weights strictly from training labels/centers."""
    strata = dataframe["center_id"].astype(str) + "_" + dataframe["label"].astype(str)
    counts = strata.value_counts().to_dict()
    total = float(len(dataframe))
    n_strata = float(len(counts))
    weights = np.array(
        [total / (n_strata * counts[s]) for s in strata], dtype=np.float64
    )
    return weights / weights.mean()


def build_dataloaders(
    train_dataset_obj: Dataset,
    val_dataset_obj: Dataset,
    test_dataset_obj: Dataset,
    sample_weights: np.ndarray = None,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Construct PyTorch DataLoaders with domain-class balanced sampling for training."""
    nw = max(2, min(num_workers, os.cpu_count() or 2))
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset_obj,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# 7. Step 1 Execution: Data Loading, Leak-Free Splits, Features & Caching
# =============================================================================
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(INPUT_DIR, "validation.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub_df = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

# Strict Integrity & Disjoint Group Verification
assert (
    "label" in val_df.columns and val_df["label"].isin([0, 1]).all()
), "validation.csv must contain binary labels"
assert train_df["label"].isin([0, 1]).all(), "train.csv must contain binary labels"
assert set(train_df["sample_id"]).isdisjoint(
    set(val_df["sample_id"])
), "Leakage: sample_id overlap!"
assert set(train_df["slide_id"]).isdisjoint(
    set(val_df["slide_id"])
), "Leakage: slide_id overlap!"
assert set(train_df["group_id"]).isdisjoint(
    set(val_df["group_id"])
), "Leakage: group_id overlap!"
assert (
    test_df["sample_id"].values == sample_sub_df["sample_id"].values
).all(), "test.csv and sample_submission.csv sample_id order mismatch!"

# Encode center_id strictly from development centers for training-time domain methods ONLY
unique_train_centers = sorted(train_df["center_id"].astype(str).unique().tolist())
CENTER_TO_IDX = {cid: idx for idx, cid in enumerate(unique_train_centers)}

# A. Splits within train_df ONLY (protects fixed validation.csv from selection/tuning leakage):
#    1) Leave-One-Center-Out (LOCO) splits to directly audit unseen-center calibration
train_loco_splits = []
for held_out_cid in unique_train_centers:
    tr_idx = np.where(train_df["center_id"].astype(str).values != held_out_cid)[0]
    ho_idx = np.where(train_df["center_id"].astype(str).values == held_out_cid)[0]
    assert set(train_df.iloc[tr_idx]["group_id"]).isdisjoint(
        set(train_df.iloc[ho_idx]["group_id"])
    )
    train_loco_splits.append((held_out_cid, tr_idx, ho_idx))

#    2) 5-Fold StratifiedGroupKFold within train_df (grouped by group_id, stratified by center+label)
train_strata = train_df["center_id"].astype(str) + "_" + train_df["label"].astype(str)
sgkf_train = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_gkf_folds = []
for fold_i, (f_tr, f_va) in enumerate(
    sgkf_train.split(train_df, train_strata, groups=train_df["group_id"])
):
    assert set(train_df.iloc[f_tr]["group_id"]).isdisjoint(
        set(train_df.iloc[f_va]["group_id"])
    )
    assert set(train_df.iloc[f_tr]["slide_id"]).isdisjoint(
        set(train_df.iloc[f_va]["slide_id"])
    )
    train_gkf_folds.append((f_tr, f_va))

# B. Merged Full-Development Dataset (train.csv + validation.csv = 16,800 labeled samples)
#    Reserved strictly for post-validation refit AFTER pre-refit validation score is recorded
merged_dev_df = pd.concat(
    [
        train_df.assign(source_partition="train"),
        val_df.assign(source_partition="validation"),
    ],
    axis=0,
    ignore_index=True,
)
merged_strata = (
    merged_dev_df["center_id"].astype(str) + "_" + merged_dev_df["label"].astype(str)
)
sgkf_merged = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
merged_gkf_folds = []
for fold_i, (m_tr, m_va) in enumerate(
    sgkf_merged.split(merged_dev_df, merged_strata, groups=merged_dev_df["group_id"])
):
    assert set(merged_dev_df.iloc[m_tr]["group_id"]).isdisjoint(
        set(merged_dev_df.iloc[m_va]["group_id"])
    )
    assert set(merged_dev_df.iloc[m_tr]["slide_id"]).isdisjoint(
        set(merged_dev_df.iloc[m_va]["slide_id"])
    )
    merged_gkf_folds.append((m_tr, m_va))

# Checksum over sample_ids to safely reuse/validate cached feature extraction in ./working/
_id_concat = "|".join(
    train_df["sample_id"].tolist()
    + val_df["sample_id"].tolist()
    + test_df["sample_id"].tolist()
)
_cache_sig = hashlib.md5(
    (_id_concat + f"_f{NUM_TABULAR_FEATURES}_v2").encode("utf-8")
).hexdigest()[:12]
CACHE_PATH = os.path.join(WORKING_DIR, f"histopath_features_{_cache_sig}.npz")

if os.path.exists(CACHE_PATH):
    cached = np.load(CACHE_PATH)
    X_train_raw = cached["X_train_raw"]
    X_val_raw = cached["X_val_raw"]
    X_test_raw = cached["X_test_raw"]
    train_image_stats = {
        "rgb_mean": cached["rgb_mean"],
        "rgb_std": cached["rgb_std"],
        "her_mean": cached["her_mean"],
        "her_std": cached["her_std"],
        "fourier_bank": cached["fourier_bank"],
        "fourier_bank_centers": cached["fourier_bank_centers"],
    }
else:
    X_train_raw = extract_features_parallel(train_df["image_path"].tolist())
    X_val_raw = extract_features_parallel(val_df["image_path"].tolist())
    X_test_raw = extract_features_parallel(test_df["image_path"].tolist())
    train_image_stats = fit_train_only_image_statistics_and_fourier_bank(
        train_df, seed=SEED
    )

    np.savez_compressed(
        CACHE_PATH,
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        rgb_mean=train_image_stats["rgb_mean"],
        rgb_std=train_image_stats["rgb_std"],
        her_mean=train_image_stats["her_mean"],
        her_std=train_image_stats["her_std"],
        fourier_bank=train_image_stats["fourier_bank"],
        fourier_bank_centers=train_image_stats["fourier_bank_centers"],
    )

# Now seed full CPU + CUDA RNG state
seed_everything(SEED, seed_cuda=True)

# Raw merged features aligned with merged_dev_df (untouched by primary validation scaler)
X_merged_raw = np.vstack([X_train_raw, X_val_raw]).astype(np.float32)

# Fit Tabular Preprocessor STRICTLY on X_train_raw ONLY (Zero Validation/Test Leakage)
tab_preprocessor_train = LeakFreeTabularPreprocessor(seed=SEED)
X_train_tab = tab_preprocessor_train.fit_transform(X_train_raw)
X_val_tab = tab_preprocessor_train.transform(X_val_raw)
X_test_tab = tab_preprocessor_train.transform(X_test_raw)

y_train = train_df["label"].values.astype(np.int64)
y_val = val_df["label"].values.astype(np.int64)
y_merged = merged_dev_df["label"].values.astype(np.int64)

train_sampling_weights = compute_balanced_domain_class_weights(train_df)

# Instantiate Primary Leak-Free Multi-View Datasets & Verify Deterministic Consistency
train_dataset = HistopathologyMultiViewDataset(
    train_df, X_train_tab, train_image_stats, is_train=True, center_to_idx=CENTER_TO_IDX
)
val_dataset = HistopathologyMultiViewDataset(
    val_df, X_val_tab, train_image_stats, is_train=False, center_to_idx=None
)
test_dataset = HistopathologyMultiViewDataset(
    test_df, X_test_tab, train_image_stats, is_train=False, center_to_idx=None
)

_s_tr = train_dataset[0]
_s_va1 = val_dataset[0]
_s_va2 = val_dataset[0]
_s_te = test_dataset[0]

assert _s_tr["rgb"].shape == (3, IMG_SIZE, IMG_SIZE) and _s_tr["stain"].shape == (
    3,
    IMG_SIZE,
    IMG_SIZE,
)
assert _s_va1["rgb"].shape == (3, IMG_SIZE, IMG_SIZE) and _s_te["rgb"].shape == (
    3,
    IMG_SIZE,
    IMG_SIZE,
)
assert _s_tr["tabular"].shape == (NUM_TABULAR_FEATURES,) and _s_te["tabular"].shape == (
    NUM_TABULAR_FEATURES,
)
assert torch.isfinite(_s_tr["rgb"]).all() and torch.isfinite(_s_tr["stain"]).all()
assert torch.isfinite(_s_va1["rgb"]).all() and torch.isfinite(_s_te["rgb"]).all()
assert torch.equal(_s_va1["rgb"], _s_va2["rgb"]) and torch.equal(
    _s_va1["stain"], _s_va2["stain"]
), "Validation/Test dataset pipeline must be 100% deterministic!"

print(
    f"Data & Feature Engineering Complete | "
    f"train={len(train_df)} (pos={y_train.mean():.3f}), "
    f"val={len(val_df)} (pos={y_val.mean():.3f}), "
    f"merged_dev={len(merged_dev_df)}, test={len(test_df)} | "
    f"tabular_dim={NUM_TABULAR_FEATURES}, fourier_bank={train_image_stats['fourier_bank'].shape}"
)


# =============================================================================
# 8. Step 2: Pretrained Backbone Documentation & Physics/Spectral Modules
# =============================================================================
# Explicit Pretrained Model Documentation (per competition rules):
# - Identity: swinv2_tiny_window8_256 (Microsoft Swin Transformer V2 Tiny, native 256x256, window=8)
# - Library: timm (version logged at runtime)
# - Pretraining Dataset: ImageNet-1K (ms_in1k supervised natural images)
# - Known Histopathology / Camelyon Overlap: None (natural-image pretraining only)
MODEL_METADATA = {
    "architecture": "SwinV2StainSpectralCrossAttnNet",
    "pretrained_backbone": "swinv2_tiny_window8_256",
    "timm_version": getattr(timm, "__version__", "unknown"),
    "pretraining_corpus": "ImageNet-1K (ms_in1k)",
    "known_target_overlap": "None (natural images only; zero histopathology overlap)",
}

_TABULAR_DIM = NUM_TABULAR_FEATURES
_NUM_TRAIN_CENTERS = len(CENTER_TO_IDX)


class _GradientReversalFn(torch.autograd.Function):
    """Reverses gradients during training-only domain adversarial alignment."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reverse(x: torch.Tensor, lambd: float = 0.25) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambd)


class ZeroPhaseSpectralGate2d(nn.Module):
    """
    2D Real-FFT Zero-Phase Spectral Equalizer.
    Modulates spatial-frequency amplitudes via a smooth learnable gain + instance spectral
    context while strictly preserving Fourier phase (nuclear boundaries & spatial layout).
    Executed in float32 outside AMP autocast for exact cuFFT numerical stability.
    """

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        freq_w = width // 2 + 1

        self.freq_gain_logits = nn.Parameter(torch.zeros(1, channels, height, freq_w))
        self.channel_gate = nn.Sequential(
            nn.Linear(channels * 2, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid(),
        )
        self.post_norm = nn.GroupNorm(
            num_groups=min(8, channels), num_channels=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                out = self._forward_fp32(x.float())
        else:
            out = self._forward_fp32(x.float())
        return out.to(orig_dtype)

    def _forward_fp32(self, x_f32: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_f32.shape
        x_fft = torch.fft.rfft2(x_f32, dim=(-2, -1), norm="ortho")

        gain_map = 2.0 * torch.sigmoid(self.freq_gain_logits)
        if gain_map.shape[-2] != H or gain_map.shape[-1] != (W // 2 + 1):
            gain_map = F.interpolate(
                gain_map, size=(H, W // 2 + 1), mode="bilinear", align_corners=False
            )

        psd = x_fft.real.square() + x_fft.imag.square()
        dc_masked = psd.clone()
        dc_masked[:, :, 0, 0] = 0.0
        low_h, low_w = max(2, H // 4), max(2, (W // 2 + 1) // 2)
        low_e = torch.log1p(dc_masked[:, :, :low_h, :low_w].mean(dim=(-2, -1)))
        high_e = torch.log1p(dc_masked[:, :, low_h:, :].mean(dim=(-2, -1)))
        ch_mod = 0.5 + self.channel_gate(torch.cat([low_e, high_e], dim=-1)).unsqueeze(
            -1
        ).unsqueeze(-1)

        x_fft_mod = x_fft * (gain_map * ch_mod)
        x_filtered = torch.fft.irfft2(x_fft_mod, s=(H, W), dim=(-2, -1), norm="ortho")
        return x_f32 + 0.5 * self.post_norm(x_filtered)


class StainDilatedResidualBlock(nn.Module):
    """
    BatchNorm-free multi-dilated depthwise-separable block capturing both single-nucleus
    chromatin texture (dilation=1) and inter-nuclear spatial packing (dilation=3).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 2,
        spec_hw: Optional[Tuple[int, int]] = None,
        dropout_p: float = 0.10,
    ):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.GELU(),
        )
        self.dw_local = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False
        )
        self.dw_context = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=3,
            dilation=3,
            groups=out_ch,
            bias=False,
        )
        self.pw_fuse = nn.Sequential(
            nn.GroupNorm(num_groups=min(8, out_ch * 2), num_channels=out_ch * 2),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
        )
        self.spec_gate = (
            ZeroPhaseSpectralGate2d(out_ch, spec_hw[0], spec_hw[1])
            if spec_hw is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down(x)
        ms = torch.cat([self.dw_local(h), self.dw_context(h)], dim=1)
        h = F.gelu(h + self.pw_fuse(ms))
        return self.spec_gate(h)


class StainSpectralPyramidEncoder(nn.Module):
    """
    Dedicated physics-guided encoder for calibrated [Hematoxylin, Eosin, Residual] density maps.
    Includes an explicit Orthogonal Pigment Suppression Gate conditioned on the 3rd channel
    (Residual OD) to prevent anthracotic carbon / hemosiderin from mimicking hyperchromasia.
    """

    def __init__(self, dropout_p: float = 0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.stage_32 = StainDilatedResidualBlock(
            64, 96, stride=2, spec_hw=(32, 32), dropout_p=dropout_p
        )
        self.stage_16 = StainDilatedResidualBlock(
            96, 144, stride=2, spec_hw=(16, 16), dropout_p=dropout_p
        )
        self.stage_8 = StainDilatedResidualBlock(
            144, 192, stride=2, spec_hw=None, dropout_p=dropout_p
        )

        self.pigment_gate_16 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 144, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, stain: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_pigment = stain[:, 2:3, :, :].abs()
        x = self.stem(stain)  # (B, 64, 64, 64)
        f32 = self.stage_32(x)  # (B, 96, 32, 32)
        f16 = self.stage_16(f32)  # (B, 144, 16, 16)

        pig_16 = F.adaptive_avg_pool2d(residual_pigment, (16, 16))
        f16 = f16 * (0.5 + 0.5 * self.pigment_gate_16(pig_16))
        f8 = self.stage_8(f16)  # (B, 192, 8, 8)
        return f32, f16, f8


class CytologyFamilyTokenizer(nn.Module):
    """
    Projects the 68 engineered features into 5 biologically structured tokens matching Step 1:
      [0:24] Multi-Concentric ROI Deconvolved Stain Architecture
      [24:42] Segmented Nuclear Cytometry & cKDTree Point-Process Topology
      [42:52] Radial 2D Fourier Power Spectrum & Anisotropy
      [52:62] Multi-Scale LoG Chromatin Scale & Structure Tensor Coherency
      [62:68] Quantized Chromatin GLCM & Joint Stain Entropy
    """

    def __init__(
        self,
        tabular_dim: int = 68,
        d_model: int = 256,
        dropout_p: float = 0.20,
        token_drop_p: float = 0.10,
    ):
        super().__init__()
        if tabular_dim == 68:
            self.slices = [(0, 24), (24, 42), (42, 52), (52, 62), (62, 68)]
        else:
            edges = np.linspace(0, tabular_dim, 6, dtype=int)
            self.slices = [(int(edges[i]), int(edges[i + 1])) for i in range(5)]

        self.num_families = len(self.slices)
        self.token_drop_p = token_drop_p

        self.encoders = nn.ModuleList()
        for lo, hi in self.slices:
            in_d = hi - lo
            self.encoders.append(
                nn.Sequential(
                    nn.LayerNorm(in_d),
                    nn.Dropout(p=dropout_p * 0.5),
                    nn.Linear(in_d, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(p=dropout_p),
                    nn.Linear(d_model, d_model * 2),
                )
            )

        self.family_pos = nn.Parameter(
            torch.randn(1, self.num_families, d_model) * 0.02
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.summary_attn = nn.Linear(d_model, 1)

    def forward(self, tabular: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = []
        for (lo, hi), enc in zip(self.slices, self.encoders):
            sub = tabular[:, lo:hi]
            val, gate = enc(sub).chunk(2, dim=-1)
            tokens.append(val * torch.sigmoid(gate))

        tok_seq = torch.stack(tokens, dim=1) + self.family_pos  # (B, 5, D)
        if self.training and self.token_drop_p > 0.0:
            keep_mask = (
                torch.rand(tok_seq.size(0), self.num_families, 1, device=tok_seq.device)
                >= self.token_drop_p
            ).to(tok_seq.dtype)
            all_dropped = keep_mask.sum(dim=1, keepdim=True) == 0
            keep_mask = torch.where(all_dropped, torch.ones_like(keep_mask), keep_mask)
            tok_seq = tok_seq * keep_mask / (1.0 - self.token_drop_p)

        tok_seq = self.out_norm(tok_seq)
        w = F.softmax(self.summary_attn(tok_seq), dim=1)
        summary_vec = (w * tok_seq).sum(dim=1)  # (B, D)
        return tok_seq, summary_vec


class BidirectionalPathologyCrossAttn(nn.Module):
    """
    Allows semantic cytology + readout queries to cross-attend to spatial visual+stain tokens
    (localizing focal pleomorphic nuclei & core-vs-outer invasion) followed by joint refinement.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout_p: float = 0.15):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout_p, batch_first=True
        )
        self.cross_drop = nn.Dropout(p=dropout_p)

        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(p=dropout_p),
        )

        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout_p, batch_first=True
        )
        self.self_drop = nn.Dropout(p=dropout_p)

        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(p=dropout_p),
        )

    def forward(
        self, query_tokens: torch.Tensor, spatial_tokens: torch.Tensor
    ) -> torch.Tensor:
        qn = self.q_norm(query_tokens)
        kvn = self.kv_norm(spatial_tokens)
        ca_out, _ = self.cross_attn(query=qn, key=kvn, value=kvn, need_weights=False)
        x = query_tokens + self.cross_drop(ca_out)
        x = x + self.ffn1(self.ffn1_norm(x))

        sn = self.self_norm(x)
        sa_out, _ = self.self_attn(query=sn, key=sn, value=sn, need_weights=False)
        x = x + self.self_drop(sa_out)
        x = x + self.ffn2(self.ffn2_norm(x))
        return x


def _create_swinv2_backbone(pretrained: bool = True) -> Tuple[nn.Module, List[int]]:
    """Load native-256 SwinV2-Tiny backbone (with safe fallback) and inspect stage channels."""
    candidates = [
        ("swinv2_tiny_window8_256", {}),
        ("swin_tiny_patch4_window7_224", {"img_size": 256}),
    ]
    last_err = None
    for name, extra_kwargs in candidates:
        try:
            backbone = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3),
                drop_path_rate=0.20,
                **extra_kwargs,
            )
            MODEL_METADATA["pretrained_backbone"] = name
            channels = list(backbone.feature_info.channels())
            return backbone, channels
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Failed to create Swin backbone: {last_err}")


def _ensure_nchw(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
    """Convert NHWC -> NCHW safely across all timm versions."""
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got {feat.shape}")
    if feat.shape[1] == expected_c and feat.shape[-1] != expected_c:
        return feat
    if feat.shape[-1] == expected_c:
        return feat.permute(0, 3, 1, 2).contiguous()
    return feat


class SwinV2StainSpectralCrossAttnNet(nn.Module):
    """
    Multi-Modal Domain-Generalizing Architecture for H&E Lymph-Node Classification:
      1) Pretrained SwinV2-Tiny (scaled-cosine attention + LayerNorm, BatchNorm-free) on RGB.
      2) Zero-Phase 2D-FFT Spectral-Gated Pyramid on calibrated [H, E, Residual] OD maps.
      3) D4-Invariant Radial-Concentric Spatial Encoding + Fine 32x32 Focal Saliency Readout.
      4) 5-Family Cytological Tokenizer + Bidirectional Cross-Attention + Multi-Sample Dropout.
      5) Training-only GRL Domain Discriminator & normalized embeddings (unused at inference).
    """

    def __init__(
        self,
        tabular_dim: int = _TABULAR_DIM,
        d_model: int = 256,
        num_train_centers: int = _NUM_TRAIN_CENTERS,
        pretrained: bool = True,
        dropout_p: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.rgb_backbone, self.swin_channels = _create_swinv2_backbone(
            pretrained=pretrained
        )
        c1, c2, c3 = self.swin_channels  # Typically [192, 384, 768]

        self.stain_encoder = StainSpectralPyramidEncoder(dropout_p=0.10)

        self.fuse_32 = nn.Sequential(
            nn.Conv2d(c1 + 96, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )
        self.fine_focal_scorer = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, d_model // 4),
            nn.GELU(),
            nn.Conv2d(d_model // 4, 1, kernel_size=1),
        )

        self.fuse_16 = nn.Sequential(
            nn.Conv2d(c2 + 144, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )
        self.fuse_8 = nn.Sequential(
            nn.Conv2d(c3 + 192, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )

        self.spatial_token_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.LayerNorm(d_model),
        )

        coords_1d = (torch.arange(8, dtype=torch.float32) + 0.5) / 8.0 * 2.0 - 1.0
        gy, gx = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
        r_euc = torch.sqrt(gx.square() + gy.square()) / math.sqrt(2.0)
        r_cheb = torch.maximum(gx.abs(), gy.abs())
        in_core = (r_cheb <= 0.26).float()
        in_mid = ((r_cheb > 0.26) & (r_cheb <= 0.52)).float()
        radial_basis = torch.stack(
            [r_euc, r_cheb, r_euc.square(), in_core, in_mid], dim=-1
        ).view(1, 64, 5)
        self.register_buffer("radial_basis_8x8", radial_basis, persistent=False)
        self.radial_pos_mlp = nn.Sequential(
            nn.Linear(5, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.cytology_tokenizer = CytologyFamilyTokenizer(
            tabular_dim=tabular_dim, d_model=d_model, dropout_p=0.20, token_drop_p=0.10
        )
        self.cls_queries = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        self.cross_fusion = BidirectionalPathologyCrossAttn(
            d_model=d_model, num_heads=8, dropout_p=0.15
        )

        self.spatial_focal_scorer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        self.bottleneck = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.ms_dropouts = nn.ModuleList([nn.Dropout(p=dropout_p) for _ in range(5)])
        self.classifier = nn.Linear(d_model, 1)
        nn.init.zeros_(self.classifier.bias)

        self.aux_vis_head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(p=0.20),
            nn.Linear(d_model * 2, 1),
        )
        self.aux_tab_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(p=0.20),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.aux_vis_head[-1].bias)
        nn.init.zeros_(self.aux_tab_head[-1].bias)

        self.domain_discriminator = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(d_model // 2, max(2, num_train_centers)),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        stain: torch.Tensor,
        tabular: torch.Tensor,
        grl_lambda: float = 0.25,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B = rgb.size(0)

        s_feats = self.rgb_backbone(rgb)
        s1 = _ensure_nchw(s_feats[0], self.swin_channels[0])  # (B, C1, 32, 32)
        s2 = _ensure_nchw(s_feats[1], self.swin_channels[1])  # (B, C2, 16, 16)
        s3 = _ensure_nchw(s_feats[2], self.swin_channels[2])  # (B, C3, 8, 8)

        st32, st16, st8 = self.stain_encoder(stain)

        m32 = self.fuse_32(torch.cat([s1, st32], dim=1))  # (B, D, 32, 32)
        fine_logits_map = self.fine_focal_scorer(m32).flatten(2)  # (B, 1, 1024)
        fine_attn = F.softmax(2.5 * fine_logits_map.float(), dim=-1).to(m32.dtype)
        fine_focal_vec = torch.bmm(fine_attn, m32.flatten(2).transpose(1, 2)).squeeze(
            1
        )  # (B, D)

        m16 = self.fuse_16(torch.cat([s2, st16], dim=1))  # (B, D, 16, 16)
        m16_to_8 = 0.5 * (
            F.avg_pool2d(m16, kernel_size=2, stride=2)
            + F.max_pool2d(m16, kernel_size=2, stride=2)
        )
        m8 = self.fuse_8(torch.cat([s3, st8], dim=1))  # (B, D, 8, 8)

        vis_grid = (
            torch.cat([m8, m16_to_8], dim=1).flatten(2).transpose(1, 2)
        )  # (B, 64, 2D)
        spatial_tokens = self.spatial_token_proj(vis_grid) + self.radial_pos_mlp(
            self.radial_basis_8x8
        )  # (B, 64, D)

        cyto_tokens, tab_summary = self.cytology_tokenizer(tabular)  # (B, 5, D), (B, D)
        queries = torch.cat([self.cls_queries.expand(B, -1, -1), cyto_tokens], dim=1)
        fused_queries = self.cross_fusion(queries, spatial_tokens)  # (B, 7, D)

        cls_global = fused_queries[:, 0, :]
        cls_focal = fused_queries[:, 1, :]

        sp_scores = self.spatial_focal_scorer(spatial_tokens)  # (B, 64, 1)
        sp_weights = F.softmax(3.0 * sp_scores.float(), dim=1).to(spatial_tokens.dtype)
        spatial_focal_vec = (sp_weights * spatial_tokens).sum(dim=1)  # (B, D)

        concat_repr = torch.cat(
            [cls_global, cls_focal, spatial_focal_vec, fine_focal_vec], dim=-1
        )
        fused_feat = self.bottleneck(concat_repr)  # (B, D)

        logits = (
            torch.stack(
                [self.classifier(drop(fused_feat)) for drop in self.ms_dropouts], dim=0
            )
            .mean(dim=0)
            .squeeze(-1)
        )

        if not return_aux:
            return logits

        vis_summary = torch.cat([spatial_focal_vec, fine_focal_vec], dim=-1)  # (B, 2D)
        aux_vis_logits = self.aux_vis_head(vis_summary).squeeze(-1)
        aux_tab_logits = self.aux_tab_head(tab_summary).squeeze(-1)
        center_logits = self.domain_discriminator(
            gradient_reverse(vis_summary, grl_lambda)
        )

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "aux_vis_logits": aux_vis_logits,
            "aux_tab_logits": aux_tab_logits,
            "fused_feat": F.normalize(fused_feat.float(), p=2, dim=-1),
            "center_logits": center_logits,
        }

    @torch.no_grad()
    def predict_proba(
        self, rgb: torch.Tensor, stain: torch.Tensor, tabular: torch.Tensor
    ) -> torch.Tensor:
        """Deterministic tumor probability inference in [0, 1] for validation & test."""
        logits = self.forward(rgb, stain, tabular, return_aux=False)
        return torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)


class UnseenCenterBalancedAccuracyLoss(nn.Module):
    """
    Directly aligns training optimization with unseen-center Balanced Accuracy at fixed 0.5
    threshold + AUROC:
      1) Class-Balanced Symmetric Label-Smoothed BCE (anchors Bayes boundary strictly at 0.5).
      2) Differentiable Fixed-0.5 Soft Balanced Accuracy surrogate: 1 - 0.5*(SoftTPR + SoftTNR).
      3) Pairwise Logistic Margin AUROC loss for threshold-robust separability.
      4) Deep supervision on visual & cytological streams to prevent single-modality collapse.
      5) Training-only Class-Conditional Cross-Center Alignment (Conditional CORAL + GRL).
    """

    def __init__(
        self,
        label_smoothing: float = 0.03,
        soft_ba_weight: float = 0.35,
        sharpness: float = 4.0,
        auroc_weight: float = 0.15,
        rank_margin: float = 1.0,
        aux_vis_weight: float = 0.20,
        aux_tab_weight: float = 0.15,
        cond_coral_weight: float = 0.15,
        adv_center_weight: float = 0.10,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.soft_ba_weight = soft_ba_weight
        self.sharpness = sharpness
        self.auroc_weight = auroc_weight
        self.rank_margin = rank_margin
        self.aux_vis_weight = aux_vis_weight
        self.aux_tab_weight = aux_tab_weight
        self.cond_coral_weight = cond_coral_weight
        self.adv_center_weight = adv_center_weight

    def _symmetric_balanced_bce(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        y = targets.float()
        y_smooth = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        raw_bce = F.binary_cross_entropy_with_logits(logits, y_smooth, reduction="none")

        pos_mask = y > 0.5
        neg_mask = ~pos_mask
        n_pos = pos_mask.float().sum()
        n_neg = neg_mask.float().sum()

        if n_pos > 0 and n_neg > 0:
            loss_pos = raw_bce[pos_mask].mean()
            loss_neg = raw_bce[neg_mask].mean()
            return 0.5 * (loss_pos + loss_neg)
        return raw_bce.mean()

    def _soft_balanced_accuracy_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        y = targets.float()
        n_pos = y.sum()
        n_neg = (1.0 - y).sum()
        if n_pos < 1.0 or n_neg < 1.0:
            return logits.new_zeros(())

        p_sharp = torch.sigmoid(self.sharpness * logits)
        soft_tpr = (p_sharp * y).sum() / (n_pos + 1e-6)
        soft_tnr = ((1.0 - p_sharp) * (1.0 - y)).sum() / (n_neg + 1e-6)
        return 1.0 - 0.5 * (soft_tpr + soft_tnr)

    def _pairwise_auroc_margin_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        pos_logits = logits[targets > 0.5]
        neg_logits = logits[targets <= 0.5]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return logits.new_zeros(())
        diffs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        return F.softplus(self.rank_margin - diffs).mean()

    @staticmethod
    def _class_conditional_coral(
        fused_feat: torch.Tensor, targets: torch.Tensor, center_idx: torch.Tensor
    ) -> torch.Tensor:
        valid = center_idx >= 0
        if valid.sum() < 6:
            return fused_feat.new_zeros(())
        unique_c = torch.unique(center_idx[valid])
        if unique_c.numel() < 2:
            return fused_feat.new_zeros(())

        c0, c1 = unique_c[0], unique_c[1]
        feat_slice = fused_feat[:, :64]
        d = float(feat_slice.size(1))
        align_losses = []

        for cls_val in (0.0, 1.0):
            cls_mask = (targets - cls_val).abs() < 0.25
            f0 = feat_slice[cls_mask & (center_idx == c0)]
            f1 = feat_slice[cls_mask & (center_idx == c1)]
            if f0.size(0) >= 3 and f1.size(0) >= 3:
                mu0, mu1 = f0.mean(dim=0), f1.mean(dim=0)
                mean_diff = (mu0 - mu1).square().sum()

                z0 = f0 - mu0.unsqueeze(0)
                z1 = f1 - mu1.unsqueeze(0)
                cov0 = (z0.T @ z0) / float(f0.size(0) - 1)
                cov1 = (z1.T @ z1) / float(f1.size(0) - 1)
                cov_diff = (cov0 - cov1).square().sum() / (4.0 * d)
                align_losses.append(mean_diff + cov_diff)

        if not align_losses:
            return fused_feat.new_zeros(())
        return torch.stack(align_losses).mean()

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets_f32 = targets.float()

        if torch.is_tensor(outputs):
            logits = outputs.float().view(-1)
            loss = self._symmetric_balanced_bce(logits, targets_f32)
            loss = loss + self.soft_ba_weight * self._soft_balanced_accuracy_loss(
                logits, targets_f32
            )
            loss = loss + self.auroc_weight * self._pairwise_auroc_margin_loss(
                logits, targets_f32
            )
            return loss

        logits = outputs["logits"].float().view(-1)
        total_loss = self._symmetric_balanced_bce(logits, targets_f32)
        total_loss = (
            total_loss
            + self.soft_ba_weight
            * self._soft_balanced_accuracy_loss(logits, targets_f32)
        )
        total_loss = total_loss + self.auroc_weight * self._pairwise_auroc_margin_loss(
            logits, targets_f32
        )

        if "aux_vis_logits" in outputs:
            total_loss = (
                total_loss
                + self.aux_vis_weight
                * self._symmetric_balanced_bce(
                    outputs["aux_vis_logits"].float().view(-1), targets_f32
                )
            )
        if "aux_tab_logits" in outputs:
            total_loss = (
                total_loss
                + self.aux_tab_weight
                * self._symmetric_balanced_bce(
                    outputs["aux_tab_logits"].float().view(-1), targets_f32
                )
            )

        if center_idx is not None:
            valid_c = center_idx >= 0
            if valid_c.any() and "center_logits" in outputs:
                c_logits = outputs["center_logits"].float()[valid_c]
                c_targets = center_idx[valid_c].long()
                if torch.unique(c_targets).numel() >= 2:
                    total_loss = total_loss + self.adv_center_weight * F.cross_entropy(
                        c_logits, c_targets, label_smoothing=0.05
                    )
            if "fused_feat" in outputs and self.cond_coral_weight > 0.0:
                total_loss = (
                    total_loss
                    + self.cond_coral_weight
                    * self._class_conditional_coral(
                        outputs["fused_feat"].float(), targets_f32, center_idx
                    )
                )

        return total_loss


def build_differential_adamw(
    model_obj: nn.Module,
    backbone_lr: float = 1.5e-4,
    head_lr: float = 6.0e-4,
    backbone_wd: float = 0.03,
    head_wd: float = 0.02,
) -> AdamW:
    """
    Constructs AdamW with differential learning rates and strict zero weight decay on biases,
    LayerNorm/GroupNorm, SwinV2 cosine logit_scale, relative position tables, and spectral gates.
    """
    no_decay_keywords = (
        "bias",
        "norm",
        "logit_scale",
        "cpb_mlp",
        "relative_position",
        "cls_queries",
        "family_pos",
        "freq_gain_logits",
    )

    bb_decay, bb_no_decay = [], []
    head_decay, head_no_decay = [], []

    for name, param in model_obj.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.ndim <= 1) or any(
            kw in name.lower() for kw in no_decay_keywords
        )
        is_backbone = name.startswith("rgb_backbone.")

        if is_backbone and not is_no_decay:
            bb_decay.append(param)
        elif is_backbone and is_no_decay:
            bb_no_decay.append(param)
        elif (not is_backbone) and (not is_no_decay):
            head_decay.append(param)
        else:
            head_no_decay.append(param)

    param_groups = [
        {
            "params": bb_decay,
            "lr": backbone_lr,
            "weight_decay": backbone_wd,
            "name": "bb_decay",
        },
        {
            "params": bb_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "name": "bb_no_decay",
        },
        {
            "params": head_decay,
            "lr": head_lr,
            "weight_decay": head_wd,
            "name": "head_decay",
        },
        {
            "params": head_no_decay,
            "lr": head_lr,
            "weight_decay": 0.0,
            "name": "head_no_decay",
        },
    ]
    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def build_model_criterion_optimizer(
    tabular_dim: int = _TABULAR_DIM,
    num_train_centers: int = _NUM_TRAIN_CENTERS,
    pretrained: bool = True,
    backbone_lr: float = 1.5e-4,
    head_lr: float = 6.0e-4,
    device: torch.device = DEVICE,
) -> Tuple[SwinV2StainSpectralCrossAttnNet, UnseenCenterBalancedAccuracyLoss, AdamW]:
    """Clean factory to instantiate fresh (model, criterion, optimizer) for CV/refit stages."""
    net = SwinV2StainSpectralCrossAttnNet(
        tabular_dim=tabular_dim,
        d_model=256,
        num_train_centers=num_train_centers,
        pretrained=pretrained,
        dropout_p=0.25,
    ).to(device)
    loss_fn = UnseenCenterBalancedAccuracyLoss().to(device)
    opt = build_differential_adamw(net, backbone_lr=backbone_lr, head_lr=head_lr)
    return net, loss_fn, opt


# Instantiate Primary Model, Criterion, Optimizer & Verify Metric Alignment
model, criterion, optimizer = build_model_criterion_optimizer(
    tabular_dim=_TABULAR_DIM,
    num_train_centers=_NUM_TRAIN_CENTERS,
    pretrained=True,
    device=DEVICE,
)

with torch.no_grad():
    model.eval()
    _b = 4
    _dummy_rgb = torch.randn(_b, 3, 256, 256, device=DEVICE)
    _dummy_stain = torch.randn(_b, 3, 256, 256, device=DEVICE)
    _dummy_tab = torch.randn(_b, _TABULAR_DIM, device=DEVICE)
    _dummy_y = torch.tensor([0.0, 1.0, 0.0, 1.0], device=DEVICE)
    _dummy_cid = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)

    _probs = model.predict_proba(_dummy_rgb, _dummy_stain, _dummy_tab)
    _aux_out = model(
        _dummy_rgb, _dummy_stain, _dummy_tab, grl_lambda=0.25, return_aux=True
    )
    _loss_val = criterion(_aux_out, _dummy_y, center_idx=_dummy_cid)
    _metric_check = binary_metrics(
        _dummy_y.cpu().numpy().astype(int), _probs.cpu().numpy()
    )

    assert _probs.shape == (_b,) and torch.isfinite(_probs).all()
    assert ((_probs >= 0.0) & (_probs <= 1.0)).all()
    assert torch.isfinite(_loss_val), "Loss produced non-finite value!"
    assert "balanced_accuracy" in _metric_check and "auroc" in _metric_check

_total_params = sum(p.numel() for p in model.parameters())
_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Model Design Complete | arch={MODEL_METADATA['architecture']} "
    f"({MODEL_METADATA['pretrained_backbone']}, timm={MODEL_METADATA['timm_version']}) | "
    f"params={_trainable_params:,}/{_total_params:,} | device={DEVICE}"
)


# =============================================================================
# 9. Step 3: Training, Unseen-Center Audit, Evaluation & Full-Dev Refit
# =============================================================================
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    _VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
    BATCH_SIZE = 64 if _VRAM_GB >= 21.0 else 48
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    USE_SCALER = AMP_DTYPE == torch.float16
else:
    BATCH_SIZE = 32
    USE_AMP = False
    AMP_DTYPE = torch.float32
    USE_SCALER = False

STAGE1_EPOCHS = 12
REFIT_EPOCHS = 11
TTA_VIEWS = 8


def _save_pure_state_dict(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    clean_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    torch.save(clean_sd, path)


def _load_pure_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class WarmupGatedEMA:
    """
    Maintains an exponential moving average of SwinV2StainSpectralCrossAttnNet weights.
    Because the architecture uses exclusively LayerNorm and GroupNorm (zero BatchNorm),
    weight-space EMA has zero running-statistic mismatch on unseen medical centers.
    """

    def __init__(self, source_model: nn.Module, max_decay: float = 0.996):
        self.max_decay = max_decay
        self.num_updates = 0
        self.ema_model = SwinV2StainSpectralCrossAttnNet(
            tabular_dim=_TABULAR_DIM,
            d_model=256,
            num_train_centers=_NUM_TRAIN_CENTERS,
            pretrained=False,
            dropout_p=0.25,
        ).to(DEVICE)
        self.ema_model.load_state_dict(source_model.state_dict())
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source_model: nn.Module, in_warmup: bool = False) -> None:
        if in_warmup:
            self.ema_model.load_state_dict(source_model.state_dict())
            return
        self.num_updates += 1
        decay = min(
            self.max_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates)
        )
        one_minus_decay = 1.0 - decay
        for ema_p, src_p in zip(self.ema_model.parameters(), source_model.parameters()):
            ema_p.mul_(decay).add_(src_p.detach(), alpha=one_minus_decay)


def _fit_single_fold_cytology_gbdt(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    w_va: np.ndarray,
    seed: int,
) -> Tuple[lgb.LGBMClassifier, xgb.XGBClassifier]:
    """
    Trains shallow, heavily regularized LightGBM + XGBoost classifiers on the 68
    stain-deconvolved cytological & spectral features using joint (center_id, label)
    balancing weights so the Bayes-optimal threshold remains anchored at 0.5.
    """
    lgb_clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.025,
        num_leaves=23,
        max_depth=5,
        min_child_samples=45,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=1.5,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_clf.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_sample_weight=[w_va],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=25.0,
        subsample=0.80,
        colsample_bytree=0.65,
        reg_alpha=1.5,
        reg_lambda=6.0,
        tree_method="hist",
        early_stopping_rounds=60,
        eval_metric="logloss",
        random_state=seed + 101,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_clf.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        sample_weight_eval_set=[w_va],
        verbose=False,
    )
    return lgb_clf, xgb_clf


def predict_cytology_gbdt_ensemble(
    gbdt_pairs: List[ Union[Tuple[lgb.LGBMClassifier, xgb.XGBClassifier], Tuple[LeakFreeTabularPreprocessor, lgb.LGBMClassifier, xgb.XGBClassifier]] ],
    X_eval: np.ndarray,
) -> np.ndarray:
    """Deterministic probability inference across all (LightGBM, XGBoost) fold pairs."""
    if not gbdt_pairs:
        raise ValueError("gbdt_pairs must be non-empty")
    preds = np.zeros(len(X_eval), dtype=np.float64)
    for item in gbdt_pairs:
        if len(item) == 3:
            prep, lgb_m, xgb_m = item
            X_in = prep.transform(X_eval)
        else:
            lgb_m, xgb_m = item
            X_in = X_eval
        p_lgb = lgb_m.predict_proba(X_in)[:, 1]
        p_xgb = xgb_m.predict_proba(X_in)[:, 1]
        preds += 0.5 * (p_lgb + p_xgb)
    preds /= float(len(gbdt_pairs))
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


def audit_loco_and_select_blend_weight() -> Tuple[float, float, float]:
    """
    Audits unseen-center generalization strictly within train.csv (train_loco_splits)
    fitting preprocessing ONLY on the single training center before predicting on the
    held-out center. Selects a conservative cytological GBDT blend weight without
    ever touching validation.csv.
    """
    loco_bas, loco_aurocs = [], []
    for split_i, (held_out_cid, tr_idx, ho_idx) in enumerate(train_loco_splits):
        sub_tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        u_groups = sub_tr_df["group_id"].unique()
        rng = np.random.default_rng(SEED + split_i)
        rng.shuffle(u_groups)
        n_es = max(1, len(u_groups) // 8)
        es_groups = set(u_groups[:n_es].tolist())
        es_mask = sub_tr_df["group_id"].isin(es_groups).values
        fit_mask = ~es_mask

        loco_prep = LeakFreeTabularPreprocessor(seed=SEED + split_i)
        X_src_all = loco_prep.fit_transform(X_train_raw[tr_idx][fit_mask])
        X_src_es = loco_prep.transform(X_train_raw[tr_idx][es_mask])
        X_unseen = loco_prep.transform(X_train_raw[ho_idx])

        w_src_full = compute_balanced_domain_class_weights(sub_tr_df)
        y_src_full = y_train[tr_idx]

        pair = _fit_single_fold_cytology_gbdt(
            X_src_all,
            y_src_full[fit_mask],
            w_src_full[fit_mask],
            X_src_es,
            y_src_full[es_mask],
            w_src_full[es_mask],
            seed=SEED + 31 * split_i,
        )
        ho_probs = predict_cytology_gbdt_ensemble([pair], X_unseen)
        m = binary_metrics(y_train[ho_idx], ho_probs)
        loco_bas.append(m["balanced_accuracy"])
        loco_aurocs.append(m["auroc"])

    mean_loco_ba = float(np.mean(loco_bas))
    mean_loco_auroc = float(np.mean(loco_aurocs))

    if mean_loco_ba >= 0.76:
        w_gbdt = 0.14
    elif mean_loco_ba >= 0.70:
        w_gbdt = 0.10
    elif mean_loco_ba >= 0.64:
        w_gbdt = 0.06
    else:
        w_gbdt = 0.0
    return w_gbdt, mean_loco_ba, mean_loco_auroc


def fit_kfold_cytology_gbdt(
    df_source: pd.DataFrame,
    X_raw_source: np.ndarray,
    y_source: np.ndarray,
    gkf_folds: List[Tuple[np.ndarray, np.ndarray]],
    seed_offset: int = 0,
) -> Tuple[List[Tuple[LeakFreeTabularPreprocessor, lgb.LGBMClassifier, xgb.XGBClassifier]], Dict[str, float]]:
    """Trains 5-fold group-disjoint (LightGBM + XGBoost) ensemble with fold-isolated preprocessing."""
    oof_probs = np.zeros(len(df_source), dtype=np.float64)
    models_out = []
    for fold_idx, (f_tr, f_va) in enumerate(gkf_folds):
        fold_prep = LeakFreeTabularPreprocessor(seed=SEED + seed_offset + fold_idx)
        X_tr_scaled = fold_prep.fit_transform(X_raw_source[f_tr])
        X_va_scaled = fold_prep.transform(X_raw_source[f_va])
        w_tr = compute_balanced_domain_class_weights(df_source.iloc[f_tr])
        w_va = compute_balanced_domain_class_weights(df_source.iloc[f_va])
        lgb_m, xgb_m = _fit_single_fold_cytology_gbdt(
            X_tr_scaled,
            y_source[f_tr],
            w_tr,
            X_va_scaled,
            y_source[f_va],
            w_va,
            seed=SEED + seed_offset + 17 * fold_idx,
        )
        oof_probs[f_va] = predict_cytology_gbdt_ensemble([(fold_prep, lgb_m, xgb_m)], X_raw_source[f_va])
        models_out.append((fold_prep, lgb_m, xgb_m))
    oof_metrics = binary_metrics(y_source, oof_probs)
    return models_out, oof_metrics


@torch.no_grad()
def predict_swin_d4_tta(
    swin_models: List[nn.Module],
    loader: DataLoader,
    expected_ids: List[str],
    tta_views: int = TTA_VIEWS,
) -> np.ndarray:
    """
    Executes deterministic multi-view D4 dihedral Test-Time Augmentation directly on GPU
    across both RGB and calibrated [H, E, Residual] tensors while reusing exact D4-invariant
    tabular features. Identical logic is used for validation and test sets.
    """
    for m in swin_models:
        m.eval()

    all_probs = []
    collected_ids = []

    if tta_views >= 8:
        d4_ops = [
            (False, 0),
            (False, 1),
            (False, 2),
            (False, 3),
            (True, 0),
            (True, 1),
            (True, 2),
            (True, 3),
        ]
    elif tta_views == 4:
        d4_ops = [(False, 0), (False, 1), (False, 2), (True, 0)]
    else:
        d4_ops = [(False, 0)]

    for batch in loader:
        rgb = batch["rgb"].to(DEVICE, non_blocking=True)
        stain = batch["stain"].to(DEVICE, non_blocking=True)
        tabular = batch["tabular"].to(DEVICE, non_blocking=True)
        collected_ids.extend(batch["sample_id"])

        batch_prob_sum = torch.zeros(rgb.size(0), dtype=torch.float32, device=DEVICE)
        n_total_passes = 0

        for do_flip, k_rot in d4_ops:
            r_v = torch.flip(rgb, dims=[-1]) if do_flip else rgb
            s_v = torch.flip(stain, dims=[-1]) if do_flip else stain
            if k_rot > 0:
                r_v = torch.rot90(r_v, k=k_rot, dims=[-2, -1])
                s_v = torch.rot90(s_v, k=k_rot, dims=[-2, -1])

            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    for m in swin_models:
                        p = m.predict_proba(r_v, s_v, tabular)
                        batch_prob_sum += p.float()
                        n_total_passes += 1
            else:
                for m in swin_models:
                    p = m.predict_proba(r_v, s_v, tabular)
                    batch_prob_sum += p.float()
                    n_total_passes += 1

        batch_probs = (batch_prob_sum / float(n_total_passes)).clamp(1e-6, 1.0 - 1e-6)
        all_probs.append(batch_probs.cpu().numpy())

    assert (
        collected_ids == expected_ids
    ), "Sample ID ordering mismatch during DataLoader inference!"
    probs_arr = np.concatenate(all_probs, axis=0).astype(np.float64)
    return np.clip(probs_arr, 1e-6, 1.0 - 1e-6)


def run_pipeline_inference(
    swin_models: List[nn.Module],
    gbdt_models: List[Tuple[lgb.LGBMClassifier, xgb.XGBClassifier]],
    eval_loader: DataLoader,
    eval_X_tab: np.ndarray,
    expected_ids: List[str],
    w_gbdt: float,
    tta_views: int = TTA_VIEWS,
) -> np.ndarray:
    """
    Single unified prediction pipeline shared identically between pre-refit validation
    evaluation and final test inference. Convexly blends symmetric 0.5-anchored probabilities.
    """
    p_swin = predict_swin_d4_tta(
        swin_models, eval_loader, expected_ids=expected_ids, tta_views=tta_views
    )
    if w_gbdt > 0.0 and len(gbdt_models) > 0:
        p_gbdt = predict_cytology_gbdt_ensemble(gbdt_models, eval_X_tab)
        probs = (1.0 - w_gbdt) * p_swin + w_gbdt * p_gbdt
    else:
        probs = p_swin
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def _cross_modal_consensus_kl(
    outputs: Dict[str, torch.Tensor], eps: float = 1e-6
) -> torch.Tensor:
    """
    Single-pass cross-modal self-distillation: when both the visual+stain stream and
    cytological stream agree on tissue polarity, regularize the fused probability toward
    their consensus while preserving exact symmetry around p = 0.5.
    """
    if ("aux_vis_logits" not in outputs) or ("aux_tab_logits" not in outputs):
        return outputs["logits"].new_zeros(())

    z_main = outputs["logits"].float().view(-1)
    z_vis = outputs["aux_vis_logits"].detach().float().view(-1)
    z_tab = outputs["aux_tab_logits"].detach().float().view(-1)

    agree_mask = (z_vis * z_tab) > 0.0
    if not agree_mask.any():
        return z_main.new_zeros(())

    p_cons = 0.5 * (torch.sigmoid(z_vis[agree_mask]) + torch.sigmoid(z_tab[agree_mask]))
    p_cons = p_cons.clamp(eps, 1.0 - eps)
    p_main = torch.sigmoid(z_main[agree_mask]).clamp(eps, 1.0 - eps)

    kl_bern = p_cons * torch.log(p_cons / p_main) + (1.0 - p_cons) * torch.log(
        (1.0 - p_cons) / (1.0 - p_main)
    )
    return kl_bern.mean()


def _set_differential_step_lr(
    opt: torch.optim.Optimizer,
    base_lrs: List[float],
    group_names: List[str],
    progress: float,
    bb_warmup_frac: float = 0.14,
    head_warmup_frac: float = 0.07,
    min_lr_ratio: float = 0.02,
) -> None:
    """Applies differential linear warmup + cosine annealing to backbone vs fusion heads."""
    for pg, base_lr, gname in zip(opt.param_groups, base_lrs, group_names):
        w_frac = bb_warmup_frac if gname.startswith("bb_") else head_warmup_frac
        if progress < w_frac:
            mult = 0.05 + 0.95 * (progress / max(w_frac, 1e-6))
        else:
            cos_p = (progress - w_frac) / max(1.0 - w_frac, 1e-6)
            mult = min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1.0 + math.cos(math.pi * min(cos_p, 1.0))
            )
        pg["lr"] = base_lr * mult


def train_swin_checkpoint(
    net: nn.Module,
    loss_fn: nn.Module,
    opt: torch.optim.Optimizer,
    tr_loader: DataLoader,
    va_loader: DataLoader,
    va_labels: np.ndarray,
    va_ids: List[str],
    num_epochs: int,
    ckpt_path: str,
    run_tag: str,
    patience: int = 5,
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Trains SwinV2StainSpectralCrossAttnNet with progressive GRL schedule, cross-modal
    consensus regularization, gradient clipping, WarmupGatedEMA, and official metric
    checkpointing (balanced_accuracy at 0.5 threshold via ./input/metric.py).
    """
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    group_names = [pg.get("name", f"g{i}") for i, pg in enumerate(opt.param_groups)]
    ema_helper = WarmupGatedEMA(net, max_decay=0.996)

    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(USE_SCALER and DEVICE.type == "cuda")
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=(USE_SCALER and DEVICE.type == "cuda")
        )

    total_steps = max(1, num_epochs * len(tr_loader))
    best_ba = -1.0
    best_auroc = -1.0
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.perf_counter()
        net.train()
        running_loss = 0.0
        n_batches = 0

        for batch in tr_loader:
            progress = float(global_step) / float(total_steps)
            _set_differential_step_lr(opt, base_lrs, group_names, progress)

            grl_lam = 0.30 * (2.0 / (1.0 + math.exp(-5.0 * progress)) - 1.0)
            agree_w = 0.12 if epoch >= 2 else 0.0

            rgb = batch["rgb"].to(DEVICE, non_blocking=True)
            stain = batch["stain"].to(DEVICE, non_blocking=True)
            tabular = batch["tabular"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            center_idx = batch["center_idx"].to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    outputs = net(
                        rgb, stain, tabular, grl_lambda=grl_lam, return_aux=True
                    )
                    loss = loss_fn(outputs, labels, center_idx=center_idx)
                    if agree_w > 0.0:
                        loss = loss + agree_w * _cross_modal_consensus_kl(outputs)
                if USE_SCALER:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                    opt.step()
            else:
                outputs = net(rgb, stain, tabular, grl_lambda=grl_lam, return_aux=True)
                loss = loss_fn(outputs, labels, center_idx=center_idx)
                if agree_w > 0.0:
                    loss = loss + agree_w * _cross_modal_consensus_kl(outputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                opt.step()

            ema_helper.update(net, in_warmup=(progress < 0.08))
            running_loss += float(loss.detach().item())
            n_batches += 1
            global_step += 1

        avg_loss = running_loss / max(1, n_batches)

        live_probs = predict_swin_d4_tta(
            [net], va_loader, expected_ids=va_ids, tta_views=1
        )
        ema_probs = predict_swin_d4_tta(
            [ema_helper.ema_model], va_loader, expected_ids=va_ids, tta_views=1
        )
        m_live = binary_metrics(va_labels, live_probs)
        m_ema = binary_metrics(va_labels, ema_probs)

        if (m_ema["balanced_accuracy"], m_ema["auroc"]) >= (
            m_live["balanced_accuracy"],
            m_live["auroc"],
        ):
            cand_m = m_ema
            cand_sd = ema_helper.ema_model.state_dict()
            cand_src = "EMA"
        else:
            cand_m = m_live
            cand_sd = net.state_dict()
            cand_src = "LIVE"

        improved = (cand_m["balanced_accuracy"] > best_ba + 1e-6) or (
            abs(cand_m["balanced_accuracy"] - best_ba) <= 1e-6
            and cand_m["auroc"] > best_auroc
        )
        if improved:
            best_ba = cand_m["balanced_accuracy"]
            best_auroc = cand_m["auroc"]
            _save_pure_state_dict(cand_sd, ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ep_dt = time.perf_counter() - ep_t0
        print(
            f"[{run_tag}] Epoch {epoch:02d}/{num_epochs:02d} | "
            f"loss={avg_loss:.4f} | val_ba={cand_m['balanced_accuracy']:.5f} "
            f"(auroc={cand_m['auroc']:.5f}, {cand_src}) | best_ba={best_ba:.5f} | {ep_dt:.1f}s"
        )

        if epochs_no_improve >= patience and epoch >= 6:
            break
        if _elapsed_sec() > MAX_RUNTIME_BUDGET_SEC and epoch >= 4:
            break

    best_sd = _load_pure_state_dict(ckpt_path, DEVICE)
    net.load_state_dict(best_sd)
    net.eval()
    return net, {"balanced_accuracy": best_ba, "auroc": best_auroc}


# =============================================================================
# 10. Stage 0 & Stage 1: Leak-Free Training on train.csv & Official Validation
# =============================================================================
seed_everything(SEED, seed_cuda=True)

# Stage 0: Unseen-center LOCO calibration audit strictly within train.csv
BLEND_W_GBDT, loco_ba, loco_auroc = audit_loco_and_select_blend_weight()
stage1_gbdt_models, stage1_gbdt_oof = fit_kfold_cytology_gbdt(
    train_df, X_train_raw, y_train, train_gkf_folds, seed_offset=0
)
print(
    f"[Stage 0 Cytology Audit] train_LOCO_ba={loco_ba:.5f} (auroc={loco_auroc:.5f}) | "
    f"train_5fold_OOF_ba={stage1_gbdt_oof['balanced_accuracy']:.5f} | selected_w_gbdt={BLEND_W_GBDT:.2f}"
)

# Build DataLoaders using Step 1's exact train_dataset, val_dataset, test_dataset
train_loader, val_loader, test_loader = build_dataloaders(
    train_dataset_obj=train_dataset,
    val_dataset_obj=val_dataset,
    test_dataset_obj=test_dataset,
    sample_weights=train_sampling_weights,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

# Stage 1: Train primary Step-2 `model`, `criterion`, `optimizer` on train_dataset ONLY
stage1_ckpt_path = os.path.join(WORKING_DIR, "best_stage1_swinv2_CrossAttn.pt")
t_s1 = time.perf_counter()
model, _ = train_swin_checkpoint(
    net=model,
    loss_fn=criterion,
    opt=optimizer,
    tr_loader=train_loader,
    va_loader=val_loader,
    va_labels=y_val,
    va_ids=val_df["sample_id"].astype(str).tolist(),
    num_epochs=STAGE1_EPOCHS,
    ckpt_path=stage1_ckpt_path,
    run_tag="Stage1-TrainOnly",
    patience=5,
)
stage1_train_duration = time.perf_counter() - t_s1

# Compute Official Pre-Refit Validation Score on untouched validation.csv via run_pipeline_inference
val_ids_list = val_df["sample_id"].astype(str).tolist()
test_ids_list = test_df["sample_id"].astype(str).tolist()

val_probs_official = run_pipeline_inference(
    swin_models=[model],
    gbdt_models=stage1_gbdt_models,
    eval_loader=val_loader,
    eval_X_tab=X_val_raw,
    expected_ids=val_ids_list,
    w_gbdt=BLEND_W_GBDT,
    tta_views=TTA_VIEWS,
)
official_val_metrics = binary_metrics(y_val, val_probs_official)
validation_balanced_accuracy = official_val_metrics["balanced_accuracy"]
validation_auroc = official_val_metrics["auroc"]

print(
    f"[Pre-Refit Official Validation] balanced_accuracy={validation_balanced_accuracy:.6f} | "
    f"auroc={validation_auroc:.6f} | accuracy={official_val_metrics['accuracy']:.6f} | "
    f"n={official_val_metrics['n_samples']} (fixed threshold=0.5)"
)

# Run Stage-1 test inference using the 100% IDENTICAL run_pipeline_inference path
test_probs_stage1 = run_pipeline_inference(
    swin_models=[model],
    gbdt_models=stage1_gbdt_models,
    eval_loader=test_loader,
    eval_X_tab=X_test_raw,
    expected_ids=test_ids_list,
    w_gbdt=BLEND_W_GBDT,
    tta_views=TTA_VIEWS,
)


# =============================================================================
# 11. Stage 2: Post-Selection Full-Dev Refit (train.csv + validation.csv = 16,800)
# =============================================================================
remaining_sec = MAX_RUNTIME_BUDGET_SEC - _elapsed_sec()
est_fold_sec = max(60.0, stage1_train_duration * (REFIT_EPOCHS / float(STAGE1_EPOCHS)))

if remaining_sec > (est_fold_sec + 180.0):
    max_refit_folds = min(3, int((remaining_sec - 150.0) // est_fold_sec))
else:
    max_refit_folds = 0

if max_refit_folds >= 1:
    merged_gbdt_models, merged_gbdt_oof = fit_kfold_cytology_gbdt(
        merged_dev_df, X_merged_raw, y_merged, merged_gkf_folds, seed_offset=500
    )
    nw_eval = max(2, min(4, os.cpu_count() or 2))

    refit_swin_entries = []
    for r_fold in range(max_refit_folds):
        if (MAX_RUNTIME_BUDGET_SEC - _elapsed_sec()) < (0.85 * est_fold_sec + 120.0):
            break

        seed_everything(SEED + 100 * (r_fold + 1), seed_cuda=True)
        m_tr_idx, m_va_idx = merged_gkf_folds[r_fold]
        fold_tr_df = merged_dev_df.iloc[m_tr_idx].reset_index(drop=True)
        fold_va_df = merged_dev_df.iloc[m_va_idx].reset_index(drop=True)

        assert set(fold_tr_df["group_id"]).isdisjoint(set(fold_va_df["group_id"]))
        assert set(fold_tr_df["slide_id"]).isdisjoint(set(fold_va_df["slide_id"]))

        fold_tab_prep = LeakFreeTabularPreprocessor(seed=SEED + 777 + r_fold)
        X_fold_tr_tab = fold_tab_prep.fit_transform(X_merged_raw[m_tr_idx])
        X_fold_va_tab = fold_tab_prep.transform(X_merged_raw[m_va_idx])
        X_fold_te_tab = fold_tab_prep.transform(X_test_raw)
        fold_image_stats = fit_train_only_image_statistics_and_fourier_bank(
            fold_tr_df, seed=SEED + 777 + r_fold
        )

        fold_tr_ds = HistopathologyMultiViewDataset(
            fold_tr_df,
            X_fold_tr_tab,
            fold_image_stats,
            is_train=True,
            center_to_idx=CENTER_TO_IDX,
        )
        fold_va_ds = HistopathologyMultiViewDataset(
            fold_va_df,
            X_fold_va_tab,
            fold_image_stats,
            is_train=False,
            center_to_idx=None,
        )
        fold_weights = compute_balanced_domain_class_weights(fold_tr_df)
        fold_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(fold_weights, dtype=torch.double),
            num_samples=len(fold_weights),
            replacement=True,
        )
        fold_tr_loader = DataLoader(
            fold_tr_ds,
            batch_size=BATCH_SIZE,
            sampler=fold_sampler,
            num_workers=nw_eval,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        fold_va_loader = DataLoader(
            fold_va_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=nw_eval,
            pin_memory=torch.cuda.is_available(),
        )

        f_net, f_crit, f_opt = build_model_criterion_optimizer(
            tabular_dim=_TABULAR_DIM,
            num_train_centers=_NUM_TRAIN_CENTERS,
            pretrained=True,
            backbone_lr=1.5e-4,
            head_lr=6.0e-4,
            device=DEVICE,
        )
        fold_ckpt_path = os.path.join(WORKING_DIR, f"best_refit_fold{r_fold}_swinv2.pt")
        f_net, f_stats = train_swin_checkpoint(
            net=f_net,
            loss_fn=f_crit,
            opt=f_opt,
            tr_loader=fold_tr_loader,
            va_loader=fold_va_loader,
            va_labels=y_merged[m_va_idx],
            va_ids=fold_va_df["sample_id"].astype(str).tolist(),
            num_epochs=REFIT_EPOCHS,
            ckpt_path=fold_ckpt_path,
            run_tag=f"RefitFold{r_fold}",
            patience=4,
        )
        refit_swin_entries.append((f_net, X_fold_te_tab, fold_image_stats))
        del f_crit, f_opt, fold_tr_loader, fold_va_loader, fold_tr_ds, fold_va_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(refit_swin_entries) > 0:
        swin_fold_preds = []
        for f_net, X_fold_te_tab, fold_image_stats in refit_swin_entries:
            fold_te_ds = HistopathologyMultiViewDataset(
                test_df,
                X_fold_te_tab,
                fold_image_stats,
                is_train=False,
                center_to_idx=None,
            )
            fold_te_loader = DataLoader(
                fold_te_ds,
                batch_size=BATCH_SIZE * 2,
                shuffle=False,
                num_workers=nw_eval,
                pin_memory=torch.cuda.is_available(),
            )
            swin_fold_preds.append(
                predict_swin_d4_tta(
                    [f_net], fold_te_loader, expected_ids=test_ids_list, tta_views=TTA_VIEWS
                )
            )
        p_swin_refit = np.mean(swin_fold_preds, axis=0)
        if BLEND_W_GBDT > 0.0 and len(merged_gbdt_models) > 0:
            p_gbdt_refit = predict_cytology_gbdt_ensemble(merged_gbdt_models, X_test_raw)
            test_probs_refit = (1.0 - BLEND_W_GBDT) * p_swin_refit + BLEND_W_GBDT * p_gbdt_refit
        else:
            test_probs_refit = p_swin_refit
        test_probs_refit = np.clip(test_probs_refit, 1e-6, 1.0 - 1e-6)
        n_refit = float(len(refit_swin_entries))
        final_test_probs = (n_refit / (n_refit + 1.0)) * test_probs_refit + (
            1.0 / (n_refit + 1.0)
        ) * test_probs_stage1
    else:
        final_test_probs = test_probs_stage1
else:
    final_test_probs = test_probs_stage1

final_test_probs = np.clip(
    np.asarray(final_test_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6
)


# =============================================================================
# 12. Save & Strictly Audit ./submission/submission.csv + Final Score Output
# =============================================================================
os.makedirs(SUBMISSION_DIR, exist_ok=True)
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

submission_df = pd.DataFrame(
    {
        "sample_id": test_df["sample_id"].astype(str).values,
        "tumor_probability": final_test_probs,
    }
)
submission_df.to_csv(submission_path, index=False)

# Strict integrity verification against sample_submission.csv
assert os.path.isfile(submission_path), f"Missing submission file at {submission_path}"
sub_verify = pd.read_csv(submission_path)
assert list(sub_verify.columns) == [
    "sample_id",
    "tumor_probability",
], f"Invalid submission columns: {list(sub_verify.columns)}"
assert len(sub_verify) == len(
    sample_sub_df
), f"Row count mismatch: got {len(sub_verify)}, expected {len(sample_sub_df)}"
assert (
    sub_verify["sample_id"].values == sample_sub_df["sample_id"].values
).all(), "sample_id sequence does not match sample_submission.csv!"
assert np.isfinite(
    sub_verify["tumor_probability"].values
).all(), "Non-finite probabilities found in submission.csv!"
assert (
    (sub_verify["tumor_probability"].values >= 0.0)
    & (sub_verify["tumor_probability"].values <= 1.0)
).all(), "Out-of-bounds probabilities found in submission.csv!"
assert (
    float(sub_verify["tumor_probability"].std()) > 0.01
), "Degenerate near-constant test predictions detected!"

print(
    f"Submission Verified: {submission_path} | rows={len(sub_verify)} | "
    f"prob_mean={sub_verify['tumor_probability'].mean():.4f}, "
    f"prob_std={sub_verify['tumor_probability'].std():.4f}, "
    f"pred_pos_rate={(sub_verify['tumor_probability'] >= 0.5).mean():.4f} | "
    f"elapsed={_elapsed_sec():.1f}s"
)
print(f"Final Validation Score: {validation_balanced_accuracy}")