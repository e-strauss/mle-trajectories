"""
End-to-End Histopathology Patch Classification Pipeline
-------------------------------------------------------
Combines:
  1. Physics-Informed Optical Stain Standardization, Dual-Stain D4 Multi-View
     Extraction, Compact Beer-Lambert OD Morphological Descriptors, and
     Leakage-Free Slide-Grouped + LOCO Data Splits.
  2. Strictly Frozen 3-Pathology-Foundation-Model Representation Bank
     (owkin/phikon-v2, Kaiko ViT-L/14-reg4 with native 0.5/0.5/0.5 norm,
     histai/hibou-L with Kaiko ViT-B/8 fallback) paired with 4 complementary
     downstream paradigms: Frozen-Backbone Spatial-MIL Probe, Multi-PFM
     Domain-Invariant MLP (DualPrototypeCosineHead + twin stain consistency),
     Center-Debiased Logistic Probe, and Fold-Internal LightGBM/XGBoost GBDTs.
  3. 5-Fold Slide-Grouped + 2-Fold LOCO CV Stacking (joint alpha_canon and
     bounded non-negative simplex logit weighting strictly inside train.csv),
     Official Pre-Refit Validation (./input/metric.py), Time-Guarded 5-Fold
     Bagged Train+Val Refit, and Verified Submission (./submission/submission.csv).
"""

import copy
import gc
import importlib.util
import inspect
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

# Prevent OpenCV thread oversubscription inside PyTorch DataLoader workers
cv2.setNumThreads(0)

# Standard paths (read-only ./input/, artifacts in ./working/, outputs in ./submission/)
INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
SUBMISSION_DIR = Path("./submission")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# Standard ImageNet normalization stats for pretrained RGB backbones (ViT / DINOv2 / Swin)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Ordered pretrained backbone candidates with provenance documentation and
# explicit native normalization overrides (fixing Kaiko 0.5/0.5/0.5 remapping).
PRETRAINED_BACKBONE_CANDIDATES: Tuple[Dict[str, Any], ...] = (
    {
        "name": "owkin/phikon-v2",
        "pretraining_data": "PANCAN-400M histopathology tiles (Apache-2.0 Owkin DINOv2 ViT-L/16)",
        "family": "vit",
        "loader": "hf_dinov2",
        "trust_remote_code": False,
        "norm_mean": [0.485, 0.456, 0.406],
        "norm_std": [0.229, 0.224, 0.225],
    },
    {
        "name": "hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms",
        "pretraining_data": "TCGA-only histopathology WSI patches (Kaiko.ai DINOv2 ViT-L/14 reg4; zero Camelyon/lymph-node overlap)",
        "family": "vit",
        "loader": "timm",
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
    },
    {
        "name": "histai/hibou-L",
        "pretraining_data": "1M+ proprietary diverse pathology WSIs (HistAI Hibou-L DINOv2 ViT-L/14 reg4)",
        "family": "vit",
        "loader": "hf_dinov2",
        "trust_remote_code": True,
        "norm_mean": [0.7068, 0.5755, 0.7220],
        "norm_std": [0.1950, 0.2316, 0.1816],
    },
    {
        "name": "hf-hub:1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms",
        "pretraining_data": "TCGA-only histopathology WSI patches (Kaiko.ai DINOv2 ViT-B/8; zero Camelyon/lymph-node overlap)",
        "family": "vit",
        "loader": "timm",
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
    },
    {
        "name": "hf-hub:1aurent/vit_large_patch14_224.kaiko_ai_towards_large_pathology_fms",
        "pretraining_data": "TCGA-only histopathology WSI patches (Kaiko.ai DINOv2 ViT-L/14; zero Camelyon/lymph-node overlap)",
        "family": "vit",
        "loader": "timm",
        "norm_mean": [0.5, 0.5, 0.5],
        "norm_std": [0.5, 0.5, 0.5],
    },
    {
        "name": "vit_large_patch14_reg4_dinov2.lvd142m",
        "pretraining_data": "LVD-142M (self-supervised natural images; no known histopathology label overlap)",
        "family": "vit",
    },
    {
        "name": "vit_large_patch14_dinov2.lvd142m",
        "pretraining_data": "LVD-142M (self-supervised natural images; no known histopathology label overlap)",
        "family": "vit",
    },
    {
        "name": "vit_base_patch14_reg4_dinov2.lvd142m",
        "pretraining_data": "LVD-142M (self-supervised natural images; no known histopathology label overlap)",
        "family": "vit",
    },
    {
        "name": "vit_base_patch14_dinov2.lvd142m",
        "pretraining_data": "LVD-142M (self-supervised natural images; no known histopathology label overlap)",
        "family": "vit",
    },
    {
        "name": "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320",
        "pretraining_data": "LAION-2B OpenCLIP -> ImageNet-12k -> ImageNet-1k (natural images; no known histopathology label overlap)",
        "family": "convnext",
    },
    {
        "name": "convnext_large.fb_in22k_ft_in1k",
        "pretraining_data": "ImageNet-22k -> ImageNet-1k (natural images; no known histopathology label overlap)",
        "family": "convnext",
    },
    {
        "name": "convnext_base.fb_in22k_ft_in1k",
        "pretraining_data": "ImageNet-22k -> ImageNet-1k (natural images; no known histopathology label overlap)",
        "family": "convnext",
    },
    {
        "name": "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
        "pretraining_data": "ImageNet-22k -> ImageNet-1k (natural images; no known histopathology overlap)",
        "family": "swinv2",
    },
    {
        "name": "swinv2_small_window8_256.ms_in1k",
        "pretraining_data": "ImageNet-1k (natural images; no known histopathology overlap)",
        "family": "swinv2",
    },
    {
        "name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "pretraining_data": "ImageNet-21k -> ImageNet-1k (natural images; no known histopathology overlap)",
        "family": "vit",
    },
)


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# STEP 1: DATA PROCESSING & PHYSICS-INFORMED FEATURE ENGINEERING
# =============================================================================
class BeerLambertHED:
    """
    Physically grounded Beer-Lambert optical density (OD) stain deconvolution
    using the Ruifrok & Johnston H&E basis, supporting:
      1. Exact lossless RGB <-> OD <-> [Hematoxylin, Eosin, Residual] conversion.
      2. Foreground tissue soft-mask estimation from optical density magnitude.
      3. Realistic unseen-center H&E stain-matrix rotation and concentration jitter.
    """

    _RGB_TO_OD_LUT: np.ndarray = (
        -np.log10((np.arange(256, dtype=np.float32) + 1.0) / 256.0)
    ).astype(np.float32)
    _LN10: np.float32 = np.float32(np.log(10.0))

    def __init__(self) -> None:
        h_vec = np.array([0.650, 0.704, 0.286], dtype=np.float32)
        e_vec = np.array([0.072, 0.990, 0.105], dtype=np.float32)
        h_vec = h_vec / np.linalg.norm(h_vec)
        e_vec = e_vec / np.linalg.norm(e_vec)
        r_vec = np.cross(h_vec, e_vec)
        r_vec = r_vec / np.linalg.norm(r_vec)

        self.stain_matrix = np.stack([h_vec, e_vec, r_vec], axis=0).astype(np.float32)
        self.stain_matrix_inv = np.linalg.inv(self.stain_matrix).astype(np.float32)

    @classmethod
    def rgb_to_od(cls, rgb_uint8: np.ndarray) -> np.ndarray:
        """Convert RGB uint8 [0, 255] to optical density OD = -log10((I + 1) / 256)."""
        if rgb_uint8.dtype == np.uint8:
            return cls._RGB_TO_OD_LUT[rgb_uint8]
        rgb_f = rgb_uint8.astype(np.float32)
        return -np.log10((rgb_f + 1.0) / 256.0)

    @classmethod
    def od_to_rgb(cls, od: np.ndarray) -> np.ndarray:
        """Convert optical density back to RGB uint8 [0, 255]."""
        od_clipped = np.clip(od, np.float32(0.0), np.float32(3.0)).astype(
            np.float32, copy=False
        )
        rgb = np.float32(256.0) * np.exp(-od_clipped * cls._LN10) - np.float32(1.0)
        return np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def compute_soft_tissue_mask(
        od: np.ndarray, low: float = 0.05, high: float = 0.20
    ) -> np.ndarray:
        """
        Smooth foreground tissue mask in [0, 1] derived from optical density magnitude.
        Avoids hard binary edges between white glass/lumen and stained cells.
        """
        od_mag = np.linalg.norm(od, axis=-1)
        return np.clip((od_mag - low) / max(high - low, 1e-6), 0.0, 1.0).astype(
            np.float32
        )

    def rgb_to_hed_concentrations(
        self, rgb_uint8: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose RGB patch into (concentrations [H, W, 3], od [H, W, 3])."""
        od = self.rgb_to_od(rgb_uint8)
        conc = od @ self.stain_matrix_inv
        return conc.astype(np.float32), od.astype(np.float32)

    def perturb_stain(
        self,
        rgb_uint8: np.ndarray,
        rng: np.random.Generator,
        conc_scale_range: Tuple[float, float] = (0.78, 1.28),
        conc_bias_range: Tuple[float, float] = (-0.05, 0.05),
        matrix_jitter_std: float = 0.04,
    ) -> np.ndarray:
        """
        Simulate unseen-hospital H&E staining chemistry and scanner spectral shift
        by perturbing both per-stain concentrations and the H&E stain basis matrix.
        """
        conc, od = self.rgb_to_hed_concentrations(rgb_uint8)
        tissue_w = self.compute_soft_tissue_mask(od)[..., None]

        # Independent multiplicative and tissue-gated additive concentration perturbation
        scales = np.array(
            [
                rng.uniform(conc_scale_range[0], conc_scale_range[1]),
                rng.uniform(conc_scale_range[0], conc_scale_range[1]),
                rng.uniform(0.85, 1.15),
            ],
            dtype=np.float32,
        )
        biases = np.array(
            [
                rng.uniform(conc_bias_range[0], conc_bias_range[1]),
                rng.uniform(conc_bias_range[0], conc_bias_range[1]),
                rng.uniform(-0.02, 0.02),
            ],
            dtype=np.float32,
        )
        conc_pert = conc * scales[None, None, :] + biases[None, None, :] * tissue_w

        # Slightly rotate Hematoxylin and Eosin stain absorption vectors
        h_pert = self.stain_matrix[0] + rng.normal(
            0.0, matrix_jitter_std, size=3
        ).astype(np.float32)
        e_pert = self.stain_matrix[1] + rng.normal(
            0.0, matrix_jitter_std, size=3
        ).astype(np.float32)
        h_pert = np.clip(h_pert, 0.02, 1.0)
        e_pert = np.clip(e_pert, 0.02, 1.0)
        h_pert /= np.linalg.norm(h_pert)
        e_pert /= np.linalg.norm(e_pert)
        r_pert = np.cross(h_pert, e_pert)
        r_norm = np.linalg.norm(r_pert)
        if r_norm < 1e-4:
            r_pert = self.stain_matrix[2]
        else:
            r_pert = (r_pert / r_norm).astype(np.float32)

        pert_matrix = np.stack([h_pert, e_pert, r_pert], axis=0)
        od_pert = np.clip(conc_pert @ pert_matrix, 0.0, 3.0)
        # Preserve background optical density smoothly while updating tissue
        od_out = (1.0 - tissue_w) * od + tissue_w * od_pert
        return self.od_to_rgb(od_out)


def read_rgb_image(image_path: Path) -> np.ndarray:
    """Read PNG image safely as RGB uint8 numpy array."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_patch_tissue_lab_stats(
    rgb_uint8: np.ndarray, hed: BeerLambertHED
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Extract foreground-tissue LAB mean/std, tissue fraction, and Beer-Lambert
    HED concentration mean/std for a single patch.
    """
    conc, od = hed.rgb_to_hed_concentrations(rgb_uint8)
    tissue_mask = np.linalg.norm(od, axis=-1) > 0.12
    tissue_frac = float(tissue_mask.mean())

    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    if tissue_frac >= 0.05:
        pixels = lab[tissue_mask]
    else:
        pixels = lab.reshape(-1, 3)

    lab_mean = pixels.mean(axis=0).astype(np.float32)
    lab_std = np.maximum(pixels.std(axis=0), 2.0).astype(np.float32)

    # Full-patch HED concentration mean/std for normalizing auxiliary OD tensors
    conc_flat = conc.reshape(-1, 3)
    od_mean = conc_flat.mean(axis=0).astype(np.float32)
    od_std = np.maximum(conc_flat.std(axis=0), 1e-3).astype(np.float32)
    return lab_mean, lab_std, tissue_frac, od_mean, od_std


def extract_od_morphological_features(
    rgb_uint8: np.ndarray, hed: Optional[BeerLambertHED] = None
) -> np.ndarray:
    """
    Extract a compact, fixed-length (46-dim) float32 Beer-Lambert Optical Density
    morphological descriptor vector from a raw RGB patch (100% pixel-only):
      - H, E, R channel quantiles (p10, p25, p50, p75, p90, p95), mean, std (24)
      - Tissue fractions at multiple OD thresholds + foreground H/E means (5)
      - High-Hematoxylin nuclear density fractions & top-10% H intensity (4)
      - 4x4 spatial quadrant variance / max / spread for H, E, and tissue (9)
      - Global, median, p90, and top-quadrant Hematoxylin-to-Eosin ratios (4)
    """
    if hed is None:
        hed = BeerLambertHED()

    conc, od = hed.rgb_to_hed_concentrations(rgb_uint8)
    od_mag = np.linalg.norm(od, axis=-1)
    h_ch = conc[..., 0]
    e_ch = conc[..., 1]
    r_ch = conc[..., 2]

    feats: List[float] = []
    qs = (10.0, 25.0, 50.0, 75.0, 90.0, 95.0)
    for ch in (h_ch, e_ch, r_ch):
        flat = ch.reshape(-1)
        q_vals = np.percentile(flat, qs)
        feats.extend(float(v) for v in q_vals)
        feats.append(float(flat.mean()))
        feats.append(float(flat.std()))

    tissue_12 = od_mag > 0.12
    tissue_25 = od_mag > 0.25
    tissue_40 = od_mag > 0.40
    tf12 = float(tissue_12.mean())
    feats.append(tf12)
    feats.append(float(tissue_25.mean()))
    feats.append(float(tissue_40.mean()))

    if tf12 >= 0.02:
        h_fg = h_ch[tissue_12]
        e_fg = e_ch[tissue_12]
        h_fg_mean = float(h_fg.mean())
        e_fg_mean = float(e_fg.mean())
    else:
        h_fg = h_ch.reshape(-1)
        e_fg = e_ch.reshape(-1)
        h_fg_mean = float(h_fg.mean())
        e_fg_mean = float(e_fg.mean())
    feats.append(h_fg_mean)
    feats.append(e_fg_mean)

    # High-Hematoxylin nuclear density & extreme hyperchromatic nuclei cues
    feats.append(float((h_ch > 0.35).mean()))
    feats.append(float((h_ch > 0.55).mean()))
    feats.append(float((h_ch > 0.75).mean()))
    h_p90 = float(np.percentile(h_ch, 90.0))
    top_h_mask = h_ch >= h_p90
    feats.append(float(h_ch[top_h_mask].mean()) if bool(top_h_mask.any()) else h_p90)

    # 4x4 spatial quadrant heterogeneity (captures focal tumor nests vs uniform stroma)
    H_img, W_img = h_ch.shape
    qh = max(1, H_img // 4)
    qw = max(1, W_img // 4)
    h_crop = h_ch[: qh * 4, : qw * 4].reshape(4, qh, 4, qw)
    e_crop = e_ch[: qh * 4, : qw * 4].reshape(4, qh, 4, qw)
    t_crop = tissue_12[: qh * 4, : qw * 4].astype(np.float32).reshape(4, qh, 4, qw)

    q_h_means = h_crop.mean(axis=(1, 3)).reshape(-1)
    q_e_means = e_crop.mean(axis=(1, 3)).reshape(-1)
    q_t_means = t_crop.mean(axis=(1, 3)).reshape(-1)

    feats.append(float(q_h_means.var()))
    feats.append(float(q_h_means.max()))
    feats.append(float(q_h_means.max() - q_h_means.min()))
    feats.append(float(q_e_means.var()))
    feats.append(float(q_e_means.max()))
    feats.append(float(q_e_means.max() - q_e_means.min()))
    feats.append(float(q_t_means.var()))
    feats.append(float(q_t_means.max()))
    feats.append(float(q_t_means.min()))

    # Hematoxylin-to-Eosin ratios (nuclear-to-cytoplasmic density proxies)
    eps = 0.05
    h_med = float(np.median(h_fg))
    e_med = float(np.median(e_fg))
    e_p90 = float(np.percentile(e_ch, 90.0))
    top_q_idx = int(np.argmax(q_h_means))
    feats.append(float(np.clip((h_fg_mean + eps) / (abs(e_fg_mean) + eps), -10.0, 10.0)))
    feats.append(float(np.clip((h_med + eps) / (abs(e_med) + eps), -10.0, 10.0)))
    feats.append(float(np.clip((h_p90 + eps) / (abs(e_p90) + eps), -10.0, 10.0)))
    feats.append(
        float(
            np.clip(
                (float(q_h_means[top_q_idx]) + eps)
                / (abs(float(q_e_means[top_q_idx])) + eps),
                -10.0,
                10.0,
            )
        )
    )

    arr = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)


class StainReferenceBank:
    """
    Fits canonical stain normalization targets and a per-slide / per-center
    stain style bank STRICTLY on training data (zero validation/test leakage).

    Used for:
      - Canonical Stain Standardization (identical across train, val, and test)
      - Virtual-Center Stain Synthesis during training via inter-center
        interpolation/extrapolation in LAB space
      - Training-fitted standardization constants for Beer-Lambert OD channels
    """

    def __init__(self) -> None:
        self.canonical_lab_mean: Optional[np.ndarray] = None
        self.canonical_lab_std: Optional[np.ndarray] = None
        self.od_channel_mean: Optional[np.ndarray] = None
        self.od_channel_std: Optional[np.ndarray] = None
        self.slide_stats: List[Dict] = []
        self.center_stats: Dict[str, Dict] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        input_dir: Path = INPUT_DIR,
        max_patches_per_slide: int = 36,
        seed: int = 42,
        num_threads: int = 8,
    ) -> "StainReferenceBank":
        """Fit slide-level and center-balanced canonical stain stats on train_df ONLY."""
        rng = np.random.default_rng(seed)
        hed = BeerLambertHED()

        sampled_rows = []
        for (_, _), grp in train_df.groupby(["center_id", "slide_id"], sort=True):
            if len(grp) > max_patches_per_slide:
                idx = rng.choice(len(grp), size=max_patches_per_slide, replace=False)
                sampled_rows.append(grp.iloc[idx])
            else:
                sampled_rows.append(grp)
        sample_df = pd.concat(sampled_rows, ignore_index=True)

        paths = [input_dir / p for p in sample_df["image_path"].tolist()]

        def _extract_one(path: Path):
            rgb = read_rgb_image(path)
            return compute_patch_tissue_lab_stats(rgb, hed)

        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            results = list(ex.map(_extract_one, paths))

        lab_means = np.stack([r[0] for r in results], axis=0)
        lab_stds = np.stack([r[1] for r in results], axis=0)
        tissue_fracs = np.array([r[2] for r in results], dtype=np.float32)

        sample_df = sample_df.copy()
        sample_df["_idx"] = np.arange(len(sample_df))

        self.slide_stats = []
        center_buckets: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for (center_id, slide_id), grp in sample_df.groupby(
            ["center_id", "slide_id"], sort=True
        ):
            idxs = grp["_idx"].values
            weights = np.maximum(tissue_fracs[idxs], 0.05)
            weights = weights / weights.sum()

            s_mean = (lab_means[idxs] * weights[:, None]).sum(axis=0)
            s_std = (lab_stds[idxs] * weights[:, None]).sum(axis=0)

            c_key = str(center_id)
            self.slide_stats.append(
                {
                    "center_id": c_key,
                    "slide_id": str(slide_id),
                    "lab_mean": s_mean.tolist(),
                    "lab_std": s_std.tolist(),
                    "n_sampled": int(len(idxs)),
                }
            )
            center_buckets.setdefault(c_key, []).append((s_mean, s_std))

        # Compute center-balanced canonical stain template so all training centers
        # contribute equally regardless of patch count imbalance.
        self.center_stats = {}
        center_mean_list = []
        center_std_list = []
        for c_key, items in sorted(center_buckets.items()):
            c_means = np.stack([x[0] for x in items], axis=0)
            c_stds = np.stack([x[1] for x in items], axis=0)
            mu_mean = c_means.mean(axis=0)
            mu_std_across_slides = np.maximum(c_means.std(axis=0), 1.5)
            sd_mean = c_stds.mean(axis=0)
            sd_std_across_slides = np.maximum(c_stds.std(axis=0), 1.0)
            self.center_stats[c_key] = {
                "lab_mean_center": mu_mean.tolist(),
                "lab_mean_spread": mu_std_across_slides.tolist(),
                "lab_std_center": sd_mean.tolist(),
                "lab_std_spread": sd_std_across_slides.tolist(),
                "n_slides": int(len(items)),
            }
            center_mean_list.append(mu_mean)
            center_std_list.append(sd_mean)

        self.canonical_lab_mean = np.mean(
            np.stack(center_mean_list, axis=0), axis=0
        ).astype(np.float32)
        self.canonical_lab_std = np.mean(
            np.stack(center_std_list, axis=0), axis=0
        ).astype(np.float32)

        # Compute training-only Beer-Lambert OD channel standardization statistics
        # directly on raw training RGB patches so OD tensors match raw inference stats.
        od_sub_idx = rng.choice(len(paths), size=min(400, len(paths)), replace=False)

        def _raw_od_stats(i: int):
            rgb = read_rgb_image(paths[i])
            conc, _ = hed.rgb_to_hed_concentrations(rgb)
            flat = conc.reshape(-1, 3)
            return flat.mean(axis=0), flat.var(axis=0)

        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            od_raw = list(ex.map(_raw_od_stats, od_sub_idx.tolist()))

        raw_means = np.stack([x[0] for x in od_raw], axis=0)
        raw_vars = np.stack([x[1] for x in od_raw], axis=0)
        global_od_mean = raw_means.mean(axis=0)
        global_od_var = raw_vars.mean(axis=0) + raw_means.var(axis=0)

        self.od_channel_mean = global_od_mean.astype(np.float32)
        self.od_channel_std = np.maximum(np.sqrt(global_od_var), 1e-3).astype(
            np.float32
        )
        self.is_fitted = True
        return self

    def sample_virtual_center_target(
        self,
        rng: np.random.Generator,
        extrapolation_range: Tuple[float, float] = (-0.35, 1.35),
        noise_scale: float = 0.6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synthesize a virtual unseen medical center stain target (lab_mean, lab_std)
        by interpolating/extrapolating between random training slides (preferring
        cross-center pairs) plus isotropic spread perturbation.
        """
        if not self.is_fitted or not self.slide_stats:
            raise RuntimeError("StainReferenceBank must be fitted before sampling.")

        n_slides = len(self.slide_stats)
        if n_slides == 1:
            base_m = np.array(self.slide_stats[0]["lab_mean"], dtype=np.float32)
            base_s = np.array(self.slide_stats[0]["lab_std"], dtype=np.float32)
            virt_m = base_m + rng.normal(0.0, [8.0, 5.0, 5.0]).astype(np.float32)
            virt_s = base_s * rng.uniform(0.8, 1.2, size=3).astype(np.float32)
        else:
            s1 = self.slide_stats[int(rng.integers(0, n_slides))]
            other_center_slides = [
                s for s in self.slide_stats if s["center_id"] != s1["center_id"]
            ]
            if other_center_slides and rng.random() < 0.75:
                s2 = other_center_slides[int(rng.integers(0, len(other_center_slides)))]
            else:
                s2 = self.slide_stats[int(rng.integers(0, n_slides))]

            lam = float(rng.uniform(extrapolation_range[0], extrapolation_range[1]))
            m1, sd1 = np.array(s1["lab_mean"], dtype=np.float32), np.array(
                s1["lab_std"], dtype=np.float32
            )
            m2, sd2 = np.array(s2["lab_mean"], dtype=np.float32), np.array(
                s2["lab_std"], dtype=np.float32
            )

            virt_m = m1 * (1.0 - lam) + m2 * lam
            virt_s = sd1 * (1.0 - lam) + sd2 * lam

            c_info = self.center_stats[s1["center_id"]]
            m_spread = np.array(c_info["lab_mean_spread"], dtype=np.float32)
            s_spread = np.array(c_info["lab_std_spread"], dtype=np.float32)
            virt_m = virt_m + rng.normal(0.0, noise_scale * m_spread).astype(np.float32)
            virt_s = virt_s + rng.normal(0.0, noise_scale * s_spread).astype(np.float32)

        virt_m = np.clip(virt_m, [80.0, 120.0, 95.0], [225.0, 185.0, 155.0]).astype(
            np.float32
        )
        virt_s = np.clip(virt_s, [8.0, 4.0, 4.0], [65.0, 32.0, 32.0]).astype(np.float32)
        return virt_m, virt_s

    def save(self, path: Path) -> None:
        """Persist fitted training stain reference bank to JSON for auditability."""
        payload = {
            "canonical_lab_mean": self.canonical_lab_mean.tolist(),
            "canonical_lab_std": self.canonical_lab_std.tolist(),
            "od_channel_mean": self.od_channel_mean.tolist(),
            "od_channel_std": self.od_channel_std.tolist(),
            "center_stats": self.center_stats,
            "slide_stats": self.slide_stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StainReferenceBank":
        """Load a previously fitted StainReferenceBank from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        bank = cls()
        bank.canonical_lab_mean = np.array(
            payload["canonical_lab_mean"], dtype=np.float32
        )
        bank.canonical_lab_std = np.array(
            payload["canonical_lab_std"], dtype=np.float32
        )
        bank.od_channel_mean = np.array(payload["od_channel_mean"], dtype=np.float32)
        bank.od_channel_std = np.array(payload["od_channel_std"], dtype=np.float32)
        bank.center_stats = payload["center_stats"]
        bank.slide_stats = payload["slide_stats"]
        bank.is_fitted = True
        return bank


class TissueMaskedStainNormalizer:
    """
    Soft Tissue-Masked Reinhard LAB Stain Standardization.

    Projects histopathology patches toward a target LAB template while:
      - Estimating source statistics strictly on foreground stained tissue pixels.
      - Clipping scale ratios to [0.45, 2.25] to avoid amplifying sensor noise.
      - Blending smoothly via optical-density soft tissue weights so white glass/lumen
        regions stay clean while background chromatic casts are attenuated.
      - Dampening strength automatically if a patch is mostly empty white space.
    """

    def __init__(
        self,
        target_lab_mean: np.ndarray,
        target_lab_std: np.ndarray,
        strength: float = 0.75,
        ratio_clip: Tuple[float, float] = (0.45, 2.25),
    ) -> None:
        self.target_mean = np.asarray(target_lab_mean, dtype=np.float32)
        self.target_std = np.asarray(target_lab_std, dtype=np.float32)
        self.strength = float(strength)
        self.ratio_clip = ratio_clip
        self.hed = BeerLambertHED()

    def apply_with_target(
        self,
        rgb_uint8: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        strength: Optional[float] = None,
    ) -> np.ndarray:
        alpha = self.strength if strength is None else float(strength)
        if alpha <= 0.0:
            return rgb_uint8

        od = self.hed.rgb_to_od(rgb_uint8)
        od_mag = np.linalg.norm(od, axis=-1)
        hard_tissue = od_mag > 0.12
        tissue_frac = float(hard_tissue.mean())

        if tissue_frac < 0.04:
            return rgb_uint8
        eff_alpha = alpha * min(1.0, tissue_frac / 0.15)

        lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
        tissue_pixels = lab[hard_tissue]
        src_mean = tissue_pixels.mean(axis=0)
        src_std = np.maximum(tissue_pixels.std(axis=0), 2.5)

        scale = np.clip(target_std / src_std, self.ratio_clip[0], self.ratio_clip[1])
        lab_norm = (lab - src_mean[None, None, :]) * scale[None, None, :] + target_mean[
            None, None, :
        ]

        soft_w = np.clip((od_mag - 0.05) / 0.15, 0.0, 1.0).astype(np.float32)
        w_l = (eff_alpha * soft_w)[..., None]
        w_ab = (eff_alpha * (0.25 + 0.75 * soft_w))[..., None]
        weights = np.concatenate([w_l, w_ab, w_ab], axis=-1)

        lab_out = (1.0 - weights) * lab + weights * lab_norm
        lab_u8 = np.clip(np.rint(lab_out), 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(lab_u8, cv2.COLOR_LAB2RGB)

    def __call__(self, rgb_uint8: np.ndarray) -> np.ndarray:
        return self.apply_with_target(
            rgb_uint8, self.target_mean, self.target_std, self.strength
        )


def apply_d4_view(img: np.ndarray, view_idx: int) -> np.ndarray:
    """Apply one of the 8 exact D4 dihedral symmetries (rotations + flips)."""
    k = view_idx % 4
    out = img
    if view_idx >= 4:
        out = np.fliplr(out)
    if k > 0:
        out = np.rot90(out, k=k)
    return np.ascontiguousarray(out)


def rgb_and_od_to_tensors(
    rgb_uint8: np.ndarray,
    hed: BeerLambertHED,
    od_mean: np.ndarray,
    od_std: np.ndarray,
    img_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a canonically standardized RGB uint8 patch into:
      1. rgb_tensor: ImageNet-normalized float32 tensor (3, img_size, img_size)
      2. od_tensor:  Training-standardized Beer-Lambert [H, E, R] tensor (3, img_size, img_size)
    """
    if rgb_uint8.shape[0] != img_size or rgb_uint8.shape[1] != img_size:
        rgb_resized = cv2.resize(
            rgb_uint8, (img_size, img_size), interpolation=cv2.INTER_AREA
        )
    else:
        rgb_resized = rgb_uint8

    rgb_f = rgb_resized.astype(np.float32) / 255.0
    rgb_norm = (rgb_f - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb_norm.transpose(2, 0, 1)))

    conc, _ = hed.rgb_to_hed_concentrations(rgb_resized)
    od_norm = (conc - od_mean[None, None, :]) / od_std[None, None, :]
    od_norm = np.clip(od_norm, -6.0, 6.0)
    od_tensor = torch.from_numpy(np.ascontiguousarray(od_norm.transpose(2, 0, 1)))

    return rgb_tensor, od_tensor


class HistopathologyTrainTransform:
    """
    Fast Single-View, Biologically Bounded Physics-Informed H&E Stain Augmentation
    pipeline (without destructive border-cropping or per-patch Reinhard evaluation drift):
      1. Exact lossless D4 dihedral symmetry (preserves border focal tumor nests).
      2. With 75% probability, biologically bounded unseen-hospital stain simulation:
         - Virtual-Center LAB transfer (strength 0.25-0.60 at p=0.40)
         - Bounded Beer-Lambert H&E stain-matrix rotation & concentration jitter (p=0.80)
         - Scanner white-balance gain (0.92-1.08), nonlinear gamma (0.88-1.15),
           optical focus, and sensor noise (p=1.0)
      3. Remaining 25% clean-stain anchor pass (optical/raw without color shift)
         to preserve crisp fixed-0.5 calibration on unaltered H&E optical density.
    """

    def __init__(
        self,
        stain_bank: StainReferenceBank,
        img_size: int = 256,
        canonical_strength: float = 0.75,
    ) -> None:
        self.stain_bank = stain_bank
        self.img_size = int(img_size)
        self.canonical_strength = float(canonical_strength)
        self.hed = BeerLambertHED()
        self.canonical_normalizer = TissueMaskedStainNormalizer(
            target_lab_mean=stain_bank.canonical_lab_mean,
            target_lab_std=stain_bank.canonical_lab_std,
            strength=canonical_strength,
        )

    def _scanner_optics_jitter(
        self,
        img: np.ndarray,
        rng: np.random.Generator,
        apply_color_optics: bool = True,
    ) -> np.ndarray:
        """Simulate unseen scanner white balance, nonlinear gamma, optical focus, and sensor noise."""
        if apply_color_optics:
            out = img.astype(np.float32)

            if rng.random() < 0.60:
                wb_gain = rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)
                out = np.clip(out * wb_gain, 0.0, 255.0)

            if rng.random() < 0.60:
                gamma = float(rng.uniform(0.88, 1.15))
                out = 255.0 * np.power(np.clip(out / 255.0, 1e-4, 1.0), gamma)

            out_u8 = np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)
        else:
            out_u8 = img

        r_focus = rng.random()
        if r_focus < 0.25:
            sigma = float(rng.uniform(0.35, 1.1))
            out_u8 = cv2.GaussianBlur(out_u8, ksize=(3, 3), sigmaX=sigma)
        elif r_focus < 0.40:
            blurred = cv2.GaussianBlur(out_u8, ksize=(3, 3), sigmaX=0.8)
            amount = float(rng.uniform(0.2, 0.55))
            out_u8 = cv2.addWeighted(out_u8, 1.0 + amount, blurred, -amount, 0)

        if rng.random() < 0.25:
            noise_std = float(rng.uniform(1.0, 3.5))
            noise = rng.normal(0.0, noise_std, size=out_u8.shape).astype(np.float32)
            out_u8 = np.clip(out_u8.astype(np.float32) + noise, 0.0, 255.0).astype(
                np.uint8
            )

        return out_u8

    def __call__(
        self, rgb_uint8: np.ndarray, rng: np.random.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        view_idx = int(rng.integers(0, 8))
        img = apply_d4_view(rgb_uint8, view_idx)

        if rng.random() < 0.75:
            if rng.random() < 0.40:
                virt_m, virt_s = self.stain_bank.sample_virtual_center_target(rng)
                img = self.canonical_normalizer.apply_with_target(
                    img,
                    target_mean=virt_m,
                    target_std=virt_s,
                    strength=float(rng.uniform(0.25, 0.60)),
                )

            if rng.random() < 0.80:
                img = self.hed.perturb_stain(img, rng)

            img = self._scanner_optics_jitter(img, rng, apply_color_optics=True)
        else:
            if rng.random() < 0.50:
                img = self._scanner_optics_jitter(img, rng, apply_color_optics=False)

        rgb_t, od_t = rgb_and_od_to_tensors(
            img,
            hed=self.hed,
            od_mean=self.stain_bank.od_channel_mean,
            od_std=self.stain_bank.od_channel_std,
            img_size=self.img_size,
        )
        return rgb_t, od_t


class HistopathologyEvalTransform:
    """
    Deterministic Evaluation / Inference Transform (100% IDENTICAL for Val and Test).
    Preserves raw H&E optical density cues by default (`include_canonical_view=False`)
    and optionally emits paired soft tissue-masked canonical stain-standardized D4 views
    (`include_canonical_view=True`, e.g., at `canonical_strength=0.50`) using identical
    pixel-only logic across val and test.
    """

    def __init__(
        self,
        stain_bank: StainReferenceBank,
        img_size: int = 256,
        canonical_strength: float = 0.75,
        tta_views: int = 1,
        include_canonical_view: bool = False,
    ) -> None:
        if tta_views not in (1, 4, 8):
            raise ValueError(f"tta_views must be 1, 4, or 8; got {tta_views}")
        self.stain_bank = stain_bank
        self.img_size = int(img_size)
        self.canonical_strength = float(canonical_strength)
        self.tta_views = int(tta_views)
        self.include_canonical_view = bool(include_canonical_view)
        self.hed = BeerLambertHED()
        self.canonical_normalizer = TissueMaskedStainNormalizer(
            target_lab_mean=stain_bank.canonical_lab_mean,
            target_lab_std=stain_bank.canonical_lab_std,
            strength=canonical_strength,
        )

    def _expand_d4(
        self, rgb_base: torch.Tensor, od_base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tta_views == 1:
            return rgb_base, od_base

        rgb_list, od_list = [], []
        for v_idx in range(self.tta_views):
            r_v, o_v = rgb_base, od_base
            if v_idx >= 4:
                r_v = torch.flip(r_v, dims=[-1])
                o_v = torch.flip(o_v, dims=[-1])
            k = v_idx % 4
            if k > 0:
                r_v = torch.rot90(r_v, k=k, dims=(-2, -1))
                o_v = torch.rot90(o_v, k=k, dims=(-2, -1))
            rgb_list.append(r_v)
            od_list.append(o_v)
        return torch.stack(rgb_list, dim=0), torch.stack(od_list, dim=0)

    def __call__(self, rgb_uint8: np.ndarray) -> Tuple[torch.Tensor, ...]:
        rgb_base, od_base = rgb_and_od_to_tensors(
            rgb_uint8,
            hed=self.hed,
            od_mean=self.stain_bank.od_channel_mean,
            od_std=self.stain_bank.od_channel_std,
            img_size=self.img_size,
        )
        rgb_out, od_out = self._expand_d4(rgb_base, od_base)

        if not self.include_canonical_view:
            return rgb_out, od_out

        canon_u8 = self.canonical_normalizer(rgb_uint8)
        rgb_c_base, od_c_base = rgb_and_od_to_tensors(
            canon_u8,
            hed=self.hed,
            od_mean=self.stain_bank.od_channel_mean,
            od_std=self.stain_bank.od_channel_std,
            img_size=self.img_size,
        )
        rgb_c_out, od_c_out = self._expand_d4(rgb_c_base, od_c_base)
        return rgb_out, od_out, rgb_c_out, od_c_out


class MultiViewExtractionDataset(Dataset):
    """
    Deterministic multi-view extraction dataset for frozen-PFM feature caching.
    Emits compact uint8 RGB tensors (CHW) so GPU-side native normalization and D4
    symmetries saturate A100 throughput without CPU/PCIe bottlenecks:
      - 'sample_id': sample identifier string
      - 'rgb_raw': raw RGB uint8 tensor (3, H, W)
      - 'rgb_canon': soft tissue-masked canonical RGB uint8 tensor (3, H, W)
                     at configurable canonical_strength=0.50 (identical on val/test)
      - 'od_feats': optional 46-dim float32 Beer-Lambert OD morphological vector
      - 'rgb_pert': (strictly for train when num_perturbed_views > 0) deterministically
                    seeded (V, 3, H, W) uint8 tensor of biologically bounded
                    virtual-center + Beer-Lambert stain-perturbed views (pixel-identical
                    across all PFMs for exact cross-PFM stain twin alignment).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        stain_bank: StainReferenceBank,
        input_dir: Path = INPUT_DIR,
        canonical_strength: float = 0.50,
        num_perturbed_views: int = 0,
        extract_od_features: bool = False,
        seed: int = 42,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.stain_bank = stain_bank
        self.input_dir = Path(input_dir)
        self.canonical_strength = float(canonical_strength)
        self.num_perturbed_views = int(num_perturbed_views)
        self.extract_od_features = bool(extract_od_features)
        self.seed = int(seed)

        self.sample_ids = self.df["sample_id"].astype(str).tolist()
        self.image_paths = [self.input_dir / p for p in self.df["image_path"].tolist()]
        self.hed = BeerLambertHED()
        self.canonical_normalizer = TissueMaskedStainNormalizer(
            target_lab_mean=stain_bank.canonical_lab_mean,
            target_lab_std=stain_bank.canonical_lab_std,
            strength=self.canonical_strength,
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _generate_perturbed_view(
        self, rgb_uint8: np.ndarray, idx: int, view_id: int
    ) -> np.ndarray:
        view_seed = (
            self.seed * 1000003 + idx * 9176 + view_id * 7919 + 17
        ) % (2**32 - 1)
        rng = np.random.default_rng(view_seed)

        # Deterministic lossless D4 orientation covering rotations and flips
        d4_idx = (view_id * 2 + (idx % 2)) % 8
        img = apply_d4_view(rgb_uint8, d4_idx)

        # Biologically bounded virtual-center LAB transfer
        if rng.random() < 0.55:
            virt_m, virt_s = self.stain_bank.sample_virtual_center_target(rng)
            img = self.canonical_normalizer.apply_with_target(
                img,
                target_mean=virt_m,
                target_std=virt_s,
                strength=float(rng.uniform(0.25, 0.60)),
            )

        # Biologically bounded Beer-Lambert H&E stain perturbation
        if rng.random() < 0.90:
            img = self.hed.perturb_stain(
                img,
                rng,
                conc_scale_range=(0.78, 1.28),
                conc_bias_range=(-0.05, 0.05),
                matrix_jitter_std=0.04,
            )

        # Bounded scanner optics (white balance, gamma, subtle focus/noise)
        out = img.astype(np.float32)
        if rng.random() < 0.65:
            wb_gain = rng.uniform(0.92, 1.08, size=(1, 1, 3)).astype(np.float32)
            out = np.clip(out * wb_gain, 0.0, 255.0)
        if rng.random() < 0.65:
            gamma = float(rng.uniform(0.88, 1.15))
            out = 255.0 * np.power(np.clip(out / 255.0, 1e-4, 1.0), gamma)
        out_u8 = np.clip(np.rint(out), 0.0, 255.0).astype(np.uint8)

        r_focus = rng.random()
        if r_focus < 0.20:
            sigma = float(rng.uniform(0.35, 1.0))
            out_u8 = cv2.GaussianBlur(out_u8, ksize=(3, 3), sigmaX=sigma)
        elif r_focus < 0.35:
            blurred = cv2.GaussianBlur(out_u8, ksize=(3, 3), sigmaX=0.8)
            amount = float(rng.uniform(0.2, 0.5))
            out_u8 = cv2.addWeighted(out_u8, 1.0 + amount, blurred, -amount, 0)

        if rng.random() < 0.20:
            noise_std = float(rng.uniform(1.0, 3.0))
            noise = rng.normal(0.0, noise_std, size=out_u8.shape).astype(np.float32)
            out_u8 = np.clip(out_u8.astype(np.float32) + noise, 0.0, 255.0).astype(
                np.uint8
            )
        return out_u8

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rgb_u8 = read_rgb_image(self.image_paths[idx])
        canon_u8 = self.canonical_normalizer(rgb_u8)

        item: Dict[str, Any] = {
            "sample_id": self.sample_ids[idx],
            "rgb_raw": torch.from_numpy(np.ascontiguousarray(rgb_u8.transpose(2, 0, 1))),
            "rgb_canon": torch.from_numpy(
                np.ascontiguousarray(canon_u8.transpose(2, 0, 1))
            ),
        }

        if self.extract_od_features:
            od_vec = extract_od_morphological_features(rgb_u8, hed=self.hed)
            item["od_feats"] = torch.from_numpy(od_vec)

        if self.num_perturbed_views > 0:
            pert_list = [
                torch.from_numpy(
                    np.ascontiguousarray(
                        self._generate_perturbed_view(rgb_u8, idx, v).transpose(2, 0, 1)
                    )
                )
                for v in range(self.num_perturbed_views)
            ]
            item["rgb_pert"] = torch.stack(pert_list, dim=0)

        return item


class HistopathologyDataset(Dataset):
    """
    Pixel-only inference-compliant Histopathology Dataset emitting:
      - 'rgb': Canonically stain-standardized RGB tensor
      - 'od':  Standardized Beer-Lambert [Hematoxylin, Eosin, Residual] OD tensor
      - 'sample_id': sample identifier string
      - 'label': float32 binary label (when available)
      - Optional training-only 'center_idx' (NEVER used as features)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        input_dir: Path = INPUT_DIR,
        is_train: bool = False,
        return_mode: str = "dual",
        include_train_meta: bool = False,
        seed: int = 42,
    ) -> None:
        if return_mode not in ("dual", "rgb", "od"):
            raise ValueError(f"Invalid return_mode: {return_mode}")
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.input_dir = Path(input_dir)
        self.is_train = bool(is_train)
        self.return_mode = return_mode
        self.include_train_meta = bool(include_train_meta and is_train)
        self.base_seed = int(seed)

        self.sample_ids = self.df["sample_id"].astype(str).tolist()
        self.image_paths = [self.input_dir / p for p in self.df["image_path"].tolist()]
        self.has_label = "label" in self.df.columns
        self.labels = (
            self.df["label"].astype(np.float32).values if self.has_label else None
        )

        if self.include_train_meta and "center_id" in self.df.columns:
            unique_centers = sorted(self.df["center_id"].astype(str).unique().tolist())
            self.center_to_idx = {c: i for i, c in enumerate(unique_centers)}
            self.center_indices = np.array(
                [self.center_to_idx[str(c)] for c in self.df["center_id"]],
                dtype=np.int64,
            )
        else:
            self.center_to_idx = {}
            self.center_indices = None

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _get_rng(self, idx: int) -> np.random.Generator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            w_seed = worker_info.seed
        else:
            w_seed = torch.initial_seed()
        entropy = (w_seed + idx * 1000003 + random.randint(0, 1_000_000)) % (2**32 - 1)
        return np.random.default_rng(entropy)

    def __getitem__(self, idx: int) -> Dict:
        rgb_u8 = read_rgb_image(self.image_paths[idx])

        rgb_canon_t = None
        od_canon_t = None
        if self.is_train:
            rng = self._get_rng(idx)
            rgb_t, od_t = self.transform(rgb_u8, rng)
        else:
            out = self.transform(rgb_u8)
            if len(out) == 4:
                rgb_t, od_t, rgb_canon_t, od_canon_t = out
            else:
                rgb_t, od_t = out

        item: Dict = {"sample_id": self.sample_ids[idx]}
        if self.return_mode in ("dual", "rgb"):
            item["rgb"] = rgb_t
            if rgb_canon_t is not None:
                item["rgb_canon"] = rgb_canon_t
        if self.return_mode in ("dual", "od"):
            item["od"] = od_t
            if od_canon_t is not None:
                item["od_canon"] = od_canon_t
        if self.has_label:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.include_train_meta and self.center_indices is not None:
            item["center_idx"] = torch.tensor(
                self.center_indices[idx], dtype=torch.long
            )
        return item


def build_slide_tempered_balanced_sampler(
    train_df: pd.DataFrame,
    slide_power: float = 0.5,
    seed: int = 42,
) -> WeightedRandomSampler:
    """
    Constructs a WeightedRandomSampler that:
      1. Balances all (center_id, label) strata equally so the classifier's
         predicted probabilities remain well-calibrated around the 0.5 threshold.
      2. Applies square-root slide-size tempering (1 / slide_count^slide_power)
         within each stratum so huge slides cannot overwhelm smaller slides.
    """
    df = train_df.reset_index(drop=True)
    slide_counts = (
        df.groupby("slide_id")["sample_id"].transform("count").values.astype(np.float64)
    )
    raw_slide_w = 1.0 / np.power(np.maximum(slide_counts, 1.0), slide_power)

    strata = df["center_id"].astype(str) + "__" + df["label"].astype(str)
    weights = np.zeros(len(df), dtype=np.float64)
    unique_strata = strata.unique()
    for s in unique_strata:
        mask = (strata == s).values
        s_w = raw_slide_w[mask]
        weights[mask] = s_w / s_w.sum()

    weights = weights / weights.sum()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(df),
        replacement=True,
        generator=generator,
    )


def _worker_init_fn(worker_id: int) -> None:
    cv2.setNumThreads(0)
    seed = (torch.initial_seed() + worker_id) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)


def load_and_audit_data(
    input_dir: Path = INPUT_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSV metadata, clean types, and audit schema/path/leakage integrity."""
    train_df = pd.read_csv(input_dir / "train.csv")
    val_df = pd.read_csv(input_dir / "validation.csv")
    test_df = pd.read_csv(input_dir / "test.csv")
    sub_df = pd.read_csv(input_dir / "sample_submission.csv")

    req_dev_cols = [
        "sample_id",
        "image_path",
        "label",
        "slide_id",
        "group_id",
        "center_id",
    ]
    for name, df in [("train.csv", train_df), ("validation.csv", val_df)]:
        missing_cols = [c for c in req_dev_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} missing required columns: {missing_cols}")
        if df[req_dev_cols].isna().any().any():
            raise ValueError(f"{name} contains NaN values in required columns.")
        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError(f"{name} contains non-binary labels.")

    if list(sub_df.columns) != ["sample_id", "tumor_probability"]:
        raise ValueError(
            f"Unexpected sample_submission columns: {list(sub_df.columns)}"
        )
    if not test_df["sample_id"].equals(sub_df["sample_id"]):
        raise ValueError(
            "test.csv sample_id order does not match sample_submission.csv!"
        )

    for col in ("sample_id", "slide_id", "group_id"):
        overlap = set(train_df[col]).intersection(set(val_df[col]))
        if overlap:
            raise ValueError(
                f"Leakage detected between train and validation on {col}: {len(overlap)} overlapping"
            )

    dev_ids = set(train_df["sample_id"]).union(set(val_df["sample_id"]))
    if dev_ids.intersection(set(test_df["sample_id"])):
        raise ValueError("Sample ID overlap detected between dev sets and test.csv!")

    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sample_paths = [
            df["image_path"].iloc[0],
            df["image_path"].iloc[len(df) // 2],
            df["image_path"].iloc[-1],
        ]
        for rel_p in sample_paths:
            full_p = input_dir / rel_p
            if not full_p.exists():
                raise FileNotFoundError(f"Missing image in {df_name}: {full_p}")

    for df in (train_df, val_df):
        df["label"] = df["label"].astype(int)
        df["center_id"] = df["center_id"].astype(str)
        df["slide_id"] = df["slide_id"].astype(str)
        df["group_id"] = df["group_id"].astype(str)

    return train_df, val_df, test_df, sub_df


def build_leakage_free_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Build all leakage-aware splitting structures:
      1. Official Fixed Split: train_df vs val_df (for official Final Validation Score)
      2. LOCO (Leave-One-Center-Out) splits within train_df (for unseen-center diagnostics)
      3. StratifiedGroupKFold within train_df (grouped strictly by group_id)
      4. Post-selection refit pool: train_df + val_df (used ONLY after reporting val score)
    """
    loco_splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    unique_centers = sorted(train_df["center_id"].unique().tolist())
    if len(unique_centers) >= 2:
        for held_out_center in unique_centers:
            tr_sub = train_df[train_df["center_id"] != held_out_center].reset_index(
                drop=True
            )
            va_sub = train_df[train_df["center_id"] == held_out_center].reset_index(
                drop=True
            )
            assert set(tr_sub["group_id"]).isdisjoint(set(va_sub["group_id"]))
            loco_splits[held_out_center] = (tr_sub, va_sub)

    train_with_folds = train_df.copy().reset_index(drop=True)
    n_unique_groups = train_with_folds["group_id"].nunique()
    eff_splits = min(n_splits, n_unique_groups)
    strat_key = (
        train_with_folds["center_id"].astype(str)
        + "_"
        + train_with_folds["label"].astype(str)
    )
    sgkf = StratifiedGroupKFold(n_splits=eff_splits, shuffle=True, random_state=seed)
    train_with_folds["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(
        sgkf.split(train_with_folds, strat_key, groups=train_with_folds["group_id"])
    ):
        train_with_folds.loc[val_idx, "fold"] = fold_idx

    for f in range(eff_splits):
        g_tr = set(train_with_folds.loc[train_with_folds["fold"] != f, "group_id"])
        g_va = set(train_with_folds.loc[train_with_folds["fold"] == f, "group_id"])
        assert g_tr.isdisjoint(g_va), f"Group leakage detected in fold {f}"

    loco_fold_indices: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if len(unique_centers) >= 2:
        for held_out_center in unique_centers:
            tr_idx = np.where(
                train_with_folds["center_id"].values != held_out_center
            )[0]
            va_idx = np.where(
                train_with_folds["center_id"].values == held_out_center
            )[0]
            loco_fold_indices.append((str(held_out_center), tr_idx, va_idx))

    full_train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    return {
        "train_df": train_with_folds,
        "val_df": val_df.reset_index(drop=True),
        "full_train_val_df": full_train_val_df,
        "loco_splits": loco_splits,
        "loco_fold_indices": loco_fold_indices,
        "n_folds": eff_splits,
    }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    test_df: pd.DataFrame,
    stain_bank: StainReferenceBank,
    img_size: int = 256,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    num_workers: Optional[int] = None,
    return_mode: str = "dual",
    eval_tta_views: int = 1,
    canonical_strength: float = 0.75,
    use_balanced_sampler: bool = True,
    include_train_meta: bool = False,
    seed: int = 42,
) -> Dict[str, Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training, validation, and test with:
      - Physics-informed Virtual-Center augmentation + canonical standardization for train
      - 100% IDENTICAL canonical standardization + optional D4 TTA for val and test
    """
    if num_workers is None:
        cpu_cnt = os.cpu_count() or 2
        num_workers = max(2, min(8, cpu_cnt))

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_tf = HistopathologyTrainTransform(
        stain_bank=stain_bank,
        img_size=img_size,
        canonical_strength=canonical_strength,
    )
    eval_tf = HistopathologyEvalTransform(
        stain_bank=stain_bank,
        img_size=img_size,
        canonical_strength=canonical_strength,
        tta_views=eval_tta_views,
    )

    train_ds = HistopathologyDataset(
        df=train_df,
        transform=train_tf,
        input_dir=INPUT_DIR,
        is_train=True,
        return_mode=return_mode,
        include_train_meta=include_train_meta,
        seed=seed,
    )

    if use_balanced_sampler:
        sampler = build_slide_tempered_balanced_sampler(
            train_df, slide_power=0.5, seed=seed
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )

    val_loader = None
    if val_df is not None and len(val_df) > 0:
        val_ds = HistopathologyDataset(
            df=val_df,
            transform=eval_tf,
            input_dir=INPUT_DIR,
            is_train=False,
            return_mode=return_mode,
            include_train_meta=False,
            seed=seed,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )

    test_ds = HistopathologyDataset(
        df=test_df,
        transform=eval_tf,
        input_dir=INPUT_DIR,
        is_train=False,
        return_mode=return_mode,
        include_train_meta=False,
        seed=seed,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def prepare_data_and_features(
    img_size: int = 256,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    eval_tta_views: int = 1,
    canonical_strength: float = 0.75,
    return_mode: str = "dual",
    force_refit_bank: bool = True,
    seed: int = 42,
) -> Dict:
    """
    End-to-end entry point for Step 1:
      - Loads & audits datasets
      - Builds leakage-free splits (Official, LOCO, StratifiedGroupKFold, Refit pool)
      - Fits training-only StainReferenceBank
      - Persists auditable split & stain artifacts to ./working/
      - Builds ready-to-use PyTorch DataLoaders
    """
    seed_everything(seed)
    train_df, val_df, test_df, sub_df = load_and_audit_data(INPUT_DIR)
    splits = build_leakage_free_splits(train_df, val_df, n_splits=5, seed=seed)

    bank_path = WORKING_DIR / "stain_reference_bank_train.json"
    if bank_path.exists() and not force_refit_bank:
        stain_bank = StainReferenceBank.load(bank_path)
    else:
        stain_bank = StainReferenceBank().fit(
            splits["train_df"],
            input_dir=INPUT_DIR,
            max_patches_per_slide=36,
            seed=seed,
        )
        stain_bank.save(bank_path)

    splits["train_df"].to_csv(WORKING_DIR / "train_with_folds.csv", index=False)

    loaders = create_dataloaders(
        train_df=splits["train_df"],
        val_df=splits["val_df"],
        test_df=test_df,
        stain_bank=stain_bank,
        img_size=img_size,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        return_mode=return_mode,
        eval_tta_views=eval_tta_views,
        canonical_strength=canonical_strength,
        use_balanced_sampler=True,
        include_train_meta=True,
        seed=seed,
    )

    return {
        "train_df": splits["train_df"],
        "val_df": splits["val_df"],
        "test_df": test_df,
        "sample_submission_df": sub_df,
        "full_train_val_df": splits["full_train_val_df"],
        "loco_splits": splits["loco_splits"],
        "loco_fold_indices": splits["loco_fold_indices"],
        "n_folds": splits["n_folds"],
        "stain_bank": stain_bank,
        "train_loader": loaders["train_loader"],
        "val_loader": loaders["val_loader"],
        "test_loader": loaders["test_loader"],
    }


# =============================================================================
# STEP 2: MODEL DESIGN, LOSS FUNCTIONS, LLRD OPTIMIZER, SCHEDULER & MODEL EMA
# =============================================================================
class _GradientReversalFn(torch.autograd.Function):
    """Identity in forward pass; negates and scales gradients in backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


class SqueezeExcite2d(nn.Module):
    """Channel attention block emphasizing diagnostic Hematoxylin/Eosin filters."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(16, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3), keepdim=True)
        w = torch.sigmoid(self.fc2(self.act(self.fc1(w))))
        return x * w


class InstanceNormResBlock(nn.Module):
    """
    BatchNorm-free multi-scale residual block using InstanceNorm2d + GroupNorm.
    Instance normalization strips per-patch stain intensity shifts while GroupNorm
    preserves relative cross-channel Hematoxylin-vs-Eosin ratios without relying
    on training-center running batch statistics.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        gn_groups = 8 if out_ch % 8 == 0 else 4

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act1 = nn.GELU()

        self.dw_conv = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=out_ch,
            bias=False,
        )
        self.pw_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(gn_groups, out_ch)
        self.se = SqueezeExcite2d(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(gn_groups, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.act_out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.pw_conv(self.dw_conv(out)))
        out = self.drop(self.se(out))
        return self.act_out(out + identity)


class BeerLambertODBranch(nn.Module):
    """
    Dedicated physics-informed encoder for standardized 3-channel Beer-Lambert
    [Hematoxylin, Eosin, Residual] Optical Density tensors (B, 3, 256, 256).
    """

    def __init__(self, out_dim: int = 256, dropout: float = 0.15) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.stage1 = InstanceNormResBlock(32, 64, stride=2, dropout=0.05)
        self.stage2 = InstanceNormResBlock(64, 128, stride=2, dropout=0.08)
        self.stage3 = InstanceNormResBlock(128, 192, stride=2, dropout=0.10)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self.proj = nn.Sequential(
            nn.Linear(192 * 3, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aux_head = nn.Linear(out_dim, 1)

    def forward(self, od: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(od)
        x = self.stage1(x)
        x = self.stage2(x)
        feat_map = self.stage3(x)

        B, C, H, W = feat_map.shape
        attn_logits = self.spatial_attn(feat_map).view(B, 1, H * W)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        flat = feat_map.view(B, C, H * W)

        attn_pool = torch.bmm(flat, attn_weights.transpose(1, 2)).squeeze(-1)
        avg_pool = flat.mean(dim=-1)
        max_pool = flat.amax(dim=-1)

        desc = torch.cat([attn_pool, avg_pool, max_pool], dim=-1)
        od_emb = self.proj(desc)
        od_logits = self.aux_head(od_emb).squeeze(-1)
        return od_emb, od_logits


class GatedMILAttentionPool(nn.Module):
    """
    Ilse et al. Gated Attention Pooling over Vision Transformer spatial tokens.
    Enables the model to focus strongly on small localized metastatic tumor nests
    without diluting signal across surrounding normal lymphocytes or glass lumen.
    """

    def __init__(
        self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.20
    ) -> None:
        super().__init__()
        self.attn_v = nn.Linear(embed_dim, hidden_dim)
        self.attn_u = nn.Linear(embed_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(
        self, spatial_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.attn_v(spatial_tokens)) * torch.sigmoid(
            self.attn_u(spatial_tokens)
        )
        scores = self.attn_w(self.drop(gated)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), spatial_tokens).squeeze(1)
        return pooled, weights


class CrossModalGatedFusion(nn.Module):
    """
    Bidirectional Gated Cross-Modal Fusion between the global RGB ViT stream
    and the Beer-Lambert Optical Density stream.
    """

    def __init__(
        self,
        rgb_dim: int,
        od_dim: int,
        fusion_dim: int = 512,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.od_proj = nn.Sequential(
            nn.Linear(od_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.gate_rgb_from_od = nn.Linear(fusion_dim, fusion_dim)
        self.gate_od_from_rgb = nn.Linear(fusion_dim, fusion_dim)

        self.dropout = float(dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

    def forward(self, rgb_emb: torch.Tensor, od_emb: torch.Tensor) -> torch.Tensor:
        r = self.rgb_proj(rgb_emb)
        o = self.od_proj(od_emb)

        r_mod = r * (0.5 + torch.sigmoid(self.gate_rgb_from_od(o)))
        o_mod = o * (0.5 + torch.sigmoid(self.gate_od_from_rgb(r)))

        fused = torch.cat([r_mod, o_mod], dim=-1)
        return self.out_proj(fused)


class LoRALinear(nn.Module):
    """
    Deepcopy-safe Low-Rank Adaptation wrapper around a frozen nn.Linear layer.
    Freezes pretrained weights and learns a low-rank residual (lora_B @ lora_A)
    initialized to zero so Epoch-1/initial forward matches pretrained features.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.in_features = int(base_layer.in_features)
        self.out_features = int(base_layer.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


def _inject_lora_into_vit(
    backbone: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> int:
    """
    Freeze all pretrained weights in a ViT backbone and inject LoRALinear adapters
    into attention (qkv, proj, query, key, value, dense) and MLP (fc1, fc2) Linear layers.
    """
    for p in backbone.parameters():
        p.requires_grad_(False)

    target_names = {"qkv", "proj", "query", "key", "value", "dense", "fc1", "fc2"}
    injected = 0

    def _replace_recursive(module: nn.Module) -> None:
        nonlocal injected
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                injected += 1
            else:
                _replace_recursive(child)

    _replace_recursive(backbone)
    return injected


class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Sample Dropout classification head (averages logits across K parallel
    dropout masks) for improved calibration around the fixed 0.5 threshold.
    """

    def __init__(
        self,
        in_dim: int,
        num_samples: int = 5,
        dropout_rate: float = 0.30,
    ) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.stack([self.fc(d(x)) for d in self.dropouts], dim=0).mean(dim=0)
        return logits.squeeze(-1)


class DualPrototypeCosineHead(nn.Module):
    """
    Bias-free, L2-normalized Dual-Prototype Cosine classification head with
    Multi-Sample Dropout. Eliminates asymmetric bias shift and unseen-center
    feature-norm drift around the fixed 0.5 probability boundary (logit = 0).
    """

    def __init__(
        self,
        in_dim: int,
        num_samples: int = 5,
        dropout_rate: float = 0.30,
        init_scale: float = 10.0,
        min_scale: float = 2.0,
        max_scale: float = 25.0,
    ) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_samples)]
        )
        proto_init = torch.empty(2, in_dim)
        nn.init.orthogonal_(proto_init)
        proto_init = F.normalize(proto_init, p=2, dim=-1)
        self.prototypes = nn.Parameter(proto_init)
        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1, eps=1e-6).to(
            dtype=x.dtype
        )
        scale = torch.clamp(
            self.logit_scale, min=self.min_scale, max=self.max_scale
        ).to(dtype=x.dtype)

        logits_list = []
        for drop in self.dropouts:
            x_norm = F.normalize(drop(x), p=2, dim=-1, eps=1e-6)
            cos_normal = (x_norm * proto_norm[0]).sum(dim=-1)
            cos_tumor = (x_norm * proto_norm[1]).sum(dim=-1)
            logits_list.append(scale * (cos_tumor - cos_normal))
        return torch.stack(logits_list, dim=0).mean(dim=0)


class ConditionalDomainDiscriminator(nn.Module):
    """
    Training-only Conditional Domain Adversarial Discriminator (CDAN).
    Conditioned on detached predicted class probabilities via the exact binary
    multilinear map [z * p, z * (1 - p)]. NEVER called at validation or test inference.
    """

    def __init__(
        self,
        embed_dim: int,
        num_centers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_centers = int(num_centers)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_centers),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        tumor_logits: torch.Tensor,
        grl_lambda: float = 1.0,
    ) -> torch.Tensor:
        p_tumor = torch.sigmoid(tumor_logits.detach()).unsqueeze(-1)
        p_normal = 1.0 - p_tumor
        z_rev = grad_reverse(embedding, grl_lambda)
        cond_feat = torch.cat([z_rev * p_tumor, z_rev * p_normal], dim=-1)
        return self.net(cond_feat)


class _HFDinov2BackboneWrapper(nn.Module):
    """
    Deepcopy-safe HuggingFace AutoModel wrapper for histopathology DINOv2 backbones
    (e.g., owkin/phikon-v2 and histai/hibou-L via trust_remote_code=True), supporting
    native/interpolated positional encodings, multi-layer hidden-state tapping on
    layers (num_blocks-7, -5, -3, -1), register-token separation, and grad checkpointing.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_path_rate: float = 0.30,
        trust_remote_code: bool = False,
        norm_mean: Optional[List[float]] = None,
        norm_std: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        cfg = AutoConfig.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if hasattr(cfg, "drop_path_rate"):
            cfg.drop_path_rate = float(drop_path_rate)

        if pretrained:
            self.model = AutoModel.from_pretrained(
                model_name, config=cfg, trust_remote_code=trust_remote_code
            )
        else:
            self.model = AutoModel.from_config(
                cfg, trust_remote_code=trust_remote_code
            )

        self.is_hf_dinov2: bool = True
        self.patch_size: int = int(getattr(cfg, "patch_size", 16))
        self.native_image_size: int = int(getattr(cfg, "image_size", 224))
        self.num_features: int = int(getattr(cfg, "hidden_size", 1024))
        self.num_hidden_layers: int = int(getattr(cfg, "num_hidden_layers", 24))
        n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.num_prefix_tokens: int = 1 + n_reg

        fwd_params = inspect.signature(self.model.forward).parameters
        self._supports_interp_pos_enc: bool = "interpolate_pos_encoding" in fwd_params
        self.dynamic_img_size: bool = bool(self._supports_interp_pos_enc)

        mean = (
            [float(x) for x in norm_mean]
            if (norm_mean is not None and len(norm_mean) == 3)
            else IMAGENET_MEAN.tolist()
        )
        std = (
            [float(x) for x in norm_std]
            if (norm_std is not None and len(norm_std) == 3)
            else IMAGENET_STD.tolist()
        )
        if norm_mean is None or norm_std is None:
            try:
                proc = AutoImageProcessor.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
                p_mean = getattr(proc, "image_mean", None)
                p_std = getattr(proc, "image_std", None)
                if p_mean is not None and len(p_mean) == 3:
                    mean = [float(x) for x in p_mean]
                if p_std is not None and len(p_std) == 3:
                    std = [float(x) for x in p_std]
            except Exception:
                pass
        self.pretrained_cfg: Dict[str, Any] = {"mean": mean, "std": std}

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if enable and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
        elif not enable and hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()

    def _call_model(
        self, x: torch.Tensor, output_hidden_states: bool = False
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "pixel_values": x,
            "return_dict": True,
        }
        if output_hidden_states:
            kwargs["output_hidden_states"] = True
        if self._supports_interp_pos_enc:
            kwargs["interpolate_pos_encoding"] = True
        try:
            return self.model(**kwargs)
        except TypeError:
            kwargs.pop("interpolate_pos_encoding", None)
            self._supports_interp_pos_enc = False
            self.dynamic_img_size = False
            return self.model(**kwargs)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self._call_model(x, output_hidden_states=False)
        return out.last_hidden_state

    def forward_features_with_taps(
        self,
        x: torch.Tensor,
        tap_indices: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = self._call_model(x, output_hidden_states=True)
        hidden_states = getattr(out, "hidden_states", None) or ()
        tapped = [
            hidden_states[idx + 1]
            for idx in tap_indices
            if (idx + 1) < len(hidden_states)
        ]
        return out.last_hidden_state, tapped


def _create_timm_backbone(
    preferred_name: Optional[str] = None,
    pretrained: bool = True,
    drop_path_rate: float = 0.30,
    exclude_backbones: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Instantiate a pretrained Vision Transformer / ConvNeXt / SwinV2 backbone via timm
    or HuggingFace AutoModel (owkin/phikon-v2, histai/hibou-L) with explicit native
    normalization metadata and family-preserving automatic fallback.
    """
    excluded = set(exclude_backbones or [])
    candidates: List[Dict[str, Any]] = []
    preferred_family: Optional[str] = None
    if preferred_name is not None and preferred_name not in excluded:
        matched = [
            c for c in PRETRAINED_BACKBONE_CANDIDATES if c["name"] == preferred_name
        ]
        if matched:
            candidates.append(matched[0])
            preferred_family = matched[0]["family"]
        else:
            low = preferred_name.lower()
            if "convnext" in low:
                family = "convnext"
            elif "swin" in low:
                family = "swinv2"
            else:
                family = "vit"
            preferred_family = family
            is_hf = ("phikon" in low) or ("hibou" in low)
            loader = "hf_dinov2" if is_hf else "timm"
            spec_custom: Dict[str, Any] = {
                "name": preferred_name,
                "pretraining_data": "Specified pretrained weights",
                "family": family,
                "loader": loader,
                "trust_remote_code": "hibou" in low,
            }
            if "kaiko_ai" in low:
                spec_custom["norm_mean"] = [0.5, 0.5, 0.5]
                spec_custom["norm_std"] = [0.5, 0.5, 0.5]
            candidates.append(spec_custom)
    if preferred_family is not None:
        for c in PRETRAINED_BACKBONE_CANDIDATES:
            if (
                c["family"] == preferred_family
                and c["name"] not in excluded
                and all(existing["name"] != c["name"] for existing in candidates)
            ):
                candidates.append(c)
    for c in PRETRAINED_BACKBONE_CANDIDATES:
        if c["name"] not in excluded and all(
            existing["name"] != c["name"] for existing in candidates
        ):
            candidates.append(c)

    last_error: Optional[Exception] = None
    for spec in candidates:
        name = spec["name"]
        spec_mean = spec.get("norm_mean")
        spec_std = spec.get("norm_std")
        if spec.get("loader") == "hf_dinov2":
            try:
                backbone = _HFDinov2BackboneWrapper(
                    model_name=name,
                    pretrained=pretrained,
                    drop_path_rate=drop_path_rate,
                    trust_remote_code=bool(spec.get("trust_remote_code", False)),
                    norm_mean=spec_mean,
                    norm_std=spec_std,
                )
                meta = {
                    "backbone_name": name,
                    "backbone_family": spec["family"],
                    "pretraining_data": spec["pretraining_data"],
                    "pretrained": pretrained,
                    "timm_version": timm.__version__,
                    "torch_version": torch.__version__,
                    "norm_mean": backbone.pretrained_cfg.get("mean"),
                    "norm_std": backbone.pretrained_cfg.get("std"),
                }
                return backbone, meta
            except Exception as exc:
                last_error = exc
                continue

        kwargs: Dict[str, Any] = {
            "pretrained": pretrained,
            "num_classes": 0,
            "drop_path_rate": drop_path_rate,
        }
        if spec["family"] == "vit":
            kwargs["dynamic_img_size"] = True

        backbone = None
        try:
            backbone = timm.create_model(name, **kwargs)
        except TypeError:
            kwargs.pop("dynamic_img_size", None)
            try:
                backbone = timm.create_model(name, **kwargs)
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc

        if backbone is not None:
            if spec_mean is not None and spec_std is not None:
                if hasattr(backbone, "pretrained_cfg") and isinstance(
                    backbone.pretrained_cfg, dict
                ):
                    backbone.pretrained_cfg["mean"] = tuple(float(x) for x in spec_mean)
                    backbone.pretrained_cfg["std"] = tuple(float(x) for x in spec_std)
                if hasattr(backbone, "default_cfg") and isinstance(
                    backbone.default_cfg, dict
                ):
                    backbone.default_cfg["mean"] = tuple(float(x) for x in spec_mean)
                    backbone.default_cfg["std"] = tuple(float(x) for x in spec_std)
            meta = {
                "backbone_name": name,
                "backbone_family": spec["family"],
                "pretraining_data": spec["pretraining_data"],
                "pretrained": pretrained,
                "timm_version": timm.__version__,
                "torch_version": torch.__version__,
                "norm_mean": list(spec_mean) if spec_mean is not None else None,
                "norm_std": list(spec_std) if spec_std is not None else None,
            }
            return backbone, meta

    raise RuntimeError(
        f"Failed to instantiate any pretrained backbone candidate. Last error: {last_error}"
    )


def _infer_backbone_input_size(backbone: nn.Module, target_size: int = 256) -> int:
    """
    Determine exact spatial resolution compatible with the backbone's patch grid.
    For patch_size=14 (DINOv2/Kaiko), selects 252 (18x18 patches) when dynamic,
    or native 224 when fixed (e.g., Hibou-L without dynamic pos interpolation).
    For patch_size=16 (Phikon-v2), selects native 256 (16x16 patches).
    """
    if getattr(backbone, "is_hf_dinov2", False):
        if not bool(getattr(backbone, "dynamic_img_size", True)):
            return int(getattr(backbone, "native_image_size", 224))
        p = int(getattr(backbone, "patch_size", 16))
        return max(p * 4, (target_size // p) * p)

    patch_embed = getattr(backbone, "patch_embed", None)
    if patch_embed is None:
        return target_size

    dynamic = bool(
        getattr(backbone, "dynamic_img_size", False)
        or getattr(patch_embed, "dynamic_img_size", False)
        or (
            hasattr(patch_embed, "strict_img_size")
            and not bool(patch_embed.strict_img_size)
        )
    )
    patch_size = getattr(patch_embed, "patch_size", (16, 16))
    p = int(patch_size[0]) if isinstance(patch_size, (tuple, list)) else int(patch_size)

    if dynamic:
        return max(p * 4, (target_size // p) * p)

    native_size = getattr(patch_embed, "img_size", None)
    if native_size is not None:
        ns = (
            int(native_size[0])
            if isinstance(native_size, (tuple, list))
            else int(native_size)
        )
        return ns
    return target_size


class _BlockCaptureHook:
    """Deepcopy-safe and gradient-checkpoint-safe forward hook for ViT block outputs."""

    def __init__(self, owner: "DualStreamStainViT", block_idx: int) -> None:
        self.owner = owner
        self.block_idx = int(block_idx)

    def __call__(
        self, _module: nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        if self.owner._capturing_blocks:
            self.owner._hook_outputs[self.block_idx] = output


class _StageCaptureHook:
    """Deepcopy-safe and gradient-checkpoint-safe forward hook for ConvNeXt stage outputs."""

    def __init__(self, owner: "DualStreamStainViT") -> None:
        self.owner = owner

    def __call__(
        self, _module: nn.Module, _inputs: Any, output: torch.Tensor
    ) -> None:
        if self.owner._capturing_stage:
            self.owner._stage_hook_output = output


def _infer_convnext_stage_dims(
    backbone: nn.Module, default_stage4: int
) -> Tuple[int, int]:
    """Infer Stage-3 (stages[2]) and Stage-4 (stages[3]) channel dimensions for ConvNeXt."""
    fi = getattr(backbone, "feature_info", None)
    if fi is not None:
        if hasattr(fi, "channels"):
            chs = list(fi.channels())
            if len(chs) >= 4:
                return int(chs[2]), int(chs[3])
        elif isinstance(fi, (list, tuple)) and len(fi) >= 4:
            if isinstance(fi[2], dict) and "num_chs" in fi[2]:
                return int(fi[2]["num_chs"]), int(fi[3]["num_chs"])

    stages = getattr(backbone, "stages", None)
    if stages is not None and len(stages) >= 4:
        dims: List[int] = []
        for st in (stages[2], stages[3]):
            dim = getattr(st, "out_chs", None)
            if dim is None and hasattr(st, "blocks") and len(st.blocks) > 0:
                blk = st.blocks[-1]
                if hasattr(blk, "conv_dw") and hasattr(blk.conv_dw, "out_channels"):
                    dim = int(blk.conv_dw.out_channels)
                elif hasattr(blk, "norm") and hasattr(blk.norm, "normalized_shape"):
                    dim = int(blk.norm.normalized_shape[0])
            if dim is not None:
                dims.append(int(dim))
        if len(dims) == 2:
            return dims[0], dims[1]

    return max(64, int(default_stage4) // 2), int(default_stage4)


class DualStreamStainViT(nn.Module):
    """
    BatchNorm-Free Dual-Stream Physics-Guided Vision Backbone (DINOv2 ViT or
    OpenCLIP ConvNeXt) for Cross-Center Histopathology Tumor Classification.
    """

    def __init__(
        self,
        backbone_name: Optional[str] = "owkin/phikon-v2",
        pretrained: bool = True,
        img_size: int = 256,
        od_dim: int = 256,
        fusion_dim: int = 512,
        drop_path_rate: float = 0.30,
        head_dropout: float = 0.30,
        num_centers: int = 2,
        use_od_stream: bool = True,
        grad_checkpointing: bool = False,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_od_stream = bool(use_od_stream)
        self.num_centers = int(num_centers)
        self.grad_checkpointing = bool(grad_checkpointing)
        self._backbone_trainable: bool = True

        self.rgb_backbone, self.provenance = _create_timm_backbone(
            preferred_name=backbone_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
        )
        if grad_checkpointing and hasattr(self.rgb_backbone, "set_grad_checkpointing"):
            self.rgb_backbone.set_grad_checkpointing(True)

        self.vit_input_size = _infer_backbone_input_size(
            self.rgb_backbone, target_size=img_size
        )
        self.rgb_dim = int(getattr(self.rgb_backbone, "num_features", 1024))

        # Affine input-statistics remapping from ImageNet normalization (DataLoader)
        # to the backbone's explicit native mean/std (prioritizing provenance overrides
        # such as 0.5/0.5/0.5 for all Kaiko pathology backbones).
        pt_cfg = dict(
            getattr(self.rgb_backbone, "pretrained_cfg", None)
            or getattr(self.rgb_backbone, "default_cfg", None)
            or {}
        )
        if self.provenance.get("norm_mean") is not None:
            pt_cfg["mean"] = self.provenance["norm_mean"]
        if self.provenance.get("norm_std") is not None:
            pt_cfg["std"] = self.provenance["norm_std"]
        if pt_cfg.get("mean") is None or pt_cfg.get("std") is None:
            try:
                resolved_cfg = timm.data.resolve_model_data_config(self.rgb_backbone)
                if isinstance(resolved_cfg, dict):
                    if (
                        pt_cfg.get("mean") is None
                        and resolved_cfg.get("mean") is not None
                    ):
                        pt_cfg["mean"] = resolved_cfg["mean"]
                    if (
                        pt_cfg.get("std") is None
                        and resolved_cfg.get("std") is not None
                    ):
                        pt_cfg["std"] = resolved_cfg["std"]
            except Exception:
                pass
        target_mean = np.asarray(pt_cfg.get("mean", IMAGENET_MEAN), dtype=np.float32)
        target_std = np.asarray(pt_cfg.get("std", IMAGENET_STD), dtype=np.float32)
        if target_mean.shape != (3,) or target_std.shape != (3,):
            target_mean = IMAGENET_MEAN.copy()
            target_std = IMAGENET_STD.copy()
        remap_scale = (IMAGENET_STD / np.maximum(target_std, 1e-6)).astype(np.float32)
        remap_shift = (
            (IMAGENET_MEAN - target_mean) / np.maximum(target_std, 1e-6)
        ).astype(np.float32)
        self.register_buffer(
            "rgb_remap_scale", torch.from_numpy(remap_scale).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "rgb_remap_shift", torch.from_numpy(remap_shift).view(1, 3, 1, 1)
        )
        self._needs_rgb_remap: bool = not (
            np.allclose(remap_scale, 1.0, atol=1e-6)
            and np.allclose(remap_shift, 0.0, atol=1e-6)
        )

        # Multi-Layer Hierarchical Token Fusion for ViT backbones (blocks 17, 19, 21, 23
        # for 24-block ViT-Large) and Stage-3 (16x16) + upsampled Stage-4 (8x8) multi-scale
        # spatial token fusion for BatchNorm-free hierarchical ConvNeXt backbones.
        self._capturing_blocks: bool = False
        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._capturing_stage: bool = False
        self._stage_hook_output: Optional[torch.Tensor] = None
        self.is_convnext: bool = (
            hasattr(self.rgb_backbone, "stages")
            and len(self.rgb_backbone.stages) >= 4
            and not hasattr(self.rgb_backbone, "blocks")
        )

        self.use_lora: bool = bool(use_lora and not self.is_convnext)
        self.lora_layers_injected: int = 0
        if self.use_lora:
            self.lora_layers_injected = _inject_lora_into_vit(
                self.rgb_backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        if (
            hasattr(self.rgb_backbone, "blocks")
            and isinstance(self.rgb_backbone.blocks, nn.ModuleList)
            and len(self.rgb_backbone.blocks) >= 7
        ):
            num_blocks = len(self.rgb_backbone.blocks)
            self.tap_indices: Tuple[int, ...] = (
                num_blocks - 7,
                num_blocks - 5,
                num_blocks - 3,
                num_blocks - 1,
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.rgb_dim) for _ in self.tap_indices]
            )
            self.layer_weights = nn.Parameter(torch.zeros(len(self.tap_indices)))
            for idx in self.tap_indices:
                self.rgb_backbone.blocks[idx].register_forward_hook(
                    _BlockCaptureHook(self, idx)
                )
        elif (
            getattr(self.rgb_backbone, "is_hf_dinov2", False)
            and int(getattr(self.rgb_backbone, "num_hidden_layers", 0)) >= 7
        ):
            num_blocks = int(self.rgb_backbone.num_hidden_layers)
            self.tap_indices = (
                num_blocks - 7,
                num_blocks - 5,
                num_blocks - 3,
                num_blocks - 1,
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(self.rgb_dim) for _ in self.tap_indices]
            )
            self.layer_weights = nn.Parameter(torch.zeros(len(self.tap_indices)))
        else:
            self.tap_indices = ()
            self.layer_norms = nn.ModuleList()
            self.layer_weights = None

        if self.is_convnext:
            self.stage3_dim, _stage4_dim = _infer_convnext_stage_dims(
                self.rgb_backbone, default_stage4=self.rgb_dim
            )
            self.stage3_proj = nn.Sequential(
                nn.LayerNorm(self.stage3_dim),
                nn.Linear(self.stage3_dim, self.rgb_dim),
            )
            self.stage4_norm = nn.LayerNorm(self.rgb_dim)
            self.multiscale_weights = nn.Parameter(torch.zeros(2))
            self.rgb_backbone.stages[2].register_forward_hook(_StageCaptureHook(self))
        else:
            self.stage3_dim = 0
            self.stage3_proj = None
            self.stage4_norm = None
            self.multiscale_weights = None

        self.rgb_spatial_norm = nn.LayerNorm(self.rgb_dim)
        self.rgb_mil_pool = GatedMILAttentionPool(
            embed_dim=self.rgb_dim, hidden_dim=256, dropout=0.20
        )
        self.rgb_head_norm = nn.LayerNorm(self.rgb_dim * 2)

        if self.use_od_stream:
            self.od_branch = BeerLambertODBranch(out_dim=od_dim, dropout=0.15)
            self.fusion = CrossModalGatedFusion(
                rgb_dim=self.rgb_dim * 2,
                od_dim=od_dim,
                fusion_dim=fusion_dim,
                dropout=0.25,
            )
        else:
            self.od_branch = None
            self.fusion = nn.Sequential(
                nn.Linear(self.rgb_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )

        self.classifier = DualPrototypeCosineHead(
            in_dim=fusion_dim,
            num_samples=5,
            dropout_rate=head_dropout,
            init_scale=10.0,
            min_scale=2.0,
            max_scale=25.0,
        )

        self.domain_discriminator = ConditionalDomainDiscriminator(
            embed_dim=fusion_dim,
            num_centers=max(2, self.num_centers),
            hidden_dim=256,
            dropout=0.25,
        )

        self.provenance["vit_input_size"] = self.vit_input_size
        self.provenance["rgb_embed_dim"] = self.rgb_dim
        self.provenance["use_od_stream"] = self.use_od_stream
        self.provenance["tap_indices"] = list(self.tap_indices)
        self.provenance["is_convnext"] = self.is_convnext
        self.provenance["use_lora"] = self.use_lora
        self.provenance["lora_layers_injected"] = int(self.lora_layers_injected)
        self.provenance["lora_rank"] = int(lora_rank) if self.use_lora else 0
        self.provenance["lora_alpha"] = float(lora_alpha) if self.use_lora else 0.0
        self.provenance["stage3_dim"] = int(self.stage3_dim)
        self.provenance["rgb_remap_active"] = bool(self._needs_rgb_remap)
        self.provenance["input_mean"] = [round(float(x), 6) for x in target_mean]
        self.provenance["input_std"] = [round(float(x), 6) for x in target_std]

    def train(self, mode: bool = True) -> "DualStreamStainViT":
        super().train(mode)
        if not self._backbone_trainable:
            self.rgb_backbone.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        """
        Toggle backbone / LoRA adapter gradients. When trainable=False, the
        pretrained backbone is locked in eval() mode with zero gradients so
        pretrained pathology representations stay strictly invariant while
        Spatial-MIL, OD branch, fusion, and cosine head train cleanly.
        """
        self._backbone_trainable = bool(trainable)
        if self.use_lora:
            for name, param in self.rgb_backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(self._backbone_trainable)
                else:
                    param.requires_grad_(False)
        else:
            for param in self.rgb_backbone.parameters():
                param.requires_grad_(self._backbone_trainable)
        if not self._backbone_trainable:
            self.rgb_backbone.eval()

    def _forward_backbone_raw_tokens(
        self, rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        if rgb.shape[-2] != self.vit_input_size or rgb.shape[-1] != self.vit_input_size:
            x_vit = F.interpolate(
                rgb,
                size=(self.vit_input_size, self.vit_input_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_vit = rgb

        if self._needs_rgb_remap:
            x_vit = (
                x_vit * self.rgb_remap_scale.to(dtype=x_vit.dtype)
                + self.rgb_remap_shift.to(dtype=x_vit.dtype)
            )

        backbone_grad = torch.is_grad_enabled() and self._backbone_trainable
        if backbone_grad and self.use_lora and self.grad_checkpointing and self.training:
            x_vit = x_vit.detach().requires_grad_(True)

        with torch.set_grad_enabled(backbone_grad):
            if (
                getattr(self.rgb_backbone, "is_hf_dinov2", False)
                and self.tap_indices
                and hasattr(self.rgb_backbone, "forward_features_with_taps")
            ):
                feats, tapped = self.rgb_backbone.forward_features_with_taps(
                    x_vit, self.tap_indices
                )
                stage3_feat = None
            else:
                self._hook_outputs.clear()
                self._stage_hook_output = None
                self._capturing_blocks = bool(self.tap_indices)
                self._capturing_stage = bool(self.is_convnext)
                try:
                    feats = self.rgb_backbone.forward_features(x_vit)
                    tapped = [
                        self._hook_outputs[idx]
                        for idx in self.tap_indices
                        if idx in self._hook_outputs
                    ]
                    stage3_feat = self._stage_hook_output
                finally:
                    self._capturing_blocks = False
                    self._capturing_stage = False
                    self._hook_outputs.clear()
                    self._stage_hook_output = None

        return feats, tapped, stage3_feat

    def _pool_rgb_from_backbone_tokens(
        self,
        feats: torch.Tensor,
        tapped: List[torch.Tensor],
        stage3_feat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self.is_convnext
            and stage3_feat is not None
            and self.stage3_proj is not None
            and self.stage4_norm is not None
            and self.multiscale_weights is not None
            and feats.ndim == 4
            and stage3_feat.ndim == 4
        ):
            s4_nchw = (
                feats if feats.shape[1] == self.rgb_dim else feats.permute(0, 3, 1, 2)
            )
            s3_nchw = (
                stage3_feat
                if stage3_feat.shape[1] == self.stage3_dim
                else stage3_feat.permute(0, 3, 1, 2)
            )
            B = s4_nchw.shape[0]
            H3, W3 = s3_nchw.shape[2], s3_nchw.shape[3]

            s4_nhwc = self.stage4_norm(s4_nchw.permute(0, 2, 3, 1))
            cls_tok = s4_nhwc.mean(dim=(1, 2))

            s4_up_nchw = F.interpolate(
                s4_nhwc.permute(0, 3, 1, 2),
                size=(H3, W3),
                mode="bilinear",
                align_corners=False,
            )
            s4_up_toks = s4_up_nchw.flatten(2).transpose(1, 2)

            s3_toks = self.stage3_proj(
                s3_nchw.permute(0, 2, 3, 1).reshape(B, H3 * W3, self.stage3_dim)
            )

            ms_w = torch.softmax(self.multiscale_weights, dim=0).to(dtype=s3_toks.dtype)
            spatial_toks = ms_w[0] * s3_toks + ms_w[1] * s4_up_toks
        else:
            if (
                self.tap_indices
                and self.layer_weights is not None
                and len(tapped) == len(self.tap_indices)
                and all(tok.ndim == 3 for tok in tapped)
            ):
                normed_layers = [
                    ln(tok) for ln, tok in zip(self.layer_norms, tapped)
                ]
                weights = torch.softmax(self.layer_weights, dim=0).to(
                    dtype=normed_layers[0].dtype
                )
                fused = (
                    torch.stack(normed_layers, dim=0) * weights.view(-1, 1, 1, 1)
                ).sum(dim=0)
            else:
                fused = feats

            if fused.ndim == 3:
                n_prefix = int(getattr(self.rgb_backbone, "num_prefix_tokens", 1))
                if n_prefix >= 1 and fused.shape[1] > n_prefix:
                    cls_tok = fused[:, 0, :]
                    spatial_toks = fused[:, n_prefix:, :]
                else:
                    cls_tok = fused.mean(dim=1)
                    spatial_toks = fused
            elif fused.ndim == 4:
                if fused.shape[-1] == self.rgb_dim:
                    spatial_toks = fused.reshape(fused.shape[0], -1, self.rgb_dim)
                else:
                    spatial_toks = fused.flatten(2).transpose(1, 2)
                cls_tok = spatial_toks.mean(dim=1)
            else:
                raise ValueError(
                    f"Unexpected backbone feature shape: {tuple(fused.shape)}"
                )

        spatial_toks = self.rgb_spatial_norm(spatial_toks)
        mil_tok, _ = self.rgb_mil_pool(spatial_toks)
        combined = torch.cat([cls_tok, mil_tok], dim=-1)
        return self.rgb_head_norm(combined)

    def _extract_rgb_representation(self, rgb: torch.Tensor) -> torch.Tensor:
        feats, tapped, stage3_feat = self._forward_backbone_raw_tokens(rgb)
        return self._pool_rgb_from_backbone_tokens(feats, tapped, stage3_feat)

    def forward_from_backbone_tokens(
        self,
        feats: torch.Tensor,
        tapped: List[torch.Tensor],
        stage3_feat: Optional[torch.Tensor],
        od: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate MIL/OD/Fusion/Cosine head from cached frozen backbone tokens."""
        rgb_emb = self._pool_rgb_from_backbone_tokens(feats, tapped, stage3_feat)
        if self.use_od_stream and self.od_branch is not None:
            if od is None:
                raise ValueError("OD tensor required when use_od_stream=True.")
            od_emb, _ = self.od_branch(od)
            fused_emb = self.fusion(rgb_emb, od_emb)
        else:
            fused_emb = self.fusion(rgb_emb)
        return self.classifier(fused_emb)

    def forward(
        self,
        rgb: torch.Tensor,
        od: Optional[torch.Tensor] = None,
        return_domain: bool = False,
        grl_lambda: float = 1.0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        rgb_emb = self._extract_rgb_representation(rgb)

        od_logits: Optional[torch.Tensor] = None
        if self.use_od_stream and self.od_branch is not None:
            if od is None:
                od = torch.zeros_like(rgb)
            od_emb, od_logits = self.od_branch(od)
            fused_emb = self.fusion(rgb_emb, od_emb)
        else:
            fused_emb = self.fusion(rgb_emb)

        logits = self.classifier(fused_emb)

        domain_logits: Optional[torch.Tensor] = None
        if return_domain and self.training:
            domain_logits = self.domain_discriminator(
                fused_emb, logits, grl_lambda=grl_lambda
            )

        return {
            "logits": logits,
            "od_logits": od_logits,
            "domain_logits": domain_logits,
            "embedding": fused_emb,
        }

    @torch.no_grad()
    def predict_proba(
        self,
        rgb: torch.Tensor,
        od: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference helper guaranteed to be 100% pixel-only and identical across
        validation and test sets. Supports both single-view (B, 3, H, W) and
        multi-view D4 TTA (B, V, 3, H, W) tensors by averaging logits across views
        prior to sigmoid (preserving Bayes symmetry around threshold 0.5).
        """
        if rgb.ndim == 5:
            B, V, C, H, W = rgb.shape
            rgb_flat = rgb.reshape(B * V, C, H, W)
            od_flat = od.reshape(B * V, C, H, W) if od is not None else None
            out = self.forward(rgb_flat, od_flat, return_domain=False)
            logits = out["logits"].reshape(B, V).mean(dim=1)
        elif rgb.ndim == 4:
            out = self.forward(rgb, od, return_domain=False)
            logits = out["logits"]
        else:
            raise ValueError(
                f"Expected 4D or 5D RGB tensor, got shape {tuple(rgb.shape)}"
            )

        probs = torch.sigmoid(logits)
        return torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)

    def get_model_provenance(self) -> Dict[str, Any]:
        """Return auditable model identity, version, pretraining corpus, and parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = dict(self.provenance)
        info["total_params"] = int(total_params)
        info["trainable_params"] = int(trainable_params)
        return info


def _apply_d4_tensor_batch(x: torch.Tensor, view_idx: int) -> torch.Tensor:
    """Apply exact D4 dihedral symmetry to a 4D GPU tensor (B, C, H, W)."""
    out = x
    if view_idx >= 4:
        out = torch.flip(out, dims=[-1])
    k = view_idx % 4
    if k > 0:
        out = torch.rot90(out, k=k, dims=(-2, -1))
    return out


class FrozenPFMExtractor(nn.Module):
    """
    Strictly frozen Pathology Foundation Model representation extractor running
    in torch.no_grad() + bfloat16/float16 AMP.

    Taps the last 4 transformer blocks (num_blocks-7, -5, -3, -1), performs
    exact register-token separation, applies multi-scale [CLS, Mean, GeM(p=3), Max]
    spatial pooling, averages across D4 views, and outputs a unit-norm concatenated
    float32 embedding (4 * D) with explicit native mean/std normalization.
    """

    def __init__(
        self,
        backbone_name: str,
        target_size: int = 256,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        exclude_backbones: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = amp_dtype

        self.backbone, self.provenance = _create_timm_backbone(
            preferred_name=backbone_name,
            pretrained=True,
            drop_path_rate=0.0,
            exclude_backbones=exclude_backbones,
        )
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval().to(self.device)

        self.embed_dim = int(getattr(self.backbone, "num_features", 1024))
        self.out_dim = self.embed_dim * 4
        self.input_size = _infer_backbone_input_size(
            self.backbone, target_size=target_size
        )

        # Resolve patch size for dynamic register-token separation & VRAM batching
        if getattr(self.backbone, "is_hf_dinov2", False):
            self.patch_size = int(getattr(self.backbone, "patch_size", 16))
        else:
            pe = getattr(self.backbone, "patch_embed", None)
            ps = getattr(pe, "patch_size", (16, 16)) if pe is not None else (16, 16)
            self.patch_size = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)

        # Resolve exact native normalization mean/std (honoring Kaiko 0.5/0.5/0.5)
        pt_cfg = dict(
            getattr(self.backbone, "pretrained_cfg", None)
            or getattr(self.backbone, "default_cfg", None)
            or {}
        )
        if self.provenance.get("norm_mean") is not None:
            pt_cfg["mean"] = self.provenance["norm_mean"]
        if self.provenance.get("norm_std") is not None:
            pt_cfg["std"] = self.provenance["norm_std"]
        if pt_cfg.get("mean") is None or pt_cfg.get("std") is None:
            try:
                resolved = timm.data.resolve_model_data_config(self.backbone)
                if isinstance(resolved, dict):
                    pt_cfg.setdefault("mean", resolved.get("mean"))
                    pt_cfg.setdefault("std", resolved.get("std"))
            except Exception:
                pass
        mean = np.asarray(pt_cfg.get("mean", IMAGENET_MEAN), dtype=np.float32)
        std = np.asarray(pt_cfg.get("std", IMAGENET_STD), dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,):
            mean = IMAGENET_MEAN.copy()
            std = IMAGENET_STD.copy()

        self.register_buffer(
            "norm_mean", torch.from_numpy(mean).view(1, 3, 1, 1).to(self.device)
        )
        self.register_buffer(
            "norm_std", torch.from_numpy(std).view(1, 3, 1, 1).to(self.device)
        )

        # Set up 4-layer hierarchical tapping: (num_blocks-7, -5, -3, -1)
        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hook_handles: List[Any] = []
        if (
            hasattr(self.backbone, "blocks")
            and isinstance(self.backbone.blocks, nn.ModuleList)
            and len(self.backbone.blocks) >= 7
        ):
            nb = len(self.backbone.blocks)
            self.tap_indices: Tuple[int, ...] = (nb - 7, nb - 5, nb - 3, nb - 1)
            for idx in self.tap_indices:
                def _make_hook(b_idx: int):
                    def _hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
                        self._hook_outputs[b_idx] = out
                    return _hook
                self._hook_handles.append(
                    self.backbone.blocks[idx].register_forward_hook(_make_hook(idx))
                )
        elif (
            getattr(self.backbone, "is_hf_dinov2", False)
            and int(getattr(self.backbone, "num_hidden_layers", 0)) >= 7
        ):
            nb = int(self.backbone.num_hidden_layers)
            self.tap_indices = (nb - 7, nb - 5, nb - 3, nb - 1)
        else:
            self.tap_indices = ()

        # Probe a dummy tensor to validate resolution & exact prefix/register count
        self._validate_resolution_and_prefix()

        self.provenance["input_size"] = int(self.input_size)
        self.provenance["patch_size"] = int(self.patch_size)
        self.provenance["embed_dim"] = int(self.embed_dim)
        self.provenance["out_dim"] = int(self.out_dim)
        self.provenance["num_prefix_tokens"] = int(self.num_prefix_tokens)
        self.provenance["tap_indices"] = list(self.tap_indices)
        self.provenance["input_mean"] = [round(float(x), 6) for x in mean]
        self.provenance["input_std"] = [round(float(x), 6) for x in std]

    @torch.no_grad()
    def _validate_resolution_and_prefix(self) -> None:
        sizes_to_try = [self.input_size]
        native_sz = int(getattr(self.backbone, "native_image_size", 224))
        if native_sz not in sizes_to_try:
            sizes_to_try.append(native_sz)
        if 224 not in sizes_to_try:
            sizes_to_try.append(224)

        last_exc: Optional[Exception] = None
        for sz in sizes_to_try:
            try:
                dummy = torch.zeros(1, 3, sz, sz, device=self.device)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    tokens = self._forward_fused_tokens(dummy)
                self.input_size = int(sz)
                n_spatial = (sz // self.patch_size) * (sz // self.patch_size)
                seq_len = int(tokens.shape[1]) if tokens.ndim == 3 else 0
                if tokens.ndim == 3 and seq_len > n_spatial and n_spatial > 0:
                    self.num_prefix_tokens = int(seq_len - n_spatial)
                else:
                    self.num_prefix_tokens = int(
                        getattr(self.backbone, "num_prefix_tokens", 1)
                    )
                return
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(
            f"FrozenPFMExtractor failed resolution check for {self.provenance['backbone_name']}: {last_exc}"
        )

    @torch.no_grad()
    def _forward_fused_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if (
            getattr(self.backbone, "is_hf_dinov2", False)
            and self.tap_indices
            and hasattr(self.backbone, "forward_features_with_taps")
        ):
            feats, tapped = self.backbone.forward_features_with_taps(
                x, self.tap_indices
            )
        else:
            self._hook_outputs.clear()
            feats = self.backbone.forward_features(x)
            tapped = [
                self._hook_outputs[idx]
                for idx in self.tap_indices
                if idx in self._hook_outputs
            ]
            self._hook_outputs.clear()

        if (
            self.tap_indices
            and len(tapped) == len(self.tap_indices)
            and all(tok.ndim == 3 for tok in tapped)
        ):
            normed = [
                F.layer_norm(tok.float(), (tok.shape[-1],)) for tok in tapped
            ]
            return torch.stack(normed, dim=0).mean(dim=0)
        return feats.float()

    @torch.no_grad()
    def _pool_tokens(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim == 3:
            n_prefix = self.num_prefix_tokens
            if n_prefix >= 1 and fused.shape[1] > n_prefix:
                cls_tok = fused[:, 0, :]
                spatial = fused[:, n_prefix:, :]
            else:
                cls_tok = fused.mean(dim=1)
                spatial = fused
        elif fused.ndim == 4:
            if fused.shape[-1] == self.embed_dim:
                spatial = fused.reshape(fused.shape[0], -1, self.embed_dim)
            else:
                spatial = fused.flatten(2).transpose(1, 2)
            cls_tok = spatial.mean(dim=1)
        else:
            raise ValueError(f"Unexpected token tensor shape: {tuple(fused.shape)}")

        cls_vec = F.normalize(cls_tok, p=2, dim=-1, eps=1e-6)
        mean_vec = F.normalize(spatial.mean(dim=1), p=2, dim=-1, eps=1e-6)
        # Signed cubic GeM(p=3) emphasizes focal high-activation tumor-nest tokens
        m3 = (spatial * spatial * spatial).mean(dim=1)
        gem_raw = torch.sign(m3) * torch.pow(m3.abs().clamp(min=1e-12), 1.0 / 3.0)
        gem_vec = F.normalize(gem_raw, p=2, dim=-1, eps=1e-6)
        max_vec = F.normalize(spatial.amax(dim=1), p=2, dim=-1, eps=1e-6)

        return torch.cat([cls_vec, mean_vec, gem_vec, max_vec], dim=-1)

    @torch.no_grad()
    def _normalize_uint8_batch(self, x_u8: torch.Tensor) -> torch.Tensor:
        x = x_u8.to(self.device, non_blocking=True).float().div_(255.0)
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return (x - self.norm_mean) / self.norm_std

    @torch.no_grad()
    def extract_d4_batch(
        self,
        x_u8: torch.Tensor,
        d4_views: int = 4,
        max_subbatch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract L2-normalized [CLS, Mean, GeM-3, Max] embeddings averaged over
        `d4_views` (1, 4, or 8) exact D4 dihedral symmetries on GPU.
        """
        if d4_views == 1:
            view_indices = (0,)
        elif d4_views == 4:
            view_indices = (0, 1, 2, 7)
        elif d4_views == 8:
            view_indices = (0, 1, 2, 3, 4, 5, 6, 7)
        else:
            raise ValueError(f"d4_views must be 1, 4, or 8; got {d4_views}")

        if max_subbatch is None:
            max_subbatch = 64 if self.patch_size <= 8 else 256

        x_norm = self._normalize_uint8_batch(x_u8)
        B = x_norm.shape[0]
        acc = torch.zeros(B, self.out_dim, device=self.device, dtype=torch.float32)

        for v_idx in view_indices:
            xv = _apply_d4_tensor_batch(x_norm, v_idx)
            for start in range(0, B, max_subbatch):
                end = min(start + max_subbatch, B)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                    dtype=self.amp_dtype,
                ):
                    fused = self._forward_fused_tokens(xv[start:end])
                acc[start:end] += self._pool_tokens(fused)

        acc.div_(float(len(view_indices)))
        # Block-wise L2-normalize each of the 4 pooling modalities and scale by 0.5
        # so total embedding norm is 1.0 with zero view-count norm shift.
        parts = acc.view(B, 4, self.embed_dim)
        parts = F.normalize(parts, p=2, dim=-1, eps=1e-6) * 0.5
        return parts.reshape(B, self.out_dim)

    def close(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


class _ResidualMLPBlock(nn.Module):
    """Pre-LayerNorm residual GELU MLP block for multi-PFM embedding fusion."""

    def __init__(self, dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MultiPFMDomainInvariantMLP(nn.Module):
    """
    Multi-PFM Domain-Invariant Residual MLP with per-PFM projection stems,
    stochastic per-PFM stream dropout (forcing cross-PFM consensus), OD
    morphological descriptor fusion, and bias-free DualPrototypeCosineHead.
    """

    def __init__(
        self,
        pfm_dims: List[int],
        od_dim: int,
        stem_dim: int = 384,
        od_hidden_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.30,
        stream_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.pfm_dims = [int(d) for d in pfm_dims]
        self.od_dim = int(od_dim)
        self.stream_dropout = float(stream_dropout)

        self.pfm_stems = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(d, stem_dim),
                    nn.LayerNorm(stem_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                )
                for d in self.pfm_dims
            ]
        )
        self.od_stem = nn.Sequential(
            nn.LayerNorm(od_dim),
            nn.Linear(od_dim, od_hidden_dim),
            nn.LayerNorm(od_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        concat_dim = stem_dim * len(self.pfm_dims) + od_hidden_dim
        self.in_proj = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res1 = _ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.res2 = _ResidualMLPBlock(hidden_dim, dropout=dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

        self.classifier = DualPrototypeCosineHead(
            in_dim=hidden_dim,
            num_samples=5,
            dropout_rate=dropout,
            init_scale=10.0,
            min_scale=2.0,
            max_scale=25.0,
        )

    def forward(
        self,
        pfm_feats: List[torch.Tensor],
        od_feats: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        projected = [stem(x) for stem, x in zip(self.pfm_stems, pfm_feats)]

        # Per-PFM stream dropout during training (never drops all streams at once)
        if self.training and self.stream_dropout > 0.0 and len(projected) > 1:
            B = projected[0].shape[0]
            device = projected[0].device
            n_streams = len(projected)
            keep = (
                torch.rand(B, n_streams, device=device) >= self.stream_dropout
            ).float()
            # Ensure at least one PFM stream survives per sample
            all_zero = keep.sum(dim=1) < 0.5
            if bool(all_zero.any().item()):
                rand_idx = torch.randint(
                    0, n_streams, (int(all_zero.sum().item()),), device=device
                )
                keep[all_zero, rand_idx] = 1.0
            scale = float(n_streams) / keep.sum(dim=1, keepdim=True).clamp(min=1.0)
            projected = [
                p * (keep[:, i : i + 1] * scale) for i, p in enumerate(projected)
            ]

        od_proj = self.od_stem(od_feats)
        h = self.in_proj(torch.cat(projected + [od_proj], dim=-1))
        h = self.res1(h)
        h = self.res2(h)
        emb = self.out_norm(h)
        logits = self.classifier(emb)
        return {
            "logits": logits,
            "od_logits": None,
            "domain_logits": None,
            "embedding": emb,
        }


class CenterDebiasedLogisticProbe:
    """
    Fold-internal Center-Debiased L2 Logistic Regression Probe.

    Estimates the 1D inter-center mean-difference unit vector u_c strictly on
    the training fold (class-balanced across centers; falling back to the top-1
    stain-perturbation shift direction when a LOCO training fold has 1 center),
    softly projects out beta * (x @ u_c) * u_c, and fits L2 LogisticRegression
    with slide-tempered class-balanced weights.
    """

    def __init__(
        self,
        C: float = 0.25,
        beta: float = 0.75,
        od_weight: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.C = float(C)
        self.beta = float(beta)
        self.od_weight = float(od_weight)
        self.seed = int(seed)
        self.od_scaler = StandardScaler()
        self.u_c: Optional[np.ndarray] = None
        self.clf: Optional[LogisticRegression] = None

    def _assemble_features(
        self,
        pfm_list: List[np.ndarray],
        od_feats: np.ndarray,
        fit_scaler: bool = False,
    ) -> np.ndarray:
        if fit_scaler:
            od_s = self.od_scaler.fit_transform(od_feats.astype(np.float32))
        else:
            od_s = self.od_scaler.transform(od_feats.astype(np.float32))
        od_s = np.clip(od_s, -6.0, 6.0).astype(np.float32)
        od_norms = np.linalg.norm(od_s, axis=1, keepdims=True)
        od_s = (od_s / np.maximum(od_norms, 1e-6)) * np.float32(self.od_weight)
        blocks = [np.asarray(x, dtype=np.float32) for x in pfm_list] + [od_s]
        return np.concatenate(blocks, axis=1)

    def _debias(self, X: np.ndarray) -> np.ndarray:
        if self.u_c is None or self.beta <= 0.0:
            return X
        proj = X @ self.u_c
        return X - np.float32(self.beta) * np.outer(proj, self.u_c)

    def fit(
        self,
        pfm_raw_list: List[np.ndarray],
        od_feats: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        sample_weights: np.ndarray,
        pfm_pert_mean_list: Optional[List[np.ndarray]] = None,
    ) -> "CenterDebiasedLogisticProbe":
        X_raw = self._assemble_features(pfm_raw_list, od_feats, fit_scaler=True)
        y = np.asarray(labels, dtype=np.int64)
        c_arr = np.asarray(centers).astype(str)
        unique_c = sorted(np.unique(c_arr).tolist())

        # Estimate unit center/stain shift vector strictly on training fold
        self.u_c = None
        if len(unique_c) >= 2:
            c_means = []
            for cid in unique_c[:2]:
                m_c = c_arr == cid
                m0 = m_c & (y == 0)
                m1 = m_c & (y == 1)
                if m0.any() and m1.any():
                    mu = 0.5 * (X_raw[m0].mean(axis=0) + X_raw[m1].mean(axis=0))
                else:
                    mu = X_raw[m_c].mean(axis=0)
                c_means.append(mu.astype(np.float32))
            diff = c_means[1] - c_means[0]
            norm = float(np.linalg.norm(diff))
            if norm > 1e-6:
                self.u_c = (diff / norm).astype(np.float32)
        elif pfm_pert_mean_list is not None:
            X_pert_tmp = self._assemble_features(
                pfm_pert_mean_list, od_feats, fit_scaler=False
            )
            delta = (X_pert_tmp - X_raw).astype(np.float32)
            # Top-1 stain-shift direction via 1-component TruncatedSVD
            svd1 = TruncatedSVD(n_components=1, random_state=self.seed)
            svd1.fit(delta)
            vec = svd1.components_[0].astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                self.u_c = (vec / norm).astype(np.float32)

        X_raw_deb = self._debias(X_raw)
        if pfm_pert_mean_list is not None:
            X_pert = self._assemble_features(
                pfm_pert_mean_list, od_feats, fit_scaler=False
            )
            X_pert_deb = self._debias(X_pert)
            X_fit = np.concatenate([X_raw_deb, X_pert_deb], axis=0)
            y_fit = np.concatenate([y, y], axis=0)
            w_fit = np.concatenate([sample_weights, sample_weights], axis=0)
        else:
            X_fit = X_raw_deb
            y_fit = y
            w_fit = sample_weights

        self.clf = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=400,
            random_state=self.seed,
        )
        self.clf.fit(X_fit, y_fit, sample_weight=w_fit)
        return self

    def predict_logits(
        self, pfm_list: List[np.ndarray], od_feats: np.ndarray
    ) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("CenterDebiasedLogisticProbe must be fitted first.")
        X = self._assemble_features(pfm_list, od_feats, fit_scaler=False)
        X_deb = self._debias(X)
        return self.clf.decision_function(X_deb).astype(np.float64)


class FoldGBDTFeatureBuilder:
    """
    Strictly fold-internal tabular feature builder for LightGBM and XGBoost:
      1. Per-PFM TruncatedSVD components (n_svd=32 per PFM).
      2. Fold-internal class-prototype cosine similarities (global + 4 pooling
         subblocks [CLS, Mean, GeM, Max] + center-debiased prototype similarity +
         centroid L2 distances), using exact slide-grouped leave-one-slide-out
         prototype subtraction on training rows to eliminate in-fold overfitting.
      3. Training-only standardized Beer-Lambert OD morphological descriptors.
      4. Cross-PFM consensus features (mean/std/min/max of prototype margins).
    """

    def __init__(self, n_svd: int = 32, seed: int = 42) -> None:
        self.n_svd = int(n_svd)
        self.seed = int(seed)
        self.svds: List[TruncatedSVD] = []
        self.od_scaler = StandardScaler()
        self.prototypes: List[Dict[str, Any]] = []
        self.n_pfms: int = 0

    @staticmethod
    def _l2_norm_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-8)

    @staticmethod
    def _l2_norm_vec(v: np.ndarray) -> np.ndarray:
        return v / max(float(np.linalg.norm(v)), 1e-8)

    def _fit_one_pfm_prototypes(
        self,
        X: np.ndarray,
        y: np.ndarray,
        centers: np.ndarray,
        slides: np.ndarray,
        X_pert_mean: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        D_total = X.shape[1]
        D_sub = D_total // 4
        y_arr = np.asarray(y, dtype=np.int64)
        c_arr = np.asarray(centers).astype(str)
        unique_c = sorted(np.unique(c_arr).tolist())

        # Class-0 and Class-1 global sums & counts for exact leave-one-slide-out
        m0 = y_arr == 0
        m1 = y_arr == 1
        sum0 = X[m0].sum(axis=0, dtype=np.float64)
        sum1 = X[m1].sum(axis=0, dtype=np.float64)
        cnt0 = int(m0.sum())
        cnt1 = int(m1.sum())

        proto0 = self._l2_norm_vec(sum0 / max(cnt0, 1))
        proto1 = self._l2_norm_vec(sum1 / max(cnt1, 1))

        # Per-subblock [CLS, Mean, GeM, Max] prototypes
        sub_p0 = []
        sub_p1 = []
        for b in range(4):
            s = b * D_sub
            e = (b + 1) * D_sub
            sub_p0.append(self._l2_norm_vec(sum0[s:e] / max(cnt0, 1)))
            sub_p1.append(self._l2_norm_vec(sum1[s:e] / max(cnt1, 1)))

        # Center/stain debiasing direction u_c for debiased cosine prototype features
        u_c = None
        if len(unique_c) >= 2:
            mus = []
            for cid in unique_c[:2]:
                mc = c_arr == cid
                mc0 = mc & m0
                mc1 = mc & m1
                if mc0.any() and mc1.any():
                    mu = 0.5 * (X[mc0].mean(axis=0) + X[mc1].mean(axis=0))
                else:
                    mu = X[mc].mean(axis=0)
                mus.append(mu.astype(np.float64))
            d_vec = mus[1] - mus[0]
            if float(np.linalg.norm(d_vec)) > 1e-6:
                u_c = self._l2_norm_vec(d_vec)
        elif X_pert_mean is not None:
            svd1 = TruncatedSVD(n_components=1, random_state=self.seed)
            svd1.fit((X_pert_mean - X).astype(np.float32))
            v = svd1.components_[0].astype(np.float64)
            if float(np.linalg.norm(v)) > 1e-6:
                u_c = self._l2_norm_vec(v)

        if u_c is not None:
            beta_d = 0.85
            p0_deb = self._l2_norm_vec(proto0 - beta_d * float(proto0 @ u_c) * u_c)
            p1_deb = self._l2_norm_vec(proto1 - beta_d * float(proto1 @ u_c) * u_c)
        else:
            p0_deb = proto0
            p1_deb = proto1

        return {
            "D_sub": D_sub,
            "sum0": sum0,
            "sum1": sum1,
            "cnt0": cnt0,
            "cnt1": cnt1,
            "proto0": proto0.astype(np.float32),
            "proto1": proto1.astype(np.float32),
            "sub_p0": [p.astype(np.float32) for p in sub_p0],
            "sub_p1": [p.astype(np.float32) for p in sub_p1],
            "u_c": u_c.astype(np.float32) if u_c is not None else None,
            "p0_deb": p0_deb.astype(np.float32),
            "p1_deb": p1_deb.astype(np.float32),
        }

    def _compute_proto_features_for_pfm(
        self,
        X: np.ndarray,
        info: Dict[str, Any],
        train_y: Optional[np.ndarray] = None,
        train_slides: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_f = np.asarray(X, dtype=np.float32)
        X_unit = self._l2_norm_rows(X_f)
        N = X_f.shape[0]
        D_sub = info["D_sub"]

        # On training fold, use exact leave-one-slide-out prototypes to avoid shrinkage bias
        if train_y is not None and train_slides is not None:
            y_arr = np.asarray(train_y, dtype=np.int64)
            s_arr = np.asarray(train_slides).astype(str)
            c0 = np.zeros(N, dtype=np.float32)
            c1 = np.zeros(N, dtype=np.float32)
            sub_c0 = [np.zeros(N, dtype=np.float32) for _ in range(4)]
            sub_c1 = [np.zeros(N, dtype=np.float32) for _ in range(4)]
            d0 = np.zeros(N, dtype=np.float32)
            d1 = np.zeros(N, dtype=np.float32)

            for sid in np.unique(s_arr):
                idx = np.where(s_arr == sid)[0]
                idx0 = idx[y_arr[idx] == 0]
                idx1 = idx[y_arr[idx] == 1]

                rem_cnt0 = max(info["cnt0"] - len(idx0), 1)
                rem_cnt1 = max(info["cnt1"] - len(idx1), 1)
                rem_sum0 = info["sum0"] - (
                    X_f[idx0].sum(axis=0, dtype=np.float64) if len(idx0) > 0 else 0.0
                )
                rem_sum1 = info["sum1"] - (
                    X_f[idx1].sum(axis=0, dtype=np.float64) if len(idx1) > 0 else 0.0
                )
                p0_s = self._l2_norm_vec(rem_sum0 / rem_cnt0).astype(np.float32)
                p1_s = self._l2_norm_vec(rem_sum1 / rem_cnt1).astype(np.float32)

                c0[idx] = X_unit[idx] @ p0_s
                c1[idx] = X_unit[idx] @ p1_s
                d0[idx] = np.linalg.norm(X_unit[idx] - p0_s[None, :], axis=1)
                d1[idx] = np.linalg.norm(X_unit[idx] - p1_s[None, :], axis=1)

                for b in range(4):
                    st, en = b * D_sub, (b + 1) * D_sub
                    xb = self._l2_norm_rows(X_f[idx, st:en])
                    bp0 = self._l2_norm_vec(rem_sum0[st:en] / rem_cnt0).astype(
                        np.float32
                    )
                    bp1 = self._l2_norm_vec(rem_sum1[st:en] / rem_cnt1).astype(
                        np.float32
                    )
                    sub_c0[b][idx] = xb @ bp0
                    sub_c1[b][idx] = xb @ bp1
        else:
            p0 = info["proto0"]
            p1 = info["proto1"]
            c0 = X_unit @ p0
            c1 = X_unit @ p1
            d0 = np.linalg.norm(X_unit - p0[None, :], axis=1)
            d1 = np.linalg.norm(X_unit - p1[None, :], axis=1)
            sub_c0 = []
            sub_c1 = []
            for b in range(4):
                st, en = b * D_sub, (b + 1) * D_sub
                xb = self._l2_norm_rows(X_f[:, st:en])
                sub_c0.append(xb @ info["sub_p0"][b])
                sub_c1.append(xb @ info["sub_p1"][b])

        # Center-debiased cosine similarity
        u_c = info["u_c"]
        if u_c is not None:
            X_deb = self._l2_norm_rows(
                X_f - np.float32(0.85) * np.outer(X_f @ u_c, u_c)
            )
            deb_c0 = X_deb @ info["p0_deb"]
            deb_c1 = X_deb @ info["p1_deb"]
        else:
            deb_c0, deb_c1 = c0, c1

        cols = [
            c0,
            c1,
            c1 - c0,
            (c1 + 1.0) / np.maximum(c0 + 1.0, 1e-4),
            deb_c0,
            deb_c1,
            deb_c1 - deb_c0,
            d0,
            d1,
            d0 - d1,
        ]
        for b in range(4):
            cols.extend([sub_c0[b], sub_c1[b], sub_c1[b] - sub_c0[b]])

        feat_mat = np.stack(cols, axis=1).astype(np.float32)
        margin = (c1 - c0).astype(np.float32)
        return feat_mat, margin

    def fit_transform(
        self,
        pfm_list: List[np.ndarray],
        od_feats: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        slides: np.ndarray,
        pfm_pert_mean_list: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        self.n_pfms = len(pfm_list)
        self.svds = []
        self.prototypes = []

        blocks: List[np.ndarray] = []
        margins: List[np.ndarray] = []

        for k, X_k in enumerate(pfm_list):
            X_f = np.asarray(X_k, dtype=np.float32)
            n_comp = min(self.n_svd, X_f.shape[1] - 1, X_f.shape[0] - 1)
            svd = TruncatedSVD(n_components=n_comp, random_state=self.seed + k)
            svd_feats = svd.fit_transform(X_f).astype(np.float32)
            self.svds.append(svd)

            X_p_k = pfm_pert_mean_list[k] if pfm_pert_mean_list is not None else None
            info = self._fit_one_pfm_prototypes(
                X_f, labels, centers, slides, X_pert_mean=X_p_k
            )
            self.prototypes.append(info)

            p_feats, margin = self._compute_proto_features_for_pfm(
                X_f, info, train_y=labels, train_slides=slides
            )
            blocks.append(svd_feats)
            blocks.append(p_feats)
            margins.append(margin)

        od_s = self.od_scaler.fit_transform(od_feats.astype(np.float32))
        blocks.append(np.clip(od_s, -6.0, 6.0).astype(np.float32))

        if len(margins) >= 2:
            m_stack = np.stack(margins, axis=1)
            consensus = np.stack(
                [
                    m_stack.mean(axis=1),
                    m_stack.std(axis=1),
                    m_stack.min(axis=1),
                    m_stack.max(axis=1),
                ],
                axis=1,
            ).astype(np.float32)
            blocks.append(consensus)

        return np.concatenate(blocks, axis=1)

    def transform(
        self,
        pfm_list: List[np.ndarray],
        od_feats: np.ndarray,
        train_y: Optional[np.ndarray] = None,
        train_slides: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        blocks: List[np.ndarray] = []
        margins: List[np.ndarray] = []

        for k, X_k in enumerate(pfm_list):
            X_f = np.asarray(X_k, dtype=np.float32)
            svd_feats = self.svds[k].transform(X_f).astype(np.float32)
            p_feats, margin = self._compute_proto_features_for_pfm(
                X_f, self.prototypes[k], train_y=train_y, train_slides=train_slides
            )
            blocks.append(svd_feats)
            blocks.append(p_feats)
            margins.append(margin)

        od_s = self.od_scaler.transform(od_feats.astype(np.float32))
        blocks.append(np.clip(od_s, -6.0, 6.0).astype(np.float32))

        if len(margins) >= 2:
            m_stack = np.stack(margins, axis=1)
            consensus = np.stack(
                [
                    m_stack.mean(axis=1),
                    m_stack.std(axis=1),
                    m_stack.min(axis=1),
                    m_stack.max(axis=1),
                ],
                axis=1,
            ).astype(np.float32)
            blocks.append(consensus)

        return np.concatenate(blocks, axis=1)


class DomainGeneralizingHistopathLoss(nn.Module):
    """
    Metric-aligned composite loss designed specifically for unseen-center
    balanced accuracy at fixed decision threshold 0.5:
      1. Label-Smoothed Focal-BCE ((1 - p_t)^gamma * BCE) prioritizing hard focal tumor patches.
      2. Differentiable Soft Balanced-Accuracy Surrogate: 1 - 0.5 * (soft_tpr + soft_tnr).
      3. Per-Center Recall-Parity Regularization: mean squared gap (soft_tpr - soft_tnr)^2
         per medical center around probability 0.5.
      4. Auxiliary OD-branch BCE loss for standalone nuclear morphology learning.
      5. Warmup-scheduled V-REx inter-center risk variance penalty.
      (Optional CDAN/consistency weights default to 0.0 for stability.)
    """

    def __init__(
        self,
        label_smoothing: float = 0.03,
        focal_gamma: float = 1.0,
        sba_weight: float = 0.30,
        parity_weight: float = 0.25,
        aux_od_weight: float = 0.15,
        vrex_weight: float = 0.25,
        cdan_weight: float = 0.0,
        logit_cons_weight: float = 0.0,
        emb_cons_weight: float = 0.0,
        min_samples_per_center: int = 2,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.focal_gamma = float(focal_gamma)
        self.sba_weight = float(sba_weight)
        self.parity_weight = float(parity_weight)
        self.aux_od_weight = float(aux_od_weight)
        self.vrex_weight = float(vrex_weight)
        self.cdan_weight = float(cdan_weight)
        self.logit_cons_weight = float(logit_cons_weight)
        self.emb_cons_weight = float(emb_cons_weight)
        self.min_samples_per_center = int(min_samples_per_center)

    def _smooth_targets(self, targets: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        if eps <= 0.0:
            return targets
        return targets * (1.0 - eps) + 0.5 * eps

    def forward(
        self,
        outputs: Dict[str, Optional[torch.Tensor]],
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
        domain_scale: float = 1.0,
        teacher_logits: Optional[torch.Tensor] = None,
        teacher_embedding: Optional[torch.Tensor] = None,
        is_perturbed: Optional[torch.Tensor] = None,
        cons_scale: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        logits = outputs["logits"]
        od_logits = outputs.get("od_logits", None)
        domain_logits = outputs.get("domain_logits", None)

        targets = labels.float().view_as(logits)
        smooth_targets = self._smooth_targets(targets)

        per_sample_bce = F.binary_cross_entropy_with_logits(
            logits, smooth_targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        if self.focal_gamma > 0.0:
            p_t = torch.where(targets >= 0.5, probs, 1.0 - probs)
            focal_mod = torch.pow(torch.clamp(1.0 - p_t, min=1e-6), self.focal_gamma)
            per_sample_cls = focal_mod * per_sample_bce
        else:
            per_sample_cls = per_sample_bce

        cls_loss = per_sample_cls.mean()
        total_loss = cls_loss

        pos_mask = targets >= 0.5
        neg_mask = ~pos_mask
        has_pos = bool(pos_mask.any().item())
        has_neg = bool(neg_mask.any().item())

        sba_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.sba_weight > 0.0 and has_pos and has_neg:
            soft_tpr = probs[pos_mask].mean()
            soft_tnr = (1.0 - probs[neg_mask]).mean()
            sba_loss = 1.0 - 0.5 * (soft_tpr + soft_tnr)
            total_loss = total_loss + self.sba_weight * sba_loss

        parity_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if self.parity_weight > 0.0:
            if center_idx is not None:
                unique_centers = torch.unique(center_idx)
                parity_terms = []
                for c in unique_centers:
                    c_mask = center_idx == c
                    c_pos = c_mask & pos_mask
                    c_neg = c_mask & neg_mask
                    if bool(c_pos.any().item()) and bool(c_neg.any().item()):
                        c_tpr = probs[c_pos].mean()
                        c_tnr = (1.0 - probs[c_neg]).mean()
                        parity_terms.append((c_tpr - c_tnr) ** 2)
                if parity_terms:
                    parity_loss = torch.stack(parity_terms).mean()
                    total_loss = total_loss + self.parity_weight * parity_loss
            elif has_pos and has_neg:
                soft_tpr = probs[pos_mask].mean()
                soft_tnr = (1.0 - probs[neg_mask]).mean()
                parity_loss = (soft_tpr - soft_tnr) ** 2
                total_loss = total_loss + self.parity_weight * parity_loss

        if od_logits is not None and self.aux_od_weight > 0.0:
            aux_loss = F.binary_cross_entropy_with_logits(
                od_logits, smooth_targets, reduction="mean"
            )
            total_loss = total_loss + self.aux_od_weight * aux_loss
        else:
            aux_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        vrex_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if center_idx is not None and self.vrex_weight > 0.0 and domain_scale > 0.0:
            unique_centers = torch.unique(center_idx)
            center_risks = []
            for c in unique_centers:
                mask = center_idx == c
                if int(mask.sum().item()) >= self.min_samples_per_center:
                    center_risks.append(per_sample_cls[mask].mean())
            if len(center_risks) >= 2:
                risk_stack = torch.stack(center_risks)
                vrex_loss = torch.var(risk_stack, unbiased=False)
                total_loss = total_loss + (self.vrex_weight * domain_scale) * vrex_loss

        cdan_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if (
            domain_logits is not None
            and center_idx is not None
            and self.cdan_weight > 0.0
            and domain_scale > 0.0
        ):
            valid_centers = torch.unique(center_idx)
            if (
                len(valid_centers) >= 2
                and int(center_idx.max().item()) < domain_logits.shape[-1]
            ):
                cdan_loss = F.cross_entropy(domain_logits, center_idx.long())
                total_loss = total_loss + (self.cdan_weight * domain_scale) * cdan_loss

        logit_cons_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        emb_cons_loss = torch.zeros((), device=logits.device, dtype=torch.float32)

        if (
            cons_scale > 0.0
            and (self.logit_cons_weight > 0.0 or self.emb_cons_weight > 0.0)
            and (teacher_logits is not None or teacher_embedding is not None)
        ):
            if is_perturbed is not None:
                pert_mask = is_perturbed.bool().view(-1)
            else:
                pert_mask = torch.ones(
                    logits.shape[0], device=logits.device, dtype=torch.bool
                )

            if int(pert_mask.sum().item()) > 0:
                if teacher_logits is not None and self.logit_cons_weight > 0.0:
                    s_logits = logits.float().view(-1)[pert_mask]
                    y_pert = targets.float().view(-1)[pert_mask]
                    t_logits_all = teacher_logits.detach().float().view(-1)
                    t_logits = (
                        t_logits_all[pert_mask]
                        if t_logits_all.shape[0] == logits.shape[0]
                        else t_logits_all
                    )

                    p_t_cons = torch.sigmoid(t_logits).clamp(1e-6, 1.0 - 1e-6)
                    p_s_cons = torch.sigmoid(s_logits).clamp(1e-6, 1.0 - 1e-6)

                    kl = p_t_cons * (torch.log(p_t_cons) - torch.log(p_s_cons)) + (
                        1.0 - p_t_cons
                    ) * (torch.log(1.0 - p_t_cons) - torch.log(1.0 - p_s_cons))
                    kl = torch.clamp(kl, min=0.0)

                    teacher_correct = ((p_t_cons >= 0.5) == (y_pert >= 0.5)).float()
                    teacher_true_conf = torch.where(
                        y_pert >= 0.5, p_t_cons, 1.0 - p_t_cons
                    )
                    gate_weight = (teacher_true_conf * teacher_correct).detach()
                    logit_cons_loss = (kl * gate_weight).mean()

                student_emb = outputs.get("embedding", None)
                if (
                    student_emb is not None
                    and teacher_embedding is not None
                    and self.emb_cons_weight > 0.0
                ):
                    s_emb = student_emb.float()[pert_mask]
                    t_emb_all = teacher_embedding.detach().float()
                    t_emb = (
                        t_emb_all[pert_mask]
                        if t_emb_all.shape[0] == student_emb.shape[0]
                        else t_emb_all
                    )
                    s_emb_norm = F.normalize(s_emb, p=2, dim=-1, eps=1e-6)
                    t_emb_norm = F.normalize(t_emb, p=2, dim=-1, eps=1e-6)
                    cos_sim = (s_emb_norm * t_emb_norm).sum(dim=-1)
                    emb_cons_loss = (1.0 - cos_sim).mean()

                total_loss = total_loss + float(cons_scale) * (
                    self.logit_cons_weight * logit_cons_loss
                    + self.emb_cons_weight * emb_cons_loss
                )

        return {
            "loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "sba_loss": sba_loss.detach(),
            "parity_loss": parity_loss.detach(),
            "aux_loss": aux_loss.detach(),
            "vrex_loss": vrex_loss.detach(),
            "cdan_loss": cdan_loss.detach(),
            "logit_cons_loss": logit_cons_loss.detach(),
            "emb_cons_loss": emb_cons_loss.detach(),
        }


def _get_backbone_layer_id(
    param_name: str,
    num_layers: int,
    stage_depths: Optional[Tuple[int, ...]] = None,
) -> int:
    """
    Map backbone parameter names to monotonic depth IDs in [0, num_layers] for LLRD.
    Supports ConvNeXt ('stem' -> 0, 'stages.{s}.blocks.{b}' & downsamples mapped
    monotonically into 12 depth groups), timm ViT/DINOv2 ('blocks.{i}'), HuggingFace
    DINOv2 ('embeddings' -> 0, 'encoder.layer.{i}' -> i+1), and SwinV2 ('layers.{i}').
    """
    if (
        param_name.startswith("stem.")
        or param_name == "stem"
        or "patch_embed" in param_name
        or "pos_embed" in param_name
        or "cls_token" in param_name
        or "reg_token" in param_name
        or "embeddings" in param_name
    ):
        return 0

    parts = param_name.split(".")
    if "stages" in parts:
        idx = parts.index("stages")
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            stage_id = int(parts[idx + 1])
            depths = (
                stage_depths
                if (stage_depths is not None and len(stage_depths) > 0)
                else (3, 3, 27, 3)
            )
            total_blocks = max(1, sum(depths))
            stage_id_clamped = min(max(stage_id, 0), len(depths) - 1)
            offset = sum(depths[:stage_id_clamped])
            block_id = 0
            if "blocks" in parts[idx + 2 :]:
                b_rel = parts[idx + 2 :].index("blocks")
                b_idx = idx + 2 + b_rel + 1
                if b_idx < len(parts) and parts[b_idx].isdigit():
                    block_id = int(parts[b_idx])
            block_pos = min(offset + block_id, total_blocks - 1)
            group_id = 1 + (block_pos * num_layers) // total_blocks
            return min(max(group_id, 1), num_layers)

    if "encoder" in parts and "layer" in parts:
        idx = parts.index("layer")
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return min(int(parts[idx + 1]) + 1, num_layers)

    if "blocks" in parts:
        idx = parts.index("blocks")
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return min(int(parts[idx + 1]) + 1, num_layers)
    if "layers" in parts:
        idx = parts.index("layers")
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            stage_id = int(parts[idx + 1])
            return min(stage_id + 1, num_layers)

    return num_layers


def _infer_backbone_num_layers(backbone: nn.Module) -> int:
    if hasattr(backbone, "num_hidden_layers") and getattr(
        backbone, "is_hf_dinov2", False
    ):
        return int(backbone.num_hidden_layers)
    if hasattr(backbone, "blocks") and isinstance(backbone.blocks, nn.ModuleList):
        return len(backbone.blocks)
    if hasattr(backbone, "stages"):
        return 12
    if hasattr(backbone, "layers") and isinstance(backbone.layers, nn.ModuleList):
        return len(backbone.layers)
    return 12


def build_optimizer_with_llrd(
    model: DualStreamStainViT,
    backbone_lr: float = 2e-4,
    head_lr: float = 3e-4,
    weight_decay: float = 0.03,
    layer_decay: float = 0.95,
) -> AdamW:
    """
    Construct AdamW optimizer with Layer-Wise Learning Rate Decay (LLRD) and
    explicit `backbone_group` tags separating LoRA/backbone parameters from
    head/fusion/OD parameters for two-stage Probe-Then-Adapt scheduling.
    """
    num_layers = _infer_backbone_num_layers(model.rgb_backbone)
    stage_depths: Optional[Tuple[int, ...]] = None
    if hasattr(model.rgb_backbone, "stages"):
        parsed_depths = tuple(
            len(st.blocks)
            for st in model.rgb_backbone.stages
            if hasattr(st, "blocks")
        )
        if parsed_depths:
            stage_depths = parsed_depths

    no_decay_keywords = (
        "bias",
        "norm",
        "pos_embed",
        "position_embeddings",
        "cls_token",
        "reg_token",
        "register_token",
        "mask_token",
        "gamma",
        "relative_position_bias",
        "prototypes",
        "logit_scale",
    )

    param_groups: Dict[Tuple[bool, float, float], List[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = (param.ndim <= 1) or any(
            kw in name.lower() for kw in no_decay_keywords
        )
        wd = 0.0 if is_no_decay else float(weight_decay)

        is_backbone = name.startswith("rgb_backbone.")
        if is_backbone:
            sub_name = name[len("rgb_backbone.") :]
            layer_id = _get_backbone_layer_id(
                sub_name, num_layers=num_layers, stage_depths=stage_depths
            )
            scale = layer_decay ** (num_layers - layer_id)
            lr = backbone_lr * scale
        else:
            lr = head_lr

        key = (is_backbone, round(lr, 12), wd)
        param_groups.setdefault(key, []).append(param)

    grouped_list = [
        {
            "params": params,
            "lr": lr,
            "weight_decay": wd,
            "backbone_group": bool(is_backbone),
        }
        for (is_backbone, lr, wd), params in sorted(
            param_groups.items(),
            key=lambda kv: (0 if kv[0][0] else 1, kv[0][1], kv[0][2]),
        )
    ]
    return AdamW(grouped_list, betas=(0.9, 0.999), eps=1e-8)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.02,
    backbone_freeze_steps: int = 0,
) -> LambdaLR:
    """
    Two-stage Probe-Then-Adapt warmup + cosine annealing scheduler:
      - Head param groups warm up linearly from step 0 and cosine-decay to min_lr_ratio.
      - Backbone / LoRA param groups (tagged `backbone_group=True`) stay at 0.0 LR
        during `backbone_freeze_steps` (Epoch 1 head-only probe), then warm up linearly
        over `warmup_ratio` (10%) of remaining steps before cosine decay.
    """
    total_steps = max(1, int(total_steps))
    freeze_steps = max(0, min(int(backbone_freeze_steps), total_steps - 1))
    head_warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

    def _head_lambda(current_step: int) -> float:
        if current_step < head_warmup_steps:
            return float(current_step + 1) / float(head_warmup_steps)
        progress = float(current_step - head_warmup_steps) / float(
            max(1, total_steps - head_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    rem_steps = max(1, total_steps - freeze_steps)
    bb_warmup_steps = max(1, int(round(rem_steps * warmup_ratio)))

    def _backbone_lambda(current_step: int) -> float:
        if current_step < freeze_steps:
            return 0.0
        bb_step = current_step - freeze_steps
        if bb_step < bb_warmup_steps:
            return float(bb_step + 1) / float(bb_warmup_steps)
        progress = float(bb_step - bb_warmup_steps) / float(
            max(1, rem_steps - bb_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    if freeze_steps > 0:
        lr_lambdas = [
            _backbone_lambda if pg.get("backbone_group", False) else _head_lambda
            for pg in optimizer.param_groups
        ]
        return LambdaLR(optimizer, lr_lambda=lr_lambdas)

    return LambdaLR(optimizer, lr_lambda=_head_lambda)


def compute_domain_schedule(
    progress: float,
    warmup_frac: float = 0.15,
    max_lambda: float = 0.5,
) -> float:
    """
    Smooth DANN/V-REx ramp-up schedule: stays near 0 during initial morphology
    learning, then increases smoothly along a logistic curve up to max_lambda.
    """
    p = min(max(float(progress), 0.0), 1.0)
    if p <= warmup_frac:
        return 0.0
    adj_p = (p - warmup_frac) / max(1e-6, 1.0 - warmup_frac)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-5.0 * adj_p)) - 1.0))


class ModelEMA:
    """
    Exponential Moving Average of model parameters for smoother decision
    boundaries and superior probability calibration on unseen medical centers.
    """

    def __init__(self, model: nn.Module, decay: float = 0.996) -> None:
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: Optional[int] = None) -> None:
        eff_decay = self.decay
        if step is not None:
            eff_decay = min(self.decay, (1.0 + step) / (10.0 + step))

        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        for k, v in model_params.items():
            if not v.requires_grad:
                continue
            ema_params[k].mul_(eff_decay).add_(v.detach(), alpha=1.0 - eff_decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for k, v in model_buffers.items():
            ema_buffers[k].copy_(v.detach())


def build_model_and_optimization(
    backbone_name: str = "owkin/phikon-v2",
    pretrained: bool = True,
    img_size: int = 256,
    num_centers: int = 2,
    use_od_stream: bool = True,
    backbone_lr: float = 2e-4,
    head_lr: float = 3e-4,
    weight_decay: float = 0.03,
    layer_decay: float = 0.95,
    total_steps: int = 2000,
    warmup_ratio: float = 0.10,
    ema_decay: float = 0.992,
    grad_checkpointing: bool = False,
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    backbone_freeze_steps: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Instantiate the complete model (with optional ViT LoRA adaptation and
    DualPrototypeCosineHead), metric-aligned loss criterion, LLRD AdamW optimizer,
    two-stage Probe-Then-Adapt cosine scheduler, and ModelEMA wrapper.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DualStreamStainViT(
        backbone_name=backbone_name,
        pretrained=pretrained,
        img_size=img_size,
        od_dim=256,
        fusion_dim=512,
        drop_path_rate=0.30,
        head_dropout=0.30,
        num_centers=num_centers,
        use_od_stream=use_od_stream,
        grad_checkpointing=grad_checkpointing,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    ).to(device)

    criterion = DomainGeneralizingHistopathLoss(
        label_smoothing=0.03,
        focal_gamma=1.0,
        sba_weight=0.30,
        parity_weight=0.25,
        aux_od_weight=0.15,
        vrex_weight=0.25,
        cdan_weight=0.0,
        logit_cons_weight=0.0,
        emb_cons_weight=0.0,
    )

    optimizer = build_optimizer_with_llrd(
        model=model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
    )

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=0.02,
        backbone_freeze_steps=backbone_freeze_steps,
    )

    ema = ModelEMA(model, decay=ema_decay)

    return {
        "model": model,
        "ema": ema,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "provenance": model.get_model_provenance(),
        "device": device,
    }


# =============================================================================
# STEP 3: TRAINING, OFFICIAL VALIDATION, POST-SELECTION REFIT & SUBMISSION
# =============================================================================
def load_official_binary_metrics(input_dir: Path = INPUT_DIR) -> Callable:
    """Load the official competition metric function directly from ./input/metric.py."""
    metric_path = Path(input_dir) / "metric.py"
    if not metric_path.exists():
        raise FileNotFoundError(f"Official metric file not found at {metric_path}")
    spec = importlib.util.spec_from_file_location("official_metric", str(metric_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec for {metric_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "binary_metrics"):
        raise AttributeError("official metric.py does not expose binary_metrics")
    return module.binary_metrics


def detect_runtime_config() -> Dict[str, Any]:
    """
    Configure process-local device (`cuda:0` or CPU), AMP precision, and VRAM-safe
    batch / gradient-accumulation settings maintaining effective batch size = 64.
    Never modifies CUDA_VISIBLE_DEVICES or assumes multiple GPUs.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if use_cuda:
        props = torch.cuda.get_device_properties(device)
        vram_gb = float(props.total_memory) / (1024.0**3)
        bf16_ok = bool(torch.cuda.is_bf16_supported())
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16

        if vram_gb >= 45.0:
            batch_size, grad_accum = 64, 1
            eval_batch_size, tta_eval_batch_size = 64, 32
            extract_batch_size, probe_batch_size = 128, 128
            grad_checkpointing = False
        elif vram_gb >= 22.0:
            batch_size, grad_accum = 32, 2
            eval_batch_size, tta_eval_batch_size = 64, 8
            extract_batch_size, probe_batch_size = 64, 64
            grad_checkpointing = True
        elif vram_gb >= 12.0:
            batch_size, grad_accum = 16, 4
            eval_batch_size, tta_eval_batch_size = 32, 4
            extract_batch_size, probe_batch_size = 32, 32
            grad_checkpointing = True
        else:
            batch_size, grad_accum = 8, 8
            eval_batch_size, tta_eval_batch_size = 16, 2
            extract_batch_size, probe_batch_size = 16, 16
            grad_checkpointing = True
        gpu_name = str(props.name)
    else:
        vram_gb = 0.0
        amp_dtype = torch.float32
        batch_size, grad_accum = 16, 4
        eval_batch_size, tta_eval_batch_size = 32, 4
        extract_batch_size, probe_batch_size = 16, 16
        grad_checkpointing = False
        gpu_name = "CPU"

    return {
        "device": device,
        "use_amp": use_cuda,
        "amp_dtype": amp_dtype,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "eval_batch_size": eval_batch_size,
        "tta_eval_batch_size": tta_eval_batch_size,
        "extract_batch_size": extract_batch_size,
        "probe_batch_size": probe_batch_size,
        "grad_checkpointing": grad_checkpointing,
        "vram_gb": round(vram_gb, 1),
        "gpu_name": gpu_name,
    }


def release_memory() -> None:
    """Cleanly release CPU/GPU memory between pipeline stages."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prob_to_logit(probs: np.ndarray) -> np.ndarray:
    """Numerically stable logit transform for calibrated probability ensembling."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def logit_to_prob(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid transform clamped strictly inside (0, 1)."""
    z = np.clip(np.asarray(logits, dtype=np.float64), -20.0, 20.0)
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-6, 1.0 - 1e-6)


def train_one_epoch(
    model: DualStreamStainViT,
    ema: ModelEMA,
    criterion: DomainGeneralizingHistopathLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    train_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_accum: int,
    global_step: int,
    total_steps: int,
    max_grad_norm: float = 1.0,
    domain_warmup_frac: float = 0.15,
    domain_max_lambda: float = 0.5,
) -> Tuple[Dict[str, float], int]:
    """
    Execute one fast single-view training epoch with mixed precision, float32
    metric-aligned loss stability (Focal-BCE + Soft Balanced-Accuracy + Recall
    Parity + Aux OD + warmup-scheduled V-REx), gradient accumulation, gradient
    clipping, and ModelEMA updates without EMA teacher double-forward overhead.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(train_loader)
    if num_batches == 0:
        raise RuntimeError("train_loader yielded 0 batches.")

    totals = {
        "loss": 0.0,
        "cls": 0.0,
        "sba": 0.0,
        "parity": 0.0,
        "aux": 0.0,
        "vrex": 0.0,
    }
    n_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        od = batch["od"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        center_idx = (
            batch["center_idx"].to(device, non_blocking=True)
            if "center_idx" in batch
            else None
        )

        window_start = (batch_idx // grad_accum) * grad_accum
        window_size = min(grad_accum, num_batches - window_start)
        is_step_boundary = ((batch_idx + 1) % grad_accum == 0) or (
            batch_idx + 1 == num_batches
        )

        progress = float(global_step) / float(max(1, total_steps))
        domain_scale = compute_domain_schedule(
            progress, warmup_frac=domain_warmup_frac, max_lambda=domain_max_lambda
        )
        use_domain = (
            getattr(criterion, "cdan_weight", 0.0) > 0.0
            and domain_scale > 0.0
            and center_idx is not None
        )

        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            outputs = model(
                rgb,
                od,
                return_domain=use_domain,
                grl_lambda=domain_scale,
            )

        outputs_fp32 = {
            k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()
        }
        loss_dict = criterion(
            outputs_fp32,
            labels,
            center_idx=center_idx,
            domain_scale=domain_scale,
        )

        scaled_loss = loss_dict["loss"] / float(window_size)
        scaler.scale(scaled_loss).backward()

        bsz = int(labels.shape[0])
        totals["loss"] += float(loss_dict["loss"].detach().item()) * bsz
        totals["cls"] += float(loss_dict["cls_loss"].item()) * bsz
        totals["sba"] += float(loss_dict["sba_loss"].item()) * bsz
        totals["parity"] += float(loss_dict["parity_loss"].item()) * bsz
        totals["aux"] += float(loss_dict["aux_loss"].item()) * bsz
        totals["vrex"] += float(loss_dict["vrex_loss"].item()) * bsz
        n_samples += bsz

        if is_step_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            prev_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step_executed = (not scaler.is_enabled()) or (
                scaler.get_scale() >= prev_scale
            )
            if step_executed:
                scheduler.step()
                global_step += 1
                ema.update(model, step=global_step)

    denom = max(1, n_samples)
    metrics = {k: v / denom for k, v in totals.items()}
    metrics["lr_head"] = float(optimizer.param_groups[-1]["lr"])
    return metrics, global_step


@torch.no_grad()
def predict_dataset(
    model: DualStreamStainViT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
    """
    100% pixel-only inference function shared identically across validation and test.
    Supports both single-view (4D) and multi-view D4 TTA (5D) DataLoader tensors via
    DualStreamStainViT.predict_proba (logit-space view averaging -> sigmoid).
    """
    model.eval()
    sample_ids: List[str] = []
    prob_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    has_labels = False

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        od = batch["od"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            probs = model.predict_proba(rgb, od)

        sample_ids.extend(batch["sample_id"])
        prob_chunks.append(probs.float().cpu().numpy())

        if "label" in batch:
            has_labels = True
            label_chunks.append(batch["label"].numpy())

    probs_arr = np.concatenate(prob_chunks, axis=0).astype(np.float64)
    if not np.isfinite(probs_arr).all():
        raise ValueError("Non-finite probabilities detected during model inference.")
    probs_arr = np.clip(probs_arr, 1e-6, 1.0 - 1e-6)

    labels_arr = (
        np.concatenate(label_chunks, axis=0).astype(np.int64) if has_labels else None
    )
    return sample_ids, probs_arr, labels_arr


@torch.no_grad()
def predict_dataset_dual(
    model: DualStreamStainViT,
    ema_model: DualStreamStainViT,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    """
    Single-pass dual evaluation for raw and EMA models on the exact same GPU
    batches, avoiding redundant validation PNG decoding and OD deconvolution
    with exact mathematical equivalence to two separate predict_dataset calls.
    """
    model.eval()
    ema_model.eval()
    sample_ids: List[str] = []
    raw_prob_chunks: List[np.ndarray] = []
    ema_prob_chunks: List[np.ndarray] = []

    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        od = batch["od"].to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            raw_probs = model.predict_proba(rgb, od)
            ema_probs = ema_model.predict_proba(rgb, od)

        sample_ids.extend(batch["sample_id"])
        raw_prob_chunks.append(raw_probs.float().cpu().numpy())
        ema_prob_chunks.append(ema_probs.float().cpu().numpy())

    raw_probs_arr = np.concatenate(raw_prob_chunks, axis=0).astype(np.float64)
    ema_probs_arr = np.concatenate(ema_prob_chunks, axis=0).astype(np.float64)
    if not np.isfinite(raw_probs_arr).all() or not np.isfinite(ema_probs_arr).all():
        raise ValueError(
            "Non-finite probabilities detected during dual model inference."
        )
    raw_probs_arr = np.clip(raw_probs_arr, 1e-6, 1.0 - 1e-6)
    ema_probs_arr = np.clip(ema_probs_arr, 1e-6, 1.0 - 1e-6)

    return (sample_ids, raw_probs_arr), (list(sample_ids), ema_probs_arr)


def evaluate_validation_predictions(
    val_df: pd.DataFrame,
    val_ids: List[str],
    val_probs: np.ndarray,
    binary_metrics_fn: Callable,
) -> Dict[str, Any]:
    """
    Compute exact official validation metrics plus auditable per-center and
    calibration diagnostics at the fixed 0.5 threshold.
    """
    expected_ids = val_df["sample_id"].astype(str).tolist()
    if val_ids != expected_ids:
        raise ValueError(
            "Validation sample_id alignment mismatch between DataLoader and val_df."
        )

    labels = val_df["label"].astype(int).values
    overall = binary_metrics_fn(labels, val_probs)

    preds_bin = (val_probs >= 0.5).astype(int)
    pos_mask = labels == 1
    neg_mask = labels == 0
    tumor_recall = float((preds_bin[pos_mask] == 1).mean())
    normal_recall = float((preds_bin[neg_mask] == 0).mean())

    per_center: Dict[str, Dict[str, float]] = {}
    for cid, grp in val_df.groupby("center_id", sort=True):
        idx = grp.index.values
        c_metrics = binary_metrics_fn(labels[idx], val_probs[idx])
        per_center[str(cid)] = {
            "balanced_accuracy": float(c_metrics["balanced_accuracy"]),
            "auroc": float(c_metrics["auroc"]),
            "n_samples": int(c_metrics["n_samples"]),
        }

    return {
        "balanced_accuracy": float(overall["balanced_accuracy"]),
        "auroc": float(overall["auroc"]),
        "accuracy": float(overall["accuracy"]),
        "n_samples": int(overall["n_samples"]),
        "tumor_recall": tumor_recall,
        "normal_recall": normal_recall,
        "pred_pos_rate": float(preds_bin.mean()),
        "mean_prob": float(val_probs.mean()),
        "per_center": per_center,
    }


def _selection_key(metrics: Dict[str, float]) -> Tuple[float, float]:
    """Primary selection criterion: official balanced_accuracy at 0.5, tie-break AUROC."""
    return (round(float(metrics["balanced_accuracy"]), 6), float(metrics["auroc"]))


def save_checkpoint(
    model: DualStreamStainViT,
    path: Path,
    metadata: Dict[str, Any],
) -> None:
    """Save CPU state_dict and auditable metadata cleanly to ./working/."""
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": cpu_state, "metadata": metadata}, path)


def load_checkpoint(
    model: DualStreamStainViT,
    path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Load saved checkpoint weights back into model safely."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return ckpt.get("metadata", {})


def write_and_verify_submission(
    sample_submission_df: pd.DataFrame,
    test_ids: List[str],
    test_probs: np.ndarray,
    submission_path: Path = SUBMISSION_DIR / "submission.csv",
) -> pd.DataFrame:
    """
    Write final predictions to ./submission/submission.csv and verify exact schema,
    ID alignment, finiteness, [0, 1] bounds, and non-trivial learned variance.
    """
    expected_ids = sample_submission_df["sample_id"].astype(str).tolist()
    if len(test_ids) != len(expected_ids) or test_ids != expected_ids:
        raise ValueError(
            "Test prediction sample_ids do not match sample_submission.csv order/length."
        )

    probs = np.asarray(test_probs, dtype=np.float64)
    if probs.ndim != 1 or len(probs) != len(expected_ids):
        raise ValueError(f"Invalid test probability shape: {probs.shape}")
    if not np.isfinite(probs).all():
        raise ValueError("Test predictions contain NaN or Inf values.")
    if ((probs < 0.0) | (probs > 1.0)).any():
        raise ValueError("Test predictions fall outside [0, 1].")
    if float(np.std(probs)) < 1e-4:
        raise ValueError("Test predictions appear constant; expected learned variance.")

    sub_out = pd.DataFrame(
        {
            "sample_id": expected_ids,
            "tumor_probability": np.clip(probs, 1e-6, 1.0 - 1e-6),
        }
    )
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    sub_out.to_csv(submission_path, index=False)

    check_df = pd.read_csv(submission_path)
    assert list(check_df.columns) == ["sample_id", "tumor_probability"]
    assert len(check_df) == len(sample_submission_df)
    assert check_df["sample_id"].equals(sample_submission_df["sample_id"])
    assert np.isfinite(check_df["tumor_probability"].values).all()
    assert (
        (check_df["tumor_probability"] >= 0.0) & (check_df["tumor_probability"] <= 1.0)
    ).all()
    return check_df


def _compute_slide_tempered_weights(
    df: pd.DataFrame, slide_power: float = 0.5
) -> np.ndarray:
    """
    Compute slide-tempered, (center_id, label)-balanced sample weights with
    mean 1.0 for tabular/linear/GBDT downstream paradigms.
    """
    sub = df.reset_index(drop=True)
    slide_counts = (
        sub.groupby("slide_id")["sample_id"]
        .transform("count")
        .values.astype(np.float64)
    )
    raw_w = 1.0 / np.power(np.maximum(slide_counts, 1.0), slide_power)
    strata = sub["center_id"].astype(str) + "__" + sub["label"].astype(str)
    weights = np.zeros(len(sub), dtype=np.float64)
    for s in strata.unique():
        mask = (strata == s).values
        sw = raw_w[mask]
        weights[mask] = sw / max(float(sw.sum()), 1e-12)
    weights = weights / max(float(weights.mean()), 1e-12)
    return weights


def _make_extraction_loader(
    df: pd.DataFrame,
    stain_bank: StainReferenceBank,
    batch_size: int,
    canonical_strength: float,
    num_perturbed_views: int = 0,
    extract_od_features: bool = False,
    seed: int = 42,
) -> DataLoader:
    cpu_cnt = os.cpu_count() or 2
    num_workers = max(2, min(8, cpu_cnt))
    ds = MultiViewExtractionDataset(
        df=df,
        stain_bank=stain_bank,
        input_dir=INPUT_DIR,
        canonical_strength=canonical_strength,
        num_perturbed_views=num_perturbed_views,
        extract_od_features=extract_od_features,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=False,
    )


def _extract_single_split_with_pfm(
    extractor: FrozenPFMExtractor,
    loader: DataLoader,
    d4_views: int = 4,
    extract_raw: bool = True,
    extract_canon: bool = True,
    extract_pert: bool = False,
    extract_od: bool = False,
) -> Dict[str, Any]:
    sample_ids: List[str] = []
    raw_chunks: List[np.ndarray] = []
    canon_chunks: List[np.ndarray] = []
    pert_chunks: List[np.ndarray] = []
    od_chunks: List[np.ndarray] = []

    for batch in loader:
        sample_ids.extend(batch["sample_id"])
        if extract_raw:
            emb_raw = extractor.extract_d4_batch(batch["rgb_raw"], d4_views=d4_views)
            raw_chunks.append(emb_raw.cpu().numpy().astype(np.float32))
        if extract_canon:
            emb_canon = extractor.extract_d4_batch(
                batch["rgb_canon"], d4_views=d4_views
            )
            canon_chunks.append(emb_canon.cpu().numpy().astype(np.float32))
        if extract_pert and "rgb_pert" in batch:
            pert_u8 = batch["rgb_pert"]  # (B, V, 3, H, W)
            B, V, C, H, W = pert_u8.shape
            pert_emb = extractor.extract_d4_batch(
                pert_u8.reshape(B * V, C, H, W), d4_views=1
            )
            pert_chunks.append(
                pert_emb.reshape(B, V, -1).cpu().numpy().astype(np.float16)
            )
        if extract_od and "od_feats" in batch:
            od_chunks.append(batch["od_feats"].numpy().astype(np.float32))

    out: Dict[str, Any] = {"sample_id": sample_ids}
    if extract_raw:
        out["raw"] = np.concatenate(raw_chunks, axis=0)
    if extract_canon:
        out["canon"] = np.concatenate(canon_chunks, axis=0)
    if extract_pert and pert_chunks:
        out["pert"] = np.concatenate(pert_chunks, axis=0)
    if extract_od and od_chunks:
        out["od"] = np.concatenate(od_chunks, axis=0)
    return out


def _extract_and_cache_pfm_bank(
    pfm_specs: List[Tuple[str, str]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stain_bank: StainReferenceBank,
    rt: Dict[str, Any],
    canonical_strength: float = 0.50,
    d4_views: int = 4,
    num_perturbed_views: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Extract and cache multi-layer D4 embeddings for all 3 frozen Pathology
    Foundation Models (raw + soft-canonical on train/val/test, plus 4 bounded
    stain-perturbed views on train and 46-dim Beer-Lambert OD descriptors).
    """
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]
    extract_bs: int = int(rt["extract_batch_size"])

    pfm_features: List[Dict[str, Any]] = []
    pfm_provenances: List[Dict[str, Any]] = []
    loaded_backbones: List[str] = []

    od_train: Optional[np.ndarray] = None
    od_val: Optional[np.ndarray] = None
    od_test: Optional[np.ndarray] = None

    for p_idx, (pfm_tag, backbone_name) in enumerate(pfm_specs):
        t0 = time.time()
        extractor = FrozenPFMExtractor(
            backbone_name=backbone_name,
            target_size=256,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            exclude_backbones=loaded_backbones,
        )
        prov = dict(extractor.provenance)
        prov["pfm_tag"] = pfm_tag
        loaded_backbones.append(prov["backbone_name"])

        print(
            f"[PFM {p_idx+1}/{len(pfm_specs)} {pfm_tag}] Loaded {prov['backbone_name']} | "
            f"in={prov['input_size']}px patch={prov['patch_size']} "
            f"prefix={prov['num_prefix_tokens']} dim={prov['out_dim']} "
            f"taps={prov['tap_indices']} "
            f"norm=({prov['input_mean']}, {prov['input_std']})",
            flush=True,
        )

        need_od = p_idx == 0
        tr_loader = _make_extraction_loader(
            train_df,
            stain_bank=stain_bank,
            batch_size=extract_bs,
            canonical_strength=canonical_strength,
            num_perturbed_views=num_perturbed_views,
            extract_od_features=need_od,
            seed=seed,
        )
        va_loader = _make_extraction_loader(
            val_df,
            stain_bank=stain_bank,
            batch_size=extract_bs,
            canonical_strength=canonical_strength,
            num_perturbed_views=0,
            extract_od_features=need_od,
            seed=seed,
        )
        te_loader = _make_extraction_loader(
            test_df,
            stain_bank=stain_bank,
            batch_size=extract_bs,
            canonical_strength=canonical_strength,
            num_perturbed_views=0,
            extract_od_features=need_od,
            seed=seed,
        )

        tr_out = _extract_single_split_with_pfm(
            extractor,
            tr_loader,
            d4_views=d4_views,
            extract_raw=True,
            extract_canon=True,
            extract_pert=True,
            extract_od=need_od,
        )
        va_out = _extract_single_split_with_pfm(
            extractor,
            va_loader,
            d4_views=d4_views,
            extract_raw=True,
            extract_canon=True,
            extract_pert=False,
            extract_od=need_od,
        )
        te_out = _extract_single_split_with_pfm(
            extractor,
            te_loader,
            d4_views=d4_views,
            extract_raw=True,
            extract_canon=True,
            extract_pert=False,
            extract_od=need_od,
        )

        assert tr_out["sample_id"] == train_df["sample_id"].astype(str).tolist()
        assert va_out["sample_id"] == val_df["sample_id"].astype(str).tolist()
        assert te_out["sample_id"] == test_df["sample_id"].astype(str).tolist()

        if need_od:
            od_train = tr_out["od"]
            od_val = va_out["od"]
            od_test = te_out["od"]
            np.savez_compressed(
                WORKING_DIR / "od_morphological_features.npz",
                train=od_train,
                val=od_val,
                test=od_test,
            )

        pert_mean_tr = tr_out["pert"].astype(np.float32).mean(axis=1)
        # Re-normalize perturbed mean block-wise to match exact norm scale of raw/canon
        D_sub = pert_mean_tr.shape[1] // 4
        pm_blocks = pert_mean_tr.reshape(len(train_df), 4, D_sub)
        pm_norms = np.linalg.norm(pm_blocks, axis=-1, keepdims=True)
        pert_mean_tr = (
            (pm_blocks / np.maximum(pm_norms, 1e-6)) * np.float32(0.5)
        ).reshape(len(train_df), -1)

        cache_path = WORKING_DIR / f"pfm_cache_{pfm_tag}.npz"
        np.savez(
            cache_path,
            train_raw=tr_out["raw"].astype(np.float16),
            train_canon=tr_out["canon"].astype(np.float16),
            train_pert=tr_out["pert"],
            val_raw=va_out["raw"].astype(np.float16),
            val_canon=va_out["canon"].astype(np.float16),
            test_raw=te_out["raw"].astype(np.float16),
            test_canon=te_out["canon"].astype(np.float16),
        )

        pfm_features.append(
            {
                "pfm_tag": pfm_tag,
                "backbone_name": prov["backbone_name"],
                "dim": int(tr_out["raw"].shape[1]),
                "train_raw": tr_out["raw"],
                "train_canon": tr_out["canon"],
                "train_pert": tr_out["pert"],  # (N_tr, V, D) float16
                "train_pert_mean": pert_mean_tr,
                "val_raw": va_out["raw"],
                "val_canon": va_out["canon"],
                "test_raw": te_out["raw"],
                "test_canon": te_out["canon"],
                "cache_path": str(cache_path),
            }
        )
        pfm_provenances.append(prov)

        elapsed = time.time() - t0
        print(
            f"[PFM {p_idx+1}/{len(pfm_specs)} {pfm_tag}] Extracted & cached -> "
            f"{cache_path.name} (D={tr_out['raw'].shape[1]}, d4={d4_views}, "
            f"pert={num_perturbed_views}) in {elapsed:.1f}s",
            flush=True,
        )

        extractor.close()
        del extractor, tr_loader, va_loader, te_loader, tr_out, va_out, te_out
        release_memory()

    assert od_train is not None and od_val is not None and od_test is not None
    return {
        "pfms": pfm_features,
        "provenances": pfm_provenances,
        "od_train": od_train,
        "od_val": od_val,
        "od_test": od_test,
    }


def _extract_refit_val_pert_and_canon(
    pfm_bank: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    refit_bank: StainReferenceBank,
    rt: Dict[str, Any],
    canonical_strength: float,
    d4_views: int,
    num_perturbed_views: int,
    need_refit_canon: bool,
    refit_seed: int,
) -> Dict[str, Any]:
    """
    For Stage 2 Train+Val Refit: extract 4 stain-perturbed views for val_df
    (and optionally refit soft-canonical views on train+val+test if alpha_canon > 0)
    using the Train+Val refit StainReferenceBank, reusing cached raw embeddings.
    """
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]
    extract_bs: int = int(rt["extract_batch_size"])

    full_pfms: List[Dict[str, Any]] = []
    for pfm in pfm_bank["pfms"]:
        t0 = time.time()
        extractor = FrozenPFMExtractor(
            backbone_name=pfm["backbone_name"],
            target_size=256,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )

        va_pert_loader = _make_extraction_loader(
            val_df,
            stain_bank=refit_bank,
            batch_size=extract_bs,
            canonical_strength=canonical_strength,
            num_perturbed_views=num_perturbed_views,
            extract_od_features=False,
            seed=refit_seed,
        )
        va_out = _extract_single_split_with_pfm(
            extractor,
            va_pert_loader,
            d4_views=d4_views,
            extract_raw=False,
            extract_canon=need_refit_canon,
            extract_pert=True,
            extract_od=False,
        )

        val_pert_fp16 = va_out["pert"]
        val_pert_mean = val_pert_fp16.astype(np.float32).mean(axis=1)
        D_sub = val_pert_mean.shape[1] // 4
        vm_b = val_pert_mean.reshape(len(val_df), 4, D_sub)
        val_pert_mean = (
            (vm_b / np.maximum(np.linalg.norm(vm_b, axis=-1, keepdims=True), 1e-6))
            * np.float32(0.5)
        ).reshape(len(val_df), -1)

        if need_refit_canon:
            tr_c_loader = _make_extraction_loader(
                train_df,
                stain_bank=refit_bank,
                batch_size=extract_bs,
                canonical_strength=canonical_strength,
                num_perturbed_views=0,
                extract_od_features=False,
                seed=refit_seed,
            )
            te_c_loader = _make_extraction_loader(
                test_df,
                stain_bank=refit_bank,
                batch_size=extract_bs,
                canonical_strength=canonical_strength,
                num_perturbed_views=0,
                extract_od_features=False,
                seed=refit_seed,
            )
            tr_c_out = _extract_single_split_with_pfm(
                extractor,
                tr_c_loader,
                d4_views=d4_views,
                extract_raw=False,
                extract_canon=True,
                extract_pert=False,
                extract_od=False,
            )
            te_c_out = _extract_single_split_with_pfm(
                extractor,
                te_c_loader,
                d4_views=d4_views,
                extract_raw=False,
                extract_canon=True,
                extract_pert=False,
                extract_od=False,
            )
            full_canon = np.concatenate([tr_c_out["canon"], va_out["canon"]], axis=0)
            test_canon = te_c_out["canon"]
            del tr_c_loader, te_c_loader, tr_c_out, te_c_out
        else:
            full_canon = np.concatenate([pfm["train_canon"], pfm["val_canon"]], axis=0)
            test_canon = pfm["test_canon"]

        full_pfms.append(
            {
                "pfm_tag": pfm["pfm_tag"],
                "backbone_name": pfm["backbone_name"],
                "dim": pfm["dim"],
                "full_raw": np.concatenate([pfm["train_raw"], pfm["val_raw"]], axis=0),
                "full_canon": full_canon,
                "full_pert": np.concatenate([pfm["train_pert"], val_pert_fp16], axis=0),
                "full_pert_mean": np.concatenate(
                    [pfm["train_pert_mean"], val_pert_mean], axis=0
                ),
                "test_raw": pfm["test_raw"],
                "test_canon": test_canon,
            }
        )

        print(
            f"[Refit PFM {pfm['pfm_tag']}] Extracted val perturbed views "
            f"(refit_canon={need_refit_canon}) in {time.time() - t0:.1f}s",
            flush=True,
        )

        extractor.close()
        del extractor, va_pert_loader, va_out
        release_memory()

    return {
        "pfms": full_pfms,
        "od_full": np.concatenate([pfm_bank["od_train"], pfm_bank["od_val"]], axis=0),
        "od_test": pfm_bank["od_test"],
    }


# -----------------------------------------------------------------------------
# Downstream Paradigm 1: Strictly Frozen-Backbone Spatial-MIL Probe (Phikon-v2)
# -----------------------------------------------------------------------------
class _HeadOnlyEMA:
    """
    Lightweight EMA tracking strictly the trainable head/MIL/OD/fusion/cosine
    parameters of a frozen-backbone DualStreamStainViT (zero backbone duplication).
    """

    def __init__(self, model: DualStreamStainViT, decay: float = 0.992) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if not k.startswith("rgb_backbone."):
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: DualStreamStainViT, step: int) -> None:
        eff_decay = min(self.decay, (1.0 + step) / (10.0 + step))
        msd = model.state_dict()
        for k, s_val in self.shadow.items():
            m_val = msd[k].detach()
            if s_val.is_floating_point():
                s_val.mul_(eff_decay).add_(m_val, alpha=1.0 - eff_decay)
            else:
                s_val.copy_(m_val)

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}


def _get_head_state_cpu(model: DualStreamStainViT) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if not k.startswith("rgb_backbone.")
    }


def _load_head_state(
    model: DualStreamStainViT, head_state: Dict[str, torch.Tensor]
) -> None:
    model.load_state_dict(head_state, strict=False)


def _reinit_spatial_mil_head(
    model: DualStreamStainViT, seed: int, device: torch.device
) -> None:
    """Re-initialize all non-backbone Spatial-MIL/OD/Fusion/Cosine head modules cleanly."""
    seed_everything(seed)
    rgb_dim = model.rgb_dim
    od_dim = 256
    fusion_dim = 512

    if model.tap_indices:
        model.layer_norms = nn.ModuleList(
            [nn.LayerNorm(rgb_dim) for _ in model.tap_indices]
        ).to(device)
        model.layer_weights = nn.Parameter(
            torch.zeros(len(model.tap_indices), device=device)
        )

    model.rgb_spatial_norm = nn.LayerNorm(rgb_dim).to(device)
    model.rgb_mil_pool = GatedMILAttentionPool(
        embed_dim=rgb_dim, hidden_dim=256, dropout=0.20
    ).to(device)
    model.rgb_head_norm = nn.LayerNorm(rgb_dim * 2).to(device)
    model.od_branch = BeerLambertODBranch(out_dim=od_dim, dropout=0.15).to(device)
    model.fusion = CrossModalGatedFusion(
        rgb_dim=rgb_dim * 2, od_dim=od_dim, fusion_dim=fusion_dim, dropout=0.25
    ).to(device)
    model.classifier = DualPrototypeCosineHead(
        in_dim=fusion_dim,
        num_samples=5,
        dropout_rate=0.30,
        init_scale=10.0,
        min_scale=2.0,
        max_scale=25.0,
    ).to(device)
    model.domain_discriminator = ConditionalDomainDiscriminator(
        embed_dim=fusion_dim,
        num_centers=max(2, model.num_centers),
        hidden_dim=256,
        dropout=0.25,
    ).to(device)
    model.set_backbone_trainable(False)


@torch.no_grad()
def _predict_spatial_mil_dual_stain(
    host_model: DualStreamStainViT,
    head_states: List[Dict[str, torch.Tensor]],
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Shared-backbone single-pass dual-stain D4 TTA inference across 1 or K fold
    Spatial-MIL heads. Runs the frozen Phikon-v2 backbone ONCE per batch and
    evaluates all fold heads on the exact same frozen tokens (returns mean logits
    across head_states for raw and soft-canonical views).
    """
    host_model.eval()
    n_heads = len(head_states)
    sample_ids: List[str] = []
    raw_chunks: List[np.ndarray] = []
    canon_chunks: List[np.ndarray] = []

    gpu_head_states = [
        {k: v.to(device, non_blocking=True) for k, v in hs.items()}
        for hs in head_states
    ]

    for batch in loader:
        sample_ids.extend(batch["sample_id"])
        rgb = batch["rgb"].to(device, non_blocking=True)
        od = batch["od"].to(device, non_blocking=True)
        has_canon = "rgb_canon" in batch and "od_canon" in batch

        if rgb.ndim == 5:
            B, V, C, H, W = rgb.shape
            rgb_flat = rgb.reshape(B * V, C, H, W)
            od_flat = od.reshape(B * V, C, H, W)
        else:
            B, V = rgb.shape[0], 1
            rgb_flat, od_flat = rgb, od

        with torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            feats, tapped, stage3 = host_model._forward_backbone_raw_tokens(rgb_flat)
            raw_sum = torch.zeros(B, device=device, dtype=torch.float32)
            for hs in gpu_head_states:
                _load_head_state(host_model, hs)
                lg = host_model.forward_from_backbone_tokens(
                    feats, tapped, stage3, od_flat
                )
                raw_sum += lg.float().reshape(B, V).mean(dim=1)
            raw_chunks.append((raw_sum / float(n_heads)).cpu().numpy())

            if has_canon:
                rgb_c = batch["rgb_canon"].to(device, non_blocking=True)
                od_c = batch["od_canon"].to(device, non_blocking=True)
                if rgb_c.ndim == 5:
                    rgb_c_flat = rgb_c.reshape(B * V, C, H, W)
                    od_c_flat = od_c.reshape(B * V, C, H, W)
                else:
                    rgb_c_flat, od_c_flat = rgb_c, od_c
                fc, tc, s3c = host_model._forward_backbone_raw_tokens(rgb_c_flat)
                canon_sum = torch.zeros(B, device=device, dtype=torch.float32)
                for hs in gpu_head_states:
                    _load_head_state(host_model, hs)
                    lg_c = host_model.forward_from_backbone_tokens(
                        fc, tc, s3c, od_c_flat
                    )
                    canon_sum += lg_c.float().reshape(B, V).mean(dim=1)
                canon_chunks.append((canon_sum / float(n_heads)).cpu().numpy())

    raw_logits = np.concatenate(raw_chunks, axis=0).astype(np.float64)
    canon_logits = (
        np.concatenate(canon_chunks, axis=0).astype(np.float64)
        if canon_chunks
        else raw_logits.copy()
    )
    return sample_ids, raw_logits, canon_logits


def _make_dual_stain_eval_loader(
    df: pd.DataFrame,
    stain_bank: StainReferenceBank,
    img_size: int,
    batch_size: int,
    canonical_strength: float,
    tta_views: int = 4,
    include_canonical_view: bool = True,
    seed: int = 42,
) -> DataLoader:
    cpu_cnt = os.cpu_count() or 2
    num_workers = max(2, min(8, cpu_cnt))
    tf = HistopathologyEvalTransform(
        stain_bank=stain_bank,
        img_size=img_size,
        canonical_strength=canonical_strength,
        tta_views=tta_views,
        include_canonical_view=include_canonical_view,
    )
    ds = HistopathologyDataset(
        df=df,
        transform=tf,
        input_dir=INPUT_DIR,
        is_train=False,
        return_mode="dual",
        include_train_meta=False,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=False,
    )


def _train_single_spatial_mil_fold(
    host_model: DualStreamStainViT,
    tr_df: pd.DataFrame,
    va_df: Optional[pd.DataFrame],
    stain_bank: StainReferenceBank,
    rt: Dict[str, Any],
    img_size: int,
    canonical_strength: float,
    num_epochs: int,
    head_lr: float,
    weight_decay: float,
    ema_decay: float,
    seed: int,
    select_on_val: bool = True,
    deadline: Optional[float] = None,
    tag: str = "mil",
) -> Tuple[Dict[str, torch.Tensor], int, List[float]]:
    """
    Train one fold of the strictly frozen-backbone owkin/phikon-v2 Spatial-MIL
    Probe with live bounded stain augmentation, DualPrototypeCosineHead, and
    head-only EMA monitoring.
    """
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]
    probe_bs: int = int(rt["probe_batch_size"])
    cpu_cnt = os.cpu_count() or 2
    num_workers = max(2, min(8, cpu_cnt))

    _reinit_spatial_mil_head(host_model, seed=seed, device=device)

    train_tf = HistopathologyTrainTransform(
        stain_bank=stain_bank,
        img_size=img_size,
        canonical_strength=canonical_strength,
    )
    train_ds = HistopathologyDataset(
        df=tr_df,
        transform=train_tf,
        input_dir=INPUT_DIR,
        is_train=True,
        return_mode="dual",
        include_train_meta=True,
        seed=seed,
    )
    sampler = build_slide_tempered_balanced_sampler(tr_df, slide_power=0.5, seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=probe_bs,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    va_labels = None
    if select_on_val and va_df is not None and len(va_df) > 0:
        val_loader = _make_dual_stain_eval_loader(
            va_df,
            stain_bank=stain_bank,
            img_size=img_size,
            batch_size=rt["eval_batch_size"],
            canonical_strength=canonical_strength,
            tta_views=1,
            include_canonical_view=False,
            seed=seed,
        )
        va_labels = va_df["label"].astype(int).values

    criterion = DomainGeneralizingHistopathLoss(
        label_smoothing=0.03,
        focal_gamma=1.0,
        sba_weight=0.30,
        parity_weight=0.25,
        aux_od_weight=0.15,
        vrex_weight=0.25,
        cdan_weight=0.0,
    )
    optimizer = build_optimizer_with_llrd(
        model=host_model,
        backbone_lr=0.0,
        head_lr=head_lr,
        weight_decay=weight_decay,
        layer_decay=1.0,
    )
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=0.10,
        min_lr_ratio=0.02,
        backbone_freeze_steps=0,
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=(use_amp and amp_dtype == torch.float16)
    )
    head_ema = _HeadOnlyEMA(host_model, decay=ema_decay)

    best_key = (-1.0, -1.0)
    best_epoch = num_epochs
    best_head_state = head_ema.state_dict_cpu()
    global_step = 0
    ep_secs: List[float] = []

    for ep in range(1, num_epochs + 1):
        if deadline is not None and ep > 3 and time.time() > deadline:
            break
        t0 = time.time()
        host_model.train()
        loss_sum, n_seen = 0.0, 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            od = batch["od"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            center_idx = (
                batch["center_idx"].to(device, non_blocking=True)
                if "center_idx" in batch
                else None
            )

            progress = float(global_step) / float(max(1, total_steps))
            domain_scale = compute_domain_schedule(
                progress, warmup_frac=0.15, max_lambda=0.5
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=amp_dtype
            ):
                outputs = host_model(rgb, od, return_domain=False)

            out_fp32 = {
                k: (v.float() if torch.is_tensor(v) else v) for k, v in outputs.items()
            }
            loss_dict = criterion(
                out_fp32, labels, center_idx=center_idx, domain_scale=domain_scale
            )
            scaler.scale(loss_dict["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in host_model.parameters() if p.requires_grad], max_norm=1.0
            )
            prev_scale = scaler.get_scale() if scaler.is_enabled() else 1.0
            scaler.step(optimizer)
            scaler.update()
            if (not scaler.is_enabled()) or (scaler.get_scale() >= prev_scale):
                scheduler.step()
                global_step += 1
                head_ema.update(host_model, step=global_step)

            bsz = int(labels.shape[0])
            loss_sum += float(loss_dict["loss"].detach().item()) * bsz
            n_seen += bsz

        ep_sec = time.time() - t0
        ep_secs.append(ep_sec)

        if select_on_val and val_loader is not None and va_labels is not None:
            raw_sd = _get_head_state_cpu(host_model)
            ema_sd = head_ema.state_dict_cpu()
            _, lg_raw, _ = _predict_spatial_mil_dual_stain(
                host_model, [raw_sd], val_loader, device, use_amp, amp_dtype
            )
            _, lg_ema, _ = _predict_spatial_mil_dual_stain(
                host_model, [ema_sd], val_loader, device, use_amp, amp_dtype
            )
            _load_head_state(host_model, raw_sd)

            p_raw = logit_to_prob(lg_raw)
            p_ema = logit_to_prob(lg_ema)
            k_raw = (
                round(float(balanced_accuracy_score(va_labels, p_raw >= 0.5)), 6),
                float(roc_auc_score(va_labels, p_raw)),
            )
            k_ema = (
                round(float(balanced_accuracy_score(va_labels, p_ema >= 0.5)), 6),
                float(roc_auc_score(va_labels, p_ema)),
            )
            if k_ema >= k_raw:
                ep_key, ep_sd = k_ema, ema_sd
            else:
                ep_key, ep_sd = k_raw, raw_sd

            if ep_key > best_key:
                best_key = ep_key
                best_epoch = ep
                best_head_state = ep_sd
        else:
            best_epoch = ep
            best_head_state = head_ema.state_dict_cpu()

    print(
        f"[{tag}] Finished {len(ep_secs)}/{num_epochs} ep "
        f"(best_ep={best_epoch}, avg={np.mean(ep_secs):.1f}s/ep"
        + (f", fold_val_bal={best_key[0]:.4f}/auc={best_key[1]:.4f}" if select_on_val else "")
        + ")",
        flush=True,
    )

    del train_loader, val_loader, optimizer, scheduler, scaler, head_ema
    release_memory()
    return best_head_state, best_epoch, ep_secs


def _run_spatial_mil_cv(
    backbone_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    loco_fold_indices: List[Tuple[str, np.ndarray, np.ndarray]],
    n_folds: int,
    stain_bank: StainReferenceBank,
    rt: Dict[str, Any],
    img_size: int = 256,
    canonical_strength: float = 0.50,
    num_epochs: int = 6,
    tta_views: int = 4,
    head_lr: float = 4.0e-4,
    weight_decay: float = 0.03,
    ema_decay: float = 0.992,
    seed: int = 42,
    deadline: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Train strictly frozen-backbone owkin/phikon-v2 Spatial-MIL Probe across
    5-Fold Slide-Grouped CV and 2-Fold LOCO CV, plus shared-backbone dual-stain
    D4 TTA inference on validation.csv and test.csv.
    """
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]
    tta_bs: int = max(4, int(rt["tta_eval_batch_size"]))

    host_model = DualStreamStainViT(
        backbone_name=backbone_name,
        pretrained=True,
        img_size=img_size,
        od_dim=256,
        fusion_dim=512,
        drop_path_rate=0.0,
        head_dropout=0.30,
        num_centers=int(train_df["center_id"].nunique()),
        use_od_stream=True,
        grad_checkpointing=False,
        use_lora=False,
    ).to(device)
    host_model.set_backbone_trainable(False)
    prov = host_model.get_model_provenance()
    print(
        f"[spatial_mil] Host Frozen Backbone: {prov['backbone_name']} "
        f"(trainable={prov['trainable_params']/1e6:.2f}M/{prov['total_params']/1e6:.1f}M, "
        f"in={prov['vit_input_size']}px, BS={rt['probe_batch_size']}, ep={num_epochs})",
        flush=True,
    )

    n_tr = len(train_df)
    oof_raw = np.zeros(n_tr, dtype=np.float64)
    oof_canon = np.zeros(n_tr, dtype=np.float64)
    loco_raw = np.zeros(n_tr, dtype=np.float64)
    loco_canon = np.zeros(n_tr, dtype=np.float64)

    fold_head_states: List[Dict[str, torch.Tensor]] = []
    best_epochs: List[int] = []

    # 1. 5-Fold Slide-Grouped CV
    for f in range(n_folds):
        tr_sub = train_df[train_df["fold"] != f].reset_index(drop=True)
        va_idx = np.where(train_df["fold"].values == f)[0]
        va_sub = train_df.iloc[va_idx].reset_index(drop=True)

        head_sd, best_ep, _ = _train_single_spatial_mil_fold(
            host_model=host_model,
            tr_df=tr_sub,
            va_df=va_sub,
            stain_bank=stain_bank,
            rt=rt,
            img_size=img_size,
            canonical_strength=canonical_strength,
            num_epochs=num_epochs,
            head_lr=head_lr,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
            seed=seed + f * 17,
            select_on_val=True,
            deadline=deadline,
            tag=f"spatial_mil_f{f}",
        )
        fold_head_states.append(head_sd)
        best_epochs.append(best_ep)

        va_tta_loader = _make_dual_stain_eval_loader(
            va_sub,
            stain_bank=stain_bank,
            img_size=img_size,
            batch_size=tta_bs,
            canonical_strength=canonical_strength,
            tta_views=tta_views,
            include_canonical_view=True,
            seed=seed,
        )
        _, lg_r, lg_c = _predict_spatial_mil_dual_stain(
            host_model, [head_sd], va_tta_loader, device, use_amp, amp_dtype
        )
        oof_raw[va_idx] = lg_r
        oof_canon[va_idx] = lg_c
        del va_tta_loader
        release_memory()

    # 2. 2-Fold LOCO CV (fixed epoch count = median OOF best epoch, zero LOCO leakage)
    loco_epochs = max(4, min(num_epochs, int(round(float(np.median(best_epochs))))))
    for l_idx, (cid, tr_idx, va_idx) in enumerate(loco_fold_indices):
        tr_sub = train_df.iloc[tr_idx].reset_index(drop=True)
        va_sub = train_df.iloc[va_idx].reset_index(drop=True)

        loco_sd, _, _ = _train_single_spatial_mil_fold(
            host_model=host_model,
            tr_df=tr_sub,
            va_df=None,
            stain_bank=stain_bank,
            rt=rt,
            img_size=img_size,
            canonical_strength=canonical_strength,
            num_epochs=loco_epochs,
            head_lr=head_lr,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
            seed=seed + 500 + l_idx * 17,
            select_on_val=False,
            deadline=deadline,
            tag=f"spatial_mil_loco_{cid[:8]}",
        )
        va_tta_loader = _make_dual_stain_eval_loader(
            va_sub,
            stain_bank=stain_bank,
            img_size=img_size,
            batch_size=tta_bs,
            canonical_strength=canonical_strength,
            tta_views=tta_views,
            include_canonical_view=True,
            seed=seed,
        )
        _, lg_r, lg_c = _predict_spatial_mil_dual_stain(
            host_model, [loco_sd], va_tta_loader, device, use_amp, amp_dtype
        )
        loco_raw[va_idx] = lg_r
        loco_canon[va_idx] = lg_c
        del va_tta_loader, loco_sd
        release_memory()

    # 3. Shared-Backbone 5-Fold Bagged Dual-Stain D4 TTA on Official Val & Test
    val_tta_loader = _make_dual_stain_eval_loader(
        val_df,
        stain_bank=stain_bank,
        img_size=img_size,
        batch_size=tta_bs,
        canonical_strength=canonical_strength,
        tta_views=tta_views,
        include_canonical_view=True,
        seed=seed,
    )
    test_tta_loader = _make_dual_stain_eval_loader(
        test_df,
        stain_bank=stain_bank,
        img_size=img_size,
        batch_size=tta_bs,
        canonical_strength=canonical_strength,
        tta_views=tta_views,
        include_canonical_view=True,
        seed=seed,
    )
    val_ids, val_raw, val_canon = _predict_spatial_mil_dual_stain(
        host_model, fold_head_states, val_tta_loader, device, use_amp, amp_dtype
    )
    test_ids, test_raw, test_canon = _predict_spatial_mil_dual_stain(
        host_model, fold_head_states, test_tta_loader, device, use_amp, amp_dtype
    )
    assert val_ids == val_df["sample_id"].astype(str).tolist()
    assert test_ids == test_df["sample_id"].astype(str).tolist()

    torch.save(
        {"head_states": fold_head_states, "best_epochs": best_epochs, "provenance": prov},
        WORKING_DIR / "spatial_mil_stage1_heads.pt",
    )

    del host_model, val_tta_loader, test_tta_loader, fold_head_states
    release_memory()

    return {
        "oof_raw": oof_raw,
        "oof_canon": oof_canon,
        "loco_raw": loco_raw,
        "loco_canon": loco_canon,
        "val_raw": val_raw,
        "val_canon": val_canon,
        "test_raw": test_raw,
        "test_canon": test_canon,
        "median_best_epoch": loco_epochs,
        "provenance": prov,
    }


def _run_spatial_mil_refit(
    backbone_name: str,
    full_train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_folds: np.ndarray,
    refit_bank: StainReferenceBank,
    rt: Dict[str, Any],
    img_size: int,
    canonical_strength: float,
    num_epochs: int,
    tta_views: int,
    head_lr: float,
    weight_decay: float,
    ema_decay: float,
    refit_seed: int,
    run_start: float,
    max_total_seconds: float,
) -> Dict[str, Any]:
    """Time-guarded 5-fold bagged Train+Val refit for frozen-backbone Spatial-MIL Probe."""
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]
    tta_bs: int = max(4, int(rt["tta_eval_batch_size"]))

    n_folds = int(full_folds.max()) + 1
    est_min_needed = 240.0
    if (time.time() - run_start) + est_min_needed >= max_total_seconds:
        print(
            "[spatial_mil] Skipping refit due to time budget; keeping pre-refit logits.",
            flush=True,
        )
        return {"completed": False, "n_refit_folds": 0, "test_raw": None, "test_canon": None}

    host_model = DualStreamStainViT(
        backbone_name=backbone_name,
        pretrained=True,
        img_size=img_size,
        od_dim=256,
        fusion_dim=512,
        drop_path_rate=0.0,
        head_dropout=0.30,
        num_centers=int(full_train_val_df["center_id"].nunique()),
        use_od_stream=True,
        grad_checkpointing=False,
        use_lora=False,
    ).to(device)
    host_model.set_backbone_trainable(False)

    refit_head_states: List[Dict[str, torch.Tensor]] = []
    fold_durations: List[float] = []
    est_test_sec = 75.0

    for f in range(n_folds):
        if f > 0:
            avg_f_sec = float(np.mean(fold_durations))
            if (time.time() - run_start) + avg_f_sec + est_test_sec + 45.0 >= max_total_seconds:
                print(
                    f"[spatial_mil] Refit time guard reached after {f}/{n_folds} folds; "
                    f"proceeding to shared-backbone test inference.",
                    flush=True,
                )
                break
        f_t0 = time.time()
        tr_sub = full_train_val_df.iloc[np.where(full_folds != f)[0]].reset_index(
            drop=True
        )
        va_sub = full_train_val_df.iloc[np.where(full_folds == f)[0]].reset_index(
            drop=True
        )
        head_sd, _, _ = _train_single_spatial_mil_fold(
            host_model=host_model,
            tr_df=tr_sub,
            va_df=va_sub,
            stain_bank=refit_bank,
            rt=rt,
            img_size=img_size,
            canonical_strength=canonical_strength,
            num_epochs=num_epochs,
            head_lr=head_lr,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
            seed=refit_seed + f * 17,
            select_on_val=True,
            deadline=run_start + max_total_seconds - est_test_sec - 45.0,
            tag=f"spatial_mil_refit_f{f}",
        )
        refit_head_states.append(head_sd)
        fold_durations.append(time.time() - f_t0)

    if len(refit_head_states) < 2:
        del host_model, refit_head_states
        release_memory()
        return {"completed": False, "n_refit_folds": 0, "test_raw": None, "test_canon": None}

    test_tta_loader = _make_dual_stain_eval_loader(
        test_df,
        stain_bank=refit_bank,
        img_size=img_size,
        batch_size=tta_bs,
        canonical_strength=canonical_strength,
        tta_views=tta_views,
        include_canonical_view=True,
        seed=refit_seed,
    )
    test_ids, test_raw, test_canon = _predict_spatial_mil_dual_stain(
        host_model, refit_head_states, test_tta_loader, device, use_amp, amp_dtype
    )
    assert test_ids == test_df["sample_id"].astype(str).tolist()

    torch.save(
        {"head_states": refit_head_states, "n_folds": len(refit_head_states)},
        WORKING_DIR / "spatial_mil_refit_heads.pt",
    )

    del host_model, test_tta_loader, refit_head_states
    release_memory()
    return {
        "completed": True,
        "n_refit_folds": len(fold_durations),
        "test_raw": test_raw,
        "test_canon": test_canon,
    }


# -----------------------------------------------------------------------------
# Downstream Paradigm 2: MultiPFMDomainInvariantMLP (Cosine Head + Twin Cons.)
# -----------------------------------------------------------------------------
@torch.no_grad()
def _predict_mlp_logits(
    model: MultiPFMDomainInvariantMLP,
    pfm_arrays: List[np.ndarray],
    od_scaled: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    N = len(od_scaled)
    chunks: List[np.ndarray] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        pfms_t = [
            torch.from_numpy(arr[start:end]).to(device, dtype=torch.float32)
            for arr in pfm_arrays
        ]
        od_t = torch.from_numpy(od_scaled[start:end]).to(device, dtype=torch.float32)
        out = model(pfms_t, od_t)
        chunks.append(out["logits"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def _train_single_mlp_fold(
    pfm_raw_tr: List[np.ndarray],
    pfm_canon_tr: List[np.ndarray],
    pfm_pert_tr: List[np.ndarray],
    od_tr: np.ndarray,
    y_tr: np.ndarray,
    c_tr_idx: np.ndarray,
    sample_weights_tr: np.ndarray,
    pfm_raw_va: Optional[List[np.ndarray]],
    od_va: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    device: torch.device,
    num_epochs: int = 18,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 0.03,
    seed: int = 42,
    select_on_val: bool = True,
) -> Tuple[MultiPFMDomainInvariantMLP, StandardScaler, int]:
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    scaler = StandardScaler()
    od_tr_s = np.clip(scaler.fit_transform(od_tr.astype(np.float32)), -6.0, 6.0).astype(
        np.float32
    )
    od_va_s = (
        np.clip(scaler.transform(od_va.astype(np.float32)), -6.0, 6.0).astype(np.float32)
        if od_va is not None
        else None
    )

    pfm_dims = [int(a.shape[1]) for a in pfm_raw_tr]
    model = MultiPFMDomainInvariantMLP(
        pfm_dims=pfm_dims,
        od_dim=od_tr_s.shape[1],
        stem_dim=384,
        od_hidden_dim=64,
        hidden_dim=512,
        dropout=0.30,
        stream_dropout=0.15,
    ).to(device)
    ema = ModelEMA(model, decay=0.985)

    criterion_clean = DomainGeneralizingHistopathLoss(
        label_smoothing=0.03,
        focal_gamma=1.0,
        sba_weight=0.30,
        parity_weight=0.25,
        aux_od_weight=0.0,
        vrex_weight=0.25,
        cdan_weight=0.0,
        logit_cons_weight=0.0,
        emb_cons_weight=0.0,
    )
    criterion_pert = DomainGeneralizingHistopathLoss(
        label_smoothing=0.03,
        focal_gamma=1.0,
        sba_weight=0.30,
        parity_weight=0.25,
        aux_od_weight=0.0,
        vrex_weight=0.25,
        cdan_weight=0.0,
        logit_cons_weight=0.35,
        emb_cons_weight=0.20,
    )

    no_decay = ("bias", "norm", "prototypes", "logit_scale")
    decay_p, no_decay_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or any(k in n.lower() for k in no_decay):
            no_decay_p.append(p)
        else:
            decay_p.append(p)
    optimizer = AdamW(
        [
            {"params": decay_p, "weight_decay": weight_decay},
            {"params": no_decay_p, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    n_tr = len(y_tr)
    steps_per_epoch = max(1, n_tr // batch_size)
    total_steps = num_epochs * steps_per_epoch
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=total_steps, warmup_ratio=0.12, min_lr_ratio=0.02
    )

    sampling_p = sample_weights_tr / max(float(sample_weights_tr.sum()), 1e-12)
    n_pert_views = int(pfm_pert_tr[0].shape[1])

    best_key = (-1.0, -1.0)
    best_epoch = num_epochs
    best_state = copy.deepcopy(ema.ema_model.state_dict())
    global_step = 0

    for ep in range(1, num_epochs + 1):
        model.train()
        sampled_indices = rng.choice(
            n_tr, size=steps_per_epoch * batch_size, replace=True, p=sampling_p
        )
        for step_i in range(steps_per_epoch):
            idx = sampled_indices[step_i * batch_size : (step_i + 1) * batch_size]
            use_canon_mask = rng.random(len(idx)) < 0.30
            v_ids = rng.integers(0, n_pert_views, size=len(idx))

            clean_pfms_t: List[torch.Tensor] = []
            pert_pfms_t: List[torch.Tensor] = []
            for p_raw, p_can, p_pert in zip(pfm_raw_tr, pfm_canon_tr, pfm_pert_tr):
                c_np = p_raw[idx].copy()
                if use_canon_mask.any():
                    c_np[use_canon_mask] = p_can[idx[use_canon_mask]]
                clean_pfms_t.append(torch.from_numpy(c_np).to(device, dtype=torch.float32))
                pert_np = p_pert[idx, v_ids].astype(np.float32)
                pert_pfms_t.append(torch.from_numpy(pert_np).to(device, dtype=torch.float32))

            od_clean_t = torch.from_numpy(od_tr_s[idx]).to(device, dtype=torch.float32)
            od_pert_t = od_clean_t + 0.05 * torch.randn_like(od_clean_t)
            y_t = torch.from_numpy(y_tr[idx]).to(device, dtype=torch.float32)
            c_t = torch.from_numpy(c_tr_idx[idx]).to(device, dtype=torch.long)

            progress = float(global_step) / float(max(1, total_steps))
            domain_scale = compute_domain_schedule(progress, warmup_frac=0.15, max_lambda=0.5)
            cons_scale = compute_domain_schedule(progress, warmup_frac=0.12, max_lambda=1.0)

            with torch.no_grad():
                teacher_out = ema.ema_model(clean_pfms_t, od_clean_t)

            optimizer.zero_grad(set_to_none=True)
            out_clean = model(clean_pfms_t, od_clean_t)
            out_pert = model(pert_pfms_t, od_pert_t)

            l_clean = criterion_clean(
                out_clean, y_t, center_idx=c_t, domain_scale=domain_scale
            )["loss"]
            l_pert = criterion_pert(
                out_pert,
                y_t,
                center_idx=c_t,
                domain_scale=domain_scale,
                teacher_logits=teacher_out["logits"],
                teacher_embedding=teacher_out["embedding"],
                cons_scale=cons_scale,
            )["loss"]
            loss = 0.5 * l_clean + 0.5 * l_pert
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            ema.update(model, step=global_step)

        if select_on_val and pfm_raw_va is not None and od_va_s is not None and y_va is not None:
            va_lg = _predict_mlp_logits(ema.ema_model, pfm_raw_va, od_va_s, device)
            va_p = logit_to_prob(va_lg)
            ep_key = (
                round(float(balanced_accuracy_score(y_va, va_p >= 0.5)), 6),
                float(roc_auc_score(y_va, va_p)),
            )
            if ep_key > best_key:
                best_key = ep_key
                best_epoch = ep
                best_state = copy.deepcopy(ema.ema_model.state_dict())
        else:
            best_epoch = ep
            best_state = copy.deepcopy(ema.ema_model.state_dict())

    ema.ema_model.load_state_dict(best_state)
    ema.ema_model.eval()
    return ema.ema_model, scaler, best_epoch


def _run_multi_pfm_mlp_cv(
    pfm_bank: Dict[str, Any],
    train_df: pd.DataFrame,
    loco_fold_indices: List[Tuple[str, np.ndarray, np.ndarray]],
    n_folds: int,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, Any]:
    pfms = pfm_bank["pfms"]
    od_tr_all = pfm_bank["od_train"]
    od_va_all = pfm_bank["od_val"]
    od_te_all = pfm_bank["od_test"]

    y_all = train_df["label"].astype(np.int64).values
    unique_c = sorted(train_df["center_id"].astype(str).unique().tolist())
    c_map = {c: i for i, c in enumerate(unique_c)}
    c_idx_all = np.array([c_map[str(c)] for c in train_df["center_id"]], dtype=np.int64)
    folds = train_df["fold"].values

    n_tr = len(train_df)
    n_val = len(od_va_all)
    n_test = len(od_te_all)

    oof_raw = np.zeros(n_tr, dtype=np.float64)
    oof_canon = np.zeros(n_tr, dtype=np.float64)
    loco_raw = np.zeros(n_tr, dtype=np.float64)
    loco_canon = np.zeros(n_tr, dtype=np.float64)
    val_raw = np.zeros(n_val, dtype=np.float64)
    val_canon = np.zeros(n_val, dtype=np.float64)
    test_raw = np.zeros(n_test, dtype=np.float64)
    test_canon = np.zeros(n_test, dtype=np.float64)

    best_epochs: List[int] = []
    for f in range(n_folds):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])

        mlp_f, scaler_f, best_ep = _train_single_mlp_fold(
            pfm_raw_tr=[p["train_raw"][tr_idx] for p in pfms],
            pfm_canon_tr=[p["train_canon"][tr_idx] for p in pfms],
            pfm_pert_tr=[p["train_pert"][tr_idx] for p in pfms],
            od_tr=od_tr_all[tr_idx],
            y_tr=y_all[tr_idx],
            c_tr_idx=c_idx_all[tr_idx],
            sample_weights_tr=w_tr,
            pfm_raw_va=[p["train_raw"][va_idx] for p in pfms],
            od_va=od_tr_all[va_idx],
            y_va=y_all[va_idx],
            device=device,
            num_epochs=18,
            seed=seed + f * 31,
            select_on_val=True,
        )
        best_epochs.append(best_ep)

        od_va_s = np.clip(scaler_f.transform(od_tr_all[va_idx]), -6.0, 6.0).astype(np.float32)
        oof_raw[va_idx] = _predict_mlp_logits(
            mlp_f, [p["train_raw"][va_idx] for p in pfms], od_va_s, device
        )
        oof_canon[va_idx] = _predict_mlp_logits(
            mlp_f, [p["train_canon"][va_idx] for p in pfms], od_va_s, device
        )

        od_val_s = np.clip(scaler_f.transform(od_va_all), -6.0, 6.0).astype(np.float32)
        od_test_s = np.clip(scaler_f.transform(od_te_all), -6.0, 6.0).astype(np.float32)
        val_raw += _predict_mlp_logits(mlp_f, [p["val_raw"] for p in pfms], od_val_s, device) / n_folds
        val_canon += _predict_mlp_logits(mlp_f, [p["val_canon"] for p in pfms], od_val_s, device) / n_folds
        test_raw += _predict_mlp_logits(mlp_f, [p["test_raw"] for p in pfms], od_test_s, device) / n_folds
        test_canon += _predict_mlp_logits(mlp_f, [p["test_canon"] for p in pfms], od_test_s, device) / n_folds

    loco_ep = max(10, min(18, int(round(float(np.median(best_epochs))))))
    for l_idx, (_, tr_idx, va_idx) in enumerate(loco_fold_indices):
        w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])
        mlp_l, scaler_l, _ = _train_single_mlp_fold(
            pfm_raw_tr=[p["train_raw"][tr_idx] for p in pfms],
            pfm_canon_tr=[p["train_canon"][tr_idx] for p in pfms],
            pfm_pert_tr=[p["train_pert"][tr_idx] for p in pfms],
            od_tr=od_tr_all[tr_idx],
            y_tr=y_all[tr_idx],
            c_tr_idx=c_idx_all[tr_idx],
            sample_weights_tr=w_tr,
            pfm_raw_va=None,
            od_va=None,
            y_va=None,
            device=device,
            num_epochs=loco_ep,
            seed=seed + 700 + l_idx * 31,
            select_on_val=False,
        )
        od_va_s = np.clip(scaler_l.transform(od_tr_all[va_idx]), -6.0, 6.0).astype(np.float32)
        loco_raw[va_idx] = _predict_mlp_logits(
            mlp_l, [p["train_raw"][va_idx] for p in pfms], od_va_s, device
        )
        loco_canon[va_idx] = _predict_mlp_logits(
            mlp_l, [p["train_canon"][va_idx] for p in pfms], od_va_s, device
        )

    return {
        "oof_raw": oof_raw,
        "oof_canon": oof_canon,
        "loco_raw": loco_raw,
        "loco_canon": loco_canon,
        "val_raw": val_raw,
        "val_canon": val_canon,
        "test_raw": test_raw,
        "test_canon": test_canon,
        "median_best_epoch": loco_ep,
    }


def _run_multi_pfm_mlp_refit(
    refit_bank_feats: Dict[str, Any],
    full_train_val_df: pd.DataFrame,
    full_folds: np.ndarray,
    device: torch.device,
    refit_seed: int,
) -> Dict[str, Any]:
    pfms = refit_bank_feats["pfms"]
    od_full = refit_bank_feats["od_full"]
    od_test = refit_bank_feats["od_test"]
    y_all = full_train_val_df["label"].astype(np.int64).values
    unique_c = sorted(full_train_val_df["center_id"].astype(str).unique().tolist())
    c_map = {c: i for i, c in enumerate(unique_c)}
    c_idx_all = np.array(
        [c_map[str(c)] for c in full_train_val_df["center_id"]], dtype=np.int64
    )

    n_folds = int(full_folds.max()) + 1
    test_raw = np.zeros(len(od_test), dtype=np.float64)
    test_canon = np.zeros(len(od_test), dtype=np.float64)

    for f in range(n_folds):
        tr_idx = np.where(full_folds != f)[0]
        va_idx = np.where(full_folds == f)[0]
        w_tr = _compute_slide_tempered_weights(full_train_val_df.iloc[tr_idx])
        mlp_f, scaler_f, _ = _train_single_mlp_fold(
            pfm_raw_tr=[p["full_raw"][tr_idx] for p in pfms],
            pfm_canon_tr=[p["full_canon"][tr_idx] for p in pfms],
            pfm_pert_tr=[p["full_pert"][tr_idx] for p in pfms],
            od_tr=od_full[tr_idx],
            y_tr=y_all[tr_idx],
            c_tr_idx=c_idx_all[tr_idx],
            sample_weights_tr=w_tr,
            pfm_raw_va=[p["full_raw"][va_idx] for p in pfms],
            od_va=od_full[va_idx],
            y_va=y_all[va_idx],
            device=device,
            num_epochs=18,
            seed=refit_seed + f * 31,
            select_on_val=True,
        )
        od_test_s = np.clip(scaler_f.transform(od_test), -6.0, 6.0).astype(np.float32)
        test_raw += _predict_mlp_logits(mlp_f, [p["test_raw"] for p in pfms], od_test_s, device) / n_folds
        test_canon += _predict_mlp_logits(mlp_f, [p["test_canon"] for p in pfms], od_test_s, device) / n_folds

    return {"completed": True, "test_raw": test_raw, "test_canon": test_canon}


# -----------------------------------------------------------------------------
# Downstream Paradigm 3: CenterDebiasedLogisticProbe (CV-tuned C and beta)
# -----------------------------------------------------------------------------
def _run_center_debiased_lr_cv(
    pfm_bank: Dict[str, Any],
    train_df: pd.DataFrame,
    loco_fold_indices: List[Tuple[str, np.ndarray, np.ndarray]],
    n_folds: int,
    seed: int = 42,
) -> Dict[str, Any]:
    pfms = pfm_bank["pfms"]
    od_tr_all = pfm_bank["od_train"]
    od_va_all = pfm_bank["od_val"]
    od_te_all = pfm_bank["od_test"]
    y_all = train_df["label"].astype(np.int64).values
    c_all = train_df["center_id"].astype(str).values
    folds = train_df["fold"].values

    grid = [
        (C_val, beta_val)
        for C_val in (0.08, 0.25, 0.80)
        for beta_val in (0.0, 0.50, 0.85)
    ]

    best_score = (-1.0, -1.0)
    best_cfg = (0.25, 0.50)
    best_out: Dict[str, Any] = {}

    for C_val, beta_val in grid:
        n_tr = len(train_df)
        oof_raw = np.zeros(n_tr, dtype=np.float64)
        oof_canon = np.zeros(n_tr, dtype=np.float64)
        loco_raw = np.zeros(n_tr, dtype=np.float64)
        loco_canon = np.zeros(n_tr, dtype=np.float64)

        for f in range(n_folds):
            tr_idx = np.where(folds != f)[0]
            va_idx = np.where(folds == f)[0]
            w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])
            probe = CenterDebiasedLogisticProbe(C=C_val, beta=beta_val, seed=seed + f)
            probe.fit(
                pfm_raw_list=[p["train_raw"][tr_idx] for p in pfms],
                od_feats=od_tr_all[tr_idx],
                labels=y_all[tr_idx],
                centers=c_all[tr_idx],
                sample_weights=w_tr,
                pfm_pert_mean_list=[p["train_pert_mean"][tr_idx] for p in pfms],
            )
            oof_raw[va_idx] = probe.predict_logits(
                [p["train_raw"][va_idx] for p in pfms], od_tr_all[va_idx]
            )
            oof_canon[va_idx] = probe.predict_logits(
                [p["train_canon"][va_idx] for p in pfms], od_tr_all[va_idx]
            )

        for l_idx, (_, tr_idx, va_idx) in enumerate(loco_fold_indices):
            w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])
            probe = CenterDebiasedLogisticProbe(
                C=C_val, beta=beta_val, seed=seed + 100 + l_idx
            )
            probe.fit(
                pfm_raw_list=[p["train_raw"][tr_idx] for p in pfms],
                od_feats=od_tr_all[tr_idx],
                labels=y_all[tr_idx],
                centers=c_all[tr_idx],
                sample_weights=w_tr,
                pfm_pert_mean_list=[p["train_pert_mean"][tr_idx] for p in pfms],
            )
            loco_raw[va_idx] = probe.predict_logits(
                [p["train_raw"][va_idx] for p in pfms], od_tr_all[va_idx]
            )
            loco_canon[va_idx] = probe.predict_logits(
                [p["train_canon"][va_idx] for p in pfms], od_tr_all[va_idx]
            )

        p_oof = logit_to_prob(oof_raw)
        p_loco = logit_to_prob(loco_raw)
        bal = 0.60 * balanced_accuracy_score(y_all, p_loco >= 0.5) + 0.40 * balanced_accuracy_score(
            y_all, p_oof >= 0.5
        )
        auc = 0.60 * roc_auc_score(y_all, p_loco) + 0.40 * roc_auc_score(y_all, p_oof)
        key = (round(float(bal), 6), float(auc))
        if key > best_score:
            best_score = key
            best_cfg = (C_val, beta_val)
            best_out = {
                "oof_raw": oof_raw,
                "oof_canon": oof_canon,
                "loco_raw": loco_raw,
                "loco_canon": loco_canon,
            }

    best_C, best_beta = best_cfg
    val_raw = np.zeros(len(od_va_all), dtype=np.float64)
    val_canon = np.zeros(len(od_va_all), dtype=np.float64)
    test_raw = np.zeros(len(od_te_all), dtype=np.float64)
    test_canon = np.zeros(len(od_te_all), dtype=np.float64)

    for f in range(n_folds):
        tr_idx = np.where(folds != f)[0]
        w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])
        probe = CenterDebiasedLogisticProbe(C=best_C, beta=best_beta, seed=seed + f)
        probe.fit(
            pfm_raw_list=[p["train_raw"][tr_idx] for p in pfms],
            od_feats=od_tr_all[tr_idx],
            labels=y_all[tr_idx],
            centers=c_all[tr_idx],
            sample_weights=w_tr,
            pfm_pert_mean_list=[p["train_pert_mean"][tr_idx] for p in pfms],
        )
        val_raw += probe.predict_logits([p["val_raw"] for p in pfms], od_va_all) / n_folds
        val_canon += probe.predict_logits([p["val_canon"] for p in pfms], od_va_all) / n_folds
        test_raw += probe.predict_logits([p["test_raw"] for p in pfms], od_te_all) / n_folds
        test_canon += probe.predict_logits([p["test_canon"] for p in pfms], od_te_all) / n_folds

    best_out.update(
        {
            "val_raw": val_raw,
            "val_canon": val_canon,
            "test_raw": test_raw,
            "test_canon": test_canon,
            "best_C": float(best_C),
            "best_beta": float(best_beta),
        }
    )
    print(
        f"[center_debiased_lr] Selected C={best_C}, beta={best_beta} "
        f"(CV obj bal={best_score[0]:.5f}, auc={best_score[1]:.5f})",
        flush=True,
    )
    return best_out


def _run_center_debiased_lr_refit(
    refit_bank_feats: Dict[str, Any],
    full_train_val_df: pd.DataFrame,
    full_folds: np.ndarray,
    best_C: float,
    best_beta: float,
    refit_seed: int,
) -> Dict[str, Any]:
    pfms = refit_bank_feats["pfms"]
    od_full = refit_bank_feats["od_full"]
    od_test = refit_bank_feats["od_test"]
    y_all = full_train_val_df["label"].astype(np.int64).values
    c_all = full_train_val_df["center_id"].astype(str).values
    n_folds = int(full_folds.max()) + 1

    test_raw = np.zeros(len(od_test), dtype=np.float64)
    test_canon = np.zeros(len(od_test), dtype=np.float64)
    for f in range(n_folds):
        tr_idx = np.where(full_folds != f)[0]
        w_tr = _compute_slide_tempered_weights(full_train_val_df.iloc[tr_idx])
        probe = CenterDebiasedLogisticProbe(
            C=best_C, beta=best_beta, seed=refit_seed + f
        )
        probe.fit(
            pfm_raw_list=[p["full_raw"][tr_idx] for p in pfms],
            od_feats=od_full[tr_idx],
            labels=y_all[tr_idx],
            centers=c_all[tr_idx],
            sample_weights=w_tr,
            pfm_pert_mean_list=[p["full_pert_mean"][tr_idx] for p in pfms],
        )
        test_raw += probe.predict_logits([p["test_raw"] for p in pfms], od_test) / n_folds
        test_canon += probe.predict_logits([p["test_canon"] for p in pfms], od_test) / n_folds

    return {"completed": True, "test_raw": test_raw, "test_canon": test_canon}


# -----------------------------------------------------------------------------
# Downstream Paradigm 4: Fold-Internal LightGBM & XGBoost GBDT Learners
# -----------------------------------------------------------------------------
def _fit_single_lgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    n_estimators: int,
    seed: int,
) -> Tuple[Any, int]:
    if lgb is not None:
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=40,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_alpha=1.0,
            reg_lambda=5.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        if X_va is not None and y_va is not None:
            clf.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )
            best_it = int(getattr(clf, "best_iteration_", 0) or n_estimators)
        else:
            clf.fit(X_tr, y_tr, sample_weight=w_tr)
            best_it = n_estimators
        return clf, max(50, best_it)

    clf = HistGradientBoostingClassifier(
        max_iter=min(n_estimators, 300),
        learning_rate=0.03,
        max_depth=4,
        min_samples_leaf=40,
        l2_regularization=5.0,
        random_state=seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    return clf, min(n_estimators, 300)


def _fit_single_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    w_tr: np.ndarray,
    X_va: Optional[np.ndarray],
    y_va: Optional[np.ndarray],
    n_estimators: int,
    seed: int,
) -> Tuple[Any, int]:
    if xgb is not None:
        kwargs: Dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": n_estimators,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 20.0,
            "subsample": 0.80,
            "colsample_bytree": 0.50,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
        }
        if X_va is not None and y_va is not None:
            kwargs["early_stopping_rounds"] = 40
            clf = xgb.XGBClassifier(**kwargs)
            clf.fit(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
            best_it = getattr(clf, "best_iteration", None)
            best_it = int(best_it) + 1 if best_it is not None else n_estimators
        else:
            clf = xgb.XGBClassifier(**kwargs)
            clf.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
            best_it = n_estimators
        return clf, max(50, best_it)

    clf = HistGradientBoostingClassifier(
        max_iter=min(n_estimators, 300),
        learning_rate=0.035,
        max_depth=5,
        min_samples_leaf=30,
        l2_regularization=3.0,
        random_state=seed + 7,
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    return clf, min(n_estimators, 300)


def _gbdt_predict_logits(clf: Any, X: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(X)[:, 1]
    return prob_to_logit(probs)


def _run_gbdt_cv(
    pfm_bank: Dict[str, Any],
    train_df: pd.DataFrame,
    loco_fold_indices: List[Tuple[str, np.ndarray, np.ndarray]],
    n_folds: int,
    n_svd: int = 32,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pfms = pfm_bank["pfms"]
    od_tr_all = pfm_bank["od_train"]
    od_va_all = pfm_bank["od_val"]
    od_te_all = pfm_bank["od_test"]
    y_all = train_df["label"].astype(np.int64).values
    c_all = train_df["center_id"].astype(str).values
    s_all = train_df["slide_id"].astype(str).values
    folds = train_df["fold"].values

    n_tr, n_val, n_test = len(train_df), len(od_va_all), len(od_te_all)

    def _empty_pack():
        return {
            "oof_raw": np.zeros(n_tr, dtype=np.float64),
            "oof_canon": np.zeros(n_tr, dtype=np.float64),
            "loco_raw": np.zeros(n_tr, dtype=np.float64),
            "loco_canon": np.zeros(n_tr, dtype=np.float64),
            "val_raw": np.zeros(n_val, dtype=np.float64),
            "val_canon": np.zeros(n_val, dtype=np.float64),
            "test_raw": np.zeros(n_test, dtype=np.float64),
            "test_canon": np.zeros(n_test, dtype=np.float64),
        }

    lgb_out = _empty_pack()
    xgb_out = _empty_pack()
    lgb_iters: List[int] = []
    xgb_iters: List[int] = []

    for f in range(n_folds):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]
        w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])

        builder = FoldGBDTFeatureBuilder(n_svd=n_svd, seed=seed + f * 13)
        X_tr_raw = builder.fit_transform(
            pfm_list=[p["train_raw"][tr_idx] for p in pfms],
            od_feats=od_tr_all[tr_idx],
            labels=y_all[tr_idx],
            centers=c_all[tr_idx],
            slides=s_all[tr_idx],
            pfm_pert_mean_list=[p["train_pert_mean"][tr_idx] for p in pfms],
        )
        X_tr_pert = builder.transform(
            pfm_list=[p["train_pert_mean"][tr_idx] for p in pfms],
            od_feats=od_tr_all[tr_idx],
            train_y=y_all[tr_idx],
            train_slides=s_all[tr_idx],
        )
        X_fit = np.concatenate([X_tr_raw, X_tr_pert], axis=0)
        y_fit = np.concatenate([y_all[tr_idx], y_all[tr_idx]], axis=0)
        w_fit = np.concatenate([w_tr, w_tr], axis=0)

        X_va_raw = builder.transform([p["train_raw"][va_idx] for p in pfms], od_tr_all[va_idx])
        X_va_can = builder.transform([p["train_canon"][va_idx] for p in pfms], od_tr_all[va_idx])
        X_val_raw = builder.transform([p["val_raw"] for p in pfms], od_va_all)
        X_val_can = builder.transform([p["val_canon"] for p in pfms], od_va_all)
        X_te_raw = builder.transform([p["test_raw"] for p in pfms], od_te_all)
        X_te_can = builder.transform([p["test_canon"] for p in pfms], od_te_all)

        clf_lgb, it_lgb = _fit_single_lgbm(
            X_fit, y_fit, w_fit, X_va_raw, y_all[va_idx], n_estimators=450, seed=seed + f
        )
        clf_xgb, it_xgb = _fit_single_xgb(
            X_fit, y_fit, w_fit, X_va_raw, y_all[va_idx], n_estimators=450, seed=seed + f
        )
        lgb_iters.append(it_lgb)
        xgb_iters.append(it_xgb)

        for clf, pack in ((clf_lgb, lgb_out), (clf_xgb, xgb_out)):
            pack["oof_raw"][va_idx] = _gbdt_predict_logits(clf, X_va_raw)
            pack["oof_canon"][va_idx] = _gbdt_predict_logits(clf, X_va_can)
            pack["val_raw"] += _gbdt_predict_logits(clf, X_val_raw) / n_folds
            pack["val_canon"] += _gbdt_predict_logits(clf, X_val_can) / n_folds
            pack["test_raw"] += _gbdt_predict_logits(clf, X_te_raw) / n_folds
            pack["test_canon"] += _gbdt_predict_logits(clf, X_te_can) / n_folds

    loco_lgb_n = max(80, min(450, int(round(float(np.median(lgb_iters))))))
    loco_xgb_n = max(80, min(450, int(round(float(np.median(xgb_iters))))))

    for l_idx, (_, tr_idx, va_idx) in enumerate(loco_fold_indices):
        w_tr = _compute_slide_tempered_weights(train_df.iloc[tr_idx])
        builder = FoldGBDTFeatureBuilder(n_svd=n_svd, seed=seed + 300 + l_idx * 13)
        X_tr_raw = builder.fit_transform(
            pfm_list=[p["train_raw"][tr_idx] for p in pfms],
            od_feats=od_tr_all[tr_idx],
            labels=y_all[tr_idx],
            centers=c_all[tr_idx],
            slides=s_all[tr_idx],
            pfm_pert_mean_list=[p["train_pert_mean"][tr_idx] for p in pfms],
        )
        X_tr_pert = builder.transform(
            pfm_list=[p["train_pert_mean"][tr_idx] for p in pfms],
            od_feats=od_tr_all[tr_idx],
            train_y=y_all[tr_idx],
            train_slides=s_all[tr_idx],
        )
        X_fit = np.concatenate([X_tr_raw, X_tr_pert], axis=0)
        y_fit = np.concatenate([y_all[tr_idx], y_all[tr_idx]], axis=0)
        w_fit = np.concatenate([w_tr, w_tr], axis=0)

        X_va_raw = builder.transform([p["train_raw"][va_idx] for p in pfms], od_tr_all[va_idx])
        X_va_can = builder.transform([p["train_canon"][va_idx] for p in pfms], od_tr_all[va_idx])

        clf_lgb, _ = _fit_single_lgbm(
            X_fit, y_fit, w_fit, None, None, n_estimators=loco_lgb_n, seed=seed + 300 + l_idx
        )
        clf_xgb, _ = _fit_single_xgb(
            X_fit, y_fit, w_fit, None, None, n_estimators=loco_xgb_n, seed=seed + 300 + l_idx
        )
        lgb_out["loco_raw"][va_idx] = _gbdt_predict_logits(clf_lgb, X_va_raw)
        lgb_out["loco_canon"][va_idx] = _gbdt_predict_logits(clf_lgb, X_va_can)
        xgb_out["loco_raw"][va_idx] = _gbdt_predict_logits(clf_xgb, X_va_raw)
        xgb_out["loco_canon"][va_idx] = _gbdt_predict_logits(clf_xgb, X_va_can)

    lgb_out["median_best_iter"] = loco_lgb_n
    xgb_out["median_best_iter"] = loco_xgb_n
    return lgb_out, xgb_out


def _run_gbdt_refit(
    refit_bank_feats: Dict[str, Any],
    full_train_val_df: pd.DataFrame,
    full_folds: np.ndarray,
    n_svd: int,
    refit_seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    pfms = refit_bank_feats["pfms"]
    od_full = refit_bank_feats["od_full"]
    od_test = refit_bank_feats["od_test"]
    y_all = full_train_val_df["label"].astype(np.int64).values
    c_all = full_train_val_df["center_id"].astype(str).values
    s_all = full_train_val_df["slide_id"].astype(str).values
    n_folds = int(full_folds.max()) + 1

    lgb_te_raw = np.zeros(len(od_test), dtype=np.float64)
    lgb_te_can = np.zeros(len(od_test), dtype=np.float64)
    xgb_te_raw = np.zeros(len(od_test), dtype=np.float64)
    xgb_te_can = np.zeros(len(od_test), dtype=np.float64)

    for f in range(n_folds):
        tr_idx = np.where(full_folds != f)[0]
        va_idx = np.where(full_folds == f)[0]
        w_tr = _compute_slide_tempered_weights(full_train_val_df.iloc[tr_idx])

        builder = FoldGBDTFeatureBuilder(n_svd=n_svd, seed=refit_seed + f * 13)
        X_tr_raw = builder.fit_transform(
            pfm_list=[p["full_raw"][tr_idx] for p in pfms],
            od_feats=od_full[tr_idx],
            labels=y_all[tr_idx],
            centers=c_all[tr_idx],
            slides=s_all[tr_idx],
            pfm_pert_mean_list=[p["full_pert_mean"][tr_idx] for p in pfms],
        )
        X_tr_pert = builder.transform(
            pfm_list=[p["full_pert_mean"][tr_idx] for p in pfms],
            od_feats=od_full[tr_idx],
            train_y=y_all[tr_idx],
            train_slides=s_all[tr_idx],
        )
        X_fit = np.concatenate([X_tr_raw, X_tr_pert], axis=0)
        y_fit = np.concatenate([y_all[tr_idx], y_all[tr_idx]], axis=0)
        w_fit = np.concatenate([w_tr, w_tr], axis=0)

        X_va_raw = builder.transform([p["full_raw"][va_idx] for p in pfms], od_full[va_idx])
        X_te_raw = builder.transform([p["test_raw"] for p in pfms], od_test)
        X_te_can = builder.transform([p["test_canon"] for p in pfms], od_test)

        clf_lgb, _ = _fit_single_lgbm(
            X_fit, y_fit, w_fit, X_va_raw, y_all[va_idx], n_estimators=450, seed=refit_seed + f
        )
        clf_xgb, _ = _fit_single_xgb(
            X_fit, y_fit, w_fit, X_va_raw, y_all[va_idx], n_estimators=450, seed=refit_seed + f
        )
        lgb_te_raw += _gbdt_predict_logits(clf_lgb, X_te_raw) / n_folds
        lgb_te_can += _gbdt_predict_logits(clf_lgb, X_te_can) / n_folds
        xgb_te_raw += _gbdt_predict_logits(clf_xgb, X_te_raw) / n_folds
        xgb_te_can += _gbdt_predict_logits(clf_xgb, X_te_can) / n_folds

    return (
        {"completed": True, "test_raw": lgb_te_raw, "test_canon": lgb_te_can},
        {"completed": True, "test_raw": xgb_te_raw, "test_canon": xgb_te_can},
    )


# -----------------------------------------------------------------------------
# Joint CV-Gated Optimization of alpha_canon & Bounded Simplex Logit Weights
# -----------------------------------------------------------------------------
def _optimize_cv_ensemble_weights_and_alpha(
    paradigm_outputs: Dict[str, Dict[str, Any]],
    y_train: np.ndarray,
    alpha_grid: Tuple[float, ...] = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50),
    max_single_weight: float = 0.65,
    n_random_simplex: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Strictly inside train.csv (zero validation.csv leakage), jointly optimize:
      - alpha_canon in [0.0, 0.5] (raw vs soft-canonical symmetric logit blend)
      - bounded non-negative simplex weights w_m in [0.0, max_single_weight], sum(w)=1
    maximizing 0.60 * LOCO_BalAcc + 0.40 * OOF_BalAcc at fixed 0.5 threshold
    (with 0.60 * LOCO_AUROC + 0.40 * OOF_AUROC tie-break).
    """
    rng = np.random.default_rng(seed)
    names = list(paradigm_outputs.keys())
    M = len(names)
    y = np.asarray(y_train, dtype=np.int64)

    # Generate structured + Dirichlet simplex candidates respecting weight cap
    cand_weights: List[np.ndarray] = [np.full(M, 1.0 / M, dtype=np.float64)]
    for i in range(M):
        for j in range(i + 1, M):
            for wi in (0.35, 0.50, 0.65):
                w = np.zeros(M, dtype=np.float64)
                w[i], w[j] = wi, 1.0 - wi
                cand_weights.append(w)
    for alpha_dir in (0.5, 1.0, 2.5):
        draws = rng.dirichlet(np.full(M, alpha_dir), size=n_random_simplex // 3)
        for w in draws:
            if float(w.max()) <= max_single_weight:
                cand_weights.append(w)

    W_mat = np.unique(np.round(np.stack(cand_weights, axis=0), 4), axis=0)
    W_mat = W_mat / W_mat.sum(axis=1, keepdims=True)

    best_key = (-1.0, -1.0)
    best_alpha = 0.0
    best_w = np.full(M, 1.0 / M, dtype=np.float64)
    best_diag: Dict[str, float] = {}

    for alpha in alpha_grid:
        oof_M = np.stack(
            [
                (1.0 - alpha) * paradigm_outputs[n]["oof_raw"]
                + alpha * paradigm_outputs[n]["oof_canon"]
                for n in names
            ],
            axis=0,
        )  # (M, N)
        loco_M = np.stack(
            [
                (1.0 - alpha) * paradigm_outputs[n]["loco_raw"]
                + alpha * paradigm_outputs[n]["loco_canon"]
                for n in names
            ],
            axis=0,
        )

        # Vectorized balanced accuracy screening across all simplex candidates
        oof_preds = (W_mat @ oof_M) >= 0.0
        loco_preds = (W_mat @ loco_M) >= 0.0
        pos_m = y == 1
        neg_m = y == 0

        oof_bal_all = 0.5 * (
            oof_preds[:, pos_m].mean(axis=1) + (~oof_preds[:, neg_m]).mean(axis=1)
        )
        loco_bal_all = 0.5 * (
            loco_preds[:, pos_m].mean(axis=1) + (~loco_preds[:, neg_m]).mean(axis=1)
        )
        obj_bal_all = 0.60 * loco_bal_all + 0.40 * oof_bal_all

        top_k = np.argsort(obj_bal_all)[-25:][::-1]
        for idx in top_k:
            w = W_mat[idx]
            oof_lg = w @ oof_M
            loco_lg = w @ loco_M
            oof_auc = float(roc_auc_score(y, logit_to_prob(oof_lg)))
            loco_auc = float(roc_auc_score(y, logit_to_prob(loco_lg)))
            obj_bal = float(obj_bal_all[idx])
            obj_auc = 0.60 * loco_auc + 0.40 * oof_auc
            key = (round(obj_bal, 6), round(obj_auc, 6))
            if key > best_key:
                best_key = key
                best_alpha = float(alpha)
                best_w = w.copy()
                best_diag = {
                    "oof_bal_acc": float(oof_bal_all[idx]),
                    "oof_auroc": oof_auc,
                    "loco_bal_acc": float(loco_bal_all[idx]),
                    "loco_auroc": loco_auc,
                    "cv_objective_bal": obj_bal,
                    "cv_objective_auc": obj_auc,
                }

    # Local coordinate refinement around best_w at best_alpha
    oof_M = np.stack(
        [
            (1.0 - best_alpha) * paradigm_outputs[n]["oof_raw"]
            + best_alpha * paradigm_outputs[n]["oof_canon"]
            for n in names
        ],
        axis=0,
    )
    loco_M = np.stack(
        [
            (1.0 - best_alpha) * paradigm_outputs[n]["loco_raw"]
            + best_alpha * paradigm_outputs[n]["loco_canon"]
            for n in names
        ],
        axis=0,
    )
    for step_sz in (0.04, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    w_try = best_w.copy()
                    w_try[i] += step_sz
                    w_try[j] -= step_sz
                    if w_try[j] < -1e-9 or w_try[i] > max_single_weight + 1e-9:
                        continue
                    w_try = np.clip(w_try, 0.0, max_single_weight)
                    w_try = w_try / w_try.sum()
                    oof_lg = w_try @ oof_M
                    loco_lg = w_try @ loco_M
                    oof_p = logit_to_prob(oof_lg)
                    loco_p = logit_to_prob(loco_lg)
                    oof_b = float(balanced_accuracy_score(y, oof_p >= 0.5))
                    loco_b = float(balanced_accuracy_score(y, loco_p >= 0.5))
                    oof_a = float(roc_auc_score(y, oof_p))
                    loco_a = float(roc_auc_score(y, loco_p))
                    obj_b = 0.60 * loco_b + 0.40 * oof_b
                    obj_a = 0.60 * loco_a + 0.40 * oof_a
                    key = (round(obj_b, 6), round(obj_a, 6))
                    if key > best_key:
                        best_key = key
                        best_w = w_try
                        best_diag = {
                            "oof_bal_acc": oof_b,
                            "oof_auroc": oof_a,
                            "loco_bal_acc": loco_b,
                            "loco_auroc": loco_a,
                            "cv_objective_bal": obj_b,
                            "cv_objective_auc": obj_a,
                        }
                        improved = True

    weights_dict = {n: round(float(best_w[i]), 5) for i, n in enumerate(names)}
    return {
        "alpha_canon": best_alpha,
        "weights": weights_dict,
        "weights_array": best_w,
        "paradigm_names": names,
        "cv_metrics": best_diag,
    }


def run_training_and_evaluation(
    backbone_name: str = "owkin/phikon-v2",
    second_backbone_name: str = "hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms",
    third_backbone_name: str = "histai/hibou-L",
    img_size: int = 256,
    canonical_strength: float = 0.50,
    spatial_mil_epochs: int = 6,
    head_lr: float = 4.0e-4,
    weight_decay: float = 0.03,
    ema_decay: float = 0.992,
    d4_views: int = 4,
    num_perturbed_views: int = 4,
    n_svd: int = 32,
    enable_post_selection_refit: bool = True,
    seed: int = 42,
    max_stage1_seconds: float = 2050.0,
    max_total_seconds: float = 3300.0,
) -> float:
    """
    Strictly Frozen 3-PFM Representation Bank + Multi-Paradigm Stacking Pipeline:
      Stage 1A: Fit training-only StainReferenceBank; extract & cache multi-layer
                D4 embeddings (raw + soft-canonical + 4 bounded stain-perturbed
                training views) and 46-dim Beer-Lambert OD descriptors across 3
                strictly frozen Pathology Foundation Models.
      Stage 1B: Train 5-Fold Slide-Grouped + 2-Fold LOCO CV across 4 complementary
                downstream paradigms (strictly fold-internal preprocessing):
                  1. Frozen-Backbone owkin/phikon-v2 Spatial-MIL Probe (live aug)
                  2. MultiPFMDomainInvariantMLP (DualPrototypeCosineHead + twin cons.)
                  3. CenterDebiasedLogisticProbe (inter-center 1D projection)
                  4. LightGBM & XGBoost GBDT learners (fold SVD + prototypes + OD)
      Stage 1C: Jointly optimize alpha_canon in [0.0, 0.5] and bounded simplex
                logit weights strictly on 0.60 * LOCO_BalAcc + 0.40 * OOF_BalAcc
                inside train.csv (zero validation.csv leakage, fixed 0.5 boundary).
      Stage 1D: Evaluate locked pre-refit ensemble on untouched validation.csv via
                ./input/metric.py and lock Final Validation Score.
      Stage 2:  Refit StainReferenceBank and 5-fold bagged downstream paradigms
                on full_train_val_df (time-guarded under 55m ceiling), blend
                pre-refit/refit test logits 0.35/0.65 -> sigmoid, and verify
                ./submission/submission.csv.
    """
    run_start = time.time()
    seed_everything(seed)

    binary_metrics_fn = load_official_binary_metrics(INPUT_DIR)
    rt = detect_runtime_config()
    device: torch.device = rt["device"]

    print(
        f"Start Tier-3 Frozen 3-PFM Multi-Paradigm Pipeline | "
        f"Device: {rt['gpu_name']} ({rt['vram_gb']}GB, amp={str(rt['amp_dtype']).split('.')[-1]}) | "
        f"extract_bs={rt['extract_batch_size']} probe_bs={rt['probe_batch_size']} "
        f"d4_views={d4_views} canon_strength={canonical_strength}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Stage 1A: Leakage-Free Splits, Training-Only StainBank & 3-PFM Caching
    # -------------------------------------------------------------------------
    bundle = prepare_data_and_features(
        img_size=img_size,
        batch_size=rt["batch_size"],
        eval_batch_size=rt["eval_batch_size"],
        eval_tta_views=1,
        canonical_strength=canonical_strength,
        return_mode="dual",
        force_refit_bank=True,
        seed=seed,
    )
    train_df: pd.DataFrame = bundle["train_df"]
    val_df: pd.DataFrame = bundle["val_df"]
    test_df: pd.DataFrame = bundle["test_df"]
    sample_submission_df: pd.DataFrame = bundle["sample_submission_df"]
    full_train_val_df: pd.DataFrame = bundle["full_train_val_df"]
    loco_fold_indices: List[Tuple[str, np.ndarray, np.ndarray]] = bundle[
        "loco_fold_indices"
    ]
    n_folds: int = int(bundle["n_folds"])
    train_stain_bank: StainReferenceBank = bundle["stain_bank"]
    del bundle
    release_memory()

    pfm_specs = [
        ("phikon_v2", backbone_name),
        ("kaiko_vitl_reg4", second_backbone_name),
        ("hibou_l", third_backbone_name),
    ]
    pfm_bank = _extract_and_cache_pfm_bank(
        pfm_specs=pfm_specs,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stain_bank=train_stain_bank,
        rt=rt,
        canonical_strength=canonical_strength,
        d4_views=d4_views,
        num_perturbed_views=num_perturbed_views,
        seed=seed,
    )

    # -------------------------------------------------------------------------
    # Stage 1B: 5-Fold Slide-Grouped + 2-Fold LOCO CV Across Downstream Paradigms
    # -------------------------------------------------------------------------
    paradigm_outputs: Dict[str, Dict[str, Any]] = {}

    t_mlp = time.time()
    paradigm_outputs["multi_pfm_mlp"] = _run_multi_pfm_mlp_cv(
        pfm_bank=pfm_bank,
        train_df=train_df,
        loco_fold_indices=loco_fold_indices,
        n_folds=n_folds,
        device=device,
        seed=seed,
    )
    print(f"[multi_pfm_mlp] CV complete in {time.time() - t_mlp:.1f}s", flush=True)

    t_lr = time.time()
    paradigm_outputs["center_debiased_lr"] = _run_center_debiased_lr_cv(
        pfm_bank=pfm_bank,
        train_df=train_df,
        loco_fold_indices=loco_fold_indices,
        n_folds=n_folds,
        seed=seed,
    )
    print(f"[center_debiased_lr] CV complete in {time.time() - t_lr:.1f}s", flush=True)

    t_gbdt = time.time()
    lgb_cv, xgb_cv = _run_gbdt_cv(
        pfm_bank=pfm_bank,
        train_df=train_df,
        loco_fold_indices=loco_fold_indices,
        n_folds=n_folds,
        n_svd=n_svd,
        seed=seed,
    )
    paradigm_outputs["lgbm_gbdt"] = lgb_cv
    paradigm_outputs["xgb_gbdt"] = xgb_cv
    print(f"[lgbm_gbdt + xgb_gbdt] CV complete in {time.time() - t_gbdt:.1f}s", flush=True)

    t_mil = time.time()
    paradigm_outputs["spatial_mil"] = _run_spatial_mil_cv(
        backbone_name=backbone_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        loco_fold_indices=loco_fold_indices,
        n_folds=n_folds,
        stain_bank=train_stain_bank,
        rt=rt,
        img_size=img_size,
        canonical_strength=canonical_strength,
        num_epochs=spatial_mil_epochs,
        tta_views=d4_views,
        head_lr=head_lr,
        weight_decay=weight_decay,
        ema_decay=ema_decay,
        seed=seed,
        deadline=run_start + max_stage1_seconds,
    )
    print(f"[spatial_mil] CV + Dual-Stain TTA complete in {time.time() - t_mil:.1f}s", flush=True)

    # -------------------------------------------------------------------------
    # Stage 1C: Strictly Train-Internal CV Stacking (alpha_canon & Simplex Weights)
    # -------------------------------------------------------------------------
    y_train = train_df["label"].astype(np.int64).values
    cv_stack = _optimize_cv_ensemble_weights_and_alpha(
        paradigm_outputs=paradigm_outputs,
        y_train=y_train,
        alpha_grid=(0.0, 0.10, 0.20, 0.30, 0.40, 0.50),
        max_single_weight=0.65,
        n_random_simplex=4000,
        seed=seed,
    )
    best_alpha = float(cv_stack["alpha_canon"])
    best_weights: Dict[str, float] = cv_stack["weights"]

    print(
        f"CV Stacking Locked (train.csv ONLY) | alpha_canon={best_alpha:.2f} | "
        f"weights={best_weights} | "
        f"LOCO bal={cv_stack['cv_metrics']['loco_bal_acc']:.5f}/auc={cv_stack['cv_metrics']['loco_auroc']:.5f} | "
        f"OOF bal={cv_stack['cv_metrics']['oof_bal_acc']:.5f}/auc={cv_stack['cv_metrics']['oof_auroc']:.5f}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Stage 1D: Official Pre-Refit Validation on Untouched validation.csv
    # -------------------------------------------------------------------------
    ref_val_ids = val_df["sample_id"].astype(str).tolist()
    ref_test_ids = test_df["sample_id"].astype(str).tolist()

    per_model_diagnostics: Dict[str, Dict[str, Any]] = {}
    ens_val_logits = np.zeros(len(val_df), dtype=np.float64)
    ens_test_pre_logits = np.zeros(len(test_df), dtype=np.float64)

    for name, out in paradigm_outputs.items():
        w = float(best_weights[name])
        oof_lg = (1.0 - best_alpha) * out["oof_raw"] + best_alpha * out["oof_canon"]
        loco_lg = (1.0 - best_alpha) * out["loco_raw"] + best_alpha * out["loco_canon"]
        val_lg = (1.0 - best_alpha) * out["val_raw"] + best_alpha * out["val_canon"]
        te_lg = (1.0 - best_alpha) * out["test_raw"] + best_alpha * out["test_canon"]

        val_probs_m = logit_to_prob(val_lg)
        val_eval_m = evaluate_validation_predictions(
            val_df, ref_val_ids, val_probs_m, binary_metrics_fn
        )

        oof_p = logit_to_prob(oof_lg)
        loco_p = logit_to_prob(loco_lg)
        oof_bal = float(balanced_accuracy_score(y_train, oof_p >= 0.5))
        oof_auc = float(roc_auc_score(y_train, oof_p))
        loco_bal = float(balanced_accuracy_score(y_train, loco_p >= 0.5))
        loco_auc = float(roc_auc_score(y_train, loco_p))

        per_model_diagnostics[name] = {
            "weight": w,
            "oof_bal_acc": round(oof_bal, 5),
            "oof_auroc": round(oof_auc, 5),
            "loco_bal_acc": round(loco_bal, 5),
            "loco_auroc": round(loco_auc, 5),
            "val_eval": val_eval_m,
        }

        c_str = ", ".join(
            f"{cid}:bal={m['balanced_accuracy']:.4f}/auc={m['auroc']:.4f}"
            for cid, m in val_eval_m["per_center"].items()
        )
        print(
            f"[{name:18s}] w={w:.3f} | CV(LOCO={loco_bal:.4f}/{loco_auc:.4f}, OOF={oof_bal:.4f}/{oof_auc:.4f}) | "
            f"Official Val bal={val_eval_m['balanced_accuracy']:.5f} auc={val_eval_m['auroc']:.5f} "
            f"rec(T={val_eval_m['tumor_recall']:.4f}, N={val_eval_m['normal_recall']:.4f}) | {c_str}",
            flush=True,
        )

        ens_val_logits += w * val_lg
        ens_test_pre_logits += w * te_lg

    ens_val_probs = logit_to_prob(ens_val_logits)
    final_val_eval = evaluate_validation_predictions(
        val_df, ref_val_ids, ens_val_probs, binary_metrics_fn
    )
    final_validation_score = float(final_val_eval["balanced_accuracy"])
    ens_test_pre_probs = logit_to_prob(ens_test_pre_logits)

    val_pred_dict: Dict[str, Any] = {
        "sample_id": ref_val_ids,
        "label": val_df["label"].astype(int).values,
        "center_id": val_df["center_id"].astype(str).values,
        "slide_id": val_df["slide_id"].astype(str).values,
        "tumor_probability": ens_val_probs,
    }
    test_pre_dict: Dict[str, Any] = {
        "sample_id": ref_test_ids,
        "tumor_probability": ens_test_pre_probs,
    }
    for name, out in paradigm_outputs.items():
        val_lg = (1.0 - best_alpha) * out["val_raw"] + best_alpha * out["val_canon"]
        te_lg = (1.0 - best_alpha) * out["test_raw"] + best_alpha * out["test_canon"]
        val_pred_dict[f"prob_{name}"] = logit_to_prob(val_lg)
        test_pre_dict[f"prob_{name}"] = logit_to_prob(te_lg)

    pd.DataFrame(val_pred_dict).to_csv(
        WORKING_DIR / "val_predictions_tta.csv", index=False
    )
    pd.DataFrame(test_pre_dict).to_csv(
        WORKING_DIR / "test_predictions_pre_refit.csv", index=False
    )

    ens_c_str = ", ".join(
        f"{cid}:bal={m['balanced_accuracy']:.4f}/auc={m['auroc']:.4f}"
        for cid, m in final_val_eval["per_center"].items()
    )
    print(
        f"Ensemble Pre-Refit Official Val (LOCKED) | "
        f"bal_acc={final_validation_score:.5f} auroc={final_val_eval['auroc']:.5f} "
        f"acc={final_val_eval['accuracy']:.5f} | "
        f"recall(T={final_val_eval['tumor_recall']:.4f}, N={final_val_eval['normal_recall']:.4f}) "
        f"pos_rate={final_val_eval['pred_pos_rate']:.4f} | {ens_c_str}",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Stage 2: Clean Train+Val 5-Fold Bagged Refit (Time-Guarded under 55m)
    # -------------------------------------------------------------------------
    refit_seed = seed + 100
    refit_status_dict: Dict[str, Dict[str, Any]] = {}
    ens_test_refit_logits = np.zeros(len(test_df), dtype=np.float64)
    any_refit_completed = False

    if enable_post_selection_refit and (time.time() - run_start) + 210.0 < max_total_seconds:
        seed_everything(refit_seed)
        print(
            f"Stage 2 Refit | Fitting Train+Val StainReferenceBank on {len(full_train_val_df)} "
            f"samples ({full_train_val_df['slide_id'].nunique()} slides)...",
            flush=True,
        )
        refit_bank = StainReferenceBank().fit(
            full_train_val_df,
            input_dir=INPUT_DIR,
            max_patches_per_slide=36,
            seed=refit_seed,
        )
        refit_bank.save(WORKING_DIR / "stain_reference_bank_trainval.json")

        # Leakage-free 5-fold Slide-Grouped split on full_train_val_df
        strat_full = (
            full_train_val_df["center_id"].astype(str)
            + "_"
            + full_train_val_df["label"].astype(str)
        )
        sgkf_full = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=refit_seed)
        full_folds = np.zeros(len(full_train_val_df), dtype=np.int64)
        for fi, (_, v_idx) in enumerate(
            sgkf_full.split(
                full_train_val_df, strat_full, groups=full_train_val_df["group_id"]
            )
        ):
            full_folds[v_idx] = fi

        need_refit_canon = bool(
            best_alpha > 0.0 and (time.time() - run_start) + 450.0 < max_total_seconds
        )
        refit_bank_feats = _extract_refit_val_pert_and_canon(
            pfm_bank=pfm_bank,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            refit_bank=refit_bank,
            rt=rt,
            canonical_strength=canonical_strength,
            d4_views=d4_views,
            num_perturbed_views=num_perturbed_views,
            need_refit_canon=need_refit_canon,
            refit_seed=refit_seed,
        )

        # Fast tabular/embedding paradigms refit first (all 4 complete in ~2 min)
        refit_mlp = _run_multi_pfm_mlp_refit(
            refit_bank_feats, full_train_val_df, full_folds, device, refit_seed
        )
        refit_lr = _run_center_debiased_lr_refit(
            refit_bank_feats,
            full_train_val_df,
            full_folds,
            best_C=float(paradigm_outputs["center_debiased_lr"]["best_C"]),
            best_beta=float(paradigm_outputs["center_debiased_lr"]["best_beta"]),
            refit_seed=refit_seed,
        )
        refit_lgb, refit_xgb = _run_gbdt_refit(
            refit_bank_feats, full_train_val_df, full_folds, n_svd=n_svd, refit_seed=refit_seed
        )

        del refit_bank_feats
        release_memory()

        # Time-guarded Spatial-MIL Probe refit (only if weighted > 0)
        if float(best_weights.get("spatial_mil", 0.0)) > 0.0:
            refit_mil = _run_spatial_mil_refit(
                backbone_name=backbone_name,
                full_train_val_df=full_train_val_df,
                test_df=test_df,
                full_folds=full_folds,
                refit_bank=refit_bank,
                rt=rt,
                img_size=img_size,
                canonical_strength=canonical_strength,
                num_epochs=int(
                    paradigm_outputs["spatial_mil"].get(
                        "median_best_epoch", spatial_mil_epochs
                    )
                ),
                tta_views=d4_views,
                head_lr=head_lr,
                weight_decay=weight_decay,
                ema_decay=ema_decay,
                refit_seed=refit_seed,
                run_start=run_start,
                max_total_seconds=max_total_seconds,
            )
        else:
            refit_mil = {
                "completed": False,
                "n_refit_folds": 0,
                "test_raw": None,
                "test_canon": None,
            }

        refit_map = {
            "multi_pfm_mlp": refit_mlp,
            "center_debiased_lr": refit_lr,
            "lgbm_gbdt": refit_lgb,
            "xgb_gbdt": refit_xgb,
            "spatial_mil": refit_mil,
        }

        for name, out in paradigm_outputs.items():
            w = float(best_weights[name])
            pre_lg = (1.0 - best_alpha) * out["test_raw"] + best_alpha * out["test_canon"]
            r_out = refit_map[name]
            if r_out["completed"] and r_out["test_raw"] is not None:
                any_refit_completed = True
                r_lg = (1.0 - best_alpha) * r_out["test_raw"] + best_alpha * r_out["test_canon"]
                corr = float(np.corrcoef(logit_to_prob(pre_lg), logit_to_prob(r_lg))[0, 1])
                refit_status_dict[name] = {
                    "completed": True,
                    "pre_vs_refit_corr": round(corr, 5),
                }
                ens_test_refit_logits += w * r_lg
                print(
                    f"[Refit {name:18s}] Complete | Pre-vs-Refit Test Corr={corr:.4f}",
                    flush=True,
                )
            else:
                refit_status_dict[name] = {
                    "completed": False,
                    "pre_vs_refit_corr": None,
                }
                ens_test_refit_logits += w * pre_lg
    else:
        for name in paradigm_outputs:
            refit_status_dict[name] = {"completed": False, "pre_vs_refit_corr": None}
        ens_test_refit_logits = ens_test_pre_logits.copy()

    # Final blend: 0.35 pre-refit + 0.65 refit in logit space -> sigmoid
    if any_refit_completed:
        final_test_logits = 0.35 * ens_test_pre_logits + 0.65 * ens_test_refit_logits
    else:
        final_test_logits = ens_test_pre_logits
    final_test_probs = logit_to_prob(final_test_logits)

    # -------------------------------------------------------------------------
    # Stage 3: Write Verified Submission & Persist Auditable Summary
    # -------------------------------------------------------------------------
    sub_df = write_and_verify_submission(
        sample_submission_df=sample_submission_df,
        test_ids=ref_test_ids,
        test_probs=final_test_probs,
        submission_path=SUBMISSION_DIR / "submission.csv",
    )

    summary = {
        "final_validation_score_pre_refit": final_validation_score,
        "final_val_eval_tta": final_val_eval,
        "cv_stacking": {
            "alpha_canon": best_alpha,
            "weights": best_weights,
            "cv_metrics": cv_stack["cv_metrics"],
        },
        "pfm_provenances": pfm_bank["provenances"],
        "paradigms": per_model_diagnostics,
        "refit_status": refit_status_dict,
        "refit_blend": {"pre_refit_weight": 0.35, "refit_weight": 0.65}
        if any_refit_completed
        else {"pre_refit_weight": 1.0, "refit_weight": 0.0},
        "d4_views": d4_views,
        "canonical_strength": canonical_strength,
        "runtime": {k: str(v) for k, v in rt.items()},
        "submission_stats": {
            "path": str(SUBMISSION_DIR / "submission.csv"),
            "n_rows": int(len(sub_df)),
            "mean_prob": float(sub_df["tumor_probability"].mean()),
            "std_prob": float(sub_df["tumor_probability"].std()),
            "pred_pos_rate_0_5": float((sub_df["tumor_probability"] >= 0.5).mean()),
        },
        "total_runtime_seconds": round(time.time() - run_start, 1),
    }
    with open(WORKING_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    refit_tag = ", ".join(
        f"{k}={'refit' if v['completed'] else 'pre'}"
        for k, v in refit_status_dict.items()
    )
    print(
        f"Submission Verified -> {SUBMISSION_DIR / 'submission.csv'} | "
        f"Stack({refit_tag}) | rows={len(sub_df)} "
        f"mean={sub_df['tumor_probability'].mean():.4f} "
        f"std={sub_df['tumor_probability'].std():.4f} "
        f"pos_rate={(sub_df['tumor_probability'] >= 0.5).mean():.4f} | "
        f"Total Time: {(time.time() - run_start)/60.0:.1f}m",
        flush=True,
    )
    print(f"Final Validation Score: {final_validation_score}", flush=True)
    return final_validation_score


if __name__ == "__main__":
    run_training_and_evaluation()
