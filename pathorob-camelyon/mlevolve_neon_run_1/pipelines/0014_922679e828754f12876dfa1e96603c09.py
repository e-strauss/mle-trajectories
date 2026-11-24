"""
End-to-End Histopathology Patch Classification Pipeline
-------------------------------------------------------
Combines:
  1. Physics-Informed Optical Stain Standardization, Virtual-Center Synthesis &
     Dual-Stream (Standardized RGB + Beer-Lambert H&E OD) Data Pipeline.
  2. BatchNorm-Free Dual-Stream Physics-Guided Vision Transformer (DualStreamStainViT)
     with Gated MIL Spatial Attention, Cross-Modal Gated Fusion, Multi-Sample Dropout,
     Composite CDAN + V-REx Domain-Generalizing Loss, LLRD AdamW, and Model EMA.
  3. Mixed-Precision Domain-Scheduled Training, Official Pre-Refit 8-View D4 TTA
     Validation (./input/metric.py), Time-Guarded Post-Selection Train+Val Refit,
     and Verified Submission Output (./submission/submission.csv).
"""

import copy
import gc
import importlib.util
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
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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

# Ordered pretrained backbone candidates with provenance documentation.
# Combines histopathology-pretrained DINOv2 ViT-Large (TCGA / PANCAN-400M) with
# BatchNorm-free hierarchical OpenCLIP ConvNeXt-Large (local 7x7 depthwise inductive bias)
# for genuine architecture and pretraining corpus diversity across unseen medical centers.
PRETRAINED_BACKBONE_CANDIDATES: Tuple[Dict[str, str], ...] = (
    {
        "name": "hf-hub:1aurent/vit_large_patch14_224.kaiko_ai_towards_large_pathology_fms",
        "pretraining_data": "TCGA-only histopathology WSI patches (Kaiko.ai DINOv2 ViT-L/14; zero Camelyon/lymph-node overlap)",
        "family": "vit",
        "loader": "timm",
    },
    {
        "name": "hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms",
        "pretraining_data": "TCGA-only histopathology WSI patches (Kaiko.ai DINOv2 ViT-L/14 reg4; zero Camelyon/lymph-node overlap)",
        "family": "vit",
        "loader": "timm",
    },
    {
        "name": "owkin/phikon-v2",
        "pretraining_data": "PANCAN-400M histopathology tiles (Apache-2.0 Owkin DINOv2 ViT-L/16)",
        "family": "vit",
        "loader": "hf_dinov2",
    },
    {
        "name": "hf-hub:1aurent/vit_base_patch8_224.kaiko_ai_towards_large_pathology_fms",
        "pretraining_data": "TCGA-only histopathology WSI patches (Kaiko.ai DINOv2 ViT-B/8; zero Camelyon/lymph-node overlap)",
        "family": "vit",
        "loader": "timm",
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
        conc_scale_range: Tuple[float, float] = (0.65, 1.40),
        conc_bias_range: Tuple[float, float] = (-0.08, 0.08),
        matrix_jitter_std: float = 0.05,
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
    Physics-Informed Paired-View Unseen-Center Stain Augmentation pipeline (without
    destructive post-augmentation per-patch Reinhard LAB normalization):
      1. Shared exact D4 symmetry + conservative scanner-magnification scale jitter.
      2. Clean unperturbed H&E anchor tensors (rgb_anchor, od_anchor) computed from
         the geometrically aligned patch for EMA teacher consistency guidance.
      3. With 85% probability (is_perturbed=True), unseen-hospital color/stain simulation:
         - Moderated Virtual-Center LAB transfer (inter-center extrapolation)
         - Widened Beer-Lambert H&E stain-matrix rotation & concentration perturbation
         - Scanner white-balance gain, nonlinear gamma, optical focus, and sensor noise
      4. Remaining 15% clean-stain anchor pass (is_perturbed=False) reusing
         (rgb_anchor, od_anchor) directly to anchor fixed 0.5-threshold calibration
         without redundant CPU OD deconvolution.
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

    def _scanner_scale_crop(
        self, img: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Simulate subtle micron-per-pixel scanner magnification differences."""
        h, w = img.shape[:2]
        if rng.random() < 0.35:
            scale = float(rng.uniform(0.94, 1.0))
            ch, cw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
            top = int(rng.integers(0, h - ch + 1))
            left = int(rng.integers(0, w - cw + 1))
            crop = img[top : top + ch, left : left + cw]
            interp = cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC
            return cv2.resize(crop, (w, h), interpolation=interp)
        return img

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
                wb_gain = rng.uniform(0.90, 1.10, size=(1, 1, 3)).astype(np.float32)
                out = np.clip(out * wb_gain, 0.0, 255.0)

            if rng.random() < 0.60:
                gamma = float(rng.uniform(0.80, 1.25))
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        view_idx = int(rng.integers(0, 8))
        clean_img = apply_d4_view(rgb_uint8, view_idx)
        clean_img = self._scanner_scale_crop(clean_img, rng)

        rgb_anchor, od_anchor = rgb_and_od_to_tensors(
            clean_img,
            hed=self.hed,
            od_mean=self.stain_bank.od_channel_mean,
            od_std=self.stain_bank.od_channel_std,
            img_size=self.img_size,
        )

        if rng.random() < 0.85:
            img = clean_img
            if rng.random() < 0.45:
                virt_m, virt_s = self.stain_bank.sample_virtual_center_target(rng)
                img = self.canonical_normalizer.apply_with_target(
                    img,
                    target_mean=virt_m,
                    target_std=virt_s,
                    strength=float(rng.uniform(0.30, 0.70)),
                )

            if rng.random() < 0.85:
                img = self.hed.perturb_stain(img, rng)

            img = self._scanner_optics_jitter(img, rng, apply_color_optics=True)
            rgb_t, od_t = rgb_and_od_to_tensors(
                img,
                hed=self.hed,
                od_mean=self.stain_bank.od_channel_mean,
                od_std=self.stain_bank.od_channel_std,
                img_size=self.img_size,
            )
            is_perturbed = True
        else:
            rgb_t, od_t = rgb_anchor, od_anchor
            is_perturbed = False

        return rgb_t, od_t, rgb_anchor, od_anchor, is_perturbed


class HistopathologyEvalTransform:
    """
    Deterministic Evaluation / Inference Transform (100% IDENTICAL for Val and Test).
    Preserves raw H&E optical density cues without per-patch Reinhard normalization,
    computes RGB and OD tensors once per image, and generates exact D4 TTA views
    via fast tensor flip/rot90.
    """

    def __init__(
        self,
        stain_bank: StainReferenceBank,
        img_size: int = 256,
        canonical_strength: float = 0.75,
        tta_views: int = 1,
    ) -> None:
        if tta_views not in (1, 4, 8):
            raise ValueError(f"tta_views must be 1, 4, or 8; got {tta_views}")
        self.stain_bank = stain_bank
        self.img_size = int(img_size)
        self.canonical_strength = float(canonical_strength)
        self.tta_views = int(tta_views)
        self.hed = BeerLambertHED()
        self.canonical_normalizer = TissueMaskedStainNormalizer(
            target_lab_mean=stain_bank.canonical_lab_mean,
            target_lab_std=stain_bank.canonical_lab_std,
            strength=canonical_strength,
        )

    def __call__(self, rgb_uint8: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_base, od_base = rgb_and_od_to_tensors(
            rgb_uint8,
            hed=self.hed,
            od_mean=self.stain_bank.od_channel_mean,
            od_std=self.stain_bank.od_channel_std,
            img_size=self.img_size,
        )

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
        rgb_anchor_t: Optional[torch.Tensor] = None
        od_anchor_t: Optional[torch.Tensor] = None
        is_perturbed: bool = False

        if self.is_train:
            rng = self._get_rng(idx)
            tf_out = self.transform(rgb_u8, rng)
            if len(tf_out) == 5:
                rgb_t, od_t, rgb_anchor_t, od_anchor_t, is_perturbed = tf_out
            else:
                rgb_t, od_t = tf_out
                rgb_anchor_t, od_anchor_t = rgb_t, od_t
        else:
            rgb_t, od_t = self.transform(rgb_u8)

        item: Dict = {"sample_id": self.sample_ids[idx]}
        if self.return_mode in ("dual", "rgb"):
            item["rgb"] = rgb_t
            if self.is_train and rgb_anchor_t is not None:
                item["rgb_anchor"] = rgb_anchor_t
        if self.return_mode in ("dual", "od"):
            item["od"] = od_t
            if self.is_train and od_anchor_t is not None:
                item["od_anchor"] = od_anchor_t
        if self.is_train:
            item["is_perturbed"] = torch.tensor(is_perturbed, dtype=torch.bool)
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

    full_train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    return {
        "train_df": train_with_folds,
        "val_df": val_df.reset_index(drop=True),
        "full_train_val_df": full_train_val_df,
        "loco_splits": loco_splits,
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
    (e.g., owkin/phikon-v2) supporting native 256px interpolate_pos_encoding,
    multi-layer hidden-state tapping on layers (17, 19, 21, 23), and gradient checkpointing.
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_path_rate: float = 0.30,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoImageProcessor, AutoModel

        cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(cfg, "drop_path_rate"):
            cfg.drop_path_rate = float(drop_path_rate)

        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, config=cfg)
        else:
            self.model = AutoModel.from_config(cfg)

        self.is_hf_dinov2: bool = True
        self.dynamic_img_size: bool = True
        self.patch_size: int = int(getattr(cfg, "patch_size", 16))
        self.num_features: int = int(getattr(cfg, "hidden_size", 1024))
        self.num_hidden_layers: int = int(getattr(cfg, "num_hidden_layers", 24))
        n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.num_prefix_tokens: int = 1 + n_reg

        mean = IMAGENET_MEAN.tolist()
        std = IMAGENET_STD.tolist()
        try:
            proc = AutoImageProcessor.from_pretrained(model_name)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(
            pixel_values=x,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        return out.last_hidden_state

    def forward_features_with_taps(
        self,
        x: torch.Tensor,
        tap_indices: Tuple[int, ...],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        out = self.model(
            pixel_values=x,
            interpolate_pos_encoding=True,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states or ()
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
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Instantiate a pretrained Vision Transformer / ConvNeXt / SwinV2 backbone via timm
    or HuggingFace AutoModel (for owkin/phikon-v2) with family-preserving automatic
    fallback and complete provenance metadata.
    """
    candidates: List[Dict[str, str]] = []
    preferred_family: Optional[str] = None
    if preferred_name is not None:
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
            loader = "hf_dinov2" if "phikon" in low else "timm"
            candidates.append(
                {
                    "name": preferred_name,
                    "pretraining_data": "Specified pretrained weights",
                    "family": family,
                    "loader": loader,
                }
            )
    if preferred_family is not None:
        for c in PRETRAINED_BACKBONE_CANDIDATES:
            if c["family"] == preferred_family and all(
                existing["name"] != c["name"] for existing in candidates
            ):
                candidates.append(c)
    for c in PRETRAINED_BACKBONE_CANDIDATES:
        if all(existing["name"] != c["name"] for existing in candidates):
            candidates.append(c)

    last_error: Optional[Exception] = None
    for spec in candidates:
        name = spec["name"]
        if spec.get("loader") == "hf_dinov2":
            try:
                backbone = _HFDinov2BackboneWrapper(
                    model_name=name,
                    pretrained=pretrained,
                    drop_path_rate=drop_path_rate,
                )
                meta = {
                    "backbone_name": name,
                    "backbone_family": spec["family"],
                    "pretraining_data": spec["pretraining_data"],
                    "pretrained": pretrained,
                    "timm_version": timm.__version__,
                    "torch_version": torch.__version__,
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

        try:
            backbone = timm.create_model(name, **kwargs)
            meta = {
                "backbone_name": name,
                "backbone_family": spec["family"],
                "pretraining_data": spec["pretraining_data"],
                "pretrained": pretrained,
                "timm_version": timm.__version__,
                "torch_version": torch.__version__,
            }
            return backbone, meta
        except TypeError:
            kwargs.pop("dynamic_img_size", None)
            try:
                backbone = timm.create_model(name, **kwargs)
                meta = {
                    "backbone_name": name,
                    "backbone_family": spec["family"],
                    "pretraining_data": spec["pretraining_data"],
                    "pretrained": pretrained,
                    "timm_version": timm.__version__,
                    "torch_version": torch.__version__,
                }
                return backbone, meta
            except Exception as exc:
                last_error = exc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to instantiate any pretrained backbone candidate. Last error: {last_error}"
    )


def _infer_backbone_input_size(backbone: nn.Module, target_size: int = 256) -> int:
    """
    Determine exact spatial resolution compatible with the backbone's patch grid.
    For patch_size=14 (DINOv2), selects 252 (18x18 patches, preserving 98.4% of 256px).
    For patch_size=16 (Phikon-v2), selects native 256 (16x16 patches).
    """
    if getattr(backbone, "is_hf_dinov2", False):
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
        backbone_name: Optional[str] = "hf-hub:1aurent/vit_large_patch14_224.kaiko_ai_towards_large_pathology_fms",
        pretrained: bool = True,
        img_size: int = 256,
        od_dim: int = 256,
        fusion_dim: int = 512,
        drop_path_rate: float = 0.30,
        head_dropout: float = 0.30,
        num_centers: int = 2,
        use_od_stream: bool = True,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.use_od_stream = bool(use_od_stream)
        self.num_centers = int(num_centers)

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
        # to the backbone's native pretrained_cfg mean/std (e.g., 0.5/0.5/0.5 for Kaiko
        # pathology DINOv2, OpenCLIP stats for ConvNeXt-Large, exact no-op for ImageNet).
        pt_cfg = dict(
            getattr(self.rgb_backbone, "pretrained_cfg", None)
            or getattr(self.rgb_backbone, "default_cfg", None)
            or {}
        )
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

        self.classifier = MultiSampleDropoutHead(
            in_dim=fusion_dim, num_samples=5, dropout_rate=head_dropout
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
        self.provenance["stage3_dim"] = int(self.stage3_dim)
        self.provenance["rgb_remap_active"] = bool(self._needs_rgb_remap)
        self.provenance["input_mean"] = [round(float(x), 6) for x in target_mean]
        self.provenance["input_std"] = [round(float(x), 6) for x in target_std]

    def _extract_rgb_representation(self, rgb: torch.Tensor) -> torch.Tensor:
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


class DomainGeneralizingHistopathLoss(nn.Module):
    """
    Composite loss designed specifically for unseen-center balanced accuracy at
    fixed decision threshold 0.5:
      1. Symmetric Label-Smoothed BCEWithLogitsLoss on fused tumor logits.
      2. Auxiliary OD-branch BCE loss for standalone nuclear morphology learning.
      3. V-REx Inter-Center Risk Extrapolation variance penalty across centers.
      4. Conditional Domain-Adversarial (CDAN) cross-entropy loss through GRL.
      5. EMA-guided cross-stain invariance consistency on stain-perturbed views:
         - Correctness-gated Bernoulli KL divergence at temperature 1.0.
         - L2-normalized cosine embedding alignment on fused representations.
    """

    def __init__(
        self,
        label_smoothing: float = 0.03,
        aux_od_weight: float = 0.15,
        vrex_weight: float = 0.25,
        cdan_weight: float = 0.12,
        logit_cons_weight: float = 0.50,
        emb_cons_weight: float = 0.25,
        min_samples_per_center: int = 2,
    ) -> None:
        super().__init__()
        self.label_smoothing = float(label_smoothing)
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
        cls_loss = per_sample_bce.mean()
        total_loss = cls_loss

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
                    center_risks.append(per_sample_bce[mask].mean())
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

        if cons_scale > 0.0 and (
            teacher_logits is not None or teacher_embedding is not None
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

                    p_t = torch.sigmoid(t_logits).clamp(1e-6, 1.0 - 1e-6)
                    p_s = torch.sigmoid(s_logits).clamp(1e-6, 1.0 - 1e-6)

                    kl = p_t * (torch.log(p_t) - torch.log(p_s)) + (1.0 - p_t) * (
                        torch.log(1.0 - p_t) - torch.log(1.0 - p_s)
                    )
                    kl = torch.clamp(kl, min=0.0)

                    teacher_correct = ((p_t >= 0.5) == (y_pert >= 0.5)).float()
                    teacher_true_conf = torch.where(y_pert >= 0.5, p_t, 1.0 - p_t)
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
    backbone_lr: float = 2e-5,
    head_lr: float = 3e-4,
    weight_decay: float = 0.03,
    layer_decay: float = 0.88,
) -> AdamW:
    """Construct AdamW optimizer with Layer-Wise Learning Rate Decay (LLRD)."""
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
    )

    param_groups: Dict[Tuple[float, float], List[nn.Parameter]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = (param.ndim <= 1) or any(
            kw in name.lower() for kw in no_decay_keywords
        )
        wd = 0.0 if is_no_decay else float(weight_decay)

        if name.startswith("rgb_backbone."):
            sub_name = name[len("rgb_backbone.") :]
            layer_id = _get_backbone_layer_id(
                sub_name, num_layers=num_layers, stage_depths=stage_depths
            )
            scale = layer_decay ** (num_layers - layer_id)
            lr = backbone_lr * scale
        else:
            lr = head_lr

        key = (round(lr, 12), wd)
        param_groups.setdefault(key, []).append(param)

    grouped_list = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in sorted(param_groups.items(), key=lambda kv: kv[0][0])
    ]
    return AdamW(grouped_list, betas=(0.9, 0.999), eps=1e-8)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.02,
) -> LambdaLR:
    """Linear warmup followed by smooth cosine annealing down to min_lr_ratio."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


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
            ema_params[k].mul_(eff_decay).add_(v.detach(), alpha=1.0 - eff_decay)

        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for k, v in model_buffers.items():
            ema_buffers[k].copy_(v.detach())


def build_model_and_optimization(
    backbone_name: str = "hf-hub:1aurent/vit_large_patch14_224.kaiko_ai_towards_large_pathology_fms",
    pretrained: bool = True,
    img_size: int = 256,
    num_centers: int = 2,
    use_od_stream: bool = True,
    backbone_lr: float = 2e-5,
    head_lr: float = 3e-4,
    weight_decay: float = 0.03,
    layer_decay: float = 0.88,
    total_steps: int = 2000,
    warmup_ratio: float = 0.10,
    ema_decay: float = 0.996,
    grad_checkpointing: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Instantiate the complete model, loss criterion, LLRD AdamW optimizer,
    cosine warmup scheduler, and ModelEMA wrapper ready for training/inference.
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
    ).to(device)

    criterion = DomainGeneralizingHistopathLoss(
        label_smoothing=0.03,
        aux_od_weight=0.15,
        vrex_weight=0.25,
        cdan_weight=0.12,
        logit_cons_weight=0.50,
        emb_cons_weight=0.25,
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
            grad_checkpointing = False
        elif vram_gb >= 22.0:
            batch_size, grad_accum = 32, 2
            eval_batch_size, tta_eval_batch_size = 64, 8
            grad_checkpointing = True
        elif vram_gb >= 12.0:
            batch_size, grad_accum = 16, 4
            eval_batch_size, tta_eval_batch_size = 32, 4
            grad_checkpointing = True
        else:
            batch_size, grad_accum = 8, 8
            eval_batch_size, tta_eval_batch_size = 16, 2
            grad_checkpointing = True
        gpu_name = str(props.name)
    else:
        vram_gb = 0.0
        amp_dtype = torch.float32
        batch_size, grad_accum = 16, 4
        eval_batch_size, tta_eval_batch_size = 32, 4
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
    cons_warmup_frac: float = 0.12,
    cons_max_lambda: float = 1.0,
) -> Tuple[Dict[str, float], int]:
    """
    Execute one training epoch with mixed precision, float32 loss stability,
    gradient accumulation, smooth CDAN/V-REx and EMA-guided cross-stain
    consistency schedules, gradient clipping, and EMA updates.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(train_loader)
    if num_batches == 0:
        raise RuntimeError("train_loader yielded 0 batches.")

    totals = {
        "loss": 0.0,
        "cls": 0.0,
        "aux": 0.0,
        "vrex": 0.0,
        "cdan": 0.0,
        "lcons": 0.0,
        "econs": 0.0,
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
        cons_scale = compute_domain_schedule(
            progress, warmup_frac=cons_warmup_frac, max_lambda=cons_max_lambda
        )
        use_domain = (domain_scale > 0.0) and (center_idx is not None)

        teacher_logits: Optional[torch.Tensor] = None
        teacher_embedding: Optional[torch.Tensor] = None
        is_perturbed: Optional[torch.Tensor] = None

        if (
            cons_scale > 0.0
            and "is_perturbed" in batch
            and "rgb_anchor" in batch
            and "od_anchor" in batch
        ):
            is_perturbed_cpu = batch["is_perturbed"].bool()
            is_perturbed = is_perturbed_cpu.to(device, non_blocking=True)
            if bool(is_perturbed_cpu.any().item()):
                rgb_anc = batch["rgb_anchor"][is_perturbed_cpu].to(
                    device, non_blocking=True
                )
                od_anc = batch["od_anchor"][is_perturbed_cpu].to(
                    device, non_blocking=True
                )
                ema.ema_model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=device.type, enabled=use_amp, dtype=amp_dtype
                    ):
                        t_out = ema.ema_model(rgb_anc, od_anc, return_domain=False)
                t_logits_masked = t_out["logits"].detach().float()
                t_emb_masked = t_out["embedding"].detach().float()

                teacher_logits = torch.zeros(
                    labels.shape[0], device=device, dtype=torch.float32
                )
                teacher_logits[is_perturbed] = t_logits_masked

                teacher_embedding = torch.zeros(
                    (labels.shape[0], t_emb_masked.shape[-1]),
                    device=device,
                    dtype=torch.float32,
                )
                teacher_embedding[is_perturbed] = t_emb_masked

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
            teacher_logits=teacher_logits,
            teacher_embedding=teacher_embedding,
            is_perturbed=is_perturbed,
            cons_scale=cons_scale,
        )

        scaled_loss = loss_dict["loss"] / float(window_size)
        scaler.scale(scaled_loss).backward()

        bsz = int(labels.shape[0])
        totals["loss"] += float(loss_dict["loss"].detach().item()) * bsz
        totals["cls"] += float(loss_dict["cls_loss"].item()) * bsz
        totals["aux"] += float(loss_dict["aux_loss"].item()) * bsz
        totals["vrex"] += float(loss_dict["vrex_loss"].item()) * bsz
        totals["cdan"] += float(loss_dict["cdan_loss"].item()) * bsz
        totals["lcons"] += float(loss_dict["logit_cons_loss"].item()) * bsz
        totals["econs"] += float(loss_dict["emb_cons_loss"].item()) * bsz
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


def _train_and_eval_single_stage1_model(
    model_tag: str,
    backbone_name: str,
    backbone_lr: float,
    layer_decay: float,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_stain_bank: StainReferenceBank,
    rt: Dict[str, Any],
    binary_metrics_fn: Callable,
    img_size: int,
    canonical_strength: float,
    num_epochs: int,
    min_epochs: int,
    early_stopping_patience: int,
    head_lr: float,
    weight_decay: float,
    ema_decay: float,
    tta_views: int,
    seed: int,
    stage1_deadline: float,
) -> Dict[str, Any]:
    """
    Execute Stage-1 training, checkpoint selection, and identical 8-view D4 TTA
    validation/test inference for one architecture on the fixed train/val split.
    """
    seed_everything(seed)
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]

    loaders = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stain_bank=train_stain_bank,
        img_size=img_size,
        batch_size=rt["batch_size"],
        eval_batch_size=rt["eval_batch_size"],
        return_mode="dual",
        eval_tta_views=1,
        canonical_strength=canonical_strength,
        use_balanced_sampler=True,
        include_train_meta=True,
        seed=seed,
    )
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]

    num_centers = int(train_df["center_id"].nunique())
    steps_per_epoch = math.ceil(len(train_loader) / float(rt["grad_accum"]))
    total_steps = num_epochs * steps_per_epoch

    components = build_model_and_optimization(
        backbone_name=backbone_name,
        pretrained=True,
        img_size=img_size,
        num_centers=num_centers,
        use_od_stream=True,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
        total_steps=total_steps,
        warmup_ratio=0.10,
        ema_decay=ema_decay,
        grad_checkpointing=rt["grad_checkpointing"],
        device=device,
    )
    model: DualStreamStainViT = components["model"]
    ema: ModelEMA = components["ema"]
    criterion: DomainGeneralizingHistopathLoss = components["criterion"]
    optimizer = components["optimizer"]
    scheduler = components["scheduler"]
    provenance = components["provenance"]

    scaler = torch.amp.GradScaler(
        device.type, enabled=(use_amp and amp_dtype == torch.float16)
    )

    print(
        f"Init [{model_tag}] | Device: {rt['gpu_name']} ({rt['vram_gb']}GB, "
        f"amp={str(amp_dtype).split('.')[-1]}) | "
        f"Backbone: {provenance['backbone_name']} ({provenance['total_params']/1e6:.1f}M params, "
        f"in={provenance['vit_input_size']}px, remap={provenance.get('rgb_remap_active', False)}) | "
        f"bb_lr={backbone_lr:.1e} llrd={layer_decay:.2f} | "
        f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)} | "
        f"EffBS={rt['batch_size']*rt['grad_accum']}"
    )

    best_ckpt_path = WORKING_DIR / f"best_val_model_{model_tag}.pt"
    best_key = (-1.0, -1.0)
    best_epoch = 1
    best_source = "ema"
    best_single_view_metrics: Dict[str, Any] = {}
    epochs_without_improve = 0
    global_step = 0
    epoch_durations: List[float] = []
    history: List[Dict[str, Any]] = []

    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.time()

        tr_stats, global_step = train_one_epoch(
            model=model,
            ema=ema,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            train_loader=train_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_accum=rt["grad_accum"],
            global_step=global_step,
            total_steps=total_steps,
        )

        (raw_ids, raw_probs), (ema_ids, ema_probs) = predict_dataset_dual(
            model, ema.ema_model, val_loader, device, use_amp, amp_dtype
        )
        raw_eval = evaluate_validation_predictions(
            val_df, raw_ids, raw_probs, binary_metrics_fn
        )
        ema_eval = evaluate_validation_predictions(
            val_df, ema_ids, ema_probs, binary_metrics_fn
        )

        if _selection_key(ema_eval) >= _selection_key(raw_eval):
            ep_source, ep_eval, ep_model = "ema", ema_eval, ema.ema_model
        else:
            ep_source, ep_eval, ep_model = "raw", raw_eval, model

        ep_key = _selection_key(ep_eval)
        improved = ep_key > best_key
        if improved:
            best_key = ep_key
            best_epoch = epoch
            best_source = ep_source
            best_single_view_metrics = ep_eval
            epochs_without_improve = 0
            save_checkpoint(
                ep_model,
                best_ckpt_path,
                metadata={
                    "model_tag": model_tag,
                    "epoch": epoch,
                    "source": ep_source,
                    "val_metrics_1view": ep_eval,
                    "provenance": provenance,
                },
            )
        else:
            epochs_without_improve += 1

        ep_sec = time.time() - ep_t0
        epoch_durations.append(ep_sec)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_stats["loss"], 5),
                "cls_loss": round(tr_stats["cls"], 5),
                "aux_loss": round(tr_stats["aux"], 5),
                "vrex_loss": round(tr_stats["vrex"], 5),
                "cdan_loss": round(tr_stats["cdan"], 5),
                "logit_cons_loss": round(tr_stats["lcons"], 5),
                "emb_cons_loss": round(tr_stats["econs"], 5),
                "val_bal_acc_raw": round(raw_eval["balanced_accuracy"], 5),
                "val_auroc_raw": round(raw_eval["auroc"], 5),
                "val_bal_acc_ema": round(ema_eval["balanced_accuracy"], 5),
                "val_auroc_ema": round(ema_eval["auroc"], 5),
                "selected": ep_source,
                "improved": improved,
                "seconds": round(ep_sec, 1),
            }
        )

        star = "*" if improved else " "
        print(
            f"[{model_tag}] Epoch {epoch:02d}/{num_epochs:02d} {star} | "
            f"loss={tr_stats['loss']:.4f} (cls={tr_stats['cls']:.4f} aux={tr_stats['aux']:.4f} "
            f"vrex={tr_stats['vrex']:.4f} cdan={tr_stats['cdan']:.4f} "
            f"lcons={tr_stats['lcons']:.4f} econs={tr_stats['econs']:.4f}) | "
            f"val_bal_acc raw={raw_eval['balanced_accuracy']:.4f} ema={ema_eval['balanced_accuracy']:.4f} | "
            f"val_auroc raw={raw_eval['auroc']:.4f} ema={ema_eval['auroc']:.4f} | {ep_sec:.1f}s"
        )

        if epoch >= min_epochs and epochs_without_improve >= early_stopping_patience:
            print(
                f"[{model_tag}] Early stopping at epoch {epoch} (best epoch {best_epoch}, "
                f"1-view bal_acc={best_single_view_metrics['balanced_accuracy']:.5f})"
            )
            break

        if time.time() > stage1_deadline and epoch < num_epochs:
            print(
                f"[{model_tag}] Stage 1 soft time guard reached after epoch {epoch}; "
                f"proceeding with best checkpoint."
            )
            break

    del loaders, train_loader, val_loader, optimizer, scheduler, ema, scaler
    release_memory()

    # Load best pre-refit checkpoint and run identical 8-view D4 TTA on Val and Test
    load_checkpoint(model, best_ckpt_path, device=device)

    tta_t0 = time.time()
    tta_loaders = create_dataloaders(
        train_df=train_df.iloc[: max(rt["batch_size"], 32)].copy(),
        val_df=val_df,
        test_df=test_df,
        stain_bank=train_stain_bank,
        img_size=img_size,
        batch_size=rt["batch_size"],
        eval_batch_size=rt["tta_eval_batch_size"],
        return_mode="dual",
        eval_tta_views=tta_views,
        canonical_strength=canonical_strength,
        use_balanced_sampler=False,
        include_train_meta=False,
        seed=seed,
    )

    val_ids_tta, val_probs_tta, _ = predict_dataset(
        model, tta_loaders["val_loader"], device, use_amp, amp_dtype
    )
    val_eval_tta = evaluate_validation_predictions(
        val_df, val_ids_tta, val_probs_tta, binary_metrics_fn
    )

    test_ids_pre, test_probs_pre, _ = predict_dataset(
        model, tta_loaders["test_loader"], device, use_amp, amp_dtype
    )
    tta_sec = time.time() - tta_t0

    pd.DataFrame(
        {
            "sample_id": val_ids_tta,
            "label": val_df["label"].astype(int).values,
            "center_id": val_df["center_id"].astype(str).values,
            "slide_id": val_df["slide_id"].astype(str).values,
            "tumor_probability": val_probs_tta,
        }
    ).to_csv(WORKING_DIR / f"val_predictions_tta_{model_tag}.csv", index=False)

    pd.DataFrame(
        {"sample_id": test_ids_pre, "tumor_probability": test_probs_pre}
    ).to_csv(WORKING_DIR / f"test_predictions_pre_refit_{model_tag}.csv", index=False)

    center_summary = ", ".join(
        f"{cid}:bal={m['balanced_accuracy']:.4f}/auc={m['auroc']:.4f}"
        for cid, m in val_eval_tta["per_center"].items()
    )
    print(
        f"[{model_tag}] Pre-Refit {tta_views}-View D4 TTA Official Val "
        f"(best_ep={best_epoch}, {best_source}) | "
        f"bal_acc={val_eval_tta['balanced_accuracy']:.5f} auroc={val_eval_tta['auroc']:.5f} "
        f"acc={val_eval_tta['accuracy']:.5f} | "
        f"recall(T={val_eval_tta['tumor_recall']:.4f}, N={val_eval_tta['normal_recall']:.4f}) "
        f"pos_rate={val_eval_tta['pred_pos_rate']:.4f} | {center_summary} | {tta_sec:.1f}s"
    )

    del tta_loaders, model
    release_memory()

    return {
        "model_tag": model_tag,
        "backbone_name": provenance["backbone_name"],
        "backbone_lr": backbone_lr,
        "layer_decay": layer_decay,
        "provenance": provenance,
        "best_ckpt_path": str(best_ckpt_path),
        "best_epoch": best_epoch,
        "best_source": best_source,
        "best_single_view_metrics": best_single_view_metrics,
        "val_ids_tta": val_ids_tta,
        "val_probs_tta": val_probs_tta,
        "val_eval_tta": val_eval_tta,
        "test_ids_pre": test_ids_pre,
        "test_probs_pre": test_probs_pre,
        "epoch_durations": epoch_durations,
        "tta_seconds": tta_sec,
        "history": history,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    }


def _run_single_model_refit(
    stage1_res: Dict[str, Any],
    full_train_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    refit_bank: StainReferenceBank,
    rt: Dict[str, Any],
    img_size: int,
    canonical_strength: float,
    num_epochs: int,
    head_lr: float,
    weight_decay: float,
    ema_decay: float,
    tta_views: int,
    refit_seed: int,
    run_start: float,
    max_total_seconds: float,
) -> Dict[str, Any]:
    """
    Time-guarded post-selection Train+Val refit for a single architecture with
    pre-launch and intra-epoch time safety checks.
    """
    model_tag = stage1_res["model_tag"]
    backbone_name = stage1_res["backbone_name"]
    backbone_lr = float(stage1_res["backbone_lr"])
    layer_decay = float(stage1_res["layer_decay"])
    best_source = stage1_res["best_source"]
    refit_epochs = max(4, min(int(stage1_res["best_epoch"]), num_epochs))

    avg_ep_sec = (
        float(np.mean(stage1_res["epoch_durations"]))
        if stage1_res["epoch_durations"]
        else 110.0
    )
    est_ep_refit = avg_ep_sec * (
        len(full_train_val_df) / max(1, stage1_res["n_train"])
    )
    est_test_tta = max(
        25.0,
        float(stage1_res["tta_seconds"])
        * (len(test_df) / max(1, len(test_df) + stage1_res["n_val"])),
    )
    est_refit_sec = est_ep_refit * refit_epochs + est_test_tta + 45.0
    elapsed_so_far = time.time() - run_start

    if elapsed_so_far + est_refit_sec >= max_total_seconds:
        print(
            f"[{model_tag}] Skipping post-selection refit "
            f"(elapsed={elapsed_so_far/60.0:.1f}m, est_refit={est_refit_sec/60.0:.1f}m, "
            f"limit={max_total_seconds/60.0:.1f}m); using pre-refit 8-view TTA logits."
        )
        return {
            "completed": False,
            "refit_epochs": 0,
            "refit_history": [],
            "test_probs_refit": None,
            "pre_vs_refit_corr": None,
        }

    seed_everything(refit_seed)
    device: torch.device = rt["device"]
    use_amp: bool = rt["use_amp"]
    amp_dtype: torch.dtype = rt["amp_dtype"]

    print(
        f"[{model_tag}] Post-Selection Refit | Training on Train+Val "
        f"({len(full_train_val_df)} samples, {full_train_val_df['slide_id'].nunique()} slides) "
        f"for {refit_epochs} epochs (est {est_refit_sec/60.0:.1f}m)..."
    )

    refit_loaders = create_dataloaders(
        train_df=full_train_val_df,
        val_df=None,
        test_df=test_df,
        stain_bank=refit_bank,
        img_size=img_size,
        batch_size=rt["batch_size"],
        eval_batch_size=rt["tta_eval_batch_size"],
        return_mode="dual",
        eval_tta_views=tta_views,
        canonical_strength=canonical_strength,
        use_balanced_sampler=True,
        include_train_meta=True,
        seed=refit_seed,
    )

    refit_train_loader = refit_loaders["train_loader"]
    refit_test_loader = refit_loaders["test_loader"]
    refit_steps_per_epoch = math.ceil(
        len(refit_train_loader) / float(rt["grad_accum"])
    )
    refit_total_steps = refit_epochs * refit_steps_per_epoch

    refit_comp = build_model_and_optimization(
        backbone_name=backbone_name,
        pretrained=True,
        img_size=img_size,
        num_centers=int(full_train_val_df["center_id"].nunique()),
        use_od_stream=True,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        layer_decay=layer_decay,
        total_steps=refit_total_steps,
        warmup_ratio=0.10,
        ema_decay=ema_decay,
        grad_checkpointing=rt["grad_checkpointing"],
        device=device,
    )
    refit_model: DualStreamStainViT = refit_comp["model"]
    refit_ema: ModelEMA = refit_comp["ema"]
    refit_criterion: DomainGeneralizingHistopathLoss = refit_comp["criterion"]
    refit_optimizer = refit_comp["optimizer"]
    refit_scheduler = refit_comp["scheduler"]
    refit_scaler = torch.amp.GradScaler(
        device.type, enabled=(use_amp and amp_dtype == torch.float16)
    )

    refit_step = 0
    refit_history: List[Dict[str, Any]] = []
    refit_ep_durations: List[float] = []

    for ep in range(1, refit_epochs + 1):
        if ep > 1:
            mean_r_ep = float(np.mean(refit_ep_durations))
            if (
                (time.time() - run_start) + mean_r_ep + est_test_tta + 35.0
                >= max_total_seconds
            ):
                print(
                    f"[{model_tag}] Intra-epoch refit time guard triggered before epoch "
                    f"{ep}/{refit_epochs}; finishing safely within ceiling."
                )
                break

        r_t0 = time.time()
        r_stats, refit_step = train_one_epoch(
            model=refit_model,
            ema=refit_ema,
            criterion=refit_criterion,
            optimizer=refit_optimizer,
            scheduler=refit_scheduler,
            scaler=refit_scaler,
            train_loader=refit_train_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_accum=rt["grad_accum"],
            global_step=refit_step,
            total_steps=refit_total_steps,
        )
        r_sec = time.time() - r_t0
        refit_ep_durations.append(r_sec)
        refit_history.append(
            {
                "epoch": ep,
                "train_loss": round(r_stats["loss"], 5),
                "cls_loss": round(r_stats["cls"], 5),
                "aux_loss": round(r_stats["aux"], 5),
                "vrex_loss": round(r_stats["vrex"], 5),
                "cdan_loss": round(r_stats["cdan"], 5),
                "logit_cons_loss": round(r_stats["lcons"], 5),
                "emb_cons_loss": round(r_stats["econs"], 5),
                "seconds": round(r_sec, 1),
            }
        )
        print(
            f"[{model_tag}] Refit Epoch {ep:02d}/{refit_epochs:02d} | "
            f"loss={r_stats['loss']:.4f} (cls={r_stats['cls']:.4f} aux={r_stats['aux']:.4f} "
            f"vrex={r_stats['vrex']:.4f} cdan={r_stats['cdan']:.4f} "
            f"lcons={r_stats['lcons']:.4f} econs={r_stats['econs']:.4f}) | {r_sec:.1f}s"
        )

    completed_epochs = len(refit_history)
    min_acceptable_epochs = max(3, refit_epochs // 2)
    if completed_epochs < min_acceptable_epochs:
        print(
            f"[{model_tag}] Refit stopped too early ({completed_epochs}/{refit_epochs} epochs); "
            f"falling back to pre-refit 8-view TTA logits."
        )
        del (
            refit_loaders,
            refit_train_loader,
            refit_test_loader,
            refit_model,
            refit_ema,
            refit_optimizer,
            refit_scheduler,
            refit_scaler,
        )
        release_memory()
        return {
            "completed": False,
            "refit_epochs": completed_epochs,
            "refit_history": refit_history,
            "test_probs_refit": None,
            "pre_vs_refit_corr": None,
        }

    refit_infer_model = refit_ema.ema_model if best_source == "ema" else refit_model
    save_checkpoint(
        refit_infer_model,
        WORKING_DIR / f"refit_trainval_model_{model_tag}.pt",
        metadata={
            "model_tag": model_tag,
            "refit_epochs": completed_epochs,
            "source": best_source,
            "seed": refit_seed,
            "provenance": refit_comp["provenance"],
        },
    )

    test_ids_refit, test_probs_refit, _ = predict_dataset(
        refit_infer_model, refit_test_loader, device, use_amp, amp_dtype
    )
    assert test_ids_refit == stage1_res["test_ids_pre"]

    pd.DataFrame(
        {"sample_id": test_ids_refit, "tumor_probability": test_probs_refit}
    ).to_csv(WORKING_DIR / f"test_predictions_refit_{model_tag}.csv", index=False)

    corr = float(np.corrcoef(stage1_res["test_probs_pre"], test_probs_refit)[0, 1])
    print(
        f"[{model_tag}] Refit Inference Complete ({completed_epochs} epochs) | "
        f"Pre-vs-Refit Test Corr={corr:.4f} | "
        f"Refit mean={test_probs_refit.mean():.4f} pos_rate={(test_probs_refit>=0.5).mean():.4f}"
    )

    del (
        refit_loaders,
        refit_train_loader,
        refit_test_loader,
        refit_model,
        refit_ema,
        refit_optimizer,
        refit_scheduler,
        refit_scaler,
    )
    release_memory()

    return {
        "completed": True,
        "refit_epochs": completed_epochs,
        "refit_history": refit_history,
        "test_probs_refit": test_probs_refit,
        "pre_vs_refit_corr": corr,
    }


def run_training_and_evaluation(
    backbone_name: str = "hf-hub:1aurent/vit_large_patch14_224.kaiko_ai_towards_large_pathology_fms",
    second_backbone_name: str = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320",
    img_size: int = 256,
    canonical_strength: float = 0.75,
    num_epochs: int = 9,
    min_epochs: int = 5,
    early_stopping_patience: int = 4,
    backbone_lr: float = 2e-5,
    second_backbone_lr: float = 5e-5,
    head_lr: float = 3e-4,
    weight_decay: float = 0.03,
    layer_decay: float = 0.88,
    second_layer_decay: float = 0.85,
    ema_decay: float = 0.993,
    tta_views: int = 8,
    enable_post_selection_refit: bool = True,
    seed: int = 42,
    max_stage1_seconds: float = 2500.0,
    max_total_seconds: float = 3300.0,
) -> float:
    """
    Full competitive heterogeneous ensemble pipeline:
      Stage 1: Sequentially train Model 1 (DINOv2 ViT-Large) and Model 2 (OpenCLIP
               ConvNeXt-Large) on the exact same train.csv split and training-only
               StainReferenceBank; evaluate each with identical 8-view D4 TTA on
               validation.csv and test.csv; combine via parameter-free 50/50
               logit-space averaging for the official pre-refit Final Validation Score.
      Stage 2: Time-guarded post-selection Train+Val refits (refit_seed=seed+100)
               under a strict 55-minute ceiling (max_total_seconds=3300.0), blending
               each architecture's pre-refit and refit test logits 0.45/0.55 if
               completed (or 1.0 pre-refit otherwise) and averaging both families
               50/50 in logit space before sigmoid for ./submission/submission.csv.
      Final:   Print exact pre-refit official validation score on the last line.
    """
    run_start = time.time()
    seed_everything(seed)

    binary_metrics_fn = load_official_binary_metrics(INPUT_DIR)
    rt = detect_runtime_config()

    # -------------------------------------------------------------------------
    # Stage 1A: Prepare Shared Leakage-Free Splits & Training-Only StainBank
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
    train_stain_bank: StainReferenceBank = bundle["stain_bank"]
    del bundle
    release_memory()

    # -------------------------------------------------------------------------
    # Stage 1B: Sequential Training & 8-View D4 TTA for Heterogeneous Family Pair
    # -------------------------------------------------------------------------
    model_specs = [
        {
            "model_tag": "vit_large",
            "backbone_name": backbone_name,
            "backbone_lr": backbone_lr,
            "layer_decay": layer_decay,
            "soft_deadline": run_start + min(max_stage1_seconds * 0.58, max_total_seconds * 0.46),
        },
        {
            "model_tag": "convnext_large",
            "backbone_name": second_backbone_name,
            "backbone_lr": second_backbone_lr,
            "layer_decay": second_layer_decay,
            "soft_deadline": run_start + min(max_stage1_seconds, max_total_seconds - 240.0),
        },
    ]

    stage1_results: List[Dict[str, Any]] = []
    for spec in model_specs:
        res = _train_and_eval_single_stage1_model(
            model_tag=spec["model_tag"],
            backbone_name=spec["backbone_name"],
            backbone_lr=spec["backbone_lr"],
            layer_decay=spec["layer_decay"],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_stain_bank=train_stain_bank,
            rt=rt,
            binary_metrics_fn=binary_metrics_fn,
            img_size=img_size,
            canonical_strength=canonical_strength,
            num_epochs=num_epochs,
            min_epochs=min_epochs,
            early_stopping_patience=early_stopping_patience,
            head_lr=head_lr,
            weight_decay=weight_decay,
            ema_decay=ema_decay,
            tta_views=tta_views,
            seed=seed,
            stage1_deadline=spec["soft_deadline"],
        )
        stage1_results.append(res)
        release_memory()

    res_m1, res_m2 = stage1_results[0], stage1_results[1]
    assert res_m1["val_ids_tta"] == res_m2["val_ids_tta"]
    assert res_m1["test_ids_pre"] == res_m2["test_ids_pre"]

    # -------------------------------------------------------------------------
    # Stage 1C: Official Pre-Refit 50/50 Logit-Space Ensemble Validation
    # -------------------------------------------------------------------------
    ens_val_logits = 0.5 * prob_to_logit(res_m1["val_probs_tta"]) + 0.5 * prob_to_logit(
        res_m2["val_probs_tta"]
    )
    ens_val_probs = logit_to_prob(ens_val_logits)
    final_val_eval = evaluate_validation_predictions(
        val_df, res_m1["val_ids_tta"], ens_val_probs, binary_metrics_fn
    )
    final_validation_score = float(final_val_eval["balanced_accuracy"])

    val_corr = float(
        np.corrcoef(res_m1["val_probs_tta"], res_m2["val_probs_tta"])[0, 1]
    )
    val_disagree = float(
        ((res_m1["val_probs_tta"] >= 0.5) != (res_m2["val_probs_tta"] >= 0.5)).mean()
    )
    ens_test_pre_logits = 0.5 * prob_to_logit(
        res_m1["test_probs_pre"]
    ) + 0.5 * prob_to_logit(res_m2["test_probs_pre"])
    ens_test_pre_probs = logit_to_prob(ens_test_pre_logits)
    test_pre_corr = float(
        np.corrcoef(res_m1["test_probs_pre"], res_m2["test_probs_pre"])[0, 1]
    )

    pd.DataFrame(
        {
            "sample_id": res_m1["val_ids_tta"],
            "label": val_df["label"].astype(int).values,
            "center_id": val_df["center_id"].astype(str).values,
            "slide_id": val_df["slide_id"].astype(str).values,
            "tumor_probability": ens_val_probs,
            f"prob_{res_m1['model_tag']}": res_m1["val_probs_tta"],
            f"prob_{res_m2['model_tag']}": res_m2["val_probs_tta"],
        }
    ).to_csv(WORKING_DIR / "val_predictions_tta.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": res_m1["test_ids_pre"],
            "tumor_probability": ens_test_pre_probs,
            f"prob_{res_m1['model_tag']}": res_m1["test_probs_pre"],
            f"prob_{res_m2['model_tag']}": res_m2["test_probs_pre"],
        }
    ).to_csv(WORKING_DIR / "test_predictions_pre_refit.csv", index=False)

    ens_center_summary = ", ".join(
        f"{cid}:bal={m['balanced_accuracy']:.4f}/auc={m['auroc']:.4f}"
        for cid, m in final_val_eval["per_center"].items()
    )
    print(
        f"Ensemble 50/50 Logit Pre-Refit {tta_views}-View D4 TTA Official Val | "
        f"bal_acc={final_validation_score:.5f} auroc={final_val_eval['auroc']:.5f} "
        f"acc={final_val_eval['accuracy']:.5f} | "
        f"recall(T={final_val_eval['tumor_recall']:.4f}, N={final_val_eval['normal_recall']:.4f}) "
        f"pos_rate={final_val_eval['pred_pos_rate']:.4f} | "
        f"val_corr={val_corr:.4f} disagree={val_disagree:.4f} test_corr={test_pre_corr:.4f} | "
        f"{ens_center_summary}"
    )

    # -------------------------------------------------------------------------
    # Stage 2: Time-Guarded Post-Selection Train+Val Refits (55-min Ceiling)
    # (Pre-refit ensemble validation score above remains the sole reported score)
    # -------------------------------------------------------------------------
    refit_seed = seed + 100
    refit_bank: Optional[StainReferenceBank] = None
    refit_results: Dict[str, Dict[str, Any]] = {}
    family_final_test_logits: List[np.ndarray] = []

    if enable_post_selection_refit:
        # Fit shared Train+Val StainReferenceBank only if at least one model can fit in budget
        can_any_refit = False
        elapsed_now = time.time() - run_start
        for res in stage1_results:
            r_ep = max(4, min(int(res["best_epoch"]), num_epochs))
            avg_ep = (
                float(np.mean(res["epoch_durations"]))
                if res["epoch_durations"]
                else 110.0
            )
            est_sec = (
                avg_ep * (len(full_train_val_df) / max(1, res["n_train"])) * r_ep
                + max(25.0, float(res["tta_seconds"]) * 0.65)
                + 60.0
            )
            if elapsed_now + est_sec < max_total_seconds:
                can_any_refit = True
                break

        if can_any_refit:
            seed_everything(refit_seed)
            refit_bank = StainReferenceBank().fit(
                full_train_val_df,
                input_dir=INPUT_DIR,
                max_patches_per_slide=36,
                seed=refit_seed,
            )
            refit_bank.save(WORKING_DIR / "stain_reference_bank_trainval.json")

    for res in stage1_results:
        tag = res["model_tag"]
        pre_logits = prob_to_logit(res["test_probs_pre"])
        if enable_post_selection_refit and refit_bank is not None:
            r_out = _run_single_model_refit(
                stage1_res=res,
                full_train_val_df=full_train_val_df,
                test_df=test_df,
                refit_bank=refit_bank,
                rt=rt,
                img_size=img_size,
                canonical_strength=canonical_strength,
                num_epochs=num_epochs,
                head_lr=head_lr,
                weight_decay=weight_decay,
                ema_decay=ema_decay,
                tta_views=tta_views,
                refit_seed=refit_seed,
                run_start=run_start,
                max_total_seconds=max_total_seconds,
            )
        else:
            r_out = {
                "completed": False,
                "refit_epochs": 0,
                "refit_history": [],
                "test_probs_refit": None,
                "pre_vs_refit_corr": None,
            }

        refit_results[tag] = r_out
        if r_out["completed"] and r_out["test_probs_refit"] is not None:
            blended_m_logits = 0.45 * pre_logits + 0.55 * prob_to_logit(
                r_out["test_probs_refit"]
            )
        else:
            blended_m_logits = pre_logits
        family_final_test_logits.append(blended_m_logits)

    final_test_ids = res_m1["test_ids_pre"]
    final_test_logits = 0.5 * family_final_test_logits[0] + 0.5 * family_final_test_logits[1]
    final_test_probs = logit_to_prob(final_test_logits)

    # -------------------------------------------------------------------------
    # Stage 3: Write Verified Submission & Persist Auditable Summary
    # -------------------------------------------------------------------------
    sub_df = write_and_verify_submission(
        sample_submission_df=sample_submission_df,
        test_ids=final_test_ids,
        test_probs=final_test_probs,
        submission_path=SUBMISSION_DIR / "submission.csv",
    )

    summary = {
        "final_validation_score_pre_refit": final_validation_score,
        "final_val_eval_tta": final_val_eval,
        "ensemble_diagnostics": {
            "val_prob_corr_m1_m2": round(val_corr, 5),
            "val_pred_disagreement_rate": round(val_disagree, 5),
            "test_pre_prob_corr_m1_m2": round(test_pre_corr, 5),
        },
        "tta_views": tta_views,
        "models": {
            res["model_tag"]: {
                "best_epoch": res["best_epoch"],
                "best_source": res["best_source"],
                "backbone_lr": res["backbone_lr"],
                "layer_decay": res["layer_decay"],
                "val_eval_tta": res["val_eval_tta"],
                "provenance": res["provenance"],
                "refit_completed": bool(refit_results[res["model_tag"]]["completed"]),
                "refit_epochs": int(refit_results[res["model_tag"]]["refit_epochs"]),
                "pre_vs_refit_corr": refit_results[res["model_tag"]][
                    "pre_vs_refit_corr"
                ],
                "history": res["history"],
                "refit_history": refit_results[res["model_tag"]]["refit_history"],
            }
            for res in stage1_results
        },
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

    refit_status = ", ".join(
        f"{r['model_tag']}={'refit('+str(refit_results[r['model_tag']]['refit_epochs'])+'ep)' if refit_results[r['model_tag']]['completed'] else 'pre-only'}"
        for r in stage1_results
    )
    print(
        f"Submission Verified -> {SUBMISSION_DIR / 'submission.csv'} | "
        f"Ensemble({refit_status}) | rows={len(sub_df)} "
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
