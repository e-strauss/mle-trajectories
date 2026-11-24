import copy
import gc
import importlib.util
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler, StandardScaler

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# =============================================================================
# Step 1: Configuration, Reproducibility & Leakage-Free Data Pipeline
# =============================================================================
DATA_CONFIG: Dict[str, Any] = {
    "input_dir": "./input",
    "working_dir": "./working",
    "submission_dir": "./submission",
    "seed": 42,
    "img_size": 256,
    "batch_size": 48,
    "eval_batch_size": 64,
    "num_workers": 6,
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
    "fourier_crop_size": 64,
    "fourier_max_per_slide": 25,
    "morph_feature_version": "v1_40d",
    "morph_clip_range": (-5.0, 5.0),
    "morph_noise_std": 0.08,
    "n_group_folds": 5,
    "aug_prob_fact": 0.55,
    "aug_prob_hed": 0.65,
    "aug_prob_hsv": 0.40,
    "aug_prob_optics": 0.35,
    "aug_prob_crop": 0.40,
    "aug_prob_erase": 0.25,
}


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds without modifying CUDA_VISIBLE_DEVICES."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    """Deterministic worker seeding for DataLoader workers."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_and_validate_metadata(
    input_dir: str = DATA_CONFIG["input_dir"],
    n_group_folds: int = DATA_CONFIG["n_group_folds"],
    seed: int = DATA_CONFIG["seed"],
) -> Dict[str, Any]:
    """
    Load train, fixed validation, test, and sample_submission tables.
    Enforces strict leakage checks and constructs within-train LOCO and
    slide-grouped K-fold splits without touching validation data.
    """
    input_path = Path(input_dir)
    train_df = pd.read_csv(input_path / "train.csv")
    val_df = pd.read_csv(input_path / "validation.csv")
    test_df = pd.read_csv(input_path / "test.csv")
    sample_sub_df = pd.read_csv(input_path / "sample_submission.csv")

    req_dev_cols = {
        "sample_id",
        "image_path",
        "label",
        "slide_id",
        "group_id",
        "center_id",
    }
    req_test_cols = {"sample_id", "image_path"}
    if not req_dev_cols.issubset(train_df.columns):
        raise ValueError(
            f"train.csv missing columns: {req_dev_cols - set(train_df.columns)}"
        )
    if not req_dev_cols.issubset(val_df.columns):
        raise ValueError(
            f"validation.csv missing columns: {req_dev_cols - set(val_df.columns)}"
        )
    if not req_test_cols.issubset(test_df.columns):
        raise ValueError(
            f"test.csv missing columns: {req_test_cols - set(test_df.columns)}"
        )
    if list(sample_sub_df.columns) != ["sample_id", "tumor_probability"]:
        raise ValueError(
            f"Unexpected sample_submission columns: {list(sample_sub_df.columns)}"
        )

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df.isna().any().any():
            raise ValueError(f"{name}.csv contains unexpected missing values.")
        if df["sample_id"].nunique() != len(df):
            raise ValueError(f"{name}.csv contains duplicate sample_id entries.")

    if not (test_df["sample_id"].values == sample_sub_df["sample_id"].values).all():
        test_df = sample_sub_df[["sample_id"]].merge(
            test_df, on="sample_id", how="left", validate="one_to_one"
        )
        if test_df["image_path"].isna().any():
            raise ValueError(
                "test.csv does not cover all sample_ids in sample_submission.csv."
            )

    overlap_samples = set(train_df["sample_id"]).intersection(set(val_df["sample_id"]))
    overlap_slides = set(train_df["slide_id"]).intersection(set(val_df["slide_id"]))
    overlap_groups = set(train_df["group_id"]).intersection(set(val_df["group_id"]))
    if overlap_samples or overlap_slides or overlap_groups:
        raise ValueError(
            f"Data leakage detected between train and validation: "
            f"samples={len(overlap_samples)}, slides={len(overlap_slides)}, groups={len(overlap_groups)}"
        )

    for name, df in [("train", train_df), ("val", val_df)]:
        df["label"] = df["label"].astype(int)
        if not set(df["label"].unique()).issubset({0, 1}):
            raise ValueError(
                f"{name}.csv has non-binary labels: {df['label'].unique()}"
            )

    for df in (train_df, val_df, test_df):
        df["full_path"] = df["image_path"].apply(lambda p: str(input_path / p))

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sample_paths = df["full_path"].iloc[[0, len(df) // 2, -1]].tolist()
        for sp in sample_paths:
            if not os.path.exists(sp):
                raise FileNotFoundError(f"Missing image file in {name}: {sp}")

    train_centers = sorted(train_df["center_id"].unique().tolist())
    center_to_idx = {c: i for i, c in enumerate(train_centers)}

    train_slides = sorted(train_df["slide_id"].unique().tolist())
    slide_to_idx = {s: i for i, s in enumerate(train_slides)}

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["center_idx"] = train_df["center_id"].map(center_to_idx).astype(int)
    train_df["slide_idx"] = train_df["slide_id"].map(slide_to_idx).astype(int)
    val_df["center_idx"] = (
        val_df["center_id"].map(lambda c: center_to_idx.get(c, -1)).astype(int)
    )

    loco_splits: List[Dict[str, Any]] = []
    for held_out_center in train_centers:
        tr_idx = np.where(train_df["center_id"].values != held_out_center)[0]
        va_idx = np.where(train_df["center_id"].values == held_out_center)[0]
        loco_splits.append(
            {
                "held_out_center": held_out_center,
                "train_idx": tr_idx,
                "val_idx": va_idx,
            }
        )

    strat_key = train_df["center_idx"].astype(str) + "_" + train_df["label"].astype(str)
    n_unique_groups = train_df["group_id"].nunique()
    actual_folds = min(n_group_folds, n_unique_groups)
    sgkf = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    group_kfold_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_idx, va_idx in sgkf.split(train_df, strat_key, groups=train_df["group_id"]):
        tr_groups = set(train_df.iloc[tr_idx]["group_id"])
        va_groups = set(train_df.iloc[va_idx]["group_id"])
        assert (
            len(tr_groups.intersection(va_groups)) == 0
        ), "Group leakage inside StratifiedGroupKFold!"
        group_kfold_splits.append((tr_idx, va_idx))

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    all_slides = sorted(train_val_df["slide_id"].unique().tolist())
    all_slide_to_idx = {s: i for i, s in enumerate(all_slides)}
    train_val_df["slide_idx"] = (
        train_val_df["slide_id"].map(all_slide_to_idx).astype(int)
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "sample_sub_df": sample_sub_df,
        "train_val_df": train_val_df,
        "center_to_idx": center_to_idx,
        "slide_to_idx": slide_to_idx,
        "loco_splits": loco_splits,
        "group_kfold_splits": group_kfold_splits,
    }


def compute_slide_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-sample training weights that balance both (center_id, label)
    strata and dampen dominance of slides with unusually many patches via
    inverse square-root slide frequency. Strictly uses training metadata only.
    """
    slide_counts = df["slide_id"].value_counts().to_dict()
    strata = df["center_id"].astype(str) + "_" + df["label"].astype(str)
    strata_counts = strata.value_counts().to_dict()

    raw_weights = np.zeros(len(df), dtype=np.float64)
    for i, (sid, st) in enumerate(zip(df["slide_id"].values, strata.values)):
        w_slide = 1.0 / math.sqrt(float(slide_counts[sid]))
        w_stratum = 1.0 / float(strata_counts[st])
        raw_weights[i] = w_slide * w_stratum

    raw_weights = raw_weights / (raw_weights.mean() + 1e-12)
    return raw_weights


def _build_radial_masks(size: int = 128) -> Tuple[List[np.ndarray], np.ndarray]:
    """Precompute rotation-invariant concentric frequency ring masks for FFT features."""
    cy, cx = size // 2, size // 2
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    edges = [1.0, 4.0, 8.0, 16.0, 28.0, 44.0, float(size // 2)]
    rings = []
    for r0, r1 in zip(edges[:-1], edges[1:]):
        rings.append((dist >= r0) & (dist < r1))
    valid_mask = dist >= 1.0
    dist_norm = dist / float(size // 2)
    return rings, np.where(valid_mask, dist_norm, 0.0)


class HistoStainPhysics:
    """
    Beer-Lambert optical density transforms, Ruifrok-Johnston H&E stain
    deconvolution, physics-guided stain perturbation, and 40-D stain-invariant
    + D4-invariant morphological feature extraction.
    """

    _RAW_MATRIX = np.array(
        [
            [0.650, 0.704, 0.286],
            [0.072, 0.990, 0.105],
            [0.268, 0.570, 0.776],
        ],
        dtype=np.float32,
    )
    HE_REF_MATRIX = _RAW_MATRIX / np.linalg.norm(_RAW_MATRIX, axis=1, keepdims=True)
    HE_INV_MATRIX = np.linalg.inv(HE_REF_MATRIX).astype(np.float32)

    _RGB_TO_OD_LUT: np.ndarray = (
        -np.log10(np.clip(np.arange(256, dtype=np.float32), 1.0, 255.0) / 255.0)
    ).astype(np.float32)
    _LN10: np.float32 = np.float32(math.log(10.0))

    _RADIAL_RINGS, _RADIAL_DIST_NORM = _build_radial_masks(128)
    MORPH_DIM = 40

    @classmethod
    def rgb_to_od_fixed(cls, rgb_uint8: np.ndarray) -> np.ndarray:
        return cls._RGB_TO_OD_LUT[rgb_uint8]

    @classmethod
    def rgb_to_od_adaptive(cls, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Adaptive white-point Beer-Lambert OD to cancel illumination and scanner
        exposure shifts across unseen medical centers.
        """
        rgb_f = rgb_uint8.astype(np.float32)
        white_ref = np.clip(
            np.percentile(rgb_f, 99.5, axis=(0, 1)), 190.0, 255.0
        ).astype(np.float32)
        ratio = np.clip(rgb_f / white_ref[None, None, :], 1.0 / 255.0, 1.0)
        return -np.log10(ratio)

    @classmethod
    def perturb_hed_stain(
        cls,
        rgb_uint8: np.ndarray,
        rng: np.random.Generator,
        conc_scale: Tuple[float, float] = (0.68, 1.38),
        conc_shift: Tuple[float, float] = (-0.055, 0.055),
        basis_jitter: float = 0.065,
    ) -> np.ndarray:
        """
        Physics-guided Beer-Lambert stain augmentation: perturbs both dye
        concentrations (alpha, beta) and the H&E stain absorption basis vectors
        to simulate unseen hospital staining protocols and scanner spectra.
        """
        od = cls.rgb_to_od_fixed(rgb_uint8)
        hed = np.matmul(od, cls.HE_INV_MATRIX)

        alpha = rng.uniform(conc_scale[0], conc_scale[1], size=(1, 1, 3)).astype(
            np.float32
        )
        beta = rng.uniform(conc_shift[0], conc_shift[1], size=(1, 1, 3)).astype(
            np.float32
        )
        hed_pert = hed * alpha + beta

        noise = rng.uniform(-basis_jitter, basis_jitter, size=(3, 3)).astype(np.float32)
        ref_pert = np.clip(cls.HE_REF_MATRIX + noise, 0.02, 1.5)
        ref_pert = ref_pert / (np.linalg.norm(ref_pert, axis=1, keepdims=True) + 1e-8)

        od_pert = np.clip(np.matmul(hed_pert, ref_pert), 0.0, 3.0)
        rgb_pert = np.clip(255.0 * np.exp(-od_pert * cls._LN10), 0.0, 255.0)
        return rgb_pert.astype(np.uint8)

    @staticmethod
    def _safe_skew_kurt(vals: np.ndarray) -> Tuple[float, float]:
        """Compute robust skewness and excess kurtosis with numerical guards."""
        if vals.size < 16:
            return 0.0, 0.0
        mean = float(vals.mean())
        diff = vals - mean
        var = float((diff * diff).mean())
        if var < 1e-8:
            return 0.0, 0.0
        std = math.sqrt(var)
        z = diff / std
        skew = float(np.clip((z * z * z).mean(), -10.0, 10.0))
        kurt = float(np.clip((z * z * z * z).mean() - 3.0, -10.0, 20.0))
        return skew, kurt

    @classmethod
    def extract_patch_morphological_features(cls, rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Extract 40 stain-invariant, D4-rotation/flip-invariant morphological
        features purely from patch pixels.
        """
        if rgb_uint8.shape[0] != 256 or rgb_uint8.shape[1] != 256:
            rgb_uint8 = cv2.resize(rgb_uint8, (256, 256), interpolation=cv2.INTER_AREA)

        od = cls.rgb_to_od_adaptive(rgb_uint8)
        hed = np.matmul(od, cls.HE_INV_MATRIX)
        H = hed[:, :, 0]
        E = hed[:, :, 1]
        od_mag = np.sqrt(np.sum(od * od, axis=-1))

        tissue_mask = od_mag > 0.08
        tissue_frac = float(tissue_mask.mean())
        use_mask = (
            tissue_mask
            if int(tissue_mask.sum()) >= 64
            else np.ones_like(tissue_mask, dtype=bool)
        )

        H_tis = H[use_mask]
        E_tis = E[use_mask]

        h_q10, h_q25, h_q50, h_q75, h_q90 = np.percentile(H_tis, [10, 25, 50, 75, 90])
        h_iqr = max(float(h_q75 - h_q25), 1e-4)
        H_std = np.clip((H - float(h_q50)) / h_iqr, -4.0, 6.0).astype(np.float32)

        e_q25, e_q50, e_q75 = np.percentile(E_tis, [25, 50, 75])
        e_iqr = max(float(e_q75 - e_q25), 1e-4)
        E_std = np.clip((E - float(e_q50)) / e_iqr, -4.0, 6.0).astype(np.float32)

        H_std_tis = H_std[use_mask]
        E_std_tis = E_std[use_mask]

        h_mean = float(H_std_tis.mean())
        h_std_val = float(H_std_tis.std())
        h_skew, h_kurt = cls._safe_skew_kurt(H_std_tis)
        h_asym = float(np.clip((h_q90 - h_q50) / (h_q50 - h_q10 + 1e-4), -5.0, 10.0))
        nuc_frac_1 = float((H_std_tis > 1.0).mean())
        nuc_frac_2 = float((H_std_tis > 2.0).mean())

        e_mean = float(E_std_tis.mean())
        e_std_val = float(E_std_tis.std())
        e_skew, e_kurt = cls._safe_skew_kurt(E_std_tis)
        e_high_frac = float((E_std_tis > 1.0).mean())

        if h_std_val > 1e-5 and e_std_val > 1e-5:
            he_corr = float(
                np.clip(
                    np.mean((H_std_tis - h_mean) * (E_std_tis - e_mean))
                    / (h_std_val * e_std_val + 1e-6),
                    -1.0,
                    1.0,
                )
            )
        else:
            he_corr = 0.0

        H_pos = np.clip(H_tis, 0.0, None)
        E_pos = np.clip(E_tis, 0.0, None)
        he_ratio = H_pos / (H_pos + E_pos + 1e-4)
        her_mean = float(he_ratio.mean())
        her_std = float(he_ratio.std())
        her_q10, her_q90 = np.percentile(he_ratio, [10, 90])

        gx_h = cv2.Sobel(H_std, cv2.CV_32F, 1, 0, ksize=3)
        gy_h = cv2.Sobel(H_std, cv2.CV_32F, 0, 1, ksize=3)
        gmag_h = np.sqrt(gx_h * gx_h + gy_h * gy_h)
        gh_mean = float(gmag_h.mean())
        gh_std = float(gmag_h.std())
        gh_q90 = float(np.percentile(gmag_h, 90))

        gx_e = cv2.Sobel(E_std, cv2.CV_32F, 1, 0, ksize=3)
        gy_e = cv2.Sobel(E_std, cv2.CV_32F, 0, 1, ksize=3)
        gmag_e = np.sqrt(gx_e * gx_e + gy_e * gy_e)
        ge_mean = float(gmag_e.mean())
        ge_std = float(gmag_e.std())

        lap_h = cv2.Laplacian(H_std, cv2.CV_32F, ksize=3)
        lap_h_abs_mean = float(np.abs(lap_h).mean())
        lap_h_std = float(lap_h.std())

        H_half = cv2.resize(H_std, (128, 128), interpolation=cv2.INTER_AREA)
        lap_h_half_std = float(cv2.Laplacian(H_half, cv2.CV_32F, ksize=3).std())

        H_blocks = H_std.reshape(4, 64, 4, 64)
        E_blocks = E_std.reshape(4, 64, 4, 64)
        T_blocks = tissue_mask.astype(np.float32).reshape(4, 64, 4, 64)

        blk_h_mean = H_blocks.mean(axis=(1, 3))
        blk_nuc_frac = (H_blocks > 1.0).mean(axis=(1, 3))
        blk_e_mean = E_blocks.mean(axis=(1, 3))
        blk_t_frac = T_blocks.mean(axis=(1, 3))

        blk_h_std = float(blk_h_mean.std())
        blk_h_range = float(blk_h_mean.max() - blk_h_mean.min())
        blk_nuc_std = float(blk_nuc_frac.std())
        blk_nuc_range = float(blk_nuc_frac.max() - blk_nuc_frac.min())
        blk_e_std = float(blk_e_mean.std())
        blk_t_std = float(blk_t_frac.std())

        H_zero = H_half - float(H_half.mean())
        fft_h = np.fft.fftshift(np.fft.fft2(H_zero))
        power = (fft_h.real * fft_h.real + fft_h.imag * fft_h.imag).astype(np.float64)
        power[64, 64] = 0.0
        p_sum = float(power.sum())
        if p_sum > 1e-10:
            p_norm = power / p_sum
            ring_feats = [
                float(math.log1p(100.0 * p_norm[ring].sum()))
                for ring in cls._RADIAL_RINGS
            ]
            spec_centroid = float((p_norm * cls._RADIAL_DIST_NORM).sum())
            p_nz = p_norm[p_norm > 1e-12]
            spec_entropy = float(-(p_nz * np.log(p_nz)).sum() / math.log(128.0 * 128.0))
        else:
            ring_feats = [0.0] * len(cls._RADIAL_RINGS)
            spec_centroid = 0.0
            spec_entropy = 0.0

        feats = np.array(
            [
                tissue_frac,
                h_mean,
                h_std_val,
                h_skew,
                h_kurt,
                h_asym,
                nuc_frac_1,
                nuc_frac_2,
                e_mean,
                e_std_val,
                e_skew,
                e_kurt,
                e_high_frac,
                he_corr,
                her_mean,
                her_std,
                float(her_q10),
                float(her_q90),
                gh_mean,
                gh_std,
                gh_q90,
                ge_mean,
                ge_std,
                lap_h_abs_mean,
                lap_h_std,
                lap_h_half_std,
                blk_h_std,
                blk_h_range,
                blk_nuc_std,
                blk_nuc_range,
                blk_e_std,
                blk_t_std,
                *ring_feats,
                spec_centroid,
                spec_entropy,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _load_rgb_uint8(path: str) -> np.ndarray:
    """Fast BGR->RGB PNG reader with explicit existence/corruption guard."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at path: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _extract_single_path_morph(path: str) -> np.ndarray:
    return HistoStainPhysics.extract_patch_morphological_features(_load_rgb_uint8(path))


class MorphologicalFeaturePipeline:
    """
    Extracts pixel-only 40-D morphological features for any split (with disk
    caching in ./working/) and applies leakage-free scaling fitted STRICTLY
    on training features.
    """

    def __init__(
        self,
        working_dir: str = DATA_CONFIG["working_dir"],
        version: str = DATA_CONFIG["morph_feature_version"],
        clip_range: Tuple[float, float] = DATA_CONFIG["morph_clip_range"],
        max_workers: int = 8,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.clip_range = clip_range
        self.max_workers = max_workers

        self.train_medians_: Optional[np.ndarray] = None
        self.robust_scaler_ = RobustScaler(quantile_range=(10.0, 90.0))
        self.std_scaler_ = StandardScaler()
        self.is_fitted_ = False

    def extract_raw(
        self,
        df: pd.DataFrame,
        cache_name: Optional[str] = None,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """
        Extract raw (unscaled) 40-D pixel-only morphological features.
        Purely deterministic per image; writes computed array to ./working/.
        """
        cache_path = None
        if cache_name is not None:
            cache_path = (
                self.working_dir
                / f"morph_raw_{cache_name}_{self.version}_{len(df)}.npy"
            )
            if cache_path.exists() and not force_recompute:
                arr = np.load(cache_path)
                if arr.shape == (len(df), HistoStainPhysics.MORPH_DIM):
                    return arr.astype(np.float32)

        paths = df["full_path"].tolist()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            feats_list = list(executor.map(_extract_single_path_morph, paths))

        raw_feats = np.stack(feats_list, axis=0).astype(np.float32)
        if cache_path is not None:
            np.save(cache_path, raw_feats)
        return raw_feats

    def fit(self, train_raw: np.ndarray) -> "MorphologicalFeaturePipeline":
        """Fit imputation medians and dual scalers STRICTLY on training features."""
        arr = train_raw.astype(np.float64).copy()
        finite_mask = np.isfinite(arr)
        medians = np.zeros(arr.shape[1], dtype=np.float64)
        for j in range(arr.shape[1]):
            col_valid = arr[finite_mask[:, j], j]
            medians[j] = float(np.median(col_valid)) if col_valid.size > 0 else 0.0
            arr[~finite_mask[:, j], j] = medians[j]

        self.train_medians_ = medians
        x_rob = self.robust_scaler_.fit_transform(arr)
        x_clip = np.clip(x_rob, self.clip_range[0], self.clip_range[1])
        self.std_scaler_.fit(x_clip)
        self.is_fitted_ = True
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Transform features using strictly training-fitted statistics."""
        if not self.is_fitted_ or self.train_medians_ is None:
            raise RuntimeError(
                "MorphologicalFeaturePipeline must be fitted on training data before transform()."
            )
        arr = raw.astype(np.float64).copy()
        bad = ~np.isfinite(arr)
        if bad.any():
            rows, cols = np.where(bad)
            arr[rows, cols] = self.train_medians_[cols]
        x_rob = self.robust_scaler_.transform(arr)
        x_clip = np.clip(x_rob, self.clip_range[0], self.clip_range[1])
        x_std = self.std_scaler_.transform(x_clip)
        return np.clip(x_std, self.clip_range[0], self.clip_range[1]).astype(np.float32)

    def fit_transform(self, train_raw: np.ndarray) -> np.ndarray:
        return self.fit(train_raw).transform(train_raw)


class FourierStyleBank:
    """
    Training-only bank of low-frequency 2D FFT amplitude spectra stratified
    across centers and slides. During training, mixes low-frequency amplitudes
    from other centers/slides with a smooth cosine-tapered frequency mask
    (eliminating Gibbs ringing) while preserving exact cellular phase geometry.
    """

    def __init__(
        self,
        crop_size: int = DATA_CONFIG["fourier_crop_size"],
        max_per_slide: int = DATA_CONFIG["fourier_max_per_slide"],
        seed: int = DATA_CONFIG["seed"],
    ) -> None:
        self.crop_size = crop_size
        self.max_per_slide = max_per_slide
        self.seed = seed

        self.amplitudes_: Optional[np.ndarray] = None
        self.center_ids_: Optional[np.ndarray] = None
        self.slide_ids_: Optional[np.ndarray] = None
        self._by_other_center: Dict[int, np.ndarray] = {}
        self._by_other_slide: Dict[int, np.ndarray] = {}
        self._all_indices: Optional[np.ndarray] = None

    def _extract_low_freq_amp(self, path: str) -> np.ndarray:
        rgb = _load_rgb_uint8(path)
        if rgb.shape[0] != 256 or rgb.shape[1] != 256:
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        img_f = rgb.astype(np.float32) / 255.0
        fft_shift = np.fft.fftshift(np.fft.fft2(img_f, axes=(0, 1)), axes=(0, 1))
        amp = np.abs(fft_shift).astype(np.float32)
        half = self.crop_size // 2
        c = 128
        return amp[c - half : c + half, c - half : c + half, :]

    def fit(self, train_df: pd.DataFrame, max_workers: int = 8) -> "FourierStyleBank":
        """Fit amplitude bank strictly on the provided training dataframe."""
        rng = np.random.default_rng(self.seed)
        sampled_rows = []
        for _, grp in train_df.groupby("slide_id", sort=True):
            n_take = min(len(grp), self.max_per_slide)
            chosen = rng.choice(grp.index.values, size=n_take, replace=False)
            sampled_rows.append(train_df.loc[chosen])
        bank_df = pd.concat(sampled_rows, axis=0, ignore_index=True)

        paths = bank_df["full_path"].tolist()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            amp_list = list(executor.map(self._extract_low_freq_amp, paths))

        self.amplitudes_ = np.stack(amp_list, axis=0).astype(np.float32)
        self.center_ids_ = bank_df["center_idx"].values.astype(np.int64)
        self.slide_ids_ = bank_df["slide_idx"].values.astype(np.int64)
        self._all_indices = np.arange(len(bank_df), dtype=np.int64)

        self._by_other_center = {}
        for cid in np.unique(self.center_ids_):
            other = np.where(self.center_ids_ != cid)[0]
            if other.size > 0:
                self._by_other_center[int(cid)] = other

        self._by_other_slide = {}
        for sid in np.unique(self.slide_ids_):
            other = np.where(self.slide_ids_ != sid)[0]
            if other.size > 0:
                self._by_other_slide[int(sid)] = other

        return self

    def mix_amplitude(
        self,
        rgb_uint8: np.ndarray,
        src_center_idx: int,
        src_slide_idx: int,
        rng: np.random.Generator,
        lam_range: Tuple[float, float] = (0.15, 0.75),
        radius_range: Tuple[int, int] = (5, 28),
        cross_center_prob: float = 0.70,
    ) -> np.ndarray:
        """
        Smooth cosine-tapered low-frequency Fourier Amplitude Mixing (FACT).
        Preferentially draws target amplitude spectrum from another center/slide
        in the training bank while preserving the exact phase spectrum.
        """
        if self.amplitudes_ is None or self._all_indices is None:
            return rgb_uint8

        if rgb_uint8.shape[0] != 256 or rgb_uint8.shape[1] != 256:
            rgb_uint8 = cv2.resize(rgb_uint8, (256, 256), interpolation=cv2.INTER_AREA)

        if rng.random() < cross_center_prob and src_center_idx in self._by_other_center:
            pool = self._by_other_center[src_center_idx]
        elif src_slide_idx in self._by_other_slide:
            pool = self._by_other_slide[src_slide_idx]
        else:
            pool = self._all_indices

        tgt_idx = int(pool[int(rng.integers(0, len(pool)))])
        tgt_amp_crop = self.amplitudes_[tgt_idx]

        img_f = rgb_uint8.astype(np.float32) / 255.0
        fft_src = np.fft.fftshift(np.fft.fft2(img_f, axes=(0, 1)), axes=(0, 1))
        amp_src = np.abs(fft_src).astype(np.float32)

        max_r = min(radius_range[1], self.crop_size // 2 - 1)
        min_r = min(radius_range[0], max_r)
        r = int(rng.integers(min_r, max_r + 1))
        lam = float(rng.uniform(lam_range[0], lam_range[1]))

        win_1d = np.hanning(2 * r + 1).astype(np.float32)
        win_2d = np.outer(win_1d, win_1d)[:, :, None] * lam

        c_img = 128
        c_bank = self.crop_size // 2
        amp_src_patch = amp_src[c_img - r : c_img + r + 1, c_img - r : c_img + r + 1, :]
        amp_tgt_patch = tgt_amp_crop[
            c_bank - r : c_bank + r + 1, c_bank - r : c_bank + r + 1, :
        ]

        amp_mixed_patch = (1.0 - win_2d) * amp_src_patch + win_2d * amp_tgt_patch
        scale_patch = amp_mixed_patch / (amp_src_patch + 1e-6)

        fft_mixed = fft_src.copy()
        fft_mixed[
            c_img - r : c_img + r + 1, c_img - r : c_img + r + 1, :
        ] *= scale_patch

        img_rec = np.fft.ifft2(
            np.fft.ifftshift(fft_mixed, axes=(0, 1)), axes=(0, 1)
        ).real
        img_rec = np.clip(img_rec * 255.0, 0.0, 255.0).astype(np.uint8)
        return img_rec


class HistopathologyAugmentor:
    """
    Training augmentation combining D4 dihedral symmetry, cross-center FACT,
    Beer-Lambert H&E optical-density perturbation, scanner optics jitter, and
    mild crop/erasing. Validation and Test preprocessing are strictly identical.
    """

    def __init__(
        self,
        img_size: int = DATA_CONFIG["img_size"],
        mean: Tuple[float, float, float] = DATA_CONFIG["imagenet_mean"],
        std: Tuple[float, float, float] = DATA_CONFIG["imagenet_std"],
        fourier_bank: Optional[FourierStyleBank] = None,
        prob_fact: float = DATA_CONFIG["aug_prob_fact"],
        prob_hed: float = DATA_CONFIG["aug_prob_hed"],
        prob_hsv: float = DATA_CONFIG["aug_prob_hsv"],
        prob_optics: float = DATA_CONFIG["aug_prob_optics"],
        prob_crop: float = DATA_CONFIG["aug_prob_crop"],
        prob_erase: float = DATA_CONFIG["aug_prob_erase"],
    ) -> None:
        self.img_size = img_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.fourier_bank = fourier_bank
        self.prob_fact = prob_fact
        self.prob_hed = prob_hed
        self.prob_hsv = prob_hsv
        self.prob_optics = prob_optics
        self.prob_crop = prob_crop
        self.prob_erase = prob_erase

    def _to_normalized_tensor(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        if rgb_uint8.shape[0] != self.img_size or rgb_uint8.shape[1] != self.img_size:
            rgb_uint8 = cv2.resize(
                rgb_uint8, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
            )
        chw = rgb_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0
        chw = (chw - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(chw))

    def transform_eval(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        """Deterministic pixel-only transform shared identically by validation and test."""
        return self._to_normalized_tensor(rgb_uint8)

    def transform_train(
        self,
        rgb_uint8: np.ndarray,
        center_idx: int = -1,
        slide_idx: int = -1,
        rng: Optional[np.random.Generator] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()

        img = rgb_uint8

        k_rot = int(rng.integers(0, 4))
        if k_rot > 0:
            img = np.rot90(img, k_rot)
        if rng.random() < 0.5:
            img = np.fliplr(img)
        img = np.ascontiguousarray(img)

        if self.fourier_bank is not None and rng.random() < self.prob_fact:
            img = self.fourier_bank.mix_amplitude(img, center_idx, slide_idx, rng)

        if rng.random() < self.prob_hed:
            img = HistoStainPhysics.perturb_hed_stain(img, rng)

        if rng.random() < self.prob_hsv:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h_shift = int(rng.integers(-6, 7))
            s_scale = float(rng.uniform(0.85, 1.15))
            hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + h_shift) % 180).astype(
                np.uint8
            )
            hsv[:, :, 1] = np.clip(
                hsv[:, :, 1].astype(np.float32) * s_scale, 0.0, 255.0
            ).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if rng.random() < self.prob_optics:
            op = int(rng.integers(0, 3))
            if op == 0:
                sigma = float(rng.uniform(0.3, 1.1))
                img = cv2.GaussianBlur(img, (3, 3), sigmaX=sigma)
            elif op == 1:
                blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=1.0)
                alpha = float(rng.uniform(0.25, 0.75))
                img = np.clip(
                    img.astype(np.float32) * (1.0 + alpha)
                    - blurred.astype(np.float32) * alpha,
                    0,
                    255,
                ).astype(np.uint8)
            else:
                gamma = float(rng.uniform(0.82, 1.22))
                lut = np.clip(
                    np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
                    0,
                    255,
                ).astype(np.uint8)
                img = lut[img]

        if rng.random() < self.prob_crop:
            h, w = img.shape[:2]
            scale = float(rng.uniform(0.84, 0.98))
            ch, cw = int(h * scale), int(w * scale)
            y0 = int(rng.integers(0, max(1, h - ch + 1)))
            x0 = int(rng.integers(0, max(1, w - cw + 1)))
            crop = img[y0 : y0 + ch, x0 : x0 + cw]
            img = cv2.resize(
                crop, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
            )

        tensor = self._to_normalized_tensor(img)

        if rng.random() < self.prob_erase:
            _, H, W = tensor.shape
            eh = int(rng.integers(H // 12, H // 5))
            ew = int(rng.integers(W // 12, W // 5))
            ey = int(rng.integers(0, H - eh + 1))
            ex = int(rng.integers(0, W - ew + 1))
            tensor[:, ey : ey + eh, ex : ex + ew] = 0.0

        return tensor


def generate_d4_tta_views(batch_images: torch.Tensor) -> List[torch.Tensor]:
    """
    Generate all 8 exact D4 dihedral symmetry views of a [B, C, H, W] batch.
    Used identically during validation and test inference.
    """
    views = []
    for k in range(4):
        rot = torch.rot90(batch_images, k=k, dims=(2, 3))
        views.append(rot)
        views.append(torch.flip(rot, dims=(3,)))
    return views


class HistopathologyDataset(Dataset):
    """
    Dual-input histopathology Dataset yielding pixel-only image + 40-D
    morphological features (with training-only domain keys excluded at eval).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        morph_features: np.ndarray,
        augmentor: HistopathologyAugmentor,
        mode: str = "train",
        morph_noise_std: float = DATA_CONFIG["morph_noise_std"],
    ) -> None:
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")
        if len(df) != len(morph_features):
            raise ValueError(
                f"Row count mismatch: df={len(df)} vs morph={len(morph_features)}"
            )

        self.paths = df["full_path"].tolist()
        self.sample_ids = df["sample_id"].tolist()
        self.morph = morph_features.astype(np.float32)
        self.augmentor = augmentor
        self.mode = mode
        self.morph_noise_std = morph_noise_std

        self.has_labels = "label" in df.columns and mode in {"train", "val"}
        self.labels = df["label"].values.astype(np.float32) if self.has_labels else None

        if mode == "train" and "center_idx" in df.columns and "slide_idx" in df.columns:
            self.center_ids = df["center_idx"].values.astype(np.int64)
            self.slide_ids = df["slide_idx"].values.astype(np.int64)
        else:
            self.center_ids = None
            self.slide_ids = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rgb = _load_rgb_uint8(self.paths[idx])
        morph_vec = self.morph[idx]

        if self.mode == "train":
            rng = np.random.default_rng()
            cid = int(self.center_ids[idx]) if self.center_ids is not None else -1
            sid = int(self.slide_ids[idx]) if self.slide_ids is not None else -1
            img_tensor = self.augmentor.transform_train(
                rgb, center_idx=cid, slide_idx=sid, rng=rng
            )
            if self.morph_noise_std > 0.0:
                morph_vec = morph_vec + rng.normal(
                    0.0, self.morph_noise_std, size=morph_vec.shape
                ).astype(np.float32)
            item: Dict[str, Any] = {
                "image": img_tensor,
                "morph": torch.from_numpy(np.ascontiguousarray(morph_vec)),
                "label": torch.tensor(self.labels[idx], dtype=torch.float32),
                "center_idx": torch.tensor(cid, dtype=torch.long),
                "slide_idx": torch.tensor(sid, dtype=torch.long),
                "sample_id": self.sample_ids[idx],
            }
            return item

        img_tensor = self.augmentor.transform_eval(rgb)
        item = {
            "image": img_tensor,
            "morph": torch.from_numpy(np.ascontiguousarray(morph_vec)),
            "sample_id": self.sample_ids[idx],
        }
        if self.has_labels and self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def build_data_pipeline(
    config: Optional[Dict[str, Any]] = None,
    use_balanced_sampler: bool = True,
    force_recompute_morph: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end leakage-free data preparation & feature engineering orchestrator.
    Fits all stateful preprocessors (MorphologicalFeaturePipeline and
    FourierStyleBank) STRICTLY on train.csv, transforms validation.csv and
    test.csv identically, and returns ready-to-use Datasets and DataLoaders.
    """
    cfg = dict(DATA_CONFIG)
    if config is not None:
        cfg.update(config)

    seed_everything(cfg["seed"])
    os.makedirs(cfg["working_dir"], exist_ok=True)
    os.makedirs(cfg["submission_dir"], exist_ok=True)

    meta = load_and_validate_metadata(
        input_dir=cfg["input_dir"],
        n_group_folds=cfg["n_group_folds"],
        seed=cfg["seed"],
    )
    train_df = meta["train_df"]
    val_df = meta["val_df"]
    test_df = meta["test_df"]

    morph_pipeline = MorphologicalFeaturePipeline(
        working_dir=cfg["working_dir"],
        version=cfg["morph_feature_version"],
        clip_range=cfg["morph_clip_range"],
        max_workers=max(4, cfg["num_workers"] * 2),
    )
    train_morph_raw = morph_pipeline.extract_raw(
        train_df, cache_name="train", force_recompute=force_recompute_morph
    )
    val_morph_raw = morph_pipeline.extract_raw(
        val_df, cache_name="val", force_recompute=force_recompute_morph
    )
    test_morph_raw = morph_pipeline.extract_raw(
        test_df, cache_name="test", force_recompute=force_recompute_morph
    )

    train_morph = morph_pipeline.fit_transform(train_morph_raw)
    val_morph = morph_pipeline.transform(val_morph_raw)
    test_morph = morph_pipeline.transform(test_morph_raw)

    fourier_bank = FourierStyleBank(
        crop_size=cfg["fourier_crop_size"],
        max_per_slide=cfg["fourier_max_per_slide"],
        seed=cfg["seed"],
    ).fit(train_df, max_workers=max(4, cfg["num_workers"] * 2))

    train_augmentor = HistopathologyAugmentor(
        img_size=cfg["img_size"],
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        fourier_bank=fourier_bank,
        prob_fact=cfg["aug_prob_fact"],
        prob_hed=cfg["aug_prob_hed"],
        prob_hsv=cfg["aug_prob_hsv"],
        prob_optics=cfg["aug_prob_optics"],
        prob_crop=cfg["aug_prob_crop"],
        prob_erase=cfg["aug_prob_erase"],
    )
    eval_augmentor = HistopathologyAugmentor(
        img_size=cfg["img_size"],
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        fourier_bank=None,
    )

    train_dataset = HistopathologyDataset(
        train_df,
        train_morph,
        train_augmentor,
        mode="train",
        morph_noise_std=float(cfg["morph_noise_std"]),
    )
    val_dataset = HistopathologyDataset(val_df, val_morph, eval_augmentor, mode="val")
    test_dataset = HistopathologyDataset(
        test_df, test_morph, eval_augmentor, mode="test"
    )

    pin_mem = torch.cuda.is_available()
    num_workers = int(cfg["num_workers"])
    persistent = num_workers > 0

    if use_balanced_sampler:
        train_weights = compute_slide_balanced_weights(train_df)
        train_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(train_weights).double(),
            num_samples=len(train_df),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent,
    )

    return {
        "config": cfg,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "sample_sub_df": meta["sample_sub_df"],
        "train_val_df": meta["train_val_df"],
        "center_to_idx": meta["center_to_idx"],
        "slide_to_idx": meta["slide_to_idx"],
        "loco_splits": meta["loco_splits"],
        "group_kfold_splits": meta["group_kfold_splits"],
        "morph_pipeline": morph_pipeline,
        "train_morph_raw": train_morph_raw,
        "val_morph_raw": val_morph_raw,
        "test_morph_raw": test_morph_raw,
        "train_morph": train_morph,
        "val_morph": val_morph,
        "test_morph": test_morph,
        "morph_dim": HistoStainPhysics.MORPH_DIM,
        "fourier_bank": fourier_bank,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


def build_refit_data_loaders(
    data_bundle: Dict[str, Any],
    use_balanced_sampler: bool = True,
) -> Dict[str, Any]:
    """
    Build post-selection train+validation refit DataLoader and matching test
    DataLoader. Must ONLY be invoked AFTER official validation evaluation and
    model selection have been completed and logged.
    """
    cfg = data_bundle["config"]
    train_val_df = data_bundle["train_val_df"]
    test_df = data_bundle["test_df"]

    tv_morph_raw = np.concatenate(
        [data_bundle["train_morph_raw"], data_bundle["val_morph_raw"]], axis=0
    )
    test_morph_raw = data_bundle["test_morph_raw"]

    refit_morph_pipeline = MorphologicalFeaturePipeline(
        working_dir=cfg["working_dir"],
        version=cfg["morph_feature_version"],
        clip_range=cfg["morph_clip_range"],
    )
    tv_morph = refit_morph_pipeline.fit_transform(tv_morph_raw)
    test_morph_refit = refit_morph_pipeline.transform(test_morph_raw)

    refit_fourier_bank = FourierStyleBank(
        crop_size=cfg["fourier_crop_size"],
        max_per_slide=cfg["fourier_max_per_slide"],
        seed=cfg["seed"] + 100,
    ).fit(train_val_df, max_workers=max(4, cfg["num_workers"] * 2))

    refit_train_aug = HistopathologyAugmentor(
        img_size=cfg["img_size"],
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        fourier_bank=refit_fourier_bank,
        prob_fact=cfg["aug_prob_fact"],
        prob_hed=cfg["aug_prob_hed"],
        prob_hsv=cfg["aug_prob_hsv"],
        prob_optics=cfg["aug_prob_optics"],
        prob_crop=cfg["aug_prob_crop"],
        prob_erase=cfg["aug_prob_erase"],
    )
    refit_eval_aug = HistopathologyAugmentor(
        img_size=cfg["img_size"],
        mean=cfg["imagenet_mean"],
        std=cfg["imagenet_std"],
        fourier_bank=None,
    )

    tv_dataset = HistopathologyDataset(
        train_val_df,
        tv_morph,
        refit_train_aug,
        mode="train",
        morph_noise_std=float(cfg["morph_noise_std"]),
    )
    test_dataset_refit = HistopathologyDataset(
        test_df, test_morph_refit, refit_eval_aug, mode="test"
    )

    pin_mem = torch.cuda.is_available()
    num_workers = int(cfg["num_workers"])
    persistent = num_workers > 0

    if use_balanced_sampler:
        tv_weights = compute_slide_balanced_weights(train_val_df)
        tv_sampler = WeightedRandomSampler(
            weights=torch.from_numpy(tv_weights).double(),
            num_samples=len(train_val_df),
            replacement=True,
        )
        tv_loader = DataLoader(
            tv_dataset,
            batch_size=cfg["batch_size"],
            sampler=tv_sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )
    else:
        tv_loader = DataLoader(
            tv_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            persistent_workers=persistent,
        )

    test_loader_refit = DataLoader(
        test_dataset_refit,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        worker_init_fn=_worker_init_fn,
        persistent_workers=persistent,
    )

    return {
        "train_val_loader": tv_loader,
        "test_loader_refit": test_loader_refit,
        "refit_morph_pipeline": refit_morph_pipeline,
        "refit_fourier_bank": refit_fourier_bank,
    }


# =============================================================================
# Step 2: Model Architecture, Composite Domain-Generalizing Loss, LLRD & EMA
# =============================================================================
PRETRAINED_MODEL_INFO: Dict[str, Any] = {
    "primary_backbone": "swinv2_small_window8_256.ms_in1k",
    "ensemble_backbones": [
        {
            "family": "swinv2",
            "primary": "swinv2_small_window8_256.ms_in1k",
            "fallbacks": [
                "swinv2_small_window8_256",
                "swinv2_tiny_window8_256.ms_in1k",
            ],
            "pretraining_dataset": "ImageNet-1K (1.28M natural-scene photographs, 1,000 classes)",
        },
        {
            "family": "convnextv2",
            "primary": "convnextv2_base.fcmae_ft_in22k_in1k",
            "fallbacks": [
                "convnextv2_base",
                "convnext_small.fb_in22k_ft_in1k",
                "convnext_small",
            ],
            "pretraining_dataset": "ImageNet-22K FCMAE self-supervised pretraining + ImageNet-1K fine-tuning",
        },
        {
            "family": "caformer",
            "primary": "caformer_s18.sail_in22k_ft_in1k",
            "fallbacks": [
                "caformer_s18",
                "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
                "swin_small_patch4_window7_224",
            ],
            "pretraining_dataset": "ImageNet-22K pretraining + ImageNet-1K fine-tuning",
        },
    ],
    "library": "timm",
    "pretraining_dataset": (
        "ImageNet-1K and ImageNet-22K natural-scene photograph corpora "
        "(supervised and FCMAE self-supervised pretraining)"
    ),
    "known_data_overlap": (
        "None known. ImageNet-1K and ImageNet-22K contain natural photographic "
        "scenes and have no known overlap with H&E lymph-node histopathology "
        "microscopy slides."
    ),
}

MODEL_CONFIG: Dict[str, Any] = {
    "backbone_name": "swinv2_small_window8_256.ms_in1k",
    "backbone_candidates": None,
    "ensemble_specs": [
        {
            "tag": "swinv2_small",
            "backbone_name": "swinv2_small_window8_256.ms_in1k",
            "backbone_candidates": [
                "swinv2_small_window8_256.ms_in1k",
                "swinv2_small_window8_256",
                "swinv2_tiny_window8_256.ms_in1k",
            ],
        },
        {
            "tag": "convnextv2_base",
            "backbone_name": "convnextv2_base.fcmae_ft_in22k_in1k",
            "backbone_candidates": [
                "convnextv2_base.fcmae_ft_in22k_in1k",
                "convnextv2_base",
                "convnext_small.fb_in22k_ft_in1k",
                "convnext_small",
            ],
        },
        {
            "tag": "caformer_s18",
            "backbone_name": "caformer_s18.sail_in22k_ft_in1k",
            "backbone_candidates": [
                "caformer_s18.sail_in22k_ft_in1k",
                "caformer_s18",
                "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
                "swin_small_patch4_window7_224",
            ],
        },
    ],
    "img_size": 256,
    "pretrained": True,
    "morph_dim": 40,
    "morph_hidden_dim": 128,
    "visual_proj_dim": 384,
    "fusion_dim": 512,
    "drop_path_rate": 0.25,
    "head_dropout": 0.30,
    "morph_dropout": 0.20,
    "morph_modality_drop_p": 0.25,
    "morph_feature_drop_p": 0.15,
    "num_msd_branches": 5,
    "use_mixstyle": True,
    "mixstyle_p": 0.50,
    "mixstyle_alpha": 0.20,
    "mixstyle_cross_center_p": 0.85,
    "dual_mixup_fraction": 0.50,
    "dual_mixup_alpha": 0.40,
    "dual_mixup_cross_center_p": 0.70,
    "label_smoothing": 0.04,
    "w_vrex": 0.35,
    "w_align": 0.15,
    "w_groupdro": 0.25,
    "groupdro_tau": 0.25,
    "w_midpoint": 0.15,
    "w_rank": 0.15,
    "rank_margin": 1.0,
    "w_morph_aux": 0.20,
    "w_vis_aux": 0.10,
    "backbone_lr": 6.0e-5,
    "head_lr": 2.5e-4,
    "llrd_decay": 0.80,
    "weight_decay": 3.0e-2,
    "min_lr_ratio": 0.02,
    "ema_decay": 0.997,
}


class CrossCenterMixStyle(nn.Module):
    """
    Feature-level style perturbation operating on 2D spatial feature maps [B, C, H, W].
    During training, mixes channel-wise instance mean/std preferentially across
    samples from DIFFERENT medical centers to suppress residual stain/scanner
    statistics in intermediate visual representations. Exact identity at eval().
    """

    def __init__(
        self,
        p: float = 0.50,
        alpha: float = 0.20,
        cross_center_p: float = 0.85,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.cross_center_p = float(cross_center_p)
        self.eps = float(eps)

    def _build_partner_indices(
        self, batch_size: int, center_idx: Optional[torch.Tensor], device: torch.device
    ) -> torch.Tensor:
        perm = torch.randperm(batch_size, device=device)
        if center_idx is None or batch_size < 2:
            return perm

        c_cpu = center_idx.detach().cpu().tolist()
        valid_centers = {c for c in c_cpu if c >= 0}
        if len(valid_centers) < 2:
            return perm

        by_other: Dict[int, List[int]] = {}
        for c in valid_centers:
            others = [idx for idx, val in enumerate(c_cpu) if val >= 0 and val != c]
            if others:
                by_other[c] = others

        perm_list = perm.cpu().tolist()
        rand_vals = torch.rand(batch_size).tolist()
        chosen = []
        for i, c in enumerate(c_cpu):
            if rand_vals[i] < self.cross_center_p and c in by_other:
                pool = by_other[c]
                pick = pool[int(torch.randint(0, len(pool), (1,)).item())]
                chosen.append(pick)
            else:
                chosen.append(perm_list[i])
        return torch.tensor(chosen, dtype=torch.long, device=device)

    def forward(
        self, x: torch.Tensor, center_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if (not self.training) or (self.p <= 0.0) or x.size(0) < 2:
            return x
        if float(torch.rand(1).item()) > self.p:
            return x

        orig_dtype = x.dtype
        x_f32 = x.float()
        B = x_f32.size(0)

        mu = x_f32.mean(dim=(2, 3), keepdim=True)
        var = x_f32.var(dim=(2, 3), unbiased=False, keepdim=True)
        sig = torch.sqrt(var + self.eps)
        x_normed = (x_f32 - mu) / sig

        perm = self._build_partner_indices(B, center_idx, x.device)
        beta_dist = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta_dist.sample((B, 1, 1, 1)).to(device=x.device, dtype=torch.float32)

        mu_mix = lam * mu + (1.0 - lam) * mu[perm].detach()
        sig_mix = lam * sig + (1.0 - lam) * sig[perm].detach()

        out = x_normed * sig_mix + mu_mix
        return out.to(orig_dtype)


class SpatialAttentionPool2d(nn.Module):
    """
    Combines gated MIL-style spatial attention pooling (capturing focal tumor
    micrometastases without dilution) with global average pooling (capturing
    overall tissue context). Purely LayerNorm-based for domain robustness.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(in_channels)
        self.attn_v = nn.Linear(in_channels, hidden_dim)
        self.attn_u = nn.Linear(in_channels, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.out_dim = in_channels * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)

        gated = torch.tanh(self.attn_v(tokens)) * torch.sigmoid(self.attn_u(tokens))
        attn_logits = self.attn_w(gated).squeeze(-1)
        attn_weights = F.softmax(attn_logits.float(), dim=-1).to(tokens.dtype)
        attn_pooled = torch.bmm(attn_weights.unsqueeze(1), tokens).squeeze(1)

        avg_pooled = tokens.mean(dim=1)
        return torch.cat([attn_pooled, avg_pooled], dim=-1)


class MorphologyEncoder(nn.Module):
    """
    Residual LayerNorm-SiLU-Dropout MLP processing the 40-D self-standardized,
    D4-invariant morphological features, with an auxiliary classifier head.
    """

    def __init__(
        self,
        in_dim: int = 40,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()
        self.aux_head = nn.Linear(hidden_dim, 1)

    def forward(self, morph: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(self.in_norm(morph))
        emb = self.act(x + self.res_block(x))
        morph_logits = self.aux_head(emb).squeeze(-1)
        return emb, morph_logits


class MultiSampleDropoutHead(nn.Module):
    """
    Averages predictions across K parallel dropout masks sharing one linear
    projection, improving probability calibration near the fixed 0.5 threshold.
    """

    def __init__(
        self, in_dim: int, dropout: float = 0.30, num_branches: int = 5
    ) -> None:
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(max(1, num_branches))]
        )
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = sum(self.fc(drop(x)) for drop in self.dropouts) / float(
            len(self.dropouts)
        )
        return logits.squeeze(-1)


def _create_hierarchical_backbone(
    backbone_name: Union[str, List[str]],
    pretrained: bool = True,
    drop_path_rate: float = 0.25,
    img_size: int = 256,
    backbone_candidates: Optional[List[str]] = None,
) -> Tuple[nn.Module, str]:
    """
    Instantiate a BatchNorm-free hierarchical backbone (SwinV2, ConvNeXtV2,
    CAFormer, or verified family fallbacks), passing `img_size` only when
    required by fixed-resolution Transformer variants.
    """
    candidate_names: List[str] = []
    if isinstance(backbone_name, (list, tuple)):
        candidate_names.extend([str(n) for n in backbone_name])
    elif backbone_name:
        candidate_names.append(str(backbone_name))
    if backbone_candidates:
        candidate_names.extend([str(n) for n in backbone_candidates])

    candidate_names.extend(
        [
            "swinv2_small_window8_256.ms_in1k",
            "swinv2_small_window8_256",
            "convnext_small.fb_in22k_ft_in1k",
        ]
    )

    seen = set()
    last_err: Optional[Exception] = None
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)

        lower = name.lower()
        needs_img_size = (
            ("swin" in lower or "vit" in lower or "eva" in lower)
            and (f"_{img_size}" not in lower)
            and (not lower.endswith(str(img_size)))
        )
        attempt_extras = (
            [{"img_size": img_size}, {}]
            if needs_img_size
            else [{}, {"img_size": img_size}]
        )

        for extra_kwargs in attempt_extras:
            try:
                kwargs: Dict[str, Any] = {
                    "pretrained": pretrained,
                    "features_only": True,
                    "out_indices": (2, 3),
                    "drop_path_rate": drop_path_rate,
                }
                kwargs.update(extra_kwargs)
                model = timm.create_model(name, **kwargs)
                return model, name
            except Exception as exc:
                last_err = exc
                continue

    raise RuntimeError(
        f"Failed to instantiate any hierarchical timm backbone. Last error: {last_err}"
    )


def apply_cross_center_dual_mixup(
    images: torch.Tensor,
    morph: torch.Tensor,
    labels: torch.Tensor,
    center_idx: Optional[torch.Tensor] = None,
    mix_fraction: float = MODEL_CONFIG["dual_mixup_fraction"],
    alpha: float = MODEL_CONFIG["dual_mixup_alpha"],
    cross_center_p: float = MODEL_CONFIG["dual_mixup_cross_center_p"],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Jointly mix RGB patches, 40-D morphological features, and labels on a
    stratified fraction (default 50%) of each training batch with Beta(alpha, alpha)
    and preferential cross-center pairing, returning `unmixed_mask` for the
    remaining clean samples so domain/group/midpoint losses operate on exact
    (center, class) subgroups every step without corruption.
    """
    B = images.size(0)
    device = images.device
    unmixed_mask = torch.ones(B, dtype=torch.bool, device=device)

    if B < 4 or mix_fraction <= 0.0 or alpha <= 0.0:
        return images, morph, labels, unmixed_mask

    lbl_cpu = (labels.detach().cpu() > 0.5).long().tolist()
    if center_idx is not None:
        c_cpu = center_idx.detach().cpu().tolist()
    else:
        c_cpu = [-1] * B

    strata: Dict[Tuple[int, int], List[int]] = {}
    for idx, (c, y) in enumerate(zip(c_cpu, lbl_cpu)):
        strata.setdefault((c, y), []).append(idx)

    mix_indices: List[int] = []
    for _, grp_indices in strata.items():
        n_grp = len(grp_indices)
        if n_grp < 2:
            continue
        perm_grp = torch.randperm(n_grp).tolist()
        n_take = int(round(n_grp * float(mix_fraction)))
        min_clean = 2 if n_grp >= 4 else 1
        n_take = max(0, min(n_take, n_grp - min_clean))
        for p_i in perm_grp[:n_take]:
            mix_indices.append(grp_indices[p_i])

    if not mix_indices:
        return images, morph, labels, unmixed_mask

    valid_centers = {c for c in c_cpu if c >= 0}
    by_other_center: Dict[int, List[int]] = {}
    if len(valid_centers) >= 2:
        for c in valid_centers:
            others = [idx for idx, val in enumerate(c_cpu) if val >= 0 and val != c]
            if others:
                by_other_center[c] = others

    partner_indices: List[int] = []
    rand_vals = torch.rand(len(mix_indices)).tolist()
    for i, r_val in zip(mix_indices, rand_vals):
        c_i = c_cpu[i]
        if r_val < float(cross_center_p) and c_i in by_other_center:
            pool = by_other_center[c_i]
            j = pool[int(torch.randint(0, len(pool), (1,)).item())]
        else:
            j = int(torch.randint(0, B - 1, (1,)).item())
            if j >= i:
                j += 1
        partner_indices.append(j)

    mix_idx_t = torch.tensor(mix_indices, dtype=torch.long, device=device)
    partner_idx_t = torch.tensor(partner_indices, dtype=torch.long, device=device)
    unmixed_mask[mix_idx_t] = False

    beta_dist = torch.distributions.Beta(float(alpha), float(alpha))
    lam = beta_dist.sample((len(mix_indices),)).to(device=device, dtype=torch.float32)
    lam_img = lam.view(-1, 1, 1, 1).to(images.dtype)
    lam_morph = lam.view(-1, 1).to(morph.dtype)
    lam_lbl = lam.to(labels.dtype)

    mixed_images = images.clone()
    mixed_morph = morph.clone()
    mixed_labels = labels.clone()

    mixed_images[mix_idx_t] = (
        lam_img * images[mix_idx_t] + (1.0 - lam_img) * images[partner_idx_t]
    )
    mixed_morph[mix_idx_t] = (
        lam_morph * morph[mix_idx_t] + (1.0 - lam_morph) * morph[partner_idx_t]
    )
    mixed_labels[mix_idx_t] = (
        lam_lbl * labels[mix_idx_t] + (1.0 - lam_lbl) * labels[partner_idx_t]
    )

    return mixed_images, mixed_morph, mixed_labels, unmixed_mask


class MorphGatedHistoNet(nn.Module):
    """
    Backbone-agnostic BatchNorm-free Hybrid Vision-Morphology Histopathology
    Classifier supporting SwinV2, ConvNeXtV2, and CAFormer backbones with
    automatic GPU input normalization remap, Cross-Center Feature MixStyle,
    Morphological Modality/Feature Dropout, MIL-inspired Spatial Attention +
    Average Pooling, 40-D MorphologyEncoder, bounded Residual FiLM + Gated
    Cross-Modal Fusion, and Multi-Sample Dropout.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = dict(MODEL_CONFIG)
        if config is not None:
            cfg.update(config)
        self.cfg = cfg

        self.backbone, self.resolved_backbone_name = _create_hierarchical_backbone(
            backbone_name=cfg["backbone_name"],
            pretrained=bool(cfg["pretrained"]),
            drop_path_rate=float(cfg["drop_path_rate"]),
            img_size=int(cfg.get("img_size", DATA_CONFIG["img_size"])),
            backbone_candidates=cfg.get("backbone_candidates", None),
        )

        # Register lossless GPU affine remap when backbone pretraining stats
        # differ from the DataLoader's ImageNet normalization.
        src_mean = np.array(DATA_CONFIG["imagenet_mean"], dtype=np.float32)
        src_std = np.array(DATA_CONFIG["imagenet_std"], dtype=np.float32)
        tgt_mean = src_mean.copy()
        tgt_std = src_std.copy()
        try:
            data_cfg = timm.data.resolve_model_data_config(self.backbone)
            if "mean" in data_cfg and "std" in data_cfg:
                cand_mean = np.array(data_cfg["mean"], dtype=np.float32)
                cand_std = np.array(data_cfg["std"], dtype=np.float32)
                if cand_mean.shape == (3,) and cand_std.shape == (3,):
                    tgt_mean = cand_mean
                    tgt_std = cand_std
        except Exception:
            pass

        needs_remap = bool(
            np.max(np.abs(tgt_mean - src_mean)) > 1e-4
            or np.max(np.abs(tgt_std - src_std)) > 1e-4
        )
        self.use_input_remap = needs_remap
        remap_scale = (src_std / np.maximum(tgt_std, 1e-6)).reshape(1, 3, 1, 1)
        remap_shift = ((src_mean - tgt_mean) / np.maximum(tgt_std, 1e-6)).reshape(
            1, 3, 1, 1
        )
        self.register_buffer(
            "input_remap_scale", torch.from_numpy(remap_scale.astype(np.float32))
        )
        self.register_buffer(
            "input_remap_shift", torch.from_numpy(remap_shift.astype(np.float32))
        )

        feat_channels = list(self.backbone.feature_info.channels())
        if len(feat_channels) < 2:
            raise ValueError(
                f"Expected at least 2 multi-scale feature stages, got {feat_channels}"
            )
        self.c3_dim = int(feat_channels[-2])
        self.c4_dim = int(feat_channels[-1])

        self.morph_modality_drop_p = float(cfg.get("morph_modality_drop_p", 0.25))
        self.morph_feature_drop_p = float(cfg.get("morph_feature_drop_p", 0.15))

        self.mixstyle = (
            CrossCenterMixStyle(
                p=float(cfg["mixstyle_p"]),
                alpha=float(cfg["mixstyle_alpha"]),
                cross_center_p=float(cfg["mixstyle_cross_center_p"]),
            )
            if bool(cfg["use_mixstyle"])
            else nn.Identity()
        )

        self.pool_s3 = SpatialAttentionPool2d(self.c3_dim, hidden_dim=128)
        self.pool_s4 = SpatialAttentionPool2d(self.c4_dim, hidden_dim=128)

        vis_raw_dim = self.pool_s3.out_dim + self.pool_s4.out_dim
        vis_dim = int(cfg["visual_proj_dim"])
        morph_dim = int(cfg["morph_dim"])
        morph_h = int(cfg["morph_hidden_dim"])
        fusion_dim = int(cfg["fusion_dim"])
        head_drop = float(cfg["head_dropout"])

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(vis_raw_dim),
            nn.Dropout(head_drop * 0.5),
            nn.Linear(vis_raw_dim, vis_dim),
            nn.LayerNorm(vis_dim),
            nn.GELU(),
        )
        self.vis_aux_head = nn.Linear(vis_dim, 1)

        self.morph_encoder = MorphologyEncoder(
            in_dim=morph_dim,
            hidden_dim=morph_h,
            dropout=float(cfg["morph_dropout"]),
        )

        self.film_gamma = nn.Linear(morph_h, vis_dim)
        self.film_beta = nn.Linear(morph_h, vis_dim)
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.morph_to_vis = nn.Sequential(
            nn.Linear(morph_h, vis_dim),
            nn.LayerNorm(vis_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(vis_dim * 2, vis_dim // 2),
            nn.GELU(),
            nn.Linear(vis_dim // 2, vis_dim),
            nn.Sigmoid(),
        )

        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(vis_dim + morph_h),
            nn.Dropout(head_drop * 0.5),
            nn.Linear(vis_dim + morph_h, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.classifier = MultiSampleDropoutHead(
            in_dim=fusion_dim,
            dropout=head_drop,
            num_branches=int(cfg["num_msd_branches"]),
        )

    @staticmethod
    def _ensure_nchw(feat: torch.Tensor, expected_c: int) -> torch.Tensor:
        if feat.ndim != 4:
            raise ValueError(f"Expected 4D feature map, got shape {tuple(feat.shape)}")
        if feat.shape[1] == expected_c:
            return feat
        if feat.shape[-1] == expected_c:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Feature map shape {tuple(feat.shape)} does not match expected channels={expected_c}"
        )

    def forward(
        self,
        image: torch.Tensor,
        morph: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.use_input_remap:
            image = image * self.input_remap_scale.to(image.dtype) + (
                self.input_remap_shift.to(image.dtype)
            )

        feats = self.backbone(image)
        f3 = self._ensure_nchw(feats[-2], self.c3_dim)
        f4 = self._ensure_nchw(feats[-1], self.c4_dim)

        if isinstance(self.mixstyle, CrossCenterMixStyle):
            f3 = self.mixstyle(f3, center_idx=center_idx)

        p3 = self.pool_s3(f3)
        p4 = self.pool_s4(f4)
        vis_emb = self.visual_proj(torch.cat([p3, p4], dim=-1))
        vis_logits = self.vis_aux_head(vis_emb).squeeze(-1)

        morph_in = morph
        if self.training:
            if self.morph_modality_drop_p > 0.0:
                mod_keep = (
                    torch.rand(
                        (morph_in.size(0), 1),
                        device=morph_in.device,
                        dtype=morph_in.dtype,
                    )
                    >= self.morph_modality_drop_p
                ).to(morph_in.dtype)
                morph_in = morph_in * mod_keep
            if self.morph_feature_drop_p > 0.0:
                feat_keep = (
                    torch.rand(
                        morph_in.shape,
                        device=morph_in.device,
                        dtype=morph_in.dtype,
                    )
                    >= self.morph_feature_drop_p
                ).to(morph_in.dtype)
                morph_in = morph_in * feat_keep

        morph_emb, morph_logits = self.morph_encoder(morph_in)

        gamma = 1.0 + 0.25 * torch.tanh(self.film_gamma(morph_emb))
        beta = 0.25 * torch.tanh(self.film_beta(morph_emb))
        vis_mod = vis_emb * gamma + beta

        morph_vis = self.morph_to_vis(morph_emb)
        gate = self.fusion_gate(torch.cat([vis_mod, morph_vis], dim=-1))
        blended_vis = gate * vis_mod + (1.0 - gate) * morph_vis

        fused_emb = self.fusion_proj(torch.cat([blended_vis, morph_emb], dim=-1))
        logits = self.classifier(fused_emb)

        if not return_dict:
            return logits

        return {
            "logits": logits,
            "morph_logits": morph_logits,
            "vis_logits": vis_logits,
            "embedding": fused_emb,
            "gate_mean": gate.mean(dim=-1, keepdim=True),
        }


MorphGatedSwinV2 = MorphGatedHistoNet


class DomainInvariantHistoLoss(nn.Module):
    """
    Multi-component loss engineered for unseen-center balanced accuracy at fixed 0.5:
      1. Symmetric label-smoothed BCE on main fused logits
      2. Soft GroupDRO worst-(center, class) risk minimization
      3. Per-Center Symmetric Logit-Midpoint calibration anchoring logit 0.0 (prob 0.5)
      4. V-REx inter-center risk variance minimization
      5. Class-Conditional L2-Normalized Center Alignment
      6. Pairwise Smooth Margin Ranking loss
      7. Auxiliary supervision on morphological & visual-only heads
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = dict(MODEL_CONFIG)
        if config is not None:
            cfg.update(config)

        self.label_smoothing = float(cfg["label_smoothing"])
        self.w_vrex = float(cfg["w_vrex"])
        self.w_align = float(cfg["w_align"])
        self.w_groupdro = float(cfg.get("w_groupdro", 0.25))
        self.groupdro_tau = float(cfg.get("groupdro_tau", 0.25))
        self.w_midpoint = float(cfg.get("w_midpoint", 0.15))
        self.w_rank = float(cfg["w_rank"])
        self.rank_margin = float(cfg["rank_margin"])
        self.w_morph_aux = float(cfg["w_morph_aux"])
        self.w_vis_aux = float(cfg["w_vis_aux"])

    def _smooth_bce_per_sample(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits_f = logits.float()
        targets_f = labels.float()
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets_f = targets_f * (1.0 - eps) + 0.5 * eps
        return F.binary_cross_entropy_with_logits(logits_f, targets_f, reduction="none")

    def _pairwise_margin_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits_f = logits.float()
        pos_mask = labels > 0.5
        neg_mask = ~pos_mask
        if int(pos_mask.sum().item()) == 0 or int(neg_mask.sum().item()) == 0:
            return logits_f.new_zeros(())
        pos_logits = logits_f[pos_mask]
        neg_logits = logits_f[neg_mask]
        diffs = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        return F.softplus(-(diffs - self.rank_margin)).mean()

    def _soft_groupdro_loss(
        self,
        per_sample_bce: torch.Tensor,
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if center_idx is None:
            return per_sample_bce.new_zeros(())
        valid_centers = torch.unique(center_idx[center_idx >= 0])
        if valid_centers.numel() < 1:
            return per_sample_bce.new_zeros(())

        group_risks = []
        for cid in valid_centers:
            c_mask = center_idx == cid
            for cls_val in (0, 1):
                cls_mask = (labels > 0.5) if cls_val == 1 else (labels <= 0.5)
                mask = c_mask & cls_mask
                if int(mask.sum().item()) >= 2:
                    group_risks.append(per_sample_bce[mask].mean())

        if len(group_risks) < 2:
            group_risks = []
            for cid in valid_centers:
                c_mask = center_idx == cid
                for cls_val in (0, 1):
                    cls_mask = (labels > 0.5) if cls_val == 1 else (labels <= 0.5)
                    mask = c_mask & cls_mask
                    if int(mask.sum().item()) >= 1:
                        group_risks.append(per_sample_bce[mask].mean())

        if len(group_risks) < 2:
            return per_sample_bce.new_zeros(())

        risks_tensor = torch.stack(group_risks).float()
        tau = max(1e-3, self.groupdro_tau)
        return tau * (
            torch.logsumexp(risks_tensor / tau, dim=0)
            - math.log(float(len(group_risks)))
        )

    def _per_center_logit_midpoint_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if center_idx is None:
            return logits.new_zeros((), dtype=torch.float32)
        logits_f = logits.float()
        valid_centers = torch.unique(center_idx[center_idx >= 0])
        if valid_centers.numel() < 1:
            return logits_f.new_zeros(())

        midpoint_terms = []
        for cid in valid_centers:
            c_mask = center_idx == cid
            pos_mask = c_mask & (labels > 0.5)
            neg_mask = c_mask & (labels <= 0.5)
            if int(pos_mask.sum().item()) >= 1 and int(neg_mask.sum().item()) >= 1:
                mean_pos = logits_f[pos_mask].mean()
                mean_neg = logits_f[neg_mask].mean()
                midpoint_terms.append((mean_pos + mean_neg) ** 2)

        if not midpoint_terms:
            return logits_f.new_zeros(())
        return torch.stack(midpoint_terms).mean()

    def _vrex_center_loss(
        self, per_sample_bce: torch.Tensor, center_idx: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if center_idx is None:
            return per_sample_bce.new_zeros(())
        valid_mask = center_idx >= 0
        if int(valid_mask.sum().item()) < 2:
            return per_sample_bce.new_zeros(())

        c_valid = center_idx[valid_mask]
        unique_centers = torch.unique(c_valid)
        if unique_centers.numel() < 2:
            return per_sample_bce.new_zeros(())

        center_risks = []
        for cid in unique_centers:
            mask = center_idx == cid
            if int(mask.sum().item()) >= 2:
                center_risks.append(per_sample_bce[mask].mean())

        if len(center_risks) < 2:
            return per_sample_bce.new_zeros(())
        risks_tensor = torch.stack(center_risks)
        return risks_tensor.var(unbiased=False)

    def _class_conditional_center_alignment(
        self,
        embedding: torch.Tensor,
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if center_idx is None:
            return embedding.new_zeros((), dtype=torch.float32)

        emb_norm = F.normalize(embedding.float(), p=2, dim=-1, eps=1e-6)
        valid_centers = torch.unique(center_idx[center_idx >= 0])
        if valid_centers.numel() < 2:
            return emb_norm.new_zeros(())

        align_terms = []
        for cls_val in (0, 1):
            cls_mask = (labels > 0.5) if cls_val == 1 else (labels <= 0.5)
            means = []
            stds = []
            for cid in valid_centers:
                mask = cls_mask & (center_idx == cid)
                if int(mask.sum().item()) >= 2:
                    sub = emb_norm[mask]
                    means.append(sub.mean(dim=0))
                    stds.append(torch.sqrt(sub.var(dim=0, unbiased=False) + 1e-5))
            if len(means) >= 2:
                for i in range(len(means)):
                    for j in range(i + 1, len(means)):
                        mean_dist = F.mse_loss(means[i], means[j], reduction="sum")
                        std_dist = F.mse_loss(stds[i], stds[j], reduction="sum")
                        align_terms.append(mean_dist + 0.5 * std_dist)

        if not align_terms:
            return emb_norm.new_zeros(())
        return torch.stack(align_terms).mean()

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
        domain_weight_scale: float = 1.0,
        unmixed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(outputs, torch.Tensor):
            logits = outputs
            morph_logits = None
            vis_logits = None
            embedding = None
        else:
            logits = outputs["logits"]
            morph_logits = outputs.get("morph_logits", None)
            vis_logits = outputs.get("vis_logits", None)
            embedding = outputs.get("embedding", None)

        per_sample_bce = self._smooth_bce_per_sample(logits, labels)
        bce_loss = per_sample_bce.mean()

        if unmixed_mask is not None and bool(unmixed_mask.any().item()):
            clean_logits = logits[unmixed_mask]
            clean_labels = labels[unmixed_mask]
            clean_bce = per_sample_bce[unmixed_mask]
            clean_centers = (
                center_idx[unmixed_mask] if center_idx is not None else None
            )
            clean_emb = embedding[unmixed_mask] if embedding is not None else None
        else:
            clean_logits = logits
            clean_labels = labels
            clean_bce = per_sample_bce
            clean_centers = center_idx
            clean_emb = embedding

        rank_loss = self._pairwise_margin_loss(clean_logits, clean_labels)

        total_loss = bce_loss + self.w_rank * rank_loss
        stats: Dict[str, float] = {
            "bce": float(bce_loss.detach().item()),
            "rank": float(rank_loss.detach().item()),
            "vrex": 0.0,
            "align": 0.0,
            "groupdro": 0.0,
            "midpoint": 0.0,
            "morph_aux": 0.0,
            "vis_aux": 0.0,
        }

        if morph_logits is not None and self.w_morph_aux > 0.0:
            morph_loss = self._smooth_bce_per_sample(morph_logits, labels).mean()
            total_loss = total_loss + self.w_morph_aux * morph_loss
            stats["morph_aux"] = float(morph_loss.detach().item())

        if vis_logits is not None and self.w_vis_aux > 0.0:
            vis_loss = self._smooth_bce_per_sample(vis_logits, labels).mean()
            total_loss = total_loss + self.w_vis_aux * vis_loss
            stats["vis_aux"] = float(vis_loss.detach().item())

        scale = max(0.0, float(domain_weight_scale))
        if scale > 0.0 and clean_centers is not None:
            if self.w_groupdro > 0.0:
                groupdro_loss = self._soft_groupdro_loss(
                    clean_bce, clean_labels, clean_centers
                )
                total_loss = total_loss + (scale * self.w_groupdro) * groupdro_loss
                stats["groupdro"] = float(groupdro_loss.detach().item())

            if self.w_midpoint > 0.0:
                midpoint_loss = self._per_center_logit_midpoint_loss(
                    clean_logits, clean_labels, clean_centers
                )
                total_loss = total_loss + (scale * self.w_midpoint) * midpoint_loss
                stats["midpoint"] = float(midpoint_loss.detach().item())

            if self.w_vrex > 0.0:
                vrex_loss = self._vrex_center_loss(clean_bce, clean_centers)
                total_loss = total_loss + (scale * self.w_vrex) * vrex_loss
                stats["vrex"] = float(vrex_loss.detach().item())

            if clean_emb is not None and self.w_align > 0.0:
                align_loss = self._class_conditional_center_alignment(
                    clean_emb, clean_labels, clean_centers
                )
                total_loss = total_loss + (scale * self.w_align) * align_loss
                stats["align"] = float(align_loss.detach().item())

        stats["total"] = float(total_loss.detach().item())
        return total_loss, stats


def _get_swin_layer_depth(
    param_name: str, max_depth: int = 6, stage2_num_blocks: int = 18
) -> int:
    """
    Map backbone parameter names across SwinV2, ConvNeXtV2, and CAFormer stages
    to monotonic depth indices [0 .. max_depth] for LLRD.
    """
    if "patch_embed" in param_name or "stem" in param_name:
        return 0

    stage_match = re.search(r"(?:layers|stages)[_.](\d+)", param_name)
    if stage_match:
        stage_idx = int(stage_match.group(1))
        if stage_idx == 0:
            return 1
        if stage_idx == 1:
            return 2
        if stage_idx == 2:
            if "downsample" in param_name:
                return 3
            blk_match = re.search(r"blocks[_.](\d+)", param_name)
            if blk_match:
                blk_idx = int(blk_match.group(1))
                bin_size = max(1, (max(3, stage2_num_blocks) + 2) // 3)
                return 3 + min(2, blk_idx // bin_size)
            return 4
        return max_depth

    return max_depth


def _is_no_weight_decay_param(param_name: str, param: torch.Tensor) -> bool:
    if param.ndim <= 1:
        return True
    no_decay_tokens = (
        "bias",
        "norm",
        "logit_scale",
        "relative_position_bias_table",
        "absolute_pos_embed",
        "cpb_mlp",
        "gamma",
        "grn",
        "layer_scale",
        "res_scale",
    )
    lower_name = param_name.lower()
    return any(tok in lower_name for tok in no_decay_tokens)


def build_optimizer(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
) -> AdamW:
    """
    Construct AdamW with Layer-Wise Learning Rate Decay across SwinV2,
    ConvNeXtV2, or CAFormer backbones, higher learning rate for custom
    multi-scale fusion heads, and strict weight-decay exclusion on
    norms/biases/GRN/layer_scale/res_scale/1D parameters.
    """
    cfg = dict(MODEL_CONFIG)
    if config is not None:
        cfg.update(config)

    backbone_lr = float(cfg["backbone_lr"])
    head_lr = float(cfg["head_lr"])
    llrd_decay = float(cfg["llrd_decay"])
    weight_decay = float(cfg["weight_decay"])
    max_depth = 6

    max_s2_blk = -1
    for name, _ in model.named_parameters():
        if name.startswith("backbone."):
            m = re.search(r"(?:layers|stages)[_.]2\.blocks[_.](\d+)", name)
            if m:
                max_s2_blk = max(max_s2_blk, int(m.group(1)))
    stage2_num_blocks = max_s2_blk + 1 if max_s2_blk >= 0 else 18

    grouped: Dict[Tuple[float, float], List[torch.Tensor]] = {}
    trainable_count = 0
    seen_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        trainable_count += 1

        wd = 0.0 if _is_no_weight_decay_param(name, param) else weight_decay
        if name.startswith("backbone."):
            depth = _get_swin_layer_depth(
                name, max_depth=max_depth, stage2_num_blocks=stage2_num_blocks
            )
            lr = backbone_lr * (llrd_decay ** (max_depth - depth))
        else:
            lr = head_lr

        key = (round(lr, 10), wd)
        grouped.setdefault(key, []).append(param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in grouped.items()
    ]
    assigned_count = sum(len(g["params"]) for g in param_groups)
    assert (
        assigned_count == trainable_count
    ), f"Parameter grouping mismatch: assigned={assigned_count} vs trainable={trainable_count}"

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = MODEL_CONFIG["min_lr_ratio"],
) -> LambdaLR:
    """Linear warmup followed by smooth cosine annealing to `min_lr_ratio * base_lr`."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    min_ratio = float(min_lr_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class ModelEMA:
    """
    Warmup-aware Exponential Moving Average of model weights for smoother
    out-of-distribution generalization and calibrated probabilities at 0.5.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = MODEL_CONFIG["ema_decay"],
        device: Optional[torch.device] = None,
    ) -> None:
        self.decay = float(decay)
        self.updates = 0
        self.ema = copy.deepcopy(model).eval()
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        eff_decay = min(self.decay, float(1 + self.updates) / float(10 + self.updates))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(eff_decay).add_(src.to(v.dtype), alpha=1.0 - eff_decay)
            else:
                v.copy_(src)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "ema_state_dict": self.ema.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.ema.load_state_dict(state["ema_state_dict"])


def build_model_components(
    morph_dim: int = MODEL_CONFIG["morph_dim"],
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    total_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Instantiate MorphGatedSwinV2, DomainInvariantHistoLoss, LLRD AdamW optimizer,
    optional cosine warmup scheduler, and ModelEMA on the assigned process-local device.
    """
    cfg = dict(MODEL_CONFIG)
    cfg["morph_dim"] = int(morph_dim)
    if config is not None:
        cfg.update(config)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MorphGatedHistoNet(cfg).to(device)
    criterion = DomainInvariantHistoLoss(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    ema = ModelEMA(model, decay=float(cfg["ema_decay"]), device=device)

    scheduler = None
    if total_steps is not None:
        w_steps = (
            int(warmup_steps)
            if warmup_steps is not None
            else max(1, int(0.10 * total_steps))
        )
        scheduler = build_scheduler(
            optimizer,
            total_steps=int(total_steps),
            warmup_steps=w_steps,
            min_lr_ratio=float(cfg["min_lr_ratio"]),
        )

    pretrained_info = dict(PRETRAINED_MODEL_INFO)
    pretrained_info["resolved_backbone"] = model.resolved_backbone_name

    return {
        "model": model,
        "criterion": criterion,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "ema": ema,
        "device": device,
        "config": cfg,
        "pretrained_info": pretrained_info,
    }


# =============================================================================
# Step 3: Training, Pre-Refit Official Validation, Optional Refit & Submission
# =============================================================================
TRAIN_CONFIG: Dict[str, Any] = {
    "epochs": 8,
    "min_epochs": 4,
    "early_stopping_patience": 3,
    "warmup_epochs": 1.0,
    "domain_ramp_epochs": 2.0,
    "grad_clip_norm": 1.0,
    "monitor_tta_views": 1,
    "final_tta_views": 8,
    "ensemble_min_weight": 0.20,
    "ensemble_weight_step": 0.05,
    "do_post_selection_refit": True,
    "min_refit_epochs": 4,
    "max_refit_epochs": 6,
    "val_model_blend_weight": 0.40,
    "refit_model_blend_weight": 0.60,
    "min_refit_corr_threshold": 0.85,
    "max_total_seconds": 3150.0,
    "min_remaining_for_refit": 260.0,
}


def load_official_metric_fn(input_dir: str = "./input"):
    """
    Load the official `binary_metrics(labels, probabilities)` function directly
    from `./input/metric.py` so local validation matches private scoring exactly.
    """
    metric_path = Path(input_dir) / "metric.py"
    if metric_path.exists():
        spec = importlib.util.spec_from_file_location(
            "official_competition_metric", str(metric_path)
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "binary_metrics"):
                return module.binary_metrics

    def _fallback_binary_metrics(labels, probabilities):
        labels_arr = np.asarray(labels)
        probs_arr = np.asarray(probabilities, dtype=float)
        predicted = (probs_arr >= 0.5).astype(int)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(labels_arr, predicted)),
            "auroc": float(roc_auc_score(labels_arr, probs_arr)),
            "accuracy": float(accuracy_score(labels_arr, predicted)),
            "n_samples": int(len(labels_arr)),
        }

    return _fallback_binary_metrics


def compute_detailed_val_metrics(
    val_df: pd.DataFrame,
    probabilities: np.ndarray,
    metric_fn,
) -> Dict[str, Any]:
    """
    Evaluate official fixed-0.5 balanced accuracy, AUROC, and accuracy on the
    full validation set, plus post-hoc per-center diagnostics (never used as input).
    """
    labels = val_df["label"].values.astype(int)
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    overall = metric_fn(labels, probs)

    per_center: Dict[str, Dict[str, float]] = {}
    center_baccs: List[float] = []
    if "center_id" in val_df.columns:
        for cid in sorted(val_df["center_id"].unique().tolist()):
            mask = val_df["center_id"].values == cid
            c_labels = labels[mask]
            c_probs = probs[mask]
            if len(np.unique(c_labels)) == 2:
                c_res = metric_fn(c_labels, c_probs)
                per_center[str(cid)] = {
                    "balanced_accuracy": float(c_res["balanced_accuracy"]),
                    "auroc": float(c_res["auroc"]),
                    "n_samples": int(c_res["n_samples"]),
                }
                center_baccs.append(float(c_res["balanced_accuracy"]))

    worst_center_bacc = (
        float(min(center_baccs))
        if center_baccs
        else float(overall["balanced_accuracy"])
    )
    center_gap = (
        float(max(center_baccs) - min(center_baccs)) if len(center_baccs) >= 2 else 0.0
    )

    return {
        "balanced_accuracy": float(overall["balanced_accuracy"]),
        "auroc": float(overall["auroc"]),
        "accuracy": float(overall["accuracy"]),
        "n_samples": int(overall["n_samples"]),
        "per_center": per_center,
        "worst_center_bacc": worst_center_bacc,
        "center_gap": center_gap,
        "pred_pos_rate": float((probs >= 0.5).mean()),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
    }


def _configure_amp(device: torch.device) -> Tuple[bool, torch.dtype, Any]:
    """Configure device-safe mixed precision (bf16 if supported, else fp16+GradScaler, else fp32)."""
    if device.type != "cuda":
        return False, torch.float32, None

    bf16_ok = (
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    if bf16_ok:
        return True, torch.bfloat16, None

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    return True, torch.float16, scaler


def _safe_load_state_dict(
    path: Path, map_location: torch.device
) -> Dict[str, torch.Tensor]:
    """Load pure tensor state_dict safely across PyTorch versions."""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception:
        return torch.load(str(path), map_location=map_location, weights_only=False)


@torch.no_grad()
def predict_with_d4_tta(
    eval_model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    n_views: int = 8,
) -> Tuple[List[str], np.ndarray]:
    """
    Strictly pixel-only inference function shared IDENTICALLY by validation and
    test loaders. Uses `generate_d4_tta_views` and averages tumor probabilities
    across `n_views` exact D4 dihedral orientations.
    """
    eval_model.eval()
    all_sample_ids: List[str] = []
    all_probs: List[np.ndarray] = []
    n_views = max(1, min(8, int(n_views)))

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        morph = batch["morph"].to(device, non_blocking=True)

        if n_views == 1:
            views = [images]
        elif n_views == 2:
            views = [
                images,
                torch.flip(torch.rot90(images, k=1, dims=(2, 3)), dims=(3,)),
            ]
        else:
            views = generate_d4_tta_views(images)[:n_views]

        prob_accum = torch.zeros(images.size(0), device=device, dtype=torch.float32)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            for view in views:
                logits = eval_model(view, morph)
                prob_accum += torch.sigmoid(logits.float())

        batch_probs = (prob_accum / float(len(views))).detach().cpu().numpy()
        all_probs.append(batch_probs.astype(np.float64))
        all_sample_ids.extend(list(batch["sample_id"]))

    probs_concat = np.concatenate(all_probs, axis=0)
    probs_concat = np.clip(probs_concat, 0.0, 1.0)
    if not np.isfinite(probs_concat).all():
        raise RuntimeError("Non-finite probabilities detected during inference.")
    return all_sample_ids, probs_concat


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema: Any,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    scaler: Any,
    epoch_idx: int,
    domain_ramp_epochs: float,
    grad_clip_norm: float,
    mixup_fraction: float = MODEL_CONFIG["dual_mixup_fraction"],
    mixup_alpha: float = MODEL_CONFIG["dual_mixup_alpha"],
    mixup_cross_center_p: float = MODEL_CONFIG["dual_mixup_cross_center_p"],
) -> Dict[str, float]:
    """
    Run one full training epoch with Cross-Center Dual-Modality Mixup, AMP,
    smooth domain-loss ramp, gradient clipping, per-step cosine warmup
    scheduling, and Model EMA updates.
    """
    model.train()
    num_steps = max(1, len(train_loader))
    ramp_steps = max(1.0, float(domain_ramp_epochs) * float(num_steps))

    running = {
        "total": 0.0,
        "bce": 0.0,
        "rank": 0.0,
        "vrex": 0.0,
        "align": 0.0,
        "groupdro": 0.0,
        "midpoint": 0.0,
        "morph_aux": 0.0,
        "vis_aux": 0.0,
    }
    valid_steps = 0

    for step_idx, batch in enumerate(train_loader):
        global_step = epoch_idx * num_steps + step_idx
        domain_scale = min(1.0, float(global_step + 1) / ramp_steps)

        images = batch["image"].to(device, non_blocking=True)
        morph = batch["morph"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        center_idx = (
            batch["center_idx"].to(device, non_blocking=True)
            if "center_idx" in batch
            else None
        )

        mixed_images, mixed_morph, mixed_labels, unmixed_mask = (
            apply_cross_center_dual_mixup(
                images=images,
                morph=morph,
                labels=labels,
                center_idx=center_idx,
                mix_fraction=mixup_fraction,
                alpha=mixup_alpha,
                cross_center_p=mixup_cross_center_p,
            )
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            outputs = model(
                mixed_images, mixed_morph, center_idx=center_idx, return_dict=True
            )
            loss, stats = criterion(
                outputs,
                mixed_labels,
                center_idx=center_idx,
                domain_weight_scale=domain_scale,
                unmixed_mask=unmixed_mask,
            )

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        for k in running:
            running[k] += float(stats.get(k, 0.0))
        valid_steps += 1

    denom = float(max(1, valid_steps))
    return {k: v / denom for k, v in running.items()}


def save_and_verify_submission(
    sample_ids: List[str],
    probabilities: np.ndarray,
    sample_sub_df: pd.DataFrame,
    submission_path: Path,
) -> pd.DataFrame:
    """
    Write `./submission/submission.csv` and verify exact schema, ID alignment,
    probability range [0, 1], finiteness, and non-degeneracy.
    """
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)

    if len(sample_ids) != len(sample_sub_df):
        raise ValueError(
            f"Prediction count mismatch: got {len(sample_ids)}, expected {len(sample_sub_df)}"
        )
    if not (np.asarray(sample_ids) == sample_sub_df["sample_id"].values).all():
        pred_df = pd.DataFrame({"sample_id": sample_ids, "tumor_probability": probs})
        sub_df = sample_sub_df[["sample_id"]].merge(
            pred_df, on="sample_id", how="left", validate="one_to_one"
        )
    else:
        sub_df = pd.DataFrame(
            {"sample_id": sample_sub_df["sample_id"].values, "tumor_probability": probs}
        )

    if list(sub_df.columns) != ["sample_id", "tumor_probability"]:
        raise ValueError(f"Invalid submission columns: {list(sub_df.columns)}")
    if sub_df["tumor_probability"].isna().any():
        raise ValueError("Submission contains NaN probabilities.")
    vals = sub_df["tumor_probability"].values.astype(np.float64)
    if not np.isfinite(vals).all() or ((vals < 0.0) | (vals > 1.0)).any():
        raise ValueError("Submission probabilities must be finite and within [0, 1].")
    if float(vals.std()) < 1e-4:
        raise ValueError(
            f"Submission probabilities appear degenerate (std={vals.std():.6f})."
        )

    sub_df.to_csv(submission_path, index=False)

    check_df = pd.read_csv(submission_path)
    assert len(check_df) == len(sample_sub_df), "Saved submission row count mismatch."
    assert list(check_df.columns) == ["sample_id", "tumor_probability"]
    assert (check_df["sample_id"].values == sample_sub_df["sample_id"].values).all()
    assert np.isfinite(check_df["tumor_probability"].values).all()
    return check_df


def _cleanup_gpu_memory(*objs: Any) -> None:
    """Move any torch modules to CPU, drop references, and reclaim GPU cache."""
    for obj in objs:
        if obj is None:
            continue
        if isinstance(obj, nn.Module):
            try:
                obj.to("cpu")
            except Exception:
                pass
        elif hasattr(obj, "ema") and isinstance(obj.ema, nn.Module):
            try:
                obj.ema.to("cpu")
            except Exception:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _optimize_constrained_simplex_weights(
    val_probs_list: List[np.ndarray],
    val_df: pd.DataFrame,
    metric_fn,
    min_weight: float = 0.20,
    step: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Constrained simplex grid search (step=0.05, each weight >= min_weight, sum=1.0)
    evaluating official fixed-0.5 balanced accuracy and tie-breaking by
    (balanced_accuracy, worst_center_bacc, auroc, -L2_distance_to_uniform).
    """
    n_models = len(val_probs_list)
    if n_models == 0:
        raise ValueError("Cannot optimize ensemble weights with zero models.")
    if n_models == 1:
        w = np.array([1.0], dtype=np.float64)
        probs = np.clip(val_probs_list[0].astype(np.float64), 0.0, 1.0)
        return w, compute_detailed_val_metrics(val_df, probs, metric_fn), probs

    scale = int(round(1.0 / float(step)))
    min_int = int(round(float(min_weight) * scale))
    if min_int * n_models > scale:
        min_int = scale // n_models

    grid_vectors: List[np.ndarray] = []

    def _recurse(depth: int, remaining: int, current: List[int]) -> None:
        if depth == n_models - 1:
            if remaining >= min_int:
                grid_vectors.append(
                    np.array(current + [remaining], dtype=np.float64) / float(scale)
                )
            return
        max_here = remaining - min_int * (n_models - 1 - depth)
        for val in range(min_int, max_here + 1):
            _recurse(depth + 1, remaining - val, current + [val])

    _recurse(0, scale, [])
    if not grid_vectors:
        grid_vectors.append(np.full(n_models, 1.0 / float(n_models), dtype=np.float64))

    uniform = np.full(n_models, 1.0 / float(n_models), dtype=np.float64)
    stacked = np.stack([p.astype(np.float64) for p in val_probs_list], axis=0)

    best_key = (-1.0, -1.0, -1.0, -1e9)
    best_weights = uniform.copy()
    best_metrics: Optional[Dict[str, Any]] = None
    best_probs: Optional[np.ndarray] = None

    for w in grid_vectors:
        ens_probs = np.clip(np.tensordot(w, stacked, axes=(0, 0)), 0.0, 1.0)
        metrics = compute_detailed_val_metrics(val_df, ens_probs, metric_fn)
        neg_l2 = -float(np.sum((w - uniform) ** 2))
        key = (
            round(float(metrics["balanced_accuracy"]), 12),
            round(float(metrics["worst_center_bacc"]), 12),
            round(float(metrics["auroc"]), 12),
            round(neg_l2, 12),
        )
        if key > best_key:
            best_key = key
            best_weights = w.copy()
            best_metrics = metrics
            best_probs = ens_probs

    assert best_metrics is not None and best_probs is not None
    return best_weights, best_metrics, best_probs


def _train_single_backbone_stage1(
    spec: Dict[str, Any],
    data_bundle: Dict[str, Any],
    cfg: Dict[str, Any],
    metric_fn,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    working_dir: Path,
    run_start_time: float,
    stage1_backbone_deadline: float,
) -> Dict[str, Any]:
    """
    Stage 1 training for one MorphGatedHistoNet backbone on train.csv with
    1-view EMA-only per-epoch monitoring, best EMA checkpoint selection by
    (balanced_accuracy, worst_center_bacc, auroc), identical 8-view D4-TTA on
    val_loader and test_loader, disk persistence, and explicit GPU cleanup.
    """
    tag = str(spec["tag"])
    seed = int(data_bundle["config"]["seed"])
    seed_everything(seed)

    val_df = data_bundle["val_df"]
    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]

    epochs = int(cfg["epochs"])
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(float(cfg["warmup_epochs"]) * len(train_loader)))
    _, _, scaler = _configure_amp(device)

    model_cfg_override = {
        "backbone_name": spec["backbone_name"],
        "backbone_candidates": spec.get("backbone_candidates", None),
        "img_size": int(data_bundle["config"]["img_size"]),
    }
    components = build_model_components(
        morph_dim=int(data_bundle["morph_dim"]),
        config=model_cfg_override,
        device=device,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )
    model = components["model"]
    criterion = components["criterion"]
    optimizer = components["optimizer"]
    scheduler = components["scheduler"]
    ema = components["ema"]
    pretrained_info = components["pretrained_info"]
    resolved_backbone = model.resolved_backbone_name

    print(
        f"[Stage 1 | {tag}] Training MorphGatedHistoNet ({resolved_backbone}) "
        f"| remap={model.use_input_remap} | device={device} "
        f"| amp={amp_dtype if use_amp else 'fp32'} | train={len(data_bundle['train_df'])} "
        f"| val={len(val_df)} | epochs={epochs}"
    )

    best_ema_key = (-1.0, -1.0, -1.0)
    best_epoch = 1
    epochs_no_improve = 0
    best_ema_ckpt_path = working_dir / f"best_ema_{tag}.pt"
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        ep_t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            train_loader=train_loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            epoch_idx=epoch,
            domain_ramp_epochs=float(cfg["domain_ramp_epochs"]),
            grad_clip_norm=float(cfg["grad_clip_norm"]),
        )

        ema_ids, ema_probs = predict_with_d4_tta(
            ema.ema,
            val_loader,
            device,
            use_amp,
            amp_dtype,
            n_views=int(cfg["monitor_tta_views"]),
        )
        assert (
            np.asarray(ema_ids) == val_df["sample_id"].values
        ).all(), "Validation sample_id order mismatch!"
        ema_metrics = compute_detailed_val_metrics(val_df, ema_probs, metric_fn)

        ema_key = (
            ema_metrics["balanced_accuracy"],
            ema_metrics["worst_center_bacc"],
            ema_metrics["auroc"],
        )

        saved_tag = ""
        if ema_key > best_ema_key:
            best_ema_key = ema_key
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(ema.ema.state_dict(), str(best_ema_ckpt_path))
            saved_tag = "*EMA"
        else:
            epochs_no_improve += 1

        ep_sec = time.time() - ep_t0
        cur_lr = float(optimizer.param_groups[-1]["lr"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_stats["total"],
                "train_bce": train_stats["bce"],
                "train_vrex": train_stats["vrex"],
                "train_align": train_stats["align"],
                "train_groupdro": train_stats["groupdro"],
                "train_midpoint": train_stats["midpoint"],
                "ema_bacc": ema_metrics["balanced_accuracy"],
                "ema_auroc": ema_metrics["auroc"],
                "ema_worst_c": ema_metrics["worst_center_bacc"],
                "ema_gap": ema_metrics["center_gap"],
                "epoch_sec": ep_sec,
            }
        )

        print(
            f"[{tag}] Epoch {epoch + 1:02d}/{epochs:02d} | loss={train_stats['total']:.4f} "
            f"(bce={train_stats['bce']:.4f}, gdro={train_stats['groupdro']:.4f}, "
            f"mid={train_stats['midpoint']:.4f}, vrex={train_stats['vrex']:.4f}, "
            f"align={train_stats['align']:.4f}) "
            f"| val_ema_bacc={ema_metrics['balanced_accuracy']:.4f} "
            f"(auc={ema_metrics['auroc']:.4f}, min_c={ema_metrics['worst_center_bacc']:.4f}, "
            f"gap={ema_metrics['center_gap']:.4f}) "
            f"| lr={cur_lr:.2e} | {ep_sec:.1f}s {saved_tag}"
        )

        now = time.time()
        if (epoch + 1) >= int(cfg["min_epochs"]) and epochs_no_improve >= int(
            cfg["early_stopping_patience"]
        ):
            print(
                f"[{tag}] Early stopping at epoch {epoch + 1} "
                f"(no EMA improvement for {epochs_no_improve} epochs)."
            )
            break
        if (epoch + 1) >= int(cfg["min_epochs"]) and now > stage1_backbone_deadline:
            print(
                f"[{tag}] Reached Stage 1 per-backbone time budget at epoch {epoch + 1}; "
                f"proceeding to 8x D4-TTA evaluation."
            )
            break
        if (now - run_start_time) > float(cfg["max_total_seconds"]) - 180.0:
            print(
                f"[{tag}] Stopping training at epoch {epoch + 1} to preserve total time budget."
            )
            break

    if not best_ema_ckpt_path.exists():
        torch.save(ema.ema.state_dict(), str(best_ema_ckpt_path))

    ema.ema.load_state_dict(_safe_load_state_dict(best_ema_ckpt_path, device))
    val_ids, val_probs_8x = predict_with_d4_tta(
        ema.ema,
        val_loader,
        device,
        use_amp,
        amp_dtype,
        n_views=int(cfg["final_tta_views"]),
    )
    assert (
        np.asarray(val_ids) == val_df["sample_id"].values
    ).all(), "Validation ID alignment error during 8x TTA."
    val_metrics_8x = compute_detailed_val_metrics(val_df, val_probs_8x, metric_fn)

    c_summary = ", ".join(
        f"{k}:{v['balanced_accuracy']:.4f}"
        for k, v in val_metrics_8x["per_center"].items()
    )
    print(
        f"[{tag}] Pre-Refit 8x D4-TTA [EMA ep={best_epoch}] | "
        f"val_bacc={val_metrics_8x['balanced_accuracy']:.4f} "
        f"| auroc={val_metrics_8x['auroc']:.4f} "
        f"| per_center=({c_summary}) | gap={val_metrics_8x['center_gap']:.4f}"
    )

    test_ids, test_probs_8x = predict_with_d4_tta(
        ema.ema,
        test_loader,
        device,
        use_amp,
        amp_dtype,
        n_views=int(cfg["final_tta_views"]),
    )

    np.save(working_dir / f"val_probs_{tag}_8x.npy", val_probs_8x)
    np.save(working_dir / f"test_probs_{tag}_8x.npy", test_probs_8x)

    avg_ep_sec = float(np.mean([h["epoch_sec"] for h in history])) if history else 60.0

    _cleanup_gpu_memory(model, ema)
    del model, criterion, optimizer, scheduler, ema, components, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "tag": tag,
        "spec": spec,
        "resolved_backbone": resolved_backbone,
        "pretrained_info": pretrained_info,
        "best_epoch": best_epoch,
        "ckpt_path": str(best_ema_ckpt_path),
        "val_metrics": val_metrics_8x,
        "val_probs": val_probs_8x,
        "test_probs": test_probs_8x,
        "test_ids": test_ids,
        "avg_epoch_sec": avg_ep_sec,
        "history": history,
    }


def run_training_and_evaluation(
    data_bundle: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end 3-architecture MorphGatedHistoNet training on train.csv,
    pre-refit 8x D4-TTA constrained simplex ensembling to freeze official
    Final Validation Score and immediately write a verified submission, followed
    by time-guarded per-backbone Train+Val refits with correlation-gated blending.
    """
    run_start_time = time.time()
    cfg = dict(TRAIN_CONFIG)
    if train_config is not None:
        cfg.update(train_config)

    if data_bundle is None:
        data_bundle = build_data_pipeline(force_recompute_morph=False)

    data_cfg = data_bundle["config"]
    seed = int(data_cfg["seed"])
    seed_everything(seed)

    working_dir = Path(data_cfg["working_dir"])
    submission_dir = Path(data_cfg["submission_dir"])
    working_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_path = submission_dir / "submission.csv"

    metric_fn = load_official_metric_fn(data_cfg["input_dir"])
    val_df = data_bundle["val_df"]
    sample_sub_df = data_bundle["sample_sub_df"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp, amp_dtype, _ = _configure_amp(device)

    ensemble_specs: List[Dict[str, Any]] = list(MODEL_CONFIG["ensemble_specs"])
    n_models = len(ensemble_specs)
    stage1_total_budget = float(cfg["max_total_seconds"]) * 0.68

    stage1_results: List[Dict[str, Any]] = []
    for idx, spec in enumerate(ensemble_specs):
        elapsed = time.time() - run_start_time
        remaining_stage1 = max(120.0, stage1_total_budget - elapsed)
        per_model_share = remaining_stage1 / float(max(1, n_models - idx))
        backbone_deadline = time.time() + per_model_share

        res = _train_single_backbone_stage1(
            spec=spec,
            data_bundle=data_bundle,
            cfg=cfg,
            metric_fn=metric_fn,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            working_dir=working_dir,
            run_start_time=run_start_time,
            stage1_backbone_deadline=backbone_deadline,
        )
        stage1_results.append(res)

    test_ids = stage1_results[0]["test_ids"]
    for res in stage1_results[1:]:
        assert (
            np.asarray(res["test_ids"]) == np.asarray(test_ids)
        ).all(), "Test sample_id order mismatch across ensemble backbones!"

    val_probs_list = [res["val_probs"] for res in stage1_results]
    pre_refit_test_probs_list = [res["test_probs"] for res in stage1_results]

    simplex_weights, ens_val_metrics, ens_val_probs = (
        _optimize_constrained_simplex_weights(
            val_probs_list=val_probs_list,
            val_df=val_df,
            metric_fn=metric_fn,
            min_weight=float(cfg["ensemble_min_weight"]),
            step=float(cfg["ensemble_weight_step"]),
        )
    )

    final_val_score = float(ens_val_metrics["balanced_accuracy"])
    weight_summary = ", ".join(
        f"{res['tag']}({res['resolved_backbone']}):{w:.2f}"
        for res, w in zip(stage1_results, simplex_weights)
    )
    ens_c_summary = ", ".join(
        f"{k}:{v['balanced_accuracy']:.4f}"
        for k, v in ens_val_metrics["per_center"].items()
    )
    print(
        f"Pre-Refit 3-Backbone 8x D4-TTA Ensemble | weights=[{weight_summary}] "
        f"| val_bacc={ens_val_metrics['balanced_accuracy']:.4f} "
        f"| auroc={ens_val_metrics['auroc']:.4f} "
        f"| per_center=({ens_c_summary}) | gap={ens_val_metrics['center_gap']:.4f}"
    )

    pre_refit_ens_test_probs = np.clip(
        sum(w * p for w, p in zip(simplex_weights, pre_refit_test_probs_list)),
        0.0,
        1.0,
    )
    np.save(working_dir / "val_oof_probs_8x.npy", ens_val_probs)
    np.save(working_dir / "test_probs_val_model_8x.npy", pre_refit_ens_test_probs)

    final_test_probs = pre_refit_ens_test_probs.copy()
    save_and_verify_submission(
        test_ids, final_test_probs, sample_sub_df, submission_path
    )
    print(
        f"Pre-refit baseline submission written and verified at {submission_path} "
        f"(elapsed={time.time() - run_start_time:.1f}s)."
    )

    # Stage 2: Time-guarded post-selection Train+Val refits using one shared loader bundle
    per_backbone_final_test_probs = [p.copy() for p in pre_refit_test_probs_list]
    refit_records: List[Dict[str, Any]] = []
    any_refit_used = False

    elapsed_before_refit = time.time() - run_start_time
    remaining_seconds = float(cfg["max_total_seconds"]) - elapsed_before_refit

    if bool(cfg["do_post_selection_refit"]) and remaining_seconds >= float(
        cfg["min_remaining_for_refit"]
    ):
        refit_data = build_refit_data_loaders(data_bundle, use_balanced_sampler=True)
        tv_loader = refit_data["train_val_loader"]
        test_loader_refit = refit_data["test_loader_refit"]
        tv_scale_factor = float(len(data_bundle["train_val_df"])) / float(
            max(1, len(data_bundle["train_df"]))
        )

        # Refit in descending order of locked ensemble weight
        refit_order = sorted(
            range(len(stage1_results)),
            key=lambda i: (-float(simplex_weights[i]), i),
        )

        for order_pos, b_idx in enumerate(refit_order):
            res = stage1_results[b_idx]
            tag = res["tag"]
            spec = res["spec"]
            rem_now = float(cfg["max_total_seconds"]) - (time.time() - run_start_time)

            target_refit_epochs = int(
                min(
                    max(int(res["best_epoch"]), int(cfg["min_refit_epochs"])),
                    int(cfg["max_refit_epochs"]),
                )
            )
            est_refit_ep_sec = max(15.0, float(res["avg_epoch_sec"]) * tv_scale_factor)
            max_affordable_epochs = int((rem_now - 95.0) / est_refit_ep_sec)
            refit_epochs = min(target_refit_epochs, max_affordable_epochs)

            if refit_epochs < int(cfg["min_refit_epochs"]):
                print(
                    f"[Stage 2 | {tag}] Skipping refit (remaining={rem_now:.0f}s, "
                    f"affordable_epochs={max_affordable_epochs} < {cfg['min_refit_epochs']})."
                )
                refit_records.append(
                    {
                        "tag": tag,
                        "resolved_backbone": res["resolved_backbone"],
                        "refit_executed": False,
                        "refit_blended": False,
                        "refit_epochs": 0,
                        "test_prob_corr": None,
                    }
                )
                continue

            seed_everything(seed + 100 + b_idx * 17)
            refit_total_steps = refit_epochs * max(1, len(tv_loader))
            refit_warmup_steps = max(
                1, int(float(cfg["warmup_epochs"]) * len(tv_loader))
            )
            _, _, refit_scaler = _configure_amp(device)
            refit_comp = build_model_components(
                morph_dim=int(data_bundle["morph_dim"]),
                config={
                    "backbone_name": spec["backbone_name"],
                    "backbone_candidates": spec.get("backbone_candidates", None),
                    "img_size": int(data_bundle["config"]["img_size"]),
                },
                device=device,
                total_steps=refit_total_steps,
                warmup_steps=refit_warmup_steps,
            )
            refit_model = refit_comp["model"]
            refit_criterion = refit_comp["criterion"]
            refit_optimizer = refit_comp["optimizer"]
            refit_scheduler = refit_comp["scheduler"]
            refit_ema = refit_comp["ema"]

            print(
                f"[Stage 2 | {tag}] Starting Train+Val refit ({refit_model.resolved_backbone_name}) "
                f"| samples={len(data_bundle['train_val_df'])} | refit_epochs={refit_epochs} "
                f"| remaining={rem_now:.0f}s"
            )

            epochs_completed = 0
            for r_ep in range(refit_epochs):
                r_t0 = time.time()
                r_stats = train_one_epoch(
                    model=refit_model,
                    criterion=refit_criterion,
                    optimizer=refit_optimizer,
                    scheduler=refit_scheduler,
                    ema=refit_ema,
                    train_loader=tv_loader,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    scaler=refit_scaler,
                    epoch_idx=r_ep,
                    domain_ramp_epochs=float(cfg["domain_ramp_epochs"]),
                    grad_clip_norm=float(cfg["grad_clip_norm"]),
                )
                epochs_completed += 1
                r_sec = time.time() - r_t0
                r_lr = float(refit_optimizer.param_groups[-1]["lr"])
                print(
                    f"[Refit {tag}] Epoch {r_ep + 1:02d}/{refit_epochs:02d} "
                    f"| loss={r_stats['total']:.4f} (bce={r_stats['bce']:.4f}, "
                    f"gdro={r_stats['groupdro']:.4f}, mid={r_stats['midpoint']:.4f}, "
                    f"vrex={r_stats['vrex']:.4f}, align={r_stats['align']:.4f}) "
                    f"| lr={r_lr:.2e} | {r_sec:.1f}s"
                )
                if (time.time() - run_start_time) > float(
                    cfg["max_total_seconds"]
                ) - 85.0:
                    print(
                        f"[Refit {tag}] Halting refit loop after epoch {r_ep + 1} to preserve inference budget."
                    )
                    break

            refit_ckpt_path = working_dir / f"refit_ema_{tag}.pt"
            torch.save(refit_ema.ema.state_dict(), str(refit_ckpt_path))

            refit_test_ids, refit_test_probs = predict_with_d4_tta(
                refit_ema.ema,
                test_loader_refit,
                device,
                use_amp,
                amp_dtype,
                n_views=int(cfg["final_tta_views"]),
            )
            assert (
                np.asarray(refit_test_ids) == np.asarray(test_ids)
            ).all(), f"Refit test sample_id order mismatch for {tag}!"
            np.save(working_dir / f"test_probs_refit_{tag}_8x.npy", refit_test_probs)

            base_probs = pre_refit_test_probs_list[b_idx]
            if (
                float(np.std(refit_test_probs)) > 1e-3
                and float(np.std(base_probs)) > 1e-3
            ):
                refit_corr = float(np.corrcoef(base_probs, refit_test_probs)[0, 1])
            else:
                refit_corr = 0.0

            blended = False
            if np.isfinite(refit_corr) and refit_corr >= float(
                cfg["min_refit_corr_threshold"]
            ):
                w_val = float(cfg["val_model_blend_weight"])
                w_ref = float(cfg["refit_model_blend_weight"])
                per_backbone_final_test_probs[b_idx] = np.clip(
                    (w_val * base_probs + w_ref * refit_test_probs) / (w_val + w_ref),
                    0.0,
                    1.0,
                )
                blended = True
                any_refit_used = True
                print(
                    f"[Refit {tag}] Blended ({w_val:.2f}/{w_ref:.2f}) | "
                    f"test_prob_corr={refit_corr:.4f}"
                )
            else:
                print(
                    f"[Refit {tag}] Skipped blend due to low correlation ({refit_corr:.4f}); "
                    f"kept validated Stage 1 predictions."
                )

            refit_records.append(
                {
                    "tag": tag,
                    "resolved_backbone": res["resolved_backbone"],
                    "refit_executed": True,
                    "refit_blended": blended,
                    "refit_epochs": epochs_completed,
                    "test_prob_corr": None
                    if not np.isfinite(refit_corr)
                    else refit_corr,
                }
            )

            _cleanup_gpu_memory(refit_model, refit_ema)
            del (
                refit_model,
                refit_criterion,
                refit_optimizer,
                refit_scheduler,
                refit_ema,
                refit_comp,
                refit_scaler,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if any_refit_used:
            final_test_probs = np.clip(
                sum(
                    w * p
                    for w, p in zip(simplex_weights, per_backbone_final_test_probs)
                ),
                0.0,
                1.0,
            )
            np.save(working_dir / "test_probs_refit_model_8x.npy", final_test_probs)
            save_and_verify_submission(
                test_ids, final_test_probs, sample_sub_df, submission_path
            )
            print(
                f"Updated final submission with locked simplex weights and correlation-gated refits."
            )

    total_runtime = time.time() - run_start_time
    report = {
        "pretrained_model_info": PRETRAINED_MODEL_INFO,
        "ensemble_backbones": [
            {
                "tag": res["tag"],
                "resolved_backbone": res["resolved_backbone"],
                "weight": float(w),
                "best_epoch": int(res["best_epoch"]),
                "pre_refit_validation_metrics": res["val_metrics"],
                "history": res["history"],
            }
            for res, w in zip(stage1_results, simplex_weights)
        ],
        "locked_simplex_weights": {
            res["tag"]: float(w) for res, w in zip(stage1_results, simplex_weights)
        },
        "pre_refit_validation_metrics": ens_val_metrics,
        "refit_used": any_refit_used,
        "refit_details": refit_records,
        "submission_path": str(submission_path),
        "submission_rows": int(len(final_test_probs)),
        "submission_pred_pos_rate": float((final_test_probs >= 0.5).mean()),
        "submission_prob_mean": float(final_test_probs.mean()),
        "submission_prob_std": float(final_test_probs.std()),
        "total_runtime_sec": float(total_runtime),
    }
    with open(working_dir / "training_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(
        f"Submission verified at {submission_path} | rows={len(final_test_probs)} "
        f"| mean={final_test_probs.mean():.4f} | std={final_test_probs.std():.4f} "
        f"| pos_rate={(final_test_probs >= 0.5).mean():.4f} | runtime={total_runtime:.1f}s"
    )
    print(f"Final Validation Score: {final_val_score}")
    return report


if __name__ == "__main__":
    run_training_and_evaluation()
