import copy
import gc
import importlib.util
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# =============================================================================
# 1. Global Configuration, Device & Reproducibility
# =============================================================================
INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

SEED = 42
IMG_SIZE = 252
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = (
    torch.bfloat16
    if (USE_AMP and torch.cuda.is_bf16_supported())
    else (torch.float16 if USE_AMP else torch.float32)
)
USE_GRAD_SCALER = USE_AMP and (AMP_DTYPE == torch.float16)

VAL_MONITOR_TTA_VIEWS = 1
FINAL_TTA_VIEWS = 8
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 5
MIN_EPOCHS = 8
TRAIN_TIME_LIMIT_SEC = 2100.0  # 35 min ceiling for Phase 1
TOTAL_TIME_LIMIT_SEC = 3050.0  # ~51 min ceiling including optional refit

PRETRAINED_MODEL_CARD = {
    "backbone_primary": "vit_base_patch14_reg4_dinov2.lvd142m",
    "backbone_fallbacks": [
        "vit_base_patch14_dinov2.lvd142m",
        "vit_base_patch14_reg4_dinov2",
        "vit_base_patch14_dinov2",
    ],
    "library": f"timm=={timm.__version__}",
    "pretraining_data": "LVD-142M curated image dataset via self-supervised DINOv2",
    "known_histopathology_overlap": "None (pretrained via self-supervised learning on LVD-142M)",
}

MODEL_CONFIG = {
    "backbone_name": PRETRAINED_MODEL_CARD["backbone_primary"],
    "pretrained": True,
    "drop_path_rate": 0.20,
    "use_mixstyle": True,
    "mixstyle_p": 0.5,
    "mixstyle_alpha": 0.1,
    "label_smoothing": 0.04,
    "coral_weight": 0.15,
    "logit_cons_weight": 0.25,
    "feat_cons_weight": 0.50,
    "cons_warmup_ratio": 0.15,
    "backbone_lr": 5.0e-5,
    "head_lr": 3.0e-4,
    "weight_decay": 5.0e-2,
    "layer_decay": 0.86,
    "ema_decay": 0.996,
    "default_epochs": 16,
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_competition_metric(input_dir: str = INPUT_DIR):
    metric_spec = importlib.util.spec_from_file_location(
        "competition_metric", os.path.join(input_dir, "metric.py")
    )
    competition_metric = importlib.util.module_from_spec(metric_spec)
    metric_spec.loader.exec_module(competition_metric)
    return competition_metric.binary_metrics


# =============================================================================
# 2. Data Loading, Integrity Verification & Leakage-Free Split Construction
# =============================================================================
def compute_balanced_sample_weights(
    df: pd.DataFrame, slide_power: float = 0.5
) -> np.ndarray:
    """
    Compute per-sample training weights that:
      1. Balance (center_id, label) joint distribution (directly supporting balanced accuracy
         and equal representation of development medical centers), and
      2. Dampen dominance of slides with disproportionately many patches via n_slide**(-slide_power).
    """
    combo = df["center_id"].astype(str) + "__" + df["label"].astype(str)
    combo_counts = combo.map(combo.value_counts()).values.astype(np.float64)
    slide_counts = (
        df["slide_id"].map(df["slide_id"].value_counts()).values.astype(np.float64)
    )

    weights = (1.0 / combo_counts) * (slide_counts ** (-slide_power))
    weights = weights / np.mean(weights)
    return weights.astype(np.float64)


def load_and_validate_metadata(input_dir: str = INPUT_DIR):
    train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(input_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(input_dir, "test.csv"))
    sample_sub_df = pd.read_csv(os.path.join(input_dir, "sample_submission.csv"))

    for df in (train_df, val_df, test_df):
        df["full_path"] = df["image_path"].apply(lambda p: os.path.join(input_dir, p))

    # Strict integrity checks
    assert train_df["sample_id"].is_unique, "Duplicate sample_id in train.csv"
    assert val_df["sample_id"].is_unique, "Duplicate sample_id in validation.csv"
    assert test_df["sample_id"].is_unique, "Duplicate sample_id in test.csv"
    assert (
        test_df["sample_id"].values == sample_sub_df["sample_id"].values
    ).all(), "test.csv and sample_submission.csv sample_id order mismatch!"

    # Ensure zero sample, slide, or group leakage between fixed train and validation sets
    assert (
        len(set(train_df["sample_id"]) & set(val_df["sample_id"])) == 0
    ), "Sample leakage!"
    assert (
        len(set(train_df["slide_id"]) & set(val_df["slide_id"])) == 0
    ), "Slide leakage!"
    assert (
        len(set(train_df["group_id"]) & set(val_df["group_id"])) == 0
    ), "Group leakage!"

    # Verify all image files exist on disk
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        missing_mask = ~df["full_path"].apply(os.path.exists)
        if missing_mask.any():
            raise FileNotFoundError(
                f"{missing_mask.sum()} missing image files detected in {name} split!"
            )

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    # Encode center_id strictly for training-time sampling/splitting (never an inference feature)
    unique_centers = sorted(train_df["center_id"].unique().tolist())
    center_to_idx = {c: i for i, c in enumerate(unique_centers)}
    train_df["center_idx"] = train_df["center_id"].map(center_to_idx).astype(int)
    val_df["center_idx"] = val_df["center_id"].map(center_to_idx).fillna(-1).astype(int)

    # Leave-One-Center-Out (LOCO) split inside train_df for unseen-center ablation/validation
    train_df["loco_fold"] = train_df["center_idx"].astype(int)

    # Group-stratified K-Fold inside train_df (keeping group_id intact)
    n_train_splits = min(5, train_df["group_id"].nunique())
    sgkf_train = StratifiedGroupKFold(n_splits=n_train_splits)
    train_df["group_fold"] = -1
    strat_key_train = (
        train_df["center_idx"].astype(str) + "_" + train_df["label"].astype(str)
    )
    for fold_idx, (_, va_idx) in enumerate(
        sgkf_train.split(train_df, strat_key_train, groups=train_df["group_id"])
    ):
        train_df.loc[va_idx, "group_fold"] = fold_idx

    # Compute center- & slide-balanced sampling weights (fit strictly on train_df)
    train_df["sample_weight"] = compute_balanced_sample_weights(train_df)

    # Merged full training dataset (train + validation) strictly for optional post-selection refit
    train_copy = train_df.copy()
    train_copy["orig_split"] = "train"
    val_copy = val_df.copy()
    val_copy["orig_split"] = "validation"
    val_copy["loco_fold"] = val_copy["center_idx"].astype(int)
    val_copy["group_fold"] = -1
    val_copy["sample_weight"] = 1.0

    full_train_df = pd.concat([train_copy, val_copy], ignore_index=True)
    n_full_splits = min(5, full_train_df["group_id"].nunique())
    sgkf_full = StratifiedGroupKFold(n_splits=n_full_splits)
    full_train_df["group_fold"] = -1
    strat_key_full = (
        full_train_df["center_idx"].astype(str)
        + "_"
        + full_train_df["label"].astype(str)
    )
    for fold_idx, (_, va_idx) in enumerate(
        sgkf_full.split(full_train_df, strat_key_full, groups=full_train_df["group_id"])
    ):
        full_train_df.loc[va_idx, "group_fold"] = fold_idx

    full_train_df["sample_weight"] = compute_balanced_sample_weights(full_train_df)

    return train_df, val_df, full_train_df, test_df, sample_sub_df


# =============================================================================
# 3. Training-Only Stain, LAB & Optical Density Statistics Fitting
# =============================================================================
# Standard Ruifrok-Johnston H&E optical density stain matrix (rows: H, E, Residual)
_RAW_HE_MATRIX = np.array(
    [
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.072, 0.990, 0.105],  # Eosin
        [0.268, 0.570, 0.776],  # Residual orthogonal component
    ],
    dtype=np.float32,
)
HE_STAIN_MATRIX = _RAW_HE_MATRIX / np.linalg.norm(_RAW_HE_MATRIX, axis=1, keepdims=True)
HE_STAIN_INV = np.linalg.inv(HE_STAIN_MATRIX).astype(np.float32)


def _extract_single_patch_color_stats(path: str):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    rgb_f = img_rgb.astype(np.float32) / 255.0
    rgb_mean = rgb_f.mean(axis=(0, 1))
    rgb_std = rgb_f.std(axis=(0, 1))

    lab_mean = img_lab.mean(axis=(0, 1))
    lab_std = img_lab.std(axis=(0, 1))

    tissue_mask = img_lab[:, :, 0] < 225.0
    tissue_frac = float(tissue_mask.mean())

    od = -np.log((img_rgb.astype(np.float32) + 1.0) / 256.0)
    he_conc = od @ HE_STAIN_INV
    if tissue_mask.sum() >= 64:
        he_mean = he_conc[tissue_mask].mean(axis=0)
        he_std = he_conc[tissue_mask].std(axis=0)
    else:
        he_mean = he_conc.mean(axis=(0, 1))
        he_std = he_conc.std(axis=(0, 1))

    return rgb_mean, rgb_std, lab_mean, lab_std, he_mean, he_std, tissue_frac


def fit_stain_statistics(
    df: pd.DataFrame, max_samples: int = 2400, seed: int = SEED
) -> dict:
    """
    Strictly fit global and per-center color/stain statistics on the given training dataframe.
    Used for Reinhard LAB normalization and cross-center LAB interpolation/extrapolation.
    """
    rng = np.random.RandomState(seed)
    strata = df["center_id"].astype(str) + "__" + df["label"].astype(str)
    unique_strata = sorted(strata.unique().tolist())
    per_stratum = max(1, max_samples // len(unique_strata))

    sampled_indices = []
    for s in unique_strata:
        idx = df.index[strata == s].values
        take = min(len(idx), per_stratum)
        sampled_indices.extend(rng.choice(idx, size=take, replace=False).tolist())

    sub_df = df.loc[sampled_indices].reset_index(drop=True)
    paths = sub_df["full_path"].tolist()

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_extract_single_patch_color_stats, paths))

    rgb_means = np.stack([r[0] for r in results], axis=0)
    rgb_stds = np.stack([r[1] for r in results], axis=0)
    lab_means = np.stack([r[2] for r in results], axis=0)
    lab_stds = np.stack([r[3] for r in results], axis=0)
    he_means = np.stack([r[4] for r in results], axis=0)
    he_stds = np.stack([r[5] for r in results], axis=0)
    tissue_fracs = np.array([r[6] for r in results], dtype=np.float32)

    valid_tissue = tissue_fracs >= 0.15
    if valid_tissue.sum() < 50:
        valid_tissue = np.ones(len(sub_df), dtype=bool)

    stats = {
        "n_sampled": int(len(sub_df)),
        "global_rgb_mean": rgb_means.mean(axis=0).tolist(),
        "global_rgb_std": np.sqrt(
            (rgb_stds**2).mean(axis=0) + rgb_means.var(axis=0)
        ).tolist(),
        "global_lab_mean": np.median(lab_means[valid_tissue], axis=0).tolist(),
        "global_lab_std": np.median(lab_stds[valid_tissue], axis=0).tolist(),
        "global_lab_mean_spread": lab_means[valid_tissue].std(axis=0).tolist(),
        "global_lab_std_spread": lab_stds[valid_tissue].std(axis=0).tolist(),
        "global_he_mean": np.median(he_means[valid_tissue], axis=0).tolist(),
        "global_he_std": np.median(he_stds[valid_tissue], axis=0).tolist(),
        "mean_tissue_fraction": float(tissue_fracs.mean()),
        "centers": {},
    }

    for cid in sorted(sub_df["center_id"].unique().tolist()):
        c_mask = (sub_df["center_id"].values == cid) & valid_tissue
        if c_mask.sum() < 10:
            c_mask = sub_df["center_id"].values == cid
        stats["centers"][str(cid)] = {
            "n_samples": int(c_mask.sum()),
            "lab_mean_center": lab_means[c_mask].mean(axis=0).tolist(),
            "lab_mean_spread": lab_means[c_mask].std(axis=0).tolist(),
            "lab_std_center": lab_stds[c_mask].mean(axis=0).tolist(),
            "lab_std_spread": lab_stds[c_mask].std(axis=0).tolist(),
            "he_mean_center": he_means[c_mask].mean(axis=0).tolist(),
            "he_std_center": he_stds[c_mask].mean(axis=0).tolist(),
        }

    return stats


# =============================================================================
# 4. Histopathology Domain-Specific Feature Engineering & Stain Augmentations
# =============================================================================
class ReinhardLABNormalizer:
    """
    Deterministic single-patch Reinhard LAB stain normalizer using a reference
    LAB template strictly fitted on training data. Safe for both validation and test.
    """

    def __init__(
        self, target_mean, target_std, strength: float = 0.75, std_clip=(0.55, 1.8)
    ):
        self.target_mean = np.asarray(target_mean, dtype=np.float32).reshape(1, 1, 3)
        self.target_std = np.asarray(target_std, dtype=np.float32).reshape(1, 1, 3)
        self.strength = float(strength)
        self.std_clip = std_clip

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean = lab.mean(axis=(0, 1), keepdims=True)
        src_std = np.maximum(lab.std(axis=(0, 1), keepdims=True), 1.5)

        scale = np.clip(self.target_std / src_std, self.std_clip[0], self.std_clip[1])
        lab_norm = (lab - src_mean) * scale + self.target_mean
        lab_out = (1.0 - self.strength) * lab + self.strength * lab_norm
        lab_out = np.clip(lab_out, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


class ReinhardLABTransferAugment:
    """
    Training-only cross-center stain distribution interpolation & extrapolation.
    Samples synthetic target LAB color statistics both between and beyond the
    development centers' empirical distributions to simulate unseen medical centers.
    """

    def __init__(
        self,
        stain_stats: dict,
        p: float = 0.55,
        extrap_range=(-0.45, 1.45),
        jitter_scale: float = 0.5,
        strength_range=(0.35, 0.95),
    ):
        self.p = p
        self.extrap_range = extrap_range
        self.jitter_scale = jitter_scale
        self.strength_range = strength_range

        self.center_profiles = []
        for _, cstat in stain_stats.get("centers", {}).items():
            self.center_profiles.append(
                {
                    "mean": np.asarray(cstat["lab_mean_center"], dtype=np.float32),
                    "mean_spread": np.asarray(
                        cstat["lab_mean_spread"], dtype=np.float32
                    ),
                    "std": np.asarray(cstat["lab_std_center"], dtype=np.float32),
                    "std_spread": np.asarray(cstat["lab_std_spread"], dtype=np.float32),
                }
            )
        self.global_mean = np.asarray(stain_stats["global_lab_mean"], dtype=np.float32)
        self.global_std = np.asarray(stain_stats["global_lab_std"], dtype=np.float32)
        self.global_mean_spread = np.asarray(
            stain_stats["global_lab_mean_spread"], dtype=np.float32
        )
        self.global_std_spread = np.asarray(
            stain_stats["global_lab_std_spread"], dtype=np.float32
        )

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return img_rgb

        if len(self.center_profiles) >= 2:
            c1, c2 = random.sample(self.center_profiles, 2)
            lam = random.uniform(self.extrap_range[0], self.extrap_range[1])
            base_mean = (1.0 - lam) * c1["mean"] + lam * c2["mean"]
            base_std = (1.0 - lam) * c1["std"] + lam * c2["std"]
            spread_m = 0.5 * (c1["mean_spread"] + c2["mean_spread"])
            spread_s = 0.5 * (c1["std_spread"] + c2["std_spread"])
        else:
            base_mean = self.global_mean
            base_std = self.global_std
            spread_m = self.global_mean_spread
            spread_s = self.global_std_spread

        target_mean = (
            base_mean
            + np.random.normal(0.0, self.jitter_scale, size=3).astype(np.float32)
            * spread_m
        )
        target_std = (
            base_std
            + np.random.normal(0.0, self.jitter_scale, size=3).astype(np.float32)
            * spread_s
        )

        target_mean = np.clip(
            target_mean, [80.0, 105.0, 95.0], [235.0, 185.0, 155.0]
        ).reshape(1, 1, 3)
        target_std = np.clip(target_std, [8.0, 3.0, 3.0], [70.0, 35.0, 30.0]).reshape(
            1, 1, 3
        )

        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        src_mean = lab.mean(axis=(0, 1), keepdims=True)
        src_std = np.maximum(lab.std(axis=(0, 1), keepdims=True), 1.5)

        scale = np.clip(target_std / src_std, 0.55, 1.75)
        lab_shifted = (lab - src_mean) * scale + target_mean
        gamma = random.uniform(self.strength_range[0], self.strength_range[1])
        lab_out = (1.0 - gamma) * lab + gamma * lab_shifted
        lab_out = np.clip(lab_out, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


class HEDStainPerturbation:
    """
    Physically grounded Beer-Lambert H&E optical density stain augmentation.
    Perturbs both dye concentrations (alpha, beta) and stain matrix spectral vectors
    to model cross-hospital reagent and scanner variations.
    """

    def __init__(
        self,
        p: float = 0.80,
        alpha_range: float = 0.25,
        beta_range: float = 0.08,
        matrix_sigma: float = 0.025,
    ):
        self.p = p
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.matrix_sigma = matrix_sigma

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return img_rgb

        od = -np.log((img_rgb.astype(np.float32) + 1.0) / 256.0)
        conc = od @ HE_STAIN_INV

        alphas = np.array(
            [
                random.uniform(1.0 - self.alpha_range, 1.0 + self.alpha_range),
                random.uniform(1.0 - self.alpha_range, 1.0 + self.alpha_range),
                random.uniform(
                    1.0 - 0.5 * self.alpha_range, 1.0 + 0.5 * self.alpha_range
                ),
            ],
            dtype=np.float32,
        )
        betas = np.array(
            [
                random.uniform(-self.beta_range, self.beta_range),
                random.uniform(-self.beta_range, self.beta_range),
                random.uniform(-0.35 * self.beta_range, 0.35 * self.beta_range),
            ],
            dtype=np.float32,
        )
        conc_new = np.maximum(
            0.0, conc * alphas.reshape(1, 1, 3) + betas.reshape(1, 1, 3)
        )

        if self.matrix_sigma > 0:
            m_pert = HE_STAIN_MATRIX + np.random.normal(
                0.0, self.matrix_sigma, size=HE_STAIN_MATRIX.shape
            ).astype(np.float32)
            m_pert = np.maximum(m_pert, 1e-3)
            m_pert = m_pert / np.linalg.norm(m_pert, axis=1, keepdims=True)
        else:
            m_pert = HE_STAIN_MATRIX

        od_new = np.clip(conc_new @ m_pert, 0.0, 5.6)
        rgb_new = np.clip(256.0 * np.exp(-od_new) - 1.0, 0.0, 255.0).astype(np.uint8)
        return rgb_new


class D4DihedralTransform:
    """
    Exact lossless D4 symmetry augmentation (8 orientations: 4 rotations x 2 reflections),
    matching the isotropic orientation invariance of histopathology microscopy patches.
    """

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        k = random.randint(0, 3)
        if k > 0:
            img_rgb = np.rot90(img_rgb, k)
        if random.random() < 0.5:
            img_rgb = np.fliplr(img_rgb)
        return np.ascontiguousarray(img_rgb)


class MicroscopyOpticsAugment:
    """
    Mild optical scanner perturbations: defocus blur, sharpening, and subtle brightness/contrast
    shifts typical of whole-slide scanner focus plane variations across centers.
    """

    def __init__(self, p_blur: float = 0.20, p_bc: float = 0.35):
        self.p_blur = p_blur
        self.p_bc = p_bc

    def __call__(self, img_rgb: np.ndarray) -> np.ndarray:
        if random.random() < self.p_bc:
            alpha = random.uniform(0.88, 1.12)
            beta = random.uniform(-12.0, 12.0)
            img_rgb = np.clip(
                img_rgb.astype(np.float32) * alpha + beta, 0.0, 255.0
            ).astype(np.uint8)

        if random.random() < self.p_blur:
            if random.random() < 0.65:
                sigma = random.uniform(0.3, 1.1)
                img_rgb = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=sigma)
            else:
                blurred = cv2.GaussianBlur(img_rgb, (3, 3), sigmaX=0.8)
                amount = random.uniform(0.2, 0.6)
                img_rgb = np.clip(
                    img_rgb.astype(np.float32) * (1.0 + amount)
                    - blurred.astype(np.float32) * amount,
                    0.0,
                    255.0,
                ).astype(np.uint8)

        return img_rgb


class HistopathologyTransformPipeline:
    """
    Composable NumPy -> PyTorch Tensor preprocessing pipeline.
    Supports pixel-aligned paired (anchor, domain) views during training while
    guaranteeing the exact same deterministic single-view path for validation and test sets.
    """

    def __init__(
        self,
        is_train: bool,
        stain_stats: dict = None,
        img_size: int = IMG_SIZE,
        use_reinhard_norm: bool = False,
        strong_stain_aug: bool = True,
        norm_mean=IMAGENET_MEAN,
        norm_std=IMAGENET_STD,
        paired_views: bool = False,
    ):
        self.is_train = is_train
        self.img_size = img_size
        self.paired_views = bool(is_train and paired_views)
        self.shared_np_ops = []

        if use_reinhard_norm:
            assert (
                stain_stats is not None
            ), "stain_stats required for Reinhard normalization"
            self.shared_np_ops.append(
                ReinhardLABNormalizer(
                    target_mean=stain_stats["global_lab_mean"],
                    target_std=stain_stats["global_lab_std"],
                    strength=0.75,
                )
            )

        if is_train:
            self.shared_np_ops.append(D4DihedralTransform())

        if self.paired_views:
            self.anchor_np_ops = [
                HEDStainPerturbation(
                    p=0.50, alpha_range=0.12, beta_range=0.04, matrix_sigma=0.01
                )
            ]
            self.anchor_tensor_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )

            self.domain_np_ops = []
            if strong_stain_aug:
                if stain_stats is not None:
                    self.domain_np_ops.append(
                        ReinhardLABTransferAugment(stain_stats, p=0.55)
                    )
                self.domain_np_ops.append(
                    HEDStainPerturbation(p=0.80, alpha_range=0.25, beta_range=0.08)
                )
                self.domain_np_ops.append(
                    MicroscopyOpticsAugment(p_blur=0.20, p_bc=0.35)
                )
            else:
                self.domain_np_ops.append(
                    HEDStainPerturbation(p=0.50, alpha_range=0.15, beta_range=0.05)
                )

            self.domain_tensor_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomErasing(
                        p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0), value="random"
                    ),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ]
            )
            self.np_ops = self.domain_np_ops
            self.tensor_transform = self.domain_tensor_transform
        else:
            self.np_ops = []
            if is_train:
                if strong_stain_aug:
                    if stain_stats is not None:
                        self.np_ops.append(
                            ReinhardLABTransferAugment(stain_stats, p=0.55)
                        )
                    self.np_ops.append(
                        HEDStainPerturbation(p=0.80, alpha_range=0.25, beta_range=0.08)
                    )
                    self.np_ops.append(MicroscopyOpticsAugment(p_blur=0.20, p_bc=0.35))
                else:
                    self.np_ops.append(
                        HEDStainPerturbation(p=0.50, alpha_range=0.15, beta_range=0.05)
                    )

            tensor_ops = [transforms.ToTensor()]
            if is_train:
                tensor_ops.append(
                    transforms.RandomErasing(
                        p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0), value="random"
                    )
                )
            tensor_ops.append(transforms.Normalize(mean=norm_mean, std=norm_std))
            self.tensor_transform = transforms.Compose(tensor_ops)

    def __call__(self, img_rgb: np.ndarray):
        if img_rgb.shape[0] != self.img_size or img_rgb.shape[1] != self.img_size:
            interp = (
                cv2.INTER_AREA
                if (img_rgb.shape[0] > self.img_size or img_rgb.shape[1] > self.img_size)
                else cv2.INTER_LINEAR
            )
            img_rgb = cv2.resize(
                img_rgb, (self.img_size, self.img_size), interpolation=interp
            )
        for op in self.shared_np_ops:
            img_rgb = op(img_rgb)

        if self.paired_views:
            anchor_img = img_rgb.copy()
            for op in self.anchor_np_ops:
                anchor_img = op(anchor_img)
            view_anchor = self.anchor_tensor_transform(anchor_img)

            domain_img = img_rgb.copy()
            for op in self.domain_np_ops:
                domain_img = op(domain_img)
            view_domain = self.domain_tensor_transform(domain_img)
            return view_anchor, view_domain

        for op in self.np_ops:
            img_rgb = op(img_rgb)
        return self.tensor_transform(img_rgb)


def build_train_transform(
    stain_stats: dict,
    img_size: int = IMG_SIZE,
    use_reinhard_norm: bool = False,
    strong_stain_aug: bool = True,
    paired_views: bool = True,
) -> HistopathologyTransformPipeline:
    return HistopathologyTransformPipeline(
        is_train=True,
        stain_stats=stain_stats,
        img_size=img_size,
        use_reinhard_norm=use_reinhard_norm,
        strong_stain_aug=strong_stain_aug,
        paired_views=paired_views,
    )


def build_eval_transform(
    stain_stats: dict = None,
    img_size: int = IMG_SIZE,
    use_reinhard_norm: bool = False,
) -> HistopathologyTransformPipeline:
    return HistopathologyTransformPipeline(
        is_train=False,
        stain_stats=stain_stats,
        img_size=img_size,
        use_reinhard_norm=use_reinhard_norm,
        strong_stain_aug=False,
        paired_views=False,
    )


def get_d4_tta_views(x: torch.Tensor, n_views: int = 8):
    """
    Generate deterministic D4 dihedral views from a batched image tensor (B, C, H, W).
    Supports n_views in {1, 4, 8}. Applied identically to val & test.
    """
    if n_views == 1:
        return [x]
    views = [
        x,
        torch.rot90(x, 1, dims=(2, 3)),
        torch.rot90(x, 2, dims=(2, 3)),
        torch.rot90(x, 3, dims=(2, 3)),
    ]
    if n_views == 4:
        return views
    x_flip = torch.flip(x, dims=(3,))
    views.extend(
        [
            x_flip,
            torch.rot90(x_flip, 1, dims=(2, 3)),
            torch.rot90(x_flip, 2, dims=(2, 3)),
            torch.rot90(x_flip, 3, dims=(2, 3)),
        ]
    )
    return views


# =============================================================================
# 5. PyTorch Dataset & DataLoader Factory
# =============================================================================
class HistopathologyDataset(Dataset):
    """
    Fast OpenCV-backed PyTorch Dataset for 256x256 H&E patches.
    Uses ONLY image pixels as model inputs; optional metadata is returned solely
    for training-time domain methods (never required or used at inference).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        is_test: bool = False,
        return_meta: bool = False,
        paired_views: bool = False,
    ):
        self.paths = df["full_path"].values
        self.sample_ids = df["sample_id"].values
        self.transform = transform
        self.is_test = is_test
        self.return_meta = return_meta
        self.paired_views = bool(paired_views and (not is_test))

        if not is_test:
            self.labels = df["label"].values.astype(np.float32)
            self.center_idx = (
                df["center_idx"].values.astype(np.int64)
                if "center_idx" in df.columns
                else np.full(len(df), -1, dtype=np.int64)
            )
        else:
            self.labels = None
            self.center_idx = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            img_rgb = np.array(Image.open(path).convert("RGB"))
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img_out = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_out = (
                (img_tensor, img_tensor.clone()) if self.paired_views else img_tensor
            )

        if self.is_test:
            return img_out, self.sample_ids[idx]

        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        if isinstance(img_out, tuple):
            img_anchor, img_domain = img_out
            if self.return_meta:
                return (
                    img_anchor,
                    img_domain,
                    label_tensor,
                    torch.tensor(self.center_idx[idx], dtype=torch.long),
                )
            return img_anchor, img_domain, label_tensor

        if self.return_meta:
            return (
                img_out,
                label_tensor,
                torch.tensor(self.center_idx[idx], dtype=torch.long),
            )
        return img_out, label_tensor


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = (SEED + worker_id * 1009) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stain_stats: dict,
    batch_size: int = 64,
    eval_batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
    use_weighted_sampler: bool = True,
    use_reinhard_norm: bool = False,
    strong_stain_aug: bool = True,
    return_train_meta: bool = False,
    paired_views: bool = True,
):
    train_tf = build_train_transform(
        stain_stats=stain_stats,
        img_size=img_size,
        use_reinhard_norm=use_reinhard_norm,
        strong_stain_aug=strong_stain_aug,
        paired_views=paired_views,
    )
    # CRITICAL: Validation and Test share the EXACT same deterministic transform object
    eval_tf = build_eval_transform(
        stain_stats=stain_stats,
        img_size=img_size,
        use_reinhard_norm=use_reinhard_norm,
    )

    train_ds = HistopathologyDataset(
        train_df,
        transform=train_tf,
        is_test=False,
        return_meta=return_train_meta,
        paired_views=paired_views,
    )
    val_ds = HistopathologyDataset(
        val_df, transform=eval_tf, is_test=False, return_meta=False
    )
    test_ds = HistopathologyDataset(
        test_df, transform=eval_tf, is_test=True, return_meta=False
    )

    pin_mem = torch.cuda.is_available()
    persist = num_workers > 0

    if use_weighted_sampler and "sample_weight" in train_df.columns:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                train_df["sample_weight"].values, dtype=torch.double
            ),
            num_samples=len(train_df),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persist,
            worker_init_fn=_worker_init_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True,
            persistent_workers=persist,
            worker_init_fn=_worker_init_fn,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
        persistent_workers=persist,
    )
    return train_loader, val_loader, test_loader


# =============================================================================
# 6. Domain Generalization & Histopathology Architectural Modules
# =============================================================================
class MixStyle(nn.Module):
    """
    Training-only Feature-Space Style Mixing (Zhou et al., ICLR 2021).
    For ViT token sequences (B, N, C), perturbs channel-wise mean and standard
    deviation strictly across spatial tokens x[:, num_prefix_tokens:, :] while
    leaving CLS and register tokens untouched. Uses PyTorch RNG so gradient
    checkpoint recomputation stays synchronized. Strictly a no-op during eval.
    """

    def __init__(
        self,
        p: float = 0.5,
        alpha: float = 0.1,
        eps: float = 1e-6,
        num_prefix_tokens: int = 0,
    ):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            (not self.training)
            or (x.size(0) < 2)
            or (float(torch.rand(1).item()) > self.p)
        ):
            return x

        orig_dtype = x.dtype
        B = x.size(0)

        if x.ndim == 3:
            n_prefix = self.num_prefix_tokens
            if n_prefix > 0:
                prefix = x[:, :n_prefix, :]
                spatial = x[:, n_prefix:, :]
            else:
                prefix = None
                spatial = x

            if spatial.size(1) < 2:
                return x

            x_f = spatial.float()
            mu = x_f.mean(dim=1, keepdim=True)
            var = x_f.var(dim=1, keepdim=True, unbiased=False)
            sig = (var + self.eps).sqrt()
            x_normed = (x_f - mu) / sig

            lmda = self.beta.sample((B, 1, 1)).to(device=x_f.device, dtype=x_f.dtype)
            perm = torch.randperm(B, device=x_f.device)

            mu_mix = lmda * mu + (1.0 - lmda) * mu[perm]
            sig_mix = lmda * sig + (1.0 - lmda) * sig[perm]

            spatial_out = (x_normed * sig_mix + mu_mix).to(orig_dtype)
            if prefix is not None:
                return torch.cat([prefix, spatial_out], dim=1)
            return spatial_out

        x_f = x.float()
        mu = x_f.mean(dim=(2, 3), keepdim=True)
        var = x_f.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_normed = (x_f - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1)).to(device=x_f.device, dtype=x_f.dtype)
        perm = torch.randperm(B, device=x_f.device)

        mu_mix = lmda * mu + (1.0 - lmda) * mu[perm]
        sig_mix = lmda * sig + (1.0 - lmda) * sig[perm]

        out = x_normed * sig_mix + mu_mix
        return out.to(orig_dtype)


class GatedSpatialAttentionPool(nn.Module):
    """
    Histopathology-inspired Gated Attention Pooling (Ilse et al., ICML 2018) over
    ViT spatial patch tokens (B, N_spatial, C) to highlight localized tumor clusters.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.attn_v = nn.Linear(in_channels, hidden_dim)
        self.attn_u = nn.Linear(in_channels, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.attn_w = nn.Linear(hidden_dim, 1)

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        if spatial_tokens.ndim == 4:
            tokens = spatial_tokens.flatten(2).transpose(1, 2)
        else:
            tokens = spatial_tokens
        tokens_norm = self.norm(tokens)

        gate = torch.tanh(self.attn_v(tokens_norm)) * torch.sigmoid(
            self.attn_u(tokens_norm)
        )
        attn_logits = self.attn_w(self.drop(gate)).squeeze(-1)

        attn_weights = torch.softmax(attn_logits.float(), dim=-1).to(tokens.dtype)
        pooled = torch.bmm(attn_weights.unsqueeze(1), tokens).squeeze(1)
        return pooled


class SpatialContextRefiner2D(nn.Module):
    """
    Lightweight strictly LayerNorm-only 2D spatial-context refiner applied
    exclusively to ViT spatial patch tokens reshaped to (B, C, H, W).
    Captures local nuclear/chromatin texture and coherent 2D cellular clusters
    via parallel 3x3 depthwise and dilated 3x3 depthwise convolutions with
    residual LayerScale gating.
    """

    def __init__(self, in_channels: int, layer_scale_init: float = 0.1):
        super().__init__()
        mid_channels = in_channels // 2
        self.norm1 = nn.LayerNorm(in_channels)
        self.pw_in = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.dw_local = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            groups=mid_channels,
            bias=True,
        )
        self.dw_dilated = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=mid_channels,
            bias=True,
        )
        self.norm2 = nn.LayerNorm(mid_channels * 2)
        self.pw_out = nn.Conv2d(
            mid_channels * 2, in_channels, kernel_size=1, bias=True
        )
        self.gamma = nn.Parameter(
            torch.full((in_channels,), float(layer_scale_init))
        )

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = spatial_tokens.shape
        H = int(round(math.sqrt(N)))
        W = H
        if H * W != N:
            return spatial_tokens

        x = self.norm1(spatial_tokens)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.gelu(self.pw_in(x))
        x_local = self.dw_local(x)
        x_dilated = self.dw_dilated(x)
        x = torch.cat([x_local, x_dilated], dim=1)

        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.norm2(x))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.pw_out(x)
        x = x.flatten(2).transpose(1, 2)

        gamma = self.gamma.to(dtype=x.dtype).view(1, 1, C)
        return spatial_tokens + (gamma * x).to(dtype=spatial_tokens.dtype)


class MultiScalePoolBlock(nn.Module):
    """
    Refines ViT spatial patch tokens with a LayerNorm-only 2D spatial-context
    module (excluding CLS and register tokens), then fuses
    [cls_token, spatial_gap, gated_spatial_attn] (3 * C dims) followed by
    LayerNorm projection to bound OOD feature norms.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        num_prefix_tokens: int = 1,
        attn_hidden: int = 128,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.spatial_refiner = SpatialContextRefiner2D(
            in_channels=in_channels, layer_scale_init=0.1
        )
        self.attn_pool = GatedSpatialAttentionPool(
            in_channels=in_channels, hidden_dim=attn_hidden, dropout=dropout
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels * 3),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 3, out_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.num_prefix_tokens > 0:
            cls_token = tokens[:, 0, :]
            spatial_tokens = tokens[:, self.num_prefix_tokens :, :]
        else:
            spatial_tokens = tokens
            cls_token = spatial_tokens.mean(dim=1)

        spatial_tokens = self.spatial_refiner(spatial_tokens)
        spatial_gap = spatial_tokens.mean(dim=1)
        gated_spatial_attn = self.attn_pool(spatial_tokens)
        fused = torch.cat([cls_token, spatial_gap, gated_spatial_attn], dim=1)
        return self.proj(fused)


class MultiSampleDropoutHead(nn.Module):
    """
    Multi-Sample Dropout binary classification head producing a single calibrated
    logit per sample.
    """

    def __init__(
        self,
        in_dim: int,
        n_drops: int = 5,
        drop_rate: float = 0.35,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(n_drops)])
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        normed = self.norm(x)
        logits = torch.stack([self.fc(d(normed)) for d in self.dropouts], dim=0).mean(
            dim=0
        )
        return logits.squeeze(-1), normed


class HistopathologyDomainInvariantNet(nn.Module):
    """
    Self-supervised LayerNorm-only DINOv2 Vision Transformer with register tokens,
    early spatial-token-only MixStyle (blocks 1 and 2), 3-depth 2D spatial-context
    feature pyramid decoder (blocks 5, 8, and 11), and gradient checkpointing for
    unseen-center histopathology.
    """

    def __init__(
        self,
        backbone_name: str = PRETRAINED_MODEL_CARD["backbone_primary"],
        pretrained: bool = True,
        drop_path_rate: float = 0.20,
        use_mixstyle: bool = True,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
        stage1_proj_dim: int = 192,
        stage2_proj_dim: int = 256,
        stage3_proj_dim: int = 512,
        head_dropout: float = 0.35,
        n_multisample_drops: int = 5,
        img_size: int = IMG_SIZE,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.img_size = int(img_size)
        self.use_grad_checkpointing = bool(use_grad_checkpointing)

        self.backbone, self.resolved_backbone_name = self._create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=self.img_size,
            drop_path_rate=drop_path_rate,
        )

        self.num_blocks = len(self.backbone.blocks)
        self.early_block_idx = min(5, self.num_blocks - 3)
        self.mid_block_idx = min(8, self.num_blocks - 2)
        self.final_block_idx = self.num_blocks - 1
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        vit_dim = int(
            getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features"))
        )
        self.expected_channels = (vit_dim, vit_dim, vit_dim)

        self.use_mixstyle = use_mixstyle
        self._mixstyle_active = True
        self.mixstyle_layers = nn.ModuleList()
        self._hook_handles = []
        if self.use_mixstyle:
            self._attach_early_mixstyle_hooks(p=mixstyle_p, alpha=mixstyle_alpha)

        self.early_norm = nn.LayerNorm(vit_dim)
        self.mid_norm = nn.LayerNorm(vit_dim)
        self.stage1_pool = MultiScalePoolBlock(
            in_channels=vit_dim,
            out_dim=stage1_proj_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            attn_hidden=128,
            dropout=0.25,
        )
        self.stage2_pool = MultiScalePoolBlock(
            in_channels=vit_dim,
            out_dim=stage2_proj_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            attn_hidden=128,
            dropout=0.25,
        )
        self.stage3_pool = MultiScalePoolBlock(
            in_channels=vit_dim,
            out_dim=stage3_proj_dim,
            num_prefix_tokens=self.num_prefix_tokens,
            attn_hidden=192,
            dropout=0.25,
        )

        self.embed_dim = stage1_proj_dim + stage2_proj_dim + stage3_proj_dim
        self.head = MultiSampleDropoutHead(
            in_dim=self.embed_dim,
            n_drops=n_multisample_drops,
            drop_rate=head_dropout,
        )

    @staticmethod
    def _create_backbone(
        backbone_name: str,
        pretrained: bool,
        img_size: int,
        drop_path_rate: float,
    ):
        candidates = [backbone_name] + [
            fb
            for fb in PRETRAINED_MODEL_CARD["backbone_fallbacks"]
            if fb != backbone_name
        ]
        last_err = None
        for name in candidates:
            try:
                try:
                    model = timm.create_model(
                        name,
                        pretrained=pretrained,
                        num_classes=0,
                        img_size=img_size,
                        drop_path_rate=drop_path_rate,
                        dynamic_img_size=True,
                    )
                except TypeError:
                    model = timm.create_model(
                        name,
                        pretrained=pretrained,
                        num_classes=0,
                        img_size=img_size,
                        drop_path_rate=drop_path_rate,
                    )
                return model, name
            except Exception as err:
                last_err = err
        raise RuntimeError(
            f"Failed to create any backbone from {candidates}. Last error: {last_err}"
        )

    def _attach_early_mixstyle_hooks(self, p: float, alpha: float) -> None:
        target_block_indices = (1, 2)
        for blk_idx in target_block_indices:
            if blk_idx >= len(self.backbone.blocks):
                continue
            mod = self.backbone.blocks[blk_idx]
            ms = MixStyle(
                p=p,
                alpha=alpha,
                num_prefix_tokens=self.num_prefix_tokens,
            )
            self.mixstyle_layers.append(ms)

            def _make_hook(mix_layer):
                def _hook(_module, _inp, output):
                    if (
                        self._mixstyle_active
                        and self.training
                        and isinstance(output, torch.Tensor)
                        and output.ndim == 3
                    ):
                        return mix_layer(output)
                    return output

                return _hook

            self._hook_handles.append(mod.register_forward_hook(_make_hook(ms)))

    def _run_block_with_mixstyle_state(
        self, blk: nn.Module, x: torch.Tensor, mix_active: bool
    ) -> torch.Tensor:
        prev = self._mixstyle_active
        self._mixstyle_active = mix_active
        try:
            return blk(x)
        finally:
            self._mixstyle_active = prev

    def _forward_backbone_multidepth(
        self, x: torch.Tensor, mix_active: bool
    ):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        patch_drop = getattr(self.backbone, "patch_drop", None)
        if patch_drop is not None:
            x = patch_drop(x)
        norm_pre = getattr(self.backbone, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)

        feat_early = None
        feat_mid = None
        use_ckpt = (
            self.use_grad_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and x.requires_grad
        )
        for idx, blk in enumerate(self.backbone.blocks):
            if use_ckpt:

                def _ckpt_fn(inp, _blk=blk, _mix=mix_active):
                    return self._run_block_with_mixstyle_state(_blk, inp, _mix)

                x = checkpoint(_ckpt_fn, x, use_reentrant=False)
            else:
                x = blk(x)
            if idx == self.early_block_idx:
                feat_early = x
            if idx == self.mid_block_idx:
                feat_mid = x

        feat_final = self.backbone.norm(x)
        if feat_early is None:
            feat_early = feat_final
        if feat_mid is None:
            feat_mid = feat_final
        return feat_early, feat_mid, feat_final

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        apply_mixstyle: bool = True,
    ):
        prev_mixstyle_active = self._mixstyle_active
        mix_active = bool(apply_mixstyle and self.training)
        self._mixstyle_active = mix_active
        try:
            feat_early, feat_mid, feat_final = self._forward_backbone_multidepth(
                x, mix_active=mix_active
            )
        finally:
            self._mixstyle_active = prev_mixstyle_active

        e1 = self.stage1_pool(self.early_norm(feat_early))
        e2 = self.stage2_pool(self.mid_norm(feat_mid))
        e3 = self.stage3_pool(feat_final)
        emb = torch.cat([e1, e2, e3], dim=1)

        logits, normed_emb = self.head(emb)
        if return_features:
            return logits, normed_emb
        return logits


# =============================================================================
# 7. Loss, Model EMA, Differential-LR Optimizer & Warmup-Cosine Scheduler
# =============================================================================
class DomainInvariantHistopathologyLoss(nn.Module):
    """
    Symmetric Label-Smoothed BCEWithLogitsLoss + Training-Only Class-Conditional Deep CORAL
    + Paired Anchor/Domain Logit and Feature Consistency Losses.
    """

    def __init__(
        self,
        label_smoothing: float = 0.04,
        coral_weight: float = 0.15,
        logit_cons_weight: float = 0.25,
        feat_cons_weight: float = 0.50,
        mean_weight: float = 1.0,
        cov_weight: float = 0.25,
        min_samples_per_group: int = 4,
        pos_weight: float = None,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.coral_weight = float(coral_weight)
        self.logit_cons_weight = float(logit_cons_weight)
        self.feat_cons_weight = float(feat_cons_weight)
        self.mean_weight = float(mean_weight)
        self.cov_weight = float(cov_weight)
        self.min_samples_per_group = int(min_samples_per_group)
        if pos_weight is not None:
            self.register_buffer(
                "pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32)
            )
        else:
            self.pos_weight = None

    def _compute_class_conditional_coral(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        center_idx: torch.Tensor,
    ) -> torch.Tensor:
        device = features.device
        valid_mask = center_idx >= 0
        if valid_mask.sum() < 2 * self.min_samples_per_group:
            return torch.zeros((), device=device, dtype=torch.float32)

        feats = F.normalize(features[valid_mask].float(), p=2, dim=1)
        lbls = (labels[valid_mask] >= 0.5).long()
        ctrs = center_idx[valid_mask].long()

        unique_centers = torch.unique(ctrs)
        if unique_centers.numel() < 2:
            return torch.zeros((), device=device, dtype=torch.float32)

        total_align = torch.zeros((), device=device, dtype=torch.float32)
        n_pairs = 0
        dim = feats.size(1)

        for cls_val in (0, 1):
            cls_mask = lbls == cls_val
            center_feats = []
            for c_id in unique_centers:
                grp = feats[cls_mask & (ctrs == c_id)]
                if grp.size(0) >= self.min_samples_per_group:
                    center_feats.append(grp)

            for i in range(len(center_feats)):
                for j in range(i + 1, len(center_feats)):
                    fi, fj = center_feats[i], center_feats[j]
                    mi = fi.mean(dim=0)
                    mj = fj.mean(dim=0)
                    mean_loss = F.mse_loss(mi, mj)

                    fi_c = fi - mi.unsqueeze(0)
                    fj_c = fj - mj.unsqueeze(0)
                    ci = (fi_c.T @ fi_c) / max(fi.size(0) - 1, 1)
                    cj = (fj_c.T @ fj_c) / max(fj.size(0) - 1, 1)
                    cov_loss = ((ci - cj) ** 2).sum() / (4.0 * dim)

                    total_align = total_align + (
                        self.mean_weight * mean_loss + self.cov_weight * cov_loss
                    )
                    n_pairs += 1

        if n_pairs == 0:
            return torch.zeros((), device=device, dtype=torch.float32)
        return total_align / float(n_pairs)

    def _compute_single_view_loss(
        self,
        logits: torch.Tensor,
        targets_1d: torch.Tensor,
        smooth_targets: torch.Tensor,
        features: torch.Tensor = None,
        center_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        logits_1d = logits.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(
            logits_1d,
            smooth_targets,
            pos_weight=self.pos_weight,
        )

        if (
            self.coral_weight > 0.0
            and features is not None
            and center_idx is not None
            and features.requires_grad
        ):
            coral = self._compute_class_conditional_coral(
                features=features,
                labels=targets_1d,
                center_idx=center_idx.view(-1),
            )
            return bce + self.coral_weight * coral

        return bce

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor = None,
        center_idx: torch.Tensor = None,
        logits_domain: torch.Tensor = None,
        features_domain: torch.Tensor = None,
        cons_scale: float = 1.0,
    ) -> torch.Tensor:
        targets_1d = labels.view(-1).float()

        if self.label_smoothing > 0.0:
            smooth_targets = (
                targets_1d * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )
        else:
            smooth_targets = targets_1d

        loss_anchor = self._compute_single_view_loss(
            logits=logits,
            targets_1d=targets_1d,
            smooth_targets=smooth_targets,
            features=features,
            center_idx=center_idx,
        )

        if logits_domain is None:
            return loss_anchor

        loss_domain = self._compute_single_view_loss(
            logits=logits_domain,
            targets_1d=targets_1d,
            smooth_targets=smooth_targets,
            features=features_domain,
            center_idx=center_idx,
        )
        total_loss = 0.5 * (loss_anchor + loss_domain)

        cons_scale_f = float(cons_scale)
        if cons_scale_f > 0.0:
            cons_term = torch.zeros((), device=logits.device, dtype=torch.float32)

            if self.logit_cons_weight > 0.0:
                logits_a = logits.view(-1).float()
                logits_d = logits_domain.view(-1).float()
                logit_cons = 0.75 * F.smooth_l1_loss(
                    logits_d, logits_a.detach(), beta=1.0
                ) + 0.25 * F.smooth_l1_loss(
                    logits_a, logits_d.detach(), beta=1.0
                )
                cons_term = cons_term + self.logit_cons_weight * logit_cons

            if (
                self.feat_cons_weight > 0.0
                and features is not None
                and features_domain is not None
            ):
                fa = F.normalize(features.float(), p=2, dim=1)
                fd = F.normalize(features_domain.float(), p=2, dim=1)
                feat_cons = (1.0 - F.cosine_similarity(fa, fd, dim=1)).mean()
                cons_term = cons_term + self.feat_cons_weight * feat_cons

            total_loss = total_loss + cons_scale_f * cons_term

        return total_loss


class ModelEMA:
    """
    Exponential Moving Average of model weights for flatter minima and improved
    probability calibration / balanced accuracy on unseen medical centers.
    """

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


def build_model(
    backbone_name: str = PRETRAINED_MODEL_CARD["backbone_primary"],
    pretrained: bool = True,
    drop_path_rate: float = 0.20,
    use_mixstyle: bool = True,
    mixstyle_p: float = 0.5,
    mixstyle_alpha: float = 0.1,
    device: torch.device = DEVICE,
    img_size: int = IMG_SIZE,
    use_grad_checkpointing: bool = True,
) -> HistopathologyDomainInvariantNet:
    model = HistopathologyDomainInvariantNet(
        backbone_name=backbone_name,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        use_mixstyle=use_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
        img_size=img_size,
        use_grad_checkpointing=use_grad_checkpointing,
    )
    return model.to(device)


def build_criterion(
    label_smoothing: float = 0.04,
    coral_weight: float = 0.15,
    logit_cons_weight: float = 0.25,
    feat_cons_weight: float = 0.50,
    device: torch.device = DEVICE,
) -> DomainInvariantHistopathologyLoss:
    return DomainInvariantHistopathologyLoss(
        label_smoothing=label_smoothing,
        coral_weight=coral_weight,
        logit_cons_weight=logit_cons_weight,
        feat_cons_weight=feat_cons_weight,
    ).to(device)


def build_optimizer(
    model: nn.Module,
    backbone_lr: float = 5.0e-5,
    head_lr: float = 3.0e-4,
    weight_decay: float = 5.0e-2,
    layer_decay: float = 0.86,
) -> AdamW:
    num_blocks = (
        len(model.backbone.blocks)
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks")
        else 12
    )

    def _get_backbone_layer_id(param_name: str) -> int:
        sub_name = param_name[len("backbone.") :]
        if (
            sub_name in ("cls_token", "reg_token", "pos_embed")
            or sub_name.startswith("patch_embed.")
            or sub_name.startswith("norm_pre.")
        ):
            return 0
        if sub_name.startswith("blocks."):
            parts = sub_name.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                return min(int(parts[1]) + 1, num_blocks)
        return num_blocks

    backbone_groups = {}
    head_decay, head_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        is_no_decay = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in name_lower
            or "gamma" in name_lower
            or "ls1" in name_lower
            or "ls2" in name_lower
            or "pos_embed" in name_lower
            or "cls_token" in name_lower
            or "reg_token" in name_lower
        )

        if name.startswith("backbone."):
            layer_id = (
                _get_backbone_layer_id(name) if layer_decay < 1.0 else num_blocks
            )
            key = (layer_id, is_no_decay)
            if key not in backbone_groups:
                backbone_groups[key] = []
            backbone_groups[key].append(param)
        else:
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)

    param_groups = []
    # Order from top backbone block (layer_id=num_blocks, peak backbone_lr) downward
    # so optimizer.param_groups[0]["lr"] accurately logs the peak backbone LR.
    for layer_id in range(num_blocks, -1, -1):
        scale = float(layer_decay) ** (num_blocks - layer_id)
        layer_lr = float(backbone_lr) * scale
        for is_no_decay, wd in ((False, weight_decay), (True, 0.0)):
            params = backbone_groups.get((layer_id, is_no_decay), [])
            if len(params) > 0:
                param_groups.append(
                    {
                        "params": params,
                        "lr": layer_lr,
                        "weight_decay": wd,
                    }
                )

    if len(head_decay) > 0:
        param_groups.append(
            {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay}
        )
    if len(head_no_decay) > 0:
        param_groups.append(
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0}
        )

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(total_steps * warmup_ratio), 1)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(
            max(total_steps - warmup_steps, 1)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# =============================================================================
# 8. Core Training & Unified Validation/Test Inference Utilities
# =============================================================================
def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits_f64 = np.asarray(logits, dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits_f64, -40.0, 40.0)))
    return np.clip(probs, 0.0, 1.0)


def train_one_epoch(
    train_model,
    ema_wrapper,
    loader,
    loss_fn,
    opt,
    sched,
    scaler,
    device,
    use_amp: bool,
    amp_dtype,
    use_grad_scaler: bool,
    base_ema_decay: float,
    global_step: int,
    max_grad_norm: float = MAX_GRAD_NORM,
    total_steps: int = None,
    cons_warmup_ratio: float = 0.15,
):
    train_model.train()
    running_loss = 0.0
    n_samples = 0

    for batch in loader:
        if len(batch) == 4:
            img_anchor, img_domain, labels, center_idx = batch
            center_idx = center_idx.to(device, non_blocking=True)
        elif len(batch) == 3:
            if batch[1].ndim == 4:
                img_anchor, img_domain, labels = batch
                center_idx = None
            else:
                img_anchor, labels, center_idx = batch
                img_domain = None
                center_idx = center_idx.to(device, non_blocking=True)
        else:
            img_anchor, labels = batch
            img_domain = None
            center_idx = None

        img_anchor = img_anchor.to(device, non_blocking=True)
        if img_domain is not None:
            img_domain = img_domain.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            if img_domain is not None:
                if total_steps is not None:
                    cons_scale = min(
                        1.0,
                        float(global_step + 1)
                        / float(max(1, int(cons_warmup_ratio * total_steps))),
                    )
                else:
                    cons_scale = 1.0
                logits_anchor, feats_anchor = train_model(
                    img_anchor, return_features=True, apply_mixstyle=False
                )
                logits_domain, feats_domain = train_model(
                    img_domain, return_features=True, apply_mixstyle=True
                )
                loss = loss_fn(
                    logits_anchor,
                    labels,
                    features=feats_anchor,
                    center_idx=center_idx,
                    logits_domain=logits_domain,
                    features_domain=feats_domain,
                    cons_scale=cons_scale,
                )
            else:
                logits, feats = train_model(img_anchor, return_features=True)
                loss = loss_fn(logits, labels, features=feats, center_idx=center_idx)

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        if use_grad_scaler:
            scale_before = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            step_executed = scaler.get_scale() >= scale_before
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)
            opt.step()
            step_executed = True

        if step_executed:
            global_step += 1
            if sched is not None:
                sched.step()
            if ema_wrapper is not None:
                ema_wrapper.decay = min(
                    base_ema_decay, (1.0 + global_step) / (10.0 + global_step)
                )
                ema_wrapper.update(train_model)

        bs = img_anchor.size(0)
        running_loss += float(loss.detach().item()) * bs
        n_samples += bs

    avg_loss = running_loss / max(n_samples, 1)
    return avg_loss, global_step


@torch.inference_mode()
def predict_logits(
    eval_model,
    loader,
    device,
    use_amp: bool,
    amp_dtype,
    tta_views: int = FINAL_TTA_VIEWS,
    is_test: bool = False,
):
    """
    Unified, deterministic inference pipeline shared identically by validation and test.
    Uses ONLY image pixels and exact D4 dihedral logit averaging (never metadata).
    """
    eval_model.eval()
    all_logits = []
    all_second = []

    for batch in loader:
        images, second = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        views = get_d4_tta_views(images, n_views=tta_views)

        batch_logit_sum = None
        for view in views:
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = eval_model(view, return_features=False)
            logits_f32 = logits.float()
            batch_logit_sum = (
                logits_f32
                if batch_logit_sum is None
                else (batch_logit_sum + logits_f32)
            )

        batch_logits = (batch_logit_sum / float(len(views))).cpu().numpy()
        all_logits.append(batch_logits)

        if is_test:
            all_second.extend(list(second))
        else:
            all_second.append(second.numpy())

    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float64)
    if is_test:
        return logits_arr, all_second
    labels_arr = np.concatenate(all_second, axis=0).astype(int)
    return logits_arr, labels_arr


def evaluate_on_validation(
    eval_model,
    loader,
    val_meta_df: pd.DataFrame,
    binary_metrics_fn,
    device,
    use_amp: bool,
    amp_dtype,
    tta_views: int = VAL_MONITOR_TTA_VIEWS,
):
    logits, labels = predict_logits(
        eval_model=eval_model,
        loader=loader,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        tta_views=tta_views,
        is_test=False,
    )
    probs = logits_to_probs(logits)
    metrics = binary_metrics_fn(labels, probs)

    per_center = {}
    if "center_id" in val_meta_df.columns and len(val_meta_df) == len(labels):
        for cid in sorted(val_meta_df["center_id"].unique().tolist()):
            mask = val_meta_df["center_id"].values == cid
            if len(np.unique(labels[mask])) == 2:
                per_center[str(cid)] = binary_metrics_fn(labels[mask], probs[mask])
    return metrics, per_center, logits, probs, labels


# =============================================================================
# 9. End-to-End Pipeline Execution
# =============================================================================
def main():
    seed_everything(SEED)
    binary_metrics = load_competition_metric(INPUT_DIR)

    # -------------------------------------------------------------------------
    # Step A: Data Preparation & Train-Only Stain Statistics
    # -------------------------------------------------------------------------
    train_df, val_df, full_train_df, test_df, sample_sub_df = (
        load_and_validate_metadata(INPUT_DIR)
    )

    # Fit stain/color statistics STRICTLY on train_df for leakage-free validation
    train_stain_stats = fit_stain_statistics(train_df, max_samples=2400, seed=SEED)

    # Fit separate stain/color statistics on full_train_df strictly for post-validation refit
    full_train_stain_stats = fit_stain_statistics(
        full_train_df, max_samples=2400, seed=SEED
    )

    # Persist prepared metadata and fitted stain statistics to ./working/
    train_df.to_csv(os.path.join(WORKING_DIR, "train_prepared.csv"), index=False)
    val_df.to_csv(os.path.join(WORKING_DIR, "val_prepared.csv"), index=False)
    full_train_df.to_csv(
        os.path.join(WORKING_DIR, "full_train_prepared.csv"), index=False
    )
    test_df.to_csv(os.path.join(WORKING_DIR, "test_prepared.csv"), index=False)

    with open(os.path.join(WORKING_DIR, "train_stain_stats.json"), "w") as f:
        json.dump(train_stain_stats, f, indent=2)
    with open(os.path.join(WORKING_DIR, "full_train_stain_stats.json"), "w") as f:
        json.dump(full_train_stain_stats, f, indent=2)

    # Build DataLoaders with return_train_meta=True so training-only Class-Conditional
    # CORAL receives center_idx (validation and test loaders never use metadata)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        stain_stats=train_stain_stats,
        batch_size=64,
        eval_batch_size=128,
        num_workers=4,
        use_weighted_sampler=True,
        use_reinhard_norm=False,
        strong_stain_aug=True,
        return_train_meta=True,
    )

    # Fast batch integrity check
    x_tr_anchor, x_tr_domain, y_tr, _ = next(iter(train_loader))
    x_va, y_va = next(iter(val_loader))
    x_te, id_te = next(iter(test_loader))
    assert (
        torch.isfinite(x_tr_anchor).all()
        and torch.isfinite(x_tr_domain).all()
        and torch.isfinite(x_va).all()
        and torch.isfinite(x_te).all()
    ), "Non-finite pixel tensor detected!"

    print(
        f"Data Ready | Train: {len(train_df)} ({train_df['slide_id'].nunique()} slides, "
        f"pos={train_df['label'].mean():.3f}) | Val: {len(val_df)} ({val_df['slide_id'].nunique()} slides, "
        f"pos={val_df['label'].mean():.3f}) | FullTrain: {len(full_train_df)} | Test: {len(test_df)}"
    )
    print(
        f"Train Centers: {dict(train_df['center_id'].value_counts())} | "
        f"Val Centers: {dict(val_df['center_id'].value_counts())}"
    )
    print(
        f"Batch Shapes | Train Anchor/Domain: {tuple(x_tr_anchor.shape)}/{tuple(x_tr_domain.shape)}, "
        f"Val: {tuple(x_va.shape)}, Test: {tuple(x_te.shape)} | "
        f"Train Global LAB Mean: {[round(v, 1) for v in train_stain_stats['global_lab_mean']]}"
    )

    # -------------------------------------------------------------------------
    # Step B: Model, EMA, Loss, Optimizer & Scheduler Construction
    # -------------------------------------------------------------------------
    seed_everything(SEED)
    model = build_model(
        backbone_name=MODEL_CONFIG["backbone_name"],
        pretrained=MODEL_CONFIG["pretrained"],
        drop_path_rate=MODEL_CONFIG["drop_path_rate"],
        use_mixstyle=MODEL_CONFIG["use_mixstyle"],
        mixstyle_p=MODEL_CONFIG["mixstyle_p"],
        mixstyle_alpha=MODEL_CONFIG["mixstyle_alpha"],
        device=DEVICE,
    )
    model_ema = ModelEMA(model, decay=MODEL_CONFIG["ema_decay"])
    criterion = build_criterion(
        label_smoothing=MODEL_CONFIG["label_smoothing"],
        coral_weight=MODEL_CONFIG["coral_weight"],
        logit_cons_weight=MODEL_CONFIG["logit_cons_weight"],
        feat_cons_weight=MODEL_CONFIG["feat_cons_weight"],
        device=DEVICE,
    )
    optimizer = build_optimizer(
        model,
        backbone_lr=MODEL_CONFIG["backbone_lr"],
        head_lr=MODEL_CONFIG["head_lr"],
        weight_decay=MODEL_CONFIG["weight_decay"],
        layer_decay=MODEL_CONFIG["layer_decay"],
    )

    num_epochs = int(MODEL_CONFIG["default_epochs"])
    phase1_total_steps = num_epochs * len(train_loader)
    scheduler = build_lr_scheduler(
        optimizer,
        total_steps=phase1_total_steps,
        warmup_ratio=0.10,
        min_lr_ratio=0.01,
    )

    # Lightweight architectural integrity check
    model.eval()
    with torch.no_grad():
        dummy_x = torch.zeros((2, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)
        dummy_logits, _ = model(dummy_x, return_features=True)
        dummy_probs = torch.sigmoid(dummy_logits)
        assert dummy_logits.shape == (
            2,
        ), f"Unexpected logits shape: {dummy_logits.shape}"
        assert (
            torch.isfinite(dummy_probs).all()
            and ((dummy_probs >= 0.0) & (dummy_probs <= 1.0)).all()
        )

    n_total_params = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model Ready | Backbone: {model.resolved_backbone_name} (MixStyle hooks={len(model.mixstyle_layers)}) | "
        f"EmbedDim: {model.embed_dim} | Params: {n_total_params / 1e6:.2f}M (trainable {n_train_params / 1e6:.2f}M) | "
        f"Device: {DEVICE} | Loss: Paired SymLS-BCE(eps={MODEL_CONFIG['label_smoothing']}) + "
        f"ClassCondCORAL(w={MODEL_CONFIG['coral_weight']}) + "
        f"LogitCons(w={MODEL_CONFIG['logit_cons_weight']}) + "
        f"FeatCons(w={MODEL_CONFIG['feat_cons_weight']})"
    )

    # -------------------------------------------------------------------------
    # Step C: Phase 1 Training on train.csv & Strict Validation on validation.csv
    # -------------------------------------------------------------------------
    run_start_time = time.time()
    seed_everything(SEED)

    scaler = make_grad_scaler(USE_GRAD_SCALER)
    base_ema_decay = float(MODEL_CONFIG["ema_decay"])
    best_ckpt_path = os.path.join(WORKING_DIR, "best_model.pt")

    best_score_tuple = (-1.0, -1.0)
    best_epoch = 0
    best_variant = "ema"
    epochs_no_improve = 0
    global_step = 0
    history = []
    epoch_durations = []

    print(
        f"Phase 1 Training Start | Epochs: {num_epochs} | Steps/Epoch: {len(train_loader)} | "
        f"AMP: {USE_AMP} ({AMP_DTYPE}) | Metric: Official binary_metrics (thr=0.5)"
    )

    for epoch in range(1, num_epochs + 1):
        ep_start = time.time()

        train_loss, global_step = train_one_epoch(
            train_model=model,
            ema_wrapper=model_ema,
            loader=train_loader,
            loss_fn=criterion,
            opt=optimizer,
            sched=scheduler,
            scaler=scaler,
            device=DEVICE,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            use_grad_scaler=USE_GRAD_SCALER,
            base_ema_decay=base_ema_decay,
            global_step=global_step,
            max_grad_norm=MAX_GRAD_NORM,
            total_steps=phase1_total_steps,
            cons_warmup_ratio=MODEL_CONFIG["cons_warmup_ratio"],
        )

        raw_metrics, raw_centers, _, _, _ = evaluate_on_validation(
            eval_model=model,
            loader=val_loader,
            val_meta_df=val_df,
            binary_metrics_fn=binary_metrics,
            device=DEVICE,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            tta_views=VAL_MONITOR_TTA_VIEWS,
        )
        ema_metrics, ema_centers, _, _, _ = evaluate_on_validation(
            eval_model=model_ema.ema,
            loader=val_loader,
            val_meta_df=val_df,
            binary_metrics_fn=binary_metrics,
            device=DEVICE,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            tta_views=VAL_MONITOR_TTA_VIEWS,
        )

        raw_tuple = (
            round(raw_metrics["balanced_accuracy"], 6),
            round(raw_metrics["auroc"], 6),
        )
        ema_tuple = (
            round(ema_metrics["balanced_accuracy"], 6),
            round(ema_metrics["auroc"], 6),
        )

        if ema_tuple >= raw_tuple:
            ep_best_tuple = ema_tuple
            ep_best_variant = "ema"
            ep_best_model = model_ema.ema
            ep_best_centers = ema_centers
        else:
            ep_best_tuple = raw_tuple
            ep_best_variant = "raw"
            ep_best_model = model
            ep_best_centers = raw_centers

        improved = False
        if ep_best_tuple > best_score_tuple:
            best_score_tuple = ep_best_tuple
            best_epoch = epoch
            best_variant = ep_best_variant
            epochs_no_improve = 0
            improved = True
            torch.save(
                {
                    "epoch": epoch,
                    "variant": best_variant,
                    "val_balanced_accuracy": best_score_tuple[0],
                    "val_auroc": best_score_tuple[1],
                    "model_config": MODEL_CONFIG,
                    "state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in ep_best_model.state_dict().items()
                    },
                },
                best_ckpt_path,
            )
        else:
            epochs_no_improve += 1

        ep_time = time.time() - ep_start
        epoch_durations.append(ep_time)
        cur_lr = optimizer.param_groups[0]["lr"]

        center_str = " ".join(
            f"c{cid}={cm['balanced_accuracy']:.4f}"
            for cid, cm in ep_best_centers.items()
        )
        marker = "*" if improved else ""
        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | Loss: {train_loss:.4f} | "
            f"Val Raw BalAcc/AUC: {raw_metrics['balanced_accuracy']:.4f}/{raw_metrics['auroc']:.4f} | "
            f"EMA BalAcc/AUC: {ema_metrics['balanced_accuracy']:.4f}/{ema_metrics['auroc']:.4f} | "
            f"{center_str} | LR: {cur_lr:.2e} | {ep_time:.1f}s {marker}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "raw_val_bal_acc": raw_metrics["balanced_accuracy"],
                "raw_val_auroc": raw_metrics["auroc"],
                "ema_val_bal_acc": ema_metrics["balanced_accuracy"],
                "ema_val_auroc": ema_metrics["auroc"],
                "lr_backbone": cur_lr,
                "epoch_sec": ep_time,
            }
        )

        if epoch >= MIN_EPOCHS and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"Early stopping at epoch {epoch} (best epoch={best_epoch}, "
                f"best 1-view Val BalAcc={best_score_tuple[0]:.4f})"
            )
            break

        if (time.time() - run_start_time) > TRAIN_TIME_LIMIT_SEC:
            print(
                f"Stopping Phase 1 at epoch {epoch} to preserve inference time budget."
            )
            break

    pd.DataFrame(history).to_csv(
        os.path.join(WORKING_DIR, "training_history.csv"), index=False
    )

    # -------------------------------------------------------------------------
    # Step D: Reload Best Checkpoint & Compute Official Pre-Refit Validation Score
    # -------------------------------------------------------------------------
    assert os.path.exists(best_ckpt_path), "Best model checkpoint was not saved!"
    best_ckpt = safe_torch_load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt["state_dict"])
    model.eval()

    final_val_metrics, final_val_centers, val_logits, val_probs, val_labels = (
        evaluate_on_validation(
            eval_model=model,
            loader=val_loader,
            val_meta_df=val_df,
            binary_metrics_fn=binary_metrics,
            device=DEVICE,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            tta_views=FINAL_TTA_VIEWS,
        )
    )

    validation_balanced_accuracy = float(final_val_metrics["balanced_accuracy"])
    validation_auroc = float(final_val_metrics["auroc"])

    val_pred_df = pd.DataFrame(
        {
            "sample_id": val_df["sample_id"].values,
            "label": val_labels,
            "center_id": val_df["center_id"].values,
            "slide_id": val_df["slide_id"].values,
            "tumor_logit": val_logits,
            "tumor_probability": val_probs,
        }
    )
    val_pred_df.to_csv(os.path.join(WORKING_DIR, "val_predictions.csv"), index=False)

    center_audit = " | ".join(
        f"Center {cid}: BalAcc={cm['balanced_accuracy']:.4f}, AUROC={cm['auroc']:.4f} (n={cm['n_samples']})"
        for cid, cm in final_val_centers.items()
    )
    print(
        f"Best Model ({best_variant}, epoch {best_epoch}) {FINAL_TTA_VIEWS}x D4-TTA Validation | "
        f"BalAcc: {validation_balanced_accuracy:.6f} | AUROC: {validation_auroc:.6f} | "
        f"Acc: {final_val_metrics['accuracy']:.6f} | PosRate@0.5: {(val_probs >= 0.5).mean():.4f}"
    )
    print(f"Per-Center Validation Audit | {center_audit}")

    # Compute test logits from the validated best checkpoint using IDENTICAL 8x D4 TTA
    best_test_logits, test_sample_ids = predict_logits(
        eval_model=model,
        loader=test_loader,
        device=DEVICE,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        tta_views=FINAL_TTA_VIEWS,
        is_test=True,
    )
    assert (
        np.asarray(test_sample_ids) == sample_sub_df["sample_id"].values
    ).all(), "Test loader sample_id order mismatch!"

    # -------------------------------------------------------------------------
    # Step E: Optional Post-Selection Refit on train + validation (full_train_df)
    #         (Validation score above is strictly frozen BEFORE refit)
    # -------------------------------------------------------------------------
    avg_epoch_sec = (
        float(np.mean(epoch_durations)) if len(epoch_durations) > 0 else 60.0
    )
    refit_epochs = max(6, min(best_epoch, num_epochs))
    est_refit_sec = (
        avg_epoch_sec * (len(full_train_df) / max(len(train_df), 1)) * refit_epochs
    )
    elapsed_before_refit = time.time() - run_start_time

    final_test_logits = best_test_logits.copy()
    refit_used = False

    if (elapsed_before_refit < 1800.0) and (
        elapsed_before_refit + est_refit_sec < TOTAL_TIME_LIMIT_SEC
    ):
        del model_ema, optimizer, scheduler, train_loader, val_loader, test_loader
        model.to("cpu")
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"Phase 2 Post-Selection Refit Start | Samples: {len(full_train_df)} "
            f"(train+val) | Epochs: {refit_epochs} | Est: {est_refit_sec:.0f}s"
        )
        seed_everything(SEED + 100)

        refit_train_loader, _, refit_test_loader = create_dataloaders(
            train_df=full_train_df,
            val_df=val_df,
            test_df=test_df,
            stain_stats=full_train_stain_stats,
            batch_size=64,
            eval_batch_size=128,
            num_workers=4,
            use_weighted_sampler=True,
            use_reinhard_norm=False,
            strong_stain_aug=True,
            return_train_meta=True,
        )

        refit_model = build_model(
            backbone_name=MODEL_CONFIG["backbone_name"],
            pretrained=MODEL_CONFIG["pretrained"],
            drop_path_rate=MODEL_CONFIG["drop_path_rate"],
            use_mixstyle=MODEL_CONFIG["use_mixstyle"],
            mixstyle_p=MODEL_CONFIG["mixstyle_p"],
            mixstyle_alpha=MODEL_CONFIG["mixstyle_alpha"],
            device=DEVICE,
        )
        refit_ema = ModelEMA(refit_model, decay=base_ema_decay)
        refit_criterion = build_criterion(
            label_smoothing=MODEL_CONFIG["label_smoothing"],
            coral_weight=MODEL_CONFIG["coral_weight"],
            logit_cons_weight=MODEL_CONFIG["logit_cons_weight"],
            feat_cons_weight=MODEL_CONFIG["feat_cons_weight"],
            device=DEVICE,
        )
        refit_optimizer = build_optimizer(
            refit_model,
            backbone_lr=MODEL_CONFIG["backbone_lr"],
            head_lr=MODEL_CONFIG["head_lr"],
            weight_decay=MODEL_CONFIG["weight_decay"],
            layer_decay=MODEL_CONFIG["layer_decay"],
        )
        refit_total_steps = refit_epochs * len(refit_train_loader)
        refit_scheduler = build_lr_scheduler(
            refit_optimizer,
            total_steps=refit_total_steps,
            warmup_ratio=0.10,
            min_lr_ratio=0.01,
        )
        refit_scaler = make_grad_scaler(USE_GRAD_SCALER)
        refit_global_step = 0

        for ep in range(1, refit_epochs + 1):
            ep_t0 = time.time()
            rf_loss, refit_global_step = train_one_epoch(
                train_model=refit_model,
                ema_wrapper=refit_ema,
                loader=refit_train_loader,
                loss_fn=refit_criterion,
                opt=refit_optimizer,
                sched=refit_scheduler,
                scaler=refit_scaler,
                device=DEVICE,
                use_amp=USE_AMP,
                amp_dtype=AMP_DTYPE,
                use_grad_scaler=USE_GRAD_SCALER,
                base_ema_decay=base_ema_decay,
                global_step=refit_global_step,
                max_grad_norm=MAX_GRAD_NORM,
                total_steps=refit_total_steps,
                cons_warmup_ratio=MODEL_CONFIG["cons_warmup_ratio"],
            )
            print(
                f"Refit Epoch {ep:02d}/{refit_epochs:02d} | Loss: {rf_loss:.4f} | "
                f"LR: {refit_optimizer.param_groups[0]['lr']:.2e} | {time.time() - ep_t0:.1f}s"
            )
            if (time.time() - run_start_time) > TOTAL_TIME_LIMIT_SEC:
                print(f"Stopping refit early at epoch {ep} within time budget.")
                break

        refit_ckpt_path = os.path.join(WORKING_DIR, "refit_model.pt")
        torch.save(
            {
                "epochs": refit_epochs,
                "model_config": MODEL_CONFIG,
                "state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in refit_ema.ema.state_dict().items()
                },
            },
            refit_ckpt_path,
        )

        refit_test_logits, refit_test_ids = predict_logits(
            eval_model=refit_ema.ema,
            loader=refit_test_loader,
            device=DEVICE,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            tta_views=FINAL_TTA_VIEWS,
            is_test=True,
        )
        assert (
            np.asarray(refit_test_ids) == sample_sub_df["sample_id"].values
        ).all(), "Refit test sample_id mismatch!"

        best_test_probs_chk = logits_to_probs(best_test_logits)
        refit_test_probs_chk = logits_to_probs(refit_test_logits)
        refit_corr = float(np.corrcoef(best_test_probs_chk, refit_test_probs_chk)[0, 1])

        if (
            np.isfinite(refit_test_logits).all()
            and np.std(refit_test_probs_chk) > 0.05
            and refit_corr >= 0.85
        ):
            final_test_logits = 0.40 * best_test_logits + 0.60 * refit_test_logits
            refit_used = True
            print(
                f"Refit Verified & Blended (0.40 val-ckpt + 0.60 train+val refit) | "
                f"Prob Corr: {refit_corr:.4f}"
            )
        else:
            print(
                f"Refit sanity check fell below threshold (corr={refit_corr:.4f}); "
                f"retaining validated best checkpoint predictions."
            )
    else:
        print(
            f"Skipping Phase 2 refit to stay within time budget "
            f"(elapsed={elapsed_before_refit:.0f}s, est_refit={est_refit_sec:.0f}s)."
        )

    # -------------------------------------------------------------------------
    # Step F: Generate, Strictly Verify & Save ./submission/submission.csv
    # -------------------------------------------------------------------------
    final_test_probs = logits_to_probs(final_test_logits)

    submission_df = pd.DataFrame(
        {
            "sample_id": sample_sub_df["sample_id"].values,
            "tumor_probability": final_test_probs.astype(np.float64),
        }
    )

    assert list(submission_df.columns) == ["sample_id", "tumor_probability"]
    assert len(submission_df) == len(sample_sub_df) == len(test_df)
    assert (
        submission_df["sample_id"].values == sample_sub_df["sample_id"].values
    ).all()
    assert submission_df["tumor_probability"].notna().all()
    assert np.isfinite(submission_df["tumor_probability"].values).all()
    assert (
        (submission_df["tumor_probability"].values >= 0.0)
        & (submission_df["tumor_probability"].values <= 1.0)
    ).all()
    assert (
        float(submission_df["tumor_probability"].std()) > 1e-3
    ), "Submission probabilities appear constant!"

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    eval_summary = {
        "pretrained_model_card": PRETRAINED_MODEL_CARD,
        "best_epoch": int(best_epoch),
        "best_variant": best_variant,
        "tta_views": int(FINAL_TTA_VIEWS),
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "validation_auroc": validation_auroc,
        "validation_accuracy": float(final_val_metrics["accuracy"]),
        "per_center_validation": final_val_centers,
        "post_selection_refit_used": bool(refit_used),
        "submission_rows": int(len(submission_df)),
        "submission_mean_prob": float(submission_df["tumor_probability"].mean()),
        "submission_pos_rate_0_5": float(
            (submission_df["tumor_probability"] >= 0.5).mean()
        ),
        "total_runtime_sec": round(time.time() - run_start_time, 1),
    }
    with open(os.path.join(WORKING_DIR, "eval_summary.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(
        f"Submission Saved: {submission_path} | Rows: {len(submission_df)} | "
        f"Prob Mean/Std: {submission_df['tumor_probability'].mean():.4f}/"
        f"{submission_df['tumor_probability'].std():.4f} | "
        f"Range: [{submission_df['tumor_probability'].min():.4f}, "
        f"{submission_df['tumor_probability'].max():.4f}] | "
        f"PosRate@0.5: {(submission_df['tumor_probability'] >= 0.5).mean():.4f} | "
        f"Total Time: {time.time() - run_start_time:.1f}s"
    )
    print(f"Final Validation Score: {validation_balanced_accuracy}")


if __name__ == "__main__":
    main()
