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


def _extract_foreground_stain_stats(path: str) -> np.ndarray:
    """Fast 2x-strided foreground-masked Beer-Lambert & CIE LAB stain calibration pass."""
    img_rgb = read_rgb_image(path)
    sub_rgb = img_rgb[::2, ::2]
    her, od = STAIN_DECONVOLVER.rgb_to_her(sub_rgb)

    H = np.maximum(her[:, :, 0], 0.0)
    E = np.maximum(her[:, :, 1], 0.0)
    od_norm = np.linalg.norm(od, axis=2)

    img_lab = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    fg_mask = od_norm > 0.12
    fg_count = int(np.count_nonzero(fg_mask))
    if fg_count >= 16:
        fg_lab = img_lab[fg_mask]
        lab_means = fg_lab.mean(axis=0)
        lab_stds = fg_lab.std(axis=0)
        fg_h = H[fg_mask]
        fg_e = E[fg_mask]
        fg_od = od_norm[fg_mask]
        her_summary = np.array(
            [
                float(fg_h.mean()),
                float(fg_h.std()),
                float(fg_e.mean()),
                float(fg_e.std()),
                float(fg_od.mean()),
                float(fg_od.std()),
            ],
            dtype=np.float32,
        )
        fg_valid = np.array([1.0], dtype=np.float32)
    else:
        lab_means = img_lab.mean(axis=(0, 1))
        lab_stds = img_lab.std(axis=(0, 1))
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
        fg_valid = np.array([0.0], dtype=np.float32)

    rgb_f = sub_rgb.astype(np.float32) / 255.0
    rgb_means = rgb_f.mean(axis=(0, 1))
    rgb_sq_means = (rgb_f * rgb_f).mean(axis=(0, 1))
    aux_stats = np.concatenate(
        [lab_means, lab_stds, rgb_means, rgb_sq_means, her_summary, fg_valid]
    ).astype(np.float32)
    return np.nan_to_num(aux_stats, nan=0.0, posinf=0.0, neginf=0.0)


def extract_foreground_stain_dataset(
    paths_series: pd.Series, max_workers: int = 12
) -> np.ndarray:
    paths_list = paths_series.tolist()
    n = len(paths_list)
    aux_out = np.zeros((n, 19), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, a_vec in enumerate(
            executor.map(_extract_foreground_stain_stats, paths_list)
        ):
            aux_out[idx] = a_vec
    return aux_out


calib_t0 = time.time()
num_io_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
aux_train = extract_foreground_stain_dataset(
    train_df["full_image_path"], max_workers=num_io_workers
)
aux_val = extract_foreground_stain_dataset(
    val_df["full_image_path"], max_workers=num_io_workers
)
assert np.isfinite(aux_train).all() and np.isfinite(aux_val).all()


# ============================================================================
# 5. Fit Stateful Stain Reference Stats STRICTLY on Training Data
# ============================================================================
def fit_stain_reference_stats(
    aux_matrix: np.ndarray, domain_expansion: float = 1.25
) -> dict:
    """
    Fits RandStainNA virtual-center foreground LAB template distribution and Beer-Lambert
    channel normalization statistics strictly on training foreground tissue (never val or test).
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

    h_mean = float(fg_aux[:, 12].mean())
    h_std = float(np.sqrt(np.mean(fg_aux[:, 13] ** 2) + np.var(fg_aux[:, 12])))
    e_mean = float(fg_aux[:, 14].mean())
    e_std = float(np.sqrt(np.mean(fg_aux[:, 15] ** 2) + np.var(fg_aux[:, 14])))
    od_mean = float(fg_aux[:, 16].mean())
    od_std = float(np.sqrt(np.mean(fg_aux[:, 17] ** 2) + np.var(fg_aux[:, 16])))

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

y_train = train_df["label"].values.astype(np.int64)
y_val = val_df["label"].values.astype(np.int64)
y_merged = merged_df["label"].values.astype(np.int64)

print(
    f"Calibrated foreground-masked Beer-Lambert & RandStainNA LAB stats strictly on train_df "
    f"in {time.time() - calib_t0:.1f}s (test_df untouched) | "
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
    train_df, stain_stats=stain_stats_train, features=None, is_train=True
)
val_dataset = HistopathologyDataset(
    val_df, stain_stats=stain_stats_train, features=None, is_train=False
)
test_dataset = HistopathologyDataset(
    test_df, stain_stats=stain_stats_train, features=None, is_train=False
)

sample_train_item = train_dataset[0]
sample_val_item = val_dataset[0]
sample_test_item = test_dataset[0]
assert (
    sample_train_item["image"].shape == (3, 256, 256)
    and torch.isfinite(sample_train_item["image"]).all()
    and "features" not in sample_train_item
)
assert (
    sample_val_item["image"].shape == (3, 256, 256)
    and torch.isfinite(sample_val_item["image"]).all()
)
assert (
    sample_test_item["image"].shape == (3, 256, 256)
    and torch.isfinite(sample_test_item["image"]).all()
)

# Persist processed splits and stain reference artifacts
train_df.to_csv(os.path.join(WORKING_DIR, "train_split.csv"), index=False)
val_df.to_csv(os.path.join(WORKING_DIR, "val_split.csv"), index=False)
merged_df.to_csv(os.path.join(WORKING_DIR, "merged_split.csv"), index=False)
test_df.to_csv(os.path.join(WORKING_DIR, "test_split.csv"), index=False)

with open(os.path.join(WORKING_DIR, "stain_stats.json"), "w") as f_json:
    json.dump(
        {
            "stain_stats_train": stain_stats_train,
            "stain_stats_merged": stain_stats_merged,
        },
        f_json,
        indent=2,
    )

print(f"Saved processed splits and stain reference stats to {WORKING_DIR}.")

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
    "vit_backbone_candidates": (
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
    ),
    "vit_img_size": 252,
    "pretrained_source": (
        "ConvNeXt-Small: ImageNet-22K pretrained + ImageNet-1K 384x384 fine-tuned (natural photos); "
        "Register-DINOv2 ViT-B/14: Self-supervised LVD-142M curated natural photographs"
    ),
    "training_data_overlap": (
        "None (0% overlap with H&E histopathology or lymph-node microscopy datasets across both backbones)"
    ),
    "num_morph_features": 46,
    "drop_path_rate": 0.25,
    "vit_drop_path_rate": 0.20,
    "head_dropout": 0.35,
    "morph_dropout": 0.20,
    "mixstyle_p": 0.50,
    "mixstyle_alpha": 0.20,
    "vit_mixstyle_p": 0.45,
    "vit_mixstyle_alpha": 0.20,
    "lr_backbone": 1.2e-4,
    "lr_vit_backbone": 4.0e-5,
    "lr_head": 5.0e-4,
    "layer_decay": 0.80,
    "weight_decay": 2.0e-2,
    "label_smoothing": 0.03,
    "ba_loss_weight": 0.25,
    "ba_margin_temp": 0.70,
    "group_dro_eta": 0.06,
    "group_dro_decay": 0.01,
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


class SpatialTokenMixStyle(nn.Module):
    """
    Training-only fp32 cross-sample style mixing applied strictly to ViT spatial tokens,
    leaving CLS and register prefix tokens intact. Deterministic identity at eval/inference.
    """

    def __init__(self, p: float = 0.45, alpha: float = 0.20, eps: float = 1e-6):
        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self._beta_dist = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x_tok: torch.Tensor, n_prefix: int = 5) -> torch.Tensor:
        if (
            (not self.training)
            or (x_tok.size(0) <= 1)
            or (torch.rand(1).item() > self.p)
        ):
            return x_tok

        orig_dtype = x_tok.dtype
        prefix_tok = x_tok[:, :n_prefix, :]
        sp_tok = x_tok[:, n_prefix:, :]

        with torch.amp.autocast(device_type=x_tok.device.type, enabled=False):
            sp_f = sp_tok.float()
            B = sp_f.size(0)

            mu = sp_f.mean(dim=1, keepdim=True)
            var = (sp_f - mu).pow(2).mean(dim=1, keepdim=True)
            sig = torch.sqrt(var + self.eps)
            sp_normed = (sp_f - mu) / sig

            perm = torch.randperm(B, device=x_tok.device)
            mu_perm = mu.detach()[perm]
            sig_perm = sig.detach()[perm]

            lmbd = self._beta_dist.sample((B, 1, 1)).to(
                device=x_tok.device, dtype=torch.float32
            )
            mu_mix = lmbd * mu.detach() + (1.0 - lmbd) * mu_perm
            sig_mix = lmbd * sig.detach() + (1.0 - lmbd) * sig_perm
            sp_mixed = sp_normed * sig_mix + mu_mix

        return torch.cat([prefix_tok, sp_mixed.to(orig_dtype)], dim=1)


class BeerLambertViTStemAdapter(nn.Module):
    """
    Differentiable on-GPU Beer-Lambert H, E, and Optical Density stem adapter for DINOv2 ViT.
    Applies fp32 smooth foreground tissue weighting (od_norm > 0.12 with low-coverage fallback)
    and projects via 14x14 stride-14 conv + ChannelLayerNorm2d + 3x3 depthwise-separable conv
    with residual gate_alpha=0.10 added strictly to spatial tokens.
    """

    def __init__(self, embed_dim: int, patch_size: int = 14):
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
            nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True
            ),
            ChannelLayerNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                groups=embed_dim,
                bias=True,
            ),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True),
            ChannelLayerNorm2d(embed_dim),
        )
        self.gate_alpha = nn.Parameter(torch.full((1, 1, embed_dim), 0.10))

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        orig_dtype = x_rgb_norm.dtype
        with torch.amp.autocast(device_type=x_rgb_norm.device.type, enabled=False):
            x_f = x_rgb_norm[:, :3, :, :].float()
            rgb_01 = torch.clamp(x_f * self.rgb_std + self.rgb_mean, 1.0 / 255.0, 1.0)
            od = -torch.log10(rgb_01)
            her = F.conv2d(od, self.stain_inv_t)

            od_norm = torch.sqrt(torch.sum(od * od, dim=1, keepdim=True) + 1e-8)
            he_od = torch.cat(
                [
                    F.relu(her[:, 0:1, :, :]),
                    F.relu(her[:, 1:2, :, :]),
                    od_norm,
                ],
                dim=1,
            )

            w_fg = torch.sigmoid((od_norm - 0.12) / 0.04)
            tissue_cov = (od_norm > 0.12).float().mean(dim=(2, 3), keepdim=True)
            t_blend = torch.clamp(tissue_cov / 0.05, 0.0, 1.0)
            alpha_cov = t_blend * t_blend * (3.0 - 2.0 * t_blend)

            p_fg = w_fg / w_fg.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            p_unif = 1.0 / float(he_od.size(2) * he_od.size(3))
            p_spatial = alpha_cov * p_fg + (1.0 - alpha_cov) * p_unif

            mu = (he_od * p_spatial).sum(dim=(2, 3), keepdim=True)
            var = ((he_od - mu).pow(2) * p_spatial).sum(dim=(2, 3), keepdim=True)
            he_od_norm = torch.clamp(
                (he_od - mu) * torch.rsqrt(var + 1e-5), -6.0, 6.0
            )
            he_od_in = he_od_norm * self.fg_norm_weight + self.fg_norm_bias

        stem_map = self.proj(he_od_in.to(orig_dtype))
        stem_tokens = stem_map.flatten(2).transpose(1, 2)
        return stem_tokens * self.gate_alpha.to(orig_dtype)


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


class MultiHeadFocalMILPool2d(nn.Module):
    """
    D4-invariant 4-head temperature-scaled Gated Focal MIL spatial attention pooling
    fused with D4-invariant soft-peak LogSumExp pooling. Captures sparse or pale-nuclei
    micrometastatic clusters across multiple spatial focus scales without background washout.
    """

    def __init__(
        self,
        in_channels: int,
        attn_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_heads = int(num_heads)

        self.attn_v = nn.Sequential(
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=True),
            nn.Tanh(),
        )
        self.attn_u = nn.Sequential(
            nn.Conv2d(in_channels, attn_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.attn_w = nn.Conv2d(attn_dim, num_heads, kernel_size=1, bias=True)

        init_temps = torch.linspace(0.35, 1.00, steps=num_heads, dtype=torch.float32)
        self.attn_log_temp = nn.Parameter(torch.log(init_temps))
        self.head_mix_logits = nn.Parameter(torch.zeros(1, num_heads, in_channels))

        self.lse_log_temp = nn.Parameter(
            torch.full((1, in_channels), math.log(2.0), dtype=torch.float32)
        )
        self.gate_alpha = nn.Parameter(torch.full((1, in_channels), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        N = H * W

        gated = self.attn_drop(self.attn_v(x) * self.attn_u(x))
        raw_scores = self.attn_w(gated).view(B, self.num_heads, N)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            tokens_f = x.float().view(B, C, N)
            attn_temp = (
                torch.exp(self.attn_log_temp.float())
                .clamp(min=0.10, max=4.0)
                .view(1, self.num_heads, 1)
            )
            weights = F.softmax(raw_scores.float() / attn_temp, dim=-1)
            head_pooled = torch.bmm(weights, tokens_f.transpose(1, 2))
            head_mix = F.softmax(self.head_mix_logits.float(), dim=1)
            mil_pooled = (head_pooled * head_mix).sum(dim=1)

            lse_beta = torch.exp(self.lse_log_temp.float()).clamp(min=0.5, max=8.0)
            tokens_clamped = torch.clamp(tokens_f, min=-12.0, max=12.0)
            lse_peak = (
                torch.logsumexp(tokens_clamped * lse_beta.unsqueeze(-1), dim=-1)
                - math.log(float(max(1, N)))
            ) / lse_beta

            pooled = mil_pooled + self.gate_alpha.float() * lse_peak

        return pooled.to(orig_dtype)


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


class DifferentiableBeerLambertMorphExtractor(nn.Module):
    """
    On-GPU fp32 differentiable, D4-invariant, foreground-standardized Beer-Lambert
    morphological descriptor module (46 features). Computed dynamically inside forward()
    so all morphological cues undergo stain randomization, D4 TTA views, and Deep CORAL.
    """

    def __init__(self, out_features: int = 46):
        super().__init__()
        self.out_features = out_features
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

        sobel_x = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
            / 8.0
        )
        sobel_y = sobel_x.t().contiguous()
        lap_k = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            )
            / 4.0
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3).repeat(4, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3).repeat(4, 1, 1, 1))
        self.register_buffer("lap_k", lap_k.view(1, 1, 3, 3).repeat(4, 1, 1, 1))

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=x_rgb_norm.device.type, enabled=False):
            x_f = x_rgb_norm[:, :3, :, :].float()
            _, _, H, W = x_f.shape
            rgb_01 = torch.clamp(x_f * self.rgb_std + self.rgb_mean, 1.0 / 255.0, 1.0)
            od = -torch.log10(rgb_01)
            her = F.conv2d(od, self.stain_inv_t)
            od_norm = torch.sqrt(torch.sum(od * od, dim=1, keepdim=True) + 1e-8)

            w_fg = torch.sigmoid((od_norm - 0.12) / 0.04)
            tissue_cov = (od_norm > 0.12).float().mean(dim=(2, 3), keepdim=True)
            t_blend = torch.clamp(tissue_cov / 0.05, 0.0, 1.0)
            alpha_cov = t_blend * t_blend * (3.0 - 2.0 * t_blend)

            p_fg = w_fg / w_fg.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            p_unif = 1.0 / float(H * W)
            p_sp = alpha_cov * p_fg + (1.0 - alpha_cov) * p_unif

            raw_4 = torch.cat(
                [
                    F.relu(her[:, 0:1, :, :]),
                    F.relu(her[:, 1:2, :, :]),
                    her[:, 2:3, :, :].abs(),
                    od_norm,
                ],
                dim=1,
            )
            mu_4 = (raw_4 * p_sp).sum(dim=(2, 3), keepdim=True)
            var_4 = ((raw_4 - mu_4).pow(2) * p_sp).sum(dim=(2, 3), keepdim=True)
            z_4 = torch.clamp((raw_4 - mu_4) * torch.rsqrt(var_4 + 1e-5), -6.0, 6.0)

            ms_var_feats = []
            corr_scale_feats = []
            for k in (4, 8, 16):
                z_k = F.avg_pool2d(z_4, kernel_size=k, stride=k)
                w_k = F.avg_pool2d(p_sp, kernel_size=k, stride=k)
                p_k = w_k / w_k.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                mu_k = (z_k * p_k).sum(dim=(2, 3), keepdim=True)
                zc_k = z_k - mu_k
                var_k = (zc_k.pow(2) * p_k).sum(dim=(2, 3))
                ms_var_feats.append(var_k)
                if k in (4, 16):
                    cov_he_k = (zc_k[:, 0:1] * zc_k[:, 1:2] * p_k).sum(dim=(2, 3))
                    denom_he = torch.sqrt(var_k[:, 0:1] * var_k[:, 1:2] + 1e-5)
                    corr_scale_feats.append(torch.clamp(cov_he_k / denom_he, -1.0, 1.0))

            z_pad = F.pad(z_4, (1, 1, 1, 1), mode="replicate")
            gx = F.conv2d(z_pad, self.sobel_x, groups=4)
            gy = F.conv2d(z_pad, self.sobel_y, groups=4)
            grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
            lap_abs = F.conv2d(z_pad, self.lap_k, groups=4).abs()

            grad_mean = (grad_mag * p_sp).sum(dim=(2, 3))
            grad_std = torch.sqrt(
                ((grad_mag - grad_mean.unsqueeze(-1).unsqueeze(-1)).pow(2) * p_sp).sum(
                    dim=(2, 3)
                )
                + 1e-6
            )
            lap_mean = (lap_abs * p_sp).sum(dim=(2, 3))

            hm = max(1, int(round(H * 0.3125)))
            wm = max(1, int(round(W * 0.3125)))
            core_mask = torch.zeros(
                (1, 1, H, W), device=x_f.device, dtype=torch.float32
            )
            core_mask[:, :, hm : H - hm, wm : W - wm] = 1.0
            rim_mask = 1.0 - core_mask

            w_core = (w_fg + 1e-4) * core_mask
            p_core = w_core / w_core.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            w_rim = (w_fg + 1e-4) * rim_mask
            p_rim = w_rim / w_rim.sum(dim=(2, 3), keepdim=True).clamp(min=1e-6)

            mu_core = (z_4 * p_core).sum(dim=(2, 3))
            mu_rim = (z_4 * p_rim).sum(dim=(2, 3))
            contrast_mu = mu_core - mu_rim

            std_core = torch.sqrt(
                ((z_4 - mu_core.unsqueeze(-1).unsqueeze(-1)).pow(2) * p_core).sum(
                    dim=(2, 3)
                )
                + 1e-5
            )
            std_rim = torch.sqrt(
                ((z_4 - mu_rim.unsqueeze(-1).unsqueeze(-1)).pow(2) * p_rim).sum(
                    dim=(2, 3)
                )
                + 1e-5
            )
            contrast_std = std_core - std_rim

            grad_ho = grad_mag[:, [0, 3], :, :]
            contrast_grad = (grad_ho * p_core).sum(dim=(2, 3)) - (
                grad_ho * p_rim
            ).sum(dim=(2, 3))

            z_h = z_4[:, 0:1, :, :]
            z_e = z_4[:, 1:2, :, :]
            exc_h_05 = (torch.sigmoid((z_h - 0.5) / 0.25) * p_sp).sum(dim=(2, 3))
            exc_h_10 = (torch.sigmoid((z_h - 1.0) / 0.25) * p_sp).sum(dim=(2, 3))
            exc_h_15 = (torch.sigmoid((z_h - 1.5) / 0.25) * p_sp).sum(dim=(2, 3))
            exc_e_10 = (torch.sigmoid((z_e - 1.0) / 0.25) * p_sp).sum(dim=(2, 3))
            vesic_clear = (
                torch.sigmoid((-z_h - 0.5) / 0.25) * w_fg * p_sp
            ).sum(dim=(2, 3))
            skew_h = (z_h.clamp(-4.0, 4.0).pow(3) * p_sp).sum(dim=(2, 3)) * 0.25
            skew_e = (z_e.clamp(-4.0, 4.0).pow(3) * p_sp).sum(dim=(2, 3)) * 0.25
            kurt_h = ((z_h.clamp(-4.0, 4.0).pow(4) * p_sp).sum(dim=(2, 3)) - 3.0) * 0.15

            corr_he_px = torch.clamp((z_h * z_e * p_sp).sum(dim=(2, 3)), -1.0, 1.0)
            fg_cov_feat = w_fg.mean(dim=(2, 3))

            feats = torch.cat(
                ms_var_feats
                + [
                    grad_mean,
                    grad_std,
                    lap_mean,
                    contrast_mu,
                    contrast_std,
                    contrast_grad,
                    exc_h_05,
                    exc_h_10,
                    exc_h_15,
                    exc_e_10,
                    vesic_clear,
                    skew_h,
                    skew_e,
                    kurt_h,
                    corr_he_px,
                ]
                + corr_scale_feats
                + [fg_cov_feat],
                dim=1,
            )
            feats = torch.clamp(
                torch.nan_to_num(feats, nan=0.0, posinf=6.0, neginf=-6.0), -8.0, 8.0
            )
            return feats


class MultiSampleClassificationHead(nn.Module):
    """Multi-Sample Dropout classification head for calibrated probabilities around p=0.5."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.35,
        num_samples: int = 5,
    ):
        super().__init__()
        self.dropouts_1 = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_samples)])
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.dropouts_2 = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_samples)])
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_acc = 0.0
        for d1, d2 in zip(self.dropouts_1, self.dropouts_2):
            logits_acc = logits_acc + self.fc_out(d2(self.fc1(d1(x))))
        return (logits_acc / float(len(self.dropouts_1))).squeeze(-1)


class StainInvariantMultiScaleConvNeXt(nn.Module):
    """
    Co-Supervised Dual-Stream ConvNeXt-Small + Self-Supervised Register-DINOv2 Network:
      - Stream 1: BatchNorm-free Multi-Scale ConvNeXt-Small + Beer-Lambert Stem + MixStyle2d + MIL/ROI
      - Stream 2: Self-Supervised Register-DINOv2 ViT-B/14 (252x252, register-free 18x18 MIL/ROI + CLS)
      - Morph Stream: On-GPU fp32 differentiable Beer-Lambert morphological extractor (46 features)
      - Cross-Modal Fusion: Gated FiLM modulation into v_fused with co-supervised stream heads
    """

    def __init__(
        self,
        backbone_candidates=MODEL_CONFIG["backbone_candidates"],
        vit_backbone_candidates=MODEL_CONFIG["vit_backbone_candidates"],
        vit_img_size: int = MODEL_CONFIG["vit_img_size"],
        pretrained: bool = True,
        num_morph_features: int = MODEL_CONFIG["num_morph_features"],
        drop_path_rate: float = MODEL_CONFIG["drop_path_rate"],
        vit_drop_path_rate: float = MODEL_CONFIG["vit_drop_path_rate"],
        head_dropout: float = MODEL_CONFIG["head_dropout"],
        morph_dropout: float = MODEL_CONFIG["morph_dropout"],
        mixstyle_p: float = MODEL_CONFIG["mixstyle_p"],
        mixstyle_alpha: float = MODEL_CONFIG["mixstyle_alpha"],
        vit_mixstyle_p: float = MODEL_CONFIG["vit_mixstyle_p"],
        vit_mixstyle_alpha: float = MODEL_CONFIG["vit_mixstyle_alpha"],
    ):
        super().__init__()
        self.num_morph_features = num_morph_features
        self.vit_img_size = int(vit_img_size)

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

        self.vit_backbone = None
        self.vit_backbone_name = None
        last_vit_err = None
        vit_candidates = (
            (vit_backbone_candidates,)
            if isinstance(vit_backbone_candidates, str)
            else tuple(vit_backbone_candidates)
        )
        for v_cand in vit_candidates:
            try:
                self.vit_backbone = timm.create_model(
                    v_cand,
                    pretrained=pretrained,
                    num_classes=0,
                    img_size=self.vit_img_size,
                    drop_path_rate=vit_drop_path_rate,
                )
                self.vit_backbone_name = v_cand
                break
            except Exception as exc:
                last_vit_err = exc
        if self.vit_backbone is None:
            raise RuntimeError(
                f"Failed to instantiate DINOv2 ViT backbone: {last_vit_err}"
            ) from last_vit_err

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            s0 = self.backbone.stem(dummy)
            f0 = self.backbone.stages[0](s0)
            f1 = self.backbone.stages[1](f0)
            f2 = self.backbone.stages[2](f1)
            f3 = self.backbone.stages[3](f2)
            c1, c2, c3 = f1.shape[1], f2.shape[1], f3.shape[1]
            c_vit = int(self.vit_backbone.embed_dim)

        vit_patch_hw = self.vit_backbone.patch_embed.patch_size
        vit_patch_size = (
            int(vit_patch_hw[0])
            if isinstance(vit_patch_hw, (tuple, list))
            else int(vit_patch_hw)
        )

        self.od_stem_adapter = BeerLambertStemAdapter(out_channels=s0.shape[1])
        self.mixstyle_s0 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)
        self.mixstyle_s1 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)

        self.norm_s1 = ChannelLayerNorm2d(c1)
        self.mil_s1 = MultiHeadFocalMILPool2d(
            c1, attn_dim=96, num_heads=4, dropout=0.15
        )
        self.proj_s1 = nn.Sequential(
            nn.Linear(c1 * 3, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )

        self.norm_s2 = ChannelLayerNorm2d(c2)
        self.mil_s2 = MultiHeadFocalMILPool2d(
            c2, attn_dim=128, num_heads=4, dropout=0.15
        )
        self.proj_s2 = nn.Sequential(
            nn.Linear(c2 * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.norm_s3 = ChannelLayerNorm2d(c3)
        self.attn_s3 = RadialBiasedD4SelfAttention2d(
            c3, num_heads=8, grid_size=8, attn_drop=0.10, proj_drop=0.15
        )
        self.mil_s3 = MultiHeadFocalMILPool2d(
            c3, attn_dim=192, num_heads=4, dropout=0.15
        )
        self.proj_s3 = nn.Sequential(
            nn.Linear(c3 * 4, 512),
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

        self.vit_stem_adapter = BeerLambertViTStemAdapter(
            embed_dim=c_vit, patch_size=vit_patch_size
        )
        self.vit_mixstyle_b2 = SpatialTokenMixStyle(
            p=vit_mixstyle_p, alpha=vit_mixstyle_alpha
        )
        self.vit_mixstyle_b5 = SpatialTokenMixStyle(
            p=vit_mixstyle_p, alpha=vit_mixstyle_alpha
        )

        self.vit_norm_b5 = ChannelLayerNorm2d(c_vit)
        self.vit_mil_b5 = MultiHeadFocalMILPool2d(
            c_vit, attn_dim=192, num_heads=4, dropout=0.15
        )
        self.vit_proj_b5 = nn.Sequential(
            nn.Linear(c_vit * 3, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )

        self.vit_norm_b8 = ChannelLayerNorm2d(c_vit)
        self.vit_mil_b8 = MultiHeadFocalMILPool2d(
            c_vit, attn_dim=192, num_heads=4, dropout=0.15
        )
        self.vit_proj_b8 = nn.Sequential(
            nn.Linear(c_vit * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.vit_norm_b11 = ChannelLayerNorm2d(c_vit)
        self.vit_mil_b11 = MultiHeadFocalMILPool2d(
            c_vit, attn_dim=192, num_heads=4, dropout=0.15
        )
        self.vit_proj_b11 = nn.Sequential(
            nn.Linear(c_vit * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        vit_concat_dim = 192 + 256 + 512
        self.vit_neck = nn.Sequential(
            nn.LayerNorm(vit_concat_dim),
            nn.Dropout(head_dropout * 0.8),
            nn.Linear(vit_concat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )

        self.morph_extractor = DifferentiableBeerLambertMorphExtractor(
            out_features=num_morph_features
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

        self.stream_gate_gen = nn.Linear(1024 + 128, 1024)
        nn.init.zeros_(self.stream_gate_gen.weight)
        nn.init.zeros_(self.stream_gate_gen.bias)

        self.film_generator = nn.Linear(128, 1024 * 2)
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        self.fusion_neck = nn.Sequential(
            nn.LayerNorm(1024 + 128),
            nn.Dropout(head_dropout * 0.7),
            nn.Linear(1024 + 128, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )

        self.fused_head = MultiSampleClassificationHead(
            512, hidden_dim=256, dropout=head_dropout, num_samples=5
        )
        self.dinov2_head = MultiSampleClassificationHead(
            512, hidden_dim=256, dropout=head_dropout, num_samples=5
        )
        self.convnext_head = MultiSampleClassificationHead(
            512, hidden_dim=256, dropout=head_dropout, num_samples=5
        )

    @staticmethod
    def _roi_box_avg_pool(feat_map: torch.Tensor, margin: float) -> torch.Tensor:
        H, W = feat_map.shape[2], feat_map.shape[3]
        h_margin = max(1, int(round(H * margin)))
        w_margin = max(1, int(round(W * margin)))
        if H - 2 * h_margin < 2:
            h_margin = max(0, (H - 2) // 2)
        if W - 2 * w_margin < 2:
            w_margin = max(0, (W - 2) // 2)
        center_crop = feat_map[:, :, h_margin : H - h_margin, w_margin : W - w_margin]
        return center_crop.mean(dim=(2, 3))

    @classmethod
    def _concentric_roi_avg_pool(cls, feat_map: torch.Tensor) -> torch.Tensor:
        roi_inner_64 = cls._roi_box_avg_pool(feat_map, margin=0.375)
        roi_outer_128 = cls._roi_box_avg_pool(feat_map, margin=0.25)
        return torch.cat([roi_inner_64, roi_outer_128], dim=1)

    def _forward_vit_stream(self, rgb_in: torch.Tensor) -> torch.Tensor:
        if rgb_in.shape[-2:] != (self.vit_img_size, self.vit_img_size):
            x_vit = F.interpolate(
                rgb_in,
                size=(self.vit_img_size, self.vit_img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_vit = rgb_in

        n_prefix = int(getattr(self.vit_backbone, "num_prefix_tokens", 1))
        x_tok = self.vit_backbone.patch_embed(x_vit)
        x_tok = self.vit_backbone._pos_embed(x_tok)
        x_tok = self.vit_backbone.patch_drop(x_tok)
        x_tok = self.vit_backbone.norm_pre(x_tok)

        stem_sp_delta = self.vit_stem_adapter(x_vit)
        x_tok = torch.cat(
            [x_tok[:, :n_prefix, :], x_tok[:, n_prefix:, :] + stem_sp_delta], dim=1
        )

        num_blocks = len(self.vit_backbone.blocks)
        idx_b2 = min(2, max(0, num_blocks - 4))
        idx_b5 = min(5, max(1, num_blocks - 3))
        idx_b8 = min(8, max(2, num_blocks - 2))

        feat_b5 = None
        feat_b8 = None
        for blk_idx, blk in enumerate(self.vit_backbone.blocks):
            x_tok = blk(x_tok)
            if blk_idx == idx_b2:
                x_tok = self.vit_mixstyle_b2(x_tok, n_prefix=n_prefix)
            if blk_idx == idx_b5:
                feat_b5 = x_tok
                x_tok = self.vit_mixstyle_b5(x_tok, n_prefix=n_prefix)
            if blk_idx == idx_b8:
                feat_b8 = x_tok
        x_final = self.vit_backbone.norm(x_tok)

        cls_tok = x_final[:, 0, :]
        sp_b5 = feat_b5[:, n_prefix:, :]
        sp_b8 = feat_b8[:, n_prefix:, :]
        sp_b11 = x_final[:, n_prefix:, :]

        B_v, N_sp, C_v = sp_b11.shape
        grid_hw = int(math.isqrt(N_sp))
        map_b5 = self.vit_norm_b5(
            sp_b5.transpose(1, 2).contiguous().view(B_v, C_v, grid_hw, grid_hw)
        )
        map_b8 = self.vit_norm_b8(
            sp_b8.transpose(1, 2).contiguous().view(B_v, C_v, grid_hw, grid_hw)
        )
        map_b11 = self.vit_norm_b11(
            sp_b11.transpose(1, 2).contiguous().view(B_v, C_v, grid_hw, grid_hw)
        )

        u_b5 = self.vit_proj_b5(
            torch.cat(
                [self.vit_mil_b5(map_b5), self._concentric_roi_avg_pool(map_b5)],
                dim=1,
            )
        )
        u_b8 = self.vit_proj_b8(
            torch.cat(
                [self.vit_mil_b8(map_b8), self._concentric_roi_avg_pool(map_b8)],
                dim=1,
            )
        )
        u_b11 = self.vit_proj_b11(
            torch.cat(
                [
                    cls_tok,
                    self.vit_mil_b11(map_b11),
                    self._concentric_roi_avg_pool(map_b11),
                ],
                dim=1,
            )
        )
        return self.vit_neck(torch.cat([u_b5, u_b8, u_b11], dim=1))

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
                [self.mil_s1(s1_norm), self._concentric_roi_avg_pool(s1_norm)],
                dim=1,
            )
        )

        s2_norm = self.norm_s2(f2)
        v2 = self.proj_s2(
            torch.cat(
                [self.mil_s2(s2_norm), self._concentric_roi_avg_pool(s2_norm)],
                dim=1,
            )
        )

        s3_norm = self.attn_s3(self.norm_s3(f3))
        v3 = self.proj_s3(
            torch.cat(
                [
                    self.mil_s3(s3_norm),
                    s3_norm.mean(dim=(2, 3)),
                    self._concentric_roi_avg_pool(s3_norm),
                ],
                dim=1,
            )
        )

        v_cnn = self.visual_neck(torch.cat([v1, v2, v3], dim=1))
        v_vit = self._forward_vit_stream(rgb_in)

        if (features is not None) and (features.size(-1) == self.num_morph_features):
            morph_raw = features.to(dtype=v_cnn.dtype)
        else:
            morph_raw = self.morph_extractor(rgb_in).to(dtype=v_cnn.dtype)

        m_embed = self.morph_encoder(self.morph_feat_drop(morph_raw))

        v_pair = torch.cat([v_cnn, v_vit], dim=1)
        stream_gates = 2.0 * torch.sigmoid(
            self.stream_gate_gen(torch.cat([v_pair, m_embed], dim=1))
        )
        film_params = torch.tanh(self.film_generator(m_embed))
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        v_mod = (v_pair * stream_gates) * (1.0 + 0.25 * gamma) + 0.25 * beta
        v_fused = self.fusion_neck(torch.cat([v_mod, m_embed], dim=1))

        logits_fused = self.fused_head(v_fused)
        logits_dinov2 = self.dinov2_head(v_vit)
        logits_convnext = self.convnext_head(v_cnn)
        logits = (
            0.50 * logits_fused + 0.25 * logits_dinov2 + 0.25 * logits_convnext
        )

        if return_aux:
            return logits, {
                "logits_fused": logits_fused,
                "logits_dinov2": logits_dinov2,
                "logits_convnext": logits_convnext,
                "v_fused": v_fused,
                "v_vit": v_vit,
                "v_cnn": v_cnn,
            }
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
                logits = self.forward(images, features, return_aux=False)
                probs = torch.sigmoid(logits.float())
            else:
                views = get_dihedral_tta_views(images)[:tta_views]
                prob_sum = 0.0
                for view_img in views:
                    view_logits = self.forward(view_img, features, return_aux=False)
                    prob_sum = prob_sum + torch.sigmoid(view_logits.float())
                probs = prob_sum / float(len(views))
            return torch.clamp(probs, 0.0, 1.0)
        finally:
            if was_training:
                self.train()


class BalancedHistopathologyLoss(nn.Module):
    """
    Co-Supervised domain-generalizing loss for Balanced Accuracy at fixed p=0.5 threshold:
      - Optimizes 0.50 * loss_fused + 0.25 * loss_dinov2 + 0.25 * loss_convnext
      - Independent stateful GroupDRO weights over (center_id, label) per stream
      - Per-Center Differentiable Soft Balanced Accuracy (SoftTPR + SoftTNR) per stream
      - Dimension-normalized fp32 Class-Conditional Deep CORAL on v_fused, v_vit, and v_cnn
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
        worst_recall_temp: float = 0.10,
        sym_weight: float = 0.30,
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
        self.worst_recall_temp = float(worst_recall_temp)
        self.sym_weight = float(sym_weight)

        init_w = torch.full(
            (self.num_groups,), 1.0 / float(self.num_groups), dtype=torch.float32
        )
        self.register_buffer("group_weights", init_w.clone())
        self.register_buffer("group_weights_dinov2", init_w.clone())
        self.register_buffer("group_weights_convnext", init_w.clone())

    def _compute_stream_loss(
        self,
        logits: torch.Tensor,
        targets_f: torch.Tensor,
        smoothed_targets: torch.Tensor,
        group_weights_buf: torch.Tensor,
        sample_weight: torch.Tensor = None,
        center_ids: torch.Tensor = None,
        v_embed: torch.Tensor = None,
    ) -> torch.Tensor:
        logits_f = logits.view(-1).float()

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
                        log_w = torch.log(group_weights_buf.clamp(min=1e-6))
                        log_w = log_w - log_w.mean()
                        log_w[idx_t] = (
                            1.0 - self.dro_decay
                        ) * log_w[idx_t] + self.dro_eta * centered
                        new_w = torch.softmax(log_w, dim=0)
                        for _ in range(6):
                            new_w = torch.clamp(new_w, min=0.12, max=0.52)
                            new_w = new_w / new_w.sum().clamp(min=1e-6)
                        group_weights_buf.copy_(new_w)

                active_w = group_weights_buf[idx_t]
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
                    err_pair_c = torch.stack([1.0 - soft_tpr_c, 1.0 - soft_tnr_c])
                    soft_ba_err_c = 0.5 * err_pair_c.sum()
                    worst_err_c = self.worst_recall_temp * (
                        torch.logsumexp(err_pair_c / self.worst_recall_temp, dim=0)
                        - math.log(2.0)
                    )
                    sym_err_c = F.smooth_l1_loss(
                        soft_tpr_c, soft_tnr_c, beta=0.02
                    )
                    center_ba_losses.append(
                        0.50 * soft_ba_err_c
                        + 0.50 * worst_err_c
                        + self.sym_weight * sym_err_c
                    )
                    center_ba_weights.append(
                        group_weights_buf[2 * c] + group_weights_buf[2 * c + 1]
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
            err_pair = torch.stack([1.0 - soft_tpr, 1.0 - soft_tnr])
            soft_ba_err = 0.5 * err_pair.sum()
            worst_err = self.worst_recall_temp * (
                torch.logsumexp(err_pair / self.worst_recall_temp, dim=0)
                - math.log(2.0)
            )
            sym_err = F.smooth_l1_loss(soft_tpr, soft_tnr, beta=0.02)
            soft_ba_loss = (
                0.50 * soft_ba_err + 0.50 * worst_err + self.sym_weight * sym_err
            )
            return (1.0 - self.ba_weight) * bce_loss + self.ba_weight * soft_ba_loss

        return bce_loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor = None,
        center_ids: torch.Tensor = None,
        v_embed=None,
        aux_outputs: dict = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            targets_f = targets.view(-1).float()
            smoothed_targets = (
                targets_f * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            )

            aux_dict = aux_outputs
            v_single = v_embed
            if isinstance(v_embed, dict):
                aux_dict = v_embed
                v_single = None

            if (aux_dict is not None) and ("logits_fused" in aux_dict):
                loss_fused = self._compute_stream_loss(
                    aux_dict["logits_fused"],
                    targets_f,
                    smoothed_targets,
                    self.group_weights,
                    sample_weight=sample_weight,
                    center_ids=center_ids,
                    v_embed=aux_dict.get("v_fused"),
                )
                loss_dinov2 = self._compute_stream_loss(
                    aux_dict["logits_dinov2"],
                    targets_f,
                    smoothed_targets,
                    self.group_weights_dinov2,
                    sample_weight=sample_weight,
                    center_ids=center_ids,
                    v_embed=aux_dict.get("v_vit"),
                )
                loss_convnext = self._compute_stream_loss(
                    aux_dict["logits_convnext"],
                    targets_f,
                    smoothed_targets,
                    self.group_weights_convnext,
                    sample_weight=sample_weight,
                    center_ids=center_ids,
                    v_embed=aux_dict.get("v_cnn"),
                )
                return (
                    0.50 * loss_fused + 0.25 * loss_dinov2 + 0.25 * loss_convnext
                )

            return self._compute_stream_loss(
                logits,
                targets_f,
                smoothed_targets,
                self.group_weights,
                sample_weight=sample_weight,
                center_ids=center_ids,
                v_embed=v_single,
            )


def build_optimizer(
    net: StainInvariantMultiScaleConvNeXt,
    lr_backbone: float = MODEL_CONFIG["lr_backbone"],
    lr_vit_backbone: float = MODEL_CONFIG["lr_vit_backbone"],
    lr_head: float = MODEL_CONFIG["lr_head"],
    layer_decay: float = MODEL_CONFIG["layer_decay"],
    weight_decay: float = MODEL_CONFIG["weight_decay"],
) -> AdamW:
    param_groups_dict = {}

    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue

        lname = name.lower()
        is_no_decay = (
            (param.ndim <= 1)
            or name.endswith(".bias")
            or ("norm" in lname)
            or ("gamma" in lname)
            or ("ls1" in lname)
            or ("ls2" in lname)
            or ("pos_embed" in name)
            or ("cls_token" in name)
            or ("reg_token" in name)
            or ("gate_alpha" in lname)
            or ("temp" in lname)
            or ("head_mix" in lname)
            or ("bias_table" in name)
        )
        wd = 0.0 if is_no_decay else weight_decay

        if name.startswith("backbone.stem.") or name.startswith("backbone.stages.0."):
            lr = lr_backbone * (layer_decay**3)
            group_tag = "backbone_cnn_s0"
        elif name.startswith("backbone.stages.1."):
            lr = lr_backbone * (layer_decay**2)
            group_tag = "backbone_cnn_s1"
        elif name.startswith("backbone.stages.2."):
            lr = lr_backbone * (layer_decay**1)
            group_tag = "backbone_cnn_s2"
        elif name.startswith("backbone.stages.3.") or name.startswith("backbone."):
            lr = lr_backbone
            group_tag = "backbone_cnn_s3"
        elif name.startswith("vit_backbone.blocks."):
            blk_idx = int(name.split(".")[2])
            if blk_idx <= 2:
                lr = lr_vit_backbone * (layer_decay**3)
                group_tag = "backbone_vit_g0"
            elif blk_idx <= 5:
                lr = lr_vit_backbone * (layer_decay**2)
                group_tag = "backbone_vit_g1"
            elif blk_idx <= 8:
                lr = lr_vit_backbone * (layer_decay**1)
                group_tag = "backbone_vit_g2"
            else:
                lr = lr_vit_backbone
                group_tag = "backbone_vit_g3"
        elif name.startswith("vit_backbone.norm.") or name.startswith(
            "vit_backbone.fc_norm."
        ):
            lr = lr_vit_backbone
            group_tag = "backbone_vit_g3"
        elif name.startswith("vit_backbone."):
            lr = lr_vit_backbone * (layer_decay**3)
            group_tag = "backbone_vit_g0"
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
    vit_backbone_candidates=MODEL_CONFIG["vit_backbone_candidates"],
    lr_backbone: float = MODEL_CONFIG["lr_backbone"],
    lr_vit_backbone: float = MODEL_CONFIG["lr_vit_backbone"],
    lr_head: float = MODEL_CONFIG["lr_head"],
):
    if pos_prior is None:
        pos_prior = float(np.mean(y_train))

    net = StainInvariantMultiScaleConvNeXt(
        backbone_candidates=backbone_candidates,
        vit_backbone_candidates=vit_backbone_candidates,
        vit_img_size=MODEL_CONFIG["vit_img_size"],
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
        lr_vit_backbone=lr_vit_backbone,
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

# Verify dual-stream forward pass, co-supervised loss, and official metric compatibility
model.eval()
with torch.no_grad():
    chk_items = [val_dataset[i] for i in range(8)]
    chk_imgs = torch.stack([it["image"] for it in chk_items], dim=0).to(DEVICE)
    chk_targets = torch.tensor(
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], device=DEVICE
    )
    chk_centers = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=DEVICE)
    chk_logits, chk_aux = model(chk_imgs, return_aux=True)
    chk_loss = criterion(
        chk_logits, chk_targets, center_ids=chk_centers, aux_outputs=chk_aux
    )
    chk_probs = model.predict_proba(chk_imgs, use_tta=True, tta_views=4)
    chk_metrics = binary_metrics(
        np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
        chk_probs.detach().cpu().numpy(),
    )

assert chk_logits.shape == (8,) and torch.isfinite(chk_logits).all()
assert torch.isfinite(chk_loss)
assert chk_probs.shape == (8,) and torch.isfinite(chk_probs).all()
assert "balanced_accuracy" in chk_metrics and "auroc" in chk_metrics

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Pretrained Backbone Identities: CNN={model.backbone_name}, ViT={model.vit_backbone_name} "
    f"(img_size={model.vit_img_size}, prefix_tokens={getattr(model.vit_backbone, 'num_prefix_tokens', 1)}) | "
    f"timm=={timm.__version__}, torch=={torch.__version__} | "
    f"Source: {MODEL_CONFIG['pretrained_source']} | "
    f"Known Training-Data Overlap: {MODEL_CONFIG['training_data_overlap']}"
)
print(
    f"Initialized Co-Supervised StainInvariantMultiScaleConvNeXt on {DEVICE} | "
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
    3100.0, max(900.0, 3350.0 - (STEP_START_TIME - SCRIPT_START_TIME))
)

if DEVICE.type == "cuda":
    vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
    TRAIN_BATCH_SIZE = 64 if vram_gb >= 22.0 else 32
    GRAD_ACCUM_STEPS = 64 // TRAIN_BATCH_SIZE
    EVAL_BATCH_SIZE = 128 if vram_gb >= 22.0 else 64
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

MAX_EPOCHS = 12
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
        features = (
            batch["features"].to(DEVICE, non_blocking=True)
            if "features" in batch
            else None
        )
        labels = batch["label"].to(DEVICE, non_blocking=True)
        center_idx = (
            batch["center_idx"].to(DEVICE, non_blocking=True)
            if "center_idx" in batch
            else None
        )

        with torch.amp.autocast(
            device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP
        ):
            logits, aux_outputs = model(images, features, return_aux=True)
            raw_loss = criterion(
                logits,
                labels,
                center_ids=center_idx,
                aux_outputs=aux_outputs,
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
    reserved_tta_sec = max(240.0, 2.0 * avg_ep_sec)
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
    c_y = y_val[idx]
    c_pred = val_preds_bin[idx]
    c_tpr = float(c_pred[c_y == 1].mean())
    c_tnr = float((c_pred[c_y == 0] == 0).mean())
    center_audit_parts.append(
        f"center_{cid}: ba={c_m['balanced_accuracy']:.4f}, tpr={c_tpr:.4f}, tnr={c_tnr:.4f}, auc={c_m['auroc']:.4f}"
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
