import copy
import gc
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
#    B) Center-held-out & 4-fold StratifiedGroupKFold strictly inside train_df
#    C) Merged (train + validation) dataset with StratifiedGroupKFold on group_id
#       preserved for post-validation artifacts
# ============================================================================
NUM_CV_FOLDS = 4
unique_train_centers = sorted(train_df["center_id"].unique().tolist())
center_to_idx = {cid: idx for idx, cid in enumerate(unique_train_centers)}
train_df["center_fold"] = train_df["center_id"].map(center_to_idx).astype(int)

# 4-fold StratifiedGroupKFold strictly within train_df (validation.csv 100% isolated)
sgkf_train = StratifiedGroupKFold(
    n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED
)
train_strat_key = (
    train_df["center_id"].astype(str) + "_" + train_df["label"].astype(str)
)
train_df["cv_fold"] = -1
for fold_idx, (tr_idx, va_idx) in enumerate(
    sgkf_train.split(train_df, y=train_strat_key, groups=train_df["group_id"])
):
    assert set(train_df.iloc[tr_idx]["group_id"]).isdisjoint(
        set(train_df.iloc[va_idx]["group_id"])
    ), f"Fold {fold_idx} group_id leakage!"
    assert set(train_df.iloc[tr_idx]["slide_id"]).isdisjoint(
        set(train_df.iloc[va_idx]["slide_id"])
    ), f"Fold {fold_idx} slide_id leakage!"
    assert set(train_df.iloc[va_idx]["slide_id"]).isdisjoint(
        set(val_df["slide_id"])
    ), f"Fold {fold_idx} overlaps validation.csv!"
    train_df.iloc[va_idx, train_df.columns.get_loc("cv_fold")] = fold_idx
assert (train_df["cv_fold"] >= 0).all(), "Unassigned cv_fold in train_df!"

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
sgkf_merged = StratifiedGroupKFold(
    n_splits=NUM_CV_FOLDS, shuffle=True, random_state=SEED
)
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
    f"with zero slide/group leakage across {NUM_CV_FOLDS} StratifiedGroupKFold splits."
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


def _get_available_ram_gib() -> float:
    """Queries OS available system memory in GiB without external dependencies."""
    try:
        with open("/proc/meminfo", "r") as f_mem:
            for line in f_mem:
                if line.startswith("MemAvailable:"):
                    kb = float(line.split()[1])
                    return kb / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return float(pages * page_size) / (1024.0**3)
    except Exception:
        pass
    return 0.0


GLOBAL_RGB_CACHE_ARRAY = None
PATH_TO_CACHE_IDX = {}
RAM_CACHE_ACTIVE = False

_cache_t0 = time.time()
_avail_ram_gib = _get_available_ram_gib()
_num_cache_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
_all_unique_paths = list(
    dict.fromkeys(
        train_df["full_image_path"].tolist()
        + val_df["full_image_path"].tolist()
        + test_df["full_image_path"].tolist()
    )
)
_est_cache_gib = (len(_all_unique_paths) * 256 * 256 * 3) / (1024.0**3)

if _avail_ram_gib >= 10.0:
    try:
        GLOBAL_RGB_CACHE_ARRAY = np.empty(
            (len(_all_unique_paths), 256, 256, 3), dtype=np.uint8
        )

        def _load_into_cache(idx_and_path):
            c_idx, c_path = idx_and_path
            GLOBAL_RGB_CACHE_ARRAY[c_idx] = read_rgb_image(c_path)

        with ThreadPoolExecutor(max_workers=_num_cache_workers) as _cache_exec:
            list(_cache_exec.map(_load_into_cache, enumerate(_all_unique_paths)))
        PATH_TO_CACHE_IDX = {p: i for i, p in enumerate(_all_unique_paths)}
        RAM_CACHE_ACTIVE = True
        print(
            f"Preloaded {len(_all_unique_paths)} uint8 RGB patches ({_est_cache_gib:.2f} GiB) into RAM "
            f"in {time.time() - _cache_t0:.1f}s (avail_ram_before={_avail_ram_gib:.1f} GiB, "
            f"avail_ram_after={_get_available_ram_gib():.1f} GiB)."
        )
    except MemoryError:
        GLOBAL_RGB_CACHE_ARRAY = None
        PATH_TO_CACHE_IDX = {}
        RAM_CACHE_ACTIVE = False
        gc.collect()
        print(
            "MemoryError during uint8 RAM caching; falling back cleanly to on-disk loading."
        )
else:
    print(
        f"Available RAM ({_avail_ram_gib:.1f} GiB) < 10.0 GiB threshold; using disk fallback."
    )


def get_rgb_image(path: str) -> np.ndarray:
    """Returns 256x256 uint8 RGB array from in-memory cache if active, else disk."""
    if RAM_CACHE_ACTIVE and (path in PATH_TO_CACHE_IDX):
        return GLOBAL_RGB_CACHE_ARRAY[PATH_TO_CACHE_IDX[path]]
    return read_rgb_image(path)


def _extract_foreground_stain_stats(path: str) -> np.ndarray:
    """Fast 2x-strided foreground-masked Beer-Lambert & CIE LAB stain calibration pass."""
    img_rgb = get_rgb_image(path)
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
# 6. Vectorized fp32 GPU Stain-Physics Augmentation & RAM-Cached Dataset
# ============================================================================
class GPUHistopathologyAugmentor(nn.Module):
    """
    Vectorized fp32 on-GPU domain randomization calibrated strictly on train.csv:
      1) Exact uniform random D4 dihedral symmetries (8 rotations/reflections)
      2) Per-channel scanner illumination / white-balance jitter in [0.93, 1.04]
      3) Zero-mean log-symmetric Beer-Lambert H&E scaling exp(Uniform(-ln 1.45, +ln 1.45))
         with foreground-weighted shifts + wider 3D stain-vector rotation (sigma=0.035)
      4) Zero-mean log-symmetric gamma jitter exp(Uniform(-ln 1.20, +ln 1.20))
      5) 3x3 optical blur / unsharp-mask MTF variation
    Deterministic identity pass-through when self.training is False.
    """

    def __init__(
        self,
        stain_stats: dict,
        p_wb: float = 0.75,
        p_stain: float = 0.85,
        p_gamma: float = 0.55,
        p_mtf: float = 0.35,
        he_max_scale: float = 1.45,
        stain_rot_sigma: float = 0.035,
        gamma_max_scale: float = 1.20,
    ):
        super().__init__()
        self.p_wb = float(p_wb)
        self.p_stain = float(p_stain)
        self.p_gamma = float(p_gamma)
        self.p_mtf = float(p_mtf)
        self.log_he_scale = math.log(float(he_max_scale))
        self.log_r_scale = math.log(1.20)
        self.stain_rot_sigma = float(stain_rot_sigma)
        self.log_gamma_scale = math.log(float(gamma_max_scale))

        ch_std = stain_stats.get("stain_channel_std", [0.22, 0.22, 0.22])
        h_shift = float(np.clip(0.30 * float(ch_std[0]), 0.05, 0.09))
        e_shift = float(np.clip(0.30 * float(ch_std[1]), 0.05, 0.09))
        r_shift = 0.025

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "stain_matrix",
            torch.tensor(STAIN_DECONVOLVER.stain_matrix, dtype=torch.float32),
        )
        self.register_buffer(
            "stain_inv",
            torch.tensor(STAIN_DECONVOLVER.stain_inv, dtype=torch.float32),
        )
        self.register_buffer(
            "shift_bounds",
            torch.tensor([h_shift, e_shift, r_shift], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.register_buffer("eye3", torch.eye(3, dtype=torch.float32))

        g3 = (
            torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            )
            / 16.0
        )
        self.register_buffer("gauss_3x3", g3.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (x.size(0) == 0):
            return x

        orig_dtype = x.dtype
        dev = x.device
        B = x.size(0)

        with torch.amp.autocast(device_type=dev.type, enabled=False):
            rgb = torch.clamp(
                x[:, :3].float() * self.imagenet_std + self.imagenet_mean,
                1.0 / 255.0,
                1.0,
            )

            # 1) Exact random D4 dihedral symmetries per sample
            g_idx = torch.randint(0, 8, (B,), device=dev)
            rgb_d4 = torch.empty_like(rgb)
            for g in range(8):
                m = g_idx == g
                if m.any():
                    sub = rgb[m]
                    k = g & 3
                    if k > 0:
                        sub = torch.rot90(sub, k=k, dims=(2, 3))
                    if g >= 4:
                        sub = torch.flip(sub, dims=(3,))
                    rgb_d4[m] = sub
            rgb = rgb_d4

            # 2) Per-channel white-balance / illumination jitter in [0.93, 1.04]
            wb_mask = torch.rand((B, 1, 1, 1), device=dev) < self.p_wb
            wb_gain = torch.empty((B, 3, 1, 1), device=dev).uniform_(0.93, 1.04)
            rgb = torch.where(
                wb_mask, torch.clamp(rgb * wb_gain, 1.0 / 255.0, 1.0), rgb
            )

            # 3) Log-symmetric Beer-Lambert H&E scaling + fg-weighted shifts + 3D stain rotation
            stain_mask = torch.rand((B, 1, 1, 1), device=dev) < self.p_stain
            od = -torch.log10(torch.clamp(rgb, 1.0 / 255.0, 1.0))
            her = torch.einsum("bchw,cd->bdhw", od, self.stain_inv)
            od_norm = torch.sqrt((od * od).sum(dim=1, keepdim=True) + 1e-8)
            w_fg = torch.sigmoid((od_norm - 0.10) / 0.04)

            log_he = torch.empty((B, 2, 1, 1), device=dev).uniform_(
                -self.log_he_scale, self.log_he_scale
            )
            log_r = torch.empty((B, 1, 1, 1), device=dev).uniform_(
                -self.log_r_scale, self.log_r_scale
            )
            scales = torch.exp(torch.cat([log_he, log_r], dim=1))
            shifts = (
                torch.rand((B, 3, 1, 1), device=dev) * 2.0 - 1.0
            ) * self.shift_bounds
            her_aug = her * scales + shifts * w_fg

            rot_std = self.stain_rot_sigma * 0.70710678
            omega = torch.randn((B, 3), device=dev) * rot_std
            theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp(min=1e-8)
            axis = omega / theta
            kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
            z0 = torch.zeros_like(kx)
            K = torch.stack(
                [
                    torch.stack([z0, -kz, ky], dim=-1),
                    torch.stack([kz, z0, -kx], dim=-1),
                    torch.stack([-ky, kx, z0], dim=-1),
                ],
                dim=1,
            )
            th = theta.unsqueeze(-1)
            R = (
                self.eye3.unsqueeze(0)
                + torch.sin(th) * K
                + (1.0 - torch.cos(th)) * torch.bmm(K, K)
            )
            mat_rot = torch.matmul(self.stain_matrix.unsqueeze(0), R.transpose(1, 2))
            mat_jit = mat_rot + torch.randn((B, 3, 3), device=dev) * rot_std
            mat_jit = torch.cat(
                [torch.clamp(mat_jit[:, :2, :], min=0.0), mat_jit[:, 2:, :]], dim=1
            )
            mat_jit = mat_jit / (
                torch.linalg.norm(mat_jit, dim=-1, keepdim=True) + 1e-6
            )

            od_rec = torch.clamp(
                torch.einsum("bdhw,bdc->bchw", her_aug, mat_jit), 0.0, 2.6
            )
            rgb_stain = torch.clamp(torch.pow(10.0, -od_rec), 1.0 / 255.0, 1.0)
            rgb = torch.where(stain_mask, rgb_stain, rgb)

            # 4) Zero-mean log-symmetric gamma jitter
            gamma_mask = torch.rand((B, 1, 1, 1), device=dev) < self.p_gamma
            gamma = torch.exp(
                torch.empty((B, 1, 1, 1), device=dev).uniform_(
                    -self.log_gamma_scale, self.log_gamma_scale
                )
            )
            rgb = torch.where(
                gamma_mask,
                torch.pow(torch.clamp(rgb, 1.0 / 255.0, 1.0), gamma),
                rgb,
            )

            # 5) 3x3 optical blur / sharpening MTF variation
            mtf_mask = torch.rand((B, 1, 1, 1), device=dev) < self.p_mtf
            blur = F.conv2d(
                F.pad(rgb, (1, 1, 1, 1), mode="reflect"), self.gauss_3x3, groups=3
            )
            is_blur = torch.rand((B, 1, 1, 1), device=dev) < 0.5
            alpha_blur = torch.empty((B, 1, 1, 1), device=dev).uniform_(0.20, 0.75)
            alpha_sharp = torch.empty((B, 1, 1, 1), device=dev).uniform_(-0.45, -0.15)
            alpha = torch.where(is_blur, alpha_blur, alpha_sharp)
            rgb_mtf = torch.clamp(rgb + alpha * (blur - rgb), 1.0 / 255.0, 1.0)
            rgb = torch.where(mtf_mask, rgb_mtf, rgb)

            out = (rgb - self.imagenet_mean) / self.imagenet_std
            if x.size(1) > 3:
                out = torch.cat([out, x[:, 3:].float()], dim=1)
            return out.to(orig_dtype)


class HistopathologyDataset(Dataset):
    """
    Consistent PyTorch Dataset for training, validation, and test inference.
    Reads preloaded uint8 RGB arrays from RAM when cached (with disk fallback),
    deferring physics stain augmentation to vectorized GPUHistopathologyAugmentor.
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
        augmentor=None,
    ):
        self.paths = df["full_image_path"].tolist()
        self.cache_indices = (
            np.array(
                [PATH_TO_CACHE_IDX.get(p, -1) for p in self.paths], dtype=np.int32
            )
            if RAM_CACHE_ACTIVE
            else None
        )
        self.has_labels = "label" in df.columns and df["label"].notna().all()
        self.labels = df["label"].values.astype(np.float32) if self.has_labels else None
        self.features = features.astype(np.float32) if features is not None else None
        self.is_train = is_train
        self.img_size = img_size
        self.return_stain_channels = return_stain_channels
        self.augmentor = augmentor if is_train else None
        self.stain_ch_mean = np.array(
            stain_stats["stain_channel_mean"], dtype=np.float32
        )
        self.stain_ch_std = np.array(stain_stats["stain_channel_std"], dtype=np.float32)
        self._mean_255 = torch.tensor(
            IMAGENET_MEAN * 255.0, dtype=torch.float32
        ).view(3, 1, 1)
        self._inv_std_255 = torch.tensor(
            1.0 / (IMAGENET_STD * 255.0), dtype=torch.float32
        ).view(3, 1, 1)

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
        if (self.cache_indices is not None) and (self.cache_indices[idx] >= 0):
            img_rgb = GLOBAL_RGB_CACHE_ARRAY[self.cache_indices[idx]]
        else:
            img_rgb = read_rgb_image(self.paths[idx])

        if self.is_train and self.augmentor is not None:
            img_rgb = self.augmentor(img_rgb.copy())

        if self.img_size != 256:
            interp = cv2.INTER_LINEAR if self.img_size > 256 else cv2.INTER_AREA
            img_rgb = cv2.resize(
                img_rgb, (self.img_size, self.img_size), interpolation=interp
            )

        if not self.return_stain_channels:
            img_tensor = (
                torch.from_numpy(img_rgb).permute(2, 0, 1).to(dtype=torch.float32)
            )
            img_tensor.sub_(self._mean_255).mul_(self._inv_std_255)
        else:
            rgb_norm = (
                img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN
            ) / IMAGENET_STD
            her, od = STAIN_DECONVOLVER.rgb_to_her(img_rgb)
            h_ch = np.maximum(her[:, :, 0], 0.0)
            e_ch = np.maximum(her[:, :, 1], 0.0)
            od_ch = np.linalg.norm(od, axis=2)
            stain_stack = np.stack([h_ch, e_ch, od_ch], axis=-1)
            stain_norm = (stain_stack - self.stain_ch_mean) / self.stain_ch_std
            full_arr = np.concatenate([rgb_norm, stain_norm], axis=-1)
            img_tensor = torch.from_numpy(np.transpose(full_arr, (2, 0, 1)).copy())

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
# 7. Model Design: Heterogeneous BatchNorm-Free SpecialistPathologyNet & Loss
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIG = {
    "specialist_priority": ("dinov2_vit", "swinv2", "convnext"),
    "dinov2_vit_candidates": (
        "vit_base_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
    ),
    "swinv2_candidates": (
        "swinv2_small_window16_256.ms_in1k",
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    ),
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
    "swinv2_img_size": 256,
    "convnext_img_size": 256,
    "pretrained_source": (
        "DINOv2-Reg4 ViT-B/14: Self-supervised LVD-142M curated natural photos; "
        "SwinV2-Small Window16: ImageNet-1K natural photos (fallback ImageNet-22K->1K); "
        "ConvNeXt-Small: ImageNet-22K pretrained + ImageNet-1K 384x384 fine-tuned natural photos"
    ),
    "training_data_overlap": (
        "None (0% overlap with H&E histopathology or lymph-node microscopy datasets across all three backbones)"
    ),
    "drop_path_rate": 0.25,
    "vit_drop_path_rate": 0.20,
    "swinv2_drop_path_rate": 0.25,
    "head_dropout": 0.30,
    "mixstyle_p": 0.50,
    "mixstyle_alpha": 0.20,
    "vit_mixstyle_p": 0.45,
    "vit_mixstyle_alpha": 0.20,
    "lr_vit_backbone": 4.0e-5,
    "lr_swinv2_backbone": 6.5e-5,
    "lr_convnext_backbone": 1.0e-4,
    "lr_backbone": 1.0e-4,
    "lr_head": 4.5e-4,
    "layer_decay": 0.80,
    "weight_decay": 2.0e-2,
    "label_smoothing": 0.025,
    "ba_loss_weight": 0.30,
    "ba_margin_temp": 0.70,
    "group_dro_eta": 0.04,
    "group_dro_decay": 0.02,
    "coral_loss_weight": 0.10,
    "sym_loss_weight": 0.35,
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


class StainChromaticityAdapter(nn.Module):
    """
    Pointwise fp32 Beer-Lambert stain chromaticity & log-concentration stem adapter:
      - Damped normalized OD chromaticity: OD / sqrt(||OD||_2^2 + 0.15^2) (3 channels)
      - Log-compressed H, E, and ||OD||_2: log1p(relu(H), relu(E), ||OD||_2) (3 channels)
    Cancels scalar stain intensity shifts without subtracting patch means (preserving
    confluent carcinoma sheets) or amplifying pale glass noise. 100% BatchNorm-free.
    """

    def __init__(
        self,
        out_channels: int,
        patch_size: int = 4,
        as_tokens: bool = False,
        init_gate: float = 0.15,
    ):
        super().__init__()
        self.as_tokens = bool(as_tokens)
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

        self.in_scale = nn.Parameter(torch.ones(1, 6, 1, 1))
        self.in_bias = nn.Parameter(torch.zeros(1, 6, 1, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=1, bias=True),
            ChannelLayerNorm2d(24),
            nn.GELU(),
            nn.Conv2d(
                24, out_channels, kernel_size=patch_size, stride=patch_size, bias=True
            ),
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
        if self.as_tokens:
            self.gate_alpha = nn.Parameter(torch.full((1, 1, out_channels), init_gate))
        else:
            self.gate_alpha = nn.Parameter(
                torch.full((1, out_channels, 1, 1), init_gate)
            )

    def forward(self, x_rgb_norm: torch.Tensor) -> torch.Tensor:
        orig_dtype = x_rgb_norm.dtype
        with torch.amp.autocast(device_type=x_rgb_norm.device.type, enabled=False):
            x_f = x_rgb_norm[:, :3, :, :].float()
            rgb_01 = torch.clamp(x_f * self.rgb_std + self.rgb_mean, 1.0 / 255.0, 1.0)
            od = -torch.log10(rgb_01)
            od_sq = (od * od).sum(dim=1, keepdim=True)
            od_norm = torch.sqrt(od_sq + 1e-8)
            chrom = od / torch.sqrt(od_sq + 0.15 * 0.15)

            her = F.conv2d(od, self.stain_inv_t)
            log_he_od = torch.log1p(
                torch.cat(
                    [
                        F.relu(her[:, 0:1, :, :]),
                        F.relu(her[:, 1:2, :, :]),
                        od_norm,
                    ],
                    dim=1,
                )
            )
            phys_in = torch.cat([chrom, log_he_od], dim=1) * self.in_scale + self.in_bias

        feat_map = self.proj(phys_in.to(orig_dtype))
        if self.as_tokens:
            tokens = feat_map.flatten(2).transpose(1, 2)
            return tokens * self.gate_alpha.to(orig_dtype)
        return feat_map * self.gate_alpha.to(orig_dtype)


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


class GatedAttentionMILPool2d(nn.Module):
    """
    Unpenalized D4-invariant 4-head Gated Attention MIL spatial pooling (Ilse et al., 2018).
    Operates directly on feature tokens with zero background OD penalty and zero patch-mean
    subtraction, preserving both pale vesicular nuclei and confluent carcinoma sheets.
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

        init_temps = torch.linspace(0.40, 1.00, steps=num_heads, dtype=torch.float32)
        self.attn_log_temp = nn.Parameter(torch.log(init_temps))
        self.head_mix_logits = nn.Parameter(torch.zeros(1, num_heads, in_channels))

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
                .clamp(min=0.15, max=4.0)
                .view(1, self.num_heads, 1)
            )
            weights = F.softmax(raw_scores.float() / attn_temp, dim=-1)
            head_pooled = torch.bmm(weights, tokens_f.transpose(1, 2))
            head_mix = F.softmax(self.head_mix_logits.float(), dim=1)
            mil_pooled = (head_pooled * head_mix).sum(dim=1)

        return mil_pooled.to(orig_dtype)


class MultiScaleStageReadout(nn.Module):
    """
    Fuses Global Average Pooling, exact D4-symmetric concentric central diagnostic ROI crops
    (37.5% and 62.5% center boxes), and unpenalized 4-head Gated Attention MIL.
    """

    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        attn_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.norm = ChannelLayerNorm2d(in_channels)
        self.mil = GatedAttentionMILPool2d(
            in_channels, attn_dim=attn_dim, num_heads=4, dropout=dropout
        )
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self._mask_cache = {}

    def _get_box_weights(
        self, H: int, W: int, box_frac: float, device: torch.device
    ) -> torch.Tensor:
        key = (H, W, round(box_frac, 5), device)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        lo_h, hi_h = H * (1.0 - box_frac) * 0.5, H * (1.0 + box_frac) * 0.5
        lo_w, hi_w = W * (1.0 - box_frac) * 0.5, W * (1.0 + box_frac) * 0.5
        idx_h = torch.arange(H, device=device, dtype=torch.float32)
        idx_w = torch.arange(W, device=device, dtype=torch.float32)
        wh = (torch.minimum(idx_h + 1.0, torch.tensor(hi_h, device=device)) - torch.maximum(idx_h, torch.tensor(lo_h, device=device))).clamp(min=0.0)
        ww = (torch.minimum(idx_w + 1.0, torch.tensor(hi_w, device=device)) - torch.maximum(idx_w, torch.tensor(lo_w, device=device))).clamp(min=0.0)
        w2d = wh[:, None] * ww[None, :]
        w2d = (w2d / w2d.sum().clamp(min=1e-6)).view(1, 1, H, W)
        self._mask_cache[key] = w2d
        return w2d

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.norm(feat_map)
        H, W = x.shape[2], x.shape[3]
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            gap = x_f.mean(dim=(2, 3))
            w_375 = self._get_box_weights(H, W, 0.375, x.device)
            w_625 = self._get_box_weights(H, W, 0.625, x.device)
            roi_375 = (x_f * w_375).sum(dim=(2, 3))
            roi_625 = (x_f * w_625).sum(dim=(2, 3))
        mil_vec = self.mil(x)
        cat_vec = torch.cat(
            [
                gap.to(orig_dtype),
                roi_375.to(orig_dtype),
                roi_625.to(orig_dtype),
                mil_vec,
            ],
            dim=1,
        )
        return self.proj(cat_vec)


class SpecialistPathologyNet(nn.Module):
    """
    Unified 100% BatchNorm-free Specialist Architecture supporting three decoupled families:
      1) 'dinov2_vit': vit_base_patch14_reg4_dinov2.lvd142m (img_size=252, blocks 2/5/8/11 + CLS)
      2) 'swinv2':     swinv2_small_window16_256.ms_in1k (img_size=256, scaled-cosine 16x16 attention)
      3) 'convnext':   convnext_small.fb_in22k_ft_in1k_384 (img_size=256, 7x7 depthwise CNN)
    Equipped with StainChromaticityAdapter, early-stage training-only MixStyle2d/SpatialTokenMixStyle,
    multi-scale GAP + concentric ROI (37.5% & 62.5%) + 4-head Gated Attention MIL readout, and a
    single clean linear logit head with logit-space D4-TTA inference.
    """

    def __init__(
        self,
        family: str = "dinov2_vit",
        backbone_candidates=None,
        pretrained: bool = True,
        head_dropout: float = MODEL_CONFIG["head_dropout"],
        mixstyle_p: float = MODEL_CONFIG["mixstyle_p"],
        mixstyle_alpha: float = MODEL_CONFIG["mixstyle_alpha"],
        vit_mixstyle_p: float = MODEL_CONFIG["vit_mixstyle_p"],
        vit_mixstyle_alpha: float = MODEL_CONFIG["vit_mixstyle_alpha"],
    ):
        super().__init__()
        self.family = str(family).lower()
        if self.family not in ("dinov2_vit", "swinv2", "convnext"):
            raise ValueError(f"Unsupported specialist family: {self.family}")

        if self.family == "dinov2_vit":
            cands = backbone_candidates or MODEL_CONFIG["dinov2_vit_candidates"]
            self.img_size = int(MODEL_CONFIG["vit_img_size"])
            dp_rate = float(MODEL_CONFIG["vit_drop_path_rate"])
        elif self.family == "swinv2":
            cands = backbone_candidates or MODEL_CONFIG["swinv2_candidates"]
            self.img_size = int(MODEL_CONFIG["swinv2_img_size"])
            dp_rate = float(MODEL_CONFIG["swinv2_drop_path_rate"])
        else:
            cands = backbone_candidates or MODEL_CONFIG["backbone_candidates"]
            self.img_size = int(MODEL_CONFIG["convnext_img_size"])
            dp_rate = float(MODEL_CONFIG["drop_path_rate"])

        if isinstance(cands, str):
            cands = (cands,)

        self.backbone = None
        self.backbone_name = None
        last_err = None
        for cand_name in cands:
            try:
                if self.family in ("dinov2_vit", "swinv2"):
                    try:
                        self.backbone = timm.create_model(
                            cand_name,
                            pretrained=pretrained,
                            num_classes=0,
                            img_size=self.img_size,
                            drop_path_rate=dp_rate,
                        )
                    except TypeError:
                        self.backbone = timm.create_model(
                            cand_name,
                            pretrained=pretrained,
                            num_classes=0,
                            drop_path_rate=dp_rate,
                        )
                else:
                    self.backbone = timm.create_model(
                        cand_name,
                        pretrained=pretrained,
                        num_classes=0,
                        drop_path_rate=dp_rate,
                    )
                self.backbone_name = cand_name
                break
            except Exception as exc:
                last_err = exc

        if self.backbone is None:
            raise RuntimeError(
                f"Failed to instantiate {self.family} backbone: {last_err}"
            ) from last_err

        if self.family == "dinov2_vit":
            c_vit = int(self.backbone.embed_dim)
            p_hw = self.backbone.patch_embed.patch_size
            p_size = int(p_hw[0]) if isinstance(p_hw, (tuple, list)) else int(p_hw)

            self.chrom_adapter = StainChromaticityAdapter(
                out_channels=c_vit, patch_size=p_size, as_tokens=True, init_gate=0.10
            )
            self.tok_mixstyle_b2 = SpatialTokenMixStyle(
                p=vit_mixstyle_p, alpha=vit_mixstyle_alpha
            )
            self.tok_mixstyle_b5 = SpatialTokenMixStyle(
                p=vit_mixstyle_p, alpha=vit_mixstyle_alpha
            )

            self.readout_b2 = MultiScaleStageReadout(c_vit, 128, attn_dim=128)
            self.readout_b5 = MultiScaleStageReadout(c_vit, 192, attn_dim=160)
            self.readout_b8 = MultiScaleStageReadout(c_vit, 256, attn_dim=192)
            self.readout_b11 = MultiScaleStageReadout(c_vit, 320, attn_dim=192)
            self.cls_proj = nn.Sequential(
                nn.LayerNorm(c_vit),
                nn.Linear(c_vit, 192),
                nn.LayerNorm(192),
                nn.GELU(),
            )
            concat_dim = 128 + 192 + 256 + 320 + 192

        elif self.family == "swinv2":
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.img_size, self.img_size)
                s0 = self.backbone.patch_embed(dummy)
                s0_nchw, _ = self._to_nchw_with_meta(s0)
                c0 = int(s0_nchw.shape[1])
                s0_stride = max(1, self.img_size // int(s0_nchw.shape[2]))
                z = self.backbone.layers[0](s0)
                z = self.backbone.layers[1](z)
                c1 = int(self._to_nchw_with_meta(z)[0].shape[1])
                z = self.backbone.layers[2](z)
                c2 = int(self._to_nchw_with_meta(z)[0].shape[1])
                z = self.backbone.layers[3](z)
                c3 = int(self._to_nchw_with_meta(z)[0].shape[1])

            self.chrom_adapter = StainChromaticityAdapter(
                out_channels=c0, patch_size=s0_stride, as_tokens=False, init_gate=0.15
            )
            self.mixstyle_s0 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)
            self.mixstyle_s1 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)

            self.readout_s1 = MultiScaleStageReadout(c1, 192, attn_dim=96)
            self.readout_s2 = MultiScaleStageReadout(c2, 256, attn_dim=128)
            self.readout_s3 = MultiScaleStageReadout(c3, 384, attn_dim=192)
            concat_dim = 192 + 256 + 384

        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.img_size, self.img_size)
                s0 = self.backbone.stem(dummy)
                c0 = int(s0.shape[1])
                s0_stride = max(1, self.img_size // int(s0.shape[2]))
                f0 = self.backbone.stages[0](s0)
                f1 = self.backbone.stages[1](f0)
                f2 = self.backbone.stages[2](f1)
                f3 = self.backbone.stages[3](f2)
                c1, c2, c3 = int(f1.shape[1]), int(f2.shape[1]), int(f3.shape[1])

            self.chrom_adapter = StainChromaticityAdapter(
                out_channels=c0, patch_size=s0_stride, as_tokens=False, init_gate=0.15
            )
            self.mixstyle_s0 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)
            self.mixstyle_s1 = MixStyle2d(p=mixstyle_p, alpha=mixstyle_alpha)

            self.readout_s1 = MultiScaleStageReadout(c1, 192, attn_dim=96)
            self.readout_s2 = MultiScaleStageReadout(c2, 256, attn_dim=128)
            self.readout_s3 = MultiScaleStageReadout(c3, 384, attn_dim=192)
            concat_dim = 192 + 256 + 384

        self.neck = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Dropout(head_dropout * 0.8),
            nn.Linear(concat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )
        self.head_drop = nn.Dropout(head_dropout)
        self.classifier = nn.Linear(512, 1)

        assert not any(
            isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.modules()
        ), f"BatchNorm layer detected in {self.family}!"

    @staticmethod
    def _to_nchw_with_meta(x: torch.Tensor):
        if x.ndim == 3:
            B, L, C = x.shape
            H = int(math.isqrt(L))
            return x.transpose(1, 2).contiguous().view(B, C, H, H), ("NLC",)
        if x.ndim == 4 and x.shape[1] == x.shape[2] and x.shape[3] != x.shape[1]:
            return x.permute(0, 3, 1, 2).contiguous(), ("NHWC",)
        return x, ("NCHW",)

    @staticmethod
    def _from_nchw_with_meta(x_nchw: torch.Tensor, meta: tuple) -> torch.Tensor:
        if meta[0] == "NLC":
            return x_nchw.flatten(2).transpose(1, 2).contiguous()
        if meta[0] == "NHWC":
            return x_nchw.permute(0, 2, 3, 1).contiguous()
        return x_nchw

    def _forward_dinov2(self, rgb_in: torch.Tensor) -> torch.Tensor:
        if rgb_in.shape[-2:] != (self.img_size, self.img_size):
            x_in = F.interpolate(
                rgb_in,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_in = rgb_in

        n_prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        x_tok = self.backbone.patch_embed(x_in)
        x_tok = self.backbone._pos_embed(x_tok)
        x_tok = self.backbone.patch_drop(x_tok)
        x_tok = self.backbone.norm_pre(x_tok)

        sp_delta = self.chrom_adapter(x_in)
        x_tok = torch.cat(
            [x_tok[:, :n_prefix, :], x_tok[:, n_prefix:, :] + sp_delta], dim=1
        )

        num_blocks = len(self.backbone.blocks)
        idx_b2 = min(2, max(0, num_blocks - 4))
        idx_b5 = min(5, max(1, num_blocks - 3))
        idx_b8 = min(8, max(2, num_blocks - 2))

        feat_b2 = None
        feat_b5 = None
        feat_b8 = None
        for blk_idx, blk in enumerate(self.backbone.blocks):
            x_tok = blk(x_tok)
            if blk_idx == idx_b2:
                feat_b2 = x_tok
                x_tok = self.tok_mixstyle_b2(x_tok, n_prefix=n_prefix)
            if blk_idx == idx_b5:
                feat_b5 = x_tok
                x_tok = self.tok_mixstyle_b5(x_tok, n_prefix=n_prefix)
            if blk_idx == idx_b8:
                feat_b8 = x_tok

        x_final = self.backbone.norm(x_tok)
        cls_tok = x_final[:, 0, :]

        B_v, N_sp, C_v = x_final[:, n_prefix:, :].shape
        gh = int(math.isqrt(N_sp))

        map_b2 = feat_b2[:, n_prefix:, :].transpose(1, 2).contiguous().view(B_v, C_v, gh, gh)
        map_b5 = feat_b5[:, n_prefix:, :].transpose(1, 2).contiguous().view(B_v, C_v, gh, gh)
        map_b8 = feat_b8[:, n_prefix:, :].transpose(1, 2).contiguous().view(B_v, C_v, gh, gh)
        map_b11 = x_final[:, n_prefix:, :].transpose(1, 2).contiguous().view(B_v, C_v, gh, gh)

        v_cat = torch.cat(
            [
                self.readout_b2(map_b2),
                self.readout_b5(map_b5),
                self.readout_b8(map_b8),
                self.readout_b11(map_b11),
                self.cls_proj(cls_tok),
            ],
            dim=1,
        )
        return self.neck(v_cat)

    def _forward_swinv2(self, rgb_in: torch.Tensor) -> torch.Tensor:
        if rgb_in.shape[-2:] != (self.img_size, self.img_size):
            x_in = F.interpolate(
                rgb_in,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_in = rgb_in

        x = self.backbone.patch_embed(x_in)
        if getattr(self.backbone, "absolute_pos_embed", None) is not None:
            x = x + self.backbone.absolute_pos_embed
        if hasattr(self.backbone, "pos_drop"):
            x = self.backbone.pos_drop(x)

        x_nchw, meta0 = self._to_nchw_with_meta(x)
        x_nchw = x_nchw + self.chrom_adapter(x_in)
        x = self._from_nchw_with_meta(x_nchw, meta0)

        x = self.backbone.layers[0](x)
        x0_nchw, meta_l0 = self._to_nchw_with_meta(x)
        x0_nchw = self.mixstyle_s0(x0_nchw)
        x = self._from_nchw_with_meta(x0_nchw, meta_l0)

        x = self.backbone.layers[1](x)
        f1_nchw, meta_l1 = self._to_nchw_with_meta(x)
        f1_styled = self.mixstyle_s1(f1_nchw)
        x = self._from_nchw_with_meta(f1_styled, meta_l1)

        x = self.backbone.layers[2](x)
        f2_nchw, _ = self._to_nchw_with_meta(x)

        x = self.backbone.layers[3](x)
        if getattr(self.backbone, "norm", None) is not None:
            x = self.backbone.norm(x)
        f3_nchw, _ = self._to_nchw_with_meta(x)

        v_cat = torch.cat(
            [
                self.readout_s1(f1_nchw),
                self.readout_s2(f2_nchw),
                self.readout_s3(f3_nchw),
            ],
            dim=1,
        )
        return self.neck(v_cat)

    def _forward_convnext(self, rgb_in: torch.Tensor) -> torch.Tensor:
        if rgb_in.shape[-2:] != (self.img_size, self.img_size):
            x_in = F.interpolate(
                rgb_in,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_in = rgb_in

        x = self.backbone.stem(x_in) + self.chrom_adapter(x_in)
        x = self.backbone.stages[0](x)
        x = self.mixstyle_s0(x)

        f1 = self.backbone.stages[1](x)
        f1_styled = self.mixstyle_s1(f1)
        f2 = self.backbone.stages[2](f1_styled)
        f3 = self.backbone.stages[3](f2)

        v_cat = torch.cat(
            [
                self.readout_s1(f1),
                self.readout_s2(f2),
                self.readout_s3(f3),
            ],
            dim=1,
        )
        return self.neck(v_cat)

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        return_aux: bool = False,
    ):
        rgb_in = images[:, :3, :, :] if images.size(1) > 3 else images
        if self.family == "dinov2_vit":
            v_embed = self._forward_dinov2(rgb_in)
        elif self.family == "swinv2":
            v_embed = self._forward_swinv2(rgb_in)
        else:
            v_embed = self._forward_convnext(rgb_in)

        logits = self.classifier(self.head_drop(v_embed)).squeeze(-1).float()
        if return_aux:
            return logits, {"v_embed": v_embed}
        return logits

    @torch.no_grad()
    def forward_logits(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        use_tta: bool = False,
        tta_views: int = 8,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            if (not use_tta) or (tta_views <= 1):
                return self.forward(images, features, return_aux=False).float()
            views = get_dihedral_tta_views(images)[:tta_views]
            logit_sum = 0.0
            for view_img in views:
                logit_sum = logit_sum + self.forward(
                    view_img, features, return_aux=False
                ).float()
            return logit_sum / float(len(views))
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def predict_proba(
        self,
        images: torch.Tensor,
        features: torch.Tensor = None,
        use_tta: bool = False,
        tta_views: int = 8,
    ) -> torch.Tensor:
        logits = self.forward_logits(
            images, features=features, use_tta=use_tta, tta_views=tta_views
        )
        return torch.clamp(torch.sigmoid(logits), 0.0, 1.0)


StainInvariantMultiScaleConvNeXt = SpecialistPathologyNet


class CenterInvariantBalancedLoss(nn.Module):
    """
    Domain-generalizing loss aligned with Balanced Accuracy at fixed logit=0 (p=0.5):
      - Stateful GroupDRO label-smoothed BCE over (center_id, label) cells
      - Per-center soft Balanced Accuracy + 0.35 * SmoothL1(soft_TPR_c, soft_TNR_c) symmetry penalty
      - Dimension-normalized fp32 Class-Conditional Deep CORAL across medical centers
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
        sym_weight: float = MODEL_CONFIG["sym_loss_weight"],
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
            if (v_single is None) and isinstance(aux_dict, dict):
                v_single = aux_dict.get("v_embed")

            return self._compute_stream_loss(
                logits,
                targets_f,
                smoothed_targets,
                self.group_weights,
                sample_weight=sample_weight,
                center_ids=center_ids,
                v_embed=v_single,
            )


BalancedHistopathologyLoss = CenterInvariantBalancedLoss


def build_optimizer(
    net: SpecialistPathologyNet,
    lr_backbone: float = None,
    lr_vit_backbone: float = MODEL_CONFIG["lr_vit_backbone"],
    lr_swinv2_backbone: float = MODEL_CONFIG["lr_swinv2_backbone"],
    lr_convnext_backbone: float = MODEL_CONFIG["lr_convnext_backbone"],
    lr_head: float = MODEL_CONFIG["lr_head"],
    layer_decay: float = MODEL_CONFIG["layer_decay"],
    weight_decay: float = MODEL_CONFIG["weight_decay"],
) -> AdamW:
    family = getattr(net, "family", "convnext")
    if lr_backbone is not None:
        base_backbone_lr = float(lr_backbone)
    elif family == "dinov2_vit":
        base_backbone_lr = float(lr_vit_backbone)
    elif family == "swinv2":
        base_backbone_lr = float(lr_swinv2_backbone)
    else:
        base_backbone_lr = float(lr_convnext_backbone)

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
            or ("pos_embed" in lname)
            or ("cls_token" in lname)
            or ("reg_token" in lname)
            or ("token" in lname)
            or ("gate" in lname)
            or ("in_scale" in lname)
            or ("in_bias" in lname)
            or ("temp" in lname)
            or ("head_mix" in lname)
            or ("mix_logits" in lname)
            or ("logit_scale" in lname)
            or ("cpb_mlp" in lname)
            or ("bias_table" in lname)
        )
        wd = 0.0 if is_no_decay else weight_decay

        if family == "dinov2_vit":
            if name.startswith("backbone.blocks."):
                blk_idx = int(name.split(".")[2])
                if blk_idx <= 2:
                    lr = base_backbone_lr * (layer_decay**3)
                    group_tag = "backbone_vit_g0"
                elif blk_idx <= 5:
                    lr = base_backbone_lr * (layer_decay**2)
                    group_tag = "backbone_vit_g1"
                elif blk_idx <= 8:
                    lr = base_backbone_lr * (layer_decay**1)
                    group_tag = "backbone_vit_g2"
                else:
                    lr = base_backbone_lr
                    group_tag = "backbone_vit_g3"
            elif name.startswith("backbone.norm.") or name.startswith(
                "backbone.fc_norm."
            ):
                lr = base_backbone_lr
                group_tag = "backbone_vit_g3"
            elif name.startswith("backbone."):
                lr = base_backbone_lr * (layer_decay**3)
                group_tag = "backbone_vit_g0"
            else:
                lr = lr_head
                group_tag = "custom_head_and_adapters"
        elif family == "swinv2":
            if (
                name.startswith("backbone.patch_embed.")
                or name.startswith("backbone.layers.0.")
                or name == "backbone.absolute_pos_embed"
            ):
                lr = base_backbone_lr * (layer_decay**3)
                group_tag = "backbone_swin_s0"
            elif name.startswith("backbone.layers.1."):
                lr = base_backbone_lr * (layer_decay**2)
                group_tag = "backbone_swin_s1"
            elif name.startswith("backbone.layers.2."):
                lr = base_backbone_lr * (layer_decay**1)
                group_tag = "backbone_swin_s2"
            elif name.startswith("backbone.layers.3.") or name.startswith("backbone."):
                lr = base_backbone_lr
                group_tag = "backbone_swin_s3"
            else:
                lr = lr_head
                group_tag = "custom_head_and_adapters"
        else:
            if name.startswith("backbone.stem.") or name.startswith(
                "backbone.stages.0."
            ):
                lr = base_backbone_lr * (layer_decay**3)
                group_tag = "backbone_cnn_s0"
            elif name.startswith("backbone.stages.1."):
                lr = base_backbone_lr * (layer_decay**2)
                group_tag = "backbone_cnn_s1"
            elif name.startswith("backbone.stages.2."):
                lr = base_backbone_lr * (layer_decay**1)
                group_tag = "backbone_cnn_s2"
            elif name.startswith("backbone.stages.3.") or name.startswith("backbone."):
                lr = base_backbone_lr
                group_tag = "backbone_cnn_s3"
            else:
                lr = lr_head
                group_tag = "custom_head_and_adapters"

        key = (group_tag, round(lr, 12), wd)
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
    family: str = "dinov2_vit",
    pos_prior: float = None,
    pretrained: bool = True,
    device: torch.device = DEVICE,
    backbone_candidates=None,
    lr_backbone: float = None,
    lr_head: float = MODEL_CONFIG["lr_head"],
):
    if pos_prior is None:
        pos_prior = float(np.mean(y_train))

    net = SpecialistPathologyNet(
        family=family,
        backbone_candidates=backbone_candidates,
        pretrained=pretrained,
    ).to(device)

    loss_fn = CenterInvariantBalancedLoss(
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
    family="dinov2_vit",
    pos_prior=train_pos_prior,
    pretrained=True,
    device=DEVICE,
)

# Verify BatchNorm-free specialist forward pass, CenterInvariantBalancedLoss, and metric.py compatibility
model.eval()
with torch.no_grad():
    chk_items = [train_dataset[i] for i in range(8)]
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
assert "v_embed" in chk_aux and chk_aux["v_embed"].shape == (8, 512)
assert torch.isfinite(chk_loss)
assert chk_probs.shape == (8,) and torch.isfinite(chk_probs).all()
assert "balanced_accuracy" in chk_metrics and "auroc" in chk_metrics

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Specialist Backbone Suite: "
    f"(1) DINOv2={MODEL_CONFIG['dinov2_vit_candidates'][0]} [active={model.backbone_name}, img_size={model.img_size}], "
    f"(2) SwinV2={MODEL_CONFIG['swinv2_candidates'][0]} [img_size={MODEL_CONFIG['swinv2_img_size']}], "
    f"(3) ConvNeXt={MODEL_CONFIG['backbone_candidates'][0]} [img_size={MODEL_CONFIG['convnext_img_size']}] | "
    f"timm=={timm.__version__}, torch=={torch.__version__} | "
    f"Source: {MODEL_CONFIG['pretrained_source']} | "
    f"Known Training-Data Overlap: {MODEL_CONFIG['training_data_overlap']}"
)
print(
    f"Initialized primary SpecialistPathologyNet ({model.family}) on {DEVICE} | "
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
TOTAL_SCRIPT_BUDGET_SEC = 3360.0
MAX_STEP_SECONDS = max(900.0, TOTAL_SCRIPT_BUDGET_SEC - (STEP_START_TIME - SCRIPT_START_TIME))

if DEVICE.type == "cuda":
    vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
    TRAIN_BATCH_SIZE = 64 if vram_gb >= 22.0 else 32
    GRAD_ACCUM_STEPS = 64 // TRAIN_BATCH_SIZE
    EVAL_BATCH_SIZE = 128 if vram_gb >= 22.0 else 64
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    USE_GRAD_SCALER = AMP_DTYPE == torch.float16
    CKPT_SAVE_DTYPE = AMP_DTYPE
else:
    TRAIN_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    EVAL_BATCH_SIZE = 32
    USE_AMP = False
    AMP_DTYPE = torch.float32
    USE_GRAD_SCALER = False
    CKPT_SAVE_DTYPE = torch.float16

FOLD_EPOCHS = 9
MAX_EPOCHS = FOLD_EPOCHS
MAX_GRAD_NORM = 1.0
EMA_MAX_DECAY = 0.994
MIN_LR_RATIO = 0.02
FOLD_VAL_TTA_VIEWS = 2
FINAL_TTA_VIEWS = 8
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pt")
CKPT_DIR = os.path.join(WORKING_DIR, "specialist_ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)


def _dataloader_worker_init_fn(worker_id: int) -> None:
    """Ensures distinct RNG streams and clean single-threaded OpenCV inside forked workers."""
    cv2.setNumThreads(0)
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class StratifiedCenterClassBatchSampler(Sampler):
    """
    Deterministic without-replacement batch sampler yielding exact balanced mini-batches
    containing batch_size // 4 samples from each (center_id, label) cell in df.
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

# Zero /dev/shm IPC overhead when uint8 patches are cached in RAM; multi-worker on disk fallback
num_dl_workers = 0 if RAM_CACHE_ACTIVE else min(8, max(2, (os.cpu_count() or 4)))
pin_mem = DEVICE.type == "cuda"
dl_worker_kwargs = (
    {"persistent_workers": True, "worker_init_fn": _dataloader_worker_init_fn}
    if num_dl_workers > 0
    else {}
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_batch_sampler,
    num_workers=num_dl_workers,
    pin_memory=pin_mem,
    **dl_worker_kwargs,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=num_dl_workers,
    pin_memory=pin_mem,
    drop_last=False,
    **dl_worker_kwargs,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=num_dl_workers,
    pin_memory=pin_mem,
    drop_last=False,
    **dl_worker_kwargs,
)


def set_step_learning_rates(
    opt: torch.optim.Optimizer,
    step_idx: int,
    total_steps: int,
    steps_in_ep: int,
) -> float:
    head_w_steps = max(1, int(round(0.6 * steps_in_ep)))
    backbone_w_steps = max(1, int(round(1.2 * steps_in_ep)))
    head_lr_now = 0.0
    for pg in opt.param_groups:
        base_lr = pg["initial_lr"]
        gname = pg.get("group_name", "")
        is_backbone = gname.startswith("backbone_")
        w_steps = backbone_w_steps if is_backbone else head_w_steps
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


@torch.no_grad()
def update_ema_weights(ema_net: nn.Module, src_net: nn.Module, opt_step: int) -> None:
    decay = min(EMA_MAX_DECAY, float(1 + opt_step) / float(6 + opt_step))
    for p_ema, p_src in zip(ema_net.parameters(), src_net.parameters()):
        p_ema.mul_(decay).add_(p_src.detach(), alpha=1.0 - decay)
    for b_ema, b_src in zip(ema_net.buffers(), src_net.buffers()):
        b_ema.copy_(b_src)


def _compact_state_dict_cpu(sd: dict, dtype: torch.dtype = CKPT_SAVE_DTYPE) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_floating_point(v):
            out[k] = v.detach().to(dtype=dtype).cpu().clone()
        else:
            out[k] = v.detach().cpu().clone()
    return out


def _load_compact_state_dict(net: nn.Module, compact_sd: dict) -> None:
    ref_sd = net.state_dict()
    cast_sd = {}
    for k, v in compact_sd.items():
        if (k in ref_sd) and torch.is_floating_point(v):
            cast_sd[k] = v.to(dtype=ref_sd[k].dtype)
        else:
            cast_sd[k] = v
    net.load_state_dict(cast_sd, strict=True)


@torch.inference_mode()
def extract_model_logits(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
    use_amp: bool = USE_AMP,
    amp_dtype: torch.dtype = AMP_DTYPE,
    use_tta: bool = False,
    tta_views: int = FINAL_TTA_VIEWS,
) -> np.ndarray:
    """Extracts exact logit-space (optionally D4-TTA averaged) predictions in float64."""
    net.eval()
    logit_chunks = []
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
            batch_logits = net.forward_logits(
                images=images,
                features=features,
                use_tta=use_tta,
                tta_views=tta_views,
            )
        logit_chunks.append(batch_logits.detach().double().cpu().numpy())
    logits_all = np.concatenate(logit_chunks, axis=0).astype(np.float64)
    return np.nan_to_num(logits_all, nan=0.0, posinf=30.0, neginf=-30.0)


def _logits_to_probs(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    probs = torch.sigmoid(torch.from_numpy(clipped)).numpy()
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs, 0.0, 1.0)


class CalibratedSpecialistEnsemble(nn.Module):
    """
    Leak-free calibrated multi-family 4-fold specialist ensemble operating strictly in logit space:
      z_cal(x) = (sum_m w_m * (mean_k z_{m,k}^{D4}(x) / sigma_m) + b*) / T*
      p(x) = sigmoid(z_cal(x))
    Executes checkpoint-sequentially over DataLoader batches to maintain <1.5 GB peak VRAM.
    """

    def __init__(
        self,
        families: list,
        ckpt_paths_by_family: dict,
        family_sigmas: dict,
        family_weights: dict,
        logit_bias: float = 0.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.families = list(families)
        self.ckpt_paths_by_family = {
            f: list(ckpt_paths_by_family[f]) for f in self.families
        }
        self.family_sigmas = {f: float(family_sigmas[f]) for f in self.families}
        self.family_weights = {f: float(family_weights[f]) for f in self.families}
        self.logit_bias = float(logit_bias)
        self.temperature = max(float(temperature), 1e-4)
        self.last_family_raw_logits = {}

    @torch.inference_mode()
    def predict_loader_logits(
        self,
        loader: DataLoader,
        device: torch.device = DEVICE,
        use_amp: bool = USE_AMP,
        amp_dtype: torch.dtype = AMP_DTYPE,
        use_tta: bool = True,
        tta_views: int = FINAL_TTA_VIEWS,
    ) -> np.ndarray:
        n_samples = len(loader.dataset)
        ens_std_logits = np.zeros(n_samples, dtype=np.float64)
        family_raw = {}

        for fam in self.families:
            ckpts = self.ckpt_paths_by_family[fam]
            net = SpecialistPathologyNet(family=fam, pretrained=False).to(device)
            net.eval()
            fam_sum = np.zeros(n_samples, dtype=np.float64)
            for ckpt_path in ckpts:
                ckpt_obj = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                _load_compact_state_dict(net, ckpt_obj["model_state_dict"])
                net.eval()
                fold_logits = extract_model_logits(
                    net,
                    loader,
                    device=device,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    use_tta=use_tta,
                    tta_views=tta_views,
                )
                fam_sum += fold_logits

            fam_mean = fam_sum / float(len(ckpts))
            family_raw[fam] = fam_mean
            ens_std_logits += self.family_weights[fam] * (
                fam_mean / max(self.family_sigmas[fam], 1e-6)
            )

            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

        self.last_family_raw_logits = family_raw
        return (ens_std_logits + self.logit_bias) / self.temperature


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
    """Unified, leak-free logit-space inference pipeline shared identically by validation and test sets."""
    if isinstance(net, CalibratedSpecialistEnsemble):
        cal_logits = net.predict_loader_logits(
            loader=loader,
            device=device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            use_tta=use_tta,
            tta_views=tta_views,
        )
        return _logits_to_probs(cal_logits)

    logits_all = extract_model_logits(
        net=net,
        loader=loader,
        device=device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_tta=use_tta,
        tta_views=tta_views,
    )
    return _logits_to_probs(logits_all)


def fit_oof_ensemble_and_calibration(
    oof_logits_by_family: dict,
    y_true: np.ndarray,
    center_ids: np.ndarray,
    min_weight: float = 0.15,
    bias_l2_reg: float = 0.015,
) -> dict:
    """
    Fits strictly on unseen-slide train.csv OOF 8-view D4-TTA logits:
      1) Per-family OOF logit std sigma_m (preserving logit=0 origin)
      2) Constrained simplex weights w_m >= 0.15, sum w_m = 1
      3) L2-regularized prior-shift bias b* in [-0.5, 0.5] maximizing 0.5 * mean_BA + 0.5 * worst_center_BA
      4) Class-and-center-balanced temperature T* > 0 preserving exact p=0.5 decision boundary
    """
    families = list(oof_logits_by_family.keys())
    m_fams = len(families)
    y_bin = np.asarray(y_true, dtype=np.int64)
    c_arr = np.asarray(center_ids).astype(str)
    u_centers = sorted(np.unique(c_arr).tolist())

    z_raw = np.column_stack(
        [np.asarray(oof_logits_by_family[f], dtype=np.float64) for f in families]
    )
    sigmas = np.maximum(np.std(z_raw, axis=0, ddof=0), 1e-6)
    z_std = z_raw / sigmas[None, :]

    c_pos_masks = [((c_arr == c) & (y_bin == 1)) for c in u_centers]
    c_neg_masks = [((c_arr == c) & (y_bin == 0)) for c in u_centers]

    cell_w = np.zeros(len(y_bin), dtype=np.float64)
    n_cells = max(1, 2 * len(u_centers))
    for pm, nm in zip(c_pos_masks, c_neg_masks):
        if np.any(pm):
            cell_w[pm] = 1.0 / (n_cells * float(np.sum(pm)))
        if np.any(nm):
            cell_w[nm] = 1.0 / (n_cells * float(np.sum(nm)))
    cell_w /= max(float(np.sum(cell_w)), 1e-12)

    def _eval_bias_grid(z_vec: np.ndarray, b_grid: np.ndarray):
        ba_per_c = []
        for pm, nm in zip(c_pos_masks, c_neg_masks):
            zp = np.sort(z_vec[pm])
            zn = np.sort(z_vec[nm])
            tpr_c = 1.0 - np.searchsorted(zp, -b_grid, side="left") / float(len(zp))
            tnr_c = np.searchsorted(zn, -b_grid, side="left") / float(len(zn))
            ba_per_c.append(0.5 * (tpr_c + tnr_c))
        ba_mat = np.vstack(ba_per_c)
        mean_ba = np.mean(ba_mat, axis=0)
        worst_ba = np.min(ba_mat, axis=0)
        hard_comp = 0.5 * mean_ba + 0.5 * worst_ba
        reg_obj = hard_comp - bias_l2_reg * (b_grid**2)
        return reg_obj, hard_comp, mean_ba, worst_ba

    def _soft_center_ba_vec(z_vec: np.ndarray, b_vals: np.ndarray, tau: float = 0.25):
        u = np.clip((z_vec[None, :] + b_vals[:, None]) / tau, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-u))
        soft_c = []
        for pm, nm in zip(c_pos_masks, c_neg_masks):
            stpr = np.mean(p[:, pm], axis=1)
            stnr = np.mean(1.0 - p[:, nm], axis=1)
            soft_c.append(0.5 * (stpr + stnr))
        soft_mat = np.vstack(soft_c)
        return 0.5 * np.mean(soft_mat, axis=0) + 0.5 * np.min(soft_mat, axis=0)

    if m_fams == 1:
        w_grid = np.array([[1.0]], dtype=np.float64)
    elif m_fams == 2:
        lo = min(min_weight, 0.5)
        w0 = np.linspace(lo, 1.0 - lo, 141)
        w_grid = np.column_stack([w0, 1.0 - w0])
    else:
        lo = min(min_weight, 1.0 / float(m_fams))
        rows = []
        steps = 100
        for i in range(steps + 1):
            w0 = i / float(steps)
            if w0 < lo - 1e-9 or w0 > 1.0 - 2.0 * lo + 1e-9:
                continue
            for j in range(steps + 1 - i):
                w1 = j / float(steps)
                w2 = 1.0 - w0 - w1
                if w1 >= lo - 1e-9 and w2 >= lo - 1e-9:
                    rows.append([w0, w1, w2])
        w_grid = (
            np.asarray(rows, dtype=np.float64)
            if len(rows) > 0
            else np.full((1, m_fams), 1.0 / float(m_fams), dtype=np.float64)
        )

    b_coarse = np.linspace(-0.5, 0.5, 401)
    best_w_score = -1e9
    best_w = w_grid[0].copy()
    y_f = y_bin.astype(np.float64)

    for w_cand in w_grid:
        z_cand = z_std @ w_cand
        reg_obj, _, _, _ = _eval_bias_grid(z_cand, b_coarse)
        b_idx = int(np.argmax(reg_obj))
        b_c = float(b_coarse[b_idx])
        soft_val = float(_soft_center_ba_vec(z_cand, np.array([b_c]), tau=0.25)[0])
        u_bce = np.clip(z_cand + b_c, -35.0, 35.0)
        bce_val = float(
            np.sum(
                cell_w
                * (np.maximum(u_bce, 0.0) - u_bce * y_f + np.log1p(np.exp(-np.abs(u_bce))))
            )
        )
        unif_pen = float(np.sum((w_cand - 1.0 / float(m_fams)) ** 2))
        w_score = (
            float(reg_obj[b_idx])
            + 0.15 * soft_val
            - 0.02 * bce_val
            - 0.005 * unif_pen
        )
        if w_score > best_w_score:
            best_w_score = w_score
            best_w = w_cand.copy()

    best_w = np.clip(best_w, min_weight if m_fams > 1 else 1.0, 1.0)
    best_w = best_w / np.sum(best_w)
    z_ens_oof = z_std @ best_w

    b_fine = np.linspace(-0.5, 0.5, 2001)
    reg_fine, hard_fine, mean_ba_fine, worst_ba_fine = _eval_bias_grid(
        z_ens_oof, b_fine
    )
    soft_fine = np.empty_like(b_fine)
    for start_i in range(0, len(b_fine), 250):
        end_i = min(len(b_fine), start_i + 250)
        soft_fine[start_i:end_i] = _soft_center_ba_vec(
            z_ens_oof, b_fine[start_i:end_i], tau=0.25
        )
    fine_total = reg_fine + 0.02 * soft_fine
    best_b_idx = int(np.argmax(fine_total))
    b_star = float(np.clip(b_fine[best_b_idx], -0.5, 0.5))

    z_shifted = z_ens_oof + b_star
    t_grid = np.geomspace(0.25, 4.0, 401)
    best_t = 1.0
    best_nll = 1e18
    for t_cand in t_grid:
        u_t = np.clip(z_shifted / float(t_cand), -40.0, 40.0)
        nll_t = float(
            np.sum(
                cell_w
                * (np.maximum(u_t, 0.0) - u_t * y_f + np.log1p(np.exp(-np.abs(u_t))))
            )
            + 0.002 * (math.log(float(t_cand)) ** 2)
        )
        if nll_t < best_nll:
            best_nll = nll_t
            best_t = float(t_cand)

    oof_cal_logits = z_shifted / best_t
    oof_cal_probs = _logits_to_probs(oof_cal_logits)
    oof_metrics = binary_metrics(y_bin, oof_cal_probs)

    return {
        "families": families,
        "family_sigmas": {f: float(sigmas[i]) for i, f in enumerate(families)},
        "family_weights": {f: float(best_w[i]) for i, f in enumerate(families)},
        "logit_bias": b_star,
        "temperature": best_t,
        "oof_cal_probs": oof_cal_probs,
        "oof_metrics": oof_metrics,
        "oof_mean_center_ba": float(mean_ba_fine[best_b_idx]),
        "oof_worst_center_ba": float(worst_ba_fine[best_b_idx]),
        "oof_composite_ba": float(hard_fine[best_b_idx]),
    }


# Release smoke-test primary model VRAM before sequential 4-fold specialist training
del model, criterion, optimizer
gc.collect()
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

gpu_augmentor = GPUHistopathologyAugmentor(stain_stats=stain_stats_train).to(DEVICE)
gpu_augmentor.train()

specialist_priority = list(MODEL_CONFIG["specialist_priority"])
completed_families = []
ckpt_paths_by_family = {}
oof_logits_by_family = {}
family_wall_times = []
measured_8tta_sec_per_sample = 0.0022

print(
    f"Starting 4-fold StratifiedGroupKFold training strictly within train.csv "
    f"(validation.csv 100% untouched) | families={specialist_priority}, "
    f"folds={NUM_CV_FOLDS}, epochs/fold={FOLD_EPOCHS}, batch={TRAIN_BATCH_SIZE}x{GRAD_ACCUM_STEPS}, "
    f"amp={AMP_DTYPE}, ram_cache={RAM_CACHE_ACTIVE}, dl_workers={num_dl_workers}"
)

for fam_idx, family in enumerate(specialist_priority):
    elapsed_script = time.time() - SCRIPT_START_TIME
    if fam_idx > 0:
        avg_prev_fam_sec = float(np.mean(family_wall_times))
        rel_cost = 0.92 if family == "swinv2" else 0.85
        est_next_fam_sec = avg_prev_fam_sec * rel_cost
        n_eval_samples = float(len(val_dataset) + len(test_dataset))
        est_final_infer_sec = max(
            120.0,
            1.25
            * float((len(completed_families) + 1) * NUM_CV_FOLDS)
            * n_eval_samples
            * measured_8tta_sec_per_sample,
        )
        if elapsed_script + est_next_fam_sec + est_final_infer_sec > TOTAL_SCRIPT_BUDGET_SEC:
            print(
                f"Wall-clock budget guard before family '{family}': elapsed={elapsed_script:.0f}s, "
                f"est_family={est_next_fam_sec:.0f}s, reserved_infer={est_final_infer_sec:.0f}s > "
                f"budget={TOTAL_SCRIPT_BUDGET_SEC:.0f}s. Proceeding with completed families={completed_families}."
            )
            break

    fam_t0 = time.time()
    fam_oof_logits = np.zeros(len(train_df), dtype=np.float64)
    fam_covered = np.zeros(len(train_df), dtype=bool)
    fam_ckpts = []
    fam_aborted = False

    for fold_idx in range(NUM_CV_FOLDS):
        fold_seed = SEED + fam_idx * 1009 + fold_idx * 101
        seed_everything(fold_seed)

        va_mask = (train_df["cv_fold"].values == fold_idx)
        tr_mask = ~va_mask
        fold_tr_df = train_df.loc[tr_mask].reset_index(drop=True)
        fold_va_df = train_df.loc[va_mask].reset_index(drop=True)
        y_fold_va = fold_va_df["label"].values.astype(np.int64)

        assert set(fold_tr_df["group_id"]).isdisjoint(set(fold_va_df["group_id"]))
        assert set(fold_tr_df["slide_id"]).isdisjoint(set(fold_va_df["slide_id"]))
        assert set(fold_va_df["slide_id"]).isdisjoint(set(val_df["slide_id"]))

        fold_tr_ds = HistopathologyDataset(
            fold_tr_df, stain_stats=stain_stats_train, features=None, is_train=True
        )
        fold_va_ds = HistopathologyDataset(
            fold_va_df, stain_stats=stain_stats_train, features=None, is_train=False
        )
        fold_sampler = StratifiedCenterClassBatchSampler(
            fold_tr_df, batch_size=TRAIN_BATCH_SIZE, seed=fold_seed
        )
        fold_tr_loader = DataLoader(
            fold_tr_ds,
            batch_sampler=fold_sampler,
            num_workers=num_dl_workers,
            pin_memory=pin_mem,
            **dl_worker_kwargs,
        )
        fold_va_loader = DataLoader(
            fold_va_ds,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=num_dl_workers,
            pin_memory=pin_mem,
            drop_last=False,
            **dl_worker_kwargs,
        )

        try:
            fold_model, fold_criterion, fold_opt = build_model_criterion_optimizer(
                family=family,
                pos_prior=float(fold_tr_df["label"].mean()),
                pretrained=True,
                device=DEVICE,
            )
        except Exception as exc:
            print(f"Skipping specialist family '{family}' due to init error: {exc}")
            fam_aborted = True
            break

        for pg in fold_opt.param_groups:
            pg["initial_lr"] = float(pg["lr"])

        fold_ema_model = copy.deepcopy(fold_model).to(DEVICE)
        fold_ema_model.eval()
        for p_ema in fold_ema_model.parameters():
            p_ema.requires_grad_(False)

        scaler = torch.amp.GradScaler(DEVICE.type, enabled=USE_GRAD_SCALER)
        fold_steps_per_ep = max(1, len(fold_tr_loader) // GRAD_ACCUM_STEPS)
        fold_total_steps = FOLD_EPOCHS * fold_steps_per_ep
        fold_opt_step = 0

        best_fold_ba = -1.0
        best_fold_auc = -1.0
        best_fold_ep = 0
        fold_ckpt_path = os.path.join(
            CKPT_DIR, f"{family}_fold{fold_idx}_best_ema.pt"
        )

        for epoch in range(1, FOLD_EPOCHS + 1):
            ep_t0 = time.time()
            fold_model.train()
            fold_criterion.train()
            gpu_augmentor.train()
            fold_sampler.set_epoch(epoch)
            fold_opt.zero_grad(set_to_none=True)

            running_loss = 0.0
            valid_batches = 0
            current_head_lr = fold_opt.param_groups[0]["lr"]

            for step_in_ep, batch in enumerate(fold_tr_loader):
                if step_in_ep % GRAD_ACCUM_STEPS == 0:
                    current_head_lr = set_step_learning_rates(
                        fold_opt, fold_opt_step, fold_total_steps, fold_steps_per_ep
                    )

                images = batch["image"].to(DEVICE, non_blocking=True)
                images = gpu_augmentor(images)
                labels = batch["label"].to(DEVICE, non_blocking=True)
                center_idx = batch["center_idx"].to(DEVICE, non_blocking=True)

                with torch.amp.autocast(
                    device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP
                ):
                    logits, aux_outputs = fold_model(images, return_aux=True)
                    raw_loss = fold_criterion(
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

                is_boundary = ((step_in_ep + 1) % GRAD_ACCUM_STEPS == 0) or (
                    (step_in_ep + 1) == len(fold_tr_loader)
                )
                if is_boundary:
                    if USE_GRAD_SCALER:
                        scaler.unscale_(fold_opt)
                        torch.nn.utils.clip_grad_norm_(
                            fold_model.parameters(), max_norm=MAX_GRAD_NORM
                        )
                        scaler.step(fold_opt)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            fold_model.parameters(), max_norm=MAX_GRAD_NORM
                        )
                        fold_opt.step()
                    fold_opt.zero_grad(set_to_none=True)

                    fold_opt_step += 1
                    update_ema_weights(fold_ema_model, fold_model, fold_opt_step)

            train_loss_ep = running_loss / float(max(1, valid_batches))
            fold_criterion.eval()

            ema_va_logits = extract_model_logits(
                fold_ema_model,
                fold_va_loader,
                device=DEVICE,
                use_amp=USE_AMP,
                amp_dtype=AMP_DTYPE,
                use_tta=True,
                tta_views=FOLD_VAL_TTA_VIEWS,
            )
            ema_va_probs = _logits_to_probs(ema_va_logits)
            ema_m = binary_metrics(y_fold_va, ema_va_probs)
            ep_ba = float(ema_m["balanced_accuracy"])
            ep_auc = float(ema_m["auroc"])

            improved = (ep_ba > best_fold_ba + 1e-6) or (
                abs(ep_ba - best_fold_ba) <= 1e-6 and ep_auc > best_fold_auc + 1e-6
            )
            if improved:
                best_fold_ba = ep_ba
                best_fold_auc = ep_auc
                best_fold_ep = epoch
                torch.save(
                    {
                        "family": family,
                        "backbone_name": fold_model.backbone_name,
                        "fold": fold_idx,
                        "epoch": epoch,
                        "fold_val_ba": best_fold_ba,
                        "fold_val_auroc": best_fold_auc,
                        "model_state_dict": _compact_state_dict_cpu(
                            fold_ema_model.state_dict(), dtype=CKPT_SAVE_DTYPE
                        ),
                    },
                    fold_ckpt_path,
                )

            ep_sec = time.time() - ep_t0
            mark = "*" if improved else ""
            print(
                f"[{family} | fold {fold_idx + 1}/{NUM_CV_FOLDS}] "
                f"Epoch {epoch:02d}/{FOLD_EPOCHS:02d} | loss={train_loss_ep:.4f} | "
                f"oof_ema_ba={ep_ba:.5f} | oof_ema_auc={ep_auc:.5f} | "
                f"lr={current_head_lr:.2e} | {ep_sec:.1f}s {mark}"
            )

            # Emergency mid-fold wall-clock guard (ensuring current family completes all 4 folds safely)
            if epoch >= 5 and epoch < FOLD_EPOCHS:
                elapsed_now = time.time() - SCRIPT_START_TIME
                rem_folds_in_fam = NUM_CV_FOLDS - 1 - fold_idx
                min_rem_fam_sec = rem_folds_in_fam * (5.0 * ep_sec + 10.0) + ep_sec
                n_eval_s = float(len(val_dataset) + len(test_dataset))
                min_final_sec = max(
                    120.0,
                    1.20
                    * float((len(completed_families) + 1) * NUM_CV_FOLDS)
                    * n_eval_s
                    * measured_8tta_sec_per_sample,
                )
                if elapsed_now + min_rem_fam_sec + min_final_sec > TOTAL_SCRIPT_BUDGET_SEC:
                    print(
                        f"Mid-fold wall-clock cap triggered at epoch {epoch} to preserve 100% 4-fold OOF coverage."
                    )
                    break

        # Extract exact 8-view D4-TTA OOF logits from saved bf16/fp16 best EMA checkpoint
        best_ckpt_obj = torch.load(
            fold_ckpt_path, map_location="cpu", weights_only=False
        )
        _load_compact_state_dict(fold_ema_model, best_ckpt_obj["model_state_dict"])
        fold_ema_model.eval()

        tta_t0 = time.time()
        fold_oof_8tta = extract_model_logits(
            fold_ema_model,
            fold_va_loader,
            device=DEVICE,
            use_amp=USE_AMP,
            amp_dtype=AMP_DTYPE,
            use_tta=True,
            tta_views=FINAL_TTA_VIEWS,
        )
        tta_dt = time.time() - tta_t0
        measured_8tta_sec_per_sample = 0.5 * measured_8tta_sec_per_sample + 0.5 * (
            tta_dt / float(max(1, len(fold_va_ds)))
        )

        fold_8tta_m = binary_metrics(y_fold_va, _logits_to_probs(fold_oof_8tta))
        print(
            f"[{family} | fold {fold_idx + 1}/{NUM_CV_FOLDS}] Best EMA ep={best_fold_ep} "
            f"8-view D4-TTA OOF | ba={fold_8tta_m['balanced_accuracy']:.5f}, "
            f"auc={fold_8tta_m['auroc']:.5f} ({tta_dt:.1f}s)"
        )

        fam_oof_logits[va_mask] = fold_oof_8tta
        fam_covered[va_mask] = True
        fam_ckpts.append(fold_ckpt_path)

        del fold_model, fold_ema_model, fold_criterion, fold_opt, scaler
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if (not fam_aborted) and bool(np.all(fam_covered)) and (len(fam_ckpts) == NUM_CV_FOLDS):
        fam_dt = time.time() - fam_t0
        family_wall_times.append(fam_dt)
        completed_families.append(family)
        ckpt_paths_by_family[family] = fam_ckpts
        oof_logits_by_family[family] = fam_oof_logits

        fam_oof_m = binary_metrics(y_train, _logits_to_probs(fam_oof_logits))
        np.save(os.path.join(WORKING_DIR, f"oof_logits_{family}.npy"), fam_oof_logits)
        print(
            f"Completed 4-fold OOF for specialist '{family}' in {fam_dt / 60.0:.2f} min | "
            f"uncalibrated OOF ba={fam_oof_m['balanced_accuracy']:.6f}, "
            f"auroc={fam_oof_m['auroc']:.6f}, logit_std={np.std(fam_oof_logits):.4f}"
        )

assert len(completed_families) >= 1, "At least one 4-fold specialist family must complete!"

# Fit leak-free OOF logit std standardization, simplex weights, regularized bias b*, and temperature T*
calib_result = fit_oof_ensemble_and_calibration(
    oof_logits_by_family=oof_logits_by_family,
    y_true=y_train,
    center_ids=train_df["center_id"].values,
    min_weight=0.15,
    bias_l2_reg=0.015,
)

oof_cal_probs = calib_result["oof_cal_probs"]
oof_preds_bin = (oof_cal_probs >= 0.5).astype(int)
oof_center_parts = []
for cid, grp in train_df.groupby("center_id"):
    idx = grp.index.values
    c_m = binary_metrics(y_train[idx], oof_cal_probs[idx])
    c_y = y_train[idx]
    c_p = oof_preds_bin[idx]
    c_tpr = float(c_p[c_y == 1].mean())
    c_tnr = float((c_p[c_y == 0] == 0).mean())
    oof_center_parts.append(
        f"center_{cid}: ba={c_m['balanced_accuracy']:.4f}, tpr={c_tpr:.4f}, tnr={c_tnr:.4f}, auc={c_m['auroc']:.4f}"
    )

print(
    f"Calibrated 4-Fold OOF Ensemble ({len(completed_families)} families, {len(completed_families) * NUM_CV_FOLDS} ckpts) | "
    f"weights={ {k: round(v, 4) for k, v in calib_result['family_weights'].items()} } | "
    f"sigmas={ {k: round(v, 4) for k, v in calib_result['family_sigmas'].items()} } | "
    f"b*={calib_result['logit_bias']:+.4f}, T*={calib_result['temperature']:.4f} | "
    f"oof_ba={calib_result['oof_metrics']['balanced_accuracy']:.6f}, "
    f"worst_center_ba={calib_result['oof_worst_center_ba']:.6f}, "
    f"oof_auroc={calib_result['oof_metrics']['auroc']:.6f} | "
    + " | ".join(oof_center_parts)
)

pd.DataFrame(
    {
        "sample_id": train_df["sample_id"].values,
        "label": y_train,
        "center_id": train_df["center_id"].values,
        "slide_id": train_df["slide_id"].values,
        "cv_fold": train_df["cv_fold"].values,
        "tumor_probability": oof_cal_probs,
    }
).to_csv(os.path.join(WORKING_DIR, "oof_predictions.csv"), index=False)

# Build unified calibrated ensemble model & persist metadata artifact
model = CalibratedSpecialistEnsemble(
    families=completed_families,
    ckpt_paths_by_family=ckpt_paths_by_family,
    family_sigmas=calib_result["family_sigmas"],
    family_weights=calib_result["family_weights"],
    logit_bias=calib_result["logit_bias"],
    temperature=calib_result["temperature"],
)

torch.save(
    {
        "families": completed_families,
        "ckpt_paths_by_family": ckpt_paths_by_family,
        "family_sigmas": calib_result["family_sigmas"],
        "family_weights": calib_result["family_weights"],
        "logit_bias": calib_result["logit_bias"],
        "temperature": calib_result["temperature"],
        "oof_metrics": calib_result["oof_metrics"],
    },
    BEST_MODEL_PATH,
)

# Execute ONE identical calibrated logit-space 8-view D4-TTA ensemble pipeline on validation.csv and test.csv
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

for fam in completed_families:
    if fam in model.last_family_raw_logits:
        fam_val_raw = model.last_family_raw_logits[fam]
        fam_val_cal = (
            fam_val_raw / max(calib_result["family_sigmas"][fam], 1e-6)
            + calib_result["logit_bias"]
        ) / calib_result["temperature"]
        fam_m_raw = binary_metrics(y_val, _logits_to_probs(fam_val_raw))
        fam_m_cal = binary_metrics(y_val, _logits_to_probs(fam_val_cal))
        print(
            f"Validation Specialist '{fam}' (4-fold D4-TTA8) | "
            f"raw_ba={fam_m_raw['balanced_accuracy']:.6f}, cal_ba={fam_m_cal['balanced_accuracy']:.6f}, "
            f"auroc={fam_m_raw['auroc']:.6f}"
        )

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
    f"Calibrated Ensemble ({len(completed_families)} families x {NUM_CV_FOLDS} folds) D4-TTA({FINAL_TTA_VIEWS}) Audit | "
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
    f"pred_pos_rate={(submission_df['tumor_probability'] >= 0.5).mean():.4f}, "
    f"elapsed={(time.time() - SCRIPT_START_TIME) / 60.0:.2f}m"
)
print(f"Final Validation Score: {validation_balanced_accuracy}")
