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
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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
FOURIER_LOW_RADIUS = 12  # Half-plane 24x12 rfft2 low-frequency crop around DC
FOURIER_BANK_SIZE = 768
MAX_RUNTIME_BUDGET_SEC = 2800.0  # Target ceiling within 1-hour hard limit

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
# 3. Training-Only H/E/R Anchors & Half-Plane rfft2 Low-Frequency Bank
# =============================================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _extract_patch_her_and_rfft_low(rel_image_path: str):
    """Extract foreground-calibrated [H, E, R] moments and 24x12 rfft2 low-frequency crop."""
    rgb = load_rgb_patch(rel_image_path).astype(np.float32)
    od = rgb_to_od(rgb)
    stain = od @ HE_INV_MATRIX

    h_pos = np.clip(stain[:, :, 0], 0.0, None)
    e_pos = np.clip(stain[:, :, 1], 0.0, None)
    r_val = stain[:, :, 2]
    her = np.stack([h_pos, e_pos, r_val], axis=-1)

    fg = np.linalg.norm(od, axis=2) > 0.12
    her_ref = her[fg] if int(fg.sum()) >= 32 else her.reshape(-1, 3)
    her_mean = her_ref.mean(axis=0).astype(np.float32)
    her_sq = (her_ref**2).mean(axis=0).astype(np.float32)

    rgb01_chw = np.transpose(rgb / 255.0, (2, 0, 1))
    amp = np.abs(np.fft.rfft2(rgb01_chw, axes=(-2, -1), norm="ortho")).astype(
        np.float32
    )
    r = FOURIER_LOW_RADIUS
    low_amp = np.concatenate([amp[:, :r, :r], amp[:, -r:, :r]], axis=1).astype(
        np.float32
    )
    return her_mean, her_sq, low_amp


def fit_train_only_image_statistics_and_fourier_bank(
    train_dataframe: pd.DataFrame,
    seed: int = SEED,
    center_to_idx: Optional[dict] = None,
) -> dict:
    """
    Fit compact foreground [H, E, Residual] anchors and stratified Cross-Center rfft2
    Low-Frequency Amplitude Bank (N, 3, 24, 12) STRICTLY on training samples.
    """
    rng = np.random.default_rng(seed)
    paths = train_dataframe["image_path"].astype(str).tolist()
    centers = train_dataframe["center_id"].astype(str).values
    unique_centers = sorted(np.unique(centers).tolist())
    if center_to_idx is None:
        center_to_idx = {cid: idx for idx, cid in enumerate(unique_centers)}

    per_center_quota = max(1, FOURIER_BANK_SIZE // max(1, len(unique_centers)))
    bank_indices = []
    for cid in unique_centers:
        c_idx = np.where(centers == cid)[0]
        chosen = rng.choice(
            c_idx, size=min(per_center_quota, len(c_idx)), replace=False
        )
        bank_indices.append((center_to_idx.get(cid, 0), chosen))

    stat_indices = rng.choice(len(paths), size=min(2048, len(paths)), replace=False)
    stat_set = set(stat_indices.tolist())
    for _, c_idx in bank_indices:
        stat_set.update(c_idx.tolist())
    eval_indices = sorted(stat_set)
    eval_paths = [paths[i] for i in eval_indices]

    n_workers = max(2, min(8, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        res = list(ex.map(_extract_patch_her_and_rfft_low, eval_paths))

    idx_to_pos = {orig_i: pos for pos, orig_i in enumerate(eval_indices)}
    her_means = np.stack([res[idx_to_pos[i]][0] for i in stat_indices], axis=0)
    her_sqs = np.stack([res[idx_to_pos[i]][1] for i in stat_indices], axis=0)

    her_mean = her_means.mean(axis=0).astype(np.float32)
    her_std = np.sqrt(np.maximum(her_sqs.mean(axis=0) - her_mean**2, 1e-6)).astype(
        np.float32
    )

    bank_arrays = []
    bank_center_ids = []
    for cid_num, c_idx in bank_indices:
        for i in c_idx:
            bank_arrays.append(res[idx_to_pos[int(i)]][2])
            bank_center_ids.append(int(cid_num))

    fourier_bank = np.stack(bank_arrays, axis=0).astype(np.float32)
    fourier_bank_centers = np.asarray(bank_center_ids, dtype=np.int64)

    return {
        "rgb_mean": IMAGENET_MEAN.copy(),
        "rgb_std": IMAGENET_STD.copy(),
        "her_mean": her_mean,
        "her_std": her_std,
        "fourier_bank": fourier_bank,
        "fourier_bank_centers": fourier_bank_centers,
    }


# =============================================================================
# 4. GPU-Batched HistopathologyPhysicsFrontend & Fast uint8 Dataset
# =============================================================================
class HistopathologyPhysicsFrontend(nn.Module):
    """
    GPU-batched float32 physics & spectral frontend:
      - Training-only: exact D4 dihedral symmetry, white-point illumination jitter,
        3D OD unit-sphere Macenko/Ruifrok-Johnston stain-basis vector rotation,
        log-uniform H&E dye concentration + gamma scaling, radial MTF + GPU rfft2
        cross-center low-frequency amplitude perturbation, and intra-class cross-center
        Beta(0.4, 0.4) pixel mixup.
      - Deterministic representation (100% identical across train, val, and test):
        ImageNet-normalized RGB + 4-channel stain-structural tensor:
        [standardized H, standardized E, scale-invariant chromatic nuclear fraction H/(H+E+eps),
         and local 5x5 standardized chromatin contrast gated by orthogonal pigment residual R].
    """

    def __init__(self, image_stats: dict):
        super().__init__()
        self.register_buffer(
            "he_stain_matrix", torch.from_numpy(HE_STAIN_MATRIX.copy())
        )
        self.register_buffer("he_inv_matrix", torch.from_numpy(HE_INV_MATRIX.copy()))
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "her_mean",
            torch.as_tensor(image_stats["her_mean"], dtype=torch.float32).view(3),
        )
        self.register_buffer(
            "her_std",
            torch.as_tensor(image_stats["her_std"], dtype=torch.float32).view(3),
        )

        f_bank = torch.as_tensor(image_stats["fourier_bank"], dtype=torch.float32)
        f_ctrs = np.asarray(image_stats["fourier_bank_centers"], dtype=np.int64)
        self.register_buffer("fourier_bank", f_bank, persistent=False)
        not0 = np.where(f_ctrs != 0)[0]
        not1 = np.where(f_ctrs != 1)[0]
        self.register_buffer(
            "bank_idx_not0",
            torch.from_numpy(not0 if len(not0) > 0 else np.arange(len(f_ctrs))),
            persistent=False,
        )
        self.register_buffer(
            "bank_idx_not1",
            torch.from_numpy(not1 if len(not1) > 0 else np.arange(len(f_ctrs))),
            persistent=False,
        )

        r = FOURIER_LOW_RADIUS
        ky = torch.cat(
            [
                torch.arange(0, r, dtype=torch.float32),
                torch.arange(-r, 0, dtype=torch.float32),
            ]
        )[:, None]
        kx = torch.arange(0, r, dtype=torch.float32)[None, :]
        rr = (torch.sqrt(ky.square() + kx.square()) / float(r)).clamp(0.0, 1.0)
        low_win = (0.5 * (1.0 + torch.cos(math.pi * rr))).view(1, 1, 2 * r, r)
        self.register_buffer("low_freq_win", low_win, persistent=False)

        fy = torch.fft.fftfreq(IMG_SIZE, d=1.0)[:, None] * 2.0
        fx = torch.fft.rfftfreq(IMG_SIZE, d=1.0)[None, :] * 2.0
        frad_sq = ((fy.square() + fx.square()) * 0.5).view(
            1, 1, IMG_SIZE, IMG_SIZE // 2 + 1
        )
        self.register_buffer("frad_norm_sq", frad_sq, persistent=False)

    @staticmethod
    def _to_unit_rgb(rgb: torch.Tensor) -> torch.Tensor:
        if rgb.dtype == torch.uint8:
            return rgb.float().div(255.0)
        x = rgb.float()
        if x.numel() > 0 and float(x.detach().max()) > 2.0:
            x = x.div(255.0)
        return x.clamp(0.0, 1.0)

    def _augment_rgb01(
        self, x: torch.Tensor, center_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B = x.size(0)
        dev = x.device

        # 1. Lossless D4 dihedral symmetry (random flip + 90-deg rotations)
        flip_m = torch.rand(B, 1, 1, 1, device=dev) < 0.5
        x = torch.where(flip_m, torch.flip(x, dims=[-1]), x)
        k_rot = torch.randint(0, 4, (B,), device=dev)
        for k in (1, 2, 3):
            mk = k_rot == k
            if mk.any():
                x[mk] = torch.rot90(x[mk], k=k, dims=(-2, -1))

        # 2. White-point illumination jitter
        do_wp = (torch.rand(B, 1, 1, 1, device=dev) < 0.75).float()
        wp_gain = torch.exp(torch.empty(B, 3, 1, 1, device=dev).uniform_(-0.06, 0.06))
        x = (x * (1.0 + do_wp * (wp_gain - 1.0))).clamp(1.0 / 255.0, 1.0)

        # 3. 3D OD unit-sphere stain-basis rotation + log-uniform dye concentration & gamma
        do_stain = torch.rand(B, 1, 1, 1, device=dev) < 0.80
        if do_stain.any():
            od = -torch.log10(x.clamp(1.0 / 255.0, 1.0))
            stain = torch.einsum("bchw,cd->bdhw", od, self.he_inv_matrix)
            tissue = (
                torch.linalg.vector_norm(od, dim=1, keepdim=True) > 0.10
            ).float()

            alpha_he = torch.exp(
                torch.empty(B, 2, 1, 1, device=dev).uniform_(-0.28, 0.28)
            )
            alpha_r = torch.exp(
                torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.18, 0.18)
            )
            beta_he = torch.empty(B, 2, 1, 1, device=dev).uniform_(-0.04, 0.04)

            st_he = stain[:, :2] * alpha_he + beta_he * tissue
            st_r = stain[:, 2:3] * alpha_r
            stain_p = torch.cat([st_he, st_r], dim=1)

            vh_base = self.he_stain_matrix[0:1].expand(B, -1)
            ve_base = self.he_stain_matrix[1:2].expand(B, -1)
            vh_rot = F.normalize(
                (vh_base + torch.randn(B, 3, device=dev) * 0.065).clamp(min=0.01),
                dim=-1,
            )
            ve_rot = F.normalize(
                (ve_base + torch.randn(B, 3, device=dev) * 0.065).clamp(min=0.01),
                dim=-1,
            )
            vr_rot = F.normalize(torch.cross(vh_rot, ve_rot, dim=-1), dim=-1)
            basis_rot = torch.stack([vh_rot, ve_rot, vr_rot], dim=1)

            od_rec = torch.einsum("bdhw,bdc->bchw", stain_p, basis_rot).clamp(0.0, 3.0)
            x_stain = torch.pow(10.0, -od_rec).clamp(1.0 / 255.0, 1.0)
            x = torch.where(do_stain, x_stain, x)

        do_gamma = (torch.rand(B, 1, 1, 1, device=dev) < 0.70).float()
        gamma = 1.0 + do_gamma * (
            torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.14, 0.14)) - 1.0
        )
        x = torch.pow(x.clamp(1e-4, 1.0), gamma)

        # 4. Joint GPU rfft2 Cross-Center Amplitude Mixing + Radial MTF Perturbation
        n_bank = self.fourier_bank.size(0)
        if n_bank > 0:
            x_fft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
            amp = x_fft.abs()
            unit_phase = x_fft / (amp + 1e-8)

            b_idx = torch.randint(0, n_bank, (B,), device=dev)
            if center_idx is not None:
                prefer_opp = torch.rand(B, device=dev) < 0.80
                idx_n0 = self.bank_idx_not0[
                    torch.randint(0, self.bank_idx_not0.numel(), (B,), device=dev)
                ]
                idx_n1 = self.bank_idx_not1[
                    torch.randint(0, self.bank_idx_not1.numel(), (B,), device=dev)
                ]
                b_idx = torch.where(prefer_opp & (center_idx == 0), idx_n0, b_idx)
                b_idx = torch.where(prefer_opp & (center_idx == 1), idx_n1, b_idx)

            tgt_low = self.fourier_bank[b_idx]
            do_fact = (torch.rand(B, 1, 1, 1, device=dev) < 0.65).float()
            alpha_f = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.15, 0.50)
            w = do_fact * alpha_f * self.low_freq_win

            r = FOURIER_LOW_RADIUS
            w_top, w_bot = w[:, :, :r, :], w[:, :, r:, :]
            amp[:, :, :r, :r] = (1.0 - w_top) * amp[:, :, :r, :r] + w_top * tgt_low[
                :, :, :r, :
            ]
            amp[:, :, -r:, :r] = (1.0 - w_bot) * amp[:, :, -r:, :r] + w_bot * tgt_low[
                :, :, r:, :
            ]

            do_mtf = (torch.rand(B, 1, 1, 1, device=dev) < 0.50).float()
            beta_mtf = do_mtf * torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.22, 0.65)
            mtf_gain = torch.exp(-beta_mtf * self.frad_norm_sq)
            amp = amp * mtf_gain

            x = torch.fft.irfft2(
                amp * unit_phase, s=(IMG_SIZE, IMG_SIZE), dim=(-2, -1), norm="ortho"
            ).clamp(0.0, 1.0)

        do_noise = (torch.rand(B, 1, 1, 1, device=dev) < 0.20).float()
        noise_sd = do_noise * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.003, 0.010)
        x = (x + torch.randn_like(x) * noise_sd).clamp(0.0, 1.0)
        return x

    @staticmethod
    def apply_intraclass_crosscenter_mixup(
        x01: torch.Tensor,
        labels: torch.Tensor,
        center_idx: torch.Tensor,
        mix_prob: float = 0.30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beta(0.4, 0.4) pixel mixup strictly between same-class opposite-center patches."""
        B = x01.size(0)
        if B < 2 or mix_prob <= 0.0:
            return x01, center_idx
        dev = x01.device
        same_cls = (labels.unsqueeze(1) - labels.unsqueeze(0)).abs() < 0.1
        diff_ctr = (
            (center_idx.unsqueeze(1) != center_idx.unsqueeze(0))
            & (center_idx.unsqueeze(1) >= 0)
            & (center_idx.unsqueeze(0) >= 0)
        )
        compat = same_cls & diff_ctr
        has_partner = compat.any(dim=1)
        if not has_partner.any():
            return x01, center_idx

        scores = torch.rand(B, B, device=dev).masked_fill(~compat, -1.0)
        partner_idx = scores.argmax(dim=1)
        do_mix = (torch.rand(B, device=dev) < mix_prob) & has_partner
        if not do_mix.any():
            return x01, center_idx

        beta_dist = torch.distributions.Beta(
            torch.tensor(0.4, device=dev), torch.tensor(0.4, device=dev)
        )
        lam = beta_dist.sample((B, 1, 1, 1))
        x_mixed = torch.where(
            do_mix.view(B, 1, 1, 1), lam * x01 + (1.0 - lam) * x01[partner_idx], x01
        )
        eff_center = torch.where(
            do_mix, torch.full_like(center_idx, -1), center_idx
        )
        return x_mixed.clamp(0.0, 1.0), eff_center

    def compute_representations(
        self, x01: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deterministic ImageNet RGB + 4-channel scale-invariant stain-structural tensor."""
        rgb_norm = (x01 - self.imagenet_mean) / self.imagenet_std

        od = -torch.log10(x01.clamp(1.0 / 255.0, 1.0))
        stain = torch.einsum("bchw,cd->bdhw", od, self.he_inv_matrix)
        h_pos = stain[:, 0:1].clamp(min=0.0)
        e_pos = stain[:, 1:2].clamp(min=0.0)
        r_val = stain[:, 2:3]

        fg = (torch.linalg.vector_norm(od, dim=1, keepdim=True) > 0.12).float()
        fg_cnt = fg.sum(dim=(-2, -1), keepdim=True)
        fg_w = fg_cnt / (fg_cnt + 256.0)
        denom = fg_cnt.clamp(min=1.0)

        fg_h_mu = (h_pos * fg).sum(dim=(-2, -1), keepdim=True) / denom
        fg_h_sd = torch.sqrt(
            (((h_pos - fg_h_mu).square()) * fg).sum(dim=(-2, -1), keepdim=True) / denom
            + 1e-5
        )
        h_mu = 0.65 * fg_w * fg_h_mu + (1.0 - 0.65 * fg_w) * self.her_mean[0]
        h_sd = 0.65 * fg_w * fg_h_sd + (1.0 - 0.65 * fg_w) * self.her_std[0] + 1e-4

        fg_e_mu = (e_pos * fg).sum(dim=(-2, -1), keepdim=True) / denom
        fg_e_sd = torch.sqrt(
            (((e_pos - fg_e_mu).square()) * fg).sum(dim=(-2, -1), keepdim=True) / denom
            + 1e-5
        )
        e_mu = 0.65 * fg_w * fg_e_mu + (1.0 - 0.65 * fg_w) * self.her_mean[1]
        e_sd = 0.65 * fg_w * fg_e_sd + (1.0 - 0.65 * fg_w) * self.her_std[1] + 1e-4

        ch_h = ((h_pos - h_mu) / h_sd).clamp(-6.0, 6.0)
        ch_e = ((e_pos - e_mu) / e_sd).clamp(-6.0, 6.0)
        ch_frac = (2.0 * (h_pos / (h_pos + e_pos + 0.03)) - 1.0).clamp(-1.0, 1.0)

        h_pad = F.pad(ch_h, (2, 2, 2, 2), mode="reflect")
        loc_mu = F.avg_pool2d(h_pad, kernel_size=5, stride=1)
        loc_var = (
            F.avg_pool2d(h_pad.square(), kernel_size=5, stride=1) - loc_mu.square()
        ).clamp(min=0.0)
        r_mag = (r_val - self.her_mean[2]).abs() / (self.her_std[2] + 1e-4)
        pigment_gate = torch.exp(-0.5 * F.relu(r_mag - 1.5))
        ch_contrast = (
            ((ch_h - loc_mu) / torch.sqrt(loc_var + 0.04)) * pigment_gate
        ).clamp(-6.0, 6.0)

        stain_4ch = torch.cat([ch_h, ch_e, ch_frac, ch_contrast], dim=1)
        rgb_norm = torch.nan_to_num(rgb_norm, nan=0.0, posinf=6.0, neginf=-6.0)
        stain_4ch = torch.nan_to_num(stain_4ch, nan=0.0, posinf=6.0, neginf=-6.0)
        return rgb_norm, stain_4ch

    def forward(
        self,
        rgb: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mixup_p: float = 0.0,
        return_center_idx: bool = False,
    ):
        if rgb.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward_fp32(
                    rgb, center_idx, labels, mixup_p, return_center_idx
                )
        return self._forward_fp32(rgb, center_idx, labels, mixup_p, return_center_idx)

    def _forward_fp32(
        self,
        rgb: torch.Tensor,
        center_idx: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        mixup_p: float,
        return_center_idx: bool,
    ):
        x01 = self._to_unit_rgb(rgb)
        eff_center = center_idx
        if self.training:
            x01 = self._augment_rgb01(x01, center_idx=center_idx)
            if (
                mixup_p > 0.0
                and (labels is not None)
                and (center_idx is not None)
            ):
                x01, eff_center = self.apply_intraclass_crosscenter_mixup(
                    x01, labels, center_idx, mix_prob=mixup_p
                )
        rgb_norm, stain_4ch = self.compute_representations(x01)
        if return_center_idx:
            return rgb_norm, stain_4ch, eff_center
        return rgb_norm, stain_4ch


class HistopathologyMultiViewDataset(Dataset):
    """
    Lightweight fast uint8 Dataset yielding raw (3, 256, 256) RGB tensors plus metadata
    for multi-worker DataLoader prefetching into GPU HistopathologyPhysicsFrontend.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_stats: Optional[dict] = None,
        is_train: bool = False,
        center_to_idx: Optional[dict] = None,
    ):
        self.sample_ids = dataframe["sample_id"].astype(str).tolist()
        self.image_paths = dataframe["image_path"].astype(str).tolist()
        self.image_stats = image_stats
        self.is_train = is_train

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

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        rgb_u8 = load_rgb_patch(self.image_paths[idx])
        rgb_chw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1), dtype=np.uint8)
        item = {
            "rgb": torch.from_numpy(rgb_chw),
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
    sample_weights: Optional[np.ndarray] = None,
    batch_size: int = 48,
    num_workers: int = 8,
):
    """Construct fast multi-worker uint8 DataLoaders with persistent workers & balanced sampling."""
    nw = max(2, min(num_workers, os.cpu_count() or 4))
    loader_kw = {
        "num_workers": nw,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": nw > 0,
    }
    if nw > 0:
        loader_kw["prefetch_factor"] = 2

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
            drop_last=True,
            **loader_kw,
        )
    else:
        train_loader = DataLoader(
            train_dataset_obj,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kw,
        )

    val_loader = DataLoader(
        val_dataset_obj,
        batch_size=batch_size * 2,
        shuffle=False,
        **loader_kw,
    )
    test_loader = DataLoader(
        test_dataset_obj,
        batch_size=batch_size * 2,
        shuffle=False,
        **loader_kw,
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

# Checksum over train sample_ids to cache train-only H/E/R anchors & rfft2 Fourier bank
_id_concat = "|".join(train_df["sample_id"].tolist())
_cache_sig = hashlib.md5(
    (_id_concat + f"_rfft{FOURIER_LOW_RADIUS}_b{FOURIER_BANK_SIZE}_v3").encode("utf-8")
).hexdigest()[:12]
CACHE_PATH = os.path.join(WORKING_DIR, f"histopath_physics_stats_{_cache_sig}.npz")

if os.path.exists(CACHE_PATH):
    cached = np.load(CACHE_PATH)
    train_image_stats = {
        "rgb_mean": cached["rgb_mean"],
        "rgb_std": cached["rgb_std"],
        "her_mean": cached["her_mean"],
        "her_std": cached["her_std"],
        "fourier_bank": cached["fourier_bank"],
        "fourier_bank_centers": cached["fourier_bank_centers"],
    }
else:
    train_image_stats = fit_train_only_image_statistics_and_fourier_bank(
        train_df, seed=SEED, center_to_idx=CENTER_TO_IDX
    )
    np.savez_compressed(
        CACHE_PATH,
        rgb_mean=train_image_stats["rgb_mean"],
        rgb_std=train_image_stats["rgb_std"],
        her_mean=train_image_stats["her_mean"],
        her_std=train_image_stats["her_std"],
        fourier_bank=train_image_stats["fourier_bank"],
        fourier_bank_centers=train_image_stats["fourier_bank_centers"],
    )

# Seed full CPU + CUDA RNG state
seed_everything(SEED, seed_cuda=True)

y_train = train_df["label"].values.astype(np.int64)
y_val = val_df["label"].values.astype(np.int64)
y_merged = merged_dev_df["label"].values.astype(np.int64)

train_sampling_weights = compute_balanced_domain_class_weights(train_df)

# Instantiate Fast uint8 Datasets & GPU Physics Frontend; Verify Deterministic Consistency
train_dataset = HistopathologyMultiViewDataset(
    train_df, train_image_stats, is_train=True, center_to_idx=CENTER_TO_IDX
)
val_dataset = HistopathologyMultiViewDataset(
    val_df, train_image_stats, is_train=False, center_to_idx=None
)
test_dataset = HistopathologyMultiViewDataset(
    test_df, train_image_stats, is_train=False, center_to_idx=None
)
primary_physics_frontend = HistopathologyPhysicsFrontend(train_image_stats).to(DEVICE)

_s_tr = train_dataset[0]
_s_va1 = val_dataset[0]
_s_va2 = val_dataset[0]
_s_te = test_dataset[0]

assert _s_tr["rgb"].shape == (3, IMG_SIZE, IMG_SIZE) and _s_tr["rgb"].dtype == torch.uint8
assert _s_va1["rgb"].shape == (3, IMG_SIZE, IMG_SIZE) and _s_te["rgb"].shape == (
    3,
    IMG_SIZE,
    IMG_SIZE,
)
assert torch.equal(_s_va1["rgb"], _s_va2["rgb"]), "Dataset reads must be deterministic!"

with torch.no_grad():
    primary_physics_frontend.eval()
    _va_b = _s_va1["rgb"].unsqueeze(0).to(DEVICE)
    _rgb_n1, _st_4c1 = primary_physics_frontend(_va_b)
    _rgb_n2, _st_4c2 = primary_physics_frontend(_va_b)
    assert _rgb_n1.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert _st_4c1.shape == (1, 4, IMG_SIZE, IMG_SIZE)
    assert torch.isfinite(_rgb_n1).all() and torch.isfinite(_st_4c1).all()
    assert torch.equal(_rgb_n1, _rgb_n2) and torch.equal(
        _st_4c1, _st_4c2
    ), "GPU physics frontend eval representation must be 100% deterministic!"

print(
    f"Data & Physics Frontend Ready | "
    f"train={len(train_df)} (pos={y_train.mean():.3f}), "
    f"val={len(val_df)} (pos={y_val.mean():.3f}), "
    f"merged_dev={len(merged_dev_df)}, test={len(test_df)} | "
    f"stain_channels=4, fourier_bank={train_image_stats['fourier_bank'].shape} ({_elapsed_sec():.1f}s)"
)


# =============================================================================
# 8. Step 2: Pretrained Backbone Documentation & Physics/Spectral Modules
# =============================================================================
# Explicit Pretrained Model Documentation (per competition rules):
# 1) convnext_small.fb_in22k_ft_in1k_384 (ConvNeXt-Small, LayerNorm2d, 7x7 depthwise convs)
#    - Pretraining Corpus: ImageNet-22K -> ImageNet-1K 384x384 supervised natural images
# 2) swinv2_tiny_window8_256 (Microsoft Swin Transformer V2 Tiny, native 256x256, window=8)
#    - Pretraining Corpus: ImageNet-1K (ms_in1k supervised natural images)
# - Library: timm (version logged at runtime)
# - Known Histopathology / Camelyon Overlap: None (natural-image pretraining only)
MODEL_METADATA = {
    "architecture": "PathologyMultiScaleStainNet",
    "pretrained_backbone": "convnext_small.fb_in22k_ft_in1k_384 + swinv2_tiny_window8_256",
    "pretrained_backbones": [
        "convnext_small.fb_in22k_ft_in1k_384",
        "swinv2_tiny_window8_256",
    ],
    "timm_version": getattr(timm, "__version__", "unknown"),
    "pretraining_corpus": "ImageNet-22K / ImageNet-1K natural images",
    "known_target_overlap": "None (natural images only; zero histopathology overlap)",
}

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


class ExtrapolativeMixStyle2d(nn.Module):
    """
    Training-only instance-level feature mean/std style extrapolation (lambda in [-0.2, 1.2])
    at early stages (Stages 0-1) to synthesize virtual stain/scanner domains outside the
    convex hull of the two development centers. Strictly no-op at eval/inference.
    """

    def __init__(self, p: float = 0.50, eps: float = 1e-5):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(
        self, x: torch.Tensor, center_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if (
            (not self.training)
            or x.size(0) < 2
            or float(torch.rand((), device=x.device)) > self.p
        ):
            return x
        orig_dtype = x.dtype
        xf = x.float()
        B = xf.size(0)
        mu = xf.mean(dim=(-2, -1), keepdim=True)
        var = xf.var(dim=(-2, -1), unbiased=False, keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (xf - mu) / sig

        if center_idx is not None and int((center_idx >= 0).sum()) >= 2:
            diff_c = (
                (center_idx.unsqueeze(1) != center_idx.unsqueeze(0))
                & (center_idx.unsqueeze(1) >= 0)
                & (center_idx.unsqueeze(0) >= 0)
            ).float()
            scores = torch.rand(B, B, device=x.device) + 0.75 * diff_c
            scores.fill_diagonal_(-1.0)
            perm = scores.argmax(dim=1)
        else:
            perm = torch.randperm(B, device=x.device)

        lam = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.2, 1.2)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = (lam * sig + (1.0 - lam) * sig[perm]).clamp(min=1e-3)
        return (x_norm * sig_mix + mu_mix).to(orig_dtype)


class StainSpectralPyramidEncoder(nn.Module):
    """
    4-scale (64x64, 32x32, 16x16, 8x8) BatchNorm-free Zero-Phase 2D-FFT Spectral-Stain
    Encoder operating on the 4-channel scale-invariant stain-structural representation
    [standardized H, standardized E, chromatic nuclear fraction H/(H+E+eps), and
    pigment-gated local 5x5 chromatin contrast].
    """

    def __init__(self, in_channels: int = 4, dropout_p: float = 0.10):
        super().__init__()
        self.stem_128 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.stage_64 = StainDilatedResidualBlock(
            48, 64, stride=2, spec_hw=(64, 64), dropout_p=dropout_p
        )
        self.stage_32 = StainDilatedResidualBlock(
            64, 96, stride=2, spec_hw=(32, 32), dropout_p=dropout_p
        )
        self.stage_16 = StainDilatedResidualBlock(
            96, 144, stride=2, spec_hw=(16, 16), dropout_p=dropout_p
        )
        self.stage_8 = StainDilatedResidualBlock(
            144, 192, stride=2, spec_hw=(8, 8), dropout_p=dropout_p
        )

        self.chromatin_gate_32 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 96, kernel_size=1),
            nn.Sigmoid(),
        )
        self.chromatin_gate_16 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 144, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, stain: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        struct_cues = stain[:, 2:4, :, :]  # [nuclear_fraction, pigment_gated_contrast]
        x128 = self.stem_128(stain)  # (B, 48, 128, 128)
        f64 = self.stage_64(x128)  # (B, 64, 64, 64)
        f32 = self.stage_32(f64)  # (B, 96, 32, 32)
        cues_32 = F.adaptive_avg_pool2d(struct_cues, (32, 32))
        f32 = f32 * (0.5 + 0.5 * self.chromatin_gate_32(cues_32))

        f16 = self.stage_16(f32)  # (B, 144, 16, 16)
        cues_16 = F.adaptive_avg_pool2d(struct_cues, (16, 16))
        f16 = f16 * (0.5 + 0.5 * self.chromatin_gate_16(cues_16))

        f8 = self.stage_8(f16)  # (B, 192, 8, 8)
        return f64, f32, f16, f8


class GeMPool2d(nn.Module):
    """D4-invariant Generalized Mean Pooling executed in float32 for numerical stability."""

    def __init__(self, init_p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.gem_p = nn.Parameter(torch.tensor(float(init_p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = F.relu(x.float()).clamp(min=self.eps)
        p = self.gem_p.float().clamp(1.0, 6.0)
        return xf.pow(p).mean(dim=(-2, -1)).pow(1.0 / p).to(orig_dtype)


class D4ConcentricRingPool2d(nn.Module):
    """
    Aggregates spatial feature maps over exact D4-symmetric concentric square rings:
      - Core (pixels 96:160 -> [3/8, 5/8]): central micrometastasis ROI
      - Mid  (pixels 32:224 \\ 96:160 -> [1/8, 7/8] \\ [3/8, 5/8]): perinodal zone
      - Outer (complement of [1/8, 7/8]): peripheral lymphoid parenchyma
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        c_lo_h, c_hi_h = (3 * height) // 8, (5 * height) // 8
        c_lo_w, c_hi_w = (3 * width) // 8, (5 * width) // 8
        m_lo_h, m_hi_h = height // 8, (7 * height) // 8
        m_lo_w, m_hi_w = width // 8, (7 * width) // 8

        core_m = torch.zeros(1, 1, height, width, dtype=torch.float32)
        core_m[:, :, c_lo_h:c_hi_h, c_lo_w:c_hi_w] = 1.0

        inner_m = torch.zeros(1, 1, height, width, dtype=torch.float32)
        inner_m[:, :, m_lo_h:m_hi_h, m_lo_w:m_hi_w] = 1.0

        mid_m = inner_m - core_m
        outer_m = 1.0 - inner_m

        self.core_slice = (c_lo_h, c_hi_h, c_lo_w, c_hi_w)
        self.register_buffer(
            "core_norm", core_m / core_m.sum().clamp(min=1.0), persistent=False
        )
        self.register_buffer(
            "mid_norm", mid_m / mid_m.sum().clamp(min=1.0), persistent=False
        )
        self.register_buffer(
            "outer_norm", outer_m / outer_m.sum().clamp(min=1.0), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        core_mean = (xf * self.core_norm).sum(dim=(-2, -1))
        lo_h, hi_h, lo_w, hi_w = self.core_slice
        core_max = xf[:, :, lo_h:hi_h, lo_w:hi_w].amax(dim=(-2, -1))
        core_repr = 0.5 * (core_mean + core_max)
        mid_mean = (xf * self.mid_norm).sum(dim=(-2, -1))
        outer_mean = (xf * self.outer_norm).sum(dim=(-2, -1))
        return torch.cat(
            [core_repr, mid_mean, outer_mean, core_repr - outer_mean], dim=-1
        ).to(x.dtype)


class GatedAttentionMIL2d(nn.Module):
    """
    D4-invariant temperature-scaled Gated-Attention Multiple Instance Learning readout
    with symmetric radial-concentric spatial prior for focal micrometastasis detection.
    """

    def __init__(
        self,
        in_dim: int,
        grid_size: int,
        hidden_dim: int = 128,
        init_temp: float = 2.5,
    ):
        super().__init__()
        self.attn_v = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attn_u = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.mil_temp = nn.Parameter(torch.tensor(float(init_temp)))

        coords = (
            torch.arange(grid_size, dtype=torch.float32) + 0.5
        ) / float(grid_size) * 2.0 - 1.0
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        r_euc = torch.sqrt(gx.square() + gy.square()) / math.sqrt(2.0)
        r_cheb = torch.maximum(gx.abs(), gy.abs())
        in_core = (r_cheb <= 0.25).float()
        in_mid = ((r_cheb > 0.25) & (r_cheb <= 0.75)).float()
        rad_basis = torch.stack([r_euc, r_cheb, in_core, in_mid], dim=-1).view(
            1, grid_size * grid_size, 4
        )
        self.register_buffer("radial_basis", rad_basis, persistent=False)
        self.radial_prior = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        orig_dtype = feat_map.dtype
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        gated = self.attn_v(tokens) * self.attn_u(tokens)
        scores = self.attn_w(gated).squeeze(-1).float()
        rad_bias = self.radial_prior(self.radial_basis).squeeze(-1).float()
        temp = self.mil_temp.float().clamp(0.5, 8.0)
        weights = F.softmax(temp * (scores + rad_bias), dim=-1).to(orig_dtype)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)


def _create_pathology_backbone(
    backbone_name: str, pretrained: bool = True
) -> Tuple[nn.Module, List[int], str]:
    """Instantiate BatchNorm-free 4-scale ConvNeXt-Small or SwinV2-Tiny backbone."""
    if "convnext" in backbone_name.lower():
        candidates = [
            (backbone_name, {}),
            ("convnext_small.fb_in22k_ft_in1k_384", {}),
            ("convnext_small.fb_in22k_ft_in1k", {}),
            ("convnext_small", {}),
        ]
        drop_path = 0.25
    elif "swin" in backbone_name.lower():
        candidates = [
            (backbone_name, {}),
            ("swinv2_tiny_window8_256", {}),
            ("swin_tiny_patch4_window7_224", {"img_size": IMG_SIZE}),
        ]
        drop_path = 0.20
    else:
        candidates = [(backbone_name, {})]
        drop_path = 0.20

    seen = set()
    last_err = None
    for cand_name, extra_kw in candidates:
        if cand_name in seen:
            continue
        seen.add(cand_name)
        try:
            backbone = timm.create_model(
                cand_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                drop_path_rate=drop_path,
                **extra_kw,
            )
            channels = list(backbone.feature_info.channels())
            return backbone, channels, cand_name
        except Exception as err:
            last_err = err
    raise RuntimeError(f"Failed to create backbone {backbone_name}: {last_err}")


def _ensure_nchw(
    feat: torch.Tensor, expected_c: int, expected_hw: Optional[int] = None
) -> torch.Tensor:
    """Convert NHWC -> NCHW safely across all timm versions and guard spatial size."""
    if feat.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got {feat.shape}")
    if feat.shape[1] == expected_c and feat.shape[-1] != expected_c:
        out = feat
    elif feat.shape[-1] == expected_c:
        out = feat.permute(0, 3, 1, 2).contiguous()
    else:
        out = feat
    if expected_hw is not None and (
        out.shape[-2] != expected_hw or out.shape[-1] != expected_hw
    ):
        out = F.interpolate(
            out, size=(expected_hw, expected_hw), mode="bilinear", align_corners=False
        )
    return out


class PathologyMultiScaleStainNet(nn.Module):
    """
    Unified BatchNorm-free 4-Scale (64x64, 32x32, 16x16, 8x8) Visual-Stain Architecture
    supporting both ConvNeXt-Small and SwinV2-Tiny:
      1) 4-stage RGB backbone (stride 4..32) + early-stage ExtrapolativeMixStyle2d.
      2) 4-scale Zero-Phase 2D-FFT StainSpectralPyramidEncoder on 4-ch stain-structural maps.
      3) Bottom-up + top-down multi-scale feature pyramid fusion.
      4) D4-symmetric Concentric Rings (core 96:160, mid, outer) + GeM + Gated-Attention MIL.
      5) Full-representation Class-Conditional CORAL + GRL domain alignment.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small.fb_in22k_ft_in1k_384",
        d_model: int = 256,
        num_train_centers: int = _NUM_TRAIN_CENTERS,
        pretrained: bool = True,
        dropout_p: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_train_centers = num_train_centers
        self.dropout_p = dropout_p

        self.rgb_backbone, self.bb_channels, self.backbone_name = (
            _create_pathology_backbone(backbone_name, pretrained=pretrained)
        )
        c0, c1, c2, c3 = self.bb_channels  # [96, 192, 384, 768]
        d_fine = d_model // 2  # 128 at 64x64

        self.mixstyle_0 = ExtrapolativeMixStyle2d(p=0.50)
        self.mixstyle_1 = ExtrapolativeMixStyle2d(p=0.50)
        self.stain_encoder = StainSpectralPyramidEncoder(in_channels=4, dropout_p=0.10)

        self.fuse_64 = nn.Sequential(
            nn.Conv2d(c0 + 64, d_fine, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_fine),
            nn.GELU(),
        )
        self.down_64_to_32 = nn.Sequential(
            nn.Conv2d(d_fine, d_fine, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, d_fine),
            nn.GELU(),
        )

        self.fuse_32 = nn.Sequential(
            nn.Conv2d(c1 + 96 + d_fine, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )
        self.down_32_to_16 = nn.Sequential(
            nn.Conv2d(d_model, d_fine, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, d_fine),
            nn.GELU(),
        )

        self.fuse_16 = nn.Sequential(
            nn.Conv2d(c2 + 144 + d_fine, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )
        self.down_16_to_8 = nn.Sequential(
            nn.Conv2d(d_model, d_fine, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, d_fine),
            nn.GELU(),
        )

        self.fuse_8 = nn.Sequential(
            nn.Conv2d(c3 + 192 + d_fine, d_model, kernel_size=1, bias=False),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )

        self.smooth_16 = nn.Sequential(
            nn.Conv2d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
            ),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )
        self.smooth_32 = nn.Sequential(
            nn.Conv2d(
                d_model, d_model, kernel_size=3, padding=1, groups=d_model, bias=False
            ),
            nn.GroupNorm(16, d_model),
            nn.GELU(),
        )

        self.ring_pool_64 = D4ConcentricRingPool2d(64, 64)
        self.ring_pool_32 = D4ConcentricRingPool2d(32, 32)
        self.ring_pool_16 = D4ConcentricRingPool2d(16, 16)
        concentric_in_dim = 4 * (d_fine + d_model + d_model)
        self.concentric_proj = nn.Sequential(
            nn.LayerNorm(concentric_in_dim),
            nn.Linear(concentric_in_dim, d_model),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.LayerNorm(d_model),
        )

        self.mil_32 = GatedAttentionMIL2d(
            d_model, grid_size=32, hidden_dim=128, init_temp=2.5
        )
        self.mil_16 = GatedAttentionMIL2d(
            d_model, grid_size=16, hidden_dim=128, init_temp=3.0
        )

        self.gem_core_64 = GeMPool2d(init_p=3.0)
        self.gem_16 = GeMPool2d(init_p=3.0)
        self.gem_8 = GeMPool2d(init_p=2.5)
        self.gem_proj = nn.Sequential(
            nn.LayerNorm(d_fine + d_model * 2),
            nn.Linear(d_fine + d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.LayerNorm(d_model),
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
        nn.init.zeros_(self.aux_vis_head[-1].bias)

        self.domain_discriminator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(d_model // 2, max(2, num_train_centers)),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        stain: torch.Tensor,
        grl_lambda: float = 0.25,
        center_idx: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        s_feats = self.rgb_backbone(rgb)
        s0 = _ensure_nchw(s_feats[0], self.bb_channels[0], 64)
        s1 = _ensure_nchw(s_feats[1], self.bb_channels[1], 32)
        s2 = _ensure_nchw(s_feats[2], self.bb_channels[2], 16)
        s3 = _ensure_nchw(s_feats[3], self.bb_channels[3], 8)

        s0 = self.mixstyle_0(s0, center_idx=center_idx)
        s1 = self.mixstyle_1(s1, center_idx=center_idx)

        st64, st32, st16, st8 = self.stain_encoder(stain)

        m64 = self.fuse_64(torch.cat([s0, st64], dim=1))  # (B, 128, 64, 64)
        m32 = self.fuse_32(
            torch.cat([s1, st32, self.down_64_to_32(m64)], dim=1)
        )  # (B, 256, 32, 32)
        m16 = self.fuse_16(
            torch.cat([s2, st16, self.down_32_to_16(m32)], dim=1)
        )  # (B, 256, 16, 16)
        m8 = self.fuse_8(
            torch.cat([s3, st8, self.down_16_to_8(m16)], dim=1)
        )  # (B, 256, 8, 8)

        p16 = self.smooth_16(m16 + F.interpolate(m8, size=(16, 16), mode="nearest"))
        p32 = self.smooth_32(m32 + F.interpolate(p16, size=(32, 32), mode="nearest"))

        rings_cat = torch.cat(
            [
                self.ring_pool_64(m64),
                self.ring_pool_32(p32),
                self.ring_pool_16(p16),
            ],
            dim=-1,
        )
        concentric_vec = self.concentric_proj(rings_cat)  # (B, D)
        mil_fine_vec = self.mil_32(p32)  # (B, D)
        mil_sem_vec = self.mil_16(p16)  # (B, D)

        gem_cat = torch.cat(
            [
                self.gem_core_64(m64[:, :, 24:40, 24:40]),
                self.gem_16(p16),
                self.gem_8(m8),
            ],
            dim=-1,
        )
        gem_vec = self.gem_proj(gem_cat)  # (B, D)

        fused_feat = self.bottleneck(
            torch.cat([concentric_vec, mil_fine_vec, mil_sem_vec, gem_vec], dim=-1)
        )  # (B, D)

        logits = (
            torch.stack(
                [self.classifier(drop(fused_feat)) for drop in self.ms_dropouts], dim=0
            )
            .mean(dim=0)
            .squeeze(-1)
        )

        if not return_aux:
            return logits

        aux_vis_logits = self.aux_vis_head(
            torch.cat([concentric_vec, mil_fine_vec], dim=-1)
        ).squeeze(-1)
        center_logits = self.domain_discriminator(
            gradient_reverse(fused_feat, grl_lambda)
        )

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "aux_vis_logits": aux_vis_logits,
            "fused_feat": F.normalize(fused_feat.float(), p=2, dim=-1),
            "center_logits": center_logits,
        }

    @torch.no_grad()
    def predict_proba(self, rgb: torch.Tensor, stain: torch.Tensor) -> torch.Tensor:
        """Deterministic tumor probability inference in [0, 1] for validation & test."""
        logits = self.forward(rgb, stain, return_aux=False)
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
        align_losses = []

        for cls_val in (0.0, 1.0):
            cls_mask = (targets - cls_val).abs() < 0.25
            f0 = fused_feat[cls_mask & (center_idx == c0)]
            f1 = fused_feat[cls_mask & (center_idx == c1)]
            if f0.size(0) >= 3 and f1.size(0) >= 3:
                mu0, mu1 = f0.mean(dim=0), f1.mean(dim=0)
                mean_diff = (mu0 - mu1).square().sum()

                z0 = f0 - mu0.unsqueeze(0)
                z1 = f1 - mu1.unsqueeze(0)
                cov0 = (z0.T @ z0) / float(f0.size(0) - 1)
                cov1 = (z1.T @ z1) / float(f1.size(0) - 1)
                cov_diff = 0.5 * (cov0 - cov1).square().sum()
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


def _infer_backbone_stage(param_name: str) -> int:
    """Map ConvNeXt or SwinV2 backbone parameter names to stage index 0..3."""
    sub = (
        param_name[len("rgb_backbone.") :]
        if param_name.startswith("rgb_backbone.")
        else param_name
    )
    if "stem" in sub or "patch_embed" in sub:
        return 0
    for s_idx in (0, 1, 2, 3):
        if (
            f"stages_{s_idx}." in sub
            or f"stages.{s_idx}." in sub
            or f"layers_{s_idx}." in sub
            or f"layers.{s_idx}." in sub
        ):
            return s_idx
    return 3


def build_differential_adamw(
    model_obj: nn.Module,
    backbone_lr: float = 8.5e-5,
    head_lr: float = 4.0e-4,
    layer_decay: float = 0.75,
    backbone_wd: float = 0.03,
    head_wd: float = 0.02,
) -> AdamW:
    """
    Constructs AdamW with 0.75 stage-wise LR decay across backbone stages (0..3) and
    strict zero weight decay on biases, norms, LayerScale gammas, cosine attention scales,
    relative position tables, GeM exponents, MIL temperatures, and spectral gates.
    """
    no_decay_keywords = (
        "bias",
        "norm",
        "gamma",
        "logit_scale",
        "cpb_mlp",
        "relative_position",
        "freq_gain_logits",
        "gem_p",
        "mil_temp",
    )

    bb_stage_groups: Dict[Tuple[int, bool], List[nn.Parameter]] = {
        (s, nd): [] for s in (0, 1, 2, 3) for nd in (False, True)
    }
    head_decay, head_no_decay = [], []

    for name, param in model_obj.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.ndim <= 1) or any(
            kw in name.lower() for kw in no_decay_keywords
        )
        if name.startswith("rgb_backbone."):
            s_idx = _infer_backbone_stage(name)
            bb_stage_groups[(s_idx, is_no_decay)].append(param)
        elif not is_no_decay:
            head_decay.append(param)
        else:
            head_no_decay.append(param)

    param_groups = []
    for s_idx in (0, 1, 2, 3):
        stage_lr = backbone_lr * (layer_decay ** float(3 - s_idx))
        if bb_stage_groups[(s_idx, False)]:
            param_groups.append(
                {
                    "params": bb_stage_groups[(s_idx, False)],
                    "lr": stage_lr,
                    "weight_decay": backbone_wd,
                    "name": f"bb_s{s_idx}_decay",
                }
            )
        if bb_stage_groups[(s_idx, True)]:
            param_groups.append(
                {
                    "params": bb_stage_groups[(s_idx, True)],
                    "lr": stage_lr,
                    "weight_decay": 0.0,
                    "name": f"bb_s{s_idx}_no_decay",
                }
            )

    if head_decay:
        param_groups.append(
            {
                "params": head_decay,
                "lr": head_lr,
                "weight_decay": head_wd,
                "name": "head_decay",
            }
        )
    if head_no_decay:
        param_groups.append(
            {
                "params": head_no_decay,
                "lr": head_lr,
                "weight_decay": 0.0,
                "name": "head_no_decay",
            }
        )

    return AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)


def build_model_criterion_optimizer(
    backbone_name: str = "convnext_small.fb_in22k_ft_in1k_384",
    num_train_centers: int = _NUM_TRAIN_CENTERS,
    pretrained: bool = True,
    backbone_lr: float = 8.5e-5,
    head_lr: float = 4.0e-4,
    layer_decay: float = 0.75,
    device: torch.device = DEVICE,
) -> Tuple[PathologyMultiScaleStainNet, UnseenCenterBalancedAccuracyLoss, AdamW]:
    """Clean factory to instantiate fresh (model, criterion, optimizer) for CV/refit folds."""
    net = PathologyMultiScaleStainNet(
        backbone_name=backbone_name,
        d_model=256,
        num_train_centers=num_train_centers,
        pretrained=pretrained,
        dropout_p=0.25,
    ).to(device)
    loss_fn = UnseenCenterBalancedAccuracyLoss().to(device)
    opt = build_differential_adamw(
        net,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        layer_decay=layer_decay,
    )
    return net, loss_fn, opt


# Instantiate Primary Model (ConvNeXt-Small), Criterion, Optimizer & Verify Both Architectures
model, criterion, optimizer = build_model_criterion_optimizer(
    backbone_name="convnext_small.fb_in22k_ft_in1k_384",
    num_train_centers=_NUM_TRAIN_CENTERS,
    pretrained=True,
    backbone_lr=8.5e-5,
    head_lr=4.0e-4,
    layer_decay=0.75,
    device=DEVICE,
)

with torch.no_grad():
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        for m in model.modules()
    ), "PathologyMultiScaleStainNet (ConvNeXt) must be 100% BatchNorm-free!"

    model.eval()
    _b = 4
    _dummy_rgb = torch.randn(_b, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _dummy_stain = torch.randn(_b, 4, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _dummy_y = torch.tensor([0.0, 1.0, 0.0, 1.0], device=DEVICE)
    _dummy_cid = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)

    _probs1 = model.predict_proba(_dummy_rgb, _dummy_stain)
    _probs2 = model.predict_proba(_dummy_rgb, _dummy_stain)
    _aux_out = model(
        _dummy_rgb,
        _dummy_stain,
        grl_lambda=0.25,
        center_idx=_dummy_cid,
        return_aux=True,
    )
    _loss_val = criterion(_aux_out, _dummy_y, center_idx=_dummy_cid)
    _metric_check = binary_metrics(
        _dummy_y.cpu().numpy().astype(int), _probs1.cpu().numpy()
    )

    assert _probs1.shape == (_b,) and torch.isfinite(_probs1).all()
    assert torch.equal(_probs1, _probs2), "Eval inference must be deterministic!"
    assert ((_probs1 >= 0.0) & (_probs1 <= 1.0)).all()
    assert torch.isfinite(_loss_val), "Loss produced non-finite value!"
    assert "balanced_accuracy" in _metric_check and "auroc" in _metric_check

    # Verify SwinV2-Tiny 4-scale forward & BatchNorm-free contract
    _swin_check = PathologyMultiScaleStainNet(
        backbone_name="swinv2_tiny_window8_256",
        d_model=256,
        num_train_centers=_NUM_TRAIN_CENTERS,
        pretrained=False,
    ).to(DEVICE)
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        for m in _swin_check.modules()
    ), "PathologyMultiScaleStainNet (SwinV2) must be 100% BatchNorm-free!"
    _swin_check.eval()
    _swin_p = _swin_check.predict_proba(_dummy_rgb, _dummy_stain)
    assert _swin_p.shape == (_b,) and torch.isfinite(_swin_p).all()
    del _swin_check

_total_params = sum(p.numel() for p in model.parameters())
_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(
    f"Model Design Complete | arch={MODEL_METADATA['architecture']} "
    f"({MODEL_METADATA['pretrained_backbone']}, timm={MODEL_METADATA['timm_version']}) | "
    f"primary_params={_trainable_params:,}/{_total_params:,} | device={DEVICE}"
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
    Maintains an exponential moving average of PathologyMultiScaleStainNet weights.
    Because the architecture uses exclusively LayerNorm and GroupNorm (zero BatchNorm),
    weight-space EMA has zero running-statistic mismatch on unseen medical centers.
    """

    def __init__(self, source_model: PathologyMultiScaleStainNet, max_decay: float = 0.996):
        self.max_decay = max_decay
        self.num_updates = 0
        self.ema_model = PathologyMultiScaleStainNet(
            backbone_name=source_model.backbone_name,
            d_model=source_model.d_model,
            num_train_centers=source_model.num_train_centers,
            pretrained=False,
            dropout_p=source_model.dropout_p,
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


@torch.no_grad()
def predict_d4_tta(
    models: List[nn.Module],
    frontend: HistopathologyPhysicsFrontend,
    loader: DataLoader,
    expected_ids: List[str],
    tta_views: int = TTA_VIEWS,
) -> np.ndarray:
    """
    Executes deterministic multi-view D4 dihedral Test-Time Augmentation directly on GPU
    through HistopathologyPhysicsFrontend + PathologyMultiScaleStainNet.
    Identical logic is used for validation and test sets.
    """
    frontend.eval()
    for m in models:
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
        rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
        collected_ids.extend(batch["sample_id"])

        batch_prob_sum = torch.zeros(rgb_u8.size(0), dtype=torch.float32, device=DEVICE)
        n_total_passes = 0

        for do_flip, k_rot in d4_ops:
            raw_v = torch.flip(rgb_u8, dims=[-1]) if do_flip else rgb_u8
            if k_rot > 0:
                raw_v = torch.rot90(raw_v, k=k_rot, dims=[-2, -1])

            r_v, s_v = frontend(raw_v)

            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    for m in models:
                        p = m.predict_proba(r_v, s_v)
                        batch_prob_sum += p.float()
                        n_total_passes += 1
            else:
                for m in models:
                    p = m.predict_proba(r_v, s_v)
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
    models_with_frontends: List[Tuple[nn.Module, HistopathologyPhysicsFrontend]],
    eval_loader: DataLoader,
    expected_ids: List[str],
    tta_views: int = TTA_VIEWS,
) -> np.ndarray:
    """
    Single unified prediction pipeline shared identically between pre-refit validation
    evaluation and final test inference.
    """
    if not models_with_frontends:
        raise ValueError("models_with_frontends must be non-empty")
    pred_list = []
    for net, fe in models_with_frontends:
        pred_list.append(
            predict_d4_tta(
                [net],
                fe,
                eval_loader,
                expected_ids=expected_ids,
                tta_views=tta_views,
            )
        )
    probs = np.mean(pred_list, axis=0)
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


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


def train_pathology_checkpoint(
    net: PathologyMultiScaleStainNet,
    frontend: HistopathologyPhysicsFrontend,
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
) -> Tuple[PathologyMultiScaleStainNet, Dict[str, float]]:
    """
    Trains PathologyMultiScaleStainNet with GPU HistopathologyPhysicsFrontend, progressive
    GRL schedule, gradient clipping, WarmupGatedEMA, and official metric checkpointing
    (balanced_accuracy at 0.5 threshold via ./input/metric.py).
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
        frontend.train()
        running_loss = 0.0
        n_batches = 0

        for batch in tr_loader:
            progress = float(global_step) / float(total_steps)
            _set_differential_step_lr(opt, base_lrs, group_names, progress)

            grl_lam = 0.30 * (2.0 / (1.0 + math.exp(-5.0 * progress)) - 1.0)

            rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            center_idx = batch["center_idx"].to(DEVICE, non_blocking=True)

            with torch.no_grad():
                rgb, stain, eff_center_idx = frontend(
                    rgb_u8,
                    center_idx=center_idx,
                    labels=labels,
                    mixup_p=0.30,
                    return_center_idx=True,
                )

            opt.zero_grad(set_to_none=True)

            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    outputs = net(
                        rgb,
                        stain,
                        grl_lambda=grl_lam,
                        center_idx=eff_center_idx,
                        return_aux=True,
                    )
                    loss = loss_fn(outputs, labels, center_idx=eff_center_idx)
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
                outputs = net(
                    rgb,
                    stain,
                    grl_lambda=grl_lam,
                    center_idx=eff_center_idx,
                    return_aux=True,
                )
                loss = loss_fn(outputs, labels, center_idx=eff_center_idx)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                opt.step()

            ema_helper.update(net, in_warmup=(progress < 0.08))
            running_loss += float(loss.detach().item())
            n_batches += 1
            global_step += 1

        avg_loss = running_loss / max(1, n_batches)

        live_probs = predict_d4_tta(
            [net], frontend, va_loader, expected_ids=va_ids, tta_views=1
        )
        ema_probs = predict_d4_tta(
            [ema_helper.ema_model], frontend, va_loader, expected_ids=va_ids, tta_views=1
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
    frontend.eval()
    return net, {"balanced_accuracy": best_ba, "auroc": best_auroc}


# =============================================================================
# 10. Stage 1: Leak-Free Training on train.csv & Official Validation
# =============================================================================
seed_everything(SEED, seed_cuda=True)

# Build DataLoaders using Step 1's exact train_dataset, val_dataset, test_dataset
train_loader, val_loader, test_loader = build_dataloaders(
    train_dataset_obj=train_dataset,
    val_dataset_obj=val_dataset,
    test_dataset_obj=test_dataset,
    sample_weights=train_sampling_weights,
    batch_size=BATCH_SIZE,
    num_workers=4,
)

val_ids_list = val_df["sample_id"].astype(str).tolist()
test_ids_list = test_df["sample_id"].astype(str).tolist()

# Stage 1A: Train primary Step-2 ConvNeXt-Small `model` on train_dataset ONLY
stage1_ckpt_path = os.path.join(WORKING_DIR, "best_stage1_convnext_small.pt")
t_s1 = time.perf_counter()
model, _ = train_pathology_checkpoint(
    net=model,
    frontend=primary_physics_frontend,
    loss_fn=criterion,
    opt=optimizer,
    tr_loader=train_loader,
    va_loader=val_loader,
    va_labels=y_val,
    va_ids=val_ids_list,
    num_epochs=STAGE1_EPOCHS,
    ckpt_path=stage1_ckpt_path,
    run_tag="Stage1-ConvNeXt-TrainOnly",
    patience=5,
)
stage1_train_duration = time.perf_counter() - t_s1

stage1_models_with_fe: List[Tuple[nn.Module, HistopathologyPhysicsFrontend]] = [
    (model, primary_physics_frontend)
]

# Stage 1B: Train complementary SwinV2-Tiny on train_dataset ONLY if runtime budget permits
if (MAX_RUNTIME_BUDGET_SEC - _elapsed_sec()) > (stage1_train_duration + 300.0):
    seed_everything(SEED + 17, seed_cuda=True)
    swin_model, swin_crit, swin_opt = build_model_criterion_optimizer(
        backbone_name="swinv2_tiny_window8_256",
        num_train_centers=_NUM_TRAIN_CENTERS,
        pretrained=True,
        backbone_lr=1.0e-4,
        head_lr=4.0e-4,
        layer_decay=0.75,
        device=DEVICE,
    )
    stage1_swin_ckpt = os.path.join(WORKING_DIR, "best_stage1_swinv2_tiny.pt")
    swin_model, _ = train_pathology_checkpoint(
        net=swin_model,
        frontend=primary_physics_frontend,
        loss_fn=swin_crit,
        opt=swin_opt,
        tr_loader=train_loader,
        va_loader=val_loader,
        va_labels=y_val,
        va_ids=val_ids_list,
        num_epochs=STAGE1_EPOCHS,
        ckpt_path=stage1_swin_ckpt,
        run_tag="Stage1-SwinV2-TrainOnly",
        patience=5,
    )
    stage1_models_with_fe.append((swin_model, primary_physics_frontend))
    del swin_crit, swin_opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Compute Official Pre-Refit Validation Score on untouched validation.csv via run_pipeline_inference
val_probs_official = run_pipeline_inference(
    models_with_frontends=stage1_models_with_fe,
    eval_loader=val_loader,
    expected_ids=val_ids_list,
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
    models_with_frontends=stage1_models_with_fe,
    eval_loader=test_loader,
    expected_ids=test_ids_list,
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
    nw_eval = max(2, min(4, os.cpu_count() or 2))
    refit_backbones = [
        "convnext_small.fb_in22k_ft_in1k_384",
        "swinv2_tiny_window8_256",
    ]
    refit_entries: List[Tuple[nn.Module, HistopathologyPhysicsFrontend]] = []

    for r_fold in range(max_refit_folds):
        if (MAX_RUNTIME_BUDGET_SEC - _elapsed_sec()) < (0.85 * est_fold_sec + 120.0):
            break

        seed_everything(SEED + 100 * (r_fold + 1), seed_cuda=True)
        m_tr_idx, m_va_idx = merged_gkf_folds[r_fold]
        fold_tr_df = merged_dev_df.iloc[m_tr_idx].reset_index(drop=True)
        fold_va_df = merged_dev_df.iloc[m_va_idx].reset_index(drop=True)

        assert set(fold_tr_df["group_id"]).isdisjoint(set(fold_va_df["group_id"]))
        assert set(fold_tr_df["slide_id"]).isdisjoint(set(fold_va_df["slide_id"]))

        fold_image_stats = fit_train_only_image_statistics_and_fourier_bank(
            fold_tr_df, seed=SEED + 777 + r_fold, center_to_idx=CENTER_TO_IDX
        )
        fold_frontend = HistopathologyPhysicsFrontend(fold_image_stats).to(DEVICE)

        fold_tr_ds = HistopathologyMultiViewDataset(
            fold_tr_df,
            fold_image_stats,
            is_train=True,
            center_to_idx=CENTER_TO_IDX,
        )
        fold_va_ds = HistopathologyMultiViewDataset(
            fold_va_df,
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

        bb_choice = refit_backbones[r_fold % len(refit_backbones)]
        bb_lr = 8.5e-5 if "convnext" in bb_choice else 1.0e-4
        f_net, f_crit, f_opt = build_model_criterion_optimizer(
            backbone_name=bb_choice,
            num_train_centers=_NUM_TRAIN_CENTERS,
            pretrained=True,
            backbone_lr=bb_lr,
            head_lr=4.0e-4,
            layer_decay=0.75,
            device=DEVICE,
        )
        fold_ckpt_path = os.path.join(WORKING_DIR, f"best_refit_fold{r_fold}.pt")
        f_net, f_stats = train_pathology_checkpoint(
            net=f_net,
            frontend=fold_frontend,
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
        refit_entries.append((f_net, fold_frontend))
        del f_crit, f_opt, fold_tr_loader, fold_va_loader, fold_tr_ds, fold_va_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(refit_entries) > 0:
        test_probs_refit = run_pipeline_inference(
            models_with_frontends=refit_entries,
            eval_loader=test_loader,
            expected_ids=test_ids_list,
            tta_views=TTA_VIEWS,
        )
        n_refit = float(len(refit_entries))
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