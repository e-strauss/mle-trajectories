import os
os.sched_setaffinity(0, {0, 1, 32, 2, 33, 34, 3, 35, 4, 36, 5, 37, 6, 38, 7, 39})
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
MAX_RUNTIME_BUDGET_SEC = 3420.0  # Target ceiling within 1-hour hard limit

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
    GPU-batched float32 physics & counterfactual scanner frontend:
      - Training-only: exact D4 dihedral symmetry, white-point illumination jitter,
        3D OD unit-sphere Macenko/Ruifrok-Johnston stain-basis rotation, log-uniform
        H&E dye concentration + gamma scaling, radial MTF + GPU rfft2 cross-center
        low-frequency swap, intra-class cross-center Beta(0.4, 0.4) pixel mixup, and
        paired (center_idx, label)-stratified K = B // 3 counterfactual scanner views.
      - Deterministic representation (100% identical across train, val, and test):
        75% orthogonal residual R suppression + 25% bounded foreground H&E soft-anchoring
        toward training-only her_mean/her_std blended (45% canonical H&E RGB + 55% raw RGB),
        plus 4-channel stain tensor: [global-anchored H, foreground-soft-anchored H,
        chromatic nuclear fraction H/(H+E+0.03), and R-gated 5x5 local chromatin contrast].
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

    @staticmethod
    def _select_stratified_pair_indices(
        B: int,
        K: int,
        labels: Optional[torch.Tensor],
        center_idx: Optional[torch.Tensor],
        dev: torch.device,
    ) -> torch.Tensor:
        """Select K unique indices stratified evenly across (center_idx, label) strata."""
        if K <= 0 or B <= 0:
            return torch.empty(0, dtype=torch.long, device=dev)
        if labels is None or center_idx is None:
            return torch.randperm(B, device=dev)[:K]

        y_bin = (labels.view(-1) > 0.5).long()
        c_clamped = center_idx.view(-1).clamp(min=0)
        strata = c_clamped * 2 + y_bin
        rand_u = torch.rand(B, device=dev)
        order = torch.argsort(strata.float() + rand_u * 0.999)
        sorted_s = strata[order]

        idx_arange = torch.arange(B, device=dev)
        is_new = torch.ones(B, dtype=torch.bool, device=dev)
        if B > 1:
            is_new[1:] = sorted_s[1:] != sorted_s[:-1]
        group_start = torch.cummax(
            torch.where(is_new, idx_arange, torch.zeros_like(idx_arange)), dim=0
        )[0]
        within_rank = idx_arange - group_start

        n_s = max(4, int(strata.max().item()) + 1)
        s_tie = torch.randperm(n_s, device=dev)[sorted_s].float() / float(n_s + 1)
        priority = within_rank.float() + s_tie + rand_u[order] * 1e-3
        chosen_pos = torch.argsort(priority)[:K]
        return order[chosen_pos]

    def _augment_rgb01(
        self,
        x: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
        counterfactual: bool = False,
    ) -> torch.Tensor:
        B = x.size(0)
        if B == 0:
            return x
        dev = x.device
        x = x.clone()

        # 1. Lossless D4 dihedral symmetry (random flip + 90-deg rotations)
        flip_m = torch.rand(B, 1, 1, 1, device=dev) < 0.5
        x = torch.where(flip_m, torch.flip(x, dims=[-1]), x)
        k_rot = torch.randint(0, 4, (B,), device=dev)
        for k in (1, 2, 3):
            mk = k_rot == k
            if mk.any():
                x[mk] = torch.rot90(x[mk], k=k, dims=(-2, -1))

        # 2. White-point illumination jitter
        wp_prob = 0.95 if counterfactual else 0.75
        wp_lim = 0.10 if counterfactual else 0.06
        do_wp = (torch.rand(B, 1, 1, 1, device=dev) < wp_prob).float()
        wp_gain = torch.exp(
            torch.empty(B, 3, 1, 1, device=dev).uniform_(-wp_lim, wp_lim)
        )
        x = (x * (1.0 + do_wp * (wp_gain - 1.0))).clamp(1.0 / 255.0, 1.0)

        # 3. 3D OD unit-sphere stain-basis rotation + log-uniform dye concentration & gamma
        stain_prob = 0.95 if counterfactual else 0.80
        he_lim = 0.42 if counterfactual else 0.28
        r_lim = 0.30 if counterfactual else 0.18
        beta_lim = 0.06 if counterfactual else 0.04
        rot_sd = 0.105 if counterfactual else 0.065

        do_stain = torch.rand(B, 1, 1, 1, device=dev) < stain_prob
        if do_stain.any():
            od = -torch.log10(x.clamp(1.0 / 255.0, 1.0))
            stain = torch.einsum("bchw,cd->bdhw", od, self.he_inv_matrix)
            tissue = (
                torch.linalg.vector_norm(od, dim=1, keepdim=True) > 0.10
            ).float()

            alpha_he = torch.exp(
                torch.empty(B, 2, 1, 1, device=dev).uniform_(-he_lim, he_lim)
            )
            alpha_r = torch.exp(
                torch.empty(B, 1, 1, 1, device=dev).uniform_(-r_lim, r_lim)
            )
            beta_he = torch.empty(B, 2, 1, 1, device=dev).uniform_(-beta_lim, beta_lim)

            st_he = stain[:, :2] * alpha_he + beta_he * tissue
            st_r = stain[:, 2:3] * alpha_r
            stain_p = torch.cat([st_he, st_r], dim=1)

            vh_base = self.he_stain_matrix[0:1].expand(B, -1)
            ve_base = self.he_stain_matrix[1:2].expand(B, -1)
            vh_rot = F.normalize(
                (vh_base + torch.randn(B, 3, device=dev) * rot_sd).clamp(min=0.01),
                dim=-1,
            )
            ve_rot = F.normalize(
                (ve_base + torch.randn(B, 3, device=dev) * rot_sd).clamp(min=0.01),
                dim=-1,
            )
            vr_rot = F.normalize(torch.cross(vh_rot, ve_rot, dim=-1), dim=-1)
            basis_rot = torch.stack([vh_rot, ve_rot, vr_rot], dim=1)

            od_rec = torch.einsum("bdhw,bdc->bchw", stain_p, basis_rot).clamp(0.0, 3.0)
            x_stain = torch.pow(10.0, -od_rec).clamp(1.0 / 255.0, 1.0)
            x = torch.where(do_stain, x_stain, x)

        gam_prob = 0.90 if counterfactual else 0.70
        gam_lim = 0.22 if counterfactual else 0.14
        do_gamma = (torch.rand(B, 1, 1, 1, device=dev) < gam_prob).float()
        gamma = 1.0 + do_gamma * (
            torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(-gam_lim, gam_lim))
            - 1.0
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
                opp_prob = 0.95 if counterfactual else 0.80
                prefer_opp = torch.rand(B, device=dev) < opp_prob
                idx_n0 = self.bank_idx_not0[
                    torch.randint(0, self.bank_idx_not0.numel(), (B,), device=dev)
                ]
                idx_n1 = self.bank_idx_not1[
                    torch.randint(0, self.bank_idx_not1.numel(), (B,), device=dev)
                ]
                b_idx = torch.where(prefer_opp & (center_idx == 0), idx_n0, b_idx)
                b_idx = torch.where(prefer_opp & (center_idx == 1), idx_n1, b_idx)

            tgt_low = self.fourier_bank[b_idx]
            fact_prob = 0.90 if counterfactual else 0.65
            f_lo, f_hi = (0.30, 0.70) if counterfactual else (0.15, 0.50)
            do_fact = (torch.rand(B, 1, 1, 1, device=dev) < fact_prob).float()
            alpha_f = torch.empty(B, 1, 1, 1, device=dev).uniform_(f_lo, f_hi)
            w = do_fact * alpha_f * self.low_freq_win

            r = FOURIER_LOW_RADIUS
            w_top, w_bot = w[:, :, :r, :], w[:, :, r:, :]
            amp[:, :, :r, :r] = (1.0 - w_top) * amp[:, :, :r, :r] + w_top * tgt_low[
                :, :, :r, :
            ]
            amp[:, :, -r:, :r] = (1.0 - w_bot) * amp[:, :, -r:, :r] + w_bot * tgt_low[
                :, :, r:, :
            ]

            mtf_prob = 0.80 if counterfactual else 0.50
            mtf_lo, mtf_hi = (-0.30, 0.85) if counterfactual else (-0.22, 0.65)
            do_mtf = (torch.rand(B, 1, 1, 1, device=dev) < mtf_prob).float()
            beta_mtf = do_mtf * torch.empty(B, 1, 1, 1, device=dev).uniform_(
                mtf_lo, mtf_hi
            )
            mtf_gain = torch.exp(-beta_mtf * self.frad_norm_sq)
            amp = amp * mtf_gain

            x = torch.fft.irfft2(
                amp * unit_phase, s=(IMG_SIZE, IMG_SIZE), dim=(-2, -1), norm="ortho"
            ).clamp(0.0, 1.0)

        n_prob = 0.35 if counterfactual else 0.20
        n_lo, n_hi = (0.004, 0.014) if counterfactual else (0.003, 0.010)
        do_noise = (torch.rand(B, 1, 1, 1, device=dev) < n_prob).float()
        noise_sd = do_noise * torch.empty(B, 1, 1, 1, device=dev).uniform_(n_lo, n_hi)
        x = (x + torch.randn_like(x) * noise_sd).clamp(0.0, 1.0)
        return x

    @staticmethod
    def apply_intraclass_crosscenter_mixup(
        x01: torch.Tensor,
        labels: torch.Tensor,
        center_idx: torch.Tensor,
        mix_prob: float = 0.30,
        exclude_mask: Optional[torch.Tensor] = None,
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
        if exclude_mask is not None:
            do_mix = do_mix & (~exclude_mask)
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
        """
        Deterministic fp32 Beer-Lambert pigment-suppressed soft-canonicalized RGB blend +
        4-channel dual-Hematoxylin stain-structural tensor (identical across train/val/test).
        """
        x01_f = x01.float().clamp(0.0, 1.0)
        od = -torch.log10(x01_f.clamp(1.0 / 255.0, 1.0))
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
        fg_e_mu = (e_pos * fg).sum(dim=(-2, -1), keepdim=True) / denom
        fg_e_sd = torch.sqrt(
            (((e_pos - fg_e_mu).square()) * fg).sum(dim=(-2, -1), keepdim=True) / denom
            + 1e-5
        )

        # 25% bounded foreground soft-anchoring toward train her_mean/her_std + 75% R attenuation
        alpha_anchor = 0.25 * fg_w
        h_scale = 1.0 + alpha_anchor * (
            (self.her_std[0] / (fg_h_sd + 1e-4)).clamp(0.65, 1.50) - 1.0
        )
        h_shift = alpha_anchor * (self.her_mean[0] - fg_h_mu).clamp(-0.25, 0.25)
        h_canon = (
            fg * ((h_pos - fg_h_mu) * h_scale + fg_h_mu + h_shift)
            + (1.0 - fg) * h_pos
        ).clamp(min=0.0)

        e_scale = 1.0 + alpha_anchor * (
            (self.her_std[1] / (fg_e_sd + 1e-4)).clamp(0.65, 1.50) - 1.0
        )
        e_shift = alpha_anchor * (self.her_mean[1] - fg_e_mu).clamp(-0.25, 0.25)
        e_canon = (
            fg * ((e_pos - fg_e_mu) * e_scale + fg_e_mu + e_shift)
            + (1.0 - fg) * e_pos
        ).clamp(min=0.0)

        r_canon = 0.25 * r_val
        stain_canon = torch.cat([h_canon, e_canon, r_canon], dim=1)
        od_canon = torch.einsum(
            "bdhw,dc->bchw", stain_canon, self.he_stain_matrix
        ).clamp(0.0, 3.0)
        x01_canon = torch.pow(10.0, -od_canon).clamp(0.0, 1.0)
        x01_blend = (0.45 * x01_canon + 0.55 * x01_f).clamp(0.0, 1.0)
        rgb_norm = (x01_blend - self.imagenet_mean) / self.imagenet_std

        # Dual Hematoxylin channels: global-anchored H + foreground-soft-anchored H
        ch_h_global = ((h_pos - self.her_mean[0]) / (self.her_std[0] + 1e-4)).clamp(
            -6.0, 6.0
        )
        h_mu_soft = 0.50 * fg_w * fg_h_mu + (1.0 - 0.50 * fg_w) * self.her_mean[0]
        h_sd_soft = (
            0.50 * fg_w * fg_h_sd + (1.0 - 0.50 * fg_w) * self.her_std[0] + 1e-4
        )
        ch_h_soft = ((h_pos - h_mu_soft) / h_sd_soft).clamp(-6.0, 6.0)
        ch_frac = (2.0 * (h_pos / (h_pos + e_pos + 0.03)) - 1.0).clamp(-1.0, 1.0)

        h_pad = F.pad(ch_h_soft, (2, 2, 2, 2), mode="reflect")
        loc_mu = F.avg_pool2d(h_pad, kernel_size=5, stride=1)
        loc_var = (
            F.avg_pool2d(h_pad.square(), kernel_size=5, stride=1) - loc_mu.square()
        ).clamp(min=0.0)
        r_mag = (r_val - self.her_mean[2]).abs() / (self.her_std[2] + 1e-4)
        pigment_gate = torch.exp(-0.5 * F.relu(r_mag - 1.5))
        ch_contrast = (
            ((ch_h_soft - loc_mu) / torch.sqrt(loc_var + 0.04)) * pigment_gate
        ).clamp(-6.0, 6.0)

        stain_4ch = torch.cat([ch_h_global, ch_h_soft, ch_frac, ch_contrast], dim=1)
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
        return_paired: bool = False,
    ):
        if rgb.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                return self._forward_fp32(
                    rgb, center_idx, labels, mixup_p, return_center_idx, return_paired
                )
        return self._forward_fp32(
            rgb, center_idx, labels, mixup_p, return_center_idx, return_paired
        )

    def _forward_fp32(
        self,
        rgb: torch.Tensor,
        center_idx: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        mixup_p: float,
        return_center_idx: bool,
        return_paired: bool,
    ):
        x01_raw = self._to_unit_rgb(rgb)
        B = x01_raw.size(0)
        dev = x01_raw.device
        eff_center = center_idx

        if return_paired and self.training and B >= 3:
            K = B // 3
            pair_idx = self._select_stratified_pair_indices(
                B, K, labels, center_idx, dev
            )
            paired_mask = torch.zeros(B, dtype=torch.bool, device=dev)
            paired_mask[pair_idx] = True

            x01 = self._augment_rgb01(
                x01_raw, center_idx=center_idx, counterfactual=False
            )
            if (
                mixup_p > 0.0
                and (labels is not None)
                and (center_idx is not None)
            ):
                x01, eff_center = self.apply_intraclass_crosscenter_mixup(
                    x01,
                    labels,
                    center_idx,
                    mix_prob=mixup_p,
                    exclude_mask=paired_mask,
                )

            sub_c = center_idx[pair_idx] if center_idx is not None else None
            x01_cf = self._augment_rgb01(
                x01_raw[pair_idx], center_idx=sub_c, counterfactual=True
            )
            x01_all = torch.cat([x01, x01_cf], dim=0)
            rgb_norm, stain_4ch = self.compute_representations(x01_all)

            if eff_center is not None:
                cf_center = torch.full(
                    (K,), -1, dtype=eff_center.dtype, device=dev
                )
                eff_center_out = torch.cat([eff_center, cf_center], dim=0)
            else:
                eff_center_out = None
            labels_out = (
                torch.cat([labels, labels[pair_idx]], dim=0)
                if labels is not None
                else None
            )
            return rgb_norm, stain_4ch, eff_center_out, labels_out, pair_idx

        x01 = x01_raw
        if self.training:
            x01 = self._augment_rgb01(x01, center_idx=center_idx, counterfactual=False)
            if (
                mixup_p > 0.0
                and (labels is not None)
                and (center_idx is not None)
            ):
                x01, eff_center = self.apply_intraclass_crosscenter_mixup(
                    x01, labels, center_idx, mix_prob=mixup_p
                )
        rgb_norm, stain_4ch = self.compute_representations(x01)
        if return_paired:
            empty_p = torch.empty(0, dtype=torch.long, device=dev)
            return rgb_norm, stain_4ch, eff_center, labels, empty_p
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

    primary_physics_frontend.train()
    _tr_b = _s_tr["rgb"].unsqueeze(0).repeat(6, 1, 1, 1).to(DEVICE)
    _tr_c = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long, device=DEVICE)
    _tr_y = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], device=DEVICE)
    _rp_rgb, _rp_st, _rp_c, _rp_y, _rp_idx = primary_physics_frontend(
        _tr_b, center_idx=_tr_c, labels=_tr_y, mixup_p=0.5, return_paired=True
    )
    assert _rp_idx.shape == (2,) and _rp_rgb.shape == (8, 3, IMG_SIZE, IMG_SIZE)
    assert _rp_st.shape == (8, 4, IMG_SIZE, IMG_SIZE) and _rp_y.shape == (8,)
    assert torch.equal(_rp_c[_rp_idx], _tr_c[_rp_idx]), "Paired anchors must stay unmixed!"
    primary_physics_frontend.eval()
seed_everything(SEED, seed_cuda=True)

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
# 1) convnextv2_base.fcmae_ft_in22k_in1k
#    - Architecture: Native 4-stage 100% BatchNorm-free FCMAE ConvNeXtV2-Base
#      ([128, 256, 512, 1024] at 64x64, 32x32, 16x16, 8x8 with LayerNorm2d + GRN)
#    - Pretraining Corpus: Self-supervised FCMAE + ImageNet-22K -> ImageNet-1K natural images
# 2) swinv2_base_window12to16_192to256.ms_in22k_ft_in1k
#    - Architecture: Microsoft Scaled-Cosine Swin Transformer V2 Base (native 256x256, window=16)
#    - Pretraining Corpus: ImageNet-22K -> ImageNet-1K 256x256 natural images
# - Library: timm (version logged at runtime)
# - Known Histopathology / Camelyon Overlap: None (natural-image pretraining only)
CONVNEXT_BACKBONE_NAME = "convnextv2_base.fcmae_ft_in22k_in1k"
SWINV2_BACKBONE_NAME = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"

MODEL_METADATA = {
    "architecture": "PathologyMultiScaleStainNet",
    "pretrained_backbone": f"{CONVNEXT_BACKBONE_NAME} + {SWINV2_BACKBONE_NAME}",
    "pretrained_backbones": [
        CONVNEXT_BACKBONE_NAME,
        "convnextv2_base.fcmae_ft_in1k",
        SWINV2_BACKBONE_NAME,
    ],
    "timm_version": getattr(timm, "__version__", "unknown"),
    "pretraining_corpus": "FCMAE + ImageNet-22K -> ImageNet-1K natural images (ConvNeXtV2-Base + SwinV2-Base)",
    "known_target_overlap": "None (natural images only; zero histopathology overlap)",
}

_NUM_TRAIN_CENTERS = len(CENTER_TO_IDX)


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
    D4-invariant Feature-Space Relational Focal-vs-Background MIL readout:
    computes radial-biased focal attention f_focus and low-attention resident background
    f_bg (via softmax(-temp * scores)) within the same patch, residually projecting
    concat([f_focus, f_focus - f_bg]) to cancel additive intra-patch stain offsets.
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
        self.rel_proj = nn.Sequential(
            nn.LayerNorm(in_dim * 2),
            nn.Linear(in_dim * 2, in_dim),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(in_dim, in_dim),
        )
        self.out_norm = nn.LayerNorm(in_dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        orig_dtype = feat_map.dtype
        tokens = feat_map.flatten(2).transpose(1, 2)  # (B, H*W, C)
        gated = self.attn_v(tokens) * self.attn_u(tokens)
        scores = self.attn_w(gated).squeeze(-1).float()
        rad_bias = self.radial_prior(self.radial_basis).squeeze(-1).float()
        temp = self.mil_temp.float().clamp(0.5, 8.0)

        w_focus = F.softmax(temp * (scores + rad_bias), dim=-1).to(orig_dtype)
        f_focus = torch.bmm(w_focus.unsqueeze(1), tokens).squeeze(1)

        w_bg = F.softmax(-temp * scores, dim=-1).to(orig_dtype)
        f_bg = torch.bmm(w_bg.unsqueeze(1), tokens).squeeze(1)

        rel_in = torch.cat([f_focus, f_focus - f_bg], dim=-1)
        return self.out_norm(f_focus + self.rel_proj(rel_in))


def _create_pathology_backbone(
    backbone_name: str, pretrained: bool = True
) -> Tuple[nn.Module, List[int], str]:
    """Instantiate native 4-stage 100% BatchNorm-free FCMAE ConvNeXtV2-Base or SwinV2-Base backbone."""
    lower_name = backbone_name.lower()

    if "swin" in lower_name:
        candidates = [
            (backbone_name, {}),
            (SWINV2_BACKBONE_NAME, {}),
            ("swinv2_base_window12to16_192to256", {}),
            ("swinv2_base_window8_256.ms_in1k", {}),
            ("swinv2_tiny_window8_256", {}),
        ]
        drop_path = 0.25
    elif "convnext" in lower_name:
        candidates = [
            (backbone_name, {}),
            (CONVNEXT_BACKBONE_NAME, {}),
            ("convnextv2_base.fcmae_ft_in1k", {}),
            ("convnextv2_base", {}),
            ("convnext_base.fb_in22k_ft_in1k", {}),
            ("convnext_small.fb_in22k_ft_in1k", {}),
        ]
        drop_path = 0.25
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


class HypersphericalPrototypeClassifier(nn.Module):
    """
    Zero-bias symmetric hyperspherical prototype classifier:
    computes tau * (cos(z_norm, u_pos) - cos(z_norm, u_neg)) over unit-normalized
    features and antipodal unit prototypes with tau clamped in [4.0, 24.0].
    """

    def __init__(self, in_dim: int, init_tau: float = 6.0):
        super().__init__()
        v_init = F.normalize(torch.randn(in_dim, dtype=torch.float32), p=2, dim=-1)
        self.proto_pos = nn.Parameter(v_init.clone())
        self.proto_neg = nn.Parameter((-v_init).clone())
        self.proto_tau = nn.Parameter(
            torch.tensor(float(init_tau), dtype=torch.float32)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
        p_pos = F.normalize(self.proto_pos.float(), p=2, dim=-1, eps=1e-6)
        p_neg = F.normalize(self.proto_neg.float(), p=2, dim=-1, eps=1e-6)
        u_pos = F.normalize(p_pos - p_neg, p=2, dim=-1, eps=1e-6)
        u_neg = -u_pos
        cos_pos = (z_norm * u_pos).sum(dim=-1)
        cos_neg = (z_norm * u_neg).sum(dim=-1)
        tau = self.proto_tau.float().clamp(4.0, 24.0)
        return tau * (cos_pos - cos_neg)


class PathologyMultiScaleStainNet(nn.Module):
    """
    Unified BatchNorm-free 4-Scale (64x64, 32x32, 16x16, 8x8) Visual-Stain Architecture
    supporting native 4-stage FCMAE ConvNeXtV2-Base ([128, 256, 512, 1024]) and Scaled-Cosine
    SwinV2-Base ([128, 256, 512, 1024]):
      1) 4-scale RGB backbone (stride 4..32) + early-stage ExtrapolativeMixStyle2d.
      2) 4-scale Zero-Phase 2D-FFT StainSpectralPyramidEncoder on 4-ch dual-H stain maps.
      3) Bottom-up + top-down multi-scale feature pyramid fusion.
      4) D4 Concentric Rings + GeM + Relational Focal-vs-Background Gated-Attention MIL.
      5) Zero-bias Symmetric Hyperspherical Prototype Classifier + Counterfactual Invariance.
    """

    def __init__(
        self,
        backbone_name: str = CONVNEXT_BACKBONE_NAME,
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
        c0, c1, c2, c3 = self.bb_channels
        d_fine = d_model // 2  # 128 at 64x64

        self.mixstyle_0 = ExtrapolativeMixStyle2d(p=0.35)
        self.mixstyle_1 = ExtrapolativeMixStyle2d(p=0.35)
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
        self.classifier = HypersphericalPrototypeClassifier(d_model, init_tau=6.0)

        self.aux_vis_norm = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(p=0.20),
        )
        self.aux_vis_head = HypersphericalPrototypeClassifier(d_model * 2, init_tau=6.0)

    def forward(
        self,
        rgb: torch.Tensor,
        stain: torch.Tensor,
        grl_lambda: float = 0.0,
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

        if self.training:
            logits = torch.stack(
                [self.classifier(drop(fused_feat)) for drop in self.ms_dropouts], dim=0
            ).mean(dim=0)
        else:
            logits = self.classifier(fused_feat)

        if not return_aux:
            return logits

        aux_vis_in = self.aux_vis_norm(
            torch.cat([concentric_vec, mil_fine_vec], dim=-1)
        )
        aux_vis_logits = self.aux_vis_head(aux_vis_in)

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits.float()),
            "aux_vis_logits": aux_vis_logits,
            "fused_feat": F.normalize(fused_feat.float(), p=2, dim=-1, eps=1e-6),
        }

    @torch.no_grad()
    def predict_proba(self, rgb: torch.Tensor, stain: torch.Tensor) -> torch.Tensor:
        """Deterministic tumor probability inference in [0, 1] for validation & test."""
        logits = self.forward(rgb, stain, return_aux=False)
        return torch.sigmoid(logits.float()).clamp(1e-6, 1.0 - 1e-6)


class UnseenCenterBalancedAccuracyLoss(nn.Module):
    """
    Directly aligns training optimization with unseen-center Balanced Accuracy at fixed 0.5
    threshold + AUROC via Counterfactual Scanner-View Invariance (without center_id at inference):
      1) Class-Balanced Symmetric Label-Smoothed BCE + moderated Soft Balanced Accuracy surrogate.
      2) Direct Paired Counterfactual Scanner-View Invariance (Smooth-L1 logit equality,
         exact symmetric Bernoulli KL probability consistency, and cosine feature alignment).
      3) Auxiliary Hyperspherical Prototype BCE + mild Pairwise Logistic Margin AUROC (0.08).
      4) Training-only Class-Conditional CORAL across unmixed development centers.
    """

    def __init__(
        self,
        num_centers: int = _NUM_TRAIN_CENTERS,
        label_smoothing: float = 0.03,
        soft_ba_weight: float = 0.20,
        sharpness: float = 3.5,
        auroc_weight: float = 0.08,
        rank_margin: float = 1.0,
        aux_vis_weight: float = 0.20,
        cond_coral_weight: float = 0.12,
        cf_logit_weight: float = 0.25,
        cf_kl_weight: float = 0.35,
        cf_cos_weight: float = 0.20,
    ):
        super().__init__()
        self.num_centers = max(2, int(num_centers))
        self.label_smoothing = label_smoothing
        self.soft_ba_weight = soft_ba_weight
        self.sharpness = sharpness
        self.auroc_weight = auroc_weight
        self.rank_margin = rank_margin
        self.aux_vis_weight = aux_vis_weight
        self.cond_coral_weight = cond_coral_weight
        self.cf_logit_weight = cf_logit_weight
        self.cf_kl_weight = cf_kl_weight
        self.cf_cos_weight = cf_cos_weight

    @torch.no_grad()
    def reset_dro_weights(self) -> None:
        return None

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

    def _counterfactual_pair_invariance(
        self,
        logits: torch.Tensor,
        fused_feat: Optional[torch.Tensor],
        pair_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pair_idx is None or pair_idx.numel() == 0:
            return logits.new_zeros(())
        k_pairs = int(pair_idx.numel())
        n_total = int(logits.numel())
        b_base = n_total - k_pairs
        if b_base <= 0:
            return logits.new_zeros(())

        z_anchor = logits[pair_idx]
        z_cf = logits[b_base:]
        loss_logit = F.smooth_l1_loss(z_anchor, z_cf, beta=0.25)

        p_anchor = torch.sigmoid(z_anchor)
        p_cf = torch.sigmoid(z_cf)
        loss_sym_kl = (0.5 * (p_anchor - p_cf) * (z_anchor - z_cf)).mean()

        inv_loss = (
            self.cf_logit_weight * loss_logit + self.cf_kl_weight * loss_sym_kl
        )
        if fused_feat is not None and self.cf_cos_weight > 0.0:
            f_anchor = F.normalize(fused_feat[pair_idx].float(), p=2, dim=-1, eps=1e-6)
            f_cf = F.normalize(fused_feat[b_base:].float(), p=2, dim=-1, eps=1e-6)
            loss_cos = (1.0 - (f_anchor * f_cf).sum(dim=-1)).mean()
            inv_loss = inv_loss + self.cf_cos_weight * loss_cos
        return inv_loss

    def forward(
        self,
        outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        targets: torch.Tensor,
        center_idx: Optional[torch.Tensor] = None,
        pair_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        targets_f32 = targets.float().view(-1)

        if torch.is_tensor(outputs):
            logits = outputs.float().view(-1)
            loss = (
                self._symmetric_balanced_bce(logits, targets_f32)
                + self.soft_ba_weight
                * self._soft_balanced_accuracy_loss(logits, targets_f32)
                + self.auroc_weight
                * self._pairwise_auroc_margin_loss(logits, targets_f32)
                + self._counterfactual_pair_invariance(logits, None, pair_idx)
            )
            return loss

        logits = outputs["logits"].float().view(-1)
        total_loss = (
            self._symmetric_balanced_bce(logits, targets_f32)
            + self.soft_ba_weight
            * self._soft_balanced_accuracy_loss(logits, targets_f32)
            + self.auroc_weight
            * self._pairwise_auroc_margin_loss(logits, targets_f32)
        )

        if "aux_vis_logits" in outputs:
            total_loss = (
                total_loss
                + self.aux_vis_weight
                * self._symmetric_balanced_bce(
                    outputs["aux_vis_logits"].float().view(-1), targets_f32
                )
            )

        fused_feat = (
            outputs["fused_feat"].float() if "fused_feat" in outputs else None
        )
        total_loss = total_loss + self._counterfactual_pair_invariance(
            logits, fused_feat, pair_idx
        )

        if (
            center_idx is not None
            and fused_feat is not None
            and self.cond_coral_weight > 0.0
        ):
            total_loss = (
                total_loss
                + self.cond_coral_weight
                * self._class_conditional_coral(
                    fused_feat, targets_f32, center_idx.view(-1)
                )
            )

        return total_loss


def _infer_backbone_stage(param_name: str) -> int:
    """Map hierarchical SwinV2 / ConvNeXt backbone parameter names to stage index 0..3."""
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


def _default_optim_hparams_for_backbone(
    backbone_name: str,
) -> Tuple[float, float, float]:
    """Return stage-accurate (backbone_lr, head_lr, layer_decay) for ConvNeXtV2-Base vs SwinV2-Base."""
    low = backbone_name.lower()
    if "convnext" in low:
        return 5.0e-5, 3.5e-4, 0.75
    if "swin" in low:
        return 5.5e-5, 3.5e-4, 0.75
    return 5.0e-5, 3.5e-4, 0.75


def build_differential_adamw(
    model_obj: nn.Module,
    backbone_lr: float = 5.0e-5,
    head_lr: float = 3.5e-4,
    layer_decay: float = 0.75,
    backbone_wd: float = 0.03,
    head_wd: float = 0.02,
) -> AdamW:
    """
    Constructs stage-accurate AdamW with:
      - 0.75 decay across ConvNeXtV2-Base stages 0..3 or SwinV2-Base stages 0..3.
      - Zero weight decay on biases, norms, GRN, LayerScale gammas, cosine attention scales,
        relative position tables, GeM/MIL/spectral gates, and hyperspherical prototypes.
    """
    no_decay_keywords = (
        "bias",
        "norm",
        "grn",
        "gamma",
        "beta",
        "ls1",
        "ls2",
        "pos_embed",
        "logit_scale",
        "cpb_mlp",
        "relative_position",
        "freq_gain_logits",
        "gem_p",
        "mil_temp",
        "proto",
    )

    num_depths = 4
    max_depth_idx = num_depths - 1

    bb_depth_groups: Dict[Tuple[int, bool], List[nn.Parameter]] = {
        (d, nd): [] for d in range(num_depths) for nd in (False, True)
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
            bb_depth_groups[(s_idx, is_no_decay)].append(param)
        elif not is_no_decay:
            head_decay.append(param)
        else:
            head_no_decay.append(param)

    param_groups = []
    for d_idx in range(num_depths):
        depth_lr = backbone_lr * (layer_decay ** float(max_depth_idx - d_idx))
        if bb_depth_groups[(d_idx, False)]:
            param_groups.append(
                {
                    "params": bb_depth_groups[(d_idx, False)],
                    "lr": depth_lr,
                    "weight_decay": backbone_wd,
                    "name": f"bb_s{d_idx}_decay",
                }
            )
        if bb_depth_groups[(d_idx, True)]:
            param_groups.append(
                {
                    "params": bb_depth_groups[(d_idx, True)],
                    "lr": depth_lr,
                    "weight_decay": 0.0,
                    "name": f"bb_s{d_idx}_no_decay",
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
    backbone_name: str = CONVNEXT_BACKBONE_NAME,
    num_train_centers: int = _NUM_TRAIN_CENTERS,
    pretrained: bool = True,
    backbone_lr: Optional[float] = None,
    head_lr: Optional[float] = None,
    layer_decay: Optional[float] = None,
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
    def_bb_lr, def_head_lr, def_decay = _default_optim_hparams_for_backbone(
        net.backbone_name
    )
    loss_fn = UnseenCenterBalancedAccuracyLoss(num_centers=num_train_centers).to(device)
    opt = build_differential_adamw(
        net,
        backbone_lr=def_bb_lr if backbone_lr is None else backbone_lr,
        head_lr=def_head_lr if head_lr is None else head_lr,
        layer_decay=def_decay if layer_decay is None else layer_decay,
    )
    return net, loss_fn, opt


# Instantiate Primary Model (ConvNeXtV2-Base), Criterion, Optimizer & Verify Both Architectures
model, criterion, optimizer = build_model_criterion_optimizer(
    backbone_name=CONVNEXT_BACKBONE_NAME,
    num_train_centers=_NUM_TRAIN_CENTERS,
    pretrained=True,
    backbone_lr=5.0e-5,
    head_lr=3.5e-4,
    layer_decay=0.75,
    device=DEVICE,
)

with torch.no_grad():
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        for m in model.modules()
    ), "PathologyMultiScaleStainNet (ConvNeXtV2-Base) must be 100% BatchNorm-free!"

    model.eval()
    criterion.eval()
    _b = 8
    _dummy_rgb = torch.randn(_b, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _dummy_stain = torch.randn(_b, 4, IMG_SIZE, IMG_SIZE, device=DEVICE)
    _dummy_y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], device=DEVICE)
    _dummy_cid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=DEVICE)

    _logits1 = model(_dummy_rgb, _dummy_stain, return_aux=False)
    _probs1 = model.predict_proba(_dummy_rgb, _dummy_stain)
    _probs2 = model.predict_proba(_dummy_rgb, _dummy_stain)
    _aux_out = model(
        _dummy_rgb,
        _dummy_stain,
        center_idx=_dummy_cid,
        return_aux=True,
    )
    _loss_single = criterion(_aux_out, _dummy_y, center_idx=_dummy_cid)

    _p_idx = torch.tensor([0, 3], dtype=torch.long, device=DEVICE)
    _rgb_p = torch.cat([_dummy_rgb, _dummy_rgb[_p_idx]], dim=0)
    _st_p = torch.cat([_dummy_stain, _dummy_stain[_p_idx]], dim=0)
    _y_p = torch.cat([_dummy_y, _dummy_y[_p_idx]], dim=0)
    _cid_p = torch.cat(
        [_dummy_cid, torch.full((2,), -1, dtype=torch.long, device=DEVICE)], dim=0
    )
    _aux_p = model(_rgb_p, _st_p, center_idx=_cid_p, return_aux=True)
    _loss_paired = criterion(_aux_p, _y_p, center_idx=_cid_p, pair_idx=_p_idx)

    _metric_check = binary_metrics(
        _dummy_y.cpu().numpy().astype(int), _probs1.cpu().numpy()
    )

    assert _logits1.shape == (_b,) and torch.isfinite(_logits1).all()
    assert _probs1.shape == (_b,) and torch.isfinite(_probs1).all()
    assert torch.equal(_probs1, _probs2), "Eval inference must be deterministic!"
    assert ((_probs1 >= 0.0) & (_probs1 <= 1.0)).all()
    assert torch.isfinite(_loss_single) and torch.isfinite(
        _loss_paired
    ), "Loss produced non-finite value!"
    assert "balanced_accuracy" in _metric_check and "auroc" in _metric_check
    criterion.train()
    criterion.reset_dro_weights()

    # Verify SwinV2-Base 4-scale forward, optimizer groups & BatchNorm-free contract
    _swin_check, _, _swin_opt_check = build_model_criterion_optimizer(
        backbone_name=SWINV2_BACKBONE_NAME,
        num_train_centers=_NUM_TRAIN_CENTERS,
        pretrained=False,
        backbone_lr=5.5e-5,
        head_lr=3.5e-4,
        layer_decay=0.75,
        device=DEVICE,
    )
    assert not any(
        isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
        for m in _swin_check.modules()
    ), "PathologyMultiScaleStainNet (SwinV2-Base) must be 100% BatchNorm-free!"
    _swin_check.eval()
    _swin_p = _swin_check.predict_proba(_dummy_rgb, _dummy_stain)
    assert _swin_p.shape == (_b,) and torch.isfinite(_swin_p).all()
    del _swin_check, _swin_opt_check

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
import copy
from scipy.optimize import minimize

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024.0**3)
    if _VRAM_GB >= 60.0:
        BATCH_SIZE = 60
    elif _VRAM_GB >= 34.0:
        BATCH_SIZE = 48
    elif _VRAM_GB >= 20.0:
        BATCH_SIZE = 36
    else:
        BATCH_SIZE = 24
    USE_AMP = True
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    USE_SCALER = AMP_DTYPE == torch.float16
else:
    _VRAM_GB = 0.0
    BATCH_SIZE = 24
    USE_AMP = False
    AMP_DTYPE = torch.float32
    USE_SCALER = False

INFER_BATCH_SIZE = max(64, BATCH_SIZE * 2)
CONVNEXT_EPOCHS = 4
SWINV2_EPOCHS = 4
STAGE1_EPOCHS = 4
REFIT_EPOCHS = 4
TTA_VIEWS = 8
TRAIN_RUNTIME_CEILING_SEC = 3420.0


def _save_pure_state_dict(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    clean_sd = {k: v.detach().cpu().clone() for k, v in state_dict.items()}
    torch.save(clean_sd, path)


def _load_pure_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class _PatchRawRGBDataset(Dataset):
    """Fast minimal RGB uint8 reader for one-time per-patch physics stat pre-extraction."""

    def __init__(self, sample_ids: List[str], image_paths: List[str]):
        self.sample_ids = sample_ids
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        rgb_u8 = load_rgb_patch(self.image_paths[idx])
        rgb_chw = np.ascontiguousarray(rgb_u8.transpose(2, 0, 1), dtype=np.uint8)
        return self.sample_ids[idx], torch.from_numpy(rgb_chw)


@torch.no_grad()
def preextract_patch_physics_cache(
    dataframe: pd.DataFrame,
    cache_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    cache_tag: str,
) -> None:
    """
    Pre-extract per-sample (her_mean, her_sq, low_amp) statistics once in memory
    (and disk cache) via GPU-batched Beer-Lambert OD + ortho rfft2 for sub-second
    leak-free fold frontend construction.
    """
    sids_all = dataframe["sample_id"].astype(str).tolist()
    paths_all = dataframe["image_path"].astype(str).tolist()
    miss_sids, miss_paths = [], []
    for sid, rel_p in zip(sids_all, paths_all):
        if sid not in cache_dict:
            miss_sids.append(sid)
            miss_paths.append(rel_p)
    if not miss_sids:
        return

    sig_hash = hashlib.md5(
        ("|".join(miss_sids) + f"_r{FOURIER_LOW_RADIUS}_v4").encode("utf-8")
    ).hexdigest()[:12]
    npz_path = os.path.join(
        WORKING_DIR, f"patch_phys_preextract_{cache_tag}_{sig_hash}.npz"
    )
    if os.path.exists(npz_path):
        loaded = np.load(npz_path)
        l_sids = loaded["sample_ids"].tolist()
        l_hm = loaded["her_means"]
        l_hsq = loaded["her_sqs"]
        l_lamp = loaded["low_amps"]
        for i, sid in enumerate(l_sids):
            cache_dict[str(sid)] = (l_hm[i], l_hsq[i], l_lamp[i])
        if all(sid in cache_dict for sid in miss_sids):
            return

    nw = max(2, min(8, os.cpu_count() or 4))
    ds = _PatchRawRGBDataset(miss_sids, miss_paths)
    dl = DataLoader(
        ds,
        batch_size=192,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    he_inv_dev = torch.from_numpy(HE_INV_MATRIX.copy()).to(
        DEVICE, dtype=torch.float32
    )
    r = FOURIER_LOW_RADIUS

    out_sids: List[str] = []
    out_hm: List[np.ndarray] = []
    out_hsq: List[np.ndarray] = []
    out_lamp: List[np.ndarray] = []

    for batch_sids, batch_u8 in dl:
        rgb_f32 = batch_u8.to(DEVICE, dtype=torch.float32, non_blocking=True)
        if rgb_f32.is_cuda:
            ctx = torch.amp.autocast("cuda", enabled=False)
        else:
            ctx = torch.amp.autocast("cpu", enabled=False)
        with ctx:
            od = -torch.log10(rgb_f32.clamp(1.0, 255.0) / 255.0)
            stain = torch.einsum("bchw,cd->bdhw", od, he_inv_dev)
            her = torch.stack(
                [stain[:, 0].clamp(min=0.0), stain[:, 1].clamp(min=0.0), stain[:, 2]],
                dim=1,
            )
            fg_f = (
                torch.linalg.vector_norm(od, dim=1, keepdim=True) > 0.12
            ).float()
            fg_cnt = fg_f.sum(dim=(-2, -1))
            use_fg = fg_cnt >= 32.0
            denom = fg_cnt.clamp(min=1.0)

            fg_mean = (her * fg_f).sum(dim=(-2, -1)) / denom
            fg_sq = (her.square() * fg_f).sum(dim=(-2, -1)) / denom
            all_mean = her.mean(dim=(-2, -1))
            all_sq = her.square().mean(dim=(-2, -1))

            hm_b = torch.where(use_fg, fg_mean, all_mean)
            hsq_b = torch.where(use_fg, fg_sq, all_sq)

            amp = torch.fft.rfft2(
                rgb_f32 / 255.0, dim=(-2, -1), norm="ortho"
            ).abs()
            lamp_b = torch.cat([amp[:, :, :r, :r], amp[:, :, -r:, :r]], dim=2)

        hm_np = hm_b.cpu().numpy().astype(np.float32)
        hsq_np = hsq_b.cpu().numpy().astype(np.float32)
        lamp_np = lamp_b.cpu().numpy().astype(np.float32)
        for i, sid in enumerate(batch_sids):
            sid_str = str(sid)
            cache_dict[sid_str] = (hm_np[i], hsq_np[i], lamp_np[i])
            out_sids.append(sid_str)
        out_hm.append(hm_np)
        out_hsq.append(hsq_np)
        out_lamp.append(lamp_np)

    np.savez(
        npz_path,
        sample_ids=np.array(out_sids),
        her_means=np.concatenate(out_hm, axis=0),
        her_sqs=np.concatenate(out_hsq, axis=0),
        low_amps=np.concatenate(out_lamp, axis=0),
    )


def build_fold_image_stats_from_cache(
    fold_train_df: pd.DataFrame,
    cache_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    seed: int = SEED,
    center_to_idx: Optional[dict] = None,
) -> dict:
    """
    Sub-second leak-free construction of H/E/R anchors and stratified cross-center
    rfft2 Fourier bank strictly from fold_train_df samples.
    """
    rng = np.random.default_rng(seed)
    sids = fold_train_df["sample_id"].astype(str).tolist()
    centers = fold_train_df["center_id"].astype(str).values
    unique_centers = sorted(np.unique(centers).tolist())
    if center_to_idx is None:
        center_to_idx = {cid: idx for idx, cid in enumerate(unique_centers)}

    her_means = np.stack([cache_dict[sid][0] for sid in sids], axis=0)
    her_sqs = np.stack([cache_dict[sid][1] for sid in sids], axis=0)
    her_mean = her_means.mean(axis=0).astype(np.float32)
    her_std = np.sqrt(np.maximum(her_sqs.mean(axis=0) - her_mean**2, 1e-6)).astype(
        np.float32
    )

    per_center_quota = max(1, FOURIER_BANK_SIZE // max(1, len(unique_centers)))
    bank_arrays = []
    bank_center_ids = []
    for cid in unique_centers:
        c_idx = np.where(centers == cid)[0]
        chosen = rng.choice(
            c_idx, size=min(per_center_quota, len(c_idx)), replace=False
        )
        c_num = int(center_to_idx.get(cid, 0))
        for idx_pos in chosen:
            bank_arrays.append(cache_dict[sids[int(idx_pos)]][2])
            bank_center_ids.append(c_num)

    return {
        "rgb_mean": IMAGENET_MEAN.copy(),
        "rgb_std": IMAGENET_STD.copy(),
        "her_mean": her_mean,
        "her_std": her_std,
        "fourier_bank": np.stack(bank_arrays, axis=0).astype(np.float32),
        "fourier_bank_centers": np.asarray(bank_center_ids, dtype=np.int64),
    }


def _stable_sigmoid_np(u: np.ndarray) -> np.ndarray:
    u_c = np.clip(np.asarray(u, dtype=np.float64), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-u_c))


def apply_platt_log_odds(
    logits: np.ndarray, temp_t: float, bias_b: float
) -> np.ndarray:
    """Calibrated equal-prior log-odds u_m(z) = (z + b_m) / T_m."""
    z = np.asarray(logits, dtype=np.float64)
    return (z + float(bias_b)) / max(float(temp_t), 1e-4)


def compute_equal_center_oof_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    centers: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes equal-center Balanced Accuracy and AUROC via ./input/metric.py so each
    (center_id, label) stratum contributes 0.25 weight regardless of center sample count.
    """
    y_int = np.asarray(labels, dtype=int)
    p_arr = np.asarray(probabilities, dtype=np.float64)
    pooled_m = binary_metrics(y_int, p_arr)
    if centers is None:
        return {
            "balanced_accuracy": pooled_m["balanced_accuracy"],
            "auroc": pooled_m["auroc"],
            "worst_center_ba": pooled_m["balanced_accuracy"],
            "pooled_ba": pooled_m["balanced_accuracy"],
        }

    c_arr = np.asarray(centers, dtype=str)
    per_center_m = []
    for cid in sorted(np.unique(c_arr).tolist()):
        mask = c_arr == cid
        if int(mask.sum()) >= 2 and set(y_int[mask].tolist()) == {0, 1}:
            per_center_m.append(binary_metrics(y_int[mask], p_arr[mask]))

    if len(per_center_m) < 2:
        return {
            "balanced_accuracy": pooled_m["balanced_accuracy"],
            "auroc": pooled_m["auroc"],
            "worst_center_ba": pooled_m["balanced_accuracy"],
            "pooled_ba": pooled_m["balanced_accuracy"],
        }

    return {
        "balanced_accuracy": float(
            np.mean([m["balanced_accuracy"] for m in per_center_m])
        ),
        "auroc": float(np.mean([m["auroc"] for m in per_center_m])),
        "worst_center_ba": float(
            np.min([m["balanced_accuracy"] for m in per_center_m])
        ),
        "pooled_ba": pooled_m["balanced_accuracy"],
    }


def fit_class_balanced_platt_scaler(
    oof_logits: np.ndarray,
    oof_labels: np.ndarray,
    oof_centers: Optional[np.ndarray] = None,
    l2_scale: float = 0.02,
    l2_bias: float = 0.05,
) -> Tuple[float, float]:
    """
    Guarded L2-regularized Equal-Center Class-Balanced Platt log-odds calibration:
    weights each (center_id, label) stratum equally (0.25 each) under l2_scale=0.02 and
    l2_bias=0.05, and falls back to temperature-only (T_m, 0.0) or uncalibrated (1.0, 0.0)
    whenever Platt bias degrades equal-center OOF balanced accuracy.
    """
    z = np.clip(
        np.nan_to_num(
            np.asarray(oof_logits, dtype=np.float64),
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
        ),
        -25.0,
        25.0,
    )
    y = np.asarray(oof_labels, dtype=np.float64)
    pos_m = y > 0.5
    n_pos = float(pos_m.sum())
    n_neg = float((~pos_m).sum())
    if n_pos < 1.0 or n_neg < 1.0:
        return 1.0, 0.0

    v = None
    if oof_centers is not None:
        c_arr = np.asarray(oof_centers, dtype=str)
        strata_masks = []
        for cid in sorted(np.unique(c_arr).tolist()):
            c_m = c_arr == cid
            m_pos = c_m & pos_m
            m_neg = c_m & (~pos_m)
            if int(m_pos.sum()) > 0 and int(m_neg.sum()) > 0:
                strata_masks.append(m_pos)
                strata_masks.append(m_neg)
        if len(strata_masks) >= 2:
            v = np.zeros_like(z, dtype=np.float64)
            stratum_w = 1.0 / float(len(strata_masks))
            for sm in strata_masks:
                v[sm] = stratum_w / float(sm.sum())
    if v is None:
        v = np.where(pos_m, 0.5 / n_pos, 0.5 / n_neg)

    def _temp_obj_and_grad(theta_t: np.ndarray):
        alpha = float(theta_t[0])
        u = np.clip(alpha * z, -35.0, 35.0)
        nll = np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u))) - y * u
        loss = float(np.sum(v * nll)) + 0.5 * l2_scale * ((alpha - 1.0) ** 2)
        p = _stable_sigmoid_np(u)
        grad_a = float(np.sum(v * (p - y) * z)) + l2_scale * (alpha - 1.0)
        return loss, np.array([grad_a], dtype=np.float64)

    def _obj_and_grad(theta: np.ndarray):
        alpha, beta = float(theta[0]), float(theta[1])
        u = np.clip(alpha * z + beta, -35.0, 35.0)
        nll = np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u))) - y * u
        loss = (
            float(np.sum(v * nll))
            + 0.5 * l2_scale * ((alpha - 1.0) ** 2)
            + 0.5 * l2_bias * (beta**2)
        )
        p = _stable_sigmoid_np(u)
        diff = v * (p - y)
        grad = np.array(
            [
                float(np.sum(diff * z)) + l2_scale * (alpha - 1.0),
                float(np.sum(diff)) + l2_bias * beta,
            ],
            dtype=np.float64,
        )
        return loss, grad

    res_temp = minimize(
        _temp_obj_and_grad,
        x0=np.array([1.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.35, 3.0)],
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 200},
    )
    alpha_temp = float(np.clip(res_temp.x[0], 0.35, 3.0))
    temp_only = 1.0 / alpha_temp

    res_full = minimize(
        _obj_and_grad,
        x0=np.array([alpha_temp, 0.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.35, 3.0), (-0.75, 0.75)],
        options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 200},
    )
    alpha_opt = float(np.clip(res_full.x[0], 0.35, 3.0))
    beta_opt = float(np.clip(res_full.x[1], -0.75, 0.75))
    temp_full = 1.0 / alpha_opt
    bias_full = beta_opt * temp_full

    raw_probs = np.clip(_stable_sigmoid_np(z), 1e-6, 1.0 - 1e-6)
    full_probs = np.clip(
        _stable_sigmoid_np(apply_platt_log_odds(z, temp_full, bias_full)),
        1e-6,
        1.0 - 1e-6,
    )
    eq_raw = compute_equal_center_oof_metrics(y.astype(int), raw_probs, oof_centers)
    eq_full = compute_equal_center_oof_metrics(y.astype(int), full_probs, oof_centers)
    comp_raw = (
        0.70 * eq_raw["balanced_accuracy"] + 0.30 * eq_raw["worst_center_ba"]
    )
    comp_full = (
        0.70 * eq_full["balanced_accuracy"] + 0.30 * eq_full["worst_center_ba"]
    )

    if (
        (comp_full < comp_raw + 3e-4)
        or (eq_full["balanced_accuracy"] < eq_raw["balanced_accuracy"] - 1e-4)
        or (eq_full["worst_center_ba"] < eq_raw["worst_center_ba"] - 1e-3)
    ):
        raw_nll, _ = _temp_obj_and_grad(np.array([1.0], dtype=np.float64))
        temp_nll, _ = _temp_obj_and_grad(np.array([alpha_temp], dtype=np.float64))
        if temp_nll < raw_nll - 1e-5:
            return float(temp_only), 0.0
        return 1.0, 0.0

    return float(temp_full), float(bias_full)


def compute_oof_ensemble_weights(
    oof_metrics_list: List[Dict[str, float]],
) -> List[float]:
    """Compute bounded [0.20, 0.80] family-level OOF ensemble weights summing to 1.0."""
    n_models = len(oof_metrics_list)
    if n_models <= 1:
        return [1.0] * n_models
    scores = np.array(
        [
            0.55 * float(m["balanced_accuracy"])
            + 0.25 * float(m.get("worst_center_ba", m["balanced_accuracy"]))
            + 0.20 * float(m["auroc"])
            for m in oof_metrics_list
        ],
        dtype=np.float64,
    )
    logits_w = np.clip((scores - scores.max()) / 0.012, -10.0, 0.0)
    w = np.exp(logits_w)
    w = w / w.sum()
    if n_models == 2:
        w = np.clip(w, 0.20, 0.80)
    else:
        w = np.clip(w, 0.40 / float(n_models), 1.60 / float(n_models))
    w = w / w.sum()
    return [float(x) for x in w]


def compute_family_balanced_fold_weights(
    fold_families: List[str],
    fold_oof_probs: List[np.ndarray],
    fold_oof_labels: List[np.ndarray],
    fold_oof_centers: List[np.ndarray],
) -> List[float]:
    """
    Pools OOF predictions within each architecture family, computes bounded [0.20, 0.80]
    family weights from pooled equal-center OOF metrics, and splits uniformly across folds
    within each family to eliminate single-fold difficulty skew.
    """
    n_folds = len(fold_families)
    if n_folds <= 1:
        return [1.0] * n_folds

    unique_fams: List[str] = []
    fam_to_indices: Dict[str, List[int]] = {}
    for idx, fam in enumerate(fold_families):
        if fam not in fam_to_indices:
            unique_fams.append(fam)
            fam_to_indices[fam] = []
        fam_to_indices[fam].append(idx)

    if len(unique_fams) <= 1:
        return [1.0 / float(n_folds)] * n_folds

    fam_metrics = []
    for fam in unique_fams:
        idxs = fam_to_indices[fam]
        p_fam = np.concatenate([fold_oof_probs[i] for i in idxs], axis=0)
        y_fam = np.concatenate([fold_oof_labels[i] for i in idxs], axis=0)
        c_fam = np.concatenate([fold_oof_centers[i] for i in idxs], axis=0)
        fam_metrics.append(compute_equal_center_oof_metrics(y_fam, p_fam, c_fam))

    fam_weights = compute_oof_ensemble_weights(fam_metrics)
    fold_weights = [0.0] * n_folds
    for fam, fw in zip(unique_fams, fam_weights):
        idxs = fam_to_indices[fam]
        per_fold_w = float(fw) / float(len(idxs))
        for i in idxs:
            fold_weights[i] = per_fold_w
    return fold_weights


class WarmupGatedEMA:
    """
    Maintains a deepcopy-based exponential moving average of PathologyMultiScaleStainNet
    weights (max_decay=0.995). Because the architecture uses exclusively LayerNorm,
    GroupNorm, and GRN (zero BatchNorm), weight-space EMA has zero running-statistic
    mismatch on unseen medical centers.
    """

    def __init__(
        self, source_model: PathologyMultiScaleStainNet, max_decay: float = 0.995
    ):
        self.max_decay = max_decay
        self.num_updates = 0
        self.ema_model = copy.deepcopy(source_model).to(DEVICE)
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
            self.max_decay, (1.0 + self.num_updates) / (2.0 + self.num_updates)
        )
        one_minus_decay = 1.0 - decay
        for ema_p, src_p in zip(self.ema_model.parameters(), source_model.parameters()):
            ema_p.mul_(decay).add_(src_p.detach(), alpha=one_minus_decay)
        for ema_b, src_b in zip(self.ema_model.buffers(), source_model.buffers()):
            ema_b.copy_(src_b.detach())


def _get_d4_ops(tta_views: int) -> List[Tuple[bool, int]]:
    if tta_views >= 8:
        return [
            (False, 0),
            (False, 1),
            (False, 2),
            (False, 3),
            (True, 0),
            (True, 1),
            (True, 2),
            (True, 3),
        ]
    if tta_views == 4:
        return [(False, 0), (False, 1), (False, 2), (True, 0)]
    return [(False, 0)]


def _stack_d4_views(
    rgb_mb: torch.Tensor, d4_ops: List[Tuple[bool, int]]
) -> torch.Tensor:
    views = []
    for do_flip, k_rot in d4_ops:
        v = torch.flip(rgb_mb, dims=[-1]) if do_flip else rgb_mb
        if k_rot > 0:
            v = torch.rot90(v, k=k_rot, dims=[-2, -1])
        views.append(v)
    return torch.cat(views, dim=0)


@torch.no_grad()
def predict_d4_tta(
    models: List[nn.Module],
    frontend: HistopathologyPhysicsFrontend,
    loader: DataLoader,
    expected_ids: List[str],
    tta_views: int = TTA_VIEWS,
    return_logits: bool = False,
    platt_params: Optional[List[Tuple[float, float]]] = None,
    model_weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Single-pass micro-batched multi-view D4 dihedral Test-Time Augmentation in logit space.
    Averages D4 views in logit space per model, applies optional Platt calibration
    u_m(z) = (z + b_m) / T_m, fuses weighted log-odds, and returns probabilities or log-odds.
    """
    frontend.eval()
    for m in models:
        m.eval()

    d4_ops = _get_d4_ops(tta_views)
    n_views = len(d4_ops)
    mb_step = max(1, (BATCH_SIZE * 2) // n_views)

    n_m = len(models)
    if platt_params is None:
        cal_list = [(1.0, 0.0)] * n_m
    else:
        cal_list = list(platt_params)
    if model_weights is None:
        w_list = [1.0 / float(n_m)] * n_m
    else:
        w_sum = sum(float(w) for w in model_weights)
        w_list = [float(w) / max(w_sum, 1e-8) for w in model_weights]

    all_log_odds: List[np.ndarray] = []
    collected_ids: List[str] = []

    for batch in loader:
        rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
        bsz = rgb_u8.size(0)
        collected_ids.extend(batch["sample_id"])
        batch_u_gpu = torch.zeros(bsz, device=DEVICE, dtype=torch.float64)

        for start in range(0, bsz, mb_step):
            end = min(bsz, start + mb_step)
            mb = end - start
            stacked_v = _stack_d4_views(rgb_u8[start:end], d4_ops)
            r_v, s_v = frontend(stacked_v)

            mb_u = torch.zeros(mb, device=DEVICE, dtype=torch.float64)
            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    for m_idx, m in enumerate(models):
                        z_all = m(r_v, s_v, return_aux=False).float()
                        z_mean = z_all.view(n_views, mb).mean(dim=0).to(torch.float64)
                        t_m, b_m = cal_list[m_idx]
                        mb_u.add_(
                            (z_mean + float(b_m)) / max(float(t_m), 1e-4),
                            alpha=w_list[m_idx],
                        )
            else:
                for m_idx, m in enumerate(models):
                    z_all = m(r_v, s_v, return_aux=False).float()
                    z_mean = z_all.view(n_views, mb).mean(dim=0).to(torch.float64)
                    t_m, b_m = cal_list[m_idx]
                    mb_u.add_(
                        (z_mean + float(b_m)) / max(float(t_m), 1e-4),
                        alpha=w_list[m_idx],
                    )

            batch_u_gpu[start:end] = mb_u

        all_log_odds.append(batch_u_gpu.cpu().numpy())

    assert (
        collected_ids == expected_ids
    ), "Sample ID ordering mismatch during DataLoader inference!"
    log_odds_arr = np.concatenate(all_log_odds, axis=0).astype(np.float64)
    if return_logits:
        return log_odds_arr
    return np.clip(_stable_sigmoid_np(log_odds_arr), 1e-6, 1.0 - 1e-6)


@torch.no_grad()
def _eval_models_custom_d4_ops(
    models: List[nn.Module],
    frontend: HistopathologyPhysicsFrontend,
    loader: DataLoader,
    expected_ids: List[str],
    d4_ops: List[Tuple[bool, int]],
) -> List[np.ndarray]:
    """Single-pass micro-batched D4-view mean logit evaluation for 1+ models on explicit d4_ops."""
    frontend.eval()
    for m in models:
        m.eval()

    n_views = len(d4_ops)
    n_m = len(models)
    mb_step = max(1, INFER_BATCH_SIZE // n_views)

    all_z_lists: List[List[np.ndarray]] = [[] for _ in range(n_m)]
    collected_ids: List[str] = []

    for batch in loader:
        rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
        bsz = rgb_u8.size(0)
        collected_ids.extend(batch["sample_id"])
        batch_z_gpus = [
            torch.empty(bsz, device=DEVICE, dtype=torch.float64) for _ in range(n_m)
        ]

        for start in range(0, bsz, mb_step):
            end = min(bsz, start + mb_step)
            mb = end - start
            stacked_v = _stack_d4_views(rgb_u8[start:end], d4_ops)
            r_v, s_v = frontend(stacked_v)

            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    for m_idx, m in enumerate(models):
                        z_all = m(r_v, s_v, return_aux=False).float()
                        batch_z_gpus[m_idx][start:end] = (
                            z_all.view(n_views, mb).mean(dim=0).to(torch.float64)
                        )
            else:
                for m_idx, m in enumerate(models):
                    z_all = m(r_v, s_v, return_aux=False).float()
                    batch_z_gpus[m_idx][start:end] = (
                        z_all.view(n_views, mb).mean(dim=0).to(torch.float64)
                    )

        for m_idx in range(n_m):
            all_z_lists[m_idx].append(batch_z_gpus[m_idx].cpu().numpy())

    assert (
        collected_ids == expected_ids
    ), "Sample ID ordering mismatch during custom D4 inference!"
    return [
        np.concatenate(z_list, axis=0).astype(np.float64) for z_list in all_z_lists
    ]


@torch.no_grad()
def _eval_live_and_ema_d4_tta(
    live_net: nn.Module,
    ema_net: nn.Module,
    frontend: HistopathologyPhysicsFrontend,
    loader: DataLoader,
    expected_ids: List[str],
    tta_views: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-pass micro-batched D4-TTA OOF evaluation for LIVE + EMA checkpoints."""
    live_z, ema_z = _eval_models_custom_d4_ops(
        [live_net, ema_net],
        frontend,
        loader,
        expected_ids=expected_ids,
        d4_ops=_get_d4_ops(tta_views),
    )
    live_probs = np.clip(_stable_sigmoid_np(live_z), 1e-6, 1.0 - 1e-6)
    ema_probs = np.clip(_stable_sigmoid_np(ema_z), 1e-6, 1.0 - 1e-6)
    return live_probs, ema_probs


@torch.no_grad()
def run_pipeline_inference(
    models_with_frontends: List[Tuple],
    eval_loader: DataLoader,
    expected_ids: List[str],
    tta_views: int = TTA_VIEWS,
    platt_calibrators: Optional[List[Tuple[float, float]]] = None,
    ensemble_weights: Optional[List[float]] = None,
    return_log_odds: bool = False,
) -> np.ndarray:
    """
    Unified calibrated log-odds prediction pipeline shared identically between pre-refit
    validation evaluation and final test inference: computes micro-batched 8-view D4-TTA
    mean logits z_m, maps to calibrated log-odds u_m(z_m) = (z_m + b_m) / T_m, and returns
    sigmoid(sum_m w_m * u_m) (or fused log-odds when return_log_odds=True).
    """
    if not models_with_frontends:
        raise ValueError("models_with_frontends must be non-empty")

    nets: List[nn.Module] = []
    fes: List[HistopathologyPhysicsFrontend] = []
    parsed_cals: List[Tuple[float, float]] = []
    parsed_weights: List[float] = []

    for idx, entry in enumerate(models_with_frontends):
        nets.append(entry[0])
        fes.append(entry[1])
        if len(entry) >= 3 and entry[2] is not None:
            parsed_cals.append((float(entry[2][0]), float(entry[2][1])))
        elif platt_calibrators is not None and idx < len(platt_calibrators):
            parsed_cals.append(
                (float(platt_calibrators[idx][0]), float(platt_calibrators[idx][1]))
            )
        else:
            parsed_cals.append((1.0, 0.0))

        if len(entry) >= 4 and entry[3] is not None:
            parsed_weights.append(float(entry[3]))
        elif ensemble_weights is not None and idx < len(ensemble_weights):
            parsed_weights.append(float(ensemble_weights[idx]))
        else:
            parsed_weights.append(1.0)

    w_arr = np.asarray(parsed_weights, dtype=np.float64)
    w_arr = w_arr / np.maximum(w_arr.sum(), 1e-8)

    if DEVICE.type == "cuda" and _VRAM_GB < 15.0:
        fused_u = np.zeros(len(expected_ids), dtype=np.float64)
        for m_idx, (net, fe) in enumerate(zip(nets, fes)):
            orig_dev = next(net.parameters()).device
            net.to(DEVICE)
            fe.to(DEVICE)
            u_m = predict_d4_tta(
                [net],
                fe,
                eval_loader,
                expected_ids=expected_ids,
                tta_views=tta_views,
                return_logits=True,
                platt_params=[parsed_cals[m_idx]],
            )
            fused_u += w_arr[m_idx] * u_m
            if orig_dev != DEVICE:
                net.to(orig_dev)
                fe.to(orig_dev)
        if return_log_odds:
            return fused_u
        return np.clip(_stable_sigmoid_np(fused_u), 1e-6, 1.0 - 1e-6)

    orig_devices = [next(net.parameters()).device for net in nets]
    for net, fe in zip(nets, fes):
        net.to(DEVICE).eval()
        fe.to(DEVICE).eval()

    d4_ops = _get_d4_ops(tta_views)
    n_views = len(d4_ops)
    mb_step = max(1, (BATCH_SIZE * 2) // n_views)

    all_fused_u: List[np.ndarray] = []
    collected_ids: List[str] = []

    try:
        for batch in eval_loader:
            rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
            bsz = rgb_u8.size(0)
            collected_ids.extend(batch["sample_id"])
            batch_u_gpu = torch.zeros(bsz, device=DEVICE, dtype=torch.float64)

            for start in range(0, bsz, mb_step):
                end = min(bsz, start + mb_step)
                mb = end - start
                stacked_v = _stack_d4_views(rgb_u8[start:end], d4_ops)
                mb_u = torch.zeros(mb, device=DEVICE, dtype=torch.float64)

                for m_idx, (net, fe) in enumerate(zip(nets, fes)):
                    r_v, s_v = fe(stacked_v)
                    if USE_AMP and DEVICE.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                            z_all = net(r_v, s_v, return_aux=False).float()
                    else:
                        z_all = net(r_v, s_v, return_aux=False).float()
                    z_mean = z_all.view(n_views, mb).mean(dim=0).to(torch.float64)
                    t_m, b_m = parsed_cals[m_idx]
                    mb_u.add_(
                        (z_mean + float(b_m)) / max(float(t_m), 1e-4),
                        alpha=float(w_arr[m_idx]),
                    )

                batch_u_gpu[start:end] = mb_u

            all_fused_u.append(batch_u_gpu.cpu().numpy())
    finally:
        for net, fe, o_dev in zip(nets, fes, orig_devices):
            if o_dev != DEVICE:
                net.to(o_dev)
                fe.to(o_dev)

    assert (
        collected_ids == expected_ids
    ), "Sample ID ordering mismatch during run_pipeline_inference!"
    fused_u = np.concatenate(all_fused_u, axis=0).astype(np.float64)
    if return_log_odds:
        return fused_u
    return np.clip(_stable_sigmoid_np(fused_u), 1e-6, 1.0 - 1e-6)


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


def _composite_center_score(metrics_dict: Dict[str, float]) -> float:
    return 0.70 * float(metrics_dict["balanced_accuracy"]) + 0.30 * float(
        metrics_dict.get("worst_center_ba", metrics_dict["balanced_accuracy"])
    )


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
    patience: int = 3,
    eval_tta_views: int = 1,
    runtime_limit_sec: float = TRAIN_RUNTIME_CEILING_SEC,
    va_centers: Optional[np.ndarray] = None,
) -> Tuple[PathologyMultiScaleStainNet, Dict[str, Union[float, np.ndarray, str]]]:
    """
    Trains PathologyMultiScaleStainNet with paired counterfactual GPU physics views,
    WarmupGatedEMA (max_decay=0.995), fast 1-view composite equal/worst-center BA ranking
    (0.70 * eq_center_ba + 0.30 * worst_center_ba via ./input/metric.py), and OOF-guarded
    Top-2 Weight-Space Model Souping via a split 4+4 D4-view workflow yielding exact
    8-view D4-TTA OOF logits on the winning checkpoint.
    """
    if hasattr(loss_fn, "reset_dro_weights"):
        loss_fn.reset_dro_weights()
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    group_names = [pg.get("name", f"g{i}") for i, pg in enumerate(opt.param_groups)]
    ema_helper = WarmupGatedEMA(net, max_decay=0.995)

    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(USE_SCALER and DEVICE.type == "cuda")
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=(USE_SCALER and DEVICE.type == "cuda")
        )

    total_steps = max(1, num_epochs * len(tr_loader))
    best_comp = -1.0
    best_ba = -1.0
    best_auroc = -1.0
    epochs_no_improve = 0
    global_step = 0
    top_k_ckpts: List[Dict] = []

    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.perf_counter()
        net.train()
        frontend.train()
        running_loss = 0.0
        n_batches = 0

        for batch in tr_loader:
            progress = float(global_step) / float(total_steps)
            _set_differential_step_lr(opt, base_lrs, group_names, progress)

            rgb_u8 = batch["rgb"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)
            center_idx = batch["center_idx"].to(DEVICE, non_blocking=True)

            with torch.no_grad():
                rgb, stain, eff_center_idx, eff_labels, pair_idx = frontend(
                    rgb_u8,
                    center_idx=center_idx,
                    labels=labels,
                    mixup_p=0.30,
                    return_paired=True,
                )

            opt.zero_grad(set_to_none=True)

            if USE_AMP and DEVICE.type == "cuda":
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    outputs = net(
                        rgb,
                        stain,
                        center_idx=eff_center_idx,
                        return_aux=True,
                    )
                    loss = loss_fn(
                        outputs,
                        eff_labels,
                        center_idx=eff_center_idx,
                        pair_idx=pair_idx,
                    )
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
                    center_idx=eff_center_idx,
                    return_aux=True,
                )
                loss = loss_fn(
                    outputs,
                    eff_labels,
                    center_idx=eff_center_idx,
                    pair_idx=pair_idx,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                opt.step()

            ema_helper.update(net, in_warmup=(progress < 0.08))
            running_loss += float(loss.detach().item())
            n_batches += 1
            global_step += 1

        avg_loss = running_loss / max(1, n_batches)

        live_probs, ema_probs = _eval_live_and_ema_d4_tta(
            net,
            ema_helper.ema_model,
            frontend,
            va_loader,
            expected_ids=va_ids,
            tta_views=1,
        )
        m_live = compute_equal_center_oof_metrics(va_labels, live_probs, va_centers)
        m_ema = compute_equal_center_oof_metrics(va_labels, ema_probs, va_centers)
        c_live = _composite_center_score(m_live)
        c_ema = _composite_center_score(m_ema)

        if (c_ema, m_ema["auroc"]) >= (c_live, m_live["auroc"]):
            cand_m = m_ema
            cand_comp = c_ema
            cand_sd_gpu = ema_helper.ema_model.state_dict()
            cand_src = "EMA"
        else:
            cand_m = m_live
            cand_comp = c_live
            cand_sd_gpu = net.state_dict()
            cand_src = "LIVE"

        qualifies_top2 = len(top_k_ckpts) < 2 or (
            cand_comp,
            cand_m["auroc"],
        ) > (top_k_ckpts[-1]["comp"], top_k_ckpts[-1]["auroc"])
        if qualifies_top2:
            cpu_clone_sd = {
                k: v.detach().cpu().clone() for k, v in cand_sd_gpu.items()
            }
            top_k_ckpts.append(
                {
                    "comp": float(cand_comp),
                    "auroc": float(cand_m["auroc"]),
                    "epoch": int(epoch),
                    "src": cand_src,
                    "state_dict": cpu_clone_sd,
                }
            )
            top_k_ckpts.sort(
                key=lambda item: (item["comp"], item["auroc"]), reverse=True
            )
            if len(top_k_ckpts) > 2:
                top_k_ckpts.pop()

        improved = (cand_comp > best_comp + 1e-6) or (
            abs(cand_comp - best_comp) <= 1e-6 and cand_m["auroc"] > best_auroc
        )
        if improved:
            best_comp = cand_comp
            best_ba = cand_m["balanced_accuracy"]
            best_auroc = cand_m["auroc"]
            _save_pure_state_dict(cand_sd_gpu, ckpt_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ep_dt = time.perf_counter() - ep_t0
        print(
            f"[{run_tag}] Epoch {epoch:02d}/{num_epochs:02d} | "
            f"loss={avg_loss:.4f} | eq_ba={cand_m['balanced_accuracy']:.5f} "
            f"(worst_c={cand_m['worst_center_ba']:.5f}, comp={cand_comp:.5f}, "
            f"auroc={cand_m['auroc']:.5f}, {cand_src}, 1v) | best_comp={best_comp:.5f} | {ep_dt:.1f}s"
        )

        if epochs_no_improve >= patience and epoch >= max(4, num_epochs - 1):
            break
        if _elapsed_sec() > runtime_limit_sec and epoch >= 2:
            break

    d4_all_8 = _get_d4_ops(8)
    d4_part1 = [d4_all_8[0], d4_all_8[1], d4_all_8[2], d4_all_8[4]]
    d4_part2 = [d4_all_8[3], d4_all_8[5], d4_all_8[6], d4_all_8[7]]

    if len(top_k_ckpts) >= 2:
        sd_best = top_k_ckpts[0]["state_dict"]
        sd_second = top_k_ckpts[1]["state_dict"]
        sd_soup = {
            k: (
                (0.5 * sd_best[k].float() + 0.5 * sd_second[k].float()).to(
                    dtype=sd_best[k].dtype
                )
                if sd_best[k].is_floating_point()
                else sd_best[k].clone()
            )
            for k in sd_best.keys()
        }

        net.load_state_dict(sd_best)
        soup_net = ema_helper.ema_model
        soup_net.load_state_dict(sd_soup)

        z4_best, z4_soup = _eval_models_custom_d4_ops(
            [net, soup_net], frontend, va_loader, expected_ids=va_ids, d4_ops=d4_part1
        )
        p4_best = np.clip(_stable_sigmoid_np(z4_best), 1e-6, 1.0 - 1e-6)
        p4_soup = np.clip(_stable_sigmoid_np(z4_soup), 1e-6, 1.0 - 1e-6)
        m4_best = compute_equal_center_oof_metrics(va_labels, p4_best, va_centers)
        m4_soup = compute_equal_center_oof_metrics(va_labels, p4_soup, va_centers)
        c4_best = _composite_center_score(m4_best)
        c4_soup = _composite_center_score(m4_soup)

        soup_accepted = (
            (
                (c4_soup > c4_best + 1e-6)
                or (
                    abs(c4_soup - c4_best) <= 1e-6
                    and m4_soup["auroc"] >= m4_best["auroc"]
                )
            )
            and (m4_soup["worst_center_ba"] >= m4_best["worst_center_ba"] - 2e-3)
        )
        ep0, ep1 = top_k_ckpts[0]["epoch"], top_k_ckpts[1]["epoch"]
        if soup_accepted:
            net.load_state_dict(sd_soup)
            _save_pure_state_dict(sd_soup, ckpt_path)
            z4_win = z4_soup
            selected_mode = f"SOUP(ep{ep0}+ep{ep1})"
        else:
            z4_win = z4_best
            selected_mode = f"BEST(ep{ep0})"

        print(
            f"[{run_tag} Soup-4v Guard] best_comp={c4_best:.5f} vs soup_comp={c4_soup:.5f} -> {selected_mode}"
        )
        del soup_net, ema_helper, top_k_ckpts
        (z4_rem,) = _eval_models_custom_d4_ops(
            [net], frontend, va_loader, expected_ids=va_ids, d4_ops=d4_part2
        )
        oof_logits_8v = 0.5 * (z4_win + z4_rem)
    else:
        del ema_helper, top_k_ckpts
        best_sd = _load_pure_state_dict(ckpt_path, DEVICE)
        net.load_state_dict(best_sd)
        selected_mode = "BEST"
        oof_logits_8v = predict_d4_tta(
            [net],
            frontend,
            va_loader,
            expected_ids=va_ids,
            tta_views=TTA_VIEWS,
            return_logits=True,
        )

    oof_probs_8v = np.clip(_stable_sigmoid_np(oof_logits_8v), 1e-6, 1.0 - 1e-6)
    final_m = compute_equal_center_oof_metrics(va_labels, oof_probs_8v, va_centers)
    net.eval()
    frontend.eval()
    return net, {
        "balanced_accuracy": final_m["balanced_accuracy"],
        "worst_center_ba": final_m["worst_center_ba"],
        "composite_ba": _composite_center_score(final_m),
        "auroc": final_m["auroc"],
        "oof_logits_8v": oof_logits_8v,
        "selected_mode": selected_mode,
    }


# =============================================================================
# 10. Stage 1: Leak-Free 5-Fold Ensemble Within train.csv & Platt Calibration
# =============================================================================
seed_everything(SEED, seed_cuda=True)

# Pre-extract per-sample (her_mean, her_sq, low_amp) STRICTLY on train.csv (val.csv untouched)
_t_pre = time.perf_counter()
dev_patch_physics_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
preextract_patch_physics_cache(
    train_df, dev_patch_physics_cache, cache_tag="train_only_13200"
)
assert len(dev_patch_physics_cache) == len(train_df)
assert set(dev_patch_physics_cache.keys()).isdisjoint(
    set(val_df["sample_id"].astype(str))
), "Leakage guard: validation.csv must remain 100% untouched during Stage 1!"
print(
    f"Pre-extracted train.csv per-patch physics cache ({len(dev_patch_physics_cache)} patches) "
    f"in {time.perf_counter() - _t_pre:.2f}s"
)

nw_workers = max(2, min(6, os.cpu_count() or 4))
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=nw_workers,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE * 2,
    shuffle=False,
    num_workers=nw_workers,
    pin_memory=torch.cuda.is_available(),
)

val_ids_list = val_df["sample_id"].astype(str).tolist()
test_ids_list = test_df["sample_id"].astype(str).tolist()

# Move unused Step-1/Step-2 global modules to CPU during Fold 0 to conserve GPU VRAM
primary_physics_frontend.cpu()
model.cpu()
criterion.cpu()

stage1_fold_specs = [
    (
        0,
        SWINV2_BACKBONE_NAME,
        5.5e-5,
        3.5e-4,
        0.75,
        SWINV2_EPOCHS,
        "swin",
        "Stage1-F0-SwinV2Base",
    ),
    (
        1,
        CONVNEXT_BACKBONE_NAME,
        5.0e-5,
        3.5e-4,
        0.75,
        CONVNEXT_EPOCHS,
        "convnext",
        "Stage1-F1-ConvNeXtV2Base",
    ),
    (
        2,
        SWINV2_BACKBONE_NAME,
        5.5e-5,
        3.5e-4,
        0.75,
        SWINV2_EPOCHS,
        "swin",
        "Stage1-F2-SwinV2Base",
    ),
    (
        3,
        CONVNEXT_BACKBONE_NAME,
        5.0e-5,
        3.5e-4,
        0.75,
        CONVNEXT_EPOCHS,
        "convnext",
        "Stage1-F3-ConvNeXtV2Base",
    ),
    (
        4,
        SWINV2_BACKBONE_NAME,
        5.5e-5,
        3.5e-4,
        0.75,
        SWINV2_EPOCHS,
        "swin",
        "Stage1-F4-SwinV2Base",
    ),
]

stage1_models_with_fe: List[Tuple[nn.Module, HistopathologyPhysicsFrontend]] = []
stage1_platt_params: List[Tuple[float, float]] = []
stage1_oof_metrics: List[Dict[str, float]] = []
stage1_fold_durations: List[float] = []
stage1_arch_durations: Dict[str, List[float]] = {"convnext": [], "swin": []}
stage1_fold_families: List[str] = []
pooled_oof_probs_list: List[np.ndarray] = []
pooled_oof_labels_list: List[np.ndarray] = []
pooled_oof_centers_list: List[np.ndarray] = []

for k_idx, (
    fold_i,
    bb_name,
    bb_lr,
    hd_lr,
    l_decay,
    fold_epochs,
    arch_fam,
    run_tag,
) in enumerate(stage1_fold_specs):
    fold_budget_deadline = TRAIN_RUNTIME_CEILING_SEC - 260.0

    seed_everything(SEED + 19 * fold_i, seed_cuda=True)
    f_tr_idx, f_va_idx = train_gkf_folds[fold_i]
    fold_tr_df = train_df.iloc[f_tr_idx].reset_index(drop=True)
    fold_va_df = train_df.iloc[f_va_idx].reset_index(drop=True)
    fold_va_labels = y_train[f_va_idx]
    fold_va_centers = fold_va_df["center_id"].astype(str).values
    fold_va_ids = fold_va_df["sample_id"].astype(str).tolist()

    assert set(fold_tr_df["group_id"]).isdisjoint(set(fold_va_df["group_id"]))
    assert set(fold_tr_df["slide_id"]).isdisjoint(set(fold_va_df["slide_id"]))

    fold_stats = build_fold_image_stats_from_cache(
        fold_tr_df,
        dev_patch_physics_cache,
        seed=SEED + 101 + fold_i,
        center_to_idx=CENTER_TO_IDX,
    )
    fold_fe = HistopathologyPhysicsFrontend(fold_stats).to(DEVICE)

    fold_tr_ds = HistopathologyMultiViewDataset(
        fold_tr_df, fold_stats, is_train=True, center_to_idx=CENTER_TO_IDX
    )
    fold_va_ds = HistopathologyMultiViewDataset(
        fold_va_df, fold_stats, is_train=False, center_to_idx=None
    )
    fold_w = compute_balanced_domain_class_weights(fold_tr_df)
    fold_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(fold_w, dtype=torch.double),
        num_samples=len(fold_w),
        replacement=True,
    )
    fold_loader_kw = {
        "num_workers": nw_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": nw_workers > 0,
    }
    if nw_workers > 0:
        fold_loader_kw["prefetch_factor"] = 2

    fold_tr_loader = DataLoader(
        fold_tr_ds,
        batch_size=BATCH_SIZE,
        sampler=fold_sampler,
        drop_last=True,
        **fold_loader_kw,
    )
    fold_va_loader = DataLoader(
        fold_va_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        **fold_loader_kw,
    )

    if fold_i == 1:
        f_net = model.to(DEVICE)
        f_crit = criterion.to(DEVICE)
        f_opt = build_differential_adamw(
            f_net,
            backbone_lr=bb_lr,
            head_lr=hd_lr,
            layer_decay=l_decay,
        )
        optimizer = f_opt
    else:
        f_net, f_crit, f_opt = build_model_criterion_optimizer(
            backbone_name=bb_name,
            num_train_centers=_NUM_TRAIN_CENTERS,
            pretrained=True,
            backbone_lr=bb_lr,
            head_lr=hd_lr,
            layer_decay=l_decay,
            device=DEVICE,
        )

    s1_ckpt_path = os.path.join(WORKING_DIR, f"best_stage1_fold{fold_i}.pt")
    t_fold0 = time.perf_counter()
    f_net, s1_ckpt_info = train_pathology_checkpoint(
        net=f_net,
        frontend=fold_fe,
        loss_fn=f_crit,
        opt=f_opt,
        tr_loader=fold_tr_loader,
        va_loader=fold_va_loader,
        va_labels=fold_va_labels,
        va_ids=fold_va_ids,
        num_epochs=fold_epochs,
        ckpt_path=s1_ckpt_path,
        run_tag=run_tag,
        patience=fold_epochs,
        eval_tta_views=1,
        runtime_limit_sec=fold_budget_deadline,
        va_centers=fold_va_centers,
    )

    # Reuse exact 8-view D4-TTA OOF mean logits from split 4+4 Soup evaluation
    oof_logits = np.asarray(s1_ckpt_info["oof_logits_8v"], dtype=np.float64)
    fold_total_sec = time.perf_counter() - t_fold0
    stage1_fold_durations.append(fold_total_sec)
    stage1_arch_durations.setdefault(arch_fam, []).append(fold_total_sec)

    raw_oof_probs = np.clip(_stable_sigmoid_np(oof_logits), 1e-6, 1.0 - 1e-6)
    raw_oof_m = compute_equal_center_oof_metrics(
        fold_va_labels, raw_oof_probs, fold_va_centers
    )

    t_m, b_m = fit_class_balanced_platt_scaler(
        oof_logits,
        fold_va_labels,
        oof_centers=fold_va_centers,
        l2_scale=0.02,
        l2_bias=0.05,
    )
    cal_oof_u = apply_platt_log_odds(oof_logits, t_m, b_m)
    cal_oof_probs = np.clip(_stable_sigmoid_np(cal_oof_u), 1e-6, 1.0 - 1e-6)
    cal_oof_m = compute_equal_center_oof_metrics(
        fold_va_labels, cal_oof_probs, fold_va_centers
    )

    print(
        f"[{run_tag} 8v-OOF Calibration] mode={s1_ckpt_info['selected_mode']} | "
        f"raw_eq_ba={raw_oof_m['balanced_accuracy']:.5f} -> "
        f"platt_eq_ba={cal_oof_m['balanced_accuracy']:.5f} (worst_c={cal_oof_m['worst_center_ba']:.5f}, "
        f"auroc={cal_oof_m['auroc']:.5f}, T={t_m:.4f}, b={b_m:+.4f}, {fold_total_sec:.1f}s)"
    )

    f_net.cpu()
    fold_fe.cpu()
    stage1_models_with_fe.append((f_net, fold_fe))
    stage1_platt_params.append((t_m, b_m))
    stage1_oof_metrics.append(cal_oof_m)
    stage1_fold_families.append(arch_fam)
    pooled_oof_probs_list.append(cal_oof_probs)
    pooled_oof_labels_list.append(fold_va_labels)
    pooled_oof_centers_list.append(fold_va_centers)

    if fold_i == 1:
        criterion.cpu()
    else:
        del f_crit, f_opt
    del fold_tr_loader, fold_va_loader, fold_tr_ds, fold_va_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

stage1_oof_weights = compute_family_balanced_fold_weights(
    stage1_fold_families,
    pooled_oof_probs_list,
    pooled_oof_labels_list,
    pooled_oof_centers_list,
)
pooled_oof_m = compute_equal_center_oof_metrics(
    np.concatenate(pooled_oof_labels_list, axis=0),
    np.concatenate(pooled_oof_probs_list, axis=0),
    np.concatenate(pooled_oof_centers_list, axis=0),
)
print(
    f"[Stage1 Pooled Train-OOF Audit] eq_center_ba={pooled_oof_m['balanced_accuracy']:.6f} | "
    f"worst_center_ba={pooled_oof_m['worst_center_ba']:.6f} | "
    f"pooled_ba={pooled_oof_m['pooled_ba']:.6f} | auroc={pooled_oof_m['auroc']:.6f} | "
    f"weights={[round(w, 4) for w in stage1_oof_weights]}"
)

# Evaluate untouched validation.csv pre-refit via run_pipeline_inference(sigmoid(sum_m w_m * u_m))
val_probs_official = run_pipeline_inference(
    models_with_frontends=stage1_models_with_fe,
    eval_loader=val_loader,
    expected_ids=val_ids_list,
    tta_views=TTA_VIEWS,
    platt_calibrators=stage1_platt_params,
    ensemble_weights=stage1_oof_weights,
)
official_val_metrics = binary_metrics(y_val, val_probs_official)
validation_balanced_accuracy = official_val_metrics["balanced_accuracy"]
validation_auroc = official_val_metrics["auroc"]

_val_center_strs = val_df["center_id"].astype(str).values
_per_center_ba = {
    cid: binary_metrics(
        y_val[_val_center_strs == cid], val_probs_official[_val_center_strs == cid]
    )["balanced_accuracy"]
    for cid in sorted(np.unique(_val_center_strs).tolist())
}
print(
    f"[Pre-Refit Official Validation] balanced_accuracy={validation_balanced_accuracy:.6f} | "
    f"auroc={validation_auroc:.6f} | accuracy={official_val_metrics['accuracy']:.6f} | "
    f"per_center_ba={_per_center_ba} | n={official_val_metrics['n_samples']} (fixed threshold=0.5)"
)

# Compute Stage-1 calibrated test log-odds & probabilities via identical run_pipeline_inference
_t_s1_test0 = time.perf_counter()
test_log_odds_stage1 = run_pipeline_inference(
    models_with_frontends=stage1_models_with_fe,
    eval_loader=test_loader,
    expected_ids=test_ids_list,
    tta_views=TTA_VIEWS,
    platt_calibrators=stage1_platt_params,
    ensemble_weights=stage1_oof_weights,
    return_log_odds=True,
)
s1_per_model_test_sec = max(
    20.0,
    (time.perf_counter() - _t_s1_test0) / float(max(1, len(stage1_models_with_fe))),
)
test_probs_stage1 = np.clip(
    _stable_sigmoid_np(test_log_odds_stage1), 1e-6, 1.0 - 1e-6
)

# Release auxiliary Stage-1 fold models from memory prior to Stage-2 full-dev refit
for _extra_net, _extra_fe in stage1_models_with_fe:
    del _extra_fe
    if _extra_net is not model:
        del _extra_net
stage1_models_with_fe = []
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =============================================================================
# 11. Stage 2: Paired 2-Fold Full-Dev Refit (train.csv + validation.csv = 16,800)
# =============================================================================
merged_scale = float(len(merged_dev_df)) / float(len(train_df))
avg_s1_fold_sec = (
    float(np.mean(stage1_fold_durations)) if stage1_fold_durations else 360.0
)
arch_est_4ep_sec = {
    fam: max(90.0, float(np.mean(durs)) * merged_scale)
    for fam, durs in stage1_arch_durations.items()
    if len(durs) > 0
}
swin_4ep_est = arch_est_4ep_sec.get("swin", max(90.0, avg_s1_fold_sec * merged_scale))
conv_4ep_est = arch_est_4ep_sec.get(
    "convnext", max(90.0, avg_s1_fold_sec * merged_scale)
)
first_refit_3ep_est = 0.77 * swin_4ep_est
remaining_sec = TRAIN_RUNTIME_CEILING_SEC - _elapsed_sec()
can_run_stage2 = remaining_sec >= (
    0.85 * first_refit_3ep_est + s1_per_model_test_sec + 45.0
)

if can_run_stage2:
    # Pre-refit validation is already recorded; extend cache to full merged_dev_df (16,800)
    preextract_patch_physics_cache(
        val_df, dev_patch_physics_cache, cache_tag="val_ext_3600"
    )
    assert len(dev_patch_physics_cache) == len(merged_dev_df)

    nw_refit = max(2, min(6, os.cpu_count() or 4))
    refit_loader_kw = {
        "num_workers": nw_refit,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": nw_refit > 0,
    }
    if nw_refit > 0:
        refit_loader_kw["prefetch_factor"] = 2

    # Dynamically pace 3 vs 4 epochs so the paired [SwinV2-Base, ConvNeXtV2-Base] refit completes cleanly
    pair_reserve_sec = 2.0 * s1_per_model_test_sec + 45.0
    avail_pair_sec = TRAIN_RUNTIME_CEILING_SEC - _elapsed_sec() - pair_reserve_sec
    pair_4ep_est = swin_4ep_est + conv_4ep_est
    paced_refit_epochs = 4 if avail_pair_sec >= 1.05 * pair_4ep_est else 3
    print(
        f"[Stage2 Pacing] avail_train_sec={avail_pair_sec:.1f}s | "
        f"pair_4ep_est={pair_4ep_est:.1f}s -> target_epochs={paced_refit_epochs}"
    )

    refit_backbones = [
        (
            SWINV2_BACKBONE_NAME,
            5.5e-5,
            3.5e-4,
            0.75,
            "swin",
            "SwinV2Base",
        ),
        (
            CONVNEXT_BACKBONE_NAME,
            5.0e-5,
            3.5e-4,
            0.75,
            "convnext",
            "ConvNeXtV2Base",
        ),
    ]
    refit_entries: List[Tuple[nn.Module, HistopathologyPhysicsFrontend]] = []
    refit_platt_params: List[Tuple[float, float]] = []
    refit_families: List[str] = []
    refit_oof_probs_list: List[np.ndarray] = []
    refit_oof_labels_list: List[np.ndarray] = []
    refit_oof_centers_list: List[np.ndarray] = []

    for r_fold in range(min(2, len(merged_gkf_folds))):
        (
            bb_choice,
            bb_lr,
            hd_lr,
            l_decay,
            r_fam,
            bb_short,
        ) = refit_backbones[r_fold]
        fam_4ep_sec = arch_est_4ep_sec.get(
            r_fam, max(90.0, avg_s1_fold_sec * merged_scale)
        )
        post_test_reserve_sec = (
            float(len(refit_entries) + 1) * s1_per_model_test_sec + 45.0
        )
        rem_for_fold = TRAIN_RUNTIME_CEILING_SEC - _elapsed_sec() - post_test_reserve_sec

        if paced_refit_epochs == 4 and (
            r_fold == 0 or rem_for_fold >= 1.02 * fam_4ep_sec
        ):
            r_epochs = 4
            this_est_sec = fam_4ep_sec
        else:
            r_epochs = 3
            this_est_sec = 0.77 * fam_4ep_sec

        if rem_for_fold < 0.82 * this_est_sec:
            break

        seed_everything(SEED + 100 * (r_fold + 1), seed_cuda=True)
        m_tr_idx, m_va_idx = merged_gkf_folds[r_fold]
        fold_tr_df = merged_dev_df.iloc[m_tr_idx].reset_index(drop=True)
        fold_va_df = merged_dev_df.iloc[m_va_idx].reset_index(drop=True)
        fold_va_labels = y_merged[m_va_idx]
        fold_va_centers = fold_va_df["center_id"].astype(str).values
        fold_va_ids = fold_va_df["sample_id"].astype(str).tolist()

        assert set(fold_tr_df["group_id"]).isdisjoint(set(fold_va_df["group_id"]))
        assert set(fold_tr_df["slide_id"]).isdisjoint(set(fold_va_df["slide_id"]))

        fold_image_stats = build_fold_image_stats_from_cache(
            fold_tr_df,
            dev_patch_physics_cache,
            seed=SEED + 777 + r_fold,
            center_to_idx=CENTER_TO_IDX,
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
            drop_last=True,
            **refit_loader_kw,
        )
        fold_va_loader = DataLoader(
            fold_va_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            **refit_loader_kw,
        )

        f_net, f_crit, f_opt = build_model_criterion_optimizer(
            backbone_name=bb_choice,
            num_train_centers=_NUM_TRAIN_CENTERS,
            pretrained=True,
            backbone_lr=bb_lr,
            head_lr=hd_lr,
            layer_decay=l_decay,
            device=DEVICE,
        )
        fold_ckpt_path = os.path.join(WORKING_DIR, f"best_refit_fold{r_fold}.pt")
        r_tag = f"RefitF{r_fold}-{bb_short}"
        t_rfold0 = time.perf_counter()
        refit_fold_deadline = TRAIN_RUNTIME_CEILING_SEC - post_test_reserve_sec
        f_net, r_ckpt_info = train_pathology_checkpoint(
            net=f_net,
            frontend=fold_frontend,
            loss_fn=f_crit,
            opt=f_opt,
            tr_loader=fold_tr_loader,
            va_loader=fold_va_loader,
            va_labels=fold_va_labels,
            va_ids=fold_va_ids,
            num_epochs=r_epochs,
            ckpt_path=fold_ckpt_path,
            run_tag=r_tag,
            patience=r_epochs,
            eval_tta_views=1,
            runtime_limit_sec=refit_fold_deadline,
            va_centers=fold_va_centers,
        )

        refit_oof_logits = np.asarray(
            r_ckpt_info["oof_logits_8v"], dtype=np.float64
        )
        actual_r_sec = time.perf_counter() - t_rfold0

        t_r, b_r = fit_class_balanced_platt_scaler(
            refit_oof_logits,
            fold_va_labels,
            oof_centers=fold_va_centers,
            l2_scale=0.02,
            l2_bias=0.05,
        )
        refit_cal_probs = np.clip(
            _stable_sigmoid_np(apply_platt_log_odds(refit_oof_logits, t_r, b_r)),
            1e-6,
            1.0 - 1e-6,
        )
        refit_cal_m = compute_equal_center_oof_metrics(
            fold_va_labels, refit_cal_probs, fold_va_centers
        )
        print(
            f"[{r_tag} 8v-OOF Calibration] mode={r_ckpt_info['selected_mode']} | "
            f"platt_eq_ba={refit_cal_m['balanced_accuracy']:.5f} "
            f"(worst_c={refit_cal_m['worst_center_ba']:.5f}, auroc={refit_cal_m['auroc']:.5f}, "
            f"T={t_r:.4f}, b={b_r:+.4f}, {actual_r_sec:.1f}s)"
        )

        f_net.cpu()
        fold_frontend.cpu()
        refit_entries.append((f_net, fold_frontend))
        refit_platt_params.append((t_r, b_r))
        refit_families.append(r_fam)
        refit_oof_probs_list.append(refit_cal_probs)
        refit_oof_labels_list.append(fold_va_labels)
        refit_oof_centers_list.append(fold_va_centers)

        del f_crit, f_opt, fold_tr_loader, fold_va_loader, fold_tr_ds, fold_va_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(refit_entries) > 0:
        refit_oof_weights = compute_family_balanced_fold_weights(
            refit_families,
            refit_oof_probs_list,
            refit_oof_labels_list,
            refit_oof_centers_list,
        )
        test_log_odds_refit = run_pipeline_inference(
            models_with_frontends=refit_entries,
            eval_loader=test_loader,
            expected_ids=test_ids_list,
            tta_views=TTA_VIEWS,
            platt_calibrators=refit_platt_params,
            ensemble_weights=refit_oof_weights,
            return_log_odds=True,
        )
        n_refit = float(len(refit_entries))
        w_refit_stage = (1.35 * n_refit) / (1.35 * n_refit + 3.5)
        print(
            f"[Stage2 Full-Dev Refit Blend] n_refit={len(refit_entries)} | "
            f"refit_weights={[round(w, 4) for w in refit_oof_weights]} | "
            f"stage2_blend_w={w_refit_stage:.4f}"
        )
        final_test_log_odds = (
            w_refit_stage * test_log_odds_refit
            + (1.0 - w_refit_stage) * test_log_odds_stage1
        )
        final_test_probs = np.clip(
            _stable_sigmoid_np(final_test_log_odds), 1e-6, 1.0 - 1e-6
        )
    else:
        final_test_probs = test_probs_stage1
else:
    final_test_probs = test_probs_stage1

final_test_probs = np.clip(
    np.asarray(final_test_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6
)


# =============================================================================
# 12. Save & Strictly Audit ./submission/submission_5faf770c419f4027b238672a7f57addf.csv + Final Score Output
# =============================================================================
os.makedirs(SUBMISSION_DIR, exist_ok=True)
submission_path = os.path.join(SUBMISSION_DIR, "submission_5faf770c419f4027b238672a7f57addf.csv")

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