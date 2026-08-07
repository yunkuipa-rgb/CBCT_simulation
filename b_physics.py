#!/usr/bin/env python3
"""
python b_physics.py \
  --ct pCT/2001.nii.gz \
  --cbct CBCT/2001.nii.gz \
  --output res_phys/phys_2001_3.nii.gz \
  --artifact-scale 4.0 \
  --pl-gamma 3.0 \
  --pl-clip-limit 1.5 \
  --proj-noise-std 2.0 \
  --sart-iters 1

Single-file NIfTI-in / NIfTI-out pseudo-CBCT simulator inspired by:

Dahiya et al., "Multitask 3D CBCT-to-CT Translation and Organs-at-Risk
Segmentation Using Physics-Based Data Augmentation", Med Phys 2021.

Pipeline reproduced at a practical level:
  1) Load planning CT and baseline CBCT
  2) Resample CBCT to CT grid if needed
  3) Approximate PL-AHE-based artifact extraction from registered CBCT
  4) Add artifact-only image to CT
  5) Rescale to [0,1]
  6) Generate 2D slice-wise projections + add Gaussian noise
  7) Reconstruct using SART (slice-wise stand-in for OS-SART)
  8) Save pseudo-CBCT as NIfTI

Notes:
- The paper's full pipeline uses deformable registration, artifact extraction with
  PL-AHE, projection generation, and OS-SART reconstruction.
- This script is a single-file practical approximation in pure Python.
- For best results, input CBCT should already be aligned to CT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from skimage import exposure
from skimage.transform import radon, iradon_sart


# ============================================================
# Config
# ============================================================

@dataclass
class SimConfig:
    # HU clipping
    ct_clip: Tuple[float, float] = (-1000.0, 1500.0)
    cbct_clip: Tuple[float, float] = (-1000.0, 1500.0)

    # Simple body mask threshold
    air_threshold_hu: float = -700.0

    # PL-AHE approximation parameters
    pl_ahe_kernel_frac: float = 0.5   # fraction of in-plane size
    pl_ahe_clip_limit: float = 1.5
    pl_gamma: float = 3.0               # "power-law" part
    artifact_scale: float = 4.0

    # Optional frequency smoothing on extracted artifact
    artifact_smooth_sigma_xy: float = 1.0

    # Projection / reconstruction
    n_angles: int = 80
    gaussian_proj_noise_std: float = 1.0
    sart_iterations: int = 2

    # Output remap
    output_hu_min: float = -1000.0
    output_hu_max: float = 1000.0

    # If enabled, preserve outside-body air from original CT
    preserve_air: bool = True


# ============================================================
# IO
# ============================================================

def load_nifti(path: str) -> nib.Nifti1Image:
    return nib.load(path)


def get_data(img: nib.Nifti1Image) -> np.ndarray:
    return img.get_fdata(dtype=np.float32)


def save_like(ref_img: nib.Nifti1Image, array: np.ndarray, out_path: str) -> None:
    out = nib.Nifti1Image(array.astype(np.float32), ref_img.affine, ref_img.header.copy())
    nib.save(out, out_path)


# ============================================================
# Helpers
# ============================================================

def clip_hu(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def normalize_to_unit(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip(x, lo, hi)
    return (x - lo) / max(hi - lo, 1e-8)


def unit_to_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x * (hi - lo) + lo


def compute_body_mask(ct_hu: np.ndarray, thresh: float) -> np.ndarray:
    return ct_hu > thresh


def maybe_resample_to_ref(
    moving: np.ndarray,
    moving_img: nib.Nifti1Image,
    ref: np.ndarray,
    ref_img: nib.Nifti1Image,
) -> np.ndarray:
    """
    Lightweight resampling by voxel-size ratio + shape correction.
    This is not deformable registration.
    """
    if moving.shape == ref.shape:
        return moving

    mv_zooms = moving_img.header.get_zooms()[:3]
    ref_zooms = ref_img.header.get_zooms()[:3]

    scale = tuple(mz / rz for mz, rz in zip(mv_zooms, ref_zooms))
    out = zoom(moving, zoom=scale, order=1)

    # Center crop/pad to exact ref shape
    result = np.zeros(ref.shape, dtype=np.float32)
    src = out
    dst_slices = []
    src_slices = []

    for a, b in zip(src.shape, ref.shape):
        if a >= b:
            start_src = (a - b) // 2
            src_slices.append(slice(start_src, start_src + b))
            dst_slices.append(slice(0, b))
        else:
            start_dst = (b - a) // 2
            src_slices.append(slice(0, a))
            dst_slices.append(slice(start_dst, start_dst + a))

    result[tuple(dst_slices)] = src[tuple(src_slices)]
    return result


# ============================================================
# Artifact extraction (PL-AHE-inspired approximation)
# ============================================================

def pl_ahe_artifact_extract_2d(
    cbct_slice_hu: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """
    Approximate 'artifact-only' extraction using:
      1) normalize slice to [0,1]
      2) power-law transform
      3) adaptive histogram equalization
      4) subtract smoothed/base version to isolate artifact-like content

    This is a practical stand-in for the paper's PL-AHE artifact extraction.
    """
    x = normalize_to_unit(cbct_slice_hu, cfg.cbct_clip[0], cfg.cbct_clip[1])

    # Power-law intensity remap
    x_pow = np.power(np.clip(x, 1e-6, 1.0), cfg.pl_gamma)

    h, w = x_pow.shape
    kernel_h = max(8, int(round(h * cfg.pl_ahe_kernel_frac)))
    kernel_w = max(8, int(round(w * cfg.pl_ahe_kernel_frac)))

    # AHE-enhanced image
    x_ahe = exposure.equalize_adapthist(
        x_pow,
        kernel_size=(kernel_h, kernel_w),
        clip_limit=cfg.pl_ahe_clip_limit,
    ).astype(np.float32)

    # Low-frequency base
    x_base = gaussian_filter(x_pow, sigma=6.0)

    # Artifact-like residual
    artifact = x_ahe - x_base
    artifact = gaussian_filter(artifact, sigma=cfg.artifact_smooth_sigma_xy)

    return artifact.astype(np.float32)


def extract_artifact_volume(cbct_hu: np.ndarray, cfg: SimConfig) -> np.ndarray:
    art = np.zeros_like(cbct_hu, dtype=np.float32)
    for z in range(cbct_hu.shape[2]):
        art[..., z] = pl_ahe_artifact_extract_2d(cbct_hu[..., z], cfg)
    return art


# ============================================================
# CT + artifact composition
# ============================================================

def compose_artifact_induced_ct(
    ct_hu: np.ndarray,
    artifact_vol_unitish: np.ndarray,
    cfg: SimConfig,
) -> np.ndarray:
    """
    Add extracted artifact field to CT.
    The paper describes pixelwise addition in overlapped regions.
    """
    ct_unit = normalize_to_unit(ct_hu, cfg.ct_clip[0], cfg.ct_clip[1])

    # Robustly scale artifact before addition
    p = np.percentile(np.abs(artifact_vol_unitish), 99.0)
    p = max(float(p), 1e-6)
    artifact_scaled = (artifact_vol_unitish / p) * 0.15 * cfg.artifact_scale

    combined = np.clip(ct_unit + artifact_scaled, 0.0, 1.0)
    return combined.astype(np.float32)


# ============================================================
# Slice-wise projection + noisy SART reconstruction
# ============================================================

def center_crop_or_pad(img: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """
    Center-crop or center-pad a 2D image to out_shape.
    """
    oh, ow = out_shape
    h, w = img.shape

    out = np.zeros((oh, ow), dtype=img.dtype)

    # source region
    src_y0 = max(0, (h - oh) // 2)
    src_x0 = max(0, (w - ow) // 2)
    src_y1 = min(h, src_y0 + oh)
    src_x1 = min(w, src_x0 + ow)

    # destination region
    dst_y0 = max(0, (oh - h) // 2)
    dst_x0 = max(0, (ow - w) // 2)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def reconstruct_slice_sart(
    img_unit: np.ndarray,
    n_angles: int,
    proj_noise_std: float,
    sart_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Slice-wise stand-in for projection generation + OS-SART.
    Always returns the same HxW shape as img_unit.
    """
    in_shape = img_unit.shape
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    sinogram = radon(img_unit, theta=theta, circle=False).astype(np.float32)

    if proj_noise_std > 0:
        sinogram = sinogram + rng.normal(
            0.0, proj_noise_std, size=sinogram.shape
        ).astype(np.float32)

    recon = None
    for _ in range(sart_iterations):
        recon = iradon_sart(sinogram, theta=theta, image=recon)
        recon = np.clip(recon, 0.0, 1.0).astype(np.float32)

    # iradon_sart may return a larger square image when circle=False
    if recon.shape != in_shape:
        recon = center_crop_or_pad(recon, in_shape)

    return recon.astype(np.float32)


def reconstruct_volume_sart(
    vol_unit: np.ndarray,
    cfg,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(vol_unit, dtype=np.float32)

    for z in range(vol_unit.shape[2]):
        recon_z = reconstruct_slice_sart(
            vol_unit[..., z],
            n_angles=cfg.n_angles,
            proj_noise_std=cfg.gaussian_proj_noise_std,
            sart_iterations=cfg.sart_iterations,
            rng=rng,
        )
        out[..., z] = recon_z

    return out


# ============================================================
# Main pipeline
# ============================================================

def simulate_pscbct(
    ct_hu: np.ndarray,
    cbct_hu: np.ndarray,
    cfg: SimConfig,
    seed: Optional[int] = None,
) -> np.ndarray:
    ct_hu = clip_hu(ct_hu.astype(np.float32), *cfg.ct_clip)
    cbct_hu = clip_hu(cbct_hu.astype(np.float32), *cfg.cbct_clip)

    body_mask = compute_body_mask(ct_hu, cfg.air_threshold_hu)

    # 1) Extract artifact-like content from CBCT
    artifact_vol = extract_artifact_volume(cbct_hu, cfg)

    # 2) Add artifact to CT and rescale to [0,1]
    induced_ct_unit = compose_artifact_induced_ct(ct_hu, artifact_vol, cfg)

    # 3) Projection + noisy reconstruction
    pscbct_unit = reconstruct_volume_sart(induced_ct_unit, cfg, seed=seed)

    # 4) Map back to HU-like range
    pscbct_hu = unit_to_range(pscbct_unit, cfg.output_hu_min, cfg.output_hu_max)

    if cfg.preserve_air:
        pscbct_hu = pscbct_hu.astype(np.float32)
        pscbct_hu[~body_mask] = np.minimum(ct_hu[~body_mask], -800.0)

    return pscbct_hu.astype(np.float32)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Single-file pseudo-CBCT simulation inspired by Dahiya et al. 2021"
    )
    p.add_argument("--ct", required=True, help="Planning CT NIfTI")
    p.add_argument("--cbct", required=True, help="Baseline CBCT NIfTI (preferably aligned to CT)")
    p.add_argument("--output", required=True, help="Output pseudo-CBCT NIfTI")
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    p.add_argument("--artifact-scale", type=float, default=1.0)
    p.add_argument("--pl-gamma", type=float, default=0.8)
    p.add_argument("--pl-clip-limit", type=float, default=0.01)
    p.add_argument("--n-angles", type=int, default=180)
    p.add_argument("--proj-noise-std", type=float, default=0.01)
    p.add_argument("--sart-iters", type=int, default=2)

    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = SimConfig(
        artifact_scale=args.artifact_scale,
        pl_gamma=args.pl_gamma,
        pl_ahe_clip_limit=args.pl_clip_limit,
        n_angles=args.n_angles,
        gaussian_proj_noise_std=args.proj_noise_std,
        sart_iterations=args.sart_iters,
    )

    ct_img = load_nifti(args.ct)
    cbct_img = load_nifti(args.cbct)

    ct = get_data(ct_img)
    cbct = get_data(cbct_img)

    # lightweight resample if needed
    cbct_rs = maybe_resample_to_ref(cbct, cbct_img, ct, ct_img)

    out = simulate_pscbct(ct, cbct_rs, cfg, seed=args.seed)

    save_like(ct_img, out, args.output)
    print(f"Saved pseudo-CBCT to: {args.output}")


if __name__ == "__main__":
    main()