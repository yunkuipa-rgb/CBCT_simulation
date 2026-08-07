#!/usr/bin/env python3
"""
ct_to_cbct_intensity_sim.py

Practical reproduction of the intensity-based CT->CBCT-style simulation
described in:

Brion et al., "Domain adversarial networks and intensity-based data augmentation
for male pelvic organ segmentation in cone beam CT", Computers in Biology and
Medicine, 2021.

Input : NIfTI CT volume
Output: NIfTI pseudo-CBCT volume

This is an intensity-domain simulator, not a projection-domain CBCT physics model.
It is intended to reproduce the paper's augmentation spirit for data generation.

Example:
    python b_brion.py \
        --input pCT/2001.nii.gz \
        --output res_brion/cbct_2001_3.nii.gz \
        --seed 1234
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class SimConfig:
    # Body masking
    air_threshold_hu: float = -700.0

    # Intensity window for simulation
    clip_min_hu: float = -1000.0
    clip_max_hu: float = 1500.0

    # Brightness / contrast
    brightness_shift_range: Tuple[float, float] = (-120.0, 120.0)
    contrast_scale_range: Tuple[float, float] = (0.75, 1.15)
    gamma_range: Tuple[float, float] = (0.85, 1.25)

    # Shading / cupping field
    shading_strength_range: Tuple[float, float] = (0.48, 1.32)
    shading_sigma_vox: Tuple[float, float, float] = (124.0, 14.0, 22.0)

    # Blur
    blur_sigma_range: Tuple[float, float] = (0.1, 0.3)

    # Noise
    gaussian_noise_std_range: Tuple[float, float] = (8.0, 30.0)
    poisson_scale_range: Tuple[float, float] = (5e3, 2e4)

    # Optional streak corruption
    streak_prob: float = 0.5
    streak_strength_range: Tuple[float, float] = (20.0, 120.0)
    streak_count_range: Tuple[int, int] = (2, 8)

    # Motion-like slice distortion
    motion_prob: float = 0.25
    motion_max_shift_vox: float = 2.0

    # Final clamp
    final_min_hu: float = -1000.0
    final_max_hu: float = 1200.0


# -----------------------------
# Utilities
# -----------------------------
def load_nifti(path: str) -> Tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    return arr, img


def save_nifti_like(
    array: np.ndarray,
    ref_img: nib.Nifti1Image,
    out_path: str,
    dtype=np.float32,
) -> None:
    out = nib.Nifti1Image(array.astype(dtype), ref_img.affine, ref_img.header.copy())
    nib.save(out, out_path)


def compute_body_mask(ct_hu: np.ndarray, air_threshold_hu: float) -> np.ndarray:
    """
    Very simple body mask from HU threshold.
    """
    return ct_hu > air_threshold_hu


def normalize_unit(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x = np.clip(x, x_min, x_max)
    return (x - x_min) / max(x_max - x_min, 1e-8)


def denormalize_unit(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    return x * (x_max - x_min) + x_min


def random_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def random_int(rng: np.random.Generator, low: int, high: int) -> int:
    # inclusive range
    return int(rng.integers(low, high + 1))


# -----------------------------
# Simulation components
# -----------------------------
def apply_brightness_contrast_gamma(
    vol_hu: np.ndarray,
    body_mask: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Pixel/intensity-level transform, matching the paper's "brightness-based"
    augmentation direction.
    """
    x = np.clip(vol_hu, cfg.clip_min_hu, cfg.clip_max_hu).copy()

    # Contrast around soft-tissue pivot
    pivot = 40.0
    contrast = random_uniform(rng, *cfg.contrast_scale_range)
    brightness = random_uniform(rng, *cfg.brightness_shift_range)

    x_body = x[body_mask]
    x[body_mask] = (x_body - pivot) * contrast + pivot + brightness

    # Gamma in normalized space
    x_norm = normalize_unit(x, cfg.clip_min_hu, cfg.clip_max_hu)
    gamma = random_uniform(rng, *cfg.gamma_range)
    x_norm[body_mask] = np.power(np.clip(x_norm[body_mask], 1e-6, 1.0), gamma)
    x = denormalize_unit(x_norm, cfg.clip_min_hu, cfg.clip_max_hu)

    return x


def generate_low_frequency_field(
    shape: Tuple[int, int, int],
    sigma_xyz: Tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Smooth multiplicative field for cupping / brightness nonuniformity.
    """
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    field = gaussian_filter(noise, sigma=sigma_xyz, mode="nearest")
    field -= field.mean()
    std = field.std()
    if std > 1e-6:
        field /= std
    return field


def apply_shading(
    vol_hu: np.ndarray,
    body_mask: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Low-frequency intensity shading to mimic CBCT cupping / nonuniformity.
    """
    strength = random_uniform(rng, *cfg.shading_strength_range)
    field = generate_low_frequency_field(vol_hu.shape, cfg.shading_sigma_vox, rng)

    # Work in normalized intensity space for stable multiplicative modulation
    x_norm = normalize_unit(vol_hu, cfg.clip_min_hu, cfg.clip_max_hu)
    shaded = x_norm.copy()

    mod = 1.0 + strength * field
    shaded[body_mask] = np.clip(x_norm[body_mask] * mod[body_mask], 0.0, 1.0)

    return denormalize_unit(shaded, cfg.clip_min_hu, cfg.clip_max_hu)


def apply_blur(
    vol_hu: np.ndarray,
    body_mask: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = random_uniform(rng, *cfg.blur_sigma_range)
    blurred = gaussian_filter(vol_hu, sigma=sigma, mode="nearest")
    out = vol_hu.copy()
    out[body_mask] = blurred[body_mask]
    return out


def apply_noise(
    vol_hu: np.ndarray,
    body_mask: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Combines Gaussian and Poisson-like noise.
    """
    x = vol_hu.copy()

    # Gaussian noise in HU
    gstd = random_uniform(rng, *cfg.gaussian_noise_std_range)
    x[body_mask] += rng.normal(0.0, gstd, size=body_mask.sum()).astype(np.float32)

    # Poisson-like noise in normalized intensity domain
    scale = random_uniform(rng, *cfg.poisson_scale_range)
    xn = normalize_unit(x, cfg.clip_min_hu, cfg.clip_max_hu)
    lam = np.clip(xn * scale, 1.0, None)
    noisy = rng.poisson(lam).astype(np.float32) / scale
    x = denormalize_unit(noisy, cfg.clip_min_hu, cfg.clip_max_hu)

    return x


def apply_streaks(
    vol_hu: np.ndarray,
    body_mask: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simple image-domain streak-like corruption.
    This is not projection-domain metal streak simulation; it is a lightweight
    artifact generator for augmentation.
    """
    if rng.uniform() >= cfg.streak_prob:
        return vol_hu

    zdim = vol_hu.shape[2]
    ydim = vol_hu.shape[1]
    xdim = vol_hu.shape[0]
    out = vol_hu.copy()

    n_streaks = random_int(rng, *cfg.streak_count_range)

    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim), indexing="ij")
    cx = (xdim - 1) / 2.0
    cy = (ydim - 1) / 2.0

    for _ in range(n_streaks):
        angle = random_uniform(rng, 0.0, np.pi)
        strength = random_uniform(rng, *cfg.streak_strength_range)
        width = random_uniform(rng, 2.0, 8.0)

        # Distance to a line through the image center
        dist = np.abs((xx - cx) * np.sin(angle) - (yy - cy) * np.cos(angle))
        line = np.exp(-(dist ** 2) / (2.0 * width ** 2)).astype(np.float32)

        # Random slice span
        z0 = random_int(rng, 0, max(0, zdim - 1))
        z1 = random_int(rng, z0, zdim - 1)

        sign = -1.0 if rng.uniform() < 0.5 else 1.0
        pattern = sign * strength * line[..., None]
        out[..., z0 : z1 + 1] += pattern

    out[~body_mask] = vol_hu[~body_mask]
    return out


def apply_motion_like_slice_shift(
    vol_hu: np.ndarray,
    body_mask: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Small per-slice in-plane shift to mimic motion inconsistency.
    """
    if rng.uniform() >= cfg.motion_prob:
        return vol_hu

    xdim, ydim, zdim = vol_hu.shape
    out = np.empty_like(vol_hu)

    grid_x, grid_y = np.meshgrid(np.arange(xdim), np.arange(ydim), indexing="ij")

    for z in range(zdim):
        dx = random_uniform(rng, -cfg.motion_max_shift_vox, cfg.motion_max_shift_vox)
        dy = random_uniform(rng, -cfg.motion_max_shift_vox, cfg.motion_max_shift_vox)

        coords = np.array(
            [
                np.clip(grid_x - dx, 0, xdim - 1),
                np.clip(grid_y - dy, 0, ydim - 1),
            ]
        )
        out[..., z] = map_coordinates(
            vol_hu[..., z],
            coords,
            order=1,
            mode="nearest",
        )

    out[~body_mask] = vol_hu[~body_mask]
    return out


def simulate_pseudo_cbct(
    ct_hu: np.ndarray,
    cfg: SimConfig,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    x = np.asarray(ct_hu, dtype=np.float32)
    x = np.clip(x, cfg.clip_min_hu, cfg.clip_max_hu)

    body_mask = compute_body_mask(x, cfg.air_threshold_hu)

    # Main intensity-domain pipeline
    x = apply_brightness_contrast_gamma(x, body_mask, cfg, rng)
    x = apply_shading(x, body_mask, cfg, rng)
    x = apply_blur(x, body_mask, cfg, rng)
    x = apply_noise(x, body_mask, cfg, rng)
    x = apply_streaks(x, body_mask, cfg, rng)
    x = apply_motion_like_slice_shift(x, body_mask, cfg, rng)

    # Keep air close to original / stable
    x[~body_mask] = np.minimum(x[~body_mask], -700.0)

    x = np.clip(x, cfg.final_min_hu, cfg.final_max_hu).astype(np.float32)
    return x


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CT NIfTI path")
    p.add_argument("--output", required=True, help="Output pseudo-CBCT NIfTI path")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    ct, ref_img = load_nifti(args.input)
    cfg = SimConfig()
    pseudo_cbct = simulate_pseudo_cbct(ct, cfg, seed=args.seed)
    save_nifti_like(pseudo_cbct, ref_img, args.output)

    print(f"Saved pseudo-CBCT to: {args.output}")


if __name__ == "__main__":
    main()