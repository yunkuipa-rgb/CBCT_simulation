#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python b_sino.py

Standalone NIfTI-in / NIfTI-out metal / CBCT artifact simulation script,
merged from the user's uploaded:
- utils.py
- simul.py
- ma_config.py
- build_geometry.py

Main usage:
    1) Set parameters in the CONFIG block below
    2) Run:
       python single_file_metal_artifact_sim_nifti.py

Requirements:
    pip install numpy scipy pandas nibabel opencv-python scikit-image matplotlib odl astra-toolbox

Notes:
- ODL + ASTRA are required for projector/backprojector.
- This script preserves the original algorithmic structure as much as possible.
- Input:  NIfTI image
- Output: NIfTI image
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
import skimage.exposure

from scipy import ndimage
from scipy.ndimage import zoom
from numpy.random import default_rng


# ============================================================
# ====================== PARAMS AT TOP =======================
# ============================================================

CONFIG = {
    # -------- IO --------
    "input_nifti": "pCT/2001.nii.gz",
    "output_nifti": "res_sino/2001_3.nii.gz",
    "xray_csv_path": "xray_characteristic_data.csv",

    # -------- mode --------
    # "ct"   -> uses add_ct_noise logic
    # "cbct" -> uses add_cbct_noise logic
    "sim_mode": "cbct",

    # -------- geometry / spacing --------
    # real pixel size in cm, same meaning as original code
    "pixel_size_cm": 0.165,

    # target in-plane size used by projector code
    # original geometry code assumes 256 x 256
    "target_nx": 256,
    "target_ny": 256,

    # projection setup
    "n_proj": 360,
    "nu_h": 256,

    # -------- volume processing --------
    # if input is 3D, process slice-by-slice
    "process_3d_as_slices": True,

    # preserve original affine/header in output
    "preserve_affine": True,

    # clip final output
    "final_clip_min": -500,
    "final_clip_max": 2500,

    # -------- simulation type thresholds --------
    # values here can override defaults from original configs
    "ct_E0": 40,
    "ct_T1": 150,
    "ct_T2": 800,
    "ct_T3": 1000,

    "cbct_E0": 40,
    "cbct_T1": 150,
    "cbct_T2": 1500,
    "cbct_T3": 2400,

    # -------- flexible artifact parameters --------
    "metal_name": "Iron",         # or "Titanium"
    "metal_density_ct": 2.0,
    "metal_density_cbct": 1.0,
    "bone_level": 1.0,
    "water_level": 1.0,

    "noise_scale_ct": 1e-2,
    "noise_scale_cbct": 1e-2,

    "percent_stripe": 0.5,
    "min_v": 0.1,
    "max_v": 0.3,
    "stripe_rate": 1.0,
    "blur_sigma_ct": 0.1,
    "blur_sigma_cbct": None,
    "scale_dark": 0.6,
    "scale_bright": 1.4,

    # random metal insertion count for CT mode
    "min_metal_blocks": 0,
    "max_metal_blocks": 2,

    # random seed
    "seed": 125,
}


# ============================================================
# ====================== GEOMETRY SETUP ======================
# ============================================================

class initialization:
    def __init__(self, n_proj):
        self.param = {}
        self.reso = 512 / 256 * 0.03

        self.param["nx_h"] = CONFIG["target_nx"]
        self.param["ny_h"] = CONFIG["target_ny"]
        self.param["nz_h"] = 1

        self.param["sx"] = self.param["nx_h"] * self.reso
        self.param["sy"] = self.param["ny_h"] * self.reso
        self.param["sz"] = self.param["nz_h"] * self.reso

        self.param["startangle"] = 0
        self.param["endangle"] = 2 * np.pi
        self.param["nProj"] = n_proj

        self.param["su"] = 2 * np.sqrt(self.param["sx"] ** 2 + self.param["sy"] ** 2)
        self.param["nu_h"] = CONFIG["nu_h"]
        self.param["dde"] = 1024 * self.reso
        self.param["dso"] = 1024 * self.reso
        self.param["u_water"] = 0.192


def imaging_geo(param):
    import odl
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param["sx"] / 2.0, -param.param["sy"] / 2.0],
        max_pt=[param.param["sx"] / 2.0, param.param["sy"] / 2.0],
        shape=[param.param["nx_h"], param.param["ny_h"]],
        dtype="float32"
    )
    angle_partition = odl.uniform_partition(
        param.param["startangle"], param.param["endangle"], param.param["nProj"]
    )
    detector_partition_h = odl.uniform_partition(
        -(param.param["su"] / 2.0), (param.param["su"] / 2.0), param.param["nu_h"]
    )
    geometry_h = odl.tomo.FanBeamGeometry(
        angle_partition, detector_partition_h,
        src_radius=param.param["dso"],
        det_radius=param.param["dde"]
    )
    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl="astra_cpu")
    FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type="Hann", frequency_scaling=1.0)
    return ray_trafo_hh, FBPOper_hh


def imaging_geo_cone(param):
    import odl
    angle_partition = odl.uniform_partition(
        param.param["startangle"], param.param["endangle"], param.param["nProj"]
    )
    reco_space_h = odl.uniform_discr(
        min_pt=[-param.param["sx"] / 2.0, -param.param["sy"] / 2.0, -param.param["sz"] / 2.0],
        max_pt=[param.param["sx"] / 2.0, param.param["sy"] / 2.0, param.param["sz"] / 2.0],
        shape=[param.param["nx_h"], param.param["ny_h"], param.param["nz_h"]],
        dtype="float32"
    )
    detector_partition_h = odl.uniform_partition(
        [-(param.param["su"] / 2.0), -(param.param["su"] / 2.0)],
        [(param.param["su"] / 2.0), (param.param["su"] / 2.0)],
        [param.param["nu_h"], param.param["nu_h"]]
    )
    geometry_h = odl.tomo.ConeBeamGeometry(
        angle_partition, detector_partition_h,
        src_radius=256, det_radius=30
    )
    ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl="astra_cuda")
    FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh)
    return ray_trafo_hh, FBPOper_hh


# ============================================================
# ======================= CORE UTILS =========================
# ============================================================

def norm(x, mean, std):
    x = np.clip(x, -500, 1000)
    x = (x + 500) / 1500
    return x


def hu2mu(hu, mu_water, mu_air):
    return hu / 1000.0 * (mu_water - mu_air) + mu_water


def mu2hu(mu, mu_water, mu_air):
    return 1000 * (mu - mu_water) / (mu_water - mu_air)


def threshold_based_weighting(image, T1, T2):
    w_bone = (image - T1) / (T2 - T1)
    w_bone = np.clip(w_bone, 0, 1)
    bone = w_bone * image

    w_water = (T2 - image) / (T2 - T1)
    w_water = np.clip(w_water, 0, 1)
    water = w_water * image
    return water, bone


def clip(a, minimum, maximum):
    clipped = a.copy()
    clipped[a > maximum] = maximum
    clipped[a < minimum] = minimum
    return clipped


def make_mask(x, p, min_v, max_v):
    m = np.ones_like(x)
    num = int(m.shape[-1] * p)
    if num <= 0:
        return m
    stripe = np.random.choice(np.arange(0, m.shape[-1], dtype=np.int32), num, replace=False)
    value = np.random.uniform(min_v, max_v, stripe.shape)
    thickness = np.random.randint(-3, 10, num)
    for i, sp in enumerate(stripe):
        if thickness[i] < 1:
            thickness[i] = 1
        m[:, sp:sp + thickness[i]] = value[i]
    return m


def make_holes(img, sx, sy):
    height, width = img.shape
    rng = default_rng()
    noise = rng.integers(0, 255, (height, width), np.uint8, True)
    blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=sx, sigmaY=sy, borderType=cv2.BORDER_DEFAULT)
    stretch = skimage.exposure.rescale_intensity(
        blur, in_range="image", out_range=(0, 255)
    ).astype(np.uint8)

    thresh = cv2.threshold(stretch, 155, 235, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def block_artifact(x, seed, scale_bright, scale_dark):
    main_body = np.ones_like(x)
    T = hu2mu(-300, 0.286, 0)
    main_body[x < T] = 0

    mask1 = np.asarray(make_holes(x, seed[0], seed[1]), dtype=np.float64)
    mask2 = np.asarray(make_holes(x, seed[2], seed[3]), dtype=np.float64)
    mask1[mask1 > 0] = 1
    mask2[mask2 > 0] = 1

    m = mask1 - mask2
    m[m > 0] = 0
    mask2 = np.abs(m)

    mask = np.ones_like(x)
    mask1 *= np.random.uniform(low=0.0, high=1.0, size=x.shape)
    mask2 *= np.random.uniform(low=0.0, high=1.0, size=x.shape)
    if np.random.rand(1)[0] > 0.5:
        mask += mask1 * (scale_bright - 1)
    else:
        mask += mask1 * (scale_bright - 1) + mask2 * (scale_dark - 1)

    img = x * mask * main_body
    return img


def polyval(c, p):
    y = 0
    for pv in c:
        y = y * p + pv
    return y


def calibration(p, total_intensity, correction_coeff, config):
    v = p / total_intensity
    ps = -np.log(np.where(v > 0, v, np.zeros_like(v) + 0.01))
    ps = polyval(correction_coeff, ps)

    sim = np.asarray(config["ifanbeam"](ps))
    sim[sim < 0] = 0
    sim = sim / config["pixel_size"]

    s = mu2hu(sim, config["mu_water"], config["mu_air"])
    s = np.clip(s, -500, 1000)
    return s


# ============================================================
# ====================== MAIN SIM CORE =======================
# ============================================================

def ct_metal_artifact_simulation(image, x_metal, config):
    data = config["data"]
    energy_composition = config["energy_composition"]
    E0 = config["E0"]
    mu_air = config["mu_air"]
    metal_name = config["metal_name"]
    metal_density = config["metal_density"]
    T1 = config["T1"]
    T2 = config["T2"]
    r = config["metal_level"]
    correction_coeff = config["correction_coeff"]
    pixel_size = config["pixel_size"]

    m0_water = data["Water"][E0]
    m0_bone = data["Bone"][E0]
    m0_metal = data[metal_name][E0]
    mu_water0 = m0_water * 1.0
    mu_metal0 = ((m0_metal - m0_bone) * r + m0_bone) * metal_density

    T1 = hu2mu(T1, mu_water0, mu_air)
    T2 = hu2mu(T2, mu_water0, mu_air)
    seed_list = np.random.randint(1, 15, 4)
    image = block_artifact(image, seed_list, config["scale_bright"], config["scale_dark"])
    x_water, x_bone = threshold_based_weighting(image, T1, T2)
    x_water *= config["water_level"]
    x_bone *= config["bone_level"]

    x_water[x_metal > 0] = 0
    x_bone[x_metal > 0] = 0
    x_metal_o = x_metal * mu_metal0

    lam = config["noise_scale"]
    degree, blur_sigma, bright, dark = np.random.uniform(
        (0.0, 0.0, 1.1, 0.4), (180.0, 2.0, 1.5, 0.9), 4
    )

    if config["blur_sigma"] is not None:
        mask = make_mask(
            x_water,
            np.random.rand(1)[0] * config["percent_stripe"] + 0.05,
            config["min_v"],
            config["max_v"]
        )
        x_water = ndimage.rotate(mask, degree, reshape=False) * x_water
        seed = np.random.randint(1, 15, 4)
    else:
        blur_sigma = 0.0

    if blur_sigma >= 0.5:
        x_water = ndimage.gaussian_filter(x_water, sigma=blur_sigma)

    d_water = config["fanbeam"](x_water) * pixel_size
    d_bone = config["fanbeam"](x_bone) * pixel_size
    d_metal = config["fanbeam"](x_metal_o) * pixel_size

    m_water = config["m_water"][energy_composition]
    m_bone = config["m_bone"][energy_composition]
    m_metal = config["m_metal"][energy_composition]
    intensity = config["m_intensity"][energy_composition]

    d_water_tmp = np.einsum("ij,k->ijk", d_water, m_water / m0_water)
    d_bone_tmp = np.einsum("ij,k->ijk", d_bone, m_bone / m0_bone)
    d_metal_tmp = np.einsum(
        "ij,k->ijk",
        d_metal,
        (m_metal / m0_metal) * r + (m_bone / m0_bone) * (1 - r)
    )

    DRR = d_water_tmp + d_bone_tmp + d_metal_tmp
    y = np.einsum("ijk,k->ijk", (np.exp(-DRR)), intensity)
    total_intensity = np.sum(intensity)

    poly_y = np.sum(y, axis=2)
    if config["blur_sigma"] is not None:
        poly_y = lam * np.random.poisson(poly_y / lam)

    x_ma = calibration(poly_y, total_intensity, correction_coeff, config)
    return x_ma


def ct_metal_artifact_simulation_3D(image, x_metal, config):
    data = config["data"]
    energy_composition = config["energy_composition"]
    E0 = config["E0"]
    mu_air = config["mu_air"]
    metal_name = config["metal_name"]
    metal_density = config["metal_density"]
    T1 = config["T1"]
    T2 = config["T2"]
    r = config["metal_level"]
    correction_coeff = config["correction_coeff"]
    pixel_size = config["pixel_size"]

    m0_water = data["Water"][E0]
    m0_bone = data["Bone"][E0]
    m0_metal = data[metal_name][E0]
    mu_water0 = m0_water * 1.0
    mu_metal0 = ((m0_metal - m0_bone) * r + m0_bone) * metal_density

    T1 = hu2mu(T1, mu_water0, mu_air)
    T2 = hu2mu(T2, mu_water0, mu_air)
    x_water, x_bone = threshold_based_weighting(image, T1, T2)
    x_water *= config["water_level"]
    x_bone *= config["bone_level"]

    x_water[x_metal > 0] = 0
    x_bone[x_metal > 0] = 0
    x_metal_o = x_metal * mu_metal0

    lam = config["noise_scale"]
    degree, blur_sigma, bright, dark = np.random.uniform(
        (0.0, 1.0, 1.3, 0.4), (180.0, 2.0, 2.5, 0.9), 4
    )

    if config["blur_sigma"] is not None:
        mask = make_mask(x_water, config["percent_stripe"], config["min_v"], config["max_v"])
        x_water = ndimage.rotate(mask, degree, reshape=False) * x_water
    else:
        blur_sigma = 1.5

    x_water = x_water * (1 - lam) + (np.random.randn(*x_water.shape)) * lam

    d_water = config["fanbeam"](x_water) * pixel_size
    d_bone = config["fanbeam"](x_bone) * pixel_size
    d_metal = config["fanbeam"](x_metal_o) * pixel_size

    m_water = config["m_water"][energy_composition]
    m_bone = config["m_bone"][energy_composition]
    m_metal = config["m_metal"][energy_composition]
    intensity = config["m_intensity"][energy_composition]

    d_water_tmp = np.einsum("ijl,k->ijlk", d_water, m_water / m0_water)
    d_bone_tmp = np.einsum("ijl,k->ijlk", d_bone, m_bone / m0_bone)
    d_metal_tmp = np.einsum(
        "ijl,k->ijlk",
        d_metal,
        (m_metal / m0_metal) * r + (m_bone / m0_bone) * (1 - r)
    )

    DRR = d_water_tmp + d_bone_tmp + d_metal_tmp
    y = np.einsum("ijlk,k->ijlk", (np.exp(-DRR)), intensity)
    total_intensity = np.sum(intensity)

    poly_y = np.sum(y, axis=2)
    x_ma = calibration(poly_y, total_intensity, correction_coeff, config)
    return x_ma


# ============================================================
# ====================== HIGH-LEVEL API ======================
# ============================================================

def recon_image(x):
    return np.clip(x, -500, 2500)


def add_mask(x, mask, min_m=0, max_m=6):
    ms = np.zeros_like(x)

    x_axis = np.sum(mask, axis=0)
    x_ids = np.nonzero(x_axis)[0]
    y_axis = np.sum(mask, axis=1)
    y_ids = np.nonzero(y_axis)[0]

    if not (len(x_ids) < 2 or len(y_ids) < 2):
        num = np.random.randint(min_m, max_m, size=1)[0]
        x_min, x_max = np.min(x_ids), np.max(x_ids)
        y_min, y_max = np.min(y_ids), np.max(y_ids)

        if not (x_min == x_max or y_min == y_max) and num > 0:
            xs = np.random.randint(x_min, x_max, size=(num,), dtype=np.int32)
            ys = np.random.randint(y_min, y_max, size=(num,), dtype=np.int32)
            r = int(max(x_max - x_min, y_max - y_min) / 5)
            max_r = max(5, r)
            rs = np.random.randint(1, max_r, size=(num,), dtype=np.int32)

            for i in range(num):
                y0 = max(0, ys[i] - rs[i])
                y1 = min(ms.shape[0], ys[i] + rs[i])
                x0 = max(0, xs[i] - rs[i])
                x1 = min(ms.shape[1], xs[i] + rs[i])
                ms[y0:y1, x0:x1] = 1

    return ms


def add_ct_noise(x, config):
    image = recon_image(x)
    mask = np.zeros_like(image)
    mask[image > config["T3"]] = 1

    ms = add_mask(
        image,
        mask,
        min_m=CONFIG["min_metal_blocks"],
        max_m=CONFIG["max_metal_blocks"]
    )

    metal = np.zeros_like(image)
    metal[(mask > 0) & (ms > 0)] = 1

    image_hu = hu2mu(image, config["mu_water"], config["mu_air"])
    x_t = ct_metal_artifact_simulation(image_hu, metal, config)
    return x_t


def add_cbct_noise(x, config):
    image = recon_image(x)
    thresh = config["T3"]
    metal = np.zeros_like(image)
    metal[image > thresh] = 1

    image_hu = hu2mu(image, config["mu_water"], config["mu_air"])
    x_t = ct_metal_artifact_simulation(image_hu, metal, config)
    return x_t


# ============================================================
# ======================= CONFIG BUILD =======================
# ============================================================

def set_config_for_ct_artifact_simulation(pixel_size):
    config = {}
    metals = ["Titanium", "Iron"]
    config["data"] = pd.read_csv(CONFIG["xray_csv_path"])
    config["E0"] = CONFIG["ct_E0"]
    config["metal_name"] = CONFIG["metal_name"]
    config["mu_water"] = config["data"].loc[config["E0"], "Water"]
    config["mu_air"] = 0
    config["T1"] = CONFIG["ct_T1"]
    config["T2"] = CONFIG["ct_T2"]
    config["T3"] = CONFIG["ct_T3"]

    config["energy_composition"] = np.arange(0, 120, dtype=np.int32)
    config["polynomial_order_for_correction"] = 3
    config["m_water"] = config["data"]["Water"]
    config["m_bone"] = config["data"]["Bone"]
    config["m_metal"] = config["data"][config["metal_name"]]
    config["m_intensity"] = config["data"]["Intensity"]
    config["correction_coeff"] = np.asarray(
        [[-1.04811676e-02], [9.82882828e-02], [9.33561802e-01], [4.56019654e-04]]
    )

    config["metal_density"] = CONFIG["metal_density_ct"]
    config["bone_level"] = CONFIG["bone_level"]
    config["water_level"] = CONFIG["water_level"]
    r = (config["metal_density"] ** 2 - 1)
    config["metal_level"] = 2.0 / (1 + np.exp(-r)) - 1
    config["noise_scale"] = CONFIG["noise_scale_ct"]
    config["percent_stripe"] = CONFIG["percent_stripe"]
    config["min_v"] = CONFIG["min_v"]
    config["max_v"] = CONFIG["max_v"]
    config["stripe_rate"] = CONFIG["stripe_rate"]
    config["blur_sigma"] = CONFIG["blur_sigma_ct"]

    config["pixel_size"] = pixel_size

    param = initialization(CONFIG["n_proj"])
    fp, bp = imaging_geo_cone(param)
    config["fanbeam"] = fp
    config["ifanbeam"] = bp
    return config


def set_config_for_cbct_artifact_simulation(pixel_size):
    config = {}
    metals = ["Titanium", "Iron"]
    config["data"] = pd.read_csv(CONFIG["xray_csv_path"])
    config["E0"] = CONFIG["cbct_E0"]
    config["metal_name"] = CONFIG["metal_name"]
    config["mu_water"] = config["data"].loc[config["E0"], "Water"]
    config["mu_air"] = 0
    config["T1"] = CONFIG["cbct_T1"]
    config["T2"] = CONFIG["cbct_T2"]
    config["T3"] = CONFIG["cbct_T3"]

    config["energy_composition"] = np.arange(0, 120, dtype=np.int32)
    config["polynomial_order_for_correction"] = 3
    config["m_water"] = config["data"]["Water"]
    config["m_bone"] = config["data"]["Bone"]
    config["m_metal"] = config["data"][config["metal_name"]]
    config["m_intensity"] = config["data"]["Intensity"]
    config["correction_coeff"] = np.asarray(
        [[-1.04811676e-02], [9.82882828e-02], [9.33561802e-01], [4.56019654e-04]]
    )

    config["metal_density"] = CONFIG["metal_density_cbct"]
    config["bone_level"] = CONFIG["bone_level"]
    config["water_level"] = CONFIG["water_level"]
    config["noise_scale"] = CONFIG["noise_scale_cbct"]
    r = (config["metal_density"] ** 2 - 1)
    config["metal_level"] = 2.0 / (1 + np.exp(-r)) - 1
    config["freqscale"] = 1
    config["blur_sigma"] = CONFIG["blur_sigma_cbct"]

    config["scale_bright"] = CONFIG["scale_bright"]
    config["scale_dark"] = CONFIG["scale_dark"]

    config["pixel_size"] = pixel_size

    param = initialization(CONFIG["n_proj"])
    fp, bp = imaging_geo(param)
    config["fanbeam"] = fp
    config["ifanbeam"] = bp
    return config


# ============================================================
# ======================= NIFTI HELPERS ======================
# ============================================================

def load_nifti(path):
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    return img, arr


def save_nifti_like(ref_img, arr, out_path):
    out = nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header.copy())
    nib.save(out, out_path)


def resize_to_target_2d(arr2d, target_hw):
    h, w = arr2d.shape
    th, tw = target_hw
    zoom_factors = (th / h, tw / w)
    out = zoom(arr2d, zoom_factors, order=1)
    return out.astype(np.float32)


def resize_back_2d(arr2d, out_hw):
    h, w = arr2d.shape
    oh, ow = out_hw
    zoom_factors = (oh / h, ow / w)
    out = zoom(arr2d, zoom_factors, order=1)
    return out.astype(np.float32)


def process_slice_2d(slice_2d, config):
    orig_shape = slice_2d.shape
    x = resize_to_target_2d(slice_2d, (CONFIG["target_nx"], CONFIG["target_ny"]))

    if CONFIG["sim_mode"].lower() == "ct":
        y = add_ct_noise(x, config)
    elif CONFIG["sim_mode"].lower() == "cbct":
        y = add_cbct_noise(x, config)
    else:
        raise ValueError(f"Unknown sim_mode: {CONFIG['sim_mode']}")

    # cone reconstruction may return singleton z
    y = np.asarray(y)
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    elif y.ndim == 3 and y.shape[0] == 1:
        y = y[0]
    elif y.ndim != 2:
        y = np.squeeze(y)

    y = resize_back_2d(y, orig_shape)
    y = np.clip(y, CONFIG["final_clip_min"], CONFIG["final_clip_max"])
    return y.astype(np.float32)


def process_volume(arr, config):
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        return process_slice_2d(arr, config)

    if arr.ndim != 3:
        raise ValueError(f"Only 2D or 3D NIfTI is supported, got shape {arr.shape}")

    out = np.zeros_like(arr, dtype=np.float32)
    for z in range(arr.shape[2]):
        print(f"Processing slice {z+1}/{arr.shape[2]}")
        out[..., z] = process_slice_2d(arr[..., z], config)
    return out


# ============================================================
# ============================ MAIN ==========================
# ============================================================

def main():
    np.random.seed(CONFIG["seed"])

    if not os.path.isfile(CONFIG["input_nifti"]):
        raise FileNotFoundError(f"Input NIfTI not found: {CONFIG['input_nifti']}")
    if not os.path.isfile(CONFIG["xray_csv_path"]):
        raise FileNotFoundError(f"X-ray CSV not found: {CONFIG['xray_csv_path']}")

    img, arr = load_nifti(CONFIG["input_nifti"])

    if CONFIG["sim_mode"].lower() == "ct":
        sim_cfg = set_config_for_ct_artifact_simulation(CONFIG["pixel_size_cm"])
    elif CONFIG["sim_mode"].lower() == "cbct":
        sim_cfg = set_config_for_cbct_artifact_simulation(CONFIG["pixel_size_cm"])
    else:
        raise ValueError("CONFIG['sim_mode'] must be 'ct' or 'cbct'")

    out = process_volume(arr, sim_cfg)
    save_nifti_like(img, out, CONFIG["output_nifti"])

    print(f"Saved output to: {CONFIG['output_nifti']}")


if __name__ == "__main__":
    main()