#!/usr/bin/env python3
"""Create cone-beam Radon sinograms for padded CT canvases.

The input CT object is kept unchanged and placed in progressively larger
air/background volumes.  The detector pixel count can scale with the edge
padding so the saved NIfTI sinogram shapes visibly change.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pydicom

from config import CBCTConfig


def parse_args() -> argparse.Namespace:
    cfg = CBCTConfig()
    geometry = cfg.geometry
    materials = cfg.materials

    parser = argparse.ArgumentParser(
        description="Forward-project a DICOM CT into cone-beam sinograms with x1/x2/x4 air canvases."
    )
    parser.add_argument(
        "--dicom-dir",
        default="/mnt/whitsett/yunkuipa/cbct_project/data/Ethos_ClinicalHeadProtocol",
        help="Folder containing the input CT DICOM series.",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/whitsett/yunkuipa/cbct_project/res_sino/ethos_conebeam_padding",
        help="Folder for output NIfTI sinograms and manifest.",
    )
    parser.add_argument(
        "--edge-scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="Canvas edge multipliers. 1 keeps the original CT size.",
    )
    parser.add_argument(
        "--n-proj",
        type=int,
        default=int(geometry["n_proj"]),
        help="Number of cone-beam projection angles.",
    )
    parser.add_argument(
        "--detector-pixels",
        type=int,
        default=int(geometry["nu_h"]),
        help="Base detector pixels per edge for the x1 canvas.",
    )
    parser.add_argument(
        "--detector-pixel-mode",
        choices=("scale", "fixed"),
        default="scale",
        help="Use 'scale' to multiply detector pixels by the canvas edge scale.",
    )
    parser.add_argument(
        "--detector-size-mm",
        type=float,
        default=None,
        help="Detector physical edge size for x1. If omitted, it is calculated from the padded object diagonal.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=float(geometry["margin"]),
        help="Detector size margin when detector-size-mm is omitted.",
    )
    parser.add_argument(
        "--dso",
        type=float,
        default=float(geometry["dso"]),
        help="Source-to-isocenter distance in mm.",
    )
    parser.add_argument(
        "--dde",
        type=float,
        default=float(geometry["dde"]),
        help="Detector-to-isocenter distance in mm.",
    )
    parser.add_argument(
        "--start-angle",
        type=float,
        default=float(geometry["start_angle"]),
        help="Start angle in radians.",
    )
    parser.add_argument(
        "--end-angle",
        type=float,
        default=float(geometry["end_angle"]),
        help="End angle in radians.",
    )
    parser.add_argument(
        "--background-hu",
        type=float,
        default=None,
        help="HU value used for the padded background. Defaults to the minimum CT HU.",
    )
    parser.add_argument(
        "--mu-water",
        type=float,
        default=float(materials["mu_water"]),
        help="Water attenuation coefficient used for HU to mu conversion.",
    )
    parser.add_argument(
        "--mu-air",
        type=float,
        default=float(materials["mu_air"]),
        help="Air attenuation coefficient used for HU to mu conversion.",
    )
    parser.add_argument(
        "--impl",
        default="astra_cuda",
        help="ODL RayTransform implementation. The project reference uses astra_cuda.",
    )
    parser.add_argument(
        "--placement",
        choices=("center", "corner-third"),
        default="center",
        help=(
            "Where to place the source CT inside the padded volume. "
            "'corner-third' places the object center at one-third of each padded edge."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the CT and write only the planned shapes to the manifest.",
    )
    return parser.parse_args()


def dicom_position_key(ds: pydicom.dataset.Dataset, tol: float = 1e-4) -> tuple[str, Any]:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return ("IPP_Z", round(float(ipp[2]) / tol))

    slice_location = getattr(ds, "SliceLocation", None)
    if slice_location is not None:
        return ("SliceLocation", round(float(slice_location) / tol))

    instance_number = getattr(ds, "InstanceNumber", None)
    if instance_number is not None:
        return ("InstanceNumber", int(instance_number))

    return ("SOPInstanceUID", str(getattr(ds, "SOPInstanceUID", "")))


def sort_dicom_slices(slices: list[pydicom.dataset.Dataset]) -> list[pydicom.dataset.Dataset]:
    def sort_key(ds: pydicom.dataset.Dataset) -> float:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])

        slice_location = getattr(ds, "SliceLocation", None)
        if slice_location is not None:
            return float(slice_location)

        instance_number = getattr(ds, "InstanceNumber", None)
        if instance_number is not None:
            return float(instance_number)

        return 0.0

    return sorted(slices, key=sort_key)


def load_ct_dicom_series(dicom_dir: str | os.PathLike[str]) -> tuple[np.ndarray, dict[str, Any]]:
    dicom_dir = str(dicom_dir)
    files = []
    for pattern in ("*.dcm", "*.DCM", "*.dicom", "*.DICOM"):
        files.extend(glob.glob(os.path.join(dicom_dir, pattern)))

    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    slices = []
    for path in files:
        try:
            ds = pydicom.dcmread(path)
        except Exception as exc:
            print(f"Skipping unreadable DICOM {path}: {exc}")
            continue
        if getattr(ds, "Modality", "") == "CT":
            slices.append(ds)

    if not slices:
        raise ValueError(f"No CT slices found in {dicom_dir}")

    slices = sort_dicom_slices(slices)
    unique = []
    seen = set()
    for ds in slices:
        key = dicom_position_key(ds)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ds)

    if len(unique) != len(slices):
        print(f"Detected {len(slices) - len(unique)} duplicate CT slice(s); using {len(unique)} unique slices.")

    volume = np.stack([ds.pixel_array for ds in unique], axis=0).astype(np.float32)
    ref = unique[0]
    slope = float(getattr(ref, "RescaleSlope", 1.0))
    intercept = float(getattr(ref, "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    pixel_spacing = [float(v) for v in getattr(ref, "PixelSpacing", [1.0, 1.0])]
    slice_spacing = infer_slice_spacing(unique)
    metadata = {
        "num_input_dicom_files": len(files),
        "num_ct_slices": len(unique),
        "duplicate_ct_slices": len(slices) - len(unique),
        "pixel_spacing": pixel_spacing,
        "slice_spacing": slice_spacing,
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "series_description": str(getattr(ref, "SeriesDescription", "")),
        "study_description": str(getattr(ref, "StudyDescription", "")),
    }
    return volume, metadata


def infer_slice_spacing(slices: list[pydicom.dataset.Dataset]) -> float:
    z_positions = []
    for ds in slices:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z_positions.append(float(ipp[2]))

    if len(z_positions) >= 2:
        diffs = np.diff(sorted(z_positions))
        diffs = np.abs(diffs[np.abs(diffs) > 1e-4])
        if diffs.size:
            return float(np.median(diffs))

    return float(getattr(slices[0], "SliceThickness", 1.0))


def hu_to_mu(volume_hu: np.ndarray, mu_water: float, mu_air: float) -> np.ndarray:
    mu = volume_hu / 1000.0 * (mu_water - mu_air) + mu_water
    return np.maximum(mu, 0.0).astype(np.float32, copy=False)


def scaled_shape(shape: tuple[int, int, int], edge_scale: float) -> tuple[int, int, int]:
    if edge_scale < 1.0:
        raise ValueError(f"edge_scale must be >= 1.0, got {edge_scale}")
    return tuple(int(math.ceil(dim * edge_scale)) for dim in shape)


def placement_start_indices(
    source_shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    placement: str,
) -> tuple[int, int, int]:
    starts = []
    for src_dim, dst_dim in zip(source_shape, target_shape):
        if dst_dim < src_dim:
            raise ValueError(f"Target shape {target_shape} is smaller than source shape {source_shape}")

        if placement == "center":
            start = (dst_dim - src_dim) // 2
        elif placement == "corner-third":
            center = dst_dim / 3.0
            start = int(round(center - src_dim / 2.0))
            start = min(max(start, 0), dst_dim - src_dim)
        else:
            raise ValueError(f"Unknown placement: {placement}")

        starts.append(start)

    return tuple(starts)


def pad_mu(
    volume_mu: np.ndarray,
    target_shape: tuple[int, int, int],
    background_mu: float,
    placement: str,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    if volume_mu.shape == target_shape:
        return np.ascontiguousarray(volume_mu, dtype=np.float32), (0, 0, 0)

    start_indices = placement_start_indices(volume_mu.shape, target_shape, placement)
    padded = np.full(target_shape, float(background_mu), dtype=np.float32)
    src_slices = []
    dst_slices = []

    for start, src_dim in zip(start_indices, volume_mu.shape):
        src_slices.append(slice(0, src_dim))
        dst_slices.append(slice(start, start + src_dim))

    padded[tuple(dst_slices)] = volume_mu[tuple(src_slices)]
    return padded, start_indices


def build_cone_beam_operator(
    shape_zyx: tuple[int, int, int],
    spacing_xyz: tuple[float, float, float],
    args: argparse.Namespace,
    edge_scale: float,
):
    import odl

    nz, ny, nx = shape_zyx
    sx = nx * spacing_xyz[0]
    sy = ny * spacing_xyz[1]
    sz = nz * spacing_xyz[2]

    reco_space = odl.uniform_discr(
        min_pt=[-sx / 2.0, -sy / 2.0, -sz / 2.0],
        max_pt=[sx / 2.0, sy / 2.0, sz / 2.0],
        shape=[nx, ny, nz],
        dtype="float32",
    )
    angle_partition = odl.uniform_partition(args.start_angle, args.end_angle, args.n_proj)

    if args.detector_pixel_mode == "scale":
        detector_pixels = int(round(args.detector_pixels * edge_scale))
    else:
        detector_pixels = int(args.detector_pixels)
    detector_pixels = max(1, detector_pixels)

    magnification = (args.dso + args.dde) / args.dso
    if args.detector_size_mm is None:
        object_diagonal = math.sqrt(sx**2 + sy**2)
        detector_size = object_diagonal * magnification * args.margin
    else:
        detector_size = args.detector_size_mm * (edge_scale if args.detector_pixel_mode == "scale" else 1.0)

    detector_partition = odl.uniform_partition(
        min_pt=[-detector_size / 2.0, -detector_size / 2.0],
        max_pt=[detector_size / 2.0, detector_size / 2.0],
        shape=[detector_pixels, detector_pixels],
    )
    geometry = odl.tomo.ConeBeamGeometry(
        angle_partition,
        detector_partition,
        src_radius=args.dso,
        det_radius=args.dde,
    )
    forward_op = odl.tomo.RayTransform(reco_space, geometry, impl=args.impl)
    info = {
        "volume_shape_zyx": [int(nz), int(ny), int(nx)],
        "volume_size_mm_xyz": [float(sx), float(sy), float(sz)],
        "spacing_mm_xyz": [float(v) for v in spacing_xyz],
        "detector_pixels": int(detector_pixels),
        "detector_size_mm": float(detector_size),
        "detector_spacing_mm": float(detector_size / detector_pixels),
        "dso_mm": float(args.dso),
        "dde_mm": float(args.dde),
        "n_proj": int(args.n_proj),
        "start_angle_rad": float(args.start_angle),
        "end_angle_rad": float(args.end_angle),
    }
    return forward_op, info


def save_sinogram_nifti(
    sinogram: np.ndarray,
    output_path: Path,
    geometry_info: dict[str, Any],
    description: str,
) -> None:
    sinogram = np.asarray(sinogram, dtype=np.float32)
    if sinogram.ndim != 3:
        raise ValueError(f"Expected a 3D sinogram, got shape {sinogram.shape}")

    # ODL cone-beam output is (angle, detector_v, detector_u). Save as
    # (detector_u, detector_v, angle) so the NIfTI axes are easy to inspect.
    nifti_data = np.transpose(sinogram, (2, 1, 0))
    detector_spacing = float(geometry_info["detector_spacing_mm"])
    angle_spacing = (
        float(geometry_info["end_angle_rad"]) - float(geometry_info["start_angle_rad"])
    ) / max(int(geometry_info["n_proj"]), 1)
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = detector_spacing
    affine[1, 1] = detector_spacing
    affine[2, 2] = angle_spacing

    image = nib.Nifti1Image(nifti_data.astype(np.float32, copy=False), affine)
    image.header.set_data_dtype(np.float32)
    image.header["descrip"] = description.encode("utf-8")[:79]
    nib.save(image, str(output_path))


def scale_label(edge_scale: float) -> str:
    if float(edge_scale).is_integer():
        return f"x{int(edge_scale)}"
    return "x" + str(edge_scale).replace(".", "p")


def placement_label(placement: str) -> str:
    return "corner13" if placement == "corner-third" else placement


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    suffix = "" if manifest.get("placement") == "center" else f"_{placement_label(str(manifest.get('placement')))}"
    manifest_path = output_dir / f"manifest{suffix}.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = str(manifest_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CT DICOM series from {args.dicom_dir}")
    volume_hu, source_metadata = load_ct_dicom_series(args.dicom_dir)
    background_hu = float(np.min(volume_hu) if args.background_hu is None else args.background_hu)
    volume_mu = hu_to_mu(volume_hu, args.mu_water, args.mu_air)
    background_mu = float(hu_to_mu(np.array([background_hu], dtype=np.float32), args.mu_water, args.mu_air)[0])

    pixel_spacing = source_metadata["pixel_spacing"]
    # DICOM PixelSpacing is row, column. The Ethos series is square in-plane, but
    # keep the mapping explicit for non-square data.
    spacing_xyz = (float(pixel_spacing[1]), float(pixel_spacing[0]), float(source_metadata["slice_spacing"]))

    manifest: dict[str, Any] = {
        "dicom_dir": str(args.dicom_dir),
        "source": source_metadata,
        "source_shape_zyx": [int(v) for v in volume_hu.shape],
        "source_hu_min": float(np.min(volume_hu)),
        "source_hu_max": float(np.max(volume_hu)),
        "background_hu": background_hu,
        "background_mu": background_mu,
        "mu_water": float(args.mu_water),
        "mu_air": float(args.mu_air),
        "detector_pixel_mode": args.detector_pixel_mode,
        "placement": args.placement,
        "sinogram_axis_order_in_nifti": ["detector_u", "detector_v", "angle"],
        "outputs": [],
    }

    for edge_scale in args.edge_scales:
        edge_scale = float(edge_scale)
        label = scale_label(edge_scale)
        target_shape = scaled_shape(tuple(int(v) for v in volume_hu.shape), edge_scale)
        print(f"\nPreparing {label}: padded volume shape z/y/x = {target_shape}")

        forward_op, geometry_info = build_cone_beam_operator(target_shape, spacing_xyz, args, edge_scale)
        output_label = label if args.placement == "center" else f"{label}_{placement_label(args.placement)}"
        start_indices = placement_start_indices(tuple(int(v) for v in volume_hu.shape), target_shape, args.placement)
        output_path = output_dir / f"ethos_conebeam_sinogram_{output_label}.nii.gz"
        output_info: dict[str, Any] = {
            "edge_scale": edge_scale,
            "label": label,
            "output_label": output_label,
            "path": str(output_path),
            "placement": args.placement,
            "object_start_index_zyx": [int(v) for v in start_indices],
            "object_end_index_zyx": [int(s + d) for s, d in zip(start_indices, volume_hu.shape)],
            "object_center_index_zyx": [float(s + (d - 1) / 2.0) for s, d in zip(start_indices, volume_hu.shape)],
            "geometry": geometry_info,
            "nifti_shape_du_dv_angle": [
                int(geometry_info["detector_pixels"]),
                int(geometry_info["detector_pixels"]),
                int(args.n_proj),
            ],
        }

        if args.dry_run:
            print(f"Dry run: would save {output_path}")
            manifest["outputs"].append(output_info)
            continue

        padded_mu_zyx, actual_start_indices = pad_mu(volume_mu, target_shape, background_mu, args.placement)
        output_info["object_start_index_zyx"] = [int(v) for v in actual_start_indices]
        padded_mu_xyz = np.transpose(padded_mu_zyx, (2, 1, 0))

        print(
            "Forward projecting "
            f"{label}: volume xyz={padded_mu_xyz.shape}, "
            f"placement={args.placement}, object_start_zyx={actual_start_indices}, "
            f"detector={geometry_info['detector_pixels']}x{geometry_info['detector_pixels']}, "
            f"n_proj={args.n_proj}"
        )
        sinogram = np.asarray(forward_op(padded_mu_xyz), dtype=np.float32)
        output_info["odl_sinogram_shape_angle_dv_du"] = [int(v) for v in sinogram.shape]
        output_info["sinogram_min"] = float(np.min(sinogram))
        output_info["sinogram_max"] = float(np.max(sinogram))
        output_info["sinogram_mean"] = float(np.mean(sinogram))

        save_sinogram_nifti(
            sinogram,
            output_path,
            geometry_info,
            f"Ethos CBCT cone-beam sinogram {output_label}",
        )
        print(f"Saved {output_path} with NIfTI shape {tuple(output_info['nifti_shape_du_dv_angle'])}")
        manifest["outputs"].append(output_info)

        del padded_mu_zyx
        del padded_mu_xyz
        del sinogram
        del forward_op

    write_manifest(output_dir, manifest)
    print(f"\nWrote manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
