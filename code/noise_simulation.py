from ctypes import Array

import numpy as np
from scipy import ndimage
from numpy.random import default_rng
from scipy.fft import fft2, ifft2, fftfreq
import cv2

import warnings
from typing import Optional, Sequence, Tuple

# usage configuration
def create_cbct_config(forward_op, reconstruction_op, cfg):
    """
    Create a comprehensive configuration dictionary for CBCT simulation
    """
    geo_config = {
        # ODL operators
        'forward_op': forward_op,
        'reconstruction_op': reconstruction_op,
    }
    config = geo_config | cfg
    
    return config

def validate_and_clip_sinogram(sinogram, max_attenuation=10.0):
    """
    Validate and clip sinogram values to reasonable range
    
    Parameters:
    -----------
    sinogram : ndarray
        Input sinogram
    max_attenuation : float
        Maximum reasonable attenuation value
        
    Returns:
    --------
    sinogram : ndarray
        Clipped sinogram
    """
    # Check for invalid values
    if np.any(np.isnan(sinogram)):
        print(f"Warning: Found {np.sum(np.isnan(sinogram))} NaN values in sinogram")
        sinogram = np.nan_to_num(sinogram, nan=0.0)
    
    if np.any(np.isinf(sinogram)):
        print(f"Warning: Found {np.sum(np.isinf(sinogram))} infinite values in sinogram")
        sinogram = np.nan_to_num(sinogram, posinf=max_attenuation, neginf=0.0)
    
    # Clip to reasonable range
    original_min, original_max = np.min(sinogram), np.max(sinogram)
    sinogram = np.clip(sinogram, 0.0, max_attenuation)
    
    if original_max > max_attenuation or original_min < 0:
        print(f"Clipped sinogram from [{original_min:.3f}, {original_max:.3f}] to [0, {max_attenuation}]")
    
    return sinogram

# Additional specialized artifact functions
def add_cone_beam_specific_artifacts(sinogram, cone_angle=10.0):
    """
    Add artifacts specific to cone beam geometry
    - Cone beam artifacts increase with cone angle
    - Z-axis sampling artifacts
    """
    cone_sinogram = sinogram.copy()
    detector_height = sinogram.shape[0] if len(sinogram.shape) > 2 else 1
    
    # Cone beam artifacts are more severe at detector edges
    if len(sinogram.shape) > 2:  # 3D case
        center_z = detector_height // 2
        z_positions = np.arange(detector_height) - center_z
        cone_factor = 1.0 + 0.1 * (np.abs(z_positions) / center_z) ** 2
        
        for z in range(detector_height):
            cone_sinogram[z, :, :] *= cone_factor[z]
    
    return cone_sinogram

def add_geometric_misalignment_artifacts(sinogram, misalignment_params=None):
    """
    Simulate geometric misalignment of source, object, and detector
    Common in CBCT due to mechanical tolerances
    """
    if misalignment_params is None:
        # Random small misalignments
        misalignment_params = {
            'source_shift_x': np.random.normal(0, 0.1),  # mm
            'source_shift_y': np.random.normal(0, 0.1),
            'detector_tilt': np.random.normal(0, 0.5),   # degrees
            'rotation_center_shift': np.random.normal(0, 0.2)  # mm
        }
    
    misaligned_sinogram = sinogram.copy()
    
    # Apply shifts and rotations based on misalignment
    # This is a simplified model - in practice, this would require
    # re-projection with corrected geometry
    
    shift_x = misalignment_params.get('source_shift_x', 0)
    shift_y = misalignment_params.get('source_shift_y', 0)
    
    if abs(shift_x) > 0.01 or abs(shift_y) > 0.01:
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        for i in range(sinogram.shape[0]):
            misaligned_sinogram[i, :] = cv2.warpAffine(
                sinogram[i, :].reshape(1, -1),
                shift_matrix,
                (sinogram.shape[1], 1)
            ).flatten()
    
    return misaligned_sinogram

def add_kV_variation_artifacts(sinogram, kV_variation=0.05):
    """
    Simulate kV (tube voltage) variations during scan
    Affects beam hardening and image contrast
    """
    kV_sinogram = sinogram.copy()
    n_projections = sinogram.shape[0]
    
    # kV variations affect the effective attenuation
    kV_variations = np.random.normal(1.0, kV_variation, n_projections)
    
    for i in range(n_projections):
        # Higher kV = more penetrating beam = less attenuation
        kV_sinogram[i, :] *= kV_variations[i]
    
    return kV_sinogram

def add_mA_variation_artifacts(sinogram, mA_variation=0.03):
    """
    Simulate mA (tube current) variations during scan
    Affects noise level and image quality
    """
    mA_sinogram = sinogram.copy()
    n_projections = sinogram.shape[0]
    
    # mA variations affect photon statistics (noise level)
    mA_variations = np.random.normal(1.0, mA_variation, n_projections)
    
    for i in range(n_projections):
        # Lower mA = fewer photons = more noise
        if mA_variations[i] < 1.0:
            noise_increase = (1.0 - mA_variations[i]) * 0.1
            mA_sinogram[i, :] += np.random.normal(0, noise_increase, sinogram.shape[1])
        mA_sinogram[i, :] *= mA_variations[i]
    
    return mA_sinogram

def add_filter_aging_artifacts(sinogram, filter_age_factor=0.02):
    """
    Simulate X-ray filter aging effects
    Filters accumulate deposits and change beam quality over time
    """
    aged_sinogram = sinogram.copy()
    detector_width = sinogram.shape[1]
    
    # Aging causes non-uniform beam hardening
    aging_pattern = 1.0 + filter_age_factor * np.random.beta(2, 5, detector_width)
    aged_sinogram *= aging_pattern[np.newaxis, :]
    
    return aged_sinogram

def hu2mu(hu, mu_water, mu_air):
    """Convert from HU to linear attenuation coefficient (mu)"""
    mu = hu / 1000.0 * (mu_water - mu_air) + mu_water
    return mu

def mu2hu(mu, mu_water, mu_air):
    """Convert from mu (linear attenuation coefficient) to HU (Hounsfield Unit)"""
    hu = 1000 * (mu - mu_water) / (mu_water - mu_air)
    return hu

def threshold_based_weighting(image, T1, T2):
    """Apply weight function to the image based on given two thresholds"""
    w_bone = (image - T1) / (T2 - T1)
    w_bone = np.clip(w_bone, 0, 1)
    bone = w_bone * image

    w_water = (T2 - image) / (T2 - T1)
    w_water = np.clip(w_water, 0, 1)
    water = w_water * image

    return water, bone

def _as_float32_3d_sinogram(sinogram: Array, name: str = "sinogram") -> Array:
    """Validate and return a 3D CBCT sinogram as float32."""
    s = np.asarray(sinogram, dtype=np.float32)
    if s.ndim != 3:
        raise ValueError(
            f"{name} must be a 3D CBCT sinogram with shape "
            f"(n_proj, det_v, det_u). Got shape {s.shape}."
        )
    return s

def _clean_log_sinogram(sinogram: Array, max_attenuation: Optional[float] = None) -> Array:
    """Replace invalid values and enforce nonnegative line integrals."""
    s = _as_float32_3d_sinogram(sinogram)
    if not np.isfinite(s).all():
        warnings.warn("Invalid values found in sinogram. NaN/Inf values were replaced.")
        upper = 10.0 if max_attenuation is None else float(max_attenuation)
        s = np.nan_to_num(s, nan=0.0, posinf=upper, neginf=0.0)
    s = np.maximum(s, 0.0)
    if max_attenuation is not None:
        s = np.minimum(s, float(max_attenuation))
    return s.astype(np.float32)

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)

# def add_beam_hardening_artifact(
#     sinogram: Array,
#     dense_sinogram: Optional[Array] = None,
#     metal_sinogram: Optional[Array] = None,
#     alpha_dense: float = 0.015,
#     beta_dense: float = 0.0,
#     alpha_metal: float = 0.06,
#     trace_noise_std: float = 0.0,
#     dense_trace_threshold: float = 0.05,
#     seed: Optional[int] = None,
# ) -> Array:
#     """
#     Add simplified polychromatic beam-hardening inconsistency.

#     This replaces the previous global-normalized threshold implementation.
#     The artifact should depend on the projected dense-material path length, not
#     on the global maximum of the total sinogram.

#     Parameters
#     ----------
#     sinogram:
#         Total log-domain projection, shape (n_proj, det_v, det_u).
#     dense_sinogram:
#         Projection of dense tissues, e.g., bone. If omitted, a weak fallback is
#         estimated from high total-projection values, but explicit dense projection
#         is preferred.
#     metal_sinogram:
#         Projection of metal material. If provided, a stronger nonlinear term is
#         applied along metal traces.
#     alpha_dense, beta_dense:
#         Dense-material nonlinear coefficients. The measured projection is reduced
#         because the beam becomes harder and more penetrating.
#     alpha_metal:
#         Stronger nonlinear coefficient for metal traces.
#     trace_noise_std:
#         Optional view/detector-dependent noise added only along dense traces.
#         This helps produce streaks after reconstruction.
#     """
#     p = _clean_log_sinogram(sinogram).astype(np.float64)

#     if dense_sinogram is None:
#         # Fallback only. It is less meaningful than a true bone/metal projection.
#         q = np.quantile(p[p > 0], 0.85) if np.any(p > 0) else 0.0
#         dense = np.maximum(p - q, 0.0)
#     else:
#         dense = _clean_log_sinogram(dense_sinogram).astype(np.float64)
#         if dense.shape != p.shape:
#             raise ValueError("dense_sinogram must have the same shape as sinogram.")

#     correction = float(alpha_dense) * dense**2 + float(beta_dense) * dense**3

#     if metal_sinogram is not None:
#         metal = _clean_log_sinogram(metal_sinogram).astype(np.float64)
#         if metal.shape != p.shape:
#             raise ValueError("metal_sinogram must have the same shape as sinogram.")
#         correction += float(alpha_metal) * metal**2
#         trace = metal > dense_trace_threshold
#     else:
#         trace = dense > dense_trace_threshold

#     p_out = p - correction

#     if trace_noise_std > 0 and np.any(trace):
#         r = _rng(seed)
#         noise = r.normal(0.0, trace_noise_std, size=p.shape)
#         p_out[trace] += noise[trace] * (1.0 + dense[trace])

#     return np.maximum(p_out, 0.0).astype(np.float32)

def add_beam_hardening_artifact(sinogram, streak_strength=0.8, density_threshold=0.6):
    """
    Simulates aggressive Beam Hardening STREAKS between dense objects in a 3D Cone Beam.
    Leaves soft tissue normal, severely penalizes dense paths.
    """
    n_proj, width, height = sinogram.shape
    sinogram_clean = np.maximum(sinogram.astype(np.float32), 0.0)
    
    s_max = np.max(sinogram_clean)
    if s_max == 0:
        return sinogram_clean
        
    # 1. Normalize to 0.0 -> 1.0 space
    s_norm = sinogram_clean / s_max
    
    # 2. Isolate the highly dense paths (bone/metal)
    # We use a threshold, so soft tissue (low s_norm) gets 0 penalty.
    # We use np.maximum to ensure the penalty mask doesn't go negative.
    dense_mask = np.maximum(s_norm - density_threshold, 0.0)
    
    # 3. Create a harsh non-linear penalty ONLY for the dense rays
    # Squaring the mask makes the penalty ramp up violently for the densest objects
    streak_penalty = streak_strength * (dense_mask ** 2)
    
    # 4. Apply the penalty to the normalized sinogram
    hardened_norm = s_norm - streak_penalty
    
    # 5. Scale back to original units and ensure no black-hole pixels
    hardened_sinogram = hardened_norm * s_max
    return np.maximum(hardened_sinogram, 0.0).astype(np.float32)

def add_scatter_artifact(sinogram, geometry_config, scatter_fraction=0.1, I0=1e5):
    """
    Simulates CBCT scatter radiation accurately in the linear photon domain.
    This creates the true "Dark Center / Cupping" artifact when reconstructed.
    """
    s0 = np.asarray(sinogram).astype(np.float64)
    s = s0.astype(np.float64, copy=True)

    if s.ndim != 3:
        raise ValueError(f"Expected 3D sinogram, got shape {s.shape}")

    # Ensure I0 is sensible (no forced constant)
    I0 = float(np.clip(I0 / scatter_fraction, 1e5, 1e9))

    # Primary intensity from log sinogram
    # If s contains negatives, exp(-s) can exceed 1; clamp to avoid I_primary > I0.
    s_clamped = np.maximum(s, 0.0)
    I_primary = I0 * np.exp(-s_clamped)

    # Interacting photons (those removed from primary beam)
    I_interacting = I0 - I_primary  # in [0, I0]

    # Blur only detector axes: axis0 is projection angle
    sigma = (0.0, scatter_fraction, scatter_fraction)
    I_scatter_base = ndimage.gaussian_filter(I_interacting, sigma=sigma, mode="nearest")

    # Scale scatter
    I_scatter = I_scatter_base * float(scatter_fraction)

    # Cap scatter so it can't exceed a fraction of primary locally (prevents washout)
    I_scatter = np.minimum(I_scatter, scatter_fraction * I_primary)

    # Add scatter
    I_total = I_primary + I_scatter
    I_total = np.maximum(I_total, 1e-12)

    s_scattered = -np.log(I_total / I0)
    s_scattered = s_scattered.astype(np.float32)

    return s_scattered

def add_truncation_artifact(sinogram, truncation_ratio=0.01):
    """
    Simulate truncation artifact - object extends beyond field of view
    Common in CBCT due to smaller detectors
    """
    detector_width = sinogram.shape[1]
    truncation_width = int(detector_width * truncation_ratio)
    
    # Create truncation mask
    truncated_sinogram = sinogram.copy()
    
    # Randomly truncate left or right side, or both
    if np.random.random() > 0.5:  # Left truncation
        truncated_sinogram[:, :truncation_width] = 0
    else:  # Right truncation
        truncated_sinogram[:, -truncation_width:] = 0
    
    return truncated_sinogram

def add_motion_artifact(sinogram, motion_type='rigid', motion_strength=1.0):
    """
    Simulate patient motion artifacts during acquisition - OPTIMIZED for 3D
    """
    motion_sinogram = sinogram.copy()
    n_projections = sinogram.shape[0]
    
    if motion_type == 'rigid':
        # Rigid body motion - shifts and rotations
        shift_x = np.random.uniform(-motion_strength, motion_strength, n_projections)
        shift_y = np.random.uniform(-motion_strength, motion_strength, n_projections)
        
        if len(sinogram.shape) == 3:
            # 3D cone beam: (projections, detector_height, detector_width)
            for i in range(n_projections):
                # Apply motion to the 2D detector image for this projection
                detector_image = motion_sinogram[i, :, :]  # 2D detector image
                
                # Check size limits for OpenCV
                if (detector_image.shape[0] < 32767 and detector_image.shape[1] < 32767):
                    shift_matrix = np.float32([[1, 0, shift_x[i]], [0, 1, shift_y[i]]])
                    motion_sinogram[i, :, :] = cv2.warpAffine(
                        detector_image.astype(np.float32),
                        shift_matrix, 
                        (detector_image.shape[1], detector_image.shape[0])
                    )
                else:
                    # Fallback for large images - use scipy instead
                    from scipy.ndimage import shift
                    motion_sinogram[i, :, :] = shift(detector_image, 
                                                   [shift_y[i], shift_x[i]], 
                                                   mode='nearest')
        else:
            # 2D fan beam: (projections, detector_width) - your original code
            for i in range(n_projections):
                if sinogram.shape[1] < 32767:
                    shift_matrix = np.float32([[1, 0, shift_x[i]], [0, 1, shift_y[i]]])
                    motion_sinogram[i, :] = cv2.warpAffine(
                        motion_sinogram[i, :].reshape(1, -1).astype(np.float32), 
                        shift_matrix, 
                        (sinogram.shape[1], 1)
                    ).flatten()
                else:
                    # Fallback for large detector
                    from scipy.ndimage import shift
                    motion_sinogram[i, :] = shift(motion_sinogram[i, :], shift_x[i], mode='nearest')
    
    elif motion_type == 'respiratory':
        # Periodic respiratory motion
        n_proj, width, height = sinogram.shape
    
        # 1. Angles and Phase
        theta = np.linspace(0, 2 * np.pi, n_proj)
        phase = np.linspace(0, 2 * np.pi * motion_strength, n_proj)
        
        # 2. Dynamic Shifts (Pixels to move for each projection)
        shift_height = 3.0 * np.sin(phase)
        shift_width = 2.0 * np.sin(phase) * np.sin(theta)
        
        # 3. Enter the Frequency Domain
        # Get the normalized frequency coordinates for our detector dimensions
        freq_w = fftfreq(width)
        freq_h = fftfreq(height)
        
        # Create a 2D grid of the frequencies (Shape: width, height)
        W, H = np.meshgrid(freq_w, freq_h, indexing='ij')
        
        # 4. Create the Phase Shift Matrix (Vectorized Broadcasting)
        # The math: exp(-i * 2 * pi * (u*dx + v*dy))
        # We broadcast the 1D shifts across the 2D frequency grid to make a 3D matrix
        phase_multiplier = np.exp(-2j * np.pi * (
            W[np.newaxis, :, :] * shift_width[:, np.newaxis, np.newaxis] + 
            H[np.newaxis, :, :] * shift_height[:, np.newaxis, np.newaxis]
        ))
        
        # 5. Execute the Shift via FFT
        # Transform to frequency -> Multiply by phase -> Transform back to spatial
        sinogram_fft = fft2(sinogram, axes=(1, 2))
        sinogram_shifted = np.real(ifft2(sinogram_fft * phase_multiplier, axes=(1, 2)))
        
        # Prevent tiny floating-point errors from creating negative attenuation
        return np.maximum(sinogram_shifted, 0.0).astype(np.float32)
    
    return motion_sinogram

def add_metal_streak_artifacts(sinogram, metal_mask, geometry_config, metal_mu=10.0):
    """
    Physically accurate metal artifact simulation - OPTIMIZED
    """
    metal_sinogram = sinogram.copy()
    
    if np.sum(metal_mask) == 0:
        return metal_sinogram
    
    # Forward project metal mask to get metal traces in sinogram
    metal_projection = geometry_config['forward_op'](metal_mask * metal_mu)
    metal_projection = np.asarray(metal_projection)
    
    # Metal causes complete absorption in some rays
    metal_traces = metal_projection > 0.1  # Threshold for metal presence
    
    # Apply photon starvation where metal blocks the beam
    metal_sinogram[metal_traces] = metal_sinogram[metal_traces] + \
                                   np.random.exponential(2.0, np.sum(metal_traces))
    
    # OPTIMIZED: Vectorized streak artifact creation
    if len(metal_traces.shape) == 3:
        # 3D case: (projections, detector_height, detector_width)
        metal_sinogram = _add_streaks_3d_vectorized(metal_sinogram, metal_traces)
    else:
        # 2D case: (projections, detector_width)
        metal_sinogram = _add_streaks_2d_vectorized(metal_sinogram, metal_traces)
    
    return metal_sinogram

def _add_streaks_2d_vectorized(sinogram, metal_traces):
    """Optimized 2D streak artifact addition"""
    
    # Find all metal pixels at once
    proj_indices, det_indices = np.where(metal_traces)
    
    if len(proj_indices) == 0:
        return sinogram
    
    # Vectorized interpolation for all metal pixels
    for proj_idx in np.unique(proj_indices):
        # Get all metal pixels for this projection
        metal_pixels = det_indices[proj_indices == proj_idx]
        
        # Use numpy broadcasting for efficient interpolation
        for mp in metal_pixels:
            # Find left and right boundaries efficiently
            row = metal_traces[proj_idx, :]
            
            # Find nearest non-metal pixels using vectorized operations
            non_metal_indices = np.where(~row)[0]
            
            if len(non_metal_indices) >= 2:
                # Find closest non-metal pixels to the left and right
                left_candidates = non_metal_indices[non_metal_indices < mp]
                right_candidates = non_metal_indices[non_metal_indices > mp]
                
                if len(left_candidates) > 0 and len(right_candidates) > 0:
                    left_idx = left_candidates[-1]  # Closest to the left
                    right_idx = right_candidates[0]  # Closest to the right
                    
                    # Linear interpolation
                    sinogram[proj_idx, mp] = (sinogram[proj_idx, left_idx] + 
                                            sinogram[proj_idx, right_idx]) / 2
                    sinogram[proj_idx, mp] += np.random.normal(0, 0.1)
    
    return sinogram

def _add_streaks_3d_vectorized(sinogram, metal_traces):
    """Optimized 3D streak artifact addition"""
    
    # Find all metal pixels at once
    proj_indices, height_indices, width_indices = np.where(metal_traces)
    
    if len(proj_indices) == 0:
        return sinogram
    
    # Group by projection and height for efficient processing
    for proj_idx in np.unique(proj_indices):
        proj_mask = proj_indices == proj_idx
        heights_in_proj = height_indices[proj_mask]
        widths_in_proj = width_indices[proj_mask]
        
        for height_idx in np.unique(heights_in_proj):
            # Get all metal pixels for this projection and detector row
            height_mask = heights_in_proj == height_idx
            metal_pixels = widths_in_proj[height_mask]
            
            # Vectorized interpolation for this detector row
            row = metal_traces[proj_idx, height_idx, :]
            non_metal_indices = np.where(~row)[0]
            
            if len(non_metal_indices) >= 2:
                # Batch process all metal pixels in this row
                for mp in metal_pixels:
                    left_candidates = non_metal_indices[non_metal_indices < mp]
                    right_candidates = non_metal_indices[non_metal_indices > mp]
                    
                    if len(left_candidates) > 0 and len(right_candidates) > 0:
                        left_idx = left_candidates[-1]
                        right_idx = right_candidates[0]
                        
                        # Linear interpolation
                        sinogram[proj_idx, height_idx, mp] = (
                            sinogram[proj_idx, height_idx, left_idx] + 
                            sinogram[proj_idx, height_idx, right_idx]) / 2
                        sinogram[proj_idx, height_idx, mp] += np.random.normal(0, 0.1)
    
    return sinogram

def add_quantum_noise(sinogram, lam=1e1):
    """
    Add quantum (Poisson) noise using lambda parameter
    
    Parameters:
    -----------
    lam : float
        Noise scaling parameter (higher = less noise)
        Range: 100-10000
        - lam = 100: Very noisy (low dose)
        - lam = 1000: Standard noise (medium dose)  
        - lam = 5000: Low noise (high dose)
    """
    measurements = sinogram # np.exp(-sinogram)
    
    if lam > 0:
        poisson_param = measurements / lam
        poisson_param = np.clip(poisson_param, 1e-10, 1e50)  # Prevent extreme values
        noisy_measurements = lam * np.random.poisson(poisson_param)
    else:
        noisy_measurements = measurements
    
    # Ensure no zero measurements (would cause log(0))
    noisy_measurements = np.maximum(noisy_measurements, 1e-10)
    
    # Convert back to attenuation: μt = -ln(I)
    noisy_sinogram = noisy_measurements #-np.log(noisy_measurements)
    
    return noisy_sinogram

def add_detector_artifacts(sinogram, bad_pixel_ratio=0.001, gain_variation=0.02):
    """
    Simulate detector imperfections - ROBUST VERSION
    """
    detector_sinogram = sinogram.copy()
    
    # Validate input
    if np.any(np.isnan(sinogram)) or np.any(np.isinf(sinogram)):
        print("Warning: Invalid values in sinogram input to detector artifacts")
        detector_sinogram = np.nan_to_num(detector_sinogram, nan=0.0, posinf=10.0, neginf=0.0)
    
    # Bad pixels - use safe random values
    n_bad_pixels = int(sinogram.size * bad_pixel_ratio)
    if n_bad_pixels > 0:
        bad_positions = np.random.choice(sinogram.size, n_bad_pixels, replace=False)
        flat_sinogram = detector_sinogram.flatten()
        
        # Use safe range for bad pixel values
        sinogram_range = np.percentile(sinogram[sinogram > 0], [10, 90])  # Use percentiles
        safe_min, safe_max = max(0, sinogram_range[0]), min(10, sinogram_range[1])
        
        if safe_max > safe_min:
            flat_sinogram[bad_positions] = np.random.uniform(safe_min, safe_max, n_bad_pixels)
        else:
            flat_sinogram[bad_positions] = 0.1  # Fallback value
        
        detector_sinogram = flat_sinogram.reshape(sinogram.shape)
    
    # Gain variations - use safe range
    gain_map = np.random.normal(1.0, gain_variation, sinogram.shape[-1])  # Last dimension
    gain_map = np.clip(gain_map, 0.5, 2.0)  # Prevent extreme gains
    
    # Apply gain correction
    if len(sinogram.shape) == 3:
        detector_sinogram *= gain_map[np.newaxis, np.newaxis, :]
    else:
        detector_sinogram *= gain_map[np.newaxis, :]
    
    return detector_sinogram

def add_ring_artifacts(sinogram, n_rings=5, ring_intensity=0.1):
    """
    Simulate ring artifacts by applying errors to detector columns (width).
    Assumes sinogram shape is (projections, width, height).
    """
    ring_sinogram = sinogram.astype(np.float32).copy()
    num_proj, det_w, det_h = ring_sinogram.shape
    
    # 1. Find the object's footprint along the WIDTH axis (index 1)
    # Average across projections (0) and height (2) to get the 1D width profile
    width_profile = np.mean(sinogram, axis=(0, 2))
    
    # Threshold to find where the object actually exists
    threshold = 0.05 * np.max(width_profile)
    valid_cols = np.where(width_profile > threshold)[0]
    
    if len(valid_cols) == 0:
        return sinogram # No object detected
        
    u_min, u_max = valid_cols[0], valid_cols[-1]
    
    # 2. Pick random detector columns ONLY within the object's footprint
    available_cols = u_max - u_min
    actual_rings = min(n_rings, available_cols)
    
    # Select random columns along the width dimension
    ring_indices = np.random.choice(np.arange(u_min, u_max), size=actual_rings, replace=False)
    
    # 3. Apply the artifact
    for pos in ring_indices:
        # Create a slight variation across projections for realism
        error_streak = np.random.normal(ring_intensity, ring_intensity * 0.1, size=num_proj)
        
        # 4. Mask out the air
        # Note the indexing: [:, pos, :] fixes the WIDTH, spanning all heights.
        object_mask = (sinogram[:, pos, :] > 1e-4).astype(np.float32)
        
        # Apply the error streak across the height, masked by the object
        ring_sinogram[:, pos, :] += error_streak[:, np.newaxis] * object_mask
        
    return ring_sinogram

def add_cupping_artifact(sinogram, cupping_strength=0.15):
    # sinogram: (360, 1024, 1024) - values are mu (attenuation)
    num_proj, det_h, det_w = sinogram.shape
    
    # 1. Project the total attenuation to find the object "footprint"
    # We take the mean across all projections to get a stable 2D map
    projection_avg = np.mean(sinogram, axis=0)
    
    # 2. Use a robust threshold (e.g., 5% of the maximum attenuation)
    # This ignores air (0) and focuses on the object
    threshold = 0.05 * np.max(projection_avg)
    mask_binary = projection_avg > threshold
    
    if not np.any(mask_binary):
        return sinogram # Object not found
    
    # 3. Find the center of the object using indices
    coords = np.argwhere(mask_binary)
    v_min, u_min = coords.min(axis=0)
    v_max, u_max = coords.max(axis=0)
    
    center_v = (v_min + v_max) / 2
    center_u = (u_min + u_max) / 2
    
    # Radius of the object on the detector
    radius_v = (v_max - v_min) / 2
    radius_u = (u_max - u_min) / 2
    
    # 4. Generate the coordinate grid
    v = np.arange(det_h)
    u = np.arange(det_w)
    uu, vv = np.meshgrid(u, v)
    
    # 5. Calculate normalized distance from the object center
    # distance is 0 at center of object, 1 at its boundaries
    dist_sq = ((uu - center_u) / radius_u)**2 + ((vv - center_v) / radius_v)**2
    
    # 6. Apply physics: Beam hardening is proportional to path length.
    # We only apply the 'dip' where the sinogram actually has values (the object).
    # Mask = 1.0 - [Strength * (1.0 - Normalized_Dist_Squared)]
    # This creates a dip in the middle (where dist=0) and fades to 1.0 at edges.
    cupping_profile = 1.0 - (cupping_strength * np.maximum(0, 1.0 - dist_sq))
    
    # Apply to the 3D volume
    return sinogram * cupping_profile[np.newaxis, :, :]


def add_incomplete_projection_artifacts(sinogram, missing_angle_ranges=None):
    """
    Simulate incomplete angular sampling (limited angle artifacts)
    Common when patient movement prevents full rotation
    """
    incomplete_sinogram = sinogram.copy()
    n_projections = sinogram.shape[0]
    
    if missing_angle_ranges is None:
        # Randomly create 1-2 missing angular ranges
        n_gaps = np.random.randint(1, 3)
        missing_angle_ranges = []
        for _ in range(n_gaps):
            gap_start = np.random.randint(0, n_projections - 20)
            gap_size = np.random.randint(10, 30)
            gap_end = min(gap_start + gap_size, n_projections)
            missing_angle_ranges.append((gap_start, gap_end))
    
    # Zero out projections in missing ranges
    for start, end in missing_angle_ranges:
        incomplete_sinogram[start:end, :] = 0
    
    return incomplete_sinogram

def add_dark_current_noise(sinogram, I0=100000, dark_current_level=0.5):
    """
    Physically accurate dark current simulation.
    Converts sinogram to intensity, adds electronic noise, and converts back.
    
    Parameters:
    - I0: Blank scan photon intensity (e.g., 50000 photons per ray)
    - dark_current_mean: Average baseline detector count (even with 0 photons)
    - dark_current_std: Variance of the dark current
    """
    s_64 = sinogram.astype(np.float64)
    
    # Calculate I_true just to act as the denominator for our noise
    I_true = I0 * np.exp(-s_64)
    
    # 2. Generate random frame-by-frame dark noise
    noise = np.random.normal(0.0, dark_current_level, s_64.shape) * dark_current_level
    noise = np.clip(noise, -dark_current_level, dark_current_level)
    
    # 3. Calculate the noise ratio: (Noise / I_true)
    # Use a tiny floor to prevent divide-by-zero in complete shadow
    I_true_safe = np.maximum(I_true, 1e-3)
    noise_ratio = noise / I_true_safe
    
    # 4. Simulate the detector "bottoming out" (clipping negative photons)
    # If noise drops the total photons below a tiny threshold (e.g., 0.001), 
    # we cap the negative ratio to prevent taking the log of a negative number.
    clip_floor = 1e-3
    min_ratio = (clip_floor / I_true_safe) - 1.0
    noise_ratio = np.maximum(noise_ratio, min_ratio)
    
    # 5. The Delta Math: S_new = S - ln(1 + ratio)
    # np.log1p(x) is mathematically equivalent to np.log(1 + x) but preserves float precision
    noisy_sinogram = s_64 - np.log1p(noise_ratio)
    
    return noisy_sinogram.astype(np.float32)

def add_lag_artifacts(sinogram, lag_factor=0.02):
    """
    Simulate detector lag (ghosting) artifacts
    Previous frame influences current frame in flat panel detectors
    """
    lagged_sinogram = sinogram.copy()
    
    for i in range(1, sinogram.shape[0]):
        # Current frame is influenced by previous frame
        lagged_sinogram[i] += lag_factor * sinogram[i-1]
    
    return lagged_sinogram

def add_bow_tie_filter_artifacts(sinogram, miscalibration_strength=0.02):
    """
    Simulate bow-tie filter effects and imperfections
    Bow-tie filters shape the X-ray beam but can introduce artifacts
    """
    filtered_sinogram = sinogram.copy().astype(np.float32)
    n_proj, width, height = sinogram.shape

    width_profile = np.sum(sinogram, axis=(0, 2))
        
    # Calculate the Center of Mass (where is the object thickest?)
    indices = np.arange(width)
    total_mass = np.sum(width_profile)
        
    if total_mass > 0:
        center_idx = np.sum(indices * width_profile) / total_mass
    else:
        center_idx = width / 2.0
    
    x = (np.arange(width) - center_idx) / (width / 2.0)
    
    # Parabola: 0 at center, thick at edges
    bow_tie_thickness = x**2  
    
    # Add mechanical imperfections/noise
    imperfections = np.sin(10 * np.pi * x) + np.random.normal(0, miscalibration_strength * 0.1, width)
    
    # Combine into the residual error array (Shape: 1D array of size 'width')
    residual_error = miscalibration_strength * bow_tie_thickness * (1.0 + imperfections)

    residual_error = (residual_error - residual_error.min()) / (residual_error.max() - residual_error.min() +1e-12)
    residual_error = miscalibration_strength * (2 * residual_error - 1) + 1
    # residual_error = residual_error.clip(0.)
    
    # --- 3. VECTORIZED BROADCASTING ---
    # We must reshape our 1D array of size (width,) to (1, width, 1)
    # This allows NumPy to broadcast it instantly across n_proj and height
    filtered_sinogram *= residual_error[np.newaxis, :, np.newaxis]
    
    return filtered_sinogram

def add_heel_effect(sinogram, heel_variation=0.2, axis='width'):
    s_64 = sinogram.astype(np.float64)
    num_proj, det_w, det_h = s_64.shape
    
    # 1. Base spatial gradient
    if 'axis' == 'vertical':
        spatial_array = np.linspace(-1, 1, det_w)
    else:
        spatial_array = np.linspace(-1, 1, det_h)
        
    # 2. Your exponential curve!
    curve = np.exp(spatial_array)
    
    # 3. Safely normalize and scale
    # First, get a curve with mean=0 and std=1 (Z-score normalization)
    normalized_curve = (curve - curve.mean()) / curve.std()
    
    # Next, multiply by your desired variation (e.g., 0.20 means +/- 20% swing)
    # Then add 1.0 so the baseline multiplier is 1.0
    heel_correction = (normalized_curve * (heel_variation / 2.0)) + 1.0
    heel_correction = np.clip(heel_correction, 0.2, 1.5)
    
    # 4. Apply multiplicatively
    if axis == 'up':
        s_new = s_64 * heel_correction[np.newaxis, np.newaxis, :]
    elif axis == 'down':
        s_new = s_64 * heel_correction[np.newaxis, np.newaxis, ::-1]
    else:
        s_new = s_64 * heel_correction[np.newaxis, :, np.newaxis]
        
    return s_new.astype(np.float32)

def add_gantry_vibration_artifacts(sinogram, vibration_frequency=2.0, vibration_amplitude=0.5):
    """
    Simulate gantry mechanical vibrations using scipy (no size limits)
    """
    from scipy.ndimage import shift
    
    vibrated_sinogram = sinogram.copy()
    n_projections = sinogram.shape[0]
    
    # Create vibration pattern
    projection_angles = np.linspace(0, 2*np.pi, n_projections)
    vibration_x = vibration_amplitude * np.sin(vibration_frequency * projection_angles)
    vibration_y = vibration_amplitude * np.cos(vibration_frequency * projection_angles * 1.3)
    
    # Apply vibrations using scipy.ndimage.shift (no size limits)
    for i in range(n_projections):
        shifts = [vibration_y[i], vibration_x[i]]  # [height_shift, width_shift]
        vibrated_sinogram[i, :, :] = shift(vibrated_sinogram[i, :, :], shifts, mode='nearest')
        
    return vibrated_sinogram

def add_partial_volume_artifacts(sinogram, voxel_size=0.5, I0=1e5):
    """
    Simulates true Partial Volume Effect (PVE) by averaging in the 
    linear photon intensity domain, generating realistic non-linear edge streaks.
    """
    # 1. Strict float64 precision 
    I0 = min(max(I0 / (voxel_size + 1e-3), 1e5), 1e7)
    s_64 = sinogram.astype(np.float64)
    
    # 2. Convert back to raw photons
    # PVE happens as physical photons hit the detector pixel area
    raw_intensity = I0 * np.exp(-s_64)
    
    # 3. Apply the spatial averaging (detector footprint / focal spot blur)
    # We blur the height and width, but NOT the projection angles (axis 0).
    # sigma format: (angle_sigma, height_sigma, width_sigma)
    blurred_intensity = ndimage.gaussian_filter(
        raw_intensity, 
        sigma=(0, voxel_size, voxel_size)
    )
    
    # 4. Prevent math domain errors (photon starvation floor)
    blurred_intensity = np.maximum(blurred_intensity, 1e-12)
    
    # 5. Convert back to the log domain!
    # This non-linear roundtrip is what creates the dark streaks between dense objects
    pv_sinogram = -np.log(blurred_intensity / float(I0))
    
    return pv_sinogram.astype(np.float32)

def add_exponential_edge_gradient_artifacts(sinogram, edge_width=10):
    """
    Simulate edge gradient artifacts at field boundaries - 3D compatible
    """
    edge_sinogram = sinogram.copy()
    
    if len(sinogram.shape) == 3:
        # 3D cone beam: (projections, detector_height, detector_width)
        detector_height, detector_width = sinogram.shape[1], sinogram.shape[2]
        edge_width = min(edge_width, min(detector_height, detector_width) // 4)
        
        # Create edge gradients for width direction (left/right edges)
        left_edge_w = np.exp(-np.arange(edge_width) / (edge_width * 0.3))
        right_edge_w = np.exp(-np.arange(edge_width)[::-1] / (edge_width * 0.3))
        
        # Apply width edge effects - broadcast correctly for 3D
        edge_sinogram[:, :, :edge_width] *= left_edge_w[np.newaxis, np.newaxis, :]
        edge_sinogram[:, :, -edge_width:] *= right_edge_w[np.newaxis, np.newaxis, :]
        
        # Create edge gradients for height direction (top/bottom edges)
        left_edge_h = np.exp(-np.arange(edge_width) / (edge_width * 0.3))
        right_edge_h = np.exp(-np.arange(edge_width)[::-1] / (edge_width * 0.3))
        
        # Apply height edge effects
        edge_sinogram[:, :edge_width, :] *= left_edge_h[np.newaxis, :, np.newaxis]
        edge_sinogram[:, -edge_width:, :] *= right_edge_h[np.newaxis, :, np.newaxis]
        
    else:
        # 2D fan beam: (projections, detector_width) - original logic
        detector_width = sinogram.shape[1]
        edge_width = min(edge_width, detector_width // 4)
        
        left_edge = np.exp(-np.arange(edge_width) / (edge_width * 0.3))
        right_edge = np.exp(-np.arange(edge_width)[::-1] / (edge_width * 0.3))
        
        edge_sinogram[:, :edge_width] *= left_edge[np.newaxis, :]
        edge_sinogram[:, -edge_width:] *= right_edge[np.newaxis, :]
    
    return edge_sinogram

def add_contrast_agent_artifacts(sinogram, geometry_config, metal_mask, metal_mu, contrast_concentration=0.01):
    """
    Simulate artifacts from contrast agents (iodine, barium, etc.)
    High Z materials cause beam hardening and streaking
    """
    metal_projection = geometry_config['forward_op'](metal_mask)
    metal_projection = np.asarray(metal_projection)
    S_ideal = sinogram + (metal_projection * metal_mu)
    
    # 2. Simulate the ARTIFACT (Non-linear Beam Hardening)
    # The denser the total path (anatomy + iodine), the more the beam hardens.
    # We square the ideal sinogram to force a massive mathematical penalty 
    # strictly along the rays that pass through the contrast agent.
    hardening_penalty = contrast_concentration * (S_ideal ** 2)
    
    # 3. Apply the penalty (Subtracting from attenuation because hard beams penetrate easier)
    S_measured = S_ideal - hardening_penalty
    
    # 4. Prevent physically impossible negative attenuation
    S_measured = np.maximum(S_measured, 0.0)
    
    return S_measured.astype(np.float32)

def add_temperature_drift_artifacts(sinogram, drift_rate=0.001):
    """
    Simulate temperature-related detector drift during scan
    Causes gradual intensity changes over time
    """
    # 1. Float32 is plenty of precision for a multiplier here
    drift_sinogram = sinogram.copy().astype(np.float32)
    n_projections = sinogram.shape[0]
    
    # 2. Create the time axis
    time_points = np.linspace(0, 1, n_projections)
    
    # 3. Create the Multiplicative Scale (Baseline is exactly 1.0)
    # E.g., if drift_rate=0.01, it drifts from 1.0 up to 1.01 over the scan
    drift_multiplier = 1.0 + (drift_rate * time_points) + \
                       1.0 * np.sin(4 * np.pi * time_points)
    
    # 4. INSTANT VECTORIZED MULTIPLICATION (No for loop!)
    # np.newaxis turns shape (N,) into (N, 1, 1) so it broadcasts across H and W
    drift_sinogram *= drift_multiplier[:, np.newaxis, np.newaxis]
    
    return drift_sinogram

def add_phosphor_afterglow_artifacts(sinogram, afterglow_factor=0.01, decay_constant=0.8):
    """
    Simulate phosphor afterglow in indirect conversion detectors
    Previous exposures leave residual signal
    """
    afterglow_sinogram = sinogram.copy()
    afterglow_memory = np.zeros_like(sinogram[0, :])
    
    for i in range(sinogram.shape[0]):
        # Add afterglow from previous exposures
        afterglow_sinogram[i, :] += afterglow_memory * afterglow_factor
        
        # Update afterglow memory (decays exponentially)
        afterglow_memory = afterglow_memory * decay_constant + sinogram[i, :]
    
    return afterglow_sinogram

def cbct_artifact_simulation_3D(image, metal_mask, config, contrast_mask=None):
    """
    Comprehensive CBCT artifact simulation with physical accuracy
    """
    # "top & bottom" assumed along axis=0 (common for z / slice dimension).
    # If your volume is (H, W, D) instead of (D, H, W), change pad_axis accordingly.
    
    # Convert HU to mu
    mu_water = config['mu_water']
    mu_air = config['mu_air']
    image_mu = hu2mu(image, mu_water, mu_air)
    
    # Separate tissue types
    print(config.get('T1', -200))
    print(config.get('T2', 1500))
    T1 = hu2mu(config.get('T1', -200), mu_water, mu_air)
    T2 = hu2mu(config.get('T2', 1500), mu_water, mu_air)
    water_component, bone_component = threshold_based_weighting(image_mu, T1, T2)
    
    # Apply tissue-specific scaling
    water_component *= config.get('water_level', 1.0)
    bone_component *= config.get('bone_level', 1.0)
    
    # Handle metal regions
    metal_component = np.zeros_like(image_mu)
    if np.sum(metal_mask) > 0 and config.get('add_metal_artifacts', True):
        metal_mu_value = config.get('metal_mu', 10.0)
        metal_component = metal_mask * metal_mu_value
        # Remove tissue where metal is present
        water_component[metal_mask > 0] = 0
        bone_component[metal_mask > 0] = 0
    
    # Combine all components
    total_image = water_component + bone_component + metal_component
    
    # Forward projection to create sinogram
    forward_op = config['forward_op']  # This should be your ODL forward operator
    sinogram = forward_op(total_image)
    p_dense = forward_op(water_component)
    p_metal = forward_op(metal_component)
    sinogram = np.asarray(sinogram)
    
    # Add various artifacts in physically meaningful order
    
    # 1. X-ray tube and beam-related artifacts
    if config.get('add_quantum_noise', True):
        lam = config.get('lam', 1e2)
        sinogram = add_quantum_noise(sinogram, lam)
        
    if config.get('add_partial_volume', True):
        print('add_partial_volume')
        voxel_size = config.get('voxel_size', 0.5)
        sinogram = add_partial_volume_artifacts(sinogram, voxel_size)
    
    # 3. Scatter (occurs during X-ray interaction with matter)
    if config.get('add_scatter', True):
        scatter_fraction = config.get('scatter_fraction', 0.1)
        sinogram = add_scatter_artifact(sinogram, config, scatter_fraction)
    
    if config.get('add_bow_tie_filter', True):
        miscalibration_strength = config.get('miscalibration_strength', 0.1)
        sinogram = add_bow_tie_filter_artifacts(sinogram, config.get('filter_profile', miscalibration_strength))
    
    # 2. Beam hardening and cupping (occurs during X-ray interaction with matter) 
    if config.get("add_beam_hardening", True):
        sinogram = add_beam_hardening_artifact(sinogram, config.get("beam_hardening_strength", 0.6))
        # sinogram = add_beam_hardening_artifact(
        #     sinogram,
        #     dense_sinogram=p_dense,
        #     metal_sinogram=p_metal,
        #     alpha_dense=config.get("alpha_dense", config.get("beam_hardening_strength", 0.015)),
        #     beta_dense=config.get("beta_dense", 0.0),
        #     alpha_metal=config.get("alpha_metal", 0.06),
        #     trace_noise_std=config.get("beam_hardening_strength", 0.1),
        #     dense_trace_threshold=config.get("dense_trace_threshold", 0.05),
        #     seed=config.get("seed", None),
        # )
    
    if config.get('add_cupping', True):
        print('add_cupping')
        cupping_strength = config.get('cupping_strength', 0.15)
        sinogram = add_cupping_artifact(sinogram, cupping_strength)
    
    # 4. High-attenuation material artifacts
    if np.sum(metal_mask) > 0 and config.get('add_metal_artifacts', True):
        sinogram = add_metal_streak_artifacts(sinogram, metal_mask, config, 
                                            config.get('metal_mu', 10.0))
    
    if np.sum(metal_mask) > 0 and config.get('add_contrast_artifacts', True):
        contrast_concentration = config.get('contrast_concentration', 5.0)
        sinogram = add_contrast_agent_artifacts(sinogram, config, metal_mask, config.get('metal_mu', 10.0), contrast_concentration)
    
    # 6. Motion artifacts (occur during acquisition)
    if config.get('add_motion', True):
        print("add motion")
        motion_type = config.get('motion_type', 'rigid')
        motion_strength = config.get('motion_strength', 1.0)
        sinogram = add_motion_artifact(sinogram, motion_type, motion_strength)
    
    # 7. Mechanical and gantry-related artifacts
    if config.get('add_gantry_vibration', True):
        print("add_gantry_vibration")
        vib_freq = config.get('vibration_frequency', 2.0)
        vib_amp = config.get('vibration_amplitude', 0.5)
        sinogram = add_gantry_vibration_artifacts(sinogram, vib_freq, vib_amp)
    
    # 8. Incomplete data artifacts
    if config.get('add_incomplete_projections', True):
        print("add_incomplete_projections")
        missing_ranges = config.get('missing_angle_ranges', None)
        sinogram = add_incomplete_projection_artifacts(sinogram, missing_ranges)
    
    if config.get('add_truncation', True):
        print("add_truncation")
        truncation_ratio = config.get('truncation_ratio', 0.1)
        sinogram = add_truncation_artifact(sinogram, truncation_ratio)
    
    # 9. Detector-related artifacts
    if config.get('add_ring_artifacts', True):
        n_rings = config.get('n_rings', 5)
        ring_intensity = config.get('ring_intensity', 0.1)
        sinogram = add_ring_artifacts(sinogram, n_rings, ring_intensity)
        print(n_rings, ring_intensity)
    
    if config.get('add_detector_artifacts', True):
        bad_pixel_ratio = config.get('bad_pixel_ratio', 0.001)
        gain_variation = config.get('gain_variation', 0.02)
        sinogram = add_detector_artifacts(sinogram, bad_pixel_ratio, gain_variation)
    
    if config.get('add_lag_artifacts', True):
        lag_factor = config.get('lag_factor', 0.02)
        sinogram = add_lag_artifacts(sinogram, lag_factor)
    
    # 11. Field and edge artifacts
    if config.get('add_edge_gradients', True):
        edge_width = config.get('edge_width', 10)
        sinogram = add_exponential_edge_gradient_artifacts(sinogram, edge_width)
    
    if config.get('add_electronic_noise', True):
        noise_std = config.get('electronic_noise_std', 0.01)
        sinogram = add_dark_current_noise(sinogram, dark_current_level=noise_std)
    
    if config.get('add_dark_current', True):
        dark_level = config.get('dark_current_level', 0.005)
        sinogram = add_dark_current_noise(sinogram, dark_current_level=dark_level)

    if config.get('add_temperature_drift', True):
        drift_rate = config.get('drift_rate', 0.001)
        sinogram = add_temperature_drift_artifacts(sinogram, drift_rate)
    
    if config.get('add_heel_effect', True):
        anode_angle = config.get('anode_angle', 0.3)
        direction = config.get('axis', 'width')
        sinogram = add_heel_effect(sinogram, anode_angle, direction)
    
    # Reconstruction
    reconstruction_op = config['reconstruction_op']  # This should be your ODL reconstruction operator
    reconstructed_image = reconstruction_op(sinogram)
    
    # Convert back to HU
    result_hu = mu2hu(reconstructed_image, mu_water, mu_air)
    result_hu = np.clip(result_hu, image.min(), image.max())
    
    return result_hu
