"""
CBCT Simulation Configuration System
====================================
"""

import numpy as np

class CBCTConfig:
    def __init__(self):
        # ================================================================
        # GEOMETRY PARAMETERS
        # ================================================================
        
        self.geometry = {
            
            # === X-RAY GEOMETRY ===
            'dso': 1000.0,           # Source-to-object distance in mm
                                    # Purpose: Distance from X-ray source to rotation center
                                    # Range: 400-1000 mm
                                    # Typical: 500-800mm (clinical CBCT)
                                    # Note: Affects magnification and cone angle
            
            'dde': 1000.0,           # Detector-to-object distance in mm
                                    # Purpose: Distance from rotation center to detector
                                    # Range: 200-800 mm
                                    # Typical: 300-600mm
                                    # Note: Total SDD = dso + dde
            
            # === DETECTOR PARAMETERS ===
            'nu_h': 512,           # Detector pixels (width/height for flat panel)
                                    # Purpose: Number of detector elements
                                    # Range: 256-2048
                                    # Typical: 512-1024 (square detector)
                                    # Note: More pixels = better resolution but slower
            
            'detector_size_mm': None,  # Detector physical size in mm (auto-calculated if None)
                                    # Purpose: Physical size of detector panel
                                    # Range: 200-500 mm
                                    # Defualt: None, the system will auto-calculate a value
                                    # Note: Must be large enough to avoid truncation
            
            'margin': 1.2,          # Safety margin for auto detector sizing
                                    # Purpose: Multiplier for minimum required detector size
                                    # Range: 1.0-1.5
                                    # Typical: 1.2 (20% margin)
            
            # === ANGULAR SAMPLING ===
            'n_proj': 360,          # Number of projection angles
                                    # Purpose: Angular sampling density
                                    # Range: 180-1200
                                    # Typical: 200-600 (clinical), 360+ (research)
                                    # Rule: ~2x object diameter in pixels for full sampling
            
            'start_angle': 0.0,     # Start angle in radians
            'end_angle': 2 * np.pi, # End angle in radians (2π for full rotation)
                                    # Purpose: Angular range of acquisition
                                    # Range: π to 2π (180° to 360°)
                                    # Typical: 2π (360°) for CBCT
            'pad_top': 0,
            'pad_bottom': 0,
        }
        
        # ================================================================
        # MATERIAL PROPERTIES
        # ================================================================
        
        self.materials = {
            # === REFERENCE MATERIALS ===
            'mu_water': 0.192,      # Water attenuation coefficient at reference energy
                                    # Purpose: Reference for HU conversion
                                    # Range: 0.15-0.25 (energy dependent)
                                    # Typical: 0.192 (80kVp equivalent)
            
            'mu_air': 0.0,          # Air attenuation coefficient
                                    # Purpose: Reference for HU conversion (fixed)
            
            # === TISSUE SEPARATION THRESHOLDS ===
            'T1': -200, # HU threshold for water/air separation (lower)
                                    # Purpose: Separate soft tissue from bone
                                    # Range: -300 to -100 HU
                                    # Typical: -200 HU
            
            'T2': 1500, # HU threshold for water/bone separation (upper)
                                    # Purpose: Separate soft tissue from bone  
                                    # Range: more than 1000 HU
            
            # === TISSUE SCALING FACTORS ===
            'water_level': 1.0,     # Soft tissue attenuation scaling
                                    # Purpose: Adjust soft tissue contrast
                                    # Range: 0.5-1.5
                                    # Typical: 1.0 (no scaling)
            
            'bone_level': 1.0,      # Bone attenuation scaling
                                    # Purpose: Adjust bone contrast
                                    # Range: 0.5-1.5  
                                    # Typical: 1.0 (no scaling)
        }
        
        # ================================================================
        # NOISE AND ARTIFACT PARAMETERS
        # ================================================================
        
        self.noise = {
            # === QUANTUM NOISE ===
            'lam': 0.02,             # lambda value in poisson noise
                                    # Purpose: Fundamental quantum noise level
                                    # Range: 0.0001 - 0.1
                                    # Note: Higher = more noise
            
            'add_quantum_noise': False,  # Enable Poisson quantum noise
            
            # === ELECTRONIC NOISE ===
            'electronic_noise_std': 0.1,  # Electronic noise standard deviation
                                    # Purpose: Detector readout noise
                                    # Range: 0.001-0.01
                                    # Typical: 0.005 (modern flat panels)
            
            'add_electronic_noise': False,
            
            # === DARK CURRENT ===
            'dark_current_level': 1e-3,  # Dark current offset level
                                    # Purpose: Temperature-dependent detector noise
                                    # Range: 1e-3 ~ 1
            
            'add_dark_current': False,
        }
        
        self.artifacts = {
            # === BEAM HARDENING ===
            'add_beam_hardening': False,
            'beam_hardening_strength': 1.0,  # Beam hardening severity multiplier
                                    # Purpose: Cupping and streak artifacts
                                    # Range: 0.5-2.0
                                    # Typical: 1.0 (realistic), 0.5 (mild), 1.5 (severe)
            
            # === CUPPING ARTIFACTS ===
            'add_cupping': False,
            'cupping_strength': 0.2,  # Cupping artifact intensity
                                    # Purpose: Center-dark, edge-bright artifacts
                                    # Range: 0.01-0.3
                                    # Mild: 0.01-0.05, Moderate: 0.05-0.15, Severe: 0.15+
            
            # === SCATTER RADIATION ===
            'add_scatter': False,
            'scatter_fraction': 0.1,  # Fraction of primary signal that is scatter
                                    # Purpose: Loss of contrast, background increase
                                    # Range: 0.01-1.0
                                    # Small FOV: 0.02-0.08, Large FOV: 0.1-0.25
                                    # Note: Higher for larger patients/FOV
            
            # === DETECTOR ARTIFACTS ===
            'add_ring_artifacts':False,
            'n_rings': 20,           # Number of ring artifacts
                                    # Purpose: Detector calibration errors
                                    # Range: 0-10
                                    # Typical: 2-5 (modern detectors)
            
            'ring_intensity': 0.5, # Ring artifact intensity
                                    # Range: 0.01-0.5
                                    # Mild: 0.01-0.03, Severe: 0.1+
            
            'add_lag_artifacts': False,
            'lag_factor': 0.1,     # Detector lag (ghosting) factor
                                    # Purpose: Previous frame influence
                                    # Range: 0.01-0.1
                                    # Typical: 0.02-0.05 (flat panels)
            
            'add_detector_artifacts': True,
            'bad_pixel_ratio': 0.001,  # Fraction of bad detector pixels
                                    # Range: 0.0001-0.01
                                    # Typical: 0.001 (0.1%)
            
            'gain_variation': 0.01, # Detector gain non-uniformity
                                    # Range: 0.01-0.1
                                    # Typical: 0.02 (2% variation)
            
            # === MOTION ARTIFACTS ===
            'add_motion': False,    # Enable patient motion simulation
            'motion_type': 'rigid', # Type: 'rigid', 'respiratory'
            'motion_strength': 1.0, # Motion amplitude in pixels
                                    # Range: 0.01-5.0
                                    # Mild: 0.5-1.0, Severe: 2.0+
            
            # === TRUNCATION ===
            'add_truncation': False,
            'truncation_ratio': 0.2,  # Fraction of FOV to truncate
                                    # Range: 0.01-0.3
                                    # Purpose: Object extends beyond FOV
            
            # === MECHANICAL ARTIFACTS ===
            'add_gantry_vibration': False,
            'vibration_frequency': 5.0,  # Vibration frequency in Hz, 1 to 5
            'vibration_amplitude': 3.0,  # Vibration amplitude in pixels 1 to 5

            # === PARTIAL VOLUME ===
            'add_partial_volume': False,
            'voxel_size': 0.01, # 0.01 to 0.75

            # === INCOMPLETE PROJECTIONS ===
            'add_incomplete_projections': False,

            # === TEMPERATURE DRIFT ===
            'add_temperature_drift': False,
            'drift_rate': 0.2, # 0.01 to 0.2 . If set to 1, HU will have large drift

            # === BOW TIE FILTER ===
            'add_bow_tie_filter': False,
            'miscalibration_strength': 0.8, # 0.1 to 0.5

            # === HEEL AFFECT ===
            'add_heel_effect': False,
            'anode_angle': 3.0, # 0.0 to 3.0
            'axis': 'down' # ['up', 'down', or 'vertical']
        }
        
        # ================================================================
        # METAL ARTIFACT PARAMETERS
        # ================================================================
        
        self.metal = {
            # === METAL SIMULATION CONTROL ===
            'add_metal_artifacts': False,
            'metal_type': 'mixed',  # Options: 'dental', 'orthopedic', 'surgical', 'mixed', 'random'
                                    # Purpose: Type of metal objects to simulate
            
            'num_metal_objects': 20,  # Number of objects
                                    # Range: 0-10
            
            # === METAL OBJECT PROPERTIES ===
            'min_radius': 10,        # Minimum metal object radius in voxels
                                    # Range: 1-20
                                    # Purpose: Smallest metal features
            
            'max_radius': 15,        # Maximum metal object radius in voxels
                                    # Range: 5-20
                                    # Purpose: Largest metal features
            
            'min_length': 10,        # Minimum metal object length in voxels
                                    # Range: 3-15
                                    # Purpose: Rod/screw length
            
            'max_length': 10,       # Maximum metal object length in voxels
                                    # Range: 10-50
                                    # Purpose: Long implants/rods
            
            'metal_mu': 3.0,       # Metal attenuation coefficient
                                    # Purpose: Metal absorption strength
                                    # Range: 2-20
            
            'intensity_variation': 0.05,  # Metal density variation
                                    # Range: 0.05-0.3
                                    # Purpose: Realistic heterogeneity
            
            # === CONTRAST AGENTS ===
            'add_contrast_artifacts': False,
            'contrast_concentration': 0.01,  # Contrast agent attenuation
                                    # Range: 0.001 - 0.01
        }


