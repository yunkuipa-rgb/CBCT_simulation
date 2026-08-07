import os
import numpy as np
import nibabel as nib
from datetime import datetime
from scipy import ndimage
from noise_simulation import create_cbct_config, cbct_artifact_simulation_3D
from cbct_geometry import imaging_geo_cone, initialization
from mask_gen import generate_3d_metal_mask
from config import CBCTConfig


class CBCTNiftiPipeline:
    """
    Complete pipeline for CBCT artifact simulation on a NIfTI volume
    """
    
    def __init__(self):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        geometry_params : dict, optional
            Parameters for ODL geometry setup
        """
        cfg = CBCTConfig()
        self.geometry_params = cfg.geometry
        self.params = cfg.metal | cfg.noise | cfg.artifacts | cfg.materials
        self.forward_op = None
        self.reconstruction_op = None
        
    def load_nifti(self, nifti_path):
        """
        Load a NIfTI volume

        Parameters:
        -----------
        nifti_path : str or Path
            Path to input NIfTI file (.nii or .nii.gz)

        Returns:
        --------
        volume : ndarray
            3D volume data in (z, y, x) order
        metadata : dict
            NIfTI metadata for reconstruction and saving
        """
        print(f"Loading NIfTI volume from: {nifti_path}")

        nii = nib.load(nifti_path)
        volume_xyz = nii.get_fdata(dtype=np.float32)

        if volume_xyz.ndim != 3:
            raise ValueError(f"Expected a 3D NIfTI volume, got shape {volume_xyz.shape}")

        # Convert from NIfTI convention (x, y, z) to pipeline convention (z, y, x)
        volume = np.transpose(volume_xyz, (2, 1, 0))

        zooms = nii.header.get_zooms()[:3]
        metadata = {
            'nifti_info': {
                'affine': nii.affine,
                'header': nii.header.copy(),
                'zooms': zooms,
                'dtype': str(volume_xyz.dtype),
            },
            'image_info': {
                'PixelSpacing': [float(zooms[0]), float(zooms[1])],
                'SliceThickness': float(zooms[2]),
                'ImageOrientationPatient': None,
                'ImagePositionPatient': None,
            }
        }

        print(f"Loaded volume shape: {volume.shape}")
        print(f"Volume range: [{np.min(volume):.1f}, {np.max(volume):.1f}]")

        return volume, metadata

    def setup_geometry(self, volume_shape, pixel_spacing=None):
        """
        Setup ODL geometry for CBCT simulation
        
        Parameters:
        -----------
        volume_shape : tuple
            Shape of the volume (depth, height, width)
        pixel_spacing : list, optional
            Pixel spacing [x, y, z] in mm
        """
        print("Setting up ODL geometry...")
        
        # Initialize geometry parameters
        param = initialization(
            n_proj=self.geometry_params['n_proj'],
        )
        
        # Adjust parameters based on actual volume
        if pixel_spacing is not None:
            param.param['reso_x'] = pixel_spacing[0]  # Use minimum spacing
            param.param['reso_y'] = pixel_spacing[1]  # Use minimum spacing
            param.param['reso_z'] = pixel_spacing[2]  # Use minimum spacing
        
        # Adjust volume parameters to match actual data
        pad_top = int(self.geometry_params['pad_top'])
        pad_bottom = int(self.geometry_params['pad_bottom'])
        depth, height, width = volume_shape
        param.param['nz_h'] = depth + pad_top + pad_bottom  # Account for padding in z dimension
        param.param['nx_h'] = width  # Note: ODL uses different axis convention
        param.param['ny_h'] = height
        
        # Recalculate physical dimensions
        param.param['sz'] = param.param['nz_h'] * param.param['reso_z']
        param.param['sx'] = param.param['nx_h'] * param.param['reso_x']
        param.param['sy'] = param.param['ny_h'] * param.param['reso_y']

        param.param['startangle'] = self.geometry_params['start_angle']
        param.param['endangle'] = self.geometry_params['end_angle']
        param.param['nProj'] = self.geometry_params['n_proj']

        # detector parameters
        param.param['nu_h'] = self.geometry_params['nu_h']
        param.param['dde'] = self.geometry_params['dde']  # detector-to-object distance
        param.param['dso'] = self.geometry_params['dso']  # source-to-object distance
        object_diagonal = np.sqrt(param.param['sx']**2 + param.param['sy']**2)
        magnification = (param.param['dso'] + param.param['dde']) / param.param['dso']
        if self.geometry_params['detector_size_mm']:
            param.param['su'] = self.geometry_params['detector_size_mm']
        else:
            param.param['su'] = object_diagonal * magnification * param.param['margin']  # margin
        
        # Create ODL operators
        self.forward_op, self.reconstruction_op = imaging_geo_cone(param)
        
        print(f"Geometry setup complete:")
        print(f"  Volume size: {param.param['sx']:.1f} x {param.param['sy']:.1f} x {param.param['sz']:.1f} mm")
        print(f"  Voxel size: {param.param['reso_x']:.3f} x {param.param['reso_y']:.3f} x {param.param['reso_z']:.3f} mm")
        print(f"  Projections: {param.param['nProj']}")
        
        return param
    
    def create_body_mask(self, volume):
        """
        Create body mask from volume
        
        Parameters:
        -----------
        volume : ndarray
            Input volume in HU
            
        Returns:
        --------
        body_mask : ndarray
            Binary body mask
        """
        print("Creating body mask...")
        
        # Threshold to separate air from tissue
        air_threshold = -500  # HU
        
        # Create initial mask
        body_mask = volume > air_threshold
        
        # Apply morphological operations to clean up the mask
        # Remove small holes
        for i in range(body_mask.shape[0]):
            body_mask[i] = ndimage.binary_fill_holes(body_mask[i])
        
        # Remove small objects
        body_mask = ndimage.binary_opening(body_mask, iterations=2)
        
        # Close remaining gaps
        body_mask = ndimage.binary_closing(body_mask, iterations=3)
        
        print(f"Body mask created: {np.sum(body_mask)} voxels ({100*np.sum(body_mask)/body_mask.size:.1f}%)")
        
        return body_mask.astype(np.float32)
    
    def apply_artifacts(self, volume, metadata):
        """
        Apply CBCT artifacts to the volume
        
        Parameters:
        -----------
        volume : ndarray
            Input volume in HU
        metadata : dict
            NIfTI metadata
            
        Returns:
        --------
        artifact_volume : ndarray
            Volume with artifacts applied
        metal_mask : ndarray
            Generated metal mask
        """
        print("Applying CBCT artifacts...")
        
        # Create body mask
        body_mask = self.create_body_mask(volume)
        
        # Generate metal mask
        metal_params = self.params.get('metal_params', {})
        metal_mask = generate_3d_metal_mask(
            image_shape=volume.shape,
            body_mask=body_mask,
            metal_type=self.params.get('metal_type', 'mixed'),
            num_objects=self.params.get('num_metal_objects', None),
            metal_params=metal_params
        )
        
        print(f"Generated metal mask: {np.sum(metal_mask)} voxels")

        # Setup CBCT simulation configuration
        cbct_config = create_cbct_config(
            forward_op=self.forward_op,
            reconstruction_op=self.reconstruction_op,
            cfg=self.params,
        )

        pad_top = int(self.geometry_params['pad_top'])
        pad_bottom = int(self.geometry_params['pad_bottom'])

        if pad_top < 0 or pad_bottom < 0:
            raise ValueError(f"pad_top and pad_bottom must be >= 0, got {pad_top}, {pad_bottom}")

        orig_shape = volume.shape
        pad_value = float(np.min(volume))

        if pad_top > 0 or pad_bottom > 0:
            volume = np.pad(
                volume,
                pad_width=((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
            metal_mask = np.pad(
                metal_mask,
                pad_width=((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        
        print(f"[cbct_artifact_simulation_3D] original shape: {orig_shape}, "
            f"pad_top={pad_top}, pad_bottom={pad_bottom}, padded shape: {volume.shape}")
        
        volume_odl = np.transpose(volume, (2, 1, 0))
        metal_mask_odl = np.transpose(metal_mask, (2, 1, 0))

        # Use transposed volumes for simulation
        artifact_volume_odl = cbct_artifact_simulation_3D(
            image=volume_odl,           # Changed from 'volume'
            metal_mask=metal_mask_odl,  # Changed from 'metal_mask'
            config=cbct_config,
            contrast_mask=None
        )

        # Convert result back to pipeline convention
        artifact_volume_np = np.asarray(artifact_volume_odl)
        artifact_volume = np.transpose(artifact_volume_np, (2, 1, 0))

        if pad_top > 0 or pad_bottom > 0:
            # Works for numpy arrays; if result_hu is an ODL element, np.asarray will still slice fine.
            artifact_volume = np.asarray(artifact_volume)
            z_end = artifact_volume.shape[0] - pad_bottom if pad_bottom > 0 else artifact_volume.shape[0]
            artifact_volume = artifact_volume[pad_top:z_end, :, :]
        
        print("Artifacts applied successfully")
        
        return artifact_volume, metal_mask
    
    def save_nifti(self, volume, output_path, metadata=None, description=""):
        """
        Save volume as NIfTI file
        
        Parameters:
        -----------
        volume : ndarray
            Volume to save
        output_path : str
            Output file path (.nii or .nii.gz)
        metadata : dict, optional
            DICOM metadata for affine matrix calculation
        description : str, optional
            Description for the NIfTI header
        """
        print(f"Saving NIfTI: {output_path}")
        
        # Reuse original NIfTI affine when available
        if metadata and 'nifti_info' in metadata and metadata['nifti_info'].get('affine') is not None:
            affine = np.array(metadata['nifti_info']['affine'], dtype=np.float64)
        else:
            pixel_spacing = metadata['image_info']['PixelSpacing'] if metadata else [1.0, 1.0]
            slice_thickness = metadata['image_info']['SliceThickness'] if metadata else 1.0
            affine = np.eye(4)
            affine[0, 0] = pixel_spacing[0]
            affine[1, 1] = pixel_spacing[1]
            affine[2, 2] = slice_thickness
        
        # Create NIfTI image
        # Note: NIfTI uses different axis convention than DICOM
        volume_nii = np.transpose(volume, (2, 1, 0))  # Convert from (z,y,x) to (x,y,z)
        
        nii_img = nib.Nifti1Image(volume_nii, affine)
        
        # Add description to header
        if description:
            nii_img.header['descrip'] = description.encode('utf-8')[:79]  # Max 80 characters
        
        # Save
        nib.save(nii_img, output_path)
        print(f"NIfTI saved: {output_path}")
    
    def process_nifti(self, input_nifti_path, output_folder):
        """
        Complete pipeline to process a NIfTI volume

        Parameters:
        -----------
        input_nifti_path : str
            Path to input NIfTI file
        output_folder : str
            Path to output folder

        Returns:
        --------
        dict : Processing results and paths
        """

        print("="*60)
        print("CBCT NIFTI PROCESSING PIPELINE")
        print("="*60)

        os.makedirs(output_folder, exist_ok=True)

        volume, metadata = self.load_nifti(input_nifti_path)

        pixel_spacing = list(metadata['image_info']['PixelSpacing']) + [metadata['image_info']['SliceThickness']]
        self.setup_geometry(volume.shape, pixel_spacing)

        artifact_volume, metal_mask = self.apply_artifacts(volume, metadata)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        artifact_nii_path = os.path.join(output_folder, f'artifact_{timestamp}.nii.gz')
        self.save_nifti(artifact_volume, artifact_nii_path, metadata, "CBCT with artifacts")

        results = {
            'input_nifti': input_nifti_path,
            'output_folder': output_folder,
            'artifact_nifti': artifact_nii_path,
            'volume_shape': volume.shape,
            'metal_voxels': int(np.sum(metal_mask)),
            'processing_timestamp': timestamp
        }

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Input: {input_nifti_path}")
        print(f"Output: {output_folder}")
        print(f"Volume shape: {volume.shape}")
        print(f"Metal voxels: {results['metal_voxels']}")
        print("Files saved:")
        print(f"  - Artifact NIfTI: {os.path.basename(artifact_nii_path)}")
        print("="*60)

        return results


if __name__ == "__main__":
    pipeline = CBCTNiftiPipeline()
    # Example:
    # pipeline.process_nifti("input.nii.gz", "./output")
