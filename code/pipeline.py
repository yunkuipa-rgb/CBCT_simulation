import os
import numpy as np
import pydicom
import nibabel as nib
import glob
from datetime import datetime
import json
from scipy import ndimage
from noise_simulation import create_cbct_config, cbct_artifact_simulation_3D
from cbct_geometry import imaging_geo_cone, initialization
from mask_gen import generate_3d_metal_mask
from config import CBCTConfig


class CBCTDicomPipeline:
    """
    Complete pipeline for CBCT artifact simulation on DICOM series
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
        
    def load_dicom_series(self, dicom_folder_path):
        """
        Load DICOM series from folder
        
        Parameters:
        -----------
        dicom_folder_path : str or Path
            Path to folder containing DICOM files
            
        Returns:
        --------
        volume : ndarray
            3D volume data
        metadata : dict
            DICOM metadata for reconstruction
        """
        print(f"Loading DICOM series from: {dicom_folder_path}")
        
        # Find all DICOM files
        dicom_files = []
        for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
            dicom_files.extend(glob.glob(os.path.join(dicom_folder_path, ext)))
        
        if not dicom_files:
            # Try files without extension
            all_files = [f for f in os.listdir(dicom_folder_path) 
                        if os.path.isfile(os.path.join(dicom_folder_path, f))]
            
            # Test if files are DICOM
            dicom_files = []
            for file in all_files:
                try:
                    filepath = os.path.join(dicom_folder_path, file)
                    pydicom.dcmread(filepath, stop_before_pixels=True)
                    dicom_files.append(filepath)
                except:
                    continue
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {dicom_folder_path}")
        
        print(f"Found {len(dicom_files)} DICOM files")
        
        # Read first file to get metadata
        first_dicom = pydicom.dcmread(dicom_files[0])
        
        # Read all DICOM files and sort by slice location or instance number
        dicom_slices = []
        for file_path in dicom_files:
            try:
                ds = pydicom.dcmread(file_path)
                if ds.Modality == "CT":
                    dicom_slices.append(ds)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        # Sort + de-duplicate slices (handles duplicated instances per z)
        dicom_slices, dedup_info = self._dedup_dicom_slices(dicom_slices)
        if dedup_info.get('duplicate', 0) > 0:
            print(f"Warning: detected {dedup_info['duplicate']} duplicated slices by {dedup_info['key_type']}; keeping {dedup_info['after']} of {dedup_info['before']} instances")
        
        # Extract volume data
        volume = np.stack([ds.pixel_array for ds in dicom_slices], axis=0)
        
        # Convert to Hounsfield units if needed
        volume = self._convert_to_hu(volume, dicom_slices[0])
        
        # Extract metadata for later reconstruction
        metadata = self._extract_dicom_metadata(dicom_slices)
        
        print(f"Loaded volume shape: {volume.shape}")
        print(f"Volume range: [{np.min(volume):.1f}, {np.max(volume):.1f}] HU")
        
        return volume, metadata
    
    def _sort_dicom_slices(self, dicom_slices):
        """Sort DICOM slices by slice location or instance number"""
        
        # Try to sort by SliceLocation first
        if hasattr(dicom_slices[0], 'SliceLocation') and dicom_slices[0].SliceLocation is not None:
            dicom_slices.sort(key=lambda x: float(x.SliceLocation))
        # Fallback to InstanceNumber
        elif hasattr(dicom_slices[0], 'InstanceNumber') and dicom_slices[0].InstanceNumber is not None:
            dicom_slices.sort(key=lambda x: int(x.InstanceNumber))
        # Fallback to ImagePositionPatient Z coordinate
        elif hasattr(dicom_slices[0], 'ImagePositionPatient') and dicom_slices[0].ImagePositionPatient is not None:
            dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        else:
            print("Warning: Cannot determine slice order, using file order")
        
        return dicom_slices

    def _slice_position_key(self, ds, tol=1e-4):
        """Return a robust per-slice position key for de-duplication.

        Preference order:
          1) ImagePositionPatient z (most reliable for CT)
          2) SliceLocation
          3) InstanceNumber (weak; no geometric meaning)
        """
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp is not None and len(ipp) >= 3:
            try:
                z = float(ipp[2])
                return ('IPP_Z', round(z / tol))
            except Exception:
                pass

        sl = getattr(ds, 'SliceLocation', None)
        if sl is not None:
            try:
                z = float(sl)
                return ('SliceLocation', round(z / tol))
            except Exception:
                pass

        inst = getattr(ds, 'InstanceNumber', None)
        if inst is not None:
            try:
                return ('InstanceNumber', int(inst))
            except Exception:
                pass

        # Last resort: SOPInstanceUID (unique but won't dedup)
        return ('SOP', str(getattr(ds, 'SOPInstanceUID', 'NA')))

    def _dedup_dicom_slices(self, dicom_slices, tol=1e-4):
        """De-duplicate slices that share the same geometric position key.

        Keeps the first slice encountered for each key after sorting.
        Returns (deduped_slices, info_dict).
        """
        if not dicom_slices:
            return dicom_slices, {'before': 0, 'after': 0, 'duplicate': 0, 'key_type': None}

        # Sort first (stable order) then dedup
        dicom_slices = self._sort_dicom_slices(dicom_slices)

        key_counts = {}
        keep = []
        seen = set()
        key_type = None

        for ds in dicom_slices:
            k = self._slice_position_key(ds, tol=tol)
            if key_type is None:
                key_type = k[0]
            key_counts[k] = key_counts.get(k, 0) + 1
            if k in seen:
                continue
            seen.add(k)
            keep.append(ds)

        duplicate = len(dicom_slices) - len(keep)
        info = {
            'before': len(dicom_slices),
            'after': len(keep),
            'duplicate': duplicate,
            'key_type': key_type,
        }
        return keep, info

    def _write_dicom_manifest(self, output_folder, manifest):
        """Write a small JSON manifest alongside the DICOM output for debugging."""
        try:
            manifest_path = os.path.join(output_folder, 'dicom_manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: could not write manifest: {e}")
    
    def _convert_to_hu(self, volume, reference_dicom):
        """Convert pixel values to Hounsfield Units"""
        
        # Get rescale parameters
        rescale_intercept = getattr(reference_dicom, 'RescaleIntercept', 0)
        rescale_slope = getattr(reference_dicom, 'RescaleSlope', 1)
        
        # Convert to float and apply rescaling
        volume = volume.astype(np.float32)
        volume = volume * rescale_slope + rescale_intercept
        
        return volume
    
    def _extract_dicom_metadata(self, dicom_slices):
        """Extract important DICOM metadata for reconstruction"""
        
        first_slice = dicom_slices[0]
        
        metadata = {
            'dicom_slices': dicom_slices,  # Keep original DICOM objects
            'patient_info': {
                'PatientID': getattr(first_slice, 'PatientID', ''),
                'PatientName': str(getattr(first_slice, 'PatientName', '')),
                'PatientBirthDate': getattr(first_slice, 'PatientBirthDate', ''),
                'PatientSex': getattr(first_slice, 'PatientSex', ''),
            },
            'study_info': {
                'StudyInstanceUID': getattr(first_slice, 'StudyInstanceUID', ''),
                'StudyDate': getattr(first_slice, 'StudyDate', ''),
                'StudyTime': getattr(first_slice, 'StudyTime', ''),
                'StudyDescription': getattr(first_slice, 'StudyDescription', ''),
                'AccessionNumber': getattr(first_slice, 'AccessionNumber', ''),
            },
            'series_info': {
                'SeriesInstanceUID': getattr(first_slice, 'SeriesInstanceUID', ''),
                'SeriesNumber': getattr(first_slice, 'SeriesNumber', ''),
                'SeriesDescription': getattr(first_slice, 'SeriesDescription', ''),
                'Modality': getattr(first_slice, 'Modality', 'CT'),
            },
            'image_info': {
                'PixelSpacing': getattr(first_slice, 'PixelSpacing', [1.0, 1.0]),
                'SliceThickness': getattr(first_slice, 'SliceThickness', 1.0),
                'ImageOrientationPatient': getattr(first_slice, 'ImageOrientationPatient', None),
                'ImagePositionPatient': getattr(first_slice, 'ImagePositionPatient', None),
                'RescaleIntercept': getattr(first_slice, 'RescaleIntercept', 0),
                'RescaleSlope': getattr(first_slice, 'RescaleSlope', 1),
                'WindowCenter': getattr(first_slice, 'WindowCenter', None),
                'WindowWidth': getattr(first_slice, 'WindowWidth', None),
            },
            'acquisition_info': {
                'KVP': getattr(first_slice, 'KVP', None),
                'ExposureTime': getattr(first_slice, 'ExposureTime', None),
                'XRayTubeCurrent': getattr(first_slice, 'XRayTubeCurrent', None),
                'FilterType': getattr(first_slice, 'FilterType', None),
                'ConvolutionKernel': getattr(first_slice, 'ConvolutionKernel', None),
            }
        }
        
        return metadata
    
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
        air_threshold = -500 # self.params['T1']  # HU
        
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
            DICOM metadata
            
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

        # Convert result back to DICOM convention
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
        
        # Create affine matrix from DICOM metadata
        if metadata and metadata['image_info']['ImagePositionPatient'] is not None:
            affine = self._create_affine_from_dicom(metadata)
        else:
            # Default affine matrix
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
    
    def _create_affine_from_dicom(self, metadata):
        """Create affine transformation matrix from DICOM metadata"""
        
        image_info = metadata['image_info']
        
        # Get orientation and position
        orientation = np.array(image_info['ImageOrientationPatient']).reshape(2, 3)
        position = np.array(image_info['ImagePositionPatient'])
        pixel_spacing = image_info['PixelSpacing']
        slice_thickness = image_info['SliceThickness']
        
        # Calculate the third direction (cross product)
        direction_z = np.cross(orientation[0], orientation[1])
        
        # Create rotation matrix
        rotation = np.column_stack([orientation[0], orientation[1], direction_z])
        
        # Create scaling matrix
        scaling = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness])
        
        # Combine rotation and scaling
        rotation_scaling = rotation @ scaling
        
        # Create full affine matrix
        affine = np.eye(4)
        affine[:3, :3] = rotation_scaling
        affine[:3, 3] = position
        
        return affine
    
    def save_dicom_series(self, volume, output_folder, metadata, description="CBCT Artifact Simulation"):
        """
        Save volume as DICOM series while preserving metadata
        
        Parameters:
        -----------
        volume : ndarray
            Volume to save
        output_folder : str
            Output folder path
        metadata : dict
            Original DICOM metadata
        description : str, optional
            Series description
        """
        print(f"Saving DICOM series: {output_folder}")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Get original DICOM slices
        original_slices = metadata['dicom_slices']
        
        # De-duplicate original slices by geometry (robust against doubled series)
        original_slices, orig_dedup_info = self._dedup_dicom_slices(list(original_slices))
        if orig_dedup_info.get('duplicate', 0) > 0:
            print(f"Warning: original metadata contained {orig_dedup_info['duplicate']} duplicated slices by {orig_dedup_info['key_type']}; using {orig_dedup_info['after']} unique slices")
        
        # Convert volume back to original data type range
        volume_scaled = self._scale_volume_for_dicom(volume, metadata)
        
        # Generate new series UID
        new_series_uid = pydicom.uid.generate_uid()
        current_time = datetime.now()
        
        # Save each slice
        for i in range(volume.shape[0]):
            # Copy original DICOM
            if i < len(original_slices):
                new_slice = original_slices[i].copy()
            else:
                # If we have more slices than original, copy the last one
                new_slice = original_slices[-1].copy()
            if new_slice.Modality != "CT":
                continue
            
            # Update pixel data
            target_dtype = new_slice.pixel_array.dtype

            # Scale and convert the data
            new_pixel_data = volume_scaled[i].astype(target_dtype)

            # Update the PixelData attribute directly
            new_slice.PixelData = new_pixel_data.tobytes()

            # Also update related DICOM tags if they exist
            if hasattr(new_slice, 'Rows'):
                new_slice.Rows = new_pixel_data.shape[0]
            if hasattr(new_slice, 'Columns'):
                new_slice.Columns = new_pixel_data.shape[1]
            
            # Update metadata
            new_slice.SeriesInstanceUID = new_series_uid
            new_slice.SOPInstanceUID = pydicom.uid.generate_uid()
            new_slice.SeriesDescription = description
            new_slice.SeriesNumber = str(int(getattr(new_slice, 'SeriesNumber', 1)) + 1000)
            new_slice.InstanceNumber = str(i + 1)
            
            # Update timestamps
            new_slice.SeriesDate = current_time.strftime('%Y%m%d')
            new_slice.SeriesTime = current_time.strftime('%H%M%S.%f')[:-3]
            new_slice.AcquisitionDate = current_time.strftime('%Y%m%d')
            new_slice.AcquisitionTime = current_time.strftime('%H%M%S.%f')[:-3]
            new_slice.ContentDate = current_time.strftime('%Y%m%d')
            new_slice.ContentTime = current_time.strftime('%H%M%S.%f')[:-3]
            
            # Update slice location if available
            if hasattr(new_slice, 'SliceLocation') and new_slice.SliceLocation is not None:
                if i < len(original_slices):
                    # Keep original slice location
                    pass
                else:
                    # Extrapolate for additional slices
                    if len(original_slices) >= 2:
                        slice_spacing = (float(original_slices[-1].SliceLocation) - 
                                       float(original_slices[-2].SliceLocation))
                        new_slice.SliceLocation = (float(original_slices[-1].SliceLocation) + 
                                                 slice_spacing * (i - len(original_slices) + 1))
            
            # Save slice
            output_filename = f"slice_{i+1:04d}.dcm"
            output_path = os.path.join(output_folder, output_filename)
            pydicom.dcmwrite(output_path, new_slice, write_like_original=False)
        
        # Write a lightweight manifest for debugging / provenance
        try:
            import platform as _platform
            import sys as _sys
            manifest = {
                'output_folder': os.path.abspath(output_folder),
                'written_instances': int(volume.shape[0]),
                'written_shape': [int(x) for x in volume.shape],
                'original_unique_slices_used': int(len(original_slices)),
                'original_total_slices_in_metadata': int(orig_dedup_info.get('before', len(metadata.get('dicom_slices', [])))),
                'original_duplicates_detected': int(orig_dedup_info.get('duplicate', 0)),
                'dedup_key_type': orig_dedup_info.get('key_type', None),
                'series_instance_uid': str(new_series_uid),
                'description': str(description),
                'timestamp_local': current_time.isoformat(),
                'environment': {
                    'python': _sys.version,
                    'platform': _platform.platform(),
                    'numpy': getattr(np, '__version__', None),
                    'pydicom': getattr(pydicom, '__version__', None),
                },
            }
            self._write_dicom_manifest(output_folder, manifest)
        except Exception as e:
            print(f"Warning: failed to build/write manifest: {e}")

        print(f"DICOM series saved: {len(volume)} slices in {output_folder}")
    
    def _scale_volume_for_dicom(self, volume, metadata):
        """Scale volume back to original DICOM pixel value range"""
        
        rescale_intercept = metadata['image_info']['RescaleIntercept']
        rescale_slope = metadata['image_info']['RescaleSlope']
        
        # Convert from HU back to pixel values
        volume_scaled = (volume - rescale_intercept) / rescale_slope
        
        # Ensure values are in reasonable range
        volume_scaled = np.clip(volume_scaled, -32768, 32767)
        
        return volume_scaled
    
    def process_dicom_series(self, input_folder, output_folder):
        """
        Complete pipeline to process DICOM series
        
        Parameters:
        -----------
        input_folder : str
            Path to input DICOM series folder
        output_folder : str
            Path to output folder
            
        Returns:
        --------
        dict : Processing results and paths
        """
        
        print("="*60)
        print("CBCT DICOM PROCESSING PIPELINE")
        print("="*60)
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Load DICOM series
        volume, metadata = self.load_dicom_series(input_folder)

        volume = volume.clip(-1024, 3000)
        
        # Setup geometry
        pixel_spacing = list(metadata['image_info']['PixelSpacing']) + [metadata['image_info']['SliceThickness']]
        self.setup_geometry(volume.shape, pixel_spacing)
        
        # Apply artifacts
        artifact_volume, metal_mask = self.apply_artifacts(volume, metadata)
        
        # Create output paths
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save original volume as NIfTI
        # original_nii_path = os.path.join(output_folder, f'original_{timestamp}.nii.gz')
        # self.save_nifti(volume, original_nii_path, metadata, "Original CBCT volume")
        
        # Save artifact volume as NIfTI
        artifact_nii_path = os.path.join(output_folder, f'artifact_{timestamp}.nii.gz')
        self.save_nifti(artifact_volume, artifact_nii_path, metadata, "CBCT with artifacts")
        
        # Save metal mask as NIfTI
        # metal_nii_path = os.path.join(output_folder, f'metal_mask_{timestamp}.nii.gz')
        # self.save_nifti(metal_mask, metal_nii_path, metadata, "Metal mask")
        
        # Save artifact volume as DICOM series
        dicom_output_folder = os.path.join(output_folder, f'dicom_artifact_{timestamp}')
        self.save_dicom_series(artifact_volume, dicom_output_folder, metadata, 
                             f"CBCT Artifact Simulation {timestamp}")
        
        # # Save processing log
        # self.save_processing_log(output_folder, artifact_config, metadata)
        
        # Results summary
        results = {
            'input_folder': input_folder,
            'output_folder': output_folder,
            # 'original_nifti': original_nii_path,
            # 'artifact_nifti': artifact_nii_path,
            # 'metal_mask_nifti': metal_nii_path,
            'dicom_series_folder': dicom_output_folder,
            'volume_shape': volume.shape,
            'metal_voxels': int(np.sum(metal_mask)),
            'processing_timestamp': timestamp
        }
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        print(f"Volume shape: {volume.shape}")
        print(f"Metal voxels: {results['metal_voxels']}")
        print(f"Files saved:")
        # print(f"  - Original NIfTI: {os.path.basename(original_nii_path)}")
        # print(f"  - Artifact NIfTI: {os.path.basename(artifact_nii_path)}")
        # print(f"  - Metal mask NIfTI: {os.path.basename(metal_nii_path)}")
        print(f"  - DICOM series: {os.path.basename(dicom_output_folder)}/")
        print("="*60)
        
        return results
