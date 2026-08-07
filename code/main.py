from pipeline_nifti_only import CBCTNiftiPipeline
from pipeline import CBCTDicomPipeline


# Create pipeline with custom geometry
pipeline = CBCTNiftiPipeline()
# pipeline = CBCTDicomPipeline()


# Process with custom settings
# results = pipeline.process_dicom_series(
#     input_folder="/mnt/whitsett/yunkuipa/cbct_project/data/1.000000-P4P100S300I00008 Gated 50.0A-57212",
#     output_folder="../result"
# )

results = pipeline.process_nifti(
    input_nifti_path="/mnt/whitsett/yunkuipa/cbct_project/pCT/0002.nii.gz",
    output_folder="../result"
)
