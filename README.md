# CBCT Simulation

This project simulates cone-beam CT (CBCT) artifacts from a 3D medical image. It supports NIfTI volumes and DICOM series.

## Setup

Create and activate the provided Conda environment:

```bash
conda env create -f environment.yml
conda activate cbct
```

## Run

Open `code/main.py` and set the input and output paths. The default example runs the NIfTI pipeline:

```python
pipeline = CBCTNiftiPipeline()
results = pipeline.process_nifti(
    input_nifti_path="/path/to/input.nii.gz",
    output_folder="../result",
)
```

To process a DICOM series instead, use the commented `CBCTDicomPipeline` example in `code/main.py` and provide the folder containing the DICOM files.

Run the simulation from the `code` directory:

```bash
cd code
python main.py
```

Results are written to the output folder specified in `main.py`.

## Configuration

Simulation settings are defined in `code/config.py` in the `CBCTConfig` class. Edit the values in its configuration groups to change the scan geometry, material properties, noise, artifacts, or metal simulation. Boolean options such as `add_quantum_noise`, `add_motion`, and `add_metal_artifacts` enable or disable individual effects.

After saving `config.py`, run `python main.py` again to use the updated settings.
