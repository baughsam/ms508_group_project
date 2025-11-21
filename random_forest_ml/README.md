Basic random forest ML algorithm

Saved files:
- rf_model.joblib ~ model
- label_encoder.joblib ~ label encoder

The saved files are used to load our trained model into the /predictor/rf_alg_predictor.py

# RDF Batch Processor & Compiler

`rdf_process_and_compile.py`

This script serves as a data processing pipeline to convert raw crystallographic files (.cif) into machine-learning-ready feature vectors (RDFs).

## Features

- **Automated Batch Processing:** Iterates through directory structures to handle thousands of CIF files automatically.
- **Physics-Informed Featurization:** Calculates the Radial Distribution Function (RDF), capturing the geometric fingerprint of the material.
- **Supercell Generation:** Automatically expands unit cells to ensure the structure is large enough for the specified cutoff distance.
- **Signal Smoothing:** Applies Gaussian filtering/broadening to discrete peaks to simulate experimental data resolution.
- **Error Handling:** Skips corrupted files or entries with missing occupancy data without crashing the pipeline.

## Dependencies

Ensure you have Python 3.x installed. You will need the Atomic Simulation Environment (ASE) and standard data science libraries:

## Configuration
- Parameters are defined in the __main__ block at the bottom of the script:
- RMAX_CUTOFF (Default: 6.0): The maximum distance (in Angstroms) to calculate interactions.
- NBINS_COUNT (Default: 200): The resolution of the RDF (number of features per sample).
- SIGMA (Default: 4): The width of the Gaussian smearing applied to peaks.

## ASSUMED LAYOUT
project_root/
│
├── data_acquisition/
│   └── cif_files_v2/
│       ├── materials_manifest.csv   <-- Input Manifest (contains ID and Label)
│       ├── material_1.cif
│       ├── material_2.cif
│       └── ...
│
├── master_csv/                      <-- Output Directory
│   └── master_rdf_dataset.csv       <-- Final Output (Input for ML script)
│
└── rdf_process_and_compile.py       <-- This Script