rdf_process_and_compile.py ~ completed script

Takes from downloaded .cif files, makes rdf, puts all rdfs in a single master csv for the random forest ML algorithm

# RDF Batch Processor & Compiler

This script serves as a data processing pipeline for materials science machine learning projects. It automates the conversion of raw crystallographic data (`.cif` files) into a structured tabular dataset (CSV) by calculating the Radial Distribution Function (RDF) for each structure.

## Features
- **Automated Batch Processing:** Iterates through a manifest of materials to process files in bulk.
- **Physics-Informed Featurization:** Calculates the Radial Distribution Function, g(r), using the ASE library.
- **Supercell Generation:** Automatically expands unit cells to ensure accurate neighbor finding up to the cutoff radius.
- **Signal Smoothing:** Applies Gaussian filtering to smooth discrete peaks, making the data more robust for machine learning models.
- **Error Handling:** Skips corrupted files or missing paths without crashing the entire pipeline.

## Dependencies
Ensure you have Python 3.x installed. You will need the following libraries:

    pip install pandas ase scipy numpy

## Directory Structure
The script assumes a specific directory layout based on relative paths. Ensure your project looks similar to this:

project_root/
│
├── data_acquisition/
│   └── cif_files_v2/
│       ├── materials_manifest.csv  <-- Input Manifest
│       ├── material_1.cif
│       ├── material_2.cif
│       └── ...
│
├── master_csv/                     <-- Output Directory (Created automatically)
│   └── master_rdf_dataset.csv      <-- Final Output
│
└── rdf_process_and_compile.py      <-- This Script

## Configuration
Parameters are defined in the `__main__` block at the bottom of the script. You can adjust these to change the resolution or range of your RDF.

- `RMAX_CUTOFF` (Default: 6.0): The maximum distance (Angstroms) to calculate neighbors.
- `NBINS_COUNT` (Default: 200): The resolution of the RDF vector (number of features).
- `SIGMA_SMEARING` (Default: 4.0): The intensity of Gaussian smoothing applied to the signal.

## Usage

1. **Prepare the Manifest:** Ensure `materials_manifest.csv` exists in your CIF folder. It must contain at least:
   - `filename`: The exact name of the .cif file.
   - `label`: The target variable.

2. **Run the Script:**
   Execute the script from the command line:

       python rdf_process_and_compile.py

3. **Check Output:**
   The script will generate `master_csv/master_rdf_dataset.csv`.

## Output Format
The resulting CSV is ready for immediate use in Scikit-Learn or PyTorch.

- **structure_id:** The filename (minus extension).
- **g(r)_1 ... g(r)_200:** The calculated RDF values (features).
- **label:** The classification or regression target copied from the manifest.

## Methodology
1. **Supercell Expansion:** The script expands the input unit cell by a factor of `int(rmax)` in all three dimensions. This ensures that the simulation box is large enough to find neighbors up to the cutoff radius.
2. **RDF Calculation:** It computes the probability of finding a particle at distance r from a reference particle.
3. **Gaussian Smearing:** A Gaussian filter is applied to the discrete histogram. This turns sharp delta-like peaks (common in perfect crystals) into smooth continuous functions, which are easier for neural networks to learn.