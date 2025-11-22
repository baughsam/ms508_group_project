rdf_process_and_compile.py ~ completed script

Takes from downloaded .cif files, makes rdf, puts all rdfs in a single master csv for the random forest ML algorithm

The discrete histogram. This turns sharp delta-like peaks (common in perfect crystals) into smooth continuous functions, which are easier for neural networks to learn.

# Random Forest Classifier for RDF Data

This script trains a Random Forest machine learning model to classify crystal structures based on Radial Distribution Function (RDF) data.

## Requirements
* Python 3.x
* Libraries: `pandas`, `scikit-learn`, `joblib`

## Inputs & Setup
The script uses a relative path to find the data. Ensure your file structure is organized as follows:
* **Input Data**: `../rdf/master_csv/master_rdf_dataset.csv`
* **Script Location**: `random_forest_ml.py` (Run the script from this folder)

## Outputs
* Console: Displays the training status, model accuracy (%), and a classification report (Precision/Recall).
* Saved Files: Generates two files in the current directory for future use:
* rf_model.joblib: The trained Random Forest model.
* label_encoder.joblib: The encoder used to convert text labels (e.g., "BCC") to numbers.