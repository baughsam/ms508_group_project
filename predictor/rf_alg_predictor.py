import pandas as pd
import joblib
import os
import glob
import argparse
import warnings

# --- Import RDF calculation functions ---
import ase.io
from ase.geometry.analysis import get_rdf
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION (MUST MATCH TRAINING SCRIPT) ---
RMAX_CUTOFF = 6.0  # Max distance in Angstroms
NBINS_COUNT = 200  # Number of bins (this is your feature vector length)
SIGMA_SMEARING = 4.0  # Amount of Gaussian smearing


# --------------------------------------------------

def calculate_rdf_in_memory(cif_path, rmax, nbins, sigma):
    """
    Calculates the RDF for a single .cif file and returns the g(r) vector.
    (This is the same helper function from your process_and_compile.py script)
    """
    try:
        atoms = ase.io.read(cif_path)
        # Use the same supercell logic from your training
        supercell = atoms * (int(rmax), int(rmax), int(rmax))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g_r, r = get_rdf(supercell, rmax=rmax, nbins=nbins)

        if sigma > 0:
            g_r = gaussian_filter1d(g_r, sigma=sigma)
        return g_r
    except Exception as e:
        print(f"  Error processing {cif_path}: {e}")
        return None


def predict_from_csv(input_file, model, encoder):
    """
    Loads a pre-processed CSV and returns a DataFrame of predictions.
    """
    print(f"Loading pre-processed file: {input_file}")
    try:
        new_data = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return None

    # Get structure IDs for the report
    try:
        structure_ids = new_data['structure_id']
    except KeyError:
        print("Warning: No 'structure_id' column found. Using row index.")
        structure_ids = new_data.index

    # Extract feature columns based on model's training
    try:
        model_features = model.feature_names_in_
        features_to_predict = new_data[model_features]
    except KeyError:
        print("Error: The input CSV does not have the correct g(r) feature columns.")
        return None

    # Make predictions
    print("Making predictions...")
    predictions_encoded = model.predict(features_to_predict)
    prediction_labels = encoder.inverse_transform(predictions_encoded)

    return pd.DataFrame({
        'structure_id': structure_ids,
        'predicted_label': prediction_labels
    })


def predict_from_cif_folder(input_folder, model, encoder):
    """
    Finds all .cif files in a folder, processes them, and returns predictions.
    """
    print(f"Scanning for .cif files in folder: {input_folder}")
    cif_files = glob.glob(os.path.join(input_folder, "*.cif"))

    if not cif_files:
        print(f"Error: No .cif files found in '{input_folder}'.")
        return None

    print(f"Found {len(cif_files)} .cif files to process.")
    results = []

    for cif_path in cif_files:
        filename = os.path.basename(cif_path)
        print(f"  Processing: {filename}")

        # 1. Process the .cif file to get the g(r) vector
        g_r_vector = calculate_rdf_in_memory(
            cif_path=cif_path,
            rmax=RMAX_CUTOFF,
            nbins=NBINS_COUNT,
            sigma=SIGMA_SMEARING
        )

        if g_r_vector is not None:
            # 2. Format for prediction
            features = [g_r_vector]

            # 3. Predict
            prediction_encoded = model.predict(features)
            prediction_label = encoder.inverse_transform(prediction_encoded)[0]

            # 4. Store result
            results.append([filename, prediction_label])
        else:
            results.append([filename, "Error: Processing Failed"])

    return pd.DataFrame(results, columns=['filename', 'predicted_label'])


# --- Main Prediction Script ---
if __name__ == "__main__":

    # 1. Setup the command-line argument parser
    parser = argparse.ArgumentParser(description="Predict crystal structures from .csv or .cif files.")

    parser.add_argument(
        "mode",
        choices=['csv', 'cif'],
        help="The prediction mode: 'csv' for a pre-processed file, 'cif' for a folder of .cif files."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input file (if mode=csv) or input folder (if mode=cif)."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="predictions.csv",
        help="Name of the output CSV file for predictions."
    )

    args = parser.parse_args()

    # 2. Load the trained model and encoder
    print("Loading trained model and encoder...")
    try:
        model = joblib.load("rf_model.joblib")
        encoder = joblib.load("label_encoder.joblib")
    except FileNotFoundError:
        print("Error: 'rf_model.joblib' or 'label_encoder.joblib' not found.")
        print("Please make sure these files are in the same directory as the script.")
        exit()

    results_df = None

    # 3. Run the selected prediction mode
    if args.mode == 'csv':
        results_df = predict_from_csv(args.input, model, encoder)

    elif args.mode == 'cif':
        results_df = predict_from_cif_folder(args.input, model, encoder)

    # 4. Save results to output file
    if results_df is not None:
        results_df.to_csv(args.output, index=False)
        print("\n--- Prediction Results ---")
        print(results_df.to_string(index=False))
        print(f"\nResults saved to '{args.output}'")
    else:
        print("\nNo predictions were made.")