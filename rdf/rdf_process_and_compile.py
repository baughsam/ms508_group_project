import os
import pandas as pd
import ase.io
from ase.geometry.analysis import get_rdf  # Using your script's import
from scipy.ndimage import gaussian_filter1d
import warnings


def calculate_rdf_in_memory(cif_path, rmax, nbins, sigma):
    """
    Calculates the RDF for a single .cif file and returns the g(r) vector.

    This logic is copied directly from your rdf_ase.py script.
    """
    try:
        # 1. Read the crystal structure
        atoms = ase.io.read(cif_path)

        # 2. Build supercell (using the exact logic from your file)
        # Note: This uses rmax as the integer multiplier
        print(f"  Building a {int(rmax)}x{int(rmax)}x{int(rmax)} supercell...")
        supercell = atoms * (int(rmax), int(rmax), int(rmax))

        # 3. Calculate the RDF
        print("  Calculating RDF...")
        # Suppress ASE warnings about cell size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g_r, r = get_rdf(supercell, rmax=rmax, nbins=nbins)

        # 4. Apply Gaussian smearing
        if sigma > 0:
            print(f"  Applying Gaussian smearing with sigma = {sigma}...")
            g_r = gaussian_filter1d(g_r, sigma=sigma)

        return g_r

    except FileNotFoundError:
        print(f"  Warning: File not found at '{cif_path}'. Skipping.")
        return None
    except Exception as e:
        print(f"  Warning: Error processing '{cif_path}': {e}. Skipping.")
        return None


def process_and_compile(manifest_path, cif_folder, output_csv, rmax, nbins, sigma):
    """
    Reads a manifest, calculates RDF for each file, and saves ONE master CSV.
    """

    # 1. Read the manifest file
    try:
        manifest_df = pd.read_csv(manifest_path)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at '{manifest_path}'")
        print("Please check your file paths and run data acquisition first.")
        return
    except Exception as e:
        print(f"Error reading manifest file: {e}")
        return

    print(f"Manifest loaded. Found {len(manifest_df)} structures to process.")

    all_data_rows = []

    # 2. Loop through each file listed in the manifest
    for index, row in manifest_df.iterrows():
        cif_filename = row['filename']
        label = row['label']
        struct_id = os.path.splitext(cif_filename)[0]  # e.g., "mp-149"

        print(f"\nProcessing {index + 1}/{len(manifest_df)}: {struct_id} ({label})")

        # 3. Construct the full .cif file path
        cif_filepath = os.path.join(cif_folder, cif_filename)

        # 4. Calculate the RDF in memory
        g_r_values = calculate_rdf_in_memory(
            cif_path=cif_filepath,
            rmax=rmax,
            nbins=nbins,
            sigma=sigma
        )

        # 5. If calculation was successful, add it to our list
        if g_r_values is not None:
            # Check for data consistency (all vectors must be the same length)
            if all_data_rows and len(g_r_values) != len(all_data_rows[0]) - 2:  # -2 for id/label
                print(f"  Error: Inconsistent data length for {struct_id}. Skipping.")
                continue

            # Build the new row: [struct_id, g(r)_1, g(r)_2, ..., label]
            new_row = [struct_id] + list(g_r_values) + [label]
            all_data_rows.append(new_row)

    # 6. Check if any data was successfully processed
    if not all_data_rows:
        print("\nNo data was successfully processed. Output file not created.")
        return

    # 7. Create headers for the final DataFrame
    num_features = len(all_data_rows[0]) - 2  # e.g., 200
    g_r_headers = [f"g(r)_{i + 1}" for i in range(num_features)]
    final_headers = ["structure_id"] + g_r_headers + ["label"]

    # 8. Create and save the final DataFrame
    print(f"\nAll processing complete. Saving {len(all_data_rows)} structures to '{output_csv}'...")
    final_df = pd.DataFrame(all_data_rows, columns=final_headers)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir:  # Handle cases where the output is in the root dir
        os.makedirs(output_dir, exist_ok=True)

    final_df.to_csv(output_csv, index=False)

    print("Master dataset successfully created!")


# ===================================================================
# --- MAIN BLOCK ---
# ===================================================================
if __name__ == "__main__":
    # --- Configuration (Copied from your two scripts) ---

    # Paths from rdf_ase.py and compile_rdf_data.py
    CIF_FOLDER = "../data_acquisition/cif_files_v2"
    MANIFEST_FILE = os.path.join(CIF_FOLDER, "materials_manifest.csv")
    MASTER_CSV_FILE = "master_csv/master_rdf_dataset.csv"

    # RDF Calculation Parameters from rdf_ase.py
    RMAX_CUTOFF = 6.0  # Max distance in Angstroms
    NBINS_COUNT = 200  # Number of bins (this is your feature vector length)
    SIGMA_SMEARING = 4.0  # Amount of Gaussian smearing
    # ---------------------------------------------------

    # Run the main processing function
    process_and_compile(
        manifest_path=MANIFEST_FILE,
        cif_folder=CIF_FOLDER,
        output_csv=MASTER_CSV_FILE,
        rmax=RMAX_CUTOFF,
        nbins=NBINS_COUNT,
        sigma=SIGMA_SMEARING
    )