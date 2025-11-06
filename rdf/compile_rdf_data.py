import pandas as pd
import os


def compile_rdf_dataset(manifest_path, rdf_folder, output_csv):
    """
    Reads a manifest CSV, finds the corresponding RDF data for each entry,
    and compiles them into a single master CSV file in the ML-ready format.

    Args:
        manifest_path (str): Path to the materials_manifest.csv file.
        rdf_folder (str): Path to the folder containing individual RDF CSVs.
        output_csv (str): Path for the final master CSV file.
    """

    # 1. Read the manifest file
    try:
        manifest_df = pd.read_csv(manifest_path)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at '{manifest_path}'.")
        print("Please check the path in the MANIFEST_FILE variable.")
        return
    except Exception as e:
        print(f"Error reading manifest file: {e}")
        return

    print(f"Found {len(manifest_df)} entries in the manifest file to process...")

    all_data_rows = []
    num_features = 0  # To store the number of g(r) points

    # 2. Loop through each file listed in the manifest
    for index, row in manifest_df.iterrows():
        cif_filename = row['filename']
        label = row['label']

        # 3. Create the structure_id and the expected RDF filename
        # e.g., "mp-149.cif" -> "mp-149"
        struct_id = os.path.splitext(cif_filename)[0]
        # e.g., "mp-149" -> "mp-149_rdf.csv"
        rdf_filename = f"{struct_id}_rdf.csv"
        rdf_filepath = os.path.join(rdf_folder, rdf_filename)

        try:
            # 4. Read the corresponding individual RDF CSV file
            df = pd.read_csv(rdf_filepath)

            # 5. Extract the g(r) vector
            g_r_values = df['g(r)'].tolist()

            # 6. Check for data consistency
            if index == 0:
                num_features = len(g_r_values)
            elif len(g_r_values) != num_features:
                print(f"Warning: Skipping {rdf_filename}. Inconsistent number of g(r) points.")
                print(f"  Expected {num_features}, but found {len(g_r_values)}.")
                continue

            # 7. Build the new row for the master file
            # [struct_id, g(r)_values..., label]
            # e.g., ["mp-149", 0.0, 0.0, ..., 1.05, "BCC"]
            new_row = [struct_id] + g_r_values + [label]
            all_data_rows.append(new_row)

        except FileNotFoundError:
            print(f"Warning: Could not find RDF file '{rdf_filename}' for '{cif_filename}'. Skipping.")
        except Exception as e:
            print(f"Error processing {rdf_filename}: {e}. Skipping this file.")

    # 8. Check if any data was successfully processed
    if not all_data_rows:
        print("No data was successfully processed. Output file not created.")
        return

    # 9. Create the headers for the final DataFrame
    g_r_headers = [f"g(r)_{i + 1}" for i in range(num_features)]
    final_headers = ["structure_id"] + g_r_headers + ["label"]

    # 10. Create and save the final DataFrame
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    final_df = pd.DataFrame(all_data_rows, columns=final_headers)
    final_df.to_csv(output_csv, index=False)

    print(f"\nSuccessfully compiled {len(all_data_rows)} structures into '{output_csv}'.")


# --- This is the main part of the script that runs ---
if __name__ == "__main__":
    # --- Configuration ---
    # 1. Set this to the path of your manifest file
    # (The one created by pull_mat_proj.py)
    MANIFEST_FILE = "../data_acquisition/cif_files_v2/materials_manifest.csv"

    # 2. Set this to the folder containing your individual RDF CSVs
    # (The folder populated by rdf_ase.py)
    FOLDER_WITH_RDF_CSVS = "./indiv_csv"

    # 3. Set this to the name of the final output file
    MASTER_CSV_FILE = "master_csv/master_rdf_dataset_test1.csv"
    # ---------------------

    compile_rdf_dataset(MANIFEST_FILE, FOLDER_WITH_RDF_CSVS, MASTER_CSV_FILE)