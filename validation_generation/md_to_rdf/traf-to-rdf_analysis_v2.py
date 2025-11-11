# ---
# TITLE: Trajectory-to-RDF Analysis Pipeline
# PURPOSE: To process .traj files from an MD simulation into averaged
#          RDFs (g(r)) for input into an ML model.
# ---
import freud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from ase.io import read


# --- 1. ANALYSIS FUNCTION (Helper) ---

def get_averaged_rdf_from_trajectory(traj_file, r_max, bins):
    """
    Calculates a time-averaged RDF from an ASE trajectory file.
    Uses freud for high-performance, accumulating analysis.
    Returns:
        (r, g_r): A tuple of (bin_centers, g(r) values)
    """
    print(f"  Processing: {traj_file}")
    rdf_computer = freud.density.RDF(bins=bins, r_max=r_max)
    try:
        frames = read(traj_file, index=':')
    except Exception as e:
        print(f"    Could not read file {traj_file}. Error: {e}")
        return None, None
    if len(frames) == 0:
        print(f"    No frames found in {traj_file}.")
        return None, None
    for atoms in frames:
        system = (atoms.get_cell(), atoms.get_positions())
        rdf_computer.compute(system, reset=False)
    print(f"    Done. Processed {len(frames)} frames.")
    return rdf_computer.bin_centers, rdf_computer.rdf


# --- 2. PLOTTING FUNCTION (Helper) ---

def plot_rdf_overlay(rdf_results, output_dir, r_max_plot):
    """
    Generates and saves the vertically-offset RDF plot.
    """
    print("--- Plotting RDF Overlay ---")
    plt.figure(figsize=(10, 7))
    for temp_K, (r, g_r) in sorted(rdf_results.items()):
        offset = (temp_K / 300.0) * 0.5
        plt.plot(r, g_r + offset, label=f"{temp_K} K")
    plt.title("RDF of Copper vs. Temperature (Vertically Offset)")
    plt.xlabel("r (Angstrom)")
    plt.ylabel("g(r) (Offset)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, r_max_plot)
    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, "rdf_vs_temp_overlay.png")
    plt.savefig(plot_save_path)
    print(f"Plot saved to: {plot_save_path}")


# --- 3. MAIN PROCESSING FUNCTION ---

def process_trajectories_to_rdf(
        traj_dir,
        rdf_txt_dir,  # Kept for saving the plot
        csv_output_file,  # Master CSV
        individual_csv_dir,  # *** NEW: Directory for individual CSVs ***
        structure_type,
        structure_label,
        r_max,
        bins,
        do_plotting=True
):
    """
    Main function: finds trajectories, computes RDFs, saves individual CSVs,
    saves a master CSV, and (optionally) plots.
    """
    print(f"--- Starting RDF Analysis ---")
    print(f"Input Trajectories: {traj_dir}")
    print(f"Output Master CSV: {csv_output_file}")
    print(f"Output Individual CSVs: {individual_csv_dir}")
    print(f"Structure Type: {structure_type}, Label: {structure_label}")
    print("-----------------------------")

    all_rdf_results = {}
    csv_data_rows = []

    try:
        all_files = os.listdir(traj_dir)
        traj_files = [f for f in all_files if f.endswith('.traj')]
        traj_files.sort(key=lambda f: int(f.split('_')[-1].split('K')[0]))
    except FileNotFoundError:
        print(f"ERROR: Trajectory directory not found: {traj_dir}")
        return
    except Exception as e:
        print(f"ERROR: Could not read trajectory files. {e}")
        return

    if not traj_files:
        print(f"No .traj files found in {traj_dir}. Stopping.")
        return

    print(f"Found {len(traj_files)} trajectory files to process.")

    # --- *** NEW: Define column order once *** ---
    gr_columns = [f"g(r)_{i + 1}" for i in range(bins)]
    column_order = ["structure_id"] + gr_columns + ["label"]

    # --- MAIN LOOP ---
    for traj_file_name in traj_files:
        traj_file_path = os.path.join(traj_dir, traj_file_name)

        try:
            temp_K = int(traj_file_name.split('_')[-1].split('K')[0])
        except:
            print(f"  Skipping file (could not parse temp): {traj_file_name}")
            continue

        r, g_r = get_averaged_rdf_from_trajectory(traj_file_path, r_max, bins)

        if r is not None:
            # 1. Store for plotting
            all_rdf_results[temp_K] = (r, g_r)

            # 2. Prepare data row
            structure_id = f"{structure_type}_T_{temp_K}K"
            label = structure_label

            row = {"structure_id": structure_id}
            for i, gr_val in enumerate(g_r):
                row[f"g(r)_{i + 1}"] = gr_val
            row["label"] = label

            # 3. Append to master list
            csv_data_rows.append(row)

            # --- *** NEW: Save individual CSV file *** ---
            # Create a single-row DataFrame
            ind_df = pd.DataFrame([row])
            # Reorder columns to match master
            ind_df = ind_df[column_order]

            # Define and save the individual CSV
            output_csv_file = os.path.join(individual_csv_dir, f"{structure_id}.csv")
            ind_df.to_csv(output_csv_file, index=False)
            # --- End of new block ---

    print("\n--- RDF Analysis Complete ---")

    # --- SAVE MASTER CSV ---
    if csv_data_rows:
        print(f"Saving {len(csv_data_rows)} RDFs to master CSV...")
        df = pd.DataFrame(csv_data_rows)
        df = df[column_order]  # Use the pre-defined column order
        df.to_csv(csv_output_file, index=False)
        print(f"Successfully saved master CSV to: {csv_output_file}")
    else:
        print("No RDF data was generated, CSV file not saved.")

    # --- PLOTTING (Conditional) ---
    if do_plotting:
        if all_rdf_results:
            plot_rdf_overlay(all_rdf_results, rdf_txt_dir, r_max)
        else:
            print("Plotting skipped (no results to plot).")
    else:
        print("Plotting skipped (do_plotting=False).")

    print("\n--- Pipeline Finished ---")


# --- 4. SCRIPT EXECUTION ---

if __name__ == "__main__":
    # --- 4a. Define Global Parameters ---
    TRAJ_DIR = "../T-dep_structures/md_temp_ramp"
    RDF_TXT_DIR = "rdf_data"  # Used for saving plot
    CSV_DIR = "validation_rdfs"
    INDIVIDUAL_CSV_DIR = "individual_csvs"  # *** NEW: Folder for individual CSVs ***

    R_MAX = 6.0
    BINS = 200

    # --- 4b. Define NEW High-Level Controls ---
    STRUCTURE_TYPE = "cu"
    STRUCTURE_LABEL = "FCC"
    DO_PLOTTING = True

    # --- 4c. Setup Directories and Filenames ---
    os.makedirs(RDF_TXT_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(INDIVIDUAL_CSV_DIR, exist_ok=True)  # *** NEW ***

    CSV_OUTPUT_FILE = os.path.join(CSV_DIR, f"{STRUCTURE_TYPE}_rdf_dataset.csv")

    # --- 4d. Run the Main Function ---
    process_trajectories_to_rdf(
        traj_dir=TRAJ_DIR,
        rdf_txt_dir=RDF_TXT_DIR,
        csv_output_file=CSV_OUTPUT_FILE,
        individual_csv_dir=INDIVIDUAL_CSV_DIR,  # *** NEW ***
        structure_type=STRUCTURE_TYPE,
        structure_label=STRUCTURE_LABEL,
        r_max=R_MAX,
        bins=BINS,
        do_plotting=DO_PLOTTING
    )