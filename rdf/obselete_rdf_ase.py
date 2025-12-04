import ase.io
from ase.geometry.analysis import get_rdf
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os


def calculate_and_save_rdf(cif_path, csv_path, rmax=6, nbins=100, sigma=1, label = "Unknown"):
    """
    Reads a .cif file, calculates its RDF, and saves the result to a CSV file.

    Args:
        cif_path (str): The file path for the input .cif file.
        csv_path (str): The file path for the output .csv file.
        rmax (float): Maximum distance (in Angstroms) for the RDF calculation.
        nbins (int): The number of bins to use for the RDF histogram.
        sigma (float): Standard deviation of the Gaussian kernel.\
        label (str): The label of the crystal structure.
    """
    try:
        # Read the crystal structure from the .cif file
        print(f"Reading structure from '{cif_path}'...")
        atoms = ase.io.read(cif_path)

        # We repeat the unit cell 10 times in the x, y, and z directions.
        print(f"Building a {rmax}x{rmax}x{rmax} supercell...")
        supercell = atoms * (int(rmax), int(rmax), int(rmax))

        # Calculate the Radial Distribution Function (RDF)
        print("Calculating RDF...")
        g_r, r = get_rdf(supercell, rmax=rmax, nbins=nbins)

        # --- Apply Gaussian smearing if sigma is greater than zero ---
        if sigma > 0:
            print(f"Applying Gaussian smearing with sigma = {sigma}...")
            g_r = gaussian_filter1d(g_r, sigma=sigma)

        # --- Write the results to a CSV file ---
        print(f"Writing RDF data and label to '{csv_path}'...")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the new header row
            writer.writerow(['r', 'g(r)', 'label'])

            #Write first row with label
            writer.writerow([r[0], g_r[0], label])

            # Write the rest of the data rows without the label
            for i in range(1, len(r)):
                writer.writerow([r[i], g_r[i], ''])  # Write an empty string for the label

        print(f"Successfully created {csv_path}")

    except FileNotFoundError:
        print(f"Error: The file '{cif_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_rdf_from_csv(csv_path, title="Radial Distribution Function"):
    """
    Reads RDF data from a CSV file and generates a plot.

    Args:
        csv_path (str): The file path for the input .csv file.
        title (str): The title for the plot.
    """
    try:
        print(f"Reading data from '{csv_path}' to create plot...")
        # Use pandas to easily read the CSV
        data = pd.read_csv(csv_path)

        # Get the columns for the x and y axes
        r = data['r']
        g_r = data['g(r)']

        # Create the plot
        plt.figure(figsize=(8, 5))
        plt.plot(r, g_r, label='g(r)')

        # Add labels and title
        plt.xlabel("Distance (r) [$\AA$]")
        plt.ylabel("g(r)")
        plt.title(title)

        # Add grid and legend for better readability
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: Plotting failed. The file '{csv_path}' was not found.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

# This block runs only when the script is executed directly
if __name__ == "__main__":

    # --- Configuration ---
    CIF_FOLDER = "../data_acquisition/cif_files_v2"
    # Assuming the manifest file you created is inside the test_cif_files folder
    MANIFEST_FILE = os.path.join(CIF_FOLDER, "materials_manifest.csv")
    OUTPUT_FOLDER = "./indiv_csv"  # Folder for the individual RDF csvs

    # RDF calculation parameters
    RMAX = 6.0
    NBINS = 200
    SIGMA = 4.0
    # ---------------------

    # 1. Ensure the output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. Read the manifest file
    try:
        manifest_df = pd.read_csv(MANIFEST_FILE)
    except FileNotFoundError:
        print(f"Error: Manifest file not found at '{MANIFEST_FILE}'")
        print("Please run the script to download CIFs first.")
        exit()  # Exit the script if no manifest
    except Exception as e:
        print(f"Error reading manifest file: {e}")
        exit()

    print(f"Found {len(manifest_df)} files to process in manifest.")

    # 3. Loop through each file in the manifest DataFrame
    for index, row in manifest_df.iterrows():
        cif_filename = row['filename']
        struct_type = row['label']

        print(f"\n--- Processing file {index + 1} of {len(manifest_df)}: {cif_filename} ---")

        # 4. Construct the full input and output file paths
        input_cif_file = os.path.join(CIF_FOLDER, cif_filename)

        # e.g., "mp-149.cif" -> "mp-149_rdf.csv"
        base_name = os.path.splitext(cif_filename)[0]
        output_csv_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_rdf.csv")

        # 5. Call the function to perform the calculation and save the file
        calculate_and_save_rdf(
            cif_path=input_cif_file,
            csv_path=output_csv_file,
            rmax=RMAX,
            nbins=NBINS,
            sigma=SIGMA,
            label=struct_type
        )

    print("\n--- Batch processing complete! ---")

    # --- Plot the RDF from the CSV file we just created ---
    #plot_rdf_from_csv(output_csv_file, title=f"RDF for {input_cif_file}")