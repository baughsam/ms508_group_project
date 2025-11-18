import os
import csv
from mp_api.client import MPRester

# --- CONFIGURATION ---

# 1. WARNING: Replace this with your new API key.
# Do NOT share this key publicly.
API_KEY = "fstLOV8SAhrIvDQ7sOIyDFNCe8DVP1pa"

# 2. Define the types of structures you want to query.
# The spacegroup numbers are correct for common structures:
# 225 = FCC (Fm-3m), 229 = BCC (Im-3m), 194 = HCP (P6_3/mmc)
STRUCTURE_TYPES = {
    "FCC": {"spacegroup_number": 225},
    "BCC": {"spacegroup_number": 229},
    "HCP": {"spacegroup_number": 194}
}

# 3. Set how many examples you want for each structure type
MAX_EXAMPLES_PER_TYPE = 10

# 4. Define your output directory and manifest file name
OUTPUT_DIR = "cif_files"
MANIFEST_FILE = "materials_manifest.csv"


# --- END CONFIGURATION ---


def pull_materials_data():
    """
    Connects to the Materials Project, downloads structures, saves them
    as .cif files, and creates a manifest CSV with their labels.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory set to: {OUTPUT_DIR}")

    # This list will hold our manifest data (e.g., ['mp-149.cif', 'BCC'])
    manifest_data = []

    try:
        # Connects to MP database with api key
        with MPRester(api_key=API_KEY) as mpr:
            for label, criteria in STRUCTURE_TYPES.items():
                print(f"\nQuerying for {label} structures (Spacegroup {criteria['spacegroup_number']})...")

                # Search for materials matching the criteria
                docs = mpr.materials.summary.search(
                    **criteria,
                    fields=["material_id", "structure"]
                )

                if not docs:
                    print(f"No materials found for {label}.")
                    continue

                print(f"Found {len(docs)} total materials for {label}. Downloading up to {MAX_EXAMPLES_PER_TYPE}...")

                saved_count = 0
                for doc in docs:
                    # Stop if we have enough examples
                    if saved_count >= MAX_EXAMPLES_PER_TYPE:
                        break

                    material_id = doc.material_id
                    structure = doc.structure

                    # Define the output filename and full path
                    cif_filename = f"{material_id}.cif"
                    cif_filepath = os.path.join(OUTPUT_DIR, cif_filename)

                    try:
                        # --- This is the key step you were missing ---
                        # Save the Pymatgen structure object as a .cif file
                        structure.to(filename=cif_filepath, fmt="cif")

                        # Add the file and its label to our manifest list
                        manifest_data.append([cif_filename, label])
                        saved_count += 1

                    except Exception as e:
                        print(f"  Error saving {material_id}: {e}")

                print(f"Successfully saved {saved_count} {label} structures.")

    except Exception as e:
        print(f"An error occurred connecting to Materials Project: {e}")
        print("Please check your API key and internet connection.")
        return

    # --- Write the final manifest file ---
    if not manifest_data:
        print("\nNo materials were downloaded. Manifest file not created.")
        return

    manifest_filepath = os.path.join(OUTPUT_DIR, MANIFEST_FILE)
    print(f"\nWriting manifest file to: {manifest_filepath}")

    # Check if the file already exists so we know if we need to write headers
    file_exists = os.path.isfile(manifest_filepath)

    # Open in 'a' (append) mode if it exists, 'w' (write) if it doesn't
    mode = 'a' if file_exists else 'w'

    print(f"\n{mode.capitalize()}ing to manifest file: {manifest_filepath}")

    try:
        with open(manifest_filepath, mode, newline='') as f:
            writer = csv.writer(f)

            # Only write the header row if it's a NEW file
            if not file_exists:
                writer.writerow(["filename", "label"])

            # Write the new data rows
            writer.writerows(manifest_data)

    except Exception as e:
        print(f"Error writing manifest file: {e}")

    print(f"\nScript finished. Total materials saved: {len(manifest_data)}")


# --- Run the main function ---
if __name__ == "__main__":
    pull_materials_data()