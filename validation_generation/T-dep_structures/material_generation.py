# ---
# TITLE: material_generation.py
# PURPOSE: To generate a dataset of "imperfect" or "noisy" crystal
#          structures by running an MD simulation at increasing
#          temperatures.
# ---

import os
import time
from ase.io import Trajectory
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.units import kB, fs
from ase.calculators.lj import LennardJones

# --- 1. SCRIPT SETUP & GLOBAL PARAMETERS ---
print("--- Initializing Materials Generation Script ---")

# --- NEW --- Record the total script start time
script_start_time = time.perf_counter()

# --- 1a. File/Directory Setup ---
OUTPUT_DIR = "md_temp_ramp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory set to: {OUTPUT_DIR}")

# --- 1b. Simulation Physics Parameters ---
# System
SUPERCELL_SIZE = (6, 6, 6)  # Use a large supercell
START_TEMP_K = 300.0  # Temperature to initialize velocities
TEMP_RAMP_K = [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1350]

# Dynamics
TIMESTEP = 4.0 * fs
FRICTION = 0.02  # Langevin friction parameter (0.01-0.05 is typical)

# --- 1c. Simulation Step Parameters ---
# Use a much longer equilibration time to ensure system is heated
STEPS_EQUILIBRATE = 10000  # 10000 steps * 4.0 fs = 40.0 ps
STEPS_PRODUCTION = 5000  # 40.0 ps (We will save frames from this part) 5000
SAVE_INTERVAL = 50  # Save a "frame" every 50 steps (5000/50 = 100 frames) 50

print(f"Equilibration: {STEPS_EQUILIBRATE} steps ({STEPS_EQUILIBRATE * TIMESTEP / fs:.1f} fs)")
print(f"Production: {STEPS_PRODUCTION} steps ({STEPS_PRODUCTION * TIMESTEP / fs:.1f} fs)")
print("------------------------------------------------")

# --- 2. BUILD THE ATOMIC STRUCTURE ---

#Copper (Cu); MT ~ 1350K
element = "cu"
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
atoms = atoms.repeat(SUPERCELL_SIZE)
print(f"Built {SUPERCELL_SIZE} Cu supercell with {len(atoms)} atoms.")

"""
#Iron (Fe); MT ~ 1810
element = "fe"
atoms = bulk('Fe', 'bcc', a=2.87, cubic=True)
atoms = atoms.repeat(SUPERCELL_SIZE)
print(f"Built {SUPERCELL_SIZE} Fe supercell with {len(atoms)} atoms.")
"""
"""
element = "mg"
#Magnesium (Mg); MT ~920
# Note: We use orthorhombic=True instead of cubic=True for hexagonal systems
atoms = bulk('Mg', 'hcp', a=3.21, covera=1.624, orthorhombic=True)
atoms = atoms.repeat(SUPERCELL_SIZE)
print(f"Built {SUPERCELL_SIZE} Mg supercell with {len(atoms)} atoms.")
"""


# --- 3. ATTACH THE CALCULATOR ---
# We need to choose the calculator based on the element because
# EMT only works for specific FCC metals (Al, Cu, Ag, Au, Ni, Pd, Pt).

element_symbol = atoms.get_chemical_symbols()[0]
if element_symbol in ['Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt', 'Al']:
    print(f"Using EMT calculator for {element_symbol}...")
    atoms.calc = EMT()

elif element_symbol == 'Fe':
    # Lennard-Jones parameters approximated for Iron
    # sigma in Angstroms, epsilon in eV
    print(f"Using Lennard-Jones calculator for {element_symbol}...")
    atoms.calc = LennardJones(sigma=2.32, epsilon=0.4)

elif element_symbol == 'Mg':
    # Lennard-Jones parameters approximated for Magnesium
    print(f"Using Lennard-Jones calculator for {element_symbol}...")
    atoms.calc = LennardJones(sigma=2.80, epsilon=0.1)

else:
    # Fallback for other elements
    print(f"Warning: No specific parameters for {element_symbol}. Using generic LJ...")
    atoms.calc = LennardJones(sigma=2.5, epsilon=0.2)

# --- 4. INITIALIZE SIMULATION ---
# Set initial velocities to the starting temperature
MaxwellBoltzmannDistribution(atoms, temperature_K=START_TEMP_K * kB)
print(f"Set initial velocities to {START_TEMP_K} K.")

# --- 5. THE TEMPERATURE RAMP LOOP ---
# This is the main loop of the script.
# We will create a new Langevin object for each temperature step.
for target_temp_K in TEMP_RAMP_K:
    print(f"\n--- Starting T = {target_temp_K} K ---")

    # --- Record the start time for this specific loop ---
    loop_start_time = time.perf_counter()

    # --- Create a NEW Langevin object for this temperature ---
    # It automatically inherits the atoms' current positions/velocities
    dyn = Langevin(atoms,
                   timestep=TIMESTEP,
                   temperature=target_temp_K * kB,
                   friction=FRICTION)

    # Run the equilibration steps...
    print(f"  Equilibrating for {STEPS_EQUILIBRATE} steps ({STEPS_EQUILIBRATE * TIMESTEP / fs:.1f} fs)...")
    dyn.run(STEPS_EQUILIBRATE)

    # ...now run the production steps and save the trajectory
    print(f"  Running production for {STEPS_PRODUCTION} steps ({STEPS_PRODUCTION * TIMESTEP / fs:.1f} fs)...")

    # Define the trajectory file name for this temperature
    traj_filename = os.path.join(OUTPUT_DIR, f"{element_symbol}_T_{int(target_temp_K)}K.traj")

    # Create the Trajectory object
    traj = Trajectory(traj_filename, 'w', atoms)

    # Attach the ENTIRE traj object.
    # The 'interval' argument should be passed to attach()
    dyn.attach(traj, interval=SAVE_INTERVAL)

    # Run the production simulation
    dyn.run(STEPS_PRODUCTION)

    # --- NO DETACH NEEDED ---
    # The 'dyn' object will be thrown away at the end of this loop
    traj.close()

    print(f"  Done. Trajectory saved to: {traj_filename}")

    # --- Calculate and print the time for this loop ---
    loop_end_time = time.perf_counter()
    print(f"  > This temperature step took {loop_end_time - loop_start_time:.2f} seconds.")

print("\n--- All Simulations Finished ---")

# --- Calculate and print the total script time ---
script_end_time = time.perf_counter()
print(f"Total script runtime: {script_end_time - script_start_time:.2f} seconds")