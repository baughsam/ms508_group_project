# ---
# TITLE: material_generation.py
# PURPOSE: To generate a dataset of "imperfect" or "noisy" crystal
#          structures by running an MD simulation at increasing
#          temperatures.
# ---

import os
import time  # --- NEW --- Import the time library
from ase.io import Trajectory
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.units import kB, fs

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
STEPS_PRODUCTION = 5000  # 40.0 ps (We will save frames from this part)
SAVE_INTERVAL = 50  # Save a "frame" every 50 steps (5000/50 = 100 frames)

print(f"Equilibration: {STEPS_EQUILIBRATE} steps ({STEPS_EQUILIBRATE * TIMESTEP / fs:.1f} fs)")
print(f"Production: {STEPS_PRODUCTION} steps ({STEPS_PRODUCTION * TIMESTEP / fs:.1f} fs)")
print("------------------------------------------------")

# --- 2. BUILD THE ATOMIC STRUCTURE ---
atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
atoms = atoms.repeat(SUPERCELL_SIZE)
print(f"Built {SUPERCELL_SIZE} Cu supercell with {len(atoms)} atoms.")

# --- 3. ATTACH THE CALCULATOR ---
atoms.calc = EMT()

# --- 4. INITIALIZE SIMULATION ---
# Set initial velocities to the starting temperature
MaxwellBoltzmannDistribution(atoms, temperature_K=START_TEMP_K * kB)
print(f"Set initial velocities to {START_TEMP_K} K.")

# --- 5. THE TEMPERATURE RAMP LOOP ---
# This is the main loop of the script.
# We will create a new Langevin object for each temperature step.
for target_temp_K in TEMP_RAMP_K:
    print(f"\n--- Starting T = {target_temp_K} K ---")

    # --- NEW --- Record the start time for this specific loop
    loop_start_time = time.perf_counter()

    # --- Create a NEW Langevin object for this temperature ---
    # It automatically inherits the atoms' current positions/velocities
    dyn = Langevin(atoms,
                   timestep=TIMESTEP,
                   temperature=target_temp_K * kB,  # Use 'temperature', not 'temperature_K'
                   friction=FRICTION)

    # Run the equilibration steps...
    print(f"  Equilibrating for {STEPS_EQUILIBRATE} steps ({STEPS_EQUILIBRATE * TIMESTEP / fs:.1f} fs)...")
    dyn.run(STEPS_EQUILIBRATE)

    # ...now run the production steps and save the trajectory
    print(f"  Running production for {STEPS_PRODUCTION} steps ({STEPS_PRODUCTION * TIMESTEP / fs:.1f} fs)...")

    # Define the trajectory file name for this temperature
    traj_filename = os.path.join(OUTPUT_DIR, f"cu_T_{int(target_temp_K)}K.traj")

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

    # --- NEW --- Calculate and print the time for this loop
    loop_end_time = time.perf_counter()
    print(f"  > This temperature step took {loop_end_time - loop_start_time:.2f} seconds.")

print("\n--- All Simulations Finished ---")

# --- NEW --- Calculate and print the total script time
script_end_time = time.perf_counter()
print(f"Total script runtime: {script_end_time - script_start_time:.2f} seconds")