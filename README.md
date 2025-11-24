Group Project for MS508 Fall 2025

# Functional Example
## Order of Operations:
1. data_acquisition 
2. rdf 
3. random_forest_ml 
4. validation_generation 
5. predictor

### data_acquisition
This is where we pull materials from materials. Materials can be pulled yourself, 
however for ease there are 30 materials in cif_files_v2 that can be used. Here we
pull 10 FCC, 10 BCC, and 10 HCP structures.

### rdf
**Note: rdf_process_and_compile.py is what should be ran. Ignore all other .py files.**

This will give you a master csv in /master_csv. It is this csv that is used to train the 
random forest ml model. 

### random_forest_ml
Trains random forest algorithm on .csv in /rdf/master_csv folder. Outputs:

- rf_model.joblib ~ model
- label_encoder.joblib ~ label encoder

### validation_generation

#### T-dep_structures
Generates .traj files of copper at an array of temperatures (300K - 1350K in this example).

You can change the STEPS_EQUILIBRATE, STEPS_PRODUCTION, SAVE_INTERVAL based on what will run well 
for your system. All .traj files are saved in md_temp_ramp

#### md_to_rdf
Here our .traj files are turned into rdfs. These rdfs must have the same bin size (200 in our case) as 
the ones produce in rdf (step 2). 
- individual_csvs ~ individual csv files for each .traj
- rdf_data ~ rdf data; plot of rdf vs temperature
- validation_rdfs ~ master csv used in predictor (step 5)

### predictor
**NOTE: Must be ran in terminal using the instructions found in the README**

Copy the outputs from random_forest_ml into this folder. Run and get your predictions.

