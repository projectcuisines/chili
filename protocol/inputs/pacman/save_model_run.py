#Use this file to save Model_outputs.npy and Model_inputs.npy whenever you want to save the data and not have to run the code again
#most useful for Monte Carlo simulations, but also can use for single runs

import os
import shutil

# Define the source file and destination folder
source_file_outputs = "Model_outputs.npy"
source_file_inputs = "Model_inputs.npy"
destination_folder = "simulation_results"
# Ensure the folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
# Construct a unique filename using the number of existing saved runs
existing_output_files = [f for f in os.listdir(destination_folder) if f.startswith("Model_outputs_run")]
existing_input_files = [f for f in os.listdir(destination_folder) if f.startswith("Model_inputs_run")]
run_number = len(existing_output_files) + 1
destination_file_outputs = os.path.join(destination_folder, f"Model_outputs_run{run_number}.npy")
destination_file_inputs = os.path.join(destination_folder, f"Model_inputs_run{run_number}.npy")
# Move and rename the file
shutil.move(source_file_outputs, destination_file_outputs)
shutil.move(source_file_inputs, destination_file_inputs)

print(f"Model outputs saved as {destination_file_outputs}")
print(f"Model inputs saved as {destination_file_inputs}")
