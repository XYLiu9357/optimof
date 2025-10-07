"""__main__.py
Runs the mof query module
"""

"""__main__.py
Initializes the mof query database by completing the following
1. Build or open a MOFMap object
2. Initialize a GUI for querying
"""

import os
import sys
import joblib
import subprocess
import numpy as np
import pandas as pd

from datetime import datetime
import torch
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from app import run_flask_client
from src.utils import MOFMap
from src.utils import (
    predict_df,
    predict_from_file,
    fill_all_unknown,
    get_nearest_neighbor,
)
from src.model_training.thermal_model import ThermalModel
from src.model_training.solvent_model import SolventModel
from src.model_training.water_stability_model import (
    WaterStabilityRF,
    WaterStabilityBoost,
)

project_path = "."
data_dir = os.path.join(project_path, "data")
data_pkl_path = os.path.join(data_dir, "all_in_one.pkl")
mof_map_file_path = os.path.join(data_dir, "mof_map.pkl")
df_all = joblib.load(data_pkl_path)


# Get current date and time as a string
def current_time():
    currentDateAndTime = datetime.now()
    year = currentDateAndTime.year
    month = currentDateAndTime.month
    day = currentDateAndTime.day
    hour = currentDateAndTime.hour
    minute = currentDateAndTime.minute
    second = currentDateAndTime.second

    date_str = f"{year}-{month}-{day}_{hour}-{minute}-{second}"
    return date_str


def helper_predict_cif(target_file: str):
    # ACAJIY_clean.cif, 1499489-acs.cgd.6b01265_1499490_clean.cif
    print(f"Input file detected, running feature extraction on file {target_file}")
    if not os.path.isfile(target_file) or target_file[-4:] != ".cif":
        print(f"**Error: {target_file} is not a valid file")
        exit(1)

    target_path = os.path.join(project_path, target_file)
    print(f"**Predicting properties of {target_path}**")

    # Property prediction (with print)
    tempeartures, solvent_flags, water_flags = predict_from_file(
        project_path, target_path
    )

    # Nearest neighbor
    prop_df = pd.DataFrame(
        data={
            "thermal": tempeartures[0],
            "solvent": solvent_flags[0],
            "water": np.argmax(water_flags, axis=1) + 1,
        }
    )
    mof_map_path = os.path.join(data_dir, "mof_map.pkl")
    target_nn = get_nearest_neighbor(mof_map_path, prop_df)
    print(f"Nearest neighbor found: {target_nn}")
    print("**Prediction complete**")
    return tempeartures, solvent_flags, water_flags


def helper_build_mof_map(filled_data_path: str):
    # Build MOFMap
    df: pd.DataFrame = joblib.load(filled_data_path)
    mof_map: MOFMap = MOFMap(df)

    # Quick correctness check
    rand_idx = np.random.randint(0, df.shape[0])
    query: np.ndarray = df.iloc[rand_idx, 1:4].values.reshape(1, -1)
    nn_result = mof_map.nearest_neighbor_query(query)
    nn_expected = df.loc[rand_idx, "name"]
    print(
        f"MOFMap: nearest neighbor found is {nn_result[0][0]}, expected {nn_expected}"
    )

    # Save MOFMap
    mof_map.export_to_file(mof_map_file_path)


def pred_multiple(all_files, num_cifs=None):
    counter = 0
    output_df = pd.DataFrame(columns=["filename", "thermal", "solvent", "water"])
    for file in all_files:
        if (
            num_cifs is not None and counter >= num_cifs
        ):  # Set maximum number of cifs to process
            break
        file_name = os.fsdecode(file)
        if file_name.endswith(".cif"):
            file_path = os.path.join(target_directory, file_name)
            tempeartures, solvent_flags, water_flags = list(
                helper_predict_cif(file_path)
            )
            cur_file_df = pd.DataFrame(
                {
                    "filename": file_name,
                    "thermal": tempeartures[0],
                    "solvent": solvent_flags[0],
                    "water": np.argmax(water_flags),
                }
            )
            output_df = pd.concat([output_df, cur_file_df])
        else:
            print(f"Warning: skipping over non-cif file {file_name}")
        counter += 1
    return output_df


# Run app client
def run():
    print("**Running Flask client")
    run_flask_client()


# Run app
if len(sys.argv) == 1:
    run()
# Fills unknown labels
elif len(sys.argv) == 2:
    if sys.argv[1] == "-f" or sys.argv[1] == "--fill":
        filled_df = fill_all_unknown(project_path, data_pkl_path)
    else:
        print("Module called with unknown argument format")
# There are a few options with 2 arguments
elif len(sys.argv) == 3:
    # Run prediction on a given cif file
    if sys.argv[1] == "-c" or sys.argv[1] == "--cif":
        helper_predict_cif(sys.argv[2])
    # Run prediction on all cif files in a directory
    elif sys.argv[1] == "-pa" or sys.argv[1] == "--predict-all":
        target_directory = sys.argv[2]
        if not os.path.isdir(target_directory):
            print(f"**Error: {target_directory} is not a valid directory")
            exit(1)
        output_df = pd.DataFrame(columns=["filename", "thermal", "solvent", "water"])
        all_files = os.listdir(target_directory)
        all_files.sort()
        output_df = pred_multiple(all_files)
        output_df.to_csv(os.path.join(project_path, f"{current_time()}_results.csv"))
        print(output_df)
    # Build MOF map data base
    elif sys.argv[1] == "-b" or sys.argv[1] == "--build-db":
        filled_data_path = sys.argv[2]
        if not os.path.isfile(filled_data_path):
            print(f"**Error: {filled_data_path} is not a valid file")
            exit(1)
        helper_build_mof_map(filled_data_path)
    else:
        print("Module called with unknown argument format")
elif len(sys.argv) == 4:
    if sys.argv[1] == "-p" or sys.argv[1] == "--predict":
        target_directory = sys.argv[2]
        if not os.path.isdir(target_directory):
            print(f"**Error: {target_directory} is not a valid directory")
            exit(1)
        num_cifs = int(sys.argv[3])
        all_files = os.listdir(target_directory)
        all_files.sort()
        output_df = pred_multiple(all_files, num_cifs)
        output_df.to_csv(os.path.join(project_path, f"{current_time()}_results.csv"))
        # print(output_df)
else:
    print("Module called with unknown argument format")
