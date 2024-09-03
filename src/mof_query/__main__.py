"""__main__.py
Initializes the mof query database by completing the following
1. Build or open a MOFMap object
2. Initialize a GUI for querying
"""

import os
import joblib
import pandas as pd
from src.model_features.data_selection import select_data
from src.mof_query import MOFMap

project_path = "."
data_dir = os.path.join(project_path, "data")
data_pkl_path = os.path.join(data_dir, "all_in_one.pkl")

# Reorganize data
if not os.path.isfile(data_pkl_path):
    thermal_path = os.path.join(data_dir, "thermal", "thermal_all_data.csv")
    solvent_path = os.path.join(data_dir, "solvent", "solvent_all_data.csv")
    water_path = os.path.join(data_dir, "water_and_haz", "data.csv")
    select_data(thermal_path, solvent_path, water_path, data_pkl_path)

df_all = joblib.load(data_pkl_path)
print(df_all.head())
