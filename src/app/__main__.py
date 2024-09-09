"""__main__.py
Initializes the mof query database by completing the following
1. Build or open a MOFMap object
2. Initialize a GUI for querying
"""

import os
import joblib
import pandas as pd
from src.model_features.data_selection import merge_data
from src.app import MOFMap

project_path = "."
data_dir = os.path.join(project_path, "data")
data_pkl_path = os.path.join(data_dir, "all_in_one.pkl")

df_all = joblib.load(data_pkl_path)
print(df_all.head())
