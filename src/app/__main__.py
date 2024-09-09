"""__main__.py
Initializes the mof query database by completing the following
1. Build or open a MOFMap object
2. Initialize a GUI for querying
"""

import os
import joblib
import pandas as pd

from src.app.predict import predict_from_file
from src.app.mof_map import MOFMap
from src.model_training.thermal_model import ThermalModel
from src.model_training.solvent_model import SolventModel

project_path = "."
data_dir = os.path.join(project_path, "data")
data_pkl_path = os.path.join(data_dir, "all_in_one.pkl")
df_all = joblib.load(data_pkl_path)

target_path = os.path.join(project_path, "test", "ABAVIJ_clean.cif")
predict_from_file(project_path, target_path)
