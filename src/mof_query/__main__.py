"""__main__.py
Initializes the mof query database by completing the following
1. Build or open a MOFMap object
2. Initialize a GUI for querying
"""

import os
import pandas as pd
from src.mof_query import MOFMap

project_path = "."
data_dir = os.path.join(project_path, "data")

# Read data and extract ground-truth labels
solvent_data_path = os.path.join(data_dir, "solvent", "solvent_all_data.csv")
thermal_data_path = os.path.join(data_dir, "thermal", "thermal_all_data.csv")
water_and_haz_data_path = os.path.join(data_dir, "water_and_haz", "data.csv")

solvent_df = pd.read_csv(solvent_data_path)
thermal_df = pd.read_csv(thermal_data_path)
water_and_haz_df = pd.read_csv(water_and_haz_data_path)

thermal_compact = thermal_df.loc[:, ["refcode", "T"]].copy()
solvent_compact = solvent_df.loc[:, ["refcode", "flag"]].copy()
water_compact = water_and_haz_df.loc[:, ["MOF_name", "water_label"]].copy()

# Merge to get ground truth table
thermal_compact = thermal_compact.rename(columns={"refcode": "name", "T": "thermal"})
solvent_compact = solvent_compact.rename(columns={"refcode": "name", "flag": "solvent"})
water_compact = water_compact.rename(
    columns={"MOF_name": "name", "water_label": "water"}
)

ground_truth = pd.merge(thermal_compact, solvent_compact, on="name", how="outer")
ground_truth = pd.merge(ground_truth, water_compact, on="name", how="outer")
print(ground_truth)

mof_map = MOFMap(mof_df=ground_truth)
