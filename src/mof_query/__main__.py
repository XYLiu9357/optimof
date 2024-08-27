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
query = (600, 0.9, 0.8)

# Read data and extract ground-truth labels
thermal_data_path = os.path.join(data_dir, "thermal", "thermal_all_data.csv")
solvent_data_path = os.path.join(data_dir, "solvent", "solvent_all_data.csv")
water_and_haz_data_path = os.path.join(data_dir, "water_and_haz", "data.csv")

thermal_df = pd.read_csv(thermal_data_path)
solvent_df = pd.read_csv(solvent_data_path)
water_df = pd.read_csv(water_and_haz_data_path)

# thermal_compact = thermal_df.loc[:, ["refcode", "T"]].copy()
# solvent_compact = solvent_df.loc[:, ["refcode", "flag"]].copy()
# water_compact = water_and_haz_df.loc[:, ["MOF_name", "water_label"]].copy()
thermal_drop_cols = ["filename", "0", "CoRE_name", "name"]  # "refcode"
solvent_drop_cols = [
    "Unnamed: 0",
    "doi",
    "filename",
    "0",
    "CoRE_name",
    "name",
]  # "refcode"
water_drop_cols = ["acid_label", "base_label", "boiling_label", "data_set"]

thermal_df = thermal_df.loc[:, ~thermal_df.columns.isin(thermal_drop_cols)]
solvent_df = solvent_df.loc[:, ~solvent_df.columns.isin(solvent_drop_cols)]
water_df = water_df.loc[:, ~water_df.columns.isin(water_drop_cols)]

# Merge ground truth dataframes
thermal_df = thermal_df.rename(columns={"refcode": "name", "T": "thermal"})
solvent_df = solvent_df.rename(columns={"refcode": "name", "flag": "solvent"})
water_df = water_df.rename(columns={"MOF_name": "name", "water_label": "water"})

ground_truth = pd.merge(thermal_df, solvent_df, on="name", how="outer")
ground_truth = pd.merge(ground_truth, water_df, on="name", how="outer")
name_col = ground_truth.pop("name")
ground_truth.insert(0, "name", name_col)
print(ground_truth)

# mof_map = MOFMap(mof_df=ground_truth)
