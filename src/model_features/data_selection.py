"""data_selection.py
Merge solvent, thermal, and water stability data into one
dataframe. Used to produce all_in_one.pkl.
"""

import os
import joblib
import pandas as pd


def select_data(thermal_data_path, solvent_data_path, water_data_path, saved_file_path):
    """select_data
    Selects the data used for model training
    1. Extracts solvent, thermal, and water stability data.
    2. Drop unused columns and rename columns
    3. Reorganize existing columns
    """
    print("**Data Selection**")
    thermal_df = pd.read_csv(thermal_data_path)
    solvent_df = pd.read_csv(solvent_data_path)
    water_df = pd.read_csv(water_data_path)

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

    thermal_df = thermal_df.rename(columns={"refcode": "name", "T": "thermal"})
    solvent_df = solvent_df.rename(columns={"refcode": "name", "flag": "solvent"})
    water_df = water_df.rename(columns={"MOF_name": "name", "water_label": "water"})

    # Gather unique columns
    thermal_columns = set(thermal_df.columns)
    solvent_columns = set(solvent_df.columns)
    water_columns = set(water_df.columns)
    unique_columns = list(thermal_columns.union(solvent_columns).union(water_columns))
    unique_columns.sort()
    print(f"Unique columns: {len(unique_columns)}")

    # Gather unique MOF names
    thermal_names = set(thermal_df.loc[:, "name"])
    solvent_names = set(solvent_df.loc[:, "name"])
    water_names = set(water_df.loc[:, "name"])
    unique_names = list(thermal_names.union(solvent_names).union(water_names))
    unique_names.sort()
    print(f"Unique MOFs: {len(unique_names)}")

    # Merge
    thermal_df = thermal_df.set_index("name")
    solvent_df = solvent_df.set_index("name")
    water_df = water_df.set_index("name")
    merged_df = thermal_df.combine_first(solvent_df)
    merged_df = merged_df.combine_first(water_df)
    merged_df = merged_df.reset_index()

    # Clean empty features
    merged_drop_cols = [
        "D_func-I-0-all",
        "D_func-I-1-all",
        "D_func-I-2-all",
        "D_func-I-3-all",
        "D_func-S-0-all",
        "D_func-T-0-all",
        "D_func-Z-0-all",
        "D_func-alpha-0-all",
        "D_func-chi-0-all",
        "D_lc-T-0-all",
        "D_lc-Z-0-all",
        "D_lc-alpha-0-all",
        "D_lc-chi-0-all",
        "D_mc-I-0-all",
        "D_mc-I-1-all",
        "D_mc-I-2-all",
        "D_mc-I-3-all",
        "D_mc-S-0-all",
        "D_mc-T-0-all",
        "D_mc-chi-0-all",
    ]
    merged_df = merged_df.loc[:, ~merged_df.columns.isin(merged_drop_cols)]
    print(f"Final dataframe shape: {merged_df.shape}")

    # Fill NaN with 0
    nan_columns = merged_df.columns[merged_df.isna().any()].tolist()
    print(f"NaN columns: {nan_columns}")
    print(f"NaN entry count: {merged_df.isna().sum().sum()}")

    # Save
    joblib.dump(merged_df, saved_file_path)
    print(f"File saved to {saved_file_path}")

    print("**Exit Data Selection**")
