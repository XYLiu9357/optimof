"""data_selection.py
Merge solvent, thermal, and water stability data into one
dataframe. Used to produce all_in_one.pkl.
"""

from pathlib import Path

import joblib
import pandas as pd


def split_data(data_dir: Path, all_in_one_data_path: Path):
    """split_data
    Split the data to be used by the three models. Split data contains no NaN values and
    can be used directly in training.
    """
    data_dir = Path(data_dir)
    all_in_one_data_path = Path(all_in_one_data_path)
    init_df = joblib.load(all_in_one_data_path)
    init_df = init_df.set_index("name")
    labels = ["thermal", "solvent", "water"]
    for label in labels:
        # Extract
        valid_rows = init_df.loc[:, label].notna()
        prep_df = init_df.loc[valid_rows, :]

        # Remove unused label columns
        label_col = prep_df.pop(label)
        prep_df = prep_df.iloc[:, 2:]
        prep_df = pd.concat([label_col, prep_df], axis=1)

        # Remove additional columns for water stability data
        if label == "water":
            water_additional_drops = [
                "D_lc-I-0-all",
                "D_lc-I-1-all",
                "D_lc-I-2-all",
                "D_lc-I-3-all",
                "D_lc-S-0-all",
                "D_mc-Z-0-all",
                "POAV",
                "PONAV",
                "cell_v",
                "lc-I-0-all",
                "mc-I-0-all",
            ]
            prep_df = prep_df.drop(water_additional_drops, axis=1)
            file_path = data_dir / "water_and_haz" / f"{label}_split_data.pkl"
            joblib.dump(prep_df, file_path)
        # Save to file
        else:
            file_path = data_dir / label / f"{label}_split_data.pkl"
            joblib.dump(prep_df, file_path)


def merge_data(
    thermal_data_path: Path,
    solvent_data_path: Path,
    water_data_path: Path,
    saved_file_path: Path,
    saved_cols_path: Path,
):
    """merge_data
    Merge the data used for model training
    1. Extracts solvent, thermal, and water stability data.
    2. Drop unused columns and rename columns
    3. Reorganize existing columns
    """

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

    # Merge
    thermal_df = thermal_df.set_index("name")
    solvent_df = solvent_df.set_index("name")
    water_df = water_df.set_index("name")
    merged_df = thermal_df.combine_first(solvent_df)
    merged_df = merged_df.combine_first(water_df)
    merged_df = merged_df.reset_index()

    # Label columns go first
    label_cols = ["name", "thermal", "solvent", "water"]
    labels_df = merged_df[label_cols]
    other_df = merged_df.drop(columns=label_cols)
    merged_df = pd.concat([labels_df, other_df], axis=1)

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

    # nan_columns = merged_df.columns[merged_df.isna().any()].tolist()
    print(f"NaN count (merged): {merged_df.isna().sum().sum()}")

    # Save
    joblib.dump(merged_df.columns, saved_cols_path)
    joblib.dump(merged_df, saved_file_path)
    print(f"File saved to {saved_file_path}")


if __name__ == "__main__":
    print("**Data Selection**")
    project_path = Path(".")
    data_dir = project_path / "data"
    all_in_one_pkl = data_dir / "all_in_one.pkl"
    all_cols_pkl = data_dir / "all_in_one_cols.pkl"

    # Reorganize data
    if not all_in_one_pkl.is_file():
        thermal_path = data_dir / "thermal" / "thermal_all_data.csv"
        solvent_path = data_dir / "solvent" / "solvent_all_data.csv"
        water_path = data_dir / "water_and_haz" / "data.csv"
        merge_data(thermal_path, solvent_path, water_path, all_in_one_pkl, all_cols_pkl)
        split_data(data_dir, all_in_one_pkl)
    else:
        print(f"Warning: data exists at {all_in_one_pkl}")

    # Check data integrity
    thermal_pkl_path = data_dir / "thermal" / "thermal_split_data.pkl"
    solvent_pkl_path = data_dir / "solvent" / "solvent_split_data.pkl"
    water_pkl_path = data_dir / "water_and_haz" / "water_split_data.pkl"

    thermal_df = joblib.load(thermal_pkl_path)
    solvent_df = joblib.load(solvent_pkl_path)
    water_df = joblib.load(water_pkl_path)

    print(
        f"NaN count (split): \n"
        + f"- Thermal: {thermal_df.isna().sum().sum()}\n"
        + f"- Solvent: {solvent_df.isna().sum().sum()}\n"
        + f"- Water: {water_df.isna().sum().sum()}"
    )
    print("**Exit Data Selection**")
