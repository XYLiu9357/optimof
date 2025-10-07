"""data_selection.py
Prepare training data from source CSVs.
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

from src.model_features.config import (
    LABEL_COLUMNS,
    SOLVENT_DROP_COLS,
    SOLVENT_RENAME_MAP,
    THERMAL_DROP_COLS,
    THERMAL_RENAME_MAP,
    WATER_DROP_COLS,
    WATER_MODEL_DROP_COLS,
    WATER_RENAME_MAP,
    get_feature_columns,
    validate_dataframe_columns,
)


def _prepare_model_data(
    csv_path: Path,
    output_path: Path,
    drop_cols: list,
    rename_map: dict,
    label_col: str,
    extra_drop_cols: list | None = None,
) -> pd.DataFrame:
    """Generic function to prepare training data from CSV.

    Args:
        csv_path: Path to source CSV file
        output_path: Path to save processed pkl file
        drop_cols: List of columns to drop
        rename_map: Dictionary for renaming columns
        label_col: Name of the label column
        extra_drop_cols: Optional additional columns to drop after main processing

    Returns:
        Cleaned DataFrame ready for training
    """
    print(f"Processing {label_col} data from {csv_path}")

    # Load and clean
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    df = df.rename(columns=rename_map)

    # Filter out rows with missing label
    initial_count = len(df)
    df = df.dropna(subset=[label_col])
    print(f"  {label_col.capitalize()}: Kept {len(df)}/{initial_count} rows with valid labels")

    # Validate
    if not validate_dataframe_columns(df, ["name", label_col]):
        raise ValueError(f"{label_col.capitalize()} data missing required columns")

    # Set name as index and move label to first column
    df = df.set_index("name")
    label_data = df.pop(label_col)
    df.insert(0, label_col, label_data)

    # Drop extra columns if specified (e.g., water-model-specific columns)
    if extra_drop_cols:
        df = df.drop(
            columns=[col for col in extra_drop_cols if col in df.columns],
            errors="ignore",
        )

    # Save
    joblib.dump(df, output_path)
    print(f"  Saved: {df.shape} → {output_path}")

    return df


def prepare_thermal_data(
    thermal_csv_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Prepare thermal stability training data from CSV.

    Args:
        thermal_csv_path: Path to thermal_all_data.csv
        output_path: Path to save thermal_clean_data.pkl

    Returns:
        Cleaned DataFrame ready for training
    """
    return _prepare_model_data(
        csv_path=thermal_csv_path,
        output_path=output_path,
        drop_cols=THERMAL_DROP_COLS,
        rename_map=THERMAL_RENAME_MAP,
        label_col="thermal",
    )


def prepare_solvent_data(
    solvent_csv_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Prepare solvent stability training data from CSV.

    Args:
        solvent_csv_path: Path to solvent_all_data.csv
        output_path: Path to save solvent_clean_data.pkl

    Returns:
        Cleaned DataFrame ready for training
    """
    return _prepare_model_data(
        csv_path=solvent_csv_path,
        output_path=output_path,
        drop_cols=SOLVENT_DROP_COLS,
        rename_map=SOLVENT_RENAME_MAP,
        label_col="solvent",
    )


def prepare_water_data(
    water_csv_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Prepare water stability training data from CSV.

    Args:
        water_csv_path: Path to water_and_haz_all_data.csv
        output_path: Path to save water_clean_data.pkl

    Returns:
        Cleaned DataFrame ready for training
    """
    return _prepare_model_data(
        csv_path=water_csv_path,
        output_path=output_path,
        drop_cols=WATER_DROP_COLS,
        rename_map=WATER_RENAME_MAP,
        label_col="water",
        extra_drop_cols=WATER_MODEL_DROP_COLS,
    )


def prepare_all_training_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare all training data from source CSVs.

    Args:
        data_dir: Root data directory containing subdirectories

    Returns:
        Tuple of (thermal_df, solvent_df, water_df)
    """
    data_dir = Path(data_dir)

    # Define paths
    thermal_csv = data_dir / "thermal" / "thermal_all_data.csv"
    solvent_csv = data_dir / "solvent" / "solvent_all_data.csv"
    water_csv = data_dir / "water_and_haz" / "water_and_haz_all_data.csv"

    thermal_out = data_dir / "thermal" / "thermal_clean_data.pkl"
    solvent_out = data_dir / "solvent" / "solvent_clean_data.pkl"
    water_out = data_dir / "water_and_haz" / "water_clean_data.pkl"

    # Validate inputs exist
    for csv_path in [thermal_csv, solvent_csv, water_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)

    # Process each dataset
    thermal_df = prepare_thermal_data(thermal_csv, thermal_out)
    solvent_df = prepare_solvent_data(solvent_csv, solvent_out)
    water_df = prepare_water_data(water_csv, water_out)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Thermal:  {thermal_df.shape[0]} MOFs × {thermal_df.shape[1]} features")
    print(f"Solvent:  {solvent_df.shape[0]} MOFs × {solvent_df.shape[1]} features")
    print(f"Water:    {water_df.shape[0]} MOFs × {water_df.shape[1]} features")
    print(f"\nNaN check:")
    print(f"  Thermal: {thermal_df.isna().sum().sum()} NaN values")
    print(f"  Solvent: {solvent_df.isna().sum().sum()} NaN values")
    print(f"  Water:   {water_df.isna().sum().sum()} NaN values")
    print("=" * 70)

    return thermal_df, solvent_df, water_df


def get_feature_column_names(split_data_path: Path) -> list:
    """Get feature column names from a data file.

    Args:
        split_data_path: Path to any data pkl file

    Returns:
        List of feature column names (excludes label columns)
    """
    df = joblib.load(split_data_path)
    return get_feature_columns(df.columns.tolist())

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DATA SELECTION")
    print("=" * 70 + "\n")

    project_path = Path(".")
    data_dir = project_path / "data"

    try:
        # Prepare all training data
        prepare_all_training_data(data_dir)

        # Validate outputs
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)

        thermal_df = joblib.load(data_dir / "thermal" / "thermal_clean_data.pkl")
        solvent_df = joblib.load(data_dir / "solvent" / "solvent_clean_data.pkl")
        water_df = joblib.load(data_dir / "water_and_haz" / "water_clean_data.pkl")

        print(f"✓ Thermal data loaded: {thermal_df.shape}")
        print(f"✓ Solvent data loaded: {solvent_df.shape}")
        print(f"✓ Water data loaded: {water_df.shape}")

        # Get feature columns
        feature_cols = get_feature_column_names(
            data_dir / "thermal" / "thermal_clean_data.pkl"
        )
        print(f"\n✓ Total feature columns: {len(feature_cols)}")

        print("\n" + "=" * 70)
        print("SUCCESS: All training data prepared")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
