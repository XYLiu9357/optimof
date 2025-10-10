"""build_mof_map.py
Build MOFMap from clean training data for nearest neighbor queries.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.mof_map import MOFMap


def build_mof_map(
    project_path: Path = Path("."),
    normalize: bool = True,
    weights: tuple = None,
):
    """Build MOF map from clean data files.

    Args:
        project_path: Root path of the project
        normalize: If True, standardize features to mean=0, std=1
        weights: Optional tuple of (thermal_weight, solvent_weight, water_weight)
                If None, uses equal weights after normalization
    """
    project_path = Path(project_path)

    print("Loading clean data files...")

    # Load the three clean datasets
    thermal_df = joblib.load(project_path / "data" / "thermal" / "thermal_clean_data.pkl")
    solvent_df = joblib.load(project_path / "data" / "solvent" / "solvent_clean_data.pkl")
    water_df = joblib.load(project_path / "data" / "water_and_haz" / "water_clean_data.pkl")

    print(f"Loaded thermal data: {thermal_df.shape[0]} samples")
    print(f"Loaded solvent data: {solvent_df.shape[0]} samples")
    print(f"Loaded water data: {water_df.shape[0]} samples")

    # Reset indices to avoid ambiguity and extract labels
    thermal_df = thermal_df.reset_index()
    solvent_df = solvent_df.reset_index()
    water_df = water_df.reset_index()

    # Create dataframes with name and label
    thermal_data = pd.DataFrame({
        'name': thermal_df['name'] if 'name' in thermal_df.columns else thermal_df.index,
        'thermal': thermal_df['thermal']
    })
    solvent_data = pd.DataFrame({
        'name': solvent_df['name'] if 'name' in solvent_df.columns else solvent_df.index,
        'solvent': solvent_df['solvent']
    })
    water_data = pd.DataFrame({
        'name': water_df['name'] if 'name' in water_df.columns else water_df.index,
        'water': water_df['water']
    })

    # Merge on name - inner join to get only samples with all three labels
    print("\nMerging datasets...")
    merged_df = thermal_data.merge(solvent_data, on='name', how='inner')
    merged_df = merged_df.merge(water_data, on='name', how='inner')

    print(f"Merged dataset: {merged_df.shape[0]} samples with all three labels")
    print(f"Columns: {merged_df.columns.tolist()}")

    # Check for NaN values
    if merged_df.isna().any().any():
        print("\nWarning: NaN values detected, dropping...")
        merged_df = merged_df.dropna()
        print(f"After dropping NaN: {merged_df.shape[0]} samples")

    # Print scale analysis before normalization
    print("\n" + "=" * 70)
    print("SCALE ANALYSIS:")
    print("=" * 70)
    feats = ['thermal', 'solvent', 'water']
    for feat in feats:
        print(f"{feat:10s}: range=[{merged_df[feat].min():7.2f}, {merged_df[feat].max():7.2f}], "
              f"mean={merged_df[feat].mean():7.2f}, std={merged_df[feat].std():7.2f}")

    # Apply normalization if requested
    scaler = None
    if normalize:
        print("\nApplying StandardScaler normalization...")
        scaler = StandardScaler()
        merged_df[feats] = scaler.fit_transform(merged_df[feats])

        print("\nAfter normalization:")
        for feat in feats:
            print(f"{feat:10s}: range=[{merged_df[feat].min():7.2f}, {merged_df[feat].max():7.2f}], "
                  f"mean={merged_df[feat].mean():7.2f}, std={merged_df[feat].std():7.2f}")

        # Save scaler for use during prediction
        scaler_path = project_path / "data" / "mof_map_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Build MOFMap
    print("\nBuilding MOFMap...")
    mof_map = MOFMap(merged_df, weights=weights)

    # Save MOFMap
    output_path = project_path / "data" / "mof_map.pkl"
    mof_map.export_to_file(output_path)
    print(f"MOFMap saved to {output_path}")

    return mof_map, merged_df, scaler


if __name__ == "__main__":
    # Parse command line args for normalization and weights
    normalize = "--no-normalize" not in sys.argv
    weights = None

    if "--weights" in sys.argv:
        idx = sys.argv.index("--weights")
        if idx + 3 < len(sys.argv):
            weights = (float(sys.argv[idx + 1]), float(sys.argv[idx + 2]), float(sys.argv[idx + 3]))
            print(f"Using custom weights: thermal={weights[0]}, solvent={weights[1]}, water={weights[2]}")

    mof_map, merged_df, scaler = build_mof_map(normalize=normalize, weights=weights)
    print("\n✓ MOFMap built successfully!")
    print(f"  Total MOFs in map: {merged_df.shape[0]}")
    print(f"  Properties indexed: thermal, solvent, water")
    print(f"  Normalization: {'enabled' if normalize else 'disabled'}")
    if weights:
        print(f"  Weights: {weights}")
