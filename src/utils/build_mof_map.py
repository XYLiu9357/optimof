"""build_mof_map.py
Build MOFMap from clean training data for nearest neighbor queries.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.model_training.water_stability_model import WaterStabilityPipeline
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
        scaler_path = project_path / "data" / "mof_map" / "mof_map_scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Build MOFMap
    print("\nBuilding MOFMap...")
    mof_map = MOFMap(merged_df, weights=weights)

    # Save MOFMap
    output_path = project_path / "data" / "mof_map" / "mof_map.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mof_map.export_to_file(output_path)
    print(f"MOFMap saved to {output_path}")

    return mof_map, merged_df, scaler


def sigmoid(logits: np.ndarray):
    """Convert logits to probabilities."""
    def _positive_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(x):
        exp = np.exp(x)
        return exp / (exp + 1)

    positive = logits >= 0
    negative = ~positive

    probs = np.empty_like(logits, dtype=np.float32)
    probs[positive] = _positive_sigmoid(logits[positive])
    probs[negative] = _negative_sigmoid(logits[negative])
    return probs


def trim_labels(df: pd.DataFrame):
    """Remove label columns from dataframe."""
    trim_targets = ["name", "thermal", "solvent", "water"]
    return df.drop(columns=[col for col in trim_targets if col in df.columns], errors='ignore')


def pred_ann(model_path: Path, scaler_path: Path, feature_df: pd.DataFrame):
    """Run ANN model prediction (for thermal and solvent models)."""
    # Load and apply scaler
    scaler: StandardScaler = joblib.load(scaler_path)
    expected_features = scaler.feature_names_in_
    feature_df = feature_df[expected_features]
    scaled_df = scaler.transform(feature_df)

    # Load model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # Use GPU if supported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = torch.tensor(scaled_df, dtype=torch.float32).to(device)

    # Run model
    with torch.no_grad():
        pred = model(feature_tensor).cpu().numpy()
    return pred


def pred_water(model_path: Path, feature_df: pd.DataFrame):
    """Run water stability model prediction."""
    model_obj = joblib.load(model_path)
    model: RandomForestClassifier = model_obj.model

    # Drop additional columns
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
    clean_water_df = feature_df.drop(columns=water_additional_drops, errors='ignore')

    # Reorder features to match the order expected by the model
    expected_features = model.feature_names_in_
    available_features = [f for f in expected_features if f in clean_water_df.columns]
    clean_water_df = clean_water_df[available_features]

    # Make prediction
    probs = model.predict_proba(clean_water_df)
    return probs


def fill_missing_labels(
    project_path: Path = Path("."),
    normalize: bool = True,
    weights: tuple = None,
):
    """Fill missing labels for MOFs that don't have all three labels.

    This function:
    1. Loads all three clean datasets
    2. Creates a union of all MOFs (outer join)
    3. For MOFs missing labels, uses the trained models to predict them
    4. Saves the complete dataset with all labels filled
    5. Builds a MOF map from the filled dataset

    Args:
        project_path: Root path of the project
        normalize: If True, standardize features to mean=0, std=1
        weights: Optional tuple of (thermal_weight, solvent_weight, water_weight)

    Returns:
        mof_map: The MOFMap object with all samples
        filled_df: DataFrame with all labels filled
        scaler: The StandardScaler used for normalization (or None)
    """
    project_path = Path(project_path)
    model_dir = project_path / "model"
    scaler_dir = model_dir / "scalers"

    print("=" * 70)
    print("FILLING MISSING LABELS FOR MOF MAP")
    print("=" * 70)
    print("\nLoading clean data files...")

    # Load the three clean datasets
    thermal_df = joblib.load(project_path / "data" / "thermal" / "thermal_clean_data.pkl")
    solvent_df = joblib.load(project_path / "data" / "solvent" / "solvent_clean_data.pkl")
    water_df = joblib.load(project_path / "data" / "water_and_haz" / "water_clean_data.pkl")

    print(f"Loaded thermal data: {thermal_df.shape[0]} samples")
    print(f"Loaded solvent data: {solvent_df.shape[0]} samples")
    print(f"Loaded water data: {water_df.shape[0]} samples")

    # Reset indices
    thermal_df = thermal_df.reset_index()
    solvent_df = solvent_df.reset_index()
    water_df = water_df.reset_index()

    # Extract name and label columns
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

    # Merge with OUTER join to get all samples
    print("\nMerging datasets with outer join...")
    merged_df = thermal_data.merge(solvent_data, on='name', how='outer')
    merged_df = merged_df.merge(water_data, on='name', how='outer')

    print(f"Total unique MOFs: {merged_df.shape[0]}")

    # Count missing labels
    thermal_missing = merged_df['thermal'].isna().sum()
    solvent_missing = merged_df['solvent'].isna().sum()
    water_missing = merged_df['water'].isna().sum()

    print(f"\nMissing labels:")
    print(f"  Thermal: {thermal_missing}")
    print(f"  Solvent: {solvent_missing}")
    print(f"  Water: {water_missing}")

    # We need to get feature data for each MOF to make predictions
    # We'll use the thermal_df as the base since it has all features
    # Map name to full feature row
    thermal_features = thermal_df.set_index('name')
    solvent_features = solvent_df.set_index('name')
    water_features = water_df.set_index('name')

    # For each dataset, keep only the feature columns (drop labels)
    thermal_features = thermal_features.drop(columns=['thermal'], errors='ignore')
    solvent_features = solvent_features.drop(columns=['solvent'], errors='ignore')
    water_features = water_features.drop(columns=['water'], errors='ignore')

    # Combine all features with outer join
    all_features = thermal_features.combine_first(solvent_features).combine_first(water_features)

    # Add features to merged_df
    merged_with_features = merged_df.set_index('name').join(all_features, how='left')

    # Check if any MOFs have no features at all
    mofs_with_features = ~merged_with_features.iloc[:, 3:].isna().all(axis=1)
    if not mofs_with_features.all():
        print(f"\nWarning: {(~mofs_with_features).sum()} MOFs have no feature data, dropping them...")
        merged_with_features = merged_with_features[mofs_with_features]

    print(f"\nMOFs with feature data: {merged_with_features.shape[0]}")

    # Now fill missing labels using models
    thermal_model_path = model_dir / "thermal_model.pkl"
    thermal_scaler_path = scaler_dir / "thermal_scaler.pkl"
    solvent_model_path = model_dir / "solvent_model.pkl"
    solvent_scaler_path = scaler_dir / "solvent_scaler.pkl"
    water_model_path = model_dir / "water_rf_model.pkl"

    # Fill thermal labels
    if thermal_missing > 0:
        print(f"\nPredicting {thermal_missing} missing thermal labels...")
        thermal_nan_mask = merged_with_features['thermal'].isna()
        if thermal_nan_mask.any():
            features_for_pred = trim_labels(merged_with_features[thermal_nan_mask])
            # Remove any rows with all NaN features
            features_for_pred = features_for_pred.dropna(how='all')
            if len(features_for_pred) > 0:
                thermal_preds = pred_ann(thermal_model_path, thermal_scaler_path, features_for_pred)
                merged_with_features.loc[features_for_pred.index, 'thermal'] = thermal_preds.flatten()
                print(f"  Filled {len(thermal_preds)} thermal labels")

    # Fill solvent labels
    if solvent_missing > 0:
        print(f"\nPredicting {solvent_missing} missing solvent labels...")
        solvent_nan_mask = merged_with_features['solvent'].isna()
        if solvent_nan_mask.any():
            features_for_pred = trim_labels(merged_with_features[solvent_nan_mask])
            features_for_pred = features_for_pred.dropna(how='all')
            if len(features_for_pred) > 0:
                solvent_logits = pred_ann(solvent_model_path, solvent_scaler_path, features_for_pred)
                solvent_preds = sigmoid(solvent_logits)
                merged_with_features.loc[features_for_pred.index, 'solvent'] = solvent_preds.flatten()
                print(f"  Filled {len(solvent_preds)} solvent labels")

    # Fill water labels
    if water_missing > 0:
        print(f"\nPredicting {water_missing} missing water labels...")
        water_nan_mask = merged_with_features['water'].isna()
        if water_nan_mask.any():
            features_for_pred = trim_labels(merged_with_features[water_nan_mask])
            features_for_pred = features_for_pred.dropna(how='all')
            if len(features_for_pred) > 0:
                water_probs = pred_water(water_model_path, features_for_pred)
                water_preds = np.argmax(water_probs, axis=1) + 1  # Convert to 1-indexed classes
                merged_with_features.loc[features_for_pred.index, 'water'] = water_preds
                print(f"  Filled {len(water_preds)} water labels")

    # Drop any remaining rows with missing labels
    print("\nDropping any remaining samples with missing labels...")
    initial_count = len(merged_with_features)
    merged_with_features = merged_with_features.dropna(subset=['thermal', 'solvent', 'water'])
    final_count = len(merged_with_features)
    if initial_count > final_count:
        print(f"  Dropped {initial_count - final_count} samples")

    # Reset index to make 'name' a column again
    filled_df = merged_with_features.reset_index()

    print(f"\nFinal dataset: {filled_df.shape[0]} samples with all three labels")

    # Save the filled dataset
    filled_data_path = project_path / "data" / "mof_map" / "mof_map_filled_data.pkl"
    filled_data_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(filled_df, filled_data_path)
    print(f"Filled dataset saved to {filled_data_path}")

    # Print scale analysis before normalization
    print("\n" + "=" * 70)
    print("SCALE ANALYSIS:")
    print("=" * 70)
    feats = ['thermal', 'solvent', 'water']
    for feat in feats:
        print(f"{feat:10s}: range=[{filled_df[feat].min():7.2f}, {filled_df[feat].max():7.2f}], "
              f"mean={filled_df[feat].mean():7.2f}, std={filled_df[feat].std():7.2f}")

    # Apply normalization if requested
    scaler = None
    if normalize:
        print("\nApplying StandardScaler normalization...")
        scaler = StandardScaler()
        filled_df[feats] = scaler.fit_transform(filled_df[feats])

        print("\nAfter normalization:")
        for feat in feats:
            print(f"{feat:10s}: range=[{filled_df[feat].min():7.2f}, {filled_df[feat].max():7.2f}], "
                  f"mean={filled_df[feat].mean():7.2f}, std={filled_df[feat].std():7.2f}")

        # Save scaler for use during prediction
        scaler_path = project_path / "data" / "mof_map" / "mof_map_filled_scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Build MOFMap from filled data
    print("\nBuilding MOFMap from filled data...")
    mof_map = MOFMap(filled_df, weights=weights)

    # Save MOFMap
    output_path = project_path / "data" / "mof_map" / "mof_map_filled.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mof_map.export_to_file(output_path)
    print(f"Filled MOFMap saved to {output_path}")

    print("\n" + "=" * 70)
    print("✓ FILLED MOF MAP BUILT SUCCESSFULLY!")
    print("=" * 70)
    print(f"  Total MOFs in filled map: {filled_df.shape[0]}")
    print(f"  Properties indexed: thermal, solvent, water")
    print(f"  Normalization: {'enabled' if normalize else 'disabled'}")
    if weights:
        print(f"  Weights: {weights}")

    return mof_map, filled_df, scaler


if __name__ == "__main__":
    # Parse command line args for normalization and weights
    normalize = "--no-normalize" not in sys.argv
    weights = None
    fill_mode = "--fill-missing" in sys.argv

    if "--weights" in sys.argv:
        idx = sys.argv.index("--weights")
        if idx + 3 < len(sys.argv):
            weights = (float(sys.argv[idx + 1]), float(sys.argv[idx + 2]), float(sys.argv[idx + 3]))
            print(f"Using custom weights: thermal={weights[0]}, solvent={weights[1]}, water={weights[2]}")

    if fill_mode:
        # Fill missing labels mode
        mof_map, filled_df, scaler = fill_missing_labels(normalize=normalize, weights=weights)
    else:
        # Standard mode - only use samples with all three labels
        mof_map, merged_df, scaler = build_mof_map(normalize=normalize, weights=weights)
        print("\n✓ MOFMap built successfully!")
        print(f"  Total MOFs in map: {merged_df.shape[0]}")
        print(f"  Properties indexed: thermal, solvent, water")
        print(f"  Normalization: {'enabled' if normalize else 'disabled'}")
        if weights:
            print(f"  Weights: {weights}")
