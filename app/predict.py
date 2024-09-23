"""predict.py
Runs the models over a set of input features. Fills unknown labels
by prediction in the all-in-one data set.

1. Call extract_features to extract feature dataframe
2. Preprocess feature dataframe: formatting & feature engineering
3. Feed data through each model
4. Output predicted labels in dataframe
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

import torch
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.model_features import extract_features
from src.model_training.thermal_model import ThermalModel
from src.model_training.solvent_model import SolventModel
from src.model_training.water_stability_model import (
    WaterStabilityRF,
    WaterStabilityBoost,
)


# Get current date and time as a string
def current_time():
    currentDateAndTime = datetime.now()
    year = currentDateAndTime.year
    month = currentDateAndTime.month
    day = currentDateAndTime.day
    hour = currentDateAndTime.hour
    minute = currentDateAndTime.minute
    second = currentDateAndTime.second

    date_str = f"{year}-{month}-{day}_{hour}-{minute}-{second}"
    return date_str


# Convert logits to probabilities
def sigmoid(logits: np.ndarray):
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


# Extract features from file and fix formatting
def extract_from_file(
    project_path: str, target_path: str, id=current_time()
) -> pd.DataFrame:
    extracted_df = extract_features(project_path, target_path, id)
    extracted_df.loc[:, "name"] = id
    all_cols = joblib.load(os.path.join(project_path, "data", "all_in_one_cols.pkl"))

    # Extract feature columns (by removing label columns in stored data)
    label_cols = ["thermal", "solvent", "water"]
    feature_cols = [col for col in all_cols if col not in label_cols]
    feature_df = extracted_df.loc[:, feature_cols]
    feature_df = feature_df.set_index("name")
    return feature_df


# Run ANN model on the given input. Thermal and solvent model share this procedure.
def pred_ann(model_path: str, scalar_path: str, input_df: pd.DataFrame):
    # Standard scale data
    scalar: StandardScaler = joblib.load(scalar_path)
    scaled_df = scalar.transform(input_df)

    # Load model
    model = torch.load(model_path, weights_only=False)
    model.eval()

    # Use GPU if supported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = torch.tensor(scaled_df, dtype=torch.float32).to(device)

    # Run model
    pred = None
    with torch.no_grad():
        pred = model(feature_tensor).cpu().numpy()
    return pred


def pred_water(model_path: str, input_df: pd.DataFrame) -> bool:
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
    clean_water_df = input_df.drop(water_additional_drops, axis=1)

    # Make prediction
    probs = model.predict_proba(clean_water_df)
    return probs


# Make predictions based on input dataframe
def predict_df(project_path: str, feature_df: pd.DataFrame):
    model_dir = os.path.join(project_path, "model")
    scalar_dir = os.path.join(model_dir, "preprocess")

    # Thermal model
    thermal_model_path = os.path.join(model_dir, "thermal_model.pkl")
    thermal_scalar_path = os.path.join(scalar_dir, "thermal_scalar.pkl")
    temperature = pred_ann(thermal_model_path, thermal_scalar_path, feature_df)
    print(f"Thermal model prediction successful: {temperature}")

    # Solvent model
    solvent_model_path = os.path.join(model_dir, "solvent_model.pkl")
    solvent_scalar_path = os.path.join(scalar_dir, "solvent_scalar.pkl")
    solvent_logits = pred_ann(solvent_model_path, solvent_scalar_path, feature_df)
    solvent_flags = sigmoid(solvent_logits)
    print(f"Solvent model prediction successful: {solvent_flags}")

    # Water stability model
    water_model_path = os.path.join(model_dir, "water_rf_model.pkl")
    water_flag = pred_water(water_model_path, feature_df)
    print(f"Water stability model prediction successful: {water_flag}")
    return temperature, solvent_flags, water_flag


# Make predictions based on CIF file
def predict_from_file(project_path: str, target_path: str) -> pd.DataFrame:
    print(f"**Predicting properties of {target_path}**")
    # Extract feature vector
    feature_df = extract_from_file(project_path, target_path)
    print("Feature extraction successful")
    predict_df(project_path, feature_df)
    print("**Prediction complete**")


# Fill unknown labels in the all-in-one dataset
