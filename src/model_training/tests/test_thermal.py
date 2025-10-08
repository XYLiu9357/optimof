"""test_thermal.py
Utility script for evaluating performance of the thermal breakdown temperature model.
Requires a test set that was not used in training.
"""

import numpy as np
import pandas as pd
import torch

from src.config.paths import (
    MODEL_DIR,
    SCALER_DIR,
    THERMAL_DATA_DIR,
    THERMAL_PERFORMANCE_DIR,
)
from src.model_training.base.data_processor import DataProcessor
from src.model_training.base.evaluator import RegressionEvaluator
from src.model_training.thermal_model import ThermalModel  # Import for unpickling


def evaluate_thermal_model():
    """Load thermal model and evaluate on test set."""
    print("**Testing thermal model accuracy**")

    # Load paths
    model_file_path = MODEL_DIR / "thermal_model.pkl"
    test_data_path = THERMAL_DATA_DIR / "thermal_test_data.pkl"
    scaler_path = SCALER_DIR / "thermal_scaler.pkl"

    # Load test data
    test_df: pd.DataFrame = pd.read_pickle(test_data_path)
    test_labels = test_df.loc[:, "thermal"].values
    test_features = test_df.loc[:, test_df.columns != "thermal"]

    # Load scaler and transform features
    data_processor = DataProcessor(label_col="thermal")
    data_processor.load_scaler(scaler_path)
    test_features_scaled = data_processor.transform(test_features)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_file_path, weights_only=False)
    model.eval()
    model.to(device)

    # Make predictions
    test_features_scaled = test_features_scaled.apply(pd.to_numeric, errors="raise")
    test_features_tensor = torch.tensor(
        test_features_scaled.values, dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        predictions = model(test_features_tensor).cpu().numpy().flatten()

    # Evaluate
    evaluator = RegressionEvaluator()
    metrics = evaluator.calculate_metrics(test_labels, predictions)
    evaluator.plot_results(test_labels, predictions, THERMAL_PERFORMANCE_DIR)

    return metrics


if __name__ == "__main__":
    evaluate_thermal_model()
