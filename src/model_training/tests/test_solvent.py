"""test_solvent.py
Utility script for evaluating performance of the solvent removal stability model.
Requires a test set that was not used in training.
"""

import numpy as np
import pandas as pd
import torch

from src.config.paths import (
    MODEL_DIR,
    SCALER_DIR,
    SOLVENT_DATA_DIR,
    SOLVENT_PERFORMANCE_DIR,
)
from src.model_training.base.data_processor import DataProcessor
from src.model_training.base.evaluator import BinaryClassificationEvaluator
from src.model_training.base.flexible_mlp import FlexibleMLP  # Import for unpickling


def evaluate_solvent_model():
    """Load solvent model and evaluate on test set."""
    print("**Testing solvent model accuracy**")

    # Load paths
    model_file_path = MODEL_DIR / "solvent_model.pkl"
    test_data_path = SOLVENT_DATA_DIR / "solvent_test_data.pkl"
    scaler_path = SCALER_DIR / "solvent_scaler.pkl"

    # Load test data
    test_df: pd.DataFrame = pd.read_pickle(test_data_path)
    test_labels = test_df.loc[:, "solvent"].values
    test_features = test_df.loc[:, test_df.columns != "solvent"]

    # Load scaler and transform features
    data_processor = DataProcessor(label_col="solvent")
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
        logits = model(test_features_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    # Binarize predictions
    binary_predictions = (probabilities >= 0.5).astype(int)

    # Evaluate
    evaluator = BinaryClassificationEvaluator()
    metrics = evaluator.calculate_metrics(test_labels, binary_predictions, probabilities)
    evaluator.plot_results(
        test_labels, binary_predictions, probabilities, SOLVENT_PERFORMANCE_DIR
    )

    return metrics


if __name__ == "__main__":
    evaluate_solvent_model()
