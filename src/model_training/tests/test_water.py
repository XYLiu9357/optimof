"""test_water.py
Utility script for evaluating performance of the water stability model.
Requires a test set that was not used in training.
"""

import joblib

from src.config.paths import MODEL_DIR, WATER_RF_PERFORMANCE_DIR
from src.model_training.base.evaluator import MultiClassEvaluator
from src.model_training.water_stability_model import WaterStabilityPipeline  # For unpickling


def evaluate_water_model(model_type: str = "rf"):
    """Load water stability model and evaluate on test set.

    Args:
        model_type: Type of model to evaluate ('rf' or 'xgboost')
    """
    print(f"**Testing water stability {model_type.upper()} model accuracy**")

    # Load model path
    if model_type == "rf":
        model_path = MODEL_DIR / "water_rf_model.pkl"
    else:
        model_path = MODEL_DIR / "water_boost_model.pkl"

    # Load pipeline (which contains model and test data)
    with open(model_path, "rb") as f:
        pipeline = joblib.load(f)

    # Get test data
    test_features, test_labels = pipeline.get_test_data()

    # Make predictions
    predictions = pipeline.model.predict(test_features)
    probabilities = pipeline.model.predict_proba(test_features)

    # Get class labels
    class_labels = pipeline.model.classes_

    # Evaluate
    evaluator = MultiClassEvaluator(class_labels)
    metrics = evaluator.calculate_metrics(
        test_labels.values, predictions, probabilities
    )
    evaluator.plot_results(
        test_labels.values, predictions, probabilities, pipeline.performance_dir
    )

    return metrics


if __name__ == "__main__":
    # Test Random Forest model
    print("Random Forest Performance:")
    evaluate_water_model(model_type="rf")

    # Uncomment to test XGBoost model
    # print("\nXGBoost Performance:")
    # evaluate_water_model(model_type="xgboost")
