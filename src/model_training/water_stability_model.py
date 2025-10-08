"""water_stability_model.py
Main model for MOF water stability prediction. Feature extraction procedures
should be invoked prior to running this training script.

Task: 4-class classification
Predictor: MOF geometric information found using Zeo++
Predictand: MOF water stability classification
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from src.config.constants import RANDOM_STATE, WATER_LABEL, WATER_LABEL_OFFSET
from src.config.paths import (
    MODEL_DIR,
    WATER_DATA_DIR,
    WATER_BOOST_PERFORMANCE_DIR,
    WATER_RF_PERFORMANCE_DIR,
)
from src.model_training.base.base_sklearn_pipeline import BaseSklearnPipeline
from src.model_training.base.evaluator import MultiClassEvaluator


class WaterStabilityPipeline(BaseSklearnPipeline):
    """Pipeline for training water stability prediction models.

    Supports both Random Forest and XGBoost models.
    """

    def __init__(self, model_type: str = "rf") -> None:
        """Initialize water stability pipeline.

        Args:
            model_type: Type of model to train ('rf' or 'xgboost')
        """
        if model_type not in ["rf", "xgboost"]:
            raise ValueError("model_type must be 'rf' or 'xgboost'")

        # Set paths based on model type
        if model_type == "rf":
            model_save_path = MODEL_DIR / "water_rf_model.pkl"
            test_data_save_path = WATER_DATA_DIR / "water_rf_test_data.pkl"
            performance_dir = WATER_RF_PERFORMANCE_DIR
            param_grid = {
                "n_estimators": [100, 150, 200, 250, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
        else:  # xgboost
            model_save_path = MODEL_DIR / "water_boost_model.pkl"
            test_data_save_path = WATER_DATA_DIR / "water_boost_test_data.pkl"
            performance_dir = WATER_BOOST_PERFORMANCE_DIR
            param_grid = {
                "n_estimators": [100, 150, 200, 250, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

        super().__init__(
            label_col=WATER_LABEL,
            model_type=model_type,
            param_grid=param_grid,
            model_save_path=model_save_path,
            test_data_save_path=test_data_save_path,
            performance_dir=performance_dir,
        )

    def prepare_data(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> None:
        """Split data and normalize labels.

        Args:
            features: Feature dataframe
            labels: Label series
        """
        # Normalize labels: {1, 2, 3, 4} -> {0, 1, 2, 3}
        normalized_labels = labels - WATER_LABEL_OFFSET

        # Call parent method
        super().prepare_data(features, normalized_labels)

    def build_model(self):
        """Build the model based on model_type.

        Returns:
            Sklearn model instance (RandomForestClassifier or XGBClassifier)
        """
        if self.model_type == "rf":
            return RandomForestClassifier(random_state=RANDOM_STATE)
        elif self.model_type == "xgboost":
            return xgb.XGBClassifier(
                objective="multi:softprob", random_state=RANDOM_STATE
            )

    def evaluate(self) -> None:
        """Evaluate model performance and generate plots."""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        if self.test_features is None or self.test_labels is None:
            raise ValueError("Test data must be prepared before evaluation")

        # Make predictions
        predictions = self.model.predict(self.test_features)
        probabilities = self.model.predict_proba(self.test_features)

        # Get class labels
        class_labels = self.model.classes_

        # Create evaluator and calculate metrics
        evaluator = MultiClassEvaluator(class_labels)
        metrics = evaluator.calculate_metrics(
            self.test_labels.values, predictions, probabilities
        )

        # Generate plots
        evaluator.plot_results(
            self.test_labels.values, predictions, probabilities, self.performance_dir
        )

        return metrics


if __name__ == "__main__":
    # Load data
    data_path = WATER_DATA_DIR / "water_clean_data.pkl"
    df_clean = joblib.load(data_path)

    features = df_clean.loc[:, df_clean.columns != WATER_LABEL]
    labels = df_clean.loc[:, WATER_LABEL]

    # Train Random Forest model
    print("**Training Random Forest Model**")
    rf_pipeline = WaterStabilityPipeline(model_type="rf")
    rf_pipeline.prepare_data(features, labels)
    rf_pipeline.train()
    rf_pipeline.save()

    print("\nRandom Forest Performance:")
    rf_pipeline.evaluate()
    print("**End of Random Forest Training**")

    # Optionally train XGBoost model
    # Uncomment below to train XGBoost model as well
    # print("\n**Training XGBoost Model**")
    # xgb_pipeline = WaterStabilityPipeline(model_type="xgboost")
    # xgb_pipeline.prepare_data(features, labels)
    # xgb_pipeline.train()
    # xgb_pipeline.save()
    #
    # print("\nXGBoost Performance:")
    # xgb_pipeline.evaluate()
    # print("**End of XGBoost Training**")
