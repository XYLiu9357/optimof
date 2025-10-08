"""Sklearn model hyperparameter tuner."""

from pathlib import Path
from typing import Dict

import optuna

from src.model_training.tune.base_tuner import BaseTuner


class SklearnTuner(BaseTuner):
    """Hyperparameter tuner for sklearn models.

    Attributes:
        features: Feature dataframe
        labels: Label series
        pipeline_class: Pipeline class to use for training
        model_type_name: Name of model type
        base_model_type: Type of model ('rf' or 'xgboost')
    """

    def __init__(
        self,
        features,
        labels,
        pipeline_class,
        model_type_name: str,
        base_model_type: str,
        study_name: str,
        n_trials: int = 50,
        timeout: int = None,
    ):
        """Initialize sklearn tuner.

        Args:
            features: Feature dataframe for training
            labels: Label series for training
            pipeline_class: Pipeline class (WaterStabilityPipeline)
            model_type_name: Model type string (e.g., 'water_rf')
            base_model_type: Base model type ('rf' or 'xgboost')
            study_name: Name for Optuna study
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
        """
        super().__init__(
            study_name=study_name,
            n_trials=n_trials,
            timeout=timeout,
            direction="maximize",  # Maximize accuracy for classification
        )

        self.features = features
        self.labels = labels
        self.pipeline_class = pipeline_class
        self.model_type_name = model_type_name
        self.base_model_type = base_model_type

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Best cross-validation accuracy achieved
        """
        # Sample hyperparameters based on model type
        if self.base_model_type == "rf":
            param_grid = {
                "n_estimators": [
                    trial.suggest_int("n_estimators", 100, 300, step=50)
                ],
                "max_depth": [trial.suggest_categorical("max_depth", [None, 10, 20, 30])],
                "min_samples_split": [
                    trial.suggest_int("min_samples_split", 2, 10, step=2)
                ],
                "min_samples_leaf": [
                    trial.suggest_int("min_samples_leaf", 1, 4)
                ],
                "bootstrap": [trial.suggest_categorical("bootstrap", [True, False])],
            }
        else:  # xgboost
            param_grid = {
                "n_estimators": [
                    trial.suggest_int("n_estimators", 100, 300, step=50)
                ],
                "max_depth": [trial.suggest_int("max_depth", 3, 10)],
                "learning_rate": [
                    trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                ],
                "subsample": [trial.suggest_float("subsample", 0.6, 1.0)],
                "colsample_bytree": [
                    trial.suggest_float("colsample_bytree", 0.6, 1.0)
                ],
            }

        # Train model and get cross-validation score
        try:
            pipeline = self.pipeline_class(model_type=self.base_model_type)

            # Override param_grid with trial's suggestions
            pipeline.param_grid = param_grid

            # Prepare data
            pipeline.prepare_data(self.features, self.labels)

            # Train with grid search (cv=5)
            pipeline.train(cv=5, n_jobs=1, verbose=0)

            # Get test accuracy
            predictions = pipeline.model.predict(pipeline.test_features)
            accuracy = (predictions == pipeline.test_labels).mean()

            # Report intermediate values for pruning
            trial.report(accuracy, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return accuracy

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a low value to mark as failed
            return 0.0

    def get_model_type(self) -> str:
        """Get model type string.

        Returns:
            Model type identifier
        """
        return self.model_type_name

    def _extract_architecture_params(self, trial: optuna.Trial) -> Dict:
        """Extract architecture parameters from best trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of architecture parameters
        """
        # For sklearn models, architecture is determined by the model type
        return {
            "model_type": self.base_model_type,
        }

    def _extract_training_params(self, trial: optuna.Trial) -> Dict:
        """Extract training parameters from best trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of training parameters
        """
        # Return the hyperparameters from the trial
        return dict(trial.params)
