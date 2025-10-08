"""Base pipeline for scikit-learn model training."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config.constants import RANDOM_STATE, TEST_SIZE
from src.model_training.base.data_processor import DataProcessor


class BaseSklearnPipeline(ABC):
    """Abstract base class for scikit-learn model pipelines.

    Attributes:
        label_col: Name of the label column
        model_type: Type of model to train (e.g., 'rf', 'xgboost')
        param_grid: Hyperparameter grid for GridSearchCV
        data_processor: DataProcessor instance for data operations
        model: The trained sklearn model
        train_features: Training features
        test_features: Test features
        train_labels: Training labels
        test_labels: Test labels
        model_save_path: Path to save the model
        test_data_save_path: Path to save test data
        performance_dir: Directory to save performance plots
    """

    def __init__(
        self,
        label_col: str,
        model_type: str,
        param_grid: Dict,
        model_save_path: Path,
        test_data_save_path: Path,
        performance_dir: Path,
    ) -> None:
        """Initialize pipeline.

        Args:
            label_col: Name of the label column in the dataframe
            model_type: Type of model to train
            param_grid: Hyperparameter grid for GridSearchCV
            model_save_path: Path to save the trained model
            test_data_save_path: Path to save test data
            performance_dir: Directory to save performance plots
        """
        self.label_col = label_col
        self.model_type = model_type
        self.param_grid = param_grid

        self.data_processor = DataProcessor(
            label_col=label_col, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        self.model = None
        self.train_features: Optional[pd.DataFrame] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.train_labels: Optional[pd.Series] = None
        self.test_labels: Optional[pd.Series] = None

        self.model_save_path = model_save_path
        self.test_data_save_path = test_data_save_path
        self.performance_dir = performance_dir

    def prepare_data(self, features: pd.DataFrame, labels: pd.Series) -> None:
        """Split data into training and testing sets.

        Note: sklearn models don't require feature scaling for tree-based methods.

        Args:
            features: Feature dataframe
            labels: Label series
        """
        # Create temporary dataframe for splitting
        df = features.copy()
        df[self.label_col] = labels

        # Split data
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = self.data_processor.split_data(df)

    @abstractmethod
    def build_model(self):
        """Build and return the base model.

        Must be implemented by subclasses.

        Returns:
            Sklearn model instance
        """
        pass

    def train(self, cv: int = 5, n_jobs: int = -1, verbose: int = 2) -> None:
        """Train the model with hyperparameter tuning.

        Args:
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 uses all processors)
            verbose: Verbosity level
        """
        if self.train_features is None or self.train_labels is None:
            raise ValueError("Data must be prepared before training")

        # Build base model
        base_model = self.build_model()

        # Run grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Fit model
        print(f"**Starting Training for {self.model_type} model**")
        grid_search.fit(self.train_features, self.train_labels)
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print("**Training Complete**")

    def save(self) -> None:
        """Save the trained model and test data."""
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        # Save model
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.model_save_path)
        print(f"**Model saved at {self.model_save_path}**")

        # Save test data
        self.test_data_save_path.parent.mkdir(parents=True, exist_ok=True)
        test_data = pd.concat([self.test_labels, self.test_features], axis=1)
        test_data.to_pickle(self.test_data_save_path)
        print(f"**Test data saved at {self.test_data_save_path}**")

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test features and labels.

        Returns:
            Tuple of (test_features, test_labels)
        """
        if self.test_features is None or self.test_labels is None:
            raise ValueError("Data must be prepared first")

        return self.test_features, self.test_labels

    @classmethod
    def load_model(cls, model_path: Path) -> "BaseSklearnPipeline":
        """Load a trained pipeline.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded pipeline instance
        """
        with open(model_path, "rb") as f:
            pipeline = joblib.load(f)
        return pipeline
