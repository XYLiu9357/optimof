"""Data processing utilities for model training."""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """Handles data loading, splitting, and scaling operations.

    Attributes:
        label_col: Name of the label column
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        scaler: StandardScaler instance for feature normalization
    """

    def __init__(
        self, label_col: str, test_size: float = 0.2, random_state: int = 0
    ) -> None:
        """Initialize DataProcessor.

        Args:
            label_col: Name of the label column in the dataframe
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility (default: 0)
        """
        self.label_col = label_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data from pickle file.

        Args:
            data_path: Path to pickle file containing the data

        Returns:
            Loaded dataframe
        """
        return joblib.load(data_path)

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.

        Args:
            df: Input dataframe containing features and labels

        Returns:
            Tuple of (train_features, test_features, train_labels, test_labels)
        """
        features = df.loc[:, df.columns != self.label_col]
        labels = df.loc[:, self.label_col]

        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=self.test_size, random_state=self.random_state
        )

        # Reset indices to ensure alignment
        test_features = test_features.reset_index(drop=True)
        test_labels = test_labels.reset_index(drop=True)

        return train_features, test_features, train_labels, test_labels

    def fit_scaler(self, train_features: pd.DataFrame) -> None:
        """Fit scaler to training features.

        Args:
            train_features: Training feature dataframe
        """
        self.scaler = StandardScaler()
        self.scaler.fit(train_features)

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler.

        Args:
            features: Feature dataframe to transform

        Returns:
            Transformed feature dataframe

        Raises:
            ValueError: If scaler has not been fitted
        """
        if self.scaler is None:
            raise ValueError("Scaler must be fitted before transforming data")

        transformed_array = self.scaler.transform(features)
        return pd.DataFrame(transformed_array, columns=features.columns)

    def save_scaler(self, scaler_path: Path) -> None:
        """Save fitted scaler to file.

        Args:
            scaler_path: Path to save the scaler

        Raises:
            ValueError: If scaler has not been fitted
        """
        if self.scaler is None:
            raise ValueError("Scaler must be fitted before saving")

        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)

    def load_scaler(self, scaler_path: Path) -> None:
        """Load scaler from file.

        Args:
            scaler_path: Path to load the scaler from
        """
        self.scaler = joblib.load(scaler_path)
