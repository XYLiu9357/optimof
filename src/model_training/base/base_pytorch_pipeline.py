"""Base pipeline for PyTorch model training."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config.constants import RANDOM_STATE, TEST_SIZE, VAL_SIZE
from src.model_training.base.config import ModelConfig, TrainingConfig
from src.model_training.base.data_processor import DataProcessor


class BasePyTorchPipeline(ABC):
    """Abstract base class for PyTorch model pipelines.

    Attributes:
        model_config: Model architecture configuration
        training_config: Training hyperparameter configuration
        label_col: Name of the label column
        data_processor: DataProcessor instance for data operations
        device: PyTorch device (CPU or CUDA)
        model: The neural network model
        train_features: Training features
        test_features: Test features
        train_labels: Training labels
        test_labels: Test labels
    """

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        label_col: str,
    ) -> None:
        """Initialize pipeline.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameter configuration
            label_col: Name of the label column in the dataframe
        """
        self.model_config = model_config
        self.training_config = training_config
        self.label_col = label_col

        self.data_processor = DataProcessor(
            label_col=label_col, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None

        self.train_features: Optional[pd.DataFrame] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.train_labels: Optional[pd.Series] = None
        self.test_labels: Optional[pd.Series] = None

    def prepare_data(self, data_path: Path, scaler_path: Path) -> None:
        """Load, split, and scale data.

        Args:
            data_path: Path to data pickle file
            scaler_path: Path to save/load the scaler
        """
        # Load data
        df = self.data_processor.load_data(data_path)

        # Split into train and test
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = self.data_processor.split_data(df)

        # Keep unscaled test features for saving
        self.test_features_unscaled = self.test_features.copy()

        # Fit and save scaler
        self.data_processor.fit_scaler(self.train_features)
        self.data_processor.save_scaler(scaler_path)

        # Transform features
        self.train_features = self.data_processor.transform(self.train_features)
        self.test_features = self.data_processor.transform(self.test_features)

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model.

        Must be implemented by subclasses.

        Returns:
            PyTorch model instance
        """
        pass

    def train(self, val_size: float = VAL_SIZE) -> float:
        """Train the model.

        Args:
            val_size: Fraction of training data to use for validation

        Returns:
            Best validation loss achieved during training
        """
        if self.model is None:
            raise ValueError("Model must be built before training")

        if self.train_features is None or self.train_labels is None:
            raise ValueError("Data must be prepared before training")

        # Split training data into train and validation
        train_features, val_features, train_labels, val_labels = train_test_split(
            self.train_features,
            self.train_labels,
            test_size=val_size,
            random_state=RANDOM_STATE,
        )

        # Ensure all data is numeric
        train_features = train_features.apply(pd.to_numeric, errors="raise")
        val_features = val_features.apply(pd.to_numeric, errors="raise")
        train_labels = train_labels.apply(pd.to_numeric, errors="raise")
        val_labels = val_labels.apply(pd.to_numeric, errors="raise")

        # Prepare tensor datasets
        tensor_train_features = torch.tensor(train_features.values, dtype=torch.float32)
        tensor_val_features = torch.tensor(val_features.values, dtype=torch.float32)
        tensor_train_labels = torch.tensor(
            train_labels.values, dtype=torch.float32
        ).unsqueeze(1)
        tensor_val_labels = torch.tensor(
            val_labels.values, dtype=torch.float32
        ).unsqueeze(1)

        train_dataset = TensorDataset(tensor_train_features, tensor_train_labels)
        val_dataset = TensorDataset(tensor_val_features, tensor_val_labels)

        # Build data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.training_config.batch_size, shuffle=False
        )

        # Loss function and optimizer
        criterion = self._get_loss_function()

        # Select optimizer based on config
        if self.training_config.optimizer == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(), lr=self.training_config.learning_rate
            )
        else:  # default to adam
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.training_config.learning_rate
            )

        # Early stopping
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model = None

        print("**Starting Training**")

        # Training loop
        for epoch in range(self.training_config.num_epochs):
            # Train
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                cur_loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    cur_loss = criterion(outputs, y_batch)
                    val_loss += cur_loss.item()

            val_loss /= len(val_loader.dataset)
            print(
                f"Epoch {epoch+1}/{self.training_config.num_epochs}, "
                f"Loss: {cur_loss.item():.4f}, Val Loss: {val_loss:.4f}"
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.training_config.patience:
                    print("Early stopping")
                    break

        # Load best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
        else:
            print("No improvement during training")

        print("**Training Complete**")

        return best_val_loss

    @abstractmethod
    def _get_loss_function(self) -> nn.Module:
        """Get the loss function for this model.

        Must be implemented by subclasses.

        Returns:
            PyTorch loss function
        """
        pass

    def save(self, model_path: Path, test_data_path: Path) -> None:
        """Save model and test data.

        Args:
            model_path: Path to save the model
            test_data_path: Path to save test data
        """
        if self.model is None:
            raise ValueError("Model must be built before saving")

        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, model_path)
        print(f"**Model saved at {model_path}**")

        # Save test data (unscaled so test script can scale it)
        test_data_path.parent.mkdir(parents=True, exist_ok=True)
        test_all = pd.concat([self.test_labels, self.test_features_unscaled], axis=1)
        test_all.to_pickle(test_data_path)
        print(f"**Test data saved at {test_data_path}**")

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test features and labels.

        Returns:
            Tuple of (test_features, test_labels)
        """
        if self.test_features is None or self.test_labels is None:
            raise ValueError("Data must be prepared first")

        return self.test_features, self.test_labels

    @classmethod
    def load_model(cls, model_path: Path, device: Optional[torch.device] = None) -> nn.Module:
        """Load a trained model.

        Args:
            model_path: Path to the saved model
            device: Device to load the model on (default: auto-detect)

        Returns:
            Loaded PyTorch model
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = torch.load(model_path, weights_only=False)
        model.eval()
        model.to(device)
        return model
