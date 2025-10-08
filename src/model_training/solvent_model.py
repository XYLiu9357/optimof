"""solvent_model.py
Main model for MOF solvent removal stability prediction. Feature extraction procedures
should be invoked prior to running this training script.

Task: Binary classification
Predictor: MOF geometric information found using Zeo++
Predictand: MOF stability upon solvent removal

* The model only uses fully connected layers for simplification.
"""

from pathlib import Path
from typing import List

import pandas as pd
import torch.nn as nn

from src.config.constants import SOLVENT_LABEL
from src.config.paths import (
    MODEL_DIR,
    SCALER_DIR,
    SOLVENT_DATA_DIR,
)
from src.model_training.base.base_pytorch_pipeline import BasePyTorchPipeline
from src.model_training.base.config import ModelConfig, TrainingConfig


class SolventModel(nn.Module):
    """Solvent ANN Model.

    ANN model that inherits PyTorch neural network module.

    Attributes:
        graph_depth: int, depth of the computation graph
        layers: List[nn.Linear], a list of network layers used in prediction
        dropout: Dropout layer for regularization
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int,
        dropout_prob: float = 0.2,
    ):
        super(SolventModel, self).__init__()

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.graph_depth: int = len(layer_sizes)
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(self.graph_depth - 1)
            ]
        )

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # No ReLU at the last layer
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
                x = self.dropout(x)
        return x  # Raw logits for BCEWithLogitsLoss


class SolventPipeline(BasePyTorchPipeline):
    """Pipeline for training solvent removal stability prediction model."""

    def __init__(
        self, model_config: ModelConfig, training_config: TrainingConfig
    ) -> None:
        """Initialize solvent model pipeline.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameter configuration
        """
        super().__init__(model_config, training_config, label_col=SOLVENT_LABEL)

    def prepare_data(self, data_path: Path, scaler_path: Path) -> None:
        """Load, split, scale data, and normalize labels.

        Args:
            data_path: Path to data pickle file
            scaler_path: Path to save/load the scaler
        """
        # Call parent method for basic data preparation
        super().prepare_data(data_path, scaler_path)

        # Normalize labels: convert {-1, 1} to {0, 1}
        self.train_labels = self._normalize_labels(self.train_labels)
        self.test_labels = self._normalize_labels(self.test_labels)

    def _normalize_labels(self, labels: pd.Series) -> pd.Series:
        """Convert {-1, 1} labels to {0, 1} labels.

        Args:
            labels: Raw labels

        Returns:
            Normalized labels
        """
        normalized = labels.copy()
        normalized[normalized < 0] = 0
        return normalized

    def build_model(self) -> nn.Module:
        """Build the solvent model.

        Returns:
            SolventModel instance
        """
        # Validate config
        assert self.model_config.output_size == 1, "Solvent model output size must be 1"
        if self.train_features is not None:
            assert (
                self.model_config.input_size == self.train_features.shape[1]
            ), f"Input size mismatch: config={self.model_config.input_size}, data={self.train_features.shape[1]}"

        self.model = SolventModel(
            self.model_config.input_size,
            self.model_config.hidden_layers,
            self.model_config.output_size,
            self.model_config.dropout_prob,
        )
        self.model.to(self.device)
        return self.model

    def _get_loss_function(self) -> nn.Module:
        """Get BCE with logits loss for binary classification.

        Returns:
            BCEWithLogitsLoss function
        """
        return nn.BCEWithLogitsLoss()


if __name__ == "__main__":
    # File paths
    data_file_path = SOLVENT_DATA_DIR / "solvent_clean_data.pkl"
    hyperparam_file_path = MODEL_DIR / "solvent_hyperparams.json"
    model_file_path = MODEL_DIR / "solvent_model.pkl"
    scaler_file_path = SCALER_DIR / "solvent_scaler.pkl"
    test_data_path = SOLVENT_DATA_DIR / "solvent_test_data.pkl"

    # Load configurations
    print("**Reading hyperparameter config**")
    model_config = ModelConfig.from_json(hyperparam_file_path)
    training_config = TrainingConfig.from_json(hyperparam_file_path)

    # Create pipeline
    pipeline = SolventPipeline(model_config, training_config)

    # Prepare data
    pipeline.prepare_data(data_file_path, scaler_file_path)

    # Build model
    pipeline.build_model()

    # Train model
    pipeline.train()

    # Save model and test data
    pipeline.save(model_file_path, test_data_path)
