"""thermal_model.py
Main model for MOF thermal stability prediction. Feature extraction procedures should be
invoked prior to running this training script.

Task: Regression
Predictor: MOF geometric information found using Zeo++
Predictand: MOF breakdown temperature

* The model only uses fully connected layers for simplification.
"""

from pathlib import Path
from typing import List

import torch.nn as nn

from src.config.constants import THERMAL_LABEL
from src.config.paths import (
    MODEL_DIR,
    SCALER_DIR,
    THERMAL_DATA_DIR,
)
from src.model_training.base.base_pytorch_pipeline import BasePyTorchPipeline
from src.model_training.base.config import ModelConfig, TrainingConfig


class ThermalModel(nn.Module):
    """Thermal ANN Model.

    ANN model that inherits PyTorch neural network module.

    Attributes:
        graph_depth: int, depth of the computation graph
        layers: List[nn.Linear], a list of network layers used in prediction
    """

    def __init__(
        self, input_size: int, hidden_layer_sizes: List[int], output_size: int
    ):
        super(ThermalModel, self).__init__()

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.graph_depth: int = len(layer_sizes)
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(self.graph_depth - 1)
            ]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # No ReLU at the last layer
            if i < len(self.layers) - 1:
                x = nn.functional.leaky_relu(x)
        return x


class ThermalPipeline(BasePyTorchPipeline):
    """Pipeline for training thermal stability prediction model."""

    def __init__(
        self, model_config: ModelConfig, training_config: TrainingConfig
    ) -> None:
        """Initialize thermal model pipeline.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameter configuration
        """
        super().__init__(model_config, training_config, label_col=THERMAL_LABEL)

    def build_model(self) -> nn.Module:
        """Build the thermal model.

        Returns:
            ThermalModel instance
        """
        # Validate config
        assert self.model_config.output_size == 1, "Thermal model output size must be 1"
        if self.train_features is not None:
            assert (
                self.model_config.input_size == self.train_features.shape[1]
            ), f"Input size mismatch: config={self.model_config.input_size}, data={self.train_features.shape[1]}"

        self.model = ThermalModel(
            self.model_config.input_size,
            self.model_config.hidden_layers,
            self.model_config.output_size,
        )
        self.model.to(self.device)
        return self.model

    def _get_loss_function(self) -> nn.Module:
        """Get MSE loss function for regression.

        Returns:
            MSE loss function
        """
        return nn.MSELoss()


if __name__ == "__main__":
    # File paths
    data_file_path = THERMAL_DATA_DIR / "thermal_clean_data.pkl"
    hyperparam_file_path = MODEL_DIR / "thermal_hyperparams.json"
    model_file_path = MODEL_DIR / "thermal_model.pkl"
    scaler_file_path = SCALER_DIR / "thermal_scaler.pkl"
    test_data_path = THERMAL_DATA_DIR / "thermal_test_data.pkl"

    # Load configurations
    print("**Reading hyperparameter config**")
    model_config = ModelConfig.from_json(hyperparam_file_path)
    training_config = TrainingConfig.from_json(hyperparam_file_path)

    # Create pipeline
    pipeline = ThermalPipeline(model_config, training_config)

    # Prepare data
    pipeline.prepare_data(data_file_path, scaler_file_path)

    # Build model
    pipeline.build_model()

    # Train model
    pipeline.train()

    # Save model and test data
    pipeline.save(model_file_path, test_data_path)
