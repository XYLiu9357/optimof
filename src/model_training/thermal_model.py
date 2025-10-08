"""thermal_model.py
Main model for MOF thermal stability prediction. Feature extraction procedures should be
invoked prior to running this training script.

Task: Regression
Predictor: MOF geometric information found using Zeo++
Predictand: MOF breakdown temperature
"""

from pathlib import Path

import torch.nn as nn

from src.config.constants import THERMAL_LABEL
from src.config.paths import (
    MODEL_DIR,
    SCALER_DIR,
    THERMAL_DATA_DIR,
)
from src.model_training.base.base_pytorch_pipeline import BasePyTorchPipeline
from src.model_training.base.config import ModelConfig, TrainingConfig
from src.model_training.base.flexible_mlp import FlexibleMLP


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
        """Build the thermal model using FlexibleMLP.

        Returns:
            FlexibleMLP instance
        """
        # Validate config
        assert self.model_config.output_size == 1, "Thermal model output size must be 1"
        if self.train_features is not None:
            assert (
                self.model_config.input_size == self.train_features.shape[1]
            ), f"Input size mismatch: config={self.model_config.input_size}, data={self.train_features.shape[1]}"

        self.model = FlexibleMLP(
            input_size=self.model_config.input_size,
            hidden_layers=self.model_config.hidden_layers,
            output_size=self.model_config.output_size,
            dropout_prob=self.model_config.dropout_prob,
            arch_type=self.model_config.arch_type,
            activation=self.model_config.activation,
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
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Train thermal stability prediction model"
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="model/structure/thermal_pytorch_structure.json",
        help="Path to structure file from hyperparameter tuning",
    )
    args = parser.parse_args()

    # File paths
    data_file_path = THERMAL_DATA_DIR / "thermal_clean_data.pkl"
    structure_file_path = Path(args.structure)
    model_file_path = MODEL_DIR / "thermal_model.pkl"
    scaler_file_path = SCALER_DIR / "thermal_scaler.pkl"
    test_data_path = THERMAL_DATA_DIR / "thermal_test_data.pkl"

    # Load configurations from structure file
    print(f"**Reading structure from {structure_file_path}**")
    with open(structure_file_path, "r") as f:
        structure = json.load(f)

    model_config = ModelConfig(**structure["architecture"])
    training_config = TrainingConfig(**structure["training"])

    print(f"Architecture: {structure['architecture']}")
    print(f"Training config: {structure['training']}")

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
