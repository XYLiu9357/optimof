"""Configuration dataclasses for model training."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    input_size: int
    hidden_layers: List[int]
    output_size: int
    dropout_prob: float = 0.2
    arch_type: str = "simple"  # 'simple', 'batchnorm', or 'residual'
    activation: str = "leaky_relu"  # 'relu', 'leaky_relu', or 'elu'

    @classmethod
    def from_json(cls, json_path: Path) -> "ModelConfig":
        """Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            ModelConfig instance
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        # Extract only ModelConfig fields
        model_config = {
            "input_size": config_dict["input_size"],
            "hidden_layers": config_dict["hidden_layers"],
            "output_size": config_dict["output_size"],
        }

        # Add optional fields if present
        for optional_field in ["dropout_prob", "arch_type", "activation"]:
            if optional_field in config_dict:
                model_config[optional_field] = config_dict[optional_field]

        return cls(**model_config)

    def to_json(self, json_path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            json_path: Path to save JSON configuration
        """
        config_dict = {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "dropout_prob": self.dropout_prob,
        }
        with open(json_path, "w") as f:
            json.dump(config_dict, f, indent=4)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float
    batch_size: int
    num_epochs: int
    patience: int
    optimizer: str = "adam"  # 'adam' or 'adamw'

    @classmethod
    def from_json(cls, json_path: Path) -> "TrainingConfig":
        """Load configuration from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            TrainingConfig instance
        """
        with open(json_path, "r") as f:
            config_dict = json.load(f)

        training_config = {
            "learning_rate": config_dict["learning_rate"],
            "batch_size": config_dict["batch_size"],
            "num_epochs": config_dict["num_epochs"],
            "patience": config_dict["patience"],
        }

        # Add optional optimizer field
        if "optimizer" in config_dict:
            training_config["optimizer"] = config_dict["optimizer"]

        return cls(**training_config)
