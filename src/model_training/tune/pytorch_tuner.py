"""PyTorch model hyperparameter tuner."""

from pathlib import Path
from typing import Dict

import optuna

from src.model_training.base.config import ModelConfig, TrainingConfig
from src.model_training.tune.base_tuner import BaseTuner


class PyTorchTuner(BaseTuner):
    """Hyperparameter tuner for PyTorch models.

    Attributes:
        data_path: Path to training data
        scaler_path: Path to save scaler
        label_col: Label column name
        pipeline_class: Pipeline class to use for training
        input_size: Input feature size
        output_size: Output size
        model_type_name: Name of model type
    """

    def __init__(
        self,
        data_path: Path,
        scaler_path: Path,
        label_col: str,
        pipeline_class,
        input_size: int,
        output_size: int,
        model_type_name: str,
        study_name: str,
        n_trials: int = 50,
        timeout: int = None,
    ):
        """Initialize PyTorch tuner.

        Args:
            data_path: Path to training data pickle
            scaler_path: Path to save scaler
            label_col: Name of label column
            pipeline_class: Pipeline class (ThermalPipeline or SolventPipeline)
            input_size: Size of input features
            output_size: Size of output (1 for regression/binary classification)
            model_type_name: Model type string (e.g., 'pytorch_regression')
            study_name: Name for Optuna study
            n_trials: Number of trials to run
            timeout: Maximum time in seconds
        """
        super().__init__(
            study_name=study_name,
            n_trials=n_trials,
            timeout=timeout,
            direction="minimize",
        )

        self.data_path = data_path
        self.scaler_path = scaler_path
        self.label_col = label_col
        self.pipeline_class = pipeline_class
        self.input_size = input_size
        self.output_size = output_size
        self.model_type_name = model_type_name

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Best validation loss achieved
        """
        # Sample architecture hyperparameters
        arch_type = trial.suggest_categorical(
            "arch_type", ["simple", "batchnorm", "residual"]
        )
        n_layers = trial.suggest_int("n_layers", 2, 7)

        hidden_layers = []
        for i in range(n_layers):
            size = trial.suggest_categorical(
                f"layer_{i}_size", [64, 128, 256, 512, 1024, 2048]
            )
            hidden_layers.append(size)

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        activation = trial.suggest_categorical(
            "activation", ["relu", "leaky_relu", "elu"]
        )

        # Sample training hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])

        # Create configurations
        model_config = ModelConfig(
            input_size=self.input_size,
            hidden_layers=hidden_layers,
            output_size=self.output_size,
            dropout_prob=dropout,
            arch_type=arch_type,
            activation=activation,
        )

        training_config = TrainingConfig(
            learning_rate=lr,
            batch_size=batch_size,
            num_epochs=50,  # Shorter for tuning
            patience=10,  # Shorter patience for tuning
            optimizer=optimizer,
        )

        # Train model and get validation loss
        try:
            pipeline = self.pipeline_class(model_config, training_config)
            pipeline.prepare_data(self.data_path, self.scaler_path)
            pipeline.build_model()
            best_val_loss = pipeline.train()

            # Report intermediate values for pruning
            trial.report(best_val_loss, step=0)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            return best_val_loss

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            # Return a high value to mark as failed
            return float("inf")

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
        n_layers = trial.params["n_layers"]
        hidden_layers = [trial.params[f"layer_{i}_size"] for i in range(n_layers)]

        return {
            "input_size": self.input_size,
            "hidden_layers": hidden_layers,
            "output_size": self.output_size,
            "dropout_prob": trial.params["dropout"],
            "arch_type": trial.params["arch_type"],
            "activation": trial.params["activation"],
        }

    def _extract_training_params(self, trial: optuna.Trial) -> Dict:
        """Extract training parameters from best trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of training parameters
        """
        return {
            "learning_rate": trial.params["learning_rate"],
            "batch_size": trial.params["batch_size"],
            "num_epochs": 1000,  # Full training will use more epochs
            "patience": 100,  # Full training will use more patience
            "optimizer": trial.params["optimizer"],
        }
