"""Base class for hyperparameter tuning."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import optuna


class BaseTuner(ABC):
    """Abstract base class for hyperparameter tuning.

    Attributes:
        study_name: Name of the Optuna study
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        study: Optuna study object
    """

    def __init__(
        self,
        study_name: str,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        direction: str = "minimize",
    ):
        """Initialize tuner.

        Args:
            study_name: Name for the study (used for database)
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (None for no limit)
            direction: 'minimize' or 'maximize'
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout

        # Create study with persistent storage
        storage_path = Path("model/structure/optuna_studies") / f"{study_name}.db"
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,  # Resume if exists
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            ),
        )

    @abstractmethod
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to optimize.

        Must be implemented by subclasses.

        Args:
            trial: Optuna trial object

        Returns:
            Value to optimize (e.g., validation loss)
        """
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Get model type string.

        Returns:
            Model type identifier
        """
        pass

    def tune(self) -> Path:
        """Run hyperparameter optimization.

        Returns:
            Path to saved structure file
        """
        print(f"\n{'='*60}")
        print(f"Starting Hyperparameter Tuning: {self.study_name}")
        print(f"{'='*60}")
        print(f"Trials: {self.n_trials}")
        print(f"Timeout: {self.timeout}s" if self.timeout else "Timeout: None")
        print(f"{'='*60}\n")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            n_jobs=1,  # Sequential for GPU safety
        )

        print(f"\n{'='*60}")
        print("Optimization Complete!")
        print(f"{'='*60}")
        print(f"Best trial: #{self.study.best_trial.number}")
        print(f"Best value: {self.study.best_trial.value:.4f}")
        print(f"{'='*60}\n")

        # Save best structure
        structure_path = self.save_best_structure()
        print(f"Best structure saved to: {structure_path}\n")

        return structure_path

    def save_best_structure(self) -> Path:
        """Save best trial configuration to JSON file.

        Returns:
            Path to saved structure file
        """
        best_trial = self.study.best_trial

        structure = {
            "model_type": self.get_model_type(),
            "architecture": self._extract_architecture_params(best_trial),
            "training": self._extract_training_params(best_trial),
            "tuning_metadata": {
                "best_trial": best_trial.number,
                "best_value": best_trial.value,
                "n_trials": len(self.study.trials),
                "study_name": self.study_name,
                "date": datetime.now().isoformat(),
                "params": best_trial.params,
            },
        }

        # Save to structure file
        structure_dir = Path("model/structure")
        structure_dir.mkdir(parents=True, exist_ok=True)
        structure_path = structure_dir / f"{self.study_name}_structure.json"

        with open(structure_path, "w") as f:
            json.dump(structure, f, indent=2)

        return structure_path

    @abstractmethod
    def _extract_architecture_params(self, trial: optuna.Trial) -> Dict:
        """Extract architecture parameters from trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of architecture parameters
        """
        pass

    @abstractmethod
    def _extract_training_params(self, trial: optuna.Trial) -> Dict:
        """Extract training parameters from trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of training parameters
        """
        pass

    def print_study_statistics(self):
        """Print statistics about the study."""
        print(f"\nStudy Statistics:")
        print(f"  Number of finished trials: {len(self.study.trials)}")
        print(f"  Number of pruned trials: {len(self.study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
        print(f"  Number of complete trials: {len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}")

        if len(self.study.trials) > 0:
            print(f"\nBest trial:")
            trial = self.study.best_trial
            print(f"  Value: {trial.value}")
            print(f"  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
