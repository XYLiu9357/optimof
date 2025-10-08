"""Hyperparameter tuning script for solvent removal stability model."""

import argparse

from src.config.paths import MODEL_DIR, SCALER_DIR, SOLVENT_DATA_DIR
from src.model_training.solvent_model import SolventPipeline
from src.model_training.tune.pytorch_tuner import PyTorchTuner


def main():
    """Run hyperparameter tuning for solvent removal stability model."""
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for solvent removal stability model"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials to run (default: 50)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum time in seconds (default: None)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="solvent_pytorch",
        help="Name for the Optuna study (default: solvent_pytorch)",
    )

    args = parser.parse_args()

    # File paths
    data_path = SOLVENT_DATA_DIR / "solvent_clean_data.pkl"
    scaler_path = SCALER_DIR / "solvent_scaler.pkl"

    # Determine input size from data
    import joblib

    df = joblib.load(data_path)
    input_size = df.shape[1] - 1  # Subtract 1 for label column

    print(f"\n{'=' * 60}")
    print("Solvent Removal Stability Hyperparameter Tuning")
    print(f"{'=' * 60}")
    print(f"Data path: {data_path}")
    print(f"Scaler path: {scaler_path}")
    print(f"Input size: {input_size}")
    print(f"Output size: 1 (binary classification)")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Timeout: {args.timeout}s" if args.timeout else "Timeout: None")
    print(f"{'=' * 60}\n")

    # Create tuner
    tuner = PyTorchTuner(
        data_path=data_path,
        scaler_path=scaler_path,
        label_col="solvent",
        pipeline_class=SolventPipeline,
        input_size=input_size,
        output_size=1,
        model_type_name="pytorch_binary_classification",
        study_name=args.study_name,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    # Run tuning
    structure_path = tuner.tune()

    print(f"\n{'=' * 60}")
    print("Tuning Complete!")
    print(f"{'=' * 60}")
    print(f"Best structure saved to: {structure_path}")
    print(f"You can now train a model using this structure.")
    print(f"{'=' * 60}\n")

    # Print statistics
    tuner.print_study_statistics()


if __name__ == "__main__":
    main()
