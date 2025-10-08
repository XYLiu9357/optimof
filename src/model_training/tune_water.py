"""Hyperparameter tuning script for water stability model."""

import argparse

import joblib

from src.config.constants import WATER_LABEL
from src.config.paths import WATER_DATA_DIR
from src.model_training.water_stability_model import WaterStabilityPipeline
from src.model_training.tune.sklearn_tuner import SklearnTuner


def main():
    """Run hyperparameter tuning for water stability model."""
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for water stability model"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="rf",
        choices=["rf", "xgboost"],
        help="Type of model to tune: 'rf' or 'xgboost' (default: rf)",
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

    args = parser.parse_args()

    # File paths
    data_path = WATER_DATA_DIR / "water_clean_data.pkl"

    # Load data
    df_clean = joblib.load(data_path)
    features = df_clean.loc[:, df_clean.columns != WATER_LABEL]
    labels = df_clean.loc[:, WATER_LABEL]

    # Set study name based on model type
    study_name = f"water_{args.model_type}"

    print(f"\n{'=' * 60}")
    print(f"Water Stability Hyperparameter Tuning ({args.model_type.upper()})")
    print(f"{'=' * 60}")
    print(f"Data path: {data_path}")
    print(f"Model type: {args.model_type}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Timeout: {args.timeout}s" if args.timeout else "Timeout: None")
    print(f"{'=' * 60}\n")

    # Create tuner
    tuner = SklearnTuner(
        features=features,
        labels=labels,
        pipeline_class=WaterStabilityPipeline,
        model_type_name=f"water_{args.model_type}",
        base_model_type=args.model_type,
        study_name=study_name,
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
