"""water_stability_model.py
Main model for MOF water stability prediction. Feature extraction procedures 
should be invoked prior to running this training script. 

Task: 4-class classification
Predictor: MOF geometric information found using Zeo++
Predictand: MOF stability upon solvent removal
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


class WaterStabilityRF:
    """
    Trains a random forest model to fit the water stability data set.

    Attributes:
        train_features: feature dataframe for training
        train_labels: label dataframe for training
        test_features: feature dataframe for testing
        test_labels: label dataframe for testing
        model: sklearn random forest model
        model_save_path: path to save model
        fig_save_dir: directory to save test figures
    """

    def __init__(
        self, project_dir: str, features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(features, labels, test_size=0.2, random_state=0)

        self.model_save_path = os.path.join(project_dir, "model", "water_rf_model.pkl")
        self.fig_save_dir = os.path.join(project_dir, "performance", "water_rf")

    # Train the model and tune hyperparameters
    def model_train(self):
        rf = RandomForestClassifier(random_state=0)

        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }

        # Run grid search
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )

        # Fit model
        grid_search.fit(self.train_features, self.train_labels)
        self.model = grid_search.best_estimator_

    # Run performance tests
    def run_perf_tests(self):
        predictions = self.model.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, predictions)
        roc_auc = roc_auc_score(
            self.test_labels,
            self.model.predict_proba(self.test_features),
            multi_class="ovr",
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(
            "\nClassification Report:\n",
            classification_report(self.test_labels, predictions),
        )

        # Confusion Matrix
        cm = confusion_matrix(self.test_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(self.fig_save_dir, "confusion_matrix.png"))

        return [accuracy, roc_auc]

    # Export model as joblib pickle
    def export_model(self):
        joblib.dump(self.model, self.model_save_path)

    def get_test_data(self):
        return [self.test_features, self.test_labels]


class WaterStabilityBoost:
    """
    Trains a gradient boosted tree model to fit the water stability data set.

    Attributes:
        train_features: feature dataframe for training
        train_labels: label dataframe for training
        test_features: feature dataframe for testing
        test_labels: label dataframe for testing
        model: xgboost tree model
        model_save_path: path to save model
        fig_save_dir: directory to save test figures
    """

    def __init__(
        self, project_dir: str, features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(features, labels, test_size=0.2, random_state=0)

        self.model_save_path = os.path.join(
            project_dir, "model", "water_boost_model.pkl"
        )
        self.fig_save_dir = os.path.join(project_dir, "performance", "water_boost")

    # Train the model and tune hyperparameters
    def model_train(self):
        self.model = None
        pass

    # Run performance tests
    def run_perf_tests(self):
        predictions = self.model.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, predictions)
        roc_auc = roc_auc_score(
            self.test_labels,
            self.model.predict_proba(self.test_features),
            multi_class="ovr",
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(
            "\nClassification Report:\n",
            classification_report(self.test_labels, predictions),
        )

        # Confusion Matrix
        cm = confusion_matrix(self.test_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(self.fig_save_dir, "confusion_matrix.png"))

        return [accuracy, roc_auc]

    # Export model as joblib pickle
    def export_model(self):
        joblib.dump(self.model, self.model_save_path)

    def get_test_data(self):
        return [self.test_features, self.test_labels]


if __name__ == "__main__":
    project_dir = "."
    data_dir = os.path.join(project_dir, "data")
    df = pd.read_csv(os.path.join(data_dir, "water_and_haz", "data.csv"))

    # Data cleansing
    removed_cols = [
        "MOF_name",
        "data_set",
        "Unnamed: 0",
        "doi",
        "filename",
        "0",
        "CoRE_name",
        "refcode",
        "name",
    ]
    other_labels = ["acid_label", "base_label", "boiling_label"]
    df_clean = df.loc[
        :, ~df.columns.isin(removed_cols) & ~df.columns.isin(other_labels)
    ]

    # Random forest
    rf_model = WaterStabilityRF(project_dir, df_clean.iloc[:, 1:], df_clean.iloc[:, 0])
    rf_model.model_train()
    rf_model.export_model()
    rf_model.run_perf_tests()
