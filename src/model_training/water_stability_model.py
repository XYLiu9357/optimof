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
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
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
        test_data_save_path: path to save test data
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
        self.test_data_save_path = os.path.join(
            project_dir, "data", "water_and_haz", "water_rf_test_data.pkl"
        )

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
        plt.close()

        # ROC Curve
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

        if len(self.test_labels.unique()) > 2:  # Multi-class case
            # Binarize the labels for multi-class ROC
            y_test_bin = label_binarize(self.test_labels, classes=self.model.classes_)
            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(len(self.model.classes_)):
                fpr[i], tpr[i], _ = roc_curve(
                    y_test_bin[:, i], self.model.predict_proba(self.test_features)[:, i]
                )
                roc_auc[i] = auc(fpr[i], tpr[i])

            for i in range(len(self.model.classes_)):
                sns.lineplot(
                    x=fpr[i],
                    y=tpr[i],
                    lw=2,
                    label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
                )
        else:  # Binary case
            fpr, tpr, _ = roc_curve(
                self.test_labels, self.model.predict_proba(self.test_features)[:, 1]
            )
            sns.lineplot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )

        sns.lineplot(x=[0, 1], y=[0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.fig_save_dir, "roc_curve.png"))
        plt.close()

        return [accuracy, roc_auc]

    # Export model as joblib pickle
    def export_model(self):
        joblib.dump(self, self.model_save_path)

    def save_test_data(self):
        test_data = pd.concat([self.test_labels, self.test_features])
        test_data.to_pickle(self.test_data_save_path)
        print(f"Test data saved to: {self.test_data_save_path}")


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
        test_data_save_path: path to save test data
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
        self.test_data_save_path = os.path.join(
            project_dir, "data", "water_and_haz", "water_boost_test_data.pkl"
        )

    # Train the model and tune hyperparameters
    def model_train(self):
        model = xgb.XGBClassifier(objective="multi:softprob", random_state=0)

        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=2,
        )
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
        plt.close()

        # ROC Curve
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

        if len(self.test_labels.unique()) > 2:  # Multi-class case
            # Binarize the labels for multi-class ROC
            y_test_bin = label_binarize(self.test_labels, classes=self.model.classes_)
            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(len(self.model.classes_)):
                fpr[i], tpr[i], _ = roc_curve(
                    y_test_bin[:, i], self.model.predict_proba(self.test_features)[:, i]
                )
                roc_auc[i] = auc(fpr[i], tpr[i])

            for i in range(len(self.model.classes_)):
                sns.lineplot(
                    x=fpr[i],
                    y=tpr[i],
                    lw=2,
                    label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
                )
        else:  # Binary case
            fpr, tpr, _ = roc_curve(
                self.test_labels, self.model.predict_proba(self.test_features)[:, 1]
            )
            sns.lineplot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )

        sns.lineplot(x=[0, 1], y=[0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.fig_save_dir, "roc_curve.png"))
        plt.close()

        return [accuracy, roc_auc]

    # Export model as joblib pickle
    def export_model(self):
        joblib.dump(self, self.model_save_path)

    def save_test_data(self):
        test_data = pd.concat([self.test_labels, self.test_features])
        test_data.to_pickle(self.test_data_save_path)
        print(f"Test data saved to: {self.test_data_save_path}")


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
    normalized_labels = df_clean.iloc[:, 0] - 1  # Labels: {0, 1, 2, 3}

    # Random forest
    print("**Training Random Forest Model")
    rf_model = WaterStabilityRF(project_dir, df_clean.iloc[:, 1:], normalized_labels)
    rf_model.model_train()
    rf_model.export_model()

    print("Random Forest Performance:")
    rf_model.run_perf_tests()
    rf_model.save_test_data()
    print("**End of Random Forest")

    # Gradient-boosted model: not as good as RF
    # print("**Training Boosted Tree Model")
    # gb_model = WaterStabilityBoost(project_dir, df_clean.iloc[:, 1:], normalized_labels)
    # gb_model.model_train()
    # gb_model.export_model()

    # print("Boosted Tree Performance:")
    # gb_model.run_perf_tests()
    # print("**End of Boosted Tree")

"""
Random Forest Performance:
Accuracy: 0.6712
ROC AUC Score: 0.8277

Classification Report:
               precision    recall  f1-score   support

           0       0.75      0.35      0.48        17
           1       0.66      0.65      0.65        68
           2       0.68      0.82      0.74       112
           3       0.56      0.23      0.32        22

    accuracy                           0.67       219
   macro avg       0.66      0.51      0.55       219
weighted avg       0.67      0.67      0.65       219

------------------------------------------------------------

Boosted Tree Performance:
Accuracy: 0.6347
ROC AUC Score: 0.7959

Classification Report:
               precision    recall  f1-score   support

           0       0.60      0.35      0.44        17
           1       0.67      0.54      0.60        68
           2       0.63      0.83      0.72       112
           3       0.43      0.14      0.21        22

    accuracy                           0.63       219
   macro avg       0.58      0.47      0.49       219
weighted avg       0.62      0.63      0.61       219
"""
