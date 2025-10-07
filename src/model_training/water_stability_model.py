"""water_stability_model.py
Main model for MOF water stability prediction. Feature extraction procedures
should be invoked prior to running this training script.

Task: 4-class classification
Predictor: MOF geometric information found using Zeo++
Predictand: MOF stability upon solvent removal
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import label_binarize


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
        self, project_dir: Path, features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(features, labels, test_size=0.2, random_state=0)

        project_dir = Path(project_dir)
        self.model_save_path = project_dir / "model" / "water_rf_model.pkl"
        self.fig_save_dir = project_dir / "performance" / "water_rf"
        self.test_data_save_path = (
            project_dir / "data" / "water_and_haz" / "water_rf_test_data.pkl"
        )

    # Train the model and tune hyperparameters
    def model_train(self):
        rf = RandomForestClassifier(random_state=0)

        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [100, 150, 200, 250, 300],
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
        # Ensure fig_save_dir is a Path object (for backward compatibility with pickled models)
        fig_save_dir = Path(self.fig_save_dir)

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
        plt.savefig(fig_save_dir / "confusion_matrix.png")
        plt.close()

        # ROC Curve
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

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

        sns.lineplot(x=[0, 1], y=[0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(fig_save_dir / "roc_curve.png")
        plt.close()

        # # SHAP value distribution calculations
        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(self.test_features)

        # # If shap_values is a dictionary (for multi-class classification), extract keys
        # if isinstance(shap_values, dict):
        #     shap_values_dict = shap_values
        # else:
        #     # If it's a list (for binary classification or single class)
        #     shap_values_dict = {i: val for i, val in enumerate(shap_values)}

        # # Concatenate SHAP values and create DataFrame
        # all_shap_values = []
        # all_classes = []
        # for class_idx, values in shap_values_dict.items():
        #     if not isinstance(values, np.ndarray):
        #         raise TypeError(
        #             f"Expected SHAP values for class {class_idx} to be a numpy array."
        #         )
        #     all_shap_values.append(values)
        #     all_classes.append(np.full(values.shape[0], fill_value=class_idx))

        # shap_values_concat = np.concatenate(all_shap_values, axis=0)
        # shap_values_df = pd.DataFrame(
        #     shap_values_concat, columns=self.test_features.columns
        # )
        # shap_values_df["class"] = np.concatenate(all_classes)

        # # SHAP summary plot
        # plt.figure(figsize=(10, 6))
        # shap_values_melted = shap_values_df.melt(
        #     id_vars="class" if "class" in shap_values_df.columns else None,
        #     var_name="Feature",
        #     value_name="SHAP Value",
        # )
        # shap_palette = sns.color_palette(["#1f77b4", "#ff69b4"])
        # sns.violinplot(
        #     x="SHAP Value",
        #     y="Feature",
        #     data=shap_values_melted,
        #     hue="class",
        #     palette=shap_palette,
        # )
        # plt.title("SHAP Value Distribution")
        # plt.savefig(os.path.join(self.fig_save_dir, "shap_value_distribution.png"))
        # plt.close()

        return [accuracy, roc_auc]

    # Export model as joblib pickle
    def export_model(self):
        joblib.dump(self, self.model_save_path)

    def save_test_data(self):
        # Ensure test_data_save_path is a Path object (for backward compatibility with pickled models)
        test_data_save_path = Path(self.test_data_save_path)
        test_data = pd.concat([self.test_labels, self.test_features])
        test_data.to_pickle(test_data_save_path)
        print(f"Test data saved to: {test_data_save_path}")


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
        self, project_dir: Path, features: pd.DataFrame, labels: pd.DataFrame
    ) -> None:
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(features, labels, test_size=0.2, random_state=0)

        project_dir = Path(project_dir)
        self.model_save_path = project_dir / "model" / "water_boost_model.pkl"
        self.fig_save_dir = project_dir / "performance" / "water_boost"
        self.test_data_save_path = (
            project_dir / "data" / "water_and_haz" / "water_boost_test_data.pkl"
        )

    # Train the model and tune hyperparameters
    def model_train(self):
        model = xgb.XGBClassifier(objective="multi:softprob", random_state=0)

        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [100, 150, 200, 250, 300],
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
        # Ensure fig_save_dir is a Path object (for backward compatibility with pickled models)
        fig_save_dir = Path(self.fig_save_dir)

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
        plt.savefig(fig_save_dir / "confusion_matrix.png")
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
        plt.savefig(fig_save_dir / "roc_curve.png")
        plt.close()

        return [accuracy, roc_auc]

    # Export model as joblib pickle
    def export_model(self):
        joblib.dump(self, self.model_save_path)

    def save_test_data(self):
        # Ensure test_data_save_path is a Path object (for backward compatibility with pickled models)
        test_data_save_path = Path(self.test_data_save_path)
        test_data = pd.concat([self.test_labels, self.test_features])
        test_data.to_pickle(test_data_save_path)
        print(f"Test data saved to: {test_data_save_path}")


if __name__ == "__main__":
    project_dir = Path(".")
    data_dir = project_dir / "data"
    solvent_data_path = data_dir / "water_and_haz" / "water_split_data.pkl"
    df_clean = joblib.load(solvent_data_path)

    # Data cleansing
    # removed_cols = [
    #     "MOF_name",
    #     "data_set",
    #     "Unnamed: 0",
    #     "doi",
    #     "filename",
    #     "0",
    #     "CoRE_name",
    #     "refcode",
    #     "name",
    # ]
    # other_labels = ["acid_label", "base_label", "boiling_label"]
    # df_clean = df.loc[
    #     :, ~df.columns.isin(removed_cols) & ~df.columns.isin(other_labels)
    # ]
    normalized_labels = df_clean.loc[:, "water"] - 1  # Labels: {0, 1, 2, 3}

    # Random forest
    print("**Training Random Forest Model")
    rf_model = WaterStabilityRF(
        project_dir, df_clean.loc[:, df_clean.columns != "water"], normalized_labels
    )
    rf_model.model_train()
    rf_model.export_model()

    print("Random Forest Performance:")
    rf_model.run_perf_tests()
    rf_model.save_test_data()
    print("**End of Random Forest Trial")

    # # Gradient-boosted model: not as good as RF
    # print("**Training Boosted Tree Model")
    # gb_model = WaterStabilityBoost(
    #     project_dir, df_clean.loc[:, df_clean.columns != "water"], normalized_labels
    # )
    # gb_model.model_train()
    # gb_model.export_model()

    # print("Boosted Tree Performance:")
    # gb_model.run_perf_tests()
    # print("**End of Boosted Tree")

"""
Random Forest Performance:
Accuracy: 0.6621
ROC AUC Score: 0.8400

Classification Report:
               precision    recall  f1-score   support

         0.0       0.80      0.48      0.60        25
         1.0       0.63      0.68      0.65        65
         2.0       0.72      0.77      0.74       115
         3.0       0.00      0.00      0.00        14

    accuracy                           0.66       219
   macro avg       0.54      0.48      0.50       219
weighted avg       0.65      0.66      0.65       219

------

Boosted Tree Performance:
Accuracy: 0.6758
ROC AUC Score: 0.8492

Classification Report:
               precision    recall  f1-score   support

         0.0       0.67      0.56      0.61        25
         1.0       0.64      0.68      0.66        65
         2.0       0.74      0.77      0.76       115
         3.0       0.11      0.07      0.09        14

    accuracy                           0.68       219
   macro avg       0.54      0.52      0.53       219
weighted avg       0.66      0.68      0.67       219
"""
