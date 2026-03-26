"""Model evaluation utilities with unified interface.

Plot specifications:
- Confusion matrices: Font size 15 for all elements.
- Other plots: Font size 15 for labels/titles, tick size 15,
    figsize=(7,5), and legend font size 12.
"""

import math
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


class RegressionEvaluator:
    """Evaluator for regression models (e.g., thermal stability)."""

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary of metric names and values
        """
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        metrics = {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

        print(f"R² score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")

        return metrics

    def plot_results(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path
    ) -> None:
        """Generate and save regression diagnostic plots.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            save_dir: Directory to save plots
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # True vs Predicted
        plt.figure(figsize=(7, 5))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot(
            [min(y_true), max(y_true)],
            [min(y_true), max(y_true)],
            color="red",
        )
        plt.xlabel("True Values", fontsize=15)
        plt.ylabel("Predicted Values", fontsize=15)
        plt.title("True vs Predicted Values", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(save_dir / "true_vs_predicted.png")
        plt.close()

        # Residuals
        residuals = y_true - y_pred
        plt.figure(figsize=(7, 5))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(0, min(y_pred), max(y_pred), colors="red")
        plt.xlabel("Predicted Values", fontsize=15)
        plt.ylabel("Residuals", fontsize=15)
        plt.title("Residuals vs Predicted Values", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(save_dir / "residuals.png")
        plt.close()

        # Residuals distribution
        plt.figure(figsize=(7, 5))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals", fontsize=15)
        plt.title("Distribution of Residuals", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(save_dir / "residual_distribution.png")
        plt.close()

        # QQ plot
        plt.figure(figsize=(7, 5))
        stats.probplot(residuals.flatten(), dist="norm", plot=plt)
        plt.title("QQ Plot of Residuals", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(save_dir / "qq_plot.png")
        plt.close()


class BinaryClassificationEvaluator:
    """Evaluator for binary classification models (e.g., solvent stability)."""

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate binary classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities

        Returns:
            Dictionary of metric names and values
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_prob)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
        }

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

        return metrics

    def plot_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        save_dir: Path,
    ) -> None:
        """Generate and save binary classification diagnostic plots.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (binary)
            y_prob: Predicted probabilities
            save_dir: Directory to save plots
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix (transposed)
        cm = confusion_matrix(y_true, y_pred).T
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"fontsize": 15})
        plt.xlabel("True Labels", fontsize=15)
        plt.ylabel("Predicted Labels", fontsize=15)
        plt.title("Confusion Matrix", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(save_dir / "confusion_matrix.png")
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="red")
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title("ROC Curve", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.legend(loc="lower right", fontsize=12)
        plt.savefig(save_dir / "roc_curve.png")
        plt.close()


class MultiClassEvaluator:
    """Evaluator for multi-class classification models (e.g., water stability)."""

    def __init__(self, class_labels: np.ndarray) -> None:
        """Initialize MultiClassEvaluator.

        Args:
            class_labels: Array of class labels
        """
        self.class_labels = class_labels

    def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Calculate multi-class classification metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (n_samples x n_classes)

        Returns:
            Dictionary of metric names and values
        """
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

        metrics = {"accuracy": accuracy, "roc_auc": roc_auc}

        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        return metrics

    def plot_results(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        save_dir: Path,
    ) -> None:
        """Generate and save multi-class classification diagnostic plots.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (n_samples x n_classes)
            save_dir: Directory to save plots
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix (transposed)
        cm = confusion_matrix(y_true, y_pred).T
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"fontsize": 15})
        plt.title("Confusion Matrix", fontsize=15)
        plt.xlabel("True", fontsize=15)
        plt.ylabel("Predicted", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.savefig(save_dir / "confusion_matrix.png")
        plt.close()

        # ROC curve for each class
        plt.figure(figsize=(7, 5))
        sns.set(style="whitegrid")

        # Binarize the labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=self.class_labels)
        fpr = {}
        tpr = {}
        roc_auc = {}

        for i in range(len(self.class_labels)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        for i in range(len(self.class_labels)):
            sns.lineplot(
                x=fpr[i],
                y=tpr[i],
                lw=2,
                label=f"Class {i} (AUC = {roc_auc[i]:.2f})",
            )

        sns.lineplot(x=[0, 1], y=[0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)
        plt.title("Receiver Operating Characteristic", fontsize=15)
        plt.tick_params(axis='both', labelsize=15)
        plt.legend(loc="lower right", fontsize=12)
        plt.savefig(save_dir / "roc_curve.png")
        plt.close()
