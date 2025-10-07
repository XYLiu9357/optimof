"""solvent_model_test.py
Utility class for evaluating performance of the solvent removal stability model.
Requires a test set that was not used in training.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from solvent_model import SolventModel


class SolventModelPerfTest:

    def __init__(
        self, model_file_path: Path, test_features: pd.DataFrame, test_labels: pd.Series
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model
        self.model = torch.load(model_file_path, weights_only=False)
        self.model.eval()  # Ensure the model is in evaluation mode

        # Convert test features and labels to tensors
        test_features = test_features.apply(pd.to_numeric, errors="raise")
        self.test_features = torch.tensor(test_features.values, dtype=torch.float32).to(
            self.device
        )

        # Test labels are already standard-scaled
        self.test_labels = (
            torch.tensor(test_labels.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device)
        )

        # Make predictions
        with torch.no_grad():
            logits = self.model(self.test_features)
            self.predictions = (
                torch.sigmoid(logits).cpu().numpy()
            )  # Convert logits to probabilities

        # Binarize predictions
        self.binary_predictions = (self.predictions >= 0.5).astype(int)

        # Move test labels to CPU for evaluation
        self.test_labels = self.test_labels.cpu().numpy()

    def calculate_accuracy(self):
        accuracy = accuracy_score(self.test_labels, self.binary_predictions)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy

    def calculate_precision(self):
        precision = precision_score(self.test_labels, self.binary_predictions)
        print(f"Precision: {precision:.4f}")
        return precision

    def calculate_recall(self):
        recall = recall_score(self.test_labels, self.binary_predictions)
        print(f"Recall: {recall:.4f}")
        return recall

    def calculate_f1(self):
        f1 = f1_score(self.test_labels, self.binary_predictions)
        print(f"F1 Score: {f1:.4f}")
        return f1

    def calculate_auc(self):
        auc = roc_auc_score(self.test_labels, self.predictions)
        print(f"AUC-ROC: {auc:.4f}")
        return auc

    def plot_confusion_matrix(self, save_dir: Path):
        cm = confusion_matrix(self.test_labels, self.binary_predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(save_dir / "confusion_matrix.png")

    def plot_roc_curve(self, save_dir: Path):
        fpr, tpr, _ = roc_curve(self.test_labels, self.predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc_score(self.test_labels, self.predictions):.4f}",
        )
        plt.plot([0, 1], [0, 1], linestyle="--", color="red")  # Diagonal line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(save_dir / "roc_curve.png")


if __name__ == "__main__":
    print("**Testing solvent model accuracy**")

    # Configure paths
    project_path = Path(".")
    data_dir = project_path / "data"

    # Change these two lines to change the model tested
    test_data_path = data_dir / "solvent" / "solvent_test_data.pkl"
    model_file_path = project_path / "model" / "solvent_model.pkl"

    # Load test data that are unused in training
    test_df: pd.DataFrame = pd.read_pickle(test_data_path)
    test_labels: pd.DataFrame = test_df.loc[:, "solvent"]
    test_features: pd.DataFrame = test_df.loc[:, test_df.columns != "solvent"]

    # Performance tests
    performance_test = SolventModelPerfTest(model_file_path, test_features, test_labels)
    save_dir = Path(".") / "performance" / "solvent"

    performance_test.calculate_accuracy()
    performance_test.calculate_precision()
    performance_test.calculate_recall()
    performance_test.calculate_f1()
    performance_test.calculate_auc()
    performance_test.plot_confusion_matrix(save_dir)
    performance_test.plot_roc_curve(save_dir)
