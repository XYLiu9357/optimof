"""test_utils.py
Utility class for evaluating performance of the model. 
Requires a test set that was not used in training. 
"""

import os
import math
import torch
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats


class ModelPerformanceTest:

    def __init__(
        self, model_file_path: str, test_features: pd.DataFrame, test_labels: pd.Series
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(model_file_path)

        # Load the model
        self.model = torch.load(model_file_path, weights_only=False)
        self.model.eval()  # Ensure the model is in evaluation mode

        # Convert test features and labels to tensors
        test_features = test_features.apply(pd.to_numeric, errors="raise")
        self.test_features = torch.tensor(test_features.values, dtype=torch.float32).to(
            self.device
        )
        self.test_labels = (
            torch.tensor(test_labels.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device)
        )

        # Make predictions
        with torch.no_grad():
            self.predictions = self.model(self.test_features).cpu().numpy()

        # Move test labels to CPU for evaluation
        self.test_labels = self.test_labels.cpu().numpy()

    def calculate_r2(self):
        r2 = r2_score(self.test_labels, self.predictions)
        print(f"R² score: {r2:.4f}")
        return r2

    def calculate_mse(self):
        mse = mean_squared_error(self.test_labels, self.predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

    def calculate_rmse(self):
        mse = mean_squared_error(self.test_labels, self.predictions)
        rmse = math.sqrt(mse)
        print(f"Root Mean Squared Error: {rmse:.4f}")
        return rmse

    def plot_actual_vs_predicted(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.test_labels, self.predictions, alpha=0.5)
        plt.plot(
            [min(self.test_labels), max(self.test_labels)],
            [min(self.test_labels), max(self.test_labels)],
            color="red",
        )  # Line of perfect fit
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.show()

    def plot_residuals(self):
        residuals = self.test_labels - self.predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(self.predictions, residuals, alpha=0.5)
        plt.hlines(0, min(self.predictions), max(self.predictions), colors="red")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.show()

    def plot_residuals_distribution(self):
        residuals = self.test_labels - self.predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.title("Distribution of Residuals")
        plt.show()

    def plot_qq(self):
        residuals = self.test_labels - self.predictions
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals.flatten(), dist="norm", plot=plt)
        plt.title("QQ Plot of Residuals")
        plt.show()

    def plot_pairwise(self):
        residuals = self.test_labels - self.predictions
        feature_analysis = pd.DataFrame(
            self.test_features.cpu().numpy(),
            columns=[f"feature_{i}" for i in range(self.test_features.shape[1])],
        )
        feature_analysis["Residuals"] = residuals
        sns.pairplot(feature_analysis)
        plt.show()


if __name__ == "__main__":
    print("**Testing thermal model accuracy**")

    # Configure paths
    project_path: str = "."
    data_dir: str = os.path.join(project_path, "data")
    test_data_path: str = os.path.join(data_dir, "thermal", "thermal_test_data.pkl")
    model_file_path: str = os.path.join(project_path, "model", "thermal_model.pkl")

    # Load test data that are unused in training
    test_df: pd.DataFrame = pd.read_pickle(test_data_path)
    test_labels: pd.DataFrame = test_df.iloc[:, 0]
    test_features: pd.DataFrame = test_df.iloc[:, 1:]

    # Performance tests
    performance_test = ModelPerformanceTest(model_file_path, test_features, test_labels)

    performance_test.calculate_r2()
    performance_test.calculate_mse()
    performance_test.plot_actual_vs_predicted()
    performance_test.plot_residuals()
    performance_test.plot_residuals_distribution()
    performance_test.plot_qq()
    performance_test.plot_pairwise()
