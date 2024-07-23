"""test_performance.py

"""


# Evaluate and visualize performance on a separate test set
def test_performance(self):
    # Ensure the model is in evaluation mode
    self.model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert test features and labels to tensors
    test_features = torch.tensor(self.test_features.values, dtype=torch.float32).to(
        device
    )
    test_labels = (
        torch.tensor(self.test_labels.values, dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    # Make predictions
    with torch.no_grad():
        predictions = self.model(test_features)

    # Move data back to CPU for evaluation
    predictions = predictions.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    # Calculate R2 score
    r2 = r2_score(test_labels, predictions)
    print(f"R2 score: {r2:.4f}")

    # Calculate Mean Squared Error
    mse = mean_squared_error(test_labels, predictions)
    print(f"Mean Squared Error: {mse:.4f}")

    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(test_labels, predictions, alpha=0.5)
    plt.plot(
        [min(test_labels), max(test_labels)],
        [min(test_labels), max(test_labels)],
        color="red",
    )  # Line of perfect fit
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()

    # Plot residuals
    residuals = test_labels - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.hlines(0, min(predictions), max(predictions), colors="red")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted Values")
    plt.show()

    # Distribution of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals")
    plt.show()

    # QQ plot for residuals
    import scipy.stats as stats

    plt.figure(figsize=(10, 6))
    stats.probplot(residuals.flatten(), dist="norm", plot=plt)
    plt.title("QQ Plot of Residuals")
    plt.show()

    # Pair plot for feature analysis (requires seaborn and pandas)
    import pandas as pd

    feature_analysis = pd.DataFrame(self.test_features)
    feature_analysis["Residuals"] = residuals
    sns.pairplot(feature_analysis)
    plt.show()
