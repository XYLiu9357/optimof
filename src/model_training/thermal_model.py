"""thermal_model.py
Main model for MOF thermal stability prediction. Feature extraction procedures should be 
invoked prior to running this training script. 

Task: Regression
Predictor: MOF geometric information found using Zeo++
Predictand: MOF breakdown temperature 

* The model only uses fully connected layers for simplification.
"""

import os
import joblib
import json
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# Model class
class ThermalModel(nn.Module):
    """Thermal ANN Model
    ANN model that inherits PyTorch neural network module

    Attributes:
        graph_depth: int, depth of the computation graph
        layers: List[nn.Linear], a list of network layers used in prediction
    """

    def __init__(
        self, input_size: int, hidden_layer_sizes: List[int], output_size: int
    ):
        super(ThermalModel, self).__init__()

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.graph_depth: int = len(layer_sizes)
        self.layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(self.graph_depth - 1)
            ]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # No ReLU at the last layer
            if i < len(self.layers) - 1:
                x = nn.functional.leaky_relu(x)
        return x


# Pipeline class
class ThermalModelPipeline:
    """Thermal ANN Model Pipeline
    Pipeline class that encapsulates the building, training, testing, and
    saving of the thermal ANN model

    Attributes:
        project_path: str, root directory of project
        hyperparams: Dict, a dictionary that contains all the hyperparameters
        device: torch.device, device used for computation

        model_input_features: np.ndarray, features used for training and validation
        model_input_labels: np.ndarray, labels used for training and validation
        test_features: np.ndarray, features used for testing
        test_labels: np.ndarray, labels used for testing

        model: ThermalModel, the ANN model used for prediction
    """

    def __init__(
        self,
        project_path: str,
        hyperparams: Dict,
        all_df: pd.DataFrame,
        save=True,
    ):
        # Store attributes
        self.project_path: str = project_path
        self.hyperparams: Dict = hyperparams
        self.train_test_split(all_df, test_size=0.2)

        # Scale features to normal distribution
        self.standard_scale_features()

        # Device setup: use GPU if possible
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = device

        # Build model
        self.model_build()
        self.model.to(device)

        # Train model
        print("**Starting Training**")
        self.model_train(val_size=0.1)
        print("**Training Complete**")

        # Save model if specified
        if save:
            model_file_path: str = os.path.join(
                project_path, "model", "thermal_model.pkl"
            )
            torch.save(self.model, model_file_path)
            print("**Model saved**")

    # Scales to normal distribution and convert back to DataFrame
    def standard_scale_features(self):
        # Instantiate the scaler
        self.scalar = StandardScaler()

        # Fit the scaler only on the training data
        self.scalar.fit(self.model_input_features)

        # Transform both training and test features using the scaler fitted on the training data
        ndarray_model_input_features = self.scalar.transform(self.model_input_features)
        ndarray_test_features = self.scalar.transform(self.test_features)

        # Convert the scaled arrays back to DataFrames
        self.model_input_features = pd.DataFrame(
            ndarray_model_input_features, columns=self.model_input_features.columns
        )
        self.test_features = pd.DataFrame(
            ndarray_test_features, columns=self.test_features.columns
        )

        # Check for NaN values
        print(
            "NaNs in scaled train features:",
            self.model_input_features.isna().sum().sum(),
        )
        print("NaNs in scaled test features:", self.test_features.isna().sum().sum())

    # Split data set into training and testing set
    def train_test_split(self, all_df: pd.DataFrame, test_size=0.2):
        all_features = all_df.loc[:, all_df.columns != "thermal"]
        all_labels = all_df.loc[:, "thermal"]
        (
            self.model_input_features,
            self.test_features,
            self.model_input_labels,
            self.test_labels,
        ) = sklearn_train_test_split(
            all_features, all_labels, test_size=test_size, random_state=0
        )

        # Reset indices to ensure alignment
        self.test_features = self.test_features.reset_index(drop=True)
        self.test_labels = self.test_labels.reset_index(drop=True)

    # Build the model with the specified hyperparameters
    def model_build(self):
        # The last layer of hidden_layer_sizes is the output layer size
        input_size: int = self.hyperparams["input_size"]
        intermediate_sizes: List[int] = self.hyperparams["hidden_layers"]
        output_size: int = self.hyperparams["output_size"]

        # Check hyperparameter validity
        assert input_size == self.model_input_features.shape[1]
        assert output_size == 1

        # Construct model
        self.model: ThermalModel = ThermalModel(
            input_size, intermediate_sizes, output_size
        )

    # Train the model
    def model_train(self, val_size=0.1):
        # Extract relevant hyperparameters
        learning_rate = self.hyperparams["learning_rate"]
        batch_size = self.hyperparams["batch_size"]
        num_epochs = self.hyperparams["num_epochs"]
        patience = self.hyperparams["patience"]

        # Split input into training set and validation set
        train_features, val_features, train_labels, val_labels = (
            sklearn_train_test_split(
                self.model_input_features,
                self.model_input_labels,
                test_size=val_size,
                random_state=0,
            )
        )

        # Ensure all data is numeric and handle NaN values
        train_features = train_features.apply(pd.to_numeric, errors="raise")
        val_features = val_features.apply(pd.to_numeric, errors="raise")
        train_labels = train_labels.apply(pd.to_numeric, errors="raise")
        val_labels = val_labels.apply(pd.to_numeric, errors="raise")

        # Prepare tensor data set.
        tensor_train_features = torch.tensor(train_features.values, dtype=torch.float32)
        tensor_val_features = torch.tensor(val_features.values, dtype=torch.float32)
        tensor_train_labels = torch.tensor(
            train_labels.values, dtype=torch.float32
        ).unsqueeze(1)
        tensor_val_labels = torch.tensor(
            val_labels.values, dtype=torch.float32
        ).unsqueeze(1)
        # Added one dimension to ensure correct shape

        train_dataset = TensorDataset(tensor_train_features, tensor_train_labels)
        val_dataset = TensorDataset(tensor_val_features, tensor_val_labels)

        # Build training and validation data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Early stopping
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model = None

        # Training loop: move batches to GPU during training
        for epoch in range(num_epochs):
            # Train
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                cur_loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

            # Validation loop
            self.model.eval()
            val_loss = 0

            # Disable gradient calculations to save memory, just in case...
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    cur_loss = criterion(outputs, y_batch)
                    val_loss += cur_loss.item()

            # Print validation loss
            val_loss /= len(val_loader.dataset)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {cur_loss.item():.4f}, Val Loss: {val_loss:.4f}"
            )

            # Check if validation loss is improved - early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping")
                    break

        # Load the best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
            print("Loaded best model with validation loss:", best_val_loss)
        else:
            print("No improvement during training")

    # Return test features and labels, for benchmarking
    def get_test_data(self):
        has_nan_features = self.test_features.isna().any().any()
        has_nan_labels = self.test_labels.isna().any().any()
        print(f"Any NaN test features: {has_nan_features}")
        print(f"Any NaN test labels: {has_nan_labels}")
        return self.test_features, self.test_labels


if __name__ == "__main__":
    # File configs
    project_path: str = "."
    hyperparam_file_name = "thermal_hyperparams.json"
    data_dir: str = os.path.join(project_path, "data", "thermal")
    data_file_path = os.path.join(data_dir, "thermal_split_data.pkl")

    # Read data: train on all features
    # removed_cols: List[str] = ["filename", "0", "CoRE_name", "refcode", "name"]
    thermal_all_df: pd.DataFrame = joblib.load(data_file_path)
    # thermal_all_df = df.loc[:, ~df.columns.isin(removed_cols)]

    # Read hyperparameters
    print("**Reading hyperparameter config**")
    hyperparam_file_path: str = os.path.join(
        project_path, "model", hyperparam_file_name
    )
    with open(hyperparam_file_path, "r") as f:
        hyperparams: Dict = json.load(f)

    # Create, build, and train the model
    pipeline: ThermalModelPipeline = ThermalModelPipeline(
        project_path, hyperparams, thermal_all_df
    )

    # Use test features and labels to benchmark performance
    model_file_name: str = "thermal_model.pkl"
    model_file_path: str = os.path.join(project_path, "model", model_file_name)
    test_features, test_labels = pipeline.get_test_data()
    test_all = pd.concat([test_labels, test_features], axis=1)

    # Save unused test data for future testing
    print(f"**Model saved at {model_file_path}**")
    test_data_path = os.path.join(data_dir, "thermal_test_data.pkl")
    test_all.to_pickle(test_data_path)
    print(f"**Test data saved at {test_data_path}**")
