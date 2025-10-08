"""Flexible MLP architecture supporting multiple variants."""

from typing import List

import torch
import torch.nn as nn


class FlexibleMLP(nn.Module):
    """Flexible MLP supporting simple, batchnorm, and residual architectures.

    Attributes:
        arch_type: Type of architecture ('simple', 'batchnorm', 'residual')
        activation_fn: Activation function to use
        layers: List of linear layers
        batchnorms: List of batch normalization layers (if arch_type='batchnorm')
        dropout: Dropout layer
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        dropout_prob: float = 0.2,
        arch_type: str = 'simple',
        activation: str = 'leaky_relu',
    ):
        """Initialize FlexibleMLP.

        Args:
            input_size: Size of input features
            hidden_layers: List of hidden layer sizes
            output_size: Size of output
            dropout_prob: Dropout probability
            arch_type: Architecture type ('simple', 'batchnorm', 'residual')
            activation: Activation function ('relu', 'leaky_relu', 'elu')
        """
        super(FlexibleMLP, self).__init__()

        self.arch_type = arch_type
        self.activation_fn = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout_prob)

        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.layer_sizes = layer_sizes

        # Build layers based on architecture type
        if arch_type == 'simple':
            self._build_simple(layer_sizes)
        elif arch_type == 'batchnorm':
            self._build_batchnorm(layer_sizes)
        elif arch_type == 'residual':
            self._build_residual(layer_sizes)
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")

    def _get_activation(self, activation: str):
        """Get activation function by name."""
        if activation == 'relu':
            return nn.functional.relu
        elif activation == 'leaky_relu':
            return nn.functional.leaky_relu
        elif activation == 'elu':
            return nn.functional.elu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def _build_simple(self, layer_sizes: List[int]):
        """Build simple fully-connected architecture."""
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

    def _build_batchnorm(self, layer_sizes: List[int]):
        """Build architecture with batch normalization."""
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        # Add batch norm for all layers except output
        self.batchnorms = nn.ModuleList([
            nn.BatchNorm1d(layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 2)  # Exclude output layer
        ])

    def _build_residual(self, layer_sizes: List[int]):
        """Build architecture with residual connections."""
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ])

        # Create projection layers for dimension mismatch
        self.projections = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):  # Exclude output layer
            if layer_sizes[i] != layer_sizes[i + 1]:
                # Need projection for residual connection
                self.projections.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            else:
                # No projection needed, dimensions match
                self.projections.append(None)

    def forward(self, x):
        """Forward pass through the network."""
        if self.arch_type == 'simple':
            return self._forward_simple(x)
        elif self.arch_type == 'batchnorm':
            return self._forward_batchnorm(x)
        elif self.arch_type == 'residual':
            return self._forward_residual(x)

    def _forward_simple(self, x):
        """Forward pass for simple architecture."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation and dropout for all layers except output
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
                x = self.dropout(x)
        return x

    def _forward_batchnorm(self, x):
        """Forward pass for batchnorm architecture."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply batchnorm, activation, and dropout for all layers except output
            if i < len(self.layers) - 1:
                x = self.batchnorms[i](x)
                x = self.activation_fn(x)
                x = self.dropout(x)
        return x

    def _forward_residual(self, x):
        """Forward pass for residual architecture."""
        for i, layer in enumerate(self.layers):
            identity = x

            # Main path
            x = layer(x)

            # Apply residual connection for all layers except output
            if i < len(self.layers) - 1:
                # Project identity if dimensions don't match
                if self.projections[i] is not None:
                    identity = self.projections[i](identity)

                # Residual connection
                x = x + identity

                # Activation and dropout after residual addition
                x = self.activation_fn(x)
                x = self.dropout(x)

        return x
