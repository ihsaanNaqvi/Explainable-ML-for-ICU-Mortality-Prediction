"""
Model definitions for ICU outcome prediction
Includes XGBoost, TCN, and Transformer architectures
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        """
        Initialize TCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolutional kernel size
            dilation: Dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.activation(self.norm(self.conv(x)))


class TransformerEncoder(nn.Module):
    """Transformer encoder for timeseries"""
    
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        """
        Initialize transformer encoder.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.embedding(x.unsqueeze(-1))
        return self.transformer(x)


class XGBoostWrapper:
    """Wrapper for XGBoost model"""
    
    def __init__(self, **kwargs):
        """Initialize XGBoost model"""
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(**kwargs)
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
