"""
Value Network (Critic) for state value estimation.
Estimates expected return from a given state.
"""

import torch
import torch.nn as nn
import numpy as np


class ValueNetwork(nn.Module):
    """Critic network that estimates state values."""
    
    def __init__(self, state_dim: int, hidden_dims: list = [128, 64]):
        """
        Initialize value network.
        
        Args:
            state_dim: Dimension of state features
            hidden_dims: List of hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer (single value)
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            State values of shape (batch_size, 1)
        """
        features = self.feature_extractor(state)
        value = self.value_head(features)
        return value
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for a single state.
        
        Args:
            state: State as numpy array
            
        Returns:
            Estimated value (float)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            value = self.forward(state_tensor)
        
        return value.item()
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        self.load_state_dict(torch.load(filepath))