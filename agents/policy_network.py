"""
Policy Network (Actor) for algorithm selection.
Outputs probability distribution over algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class PolicyNetwork(nn.Module):
    """Actor network that outputs action probabilities."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 64]):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state features
            action_dim: Number of algorithms to choose from
            hidden_dims: List of hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output layer (logits for each action)
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
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
            Action logits of shape (batch_size, action_dim)
        """
        features = self.feature_extractor(state)
        logits = self.action_head(features)
        return logits
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probability distribution.
        
        Args:
            state: State tensor
            
        Returns:
            Action probabilities (softmax over logits)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Select an action given a state.
        
        Args:
            state: State as numpy array
            deterministic: If True, select argmax. If False, sample from distribution.
            
        Returns:
            Tuple of (action, log_probability)
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            probs = self.get_action_probs(state_tensor)
        
        if deterministic:
            # Select action with highest probability
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action]).item()
        else:
            # Sample from probability distribution
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of actions.
        Used during training.
        
        Args:
            states: Batch of states (batch_size, state_dim)
            actions: Batch of actions (batch_size,)
            
        Returns:
            Tuple of (log_probs, entropy)
        """
        probs = self.get_action_probs(states)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
    
    def save(self, filepath: str):
        """Save model weights."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        self.load_state_dict(torch.load(filepath))


class FeatureEmbeddingNetwork(nn.Module):
    """
    Alternative architecture with separate feature processing.
    Processes landscape and history features separately before fusion.
    """
    
    def __init__(self, landscape_dim: int, history_dim: int, action_dim: int):
        """
        Initialize feature embedding network.
        
        Args:
            landscape_dim: Dimension of landscape features (should be 9)
            history_dim: Dimension of history features (should be 2 * num_algorithms)
            action_dim: Number of algorithms
        """
        super(FeatureEmbeddingNetwork, self).__init__()
        
        self.landscape_dim = landscape_dim
        self.history_dim = history_dim
        self.action_dim = action_dim
        
        # Landscape feature processor
        self.landscape_encoder = nn.Sequential(
            nn.Linear(landscape_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # History feature processor
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.Tanh()
        )
        
        # Action head
        self.action_head = nn.Linear(64, action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with feature separation.
        
        Args:
            state: Full state tensor (batch_size, landscape_dim + history_dim)
            
        Returns:
            Action logits
        """
        # Split state into landscape and history
        landscape = state[:, :self.landscape_dim]
        history = state[:, self.landscape_dim:]
        
        # Encode separately
        landscape_features = self.landscape_encoder(landscape)
        history_features = self.history_encoder(history)
        
        # Fuse features
        fused = torch.cat([landscape_features, history_features], dim=-1)
        fused_features = self.fusion(fused)
        
        # Output action logits
        logits = self.action_head(fused_features)
        return logits
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """Select action (same as PolicyNetwork)."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.get_action_probs(state_tensor)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action]).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        
        return action, log_prob
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions (same as PolicyNetwork)."""
        probs = self.get_action_probs(states)
        dist = torch.distributions.Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
    
    def save(self, filepath: str):
        """Save model."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str):
        """Load model."""
        self.load_state_dict(torch.load(filepath))