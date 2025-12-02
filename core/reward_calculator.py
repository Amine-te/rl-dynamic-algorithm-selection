import numpy as np
from typing import Optional

class RewardCalculator:
    """Calculates rewards for the RL agent."""
    
    def __init__(self, reward_type: str = 'improvement_with_efficiency'):
        """
        Initialize reward calculator.
        
        Args:
            reward_type: Type of reward function to use
                - 'improvement': Simple cost improvement
                - 'improvement_with_efficiency': Improvement scaled by budget efficiency
                - 'normalized_improvement': Normalized by initial cost
        """
        self.reward_type = reward_type
        self.initial_cost = None
        self.previous_best_cost = None
        self.cumulative_reward = 0.0
    
    def reset(self, initial_cost: float) -> None:
        """
        Reset calculator for new episode.
        
        Args:
            initial_cost: Initial solution cost
        """
        self.initial_cost = initial_cost
        self.previous_best_cost = initial_cost
        self.cumulative_reward = 0.0
    
    def calculate_reward(self, current_best_cost: float, 
                        evaluations_used: int, 
                        max_evaluations: int) -> float:
        """
        Calculate reward for the current step.
        
        Args:
            current_best_cost: Best cost found so far
            evaluations_used: Total evaluations used
            max_evaluations: Maximum evaluations allowed
            
        Returns:
            Reward value (float)
        """
        if self.previous_best_cost is None:
            self.previous_best_cost = current_best_cost
            return 0.0
        
        if self.reward_type == 'improvement':
            reward = self._improvement_reward(current_best_cost)
        
        elif self.reward_type == 'improvement_with_efficiency':
            reward = self._improvement_with_efficiency_reward(
                current_best_cost, evaluations_used, max_evaluations
            )
        
        elif self.reward_type == 'normalized_improvement':
            reward = self._normalized_improvement_reward(current_best_cost)
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        # Update previous best
        self.previous_best_cost = current_best_cost
        self.cumulative_reward += reward
        
        return reward
    
    def _improvement_reward(self, current_best_cost: float) -> float:
        """
        Simple improvement reward: r_t = cost_{t-1} - cost_t
        
        Args:
            current_best_cost: Current best cost
            
        Returns:
            Reward value
        """
        improvement = self.previous_best_cost - current_best_cost
        return improvement
    
    def _improvement_with_efficiency_reward(self, current_best_cost: float,
                                            evaluations_used: int,
                                            max_evaluations: int) -> float:
        """
        Improvement with efficiency bonus.
        Formula: r_t = (cost_{t-1} - cost_t) / cost_0 × (MaxEvals / Evals_total)
        
        Args:
            current_best_cost: Current best cost
            evaluations_used: Total evaluations used
            max_evaluations: Maximum evaluations allowed
            
        Returns:
            Reward value
        """
        if self.initial_cost is None or self.initial_cost == 0:
            return 0.0
        
        # Normalized improvement
        improvement = (self.previous_best_cost - current_best_cost) / self.initial_cost
        
        # Efficiency factor (higher reward for finding good solutions early)
        if evaluations_used > 0:
            efficiency_factor = max_evaluations / evaluations_used
        else:
            efficiency_factor = 1.0
        
        # Cap efficiency factor to prevent extreme values
        efficiency_factor = min(efficiency_factor, 10.0)
        
        reward = improvement * efficiency_factor
        
        return reward
    
    def _normalized_improvement_reward(self, current_best_cost: float) -> float:
        """
        Normalized improvement: r_t = (cost_{t-1} - cost_t) / cost_0
        
        Args:
            current_best_cost: Current best cost
            
        Returns:
            Reward value
        """
        if self.initial_cost is None or self.initial_cost == 0:
            return 0.0
        
        improvement = (self.previous_best_cost - current_best_cost) / self.initial_cost
        return improvement
    
    def get_cumulative_reward(self) -> float:
        """
        Get cumulative reward for the episode.
        
        Returns:
            Cumulative reward
        """
        return self.cumulative_reward
    
    def compute_episode_return(self, rewards: list, gamma: float = 0.99) -> float:
        """
        Compute discounted return for an episode.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor
            
        Returns:
            Discounted return
        """
        discounted_return = 0.0
        for t, reward in enumerate(rewards):
            discounted_return += (gamma ** t) * reward
        
        return discounted_return
    
    def compute_advantages(self, rewards: list, values: list, 
                            gamma: float = 0.99, lambda_gae: float = 0.95) -> list:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates from critic
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            
        Returns:
            List of advantage estimates
        """
        advantages = []
        gae = 0.0
        
        # Process in reverse (from last step to first)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0  # Terminal state
            else:
                next_value = values[t + 1]
            
            # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * A_{t+1}
            gae = delta + gamma * lambda_gae * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def normalize_rewards(self, rewards: list, epsilon: float = 1e-8) -> list:
        """
        Normalize rewards to have zero mean and unit variance.
        
        Args:
            rewards: List of rewards
            epsilon: Small constant for numerical stability
            
        Returns:
            Normalized rewards
        """
        if len(rewards) == 0:
            return rewards
        
        rewards_array = np.array(rewards)
        mean = np.mean(rewards_array)
        std = np.std(rewards_array)
        
        if std < epsilon:
            return [0.0] * len(rewards)
        
        normalized = (rewards_array - mean) / (std + epsilon)
        return normalized.tolist()