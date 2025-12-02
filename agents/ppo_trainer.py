"""
PPO Trainer for Dynamic Algorithm Selection.
Implements Proximal Policy Optimization for training the selection policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class PPOTrainer:
    """Proximal Policy Optimization trainer."""
    
    def __init__(
        self,
        policy_network,
        value_network,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        device: str = 'cpu'
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy_network: Actor network
            value_network: Critic network
            learning_rate: Learning rate for both networks
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs to update on collected data
            device: 'cpu' or 'cuda'
        """
        self.policy_network = policy_network.to(device)
        self.value_network = value_network.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        # Optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(), 
            lr=learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(), 
            lr=learning_rate
        )
        
        # Training statistics
        self.training_stats = defaultdict(list)
    
    def collect_trajectories(
        self, 
        env, 
        num_episodes: int
    ) -> Dict[str, List]:
        """
        Collect trajectories by running episodes in the environment.
        
        Args:
            env: OptimizationEnvironment instance
            num_episodes: Number of episodes to collect
            
        Returns:
            Dictionary containing trajectory data
        """
        trajectories = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'best_costs': []
        }
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            episode_dones = []
            
            while not done:
                # Select action using policy
                action, log_prob = self.policy_network.select_action(state)
                
                # Get value estimate
                value = self.value_network.get_value(state)
                
                # Take step in environment
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_values.append(value)
                episode_log_probs.append(log_prob)
                episode_dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # Store episode data
            trajectories['states'].extend(episode_states)
            trajectories['actions'].extend(episode_actions)
            trajectories['rewards'].extend(episode_rewards)
            trajectories['values'].extend(episode_values)
            trajectories['log_probs'].extend(episode_log_probs)
            trajectories['dones'].extend(episode_dones)
            trajectories['episode_rewards'].append(episode_reward)
            trajectories['episode_lengths'].append(episode_length)
            
            # Store best cost from episode
            episode_info = env.get_episode_info()
            trajectories['best_costs'].append(episode_info['best_cost'])
        
        return trajectories
    
    def compute_advantages(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0  # Terminal state value
        
        # Process in reverse (from last step to first)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
                gae = 0
            
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE: A = delta + gamma * lambda * A_next
            gae = delta + self.gamma * self.gae_lambda * gae
            
            # Return: R = A + V(s)
            return_t = gae + values[t]
            
            advantages.insert(0, gae)
            returns.insert(0, return_t)
            
            next_value = values[t]
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, trajectories: Dict[str, List]) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            trajectories: Dictionary containing trajectory data
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(trajectories['states'])).to(self.device)
        actions = torch.LongTensor(trajectories['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(trajectories['log_probs']).to(self.device)
        rewards = trajectories['rewards']
        values_list = trajectories['values']
        dones = trajectories['dones']
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values_list, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Store metrics
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'total_loss': 0,
            'approx_kl': 0
        }
        
        # Multiple update epochs on the same data
        for epoch in range(self.update_epochs):
            # Evaluate current policy on collected states and actions
            new_log_probs, entropy = self.policy_network.evaluate_actions(states, actions)
            
            # Get current value estimates
            values = self.value_network(states).squeeze()
            
            # === Policy Loss (PPO clipped objective) ===
            # Ratio: π_new(a|s) / π_old(a|s)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Surrogate objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # === Value Loss ===
            value_loss = nn.MSELoss()(values, returns)
            
            # === Entropy Bonus (for exploration) ===
            entropy_loss = -entropy.mean()
            
            # === Total Loss ===
            total_loss = (
                policy_loss + 
                self.value_loss_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # === Update Policy Network ===
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # === Update Value Network ===
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # Approximate KL divergence (for monitoring)
            with torch.no_grad():
                approx_kl = (old_log_probs - new_log_probs).mean().item()
            
            # Accumulate metrics
            metrics['policy_loss'] += policy_loss.item()
            metrics['value_loss'] += value_loss.item()
            metrics['entropy'] += entropy.mean().item()
            metrics['total_loss'] += total_loss.item()
            metrics['approx_kl'] += approx_kl
        
        # Average metrics over epochs
        for key in metrics:
            metrics[key] /= self.update_epochs
        
        return metrics
    
    def train(
        self, 
        env, 
        num_iterations: int,
        episodes_per_iteration: int = 16,
        save_interval: int = 10,
        checkpoint_dir: str = 'checkpoints',
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            env: OptimizationEnvironment instance
            num_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            save_interval: Save checkpoint every N iterations
            checkpoint_dir: Directory to save checkpoints
            verbose: Print training progress
            
        Returns:
            Dictionary of training history
        """
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        history = defaultdict(list)
        
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = self.collect_trajectories(env, episodes_per_iteration)
            
            # Update networks
            metrics = self.update(trajectories)
            
            # Store metrics
            for key, value in metrics.items():
                history[key].append(value)
            
            # Episode statistics
            mean_episode_reward = np.mean(trajectories['episode_rewards'])
            mean_episode_length = np.mean(trajectories['episode_lengths'])
            mean_best_cost = np.mean(trajectories['best_costs'])
            
            history['mean_episode_reward'].append(mean_episode_reward)
            history['mean_episode_length'].append(mean_episode_length)
            history['mean_best_cost'].append(mean_best_cost)
            
            # Print progress
            if verbose and (iteration + 1) % 5 == 0:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"{'='*60}")
                print(f"Mean Episode Reward: {mean_episode_reward:.4f}")
                print(f"Mean Best Cost: {mean_best_cost:.2f}")
                print(f"Mean Episode Length: {mean_episode_length:.1f}")
                print(f"Policy Loss: {metrics['policy_loss']:.4f}")
                print(f"Value Loss: {metrics['value_loss']:.4f}")
                print(f"Entropy: {metrics['entropy']:.4f}")
                print(f"Approx KL: {metrics['approx_kl']:.6f}")
            
            # Save checkpoint
            if (iteration + 1) % save_interval == 0:
                self.save_checkpoint(
                    os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration + 1}.pt'),
                    iteration + 1
                )
                if verbose:
                    print(f"✓ Checkpoint saved at iteration {iteration + 1}")
        
        return dict(history)
    
    def save_checkpoint(self, filepath: str, iteration: int):
        """
        Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            iteration: Current iteration number
        """
        checkpoint = {
            'iteration': iteration,
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': dict(self.training_stats)
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.training_stats = defaultdict(list, checkpoint['training_stats'])
        
        return checkpoint['iteration']