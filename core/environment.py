import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from .context_manager import ContextManager
from .state_extractor import StateExtractor
from .reward_calculator import RewardCalculator


class OptimizationEnvironment:
    """MDP Environment for dynamic algorithm selection."""
    
    def __init__(self, problem: Any, algorithms: List[Any], 
                max_evaluations: int = 10000,
                schedule_interval: int = 500,
                reward_type: str = 'improvement_with_efficiency'):
        """
        Initialize optimization environment.
        
        Args:
            problem: Problem instance to solve
            algorithms: List of algorithm instances
            max_evaluations: Maximum function evaluations per episode
            schedule_interval: Evaluations between algorithm selections
            reward_type: Type of reward function to use
        """
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.schedule_interval = schedule_interval
        
        # Initialize components
        self.context_manager = ContextManager(algorithms)
        self.algorithm_names = self.context_manager.get_algorithm_names()
        self.num_algorithms = len(self.algorithm_names)
        
        self.state_extractor = StateExtractor(
            problem, 
            self.algorithm_names,
            sample_size=50
        )
        
        self.reward_calculator = RewardCalculator(reward_type=reward_type)
        
        # Episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_done = False
        
        # For tracking
        self.selection_history = []
        self.cost_history = []
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            Initial state observation
        """
        # Initialize all algorithms
        self.context_manager.initialize_all()
        
        # Reset state extractor
        self.state_extractor.reset()
        
        # Reset reward calculator
        initial_best = self.context_manager.get_best()[1]
        self.reward_calculator.reset(initial_best)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.episode_done = False
        self.selection_history = []
        self.cost_history = [initial_best]
        
        # Extract initial state
        state = self.state_extractor.extract_features(
            self.context_manager, 
            self.max_evaluations
        )
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Algorithm index to select (0 to num_algorithms-1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.episode_done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")
        
        if action < 0 or action >= self.num_algorithms:
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.num_algorithms-1}]")
        
        # Select algorithm
        algorithm_name = self.algorithm_names[action]
        self.context_manager.select_algorithm(algorithm_name)
        
        # Store cost before step
        cost_before = self.context_manager.get_best()[1]
        
        # Execute algorithm for schedule_interval evaluations
        best_solution, best_cost = self.context_manager.step_current(self.schedule_interval)
        
        # Update state extractor history
        self.state_extractor.update_history(
            algorithm_name, 
            cost_before, 
            best_cost, 
            best_cost
        )
        
        # Calculate reward
        total_evals = self.context_manager.get_total_evaluations()
        reward = self.reward_calculator.calculate_reward(
            best_cost, 
            total_evals, 
            self.max_evaluations
        )
        
        # Update tracking
        self.current_step += 1
        self.episode_rewards.append(reward)
        self.selection_history.append(algorithm_name)
        self.cost_history.append(best_cost)
        
        # Check if episode is done
        done = total_evals >= self.max_evaluations
        self.episode_done = done
        
        # Extract next state
        next_state = self.state_extractor.extract_features(
            self.context_manager, 
            self.max_evaluations
        )
        
        # Info dictionary
        info = {
            'algorithm_selected': algorithm_name,
            'best_cost': best_cost,
            'total_evaluations': total_evals,
            'step': self.current_step,
            'cost_improvement': cost_before - best_cost
        }
        
        return next_state, reward, done, info
    
    def get_observation_space_size(self) -> int:
        """
        Get size of observation space.
        
        Returns:
            Dimension of state features
        """
        return self.state_extractor.get_feature_dimension()
    
    def get_action_space_size(self) -> int:
        """
        Get size of action space.
        
        Returns:
            Number of algorithms
        """
        return self.num_algorithms
    
    def get_episode_info(self) -> Dict[str, Any]:
        """
        Get information about current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        best_solution, best_cost = self.context_manager.get_best()
        
        return {
            'best_cost': best_cost,
            'total_reward': sum(self.episode_rewards),
            'total_steps': self.current_step,
            'total_evaluations': self.context_manager.get_total_evaluations(),
            'selection_history': self.selection_history.copy(),
            'cost_history': self.cost_history.copy(),
            'episode_done': self.episode_done
        }
    
    def get_algorithm_selection_counts(self) -> Dict[str, int]:
        """
        Get selection counts for each algorithm in current episode.
        
        Returns:
            Dictionary mapping algorithm names to selection counts
        """
        counts = {name: 0 for name in self.algorithm_names}
        for alg_name in self.selection_history:
            counts[alg_name] += 1
        return counts
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            String representation if mode='ansi', None otherwise
        """
        info = self.get_episode_info()
        selection_counts = self.get_algorithm_selection_counts()
        
        output = [
            f"\n{'='*50}",
            f"Optimization Environment State",
            f"{'='*50}",
            f"Step: {self.current_step}",
            f"Best Cost: {info['best_cost']:.4f}",
            f"Total Evaluations: {info['total_evaluations']} / {self.max_evaluations}",
            f"Total Reward: {info['total_reward']:.4f}",
            f"\nAlgorithm Selection Counts:",
        ]
        
        for alg_name, count in selection_counts.items():
            percentage = (count / max(1, self.current_step)) * 100
            output.append(f"  {alg_name}: {count} ({percentage:.1f}%)")
        
        if len(self.cost_history) > 1:
            initial_cost = self.cost_history[0]
            current_cost = self.cost_history[-1]
            improvement = initial_cost - current_cost
            improvement_pct = (improvement / initial_cost) * 100
            output.append(f"\nImprovement: {improvement:.4f} ({improvement_pct:.2f}%)")
        
        output.append(f"{'='*50}\n")
        
        result = '\n'.join(output)
        
        if mode == 'human':
            print(result)
            return None
        else:
            return result
    
    def close(self) -> None:
        """Clean up resources."""
        pass