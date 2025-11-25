import numpy as np
from typing import List, Tuple, Dict, Any
from .base_algorithm import BaseAlgorithm


class SimulatedAnnealing(BaseAlgorithm):
    """Simulated Annealing for TSP using 2-opt neighborhood."""
    
    def __init__(self, problem: Any, initial_temperature: float = 100.0,
                cooling_rate: float = 0.995, min_temperature: float = 0.01):
        """
        Initialize Simulated Annealing.
        
        Args:
            problem: TSP problem instance
            initial_temperature: Starting temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            min_temperature: Minimum temperature threshold
        """
        super().__init__(problem, name='SA')
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        
        self.current_solution = None
        self.current_cost = float('inf')
        self.temperature = initial_temperature
        self.cooling_step = 0
        self.acceptance_history = []
    
    def initialize(self, **kwargs) -> None:
        """Initialize with a random solution."""
        self.current_solution = self.problem.random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.temperature = self.initial_temperature
        self.cooling_step = 0
        self.acceptance_history = []
        self.evaluations_used = 1
        
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Execute SA for specified number of evaluations.
        
        Args:
            num_evaluations: Number of evaluations to perform
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations and self.temperature > self.min_temperature:
            # Generate neighbor using 2-opt
            neighbor = self._two_opt_neighbor(self.current_solution)
            neighbor_cost = self.problem.evaluate(neighbor)
            evals_done += 1
            self.evaluations_used += 1
            
            # Decide whether to accept neighbor
            if self._accept(neighbor_cost, self.current_cost):
                self.current_solution = neighbor
                self.current_cost = neighbor_cost
                self.acceptance_history.append(1)
                
                # Update best if improved
                self.update_best(neighbor.copy(), neighbor_cost)
            else:
                self.acceptance_history.append(0)
            
            # Cool down temperature
            self.temperature *= self.cooling_rate
            self.cooling_step += 1
            
            # Keep acceptance history manageable
            if len(self.acceptance_history) > 100:
                self.acceptance_history.pop(0)
        
        return self.get_best()
    
    def _two_opt_neighbor(self, solution: List[int]) -> List[int]:
        """
        Generate neighbor using 2-opt move (reverse a segment).
        
        Args:
            solution: Current solution
            
        Returns:
            Neighbor solution
        """
        neighbor = solution.copy()
        size = len(neighbor)
        
        # Choose two random positions
        i, j = sorted(np.random.choice(size, 2, replace=False))
        
        # Reverse the segment between i and j
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        
        return neighbor
    
    def _accept(self, new_cost: float, current_cost: float) -> bool:
        """
        Decide whether to accept a new solution.
        
        Args:
            new_cost: Cost of new solution
            current_cost: Cost of current solution
            
        Returns:
            True if accepted, False otherwise
        """
        # Always accept improvements
        if new_cost < current_cost:
            return True
        
        # Accept worse solutions with probability based on temperature
        delta = new_cost - current_cost
        probability = np.exp(-delta / self.temperature)
        return np.random.random() < probability
    
    def get_state(self) -> Dict[str, Any]:
        """Get current SA state."""
        return {
            'current_solution': self.current_solution.copy(),
            'current_cost': self.current_cost,
            'temperature': self.temperature,
            'cooling_step': self.cooling_step,
            'acceptance_history': self.acceptance_history.copy(),
            'best_solution': self.best_solution.copy() if self.best_solution else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore SA state."""
        self.current_solution = state['current_solution'].copy()
        self.current_cost = state['current_cost']
        self.temperature = state['temperature']
        self.cooling_step = state['cooling_step']
        self.acceptance_history = state['acceptance_history'].copy()
        self.best_solution = state['best_solution'].copy() if state['best_solution'] else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']