import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
from .base_algorithm import BaseAlgorithm


class TabuSearch(BaseAlgorithm):
    """Tabu Search for TSP with adaptive tenure."""
    
    def __init__(self, problem: Any, tabu_tenure: int = 10, 
                aspiration_enabled: bool = True):
        """
        Initialize Tabu Search.
        
        Args:
            problem: TSP problem instance
            tabu_tenure: Number of iterations a move stays tabu
            aspiration_enabled: Allow tabu moves if they improve best solution
        """
        super().__init__(problem, name='TS')
        self.tabu_tenure = tabu_tenure
        self.aspiration_enabled = aspiration_enabled
        
        self.current_solution = None
        self.current_cost = float('inf')
        self.tabu_list = deque(maxlen=tabu_tenure)
        self.iteration = 0
        self.aspiration_value = float('inf')
    
    def initialize(self, **kwargs) -> None:
        """Initialize with a random solution."""
        self.current_solution = self.problem.random_solution()
        self.current_cost = self.problem.evaluate(self.current_solution)
        self.tabu_list = deque(maxlen=self.tabu_tenure)
        self.iteration = 0
        self.evaluations_used = 1
        
        self.best_solution = self.current_solution.copy()
        self.best_cost = self.current_cost
        self.aspiration_value = self.best_cost
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Execute TS for specified number of evaluations.
        
        Args:
            num_evaluations: Number of evaluations to perform
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Generate neighborhood (2-opt moves)
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None
            
            # Sample neighborhood (not exhaustive for efficiency)
            num_neighbors = min(20, self.problem.get_problem_size() * 2)
            
            for _ in range(num_neighbors):
                if evals_done >= num_evaluations:
                    break
                
                neighbor, move = self._generate_neighbor(self.current_solution)
                neighbor_cost = self.problem.evaluate(neighbor)
                evals_done += 1
                self.evaluations_used += 1
                
                # Check if move is tabu
                is_tabu = move in self.tabu_list
                
                # Aspiration criterion: accept tabu move if it's better than best known
                aspiration = self.aspiration_enabled and neighbor_cost < self.aspiration_value
                
                # Accept if not tabu or aspiration criterion met
                if (not is_tabu or aspiration) and neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move
            
            # Move to best neighbor found
            if best_neighbor is not None:
                self.current_solution = best_neighbor
                self.current_cost = best_neighbor_cost
                
                # Add move to tabu list
                self.tabu_list.append(best_move)
                
                # Update best solution
                if self.update_best(best_neighbor.copy(), best_neighbor_cost):
                    self.aspiration_value = best_neighbor_cost
            
            self.iteration += 1
        
        return self.get_best()
    
    def _generate_neighbor(self, solution: List[int]) -> Tuple[List[int], Tuple[int, int]]:
        """
        Generate a neighbor using 2-opt and return the move.
        
        Args:
            solution: Current solution
            
        Returns:
            Tuple of (neighbor_solution, move)
            where move is (i, j) representing the swap positions
        """
        neighbor = solution.copy()
        size = len(neighbor)
        
        # Choose two random positions
        i, j = sorted(np.random.choice(size, 2, replace=False))
        
        # Reverse the segment between i and j
        neighbor[i:j+1] = list(reversed(neighbor[i:j+1]))
        
        # Move is the pair of indices (for tabu list)
        move = (i, j)
        
        return neighbor, move
    
    def get_state(self) -> Dict[str, Any]:
        """Get current TS state."""
        return {
            'current_solution': self.current_solution.copy(),
            'current_cost': self.current_cost,
            'tabu_list': list(self.tabu_list),
            'iteration': self.iteration,
            'aspiration_value': self.aspiration_value,
            'best_solution': self.best_solution.copy() if self.best_solution else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore TS state."""
        self.current_solution = state['current_solution'].copy()
        self.current_cost = state['current_cost']
        self.tabu_list = deque(state['tabu_list'], maxlen=self.tabu_tenure)
        self.iteration = state['iteration']
        self.aspiration_value = state['aspiration_value']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']