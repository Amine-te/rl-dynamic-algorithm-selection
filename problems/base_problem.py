from abc import ABC, abstractmethod
from typing import Any, Tuple
import numpy as np


class BaseProblem(ABC):
    """Abstract base class for optimization problems."""
    
    def __init__(self, instance_data: Any):
        """
        Initialize problem with instance data.
        
        Args:
            instance_data: Problem-specific data (e.g., distance matrix for TSP)
        """
        self.instance_data = instance_data
        self.best_known_cost = None  # If available from benchmarks
    
    @abstractmethod
    def evaluate(self, solution: Any) -> float:
        """
        Evaluate a solution and return its cost.
        
        Args:
            solution: Problem-specific solution representation
            
        Returns:
            Cost value (lower is better)
        """
        pass
    
    @abstractmethod
    def random_solution(self) -> Any:
        """
        Generate a random valid solution.
        
        Returns:
            A random solution
        """
        pass
    
    @abstractmethod
    def get_problem_size(self) -> int:
        """
        Return the problem size (e.g., number of cities for TSP).
        
        Returns:
            Integer representing problem size
        """
        pass
    
    def is_valid(self, solution: Any) -> bool:
        """
        Check if solution is valid (default: always valid).
        Override for constrained problems.
        
        Args:
            solution: Solution to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True