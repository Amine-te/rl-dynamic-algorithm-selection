from typing import Dict, Any, List
import copy


class ContextManager:
    """Manages algorithm states for dynamic switching."""
    
    def __init__(self, algorithms: List[Any]):
        """
        Initialize context manager.
        
        Args:
            algorithms: List of algorithm instances
        """
        self.algorithms = {alg.name: alg for alg in algorithms}
        self.context_memory = {alg.name: None for alg in algorithms}
        
        # Common state shared across algorithms
        self.common_context = {
            'best_solution': None,
            'best_cost': float('inf'),
            'elite_solutions': [],  # Top N solutions found
            'total_evaluations': 0
        }
        
        self.current_algorithm = None
    
    def initialize_all(self) -> None:
        """Initialize all algorithms."""
        for name, alg in self.algorithms.items():
            alg.initialize()
            self.context_memory[name] = alg.get_state()
            
            # Update common context with best found
            if alg.best_cost < self.common_context['best_cost']:
                self.common_context['best_solution'] = alg.best_solution.copy()
                self.common_context['best_cost'] = alg.best_cost
            
            self.common_context['total_evaluations'] += alg.evaluations_used
    
    def select_algorithm(self, algorithm_name: str) -> None:
        """
        Select and restore an algorithm's state.
        
        Args:
            algorithm_name: Name of algorithm to select
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        # Save current algorithm state if one is active
        if self.current_algorithm is not None:
            self.context_memory[self.current_algorithm] = self.algorithms[self.current_algorithm].get_state()
        
        # Restore selected algorithm state
        self.current_algorithm = algorithm_name
        selected_alg = self.algorithms[algorithm_name]
        
        if self.context_memory[algorithm_name] is not None:
            selected_alg.set_state(self.context_memory[algorithm_name])
        
        # Update algorithm with common best if better
        if self.common_context['best_cost'] < selected_alg.best_cost:
            selected_alg.best_solution = self.common_context['best_solution'].copy()
            selected_alg.best_cost = self.common_context['best_cost']
    
    def step_current(self, num_evaluations: int) -> tuple:
        """
        Execute current algorithm for specified evaluations.
        
        Args:
            num_evaluations: Number of evaluations to perform
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        if self.current_algorithm is None:
            raise RuntimeError("No algorithm selected. Call select_algorithm() first.")
        
        alg = self.algorithms[self.current_algorithm]
        best_solution, best_cost = alg.step(num_evaluations)
        
        # Update common context
        if best_cost < self.common_context['best_cost']:
            self.common_context['best_solution'] = best_solution.copy()
            self.common_context['best_cost'] = best_cost
        
        self.common_context['total_evaluations'] += num_evaluations
        
        # Update elite solutions (keep top 5)
        self._update_elite_solutions(best_solution, best_cost)
        
        return best_solution, best_cost
    
    def _update_elite_solutions(self, solution: Any, cost: float, max_elite: int = 5) -> None:
        """
        Update elite solutions list.
        
        Args:
            solution: Solution to potentially add
            cost: Cost of the solution
            max_elite: Maximum number of elite solutions to keep
        """
        # Add to elite list
        self.common_context['elite_solutions'].append({
            'solution': solution.copy(),
            'cost': cost
        })
        
        # Sort by cost and keep top N
        self.common_context['elite_solutions'].sort(key=lambda x: x['cost'])
        self.common_context['elite_solutions'] = self.common_context['elite_solutions'][:max_elite]
    
    def get_best(self) -> tuple:
        """
        Get overall best solution found.
        
        Returns:
            Tuple of (best_solution, best_cost)
        """
        return self.common_context['best_solution'], self.common_context['best_cost']
    
    def get_current_algorithm(self) -> str:
        """
        Get name of currently active algorithm.
        
        Returns:
            Algorithm name
        """
        return self.current_algorithm
    
    def get_total_evaluations(self) -> int:
        """
        Get total evaluations across all algorithms.
        
        Returns:
            Total number of evaluations
        """
        return self.common_context['total_evaluations']
    
    def get_algorithm_names(self) -> List[str]:
        """
        Get list of available algorithm names.
        
        Returns:
            List of algorithm names
        """
        return list(self.algorithms.keys())
    
    def get_all_states(self) -> Dict[str, Any]:
        """
        Get complete state of all algorithms and common context.
        
        Returns:
            Dictionary containing all states
        """
        # Save current algorithm state
        if self.current_algorithm is not None:
            self.context_memory[self.current_algorithm] = self.algorithms[self.current_algorithm].get_state()
        
        return {
            'context_memory': copy.deepcopy(self.context_memory),
            'common_context': copy.deepcopy(self.common_context),
            'current_algorithm': self.current_algorithm
        }
    
    def set_all_states(self, state: Dict[str, Any]) -> None:
        """
        Restore complete state of all algorithms.
        
        Args:
            state: Dictionary containing all states
        """
        self.context_memory = copy.deepcopy(state['context_memory'])
        self.common_context = copy.deepcopy(state['common_context'])
        self.current_algorithm = state['current_algorithm']
        
        # Restore current algorithm if one is active
        if self.current_algorithm is not None:
            self.algorithms[self.current_algorithm].set_state(
                self.context_memory[self.current_algorithm]
            )