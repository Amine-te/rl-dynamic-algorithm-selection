import numpy as np
from typing import List, Tuple, Dict, Any
from .base_algorithm import BaseAlgorithm


class GeneticAlgorithm(BaseAlgorithm):
    """Genetic Algorithm for TSP using Order Crossover (OX)."""
    
    def __init__(self, problem: Any, population_size: int = 50, 
                mutation_rate: float = 0.1, crossover_rate: float = 0.8):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problem: TSP problem instance
            population_size: Number of individuals in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        super().__init__(problem, name='GA')
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        self.population = []
        self.fitness_values = []
        self.generation = 0
    
    def initialize(self, **kwargs) -> None:
        """Initialize population with random solutions."""
        self.population = []
        self.fitness_values = []
        self.generation = 0
        self.evaluations_used = 0
        self.best_solution = None
        self.best_cost = float('inf')
        
        # Create initial population
        for _ in range(self.population_size):
            individual = self.problem.random_solution()
            cost = self.problem.evaluate(individual)
            self.population.append(individual)
            self.fitness_values.append(cost)
            self.evaluations_used += 1
            self.update_best(individual.copy(), cost)
    
    def step(self, num_evaluations: int) -> Tuple[Any, float]:
        """
        Execute GA for specified number of evaluations.
        
        Args:
            num_evaluations: Number of evaluations to perform
            
        Returns:
            Tuple of (best_solution, best_cost)
        """
        evals_done = 0
        
        while evals_done < num_evaluations:
            # Create new population
            new_population = []
            new_fitness = []
            
            # Elitism: keep best individual
            best_idx = np.argmin(self.fitness_values)
            new_population.append(self.population[best_idx].copy())
            new_fitness.append(self.fitness_values[best_idx])
            
            # Generate rest of population
            while len(new_population) < self.population_size and evals_done < num_evaluations:
                # Selection (tournament)
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = self._order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < self.mutation_rate:
                    child = self._swap_mutation(child)
                
                # Evaluate child
                cost = self.problem.evaluate(child)
                new_population.append(child)
                new_fitness.append(cost)
                
                evals_done += 1
                self.evaluations_used += 1
                self.update_best(child.copy(), cost)
            
            # Replace population
            self.population = new_population
            self.fitness_values = new_fitness
            self.generation += 1
        
        return self.get_best()
    
    def _tournament_selection(self, tournament_size: int = 3) -> List[int]:
        """
        Select individual using tournament selection.
        
        Args:
            tournament_size: Number of individuals in tournament
            
        Returns:
            Selected individual
        """
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        tournament_fitness = [self.fitness_values[i] for i in indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX) operator for permutations.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child solution
        """
        size = len(parent1)
        
        # Choose two random crossover points
        point1, point2 = sorted(np.random.choice(size, 2, replace=False))
        
        # Create child with segment from parent1
        child = [-1] * size
        child[point1:point2] = parent1[point1:point2]
        
        # Fill remaining positions with parent2's order
        current_pos = point2
        for city in parent2[point2:] + parent2[:point2]:
            if city not in child:
                child[current_pos] = city
                current_pos = (current_pos + 1) % size
        
        return child
    
    def _swap_mutation(self, individual: List[int]) -> List[int]:
        """
        Swap mutation: randomly swap two cities.
        
        Args:
            individual: Solution to mutate
            
        Returns:
            Mutated solution
        """
        mutated = individual.copy()
        idx1, idx2 = np.random.choice(len(mutated), 2, replace=False)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def get_state(self) -> Dict[str, Any]:
        """Get current GA state."""
        return {
            'population': [ind.copy() for ind in self.population],
            'fitness_values': self.fitness_values.copy(),
            'generation': self.generation,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'best_solution': self.best_solution.copy() if self.best_solution else None,
            'best_cost': self.best_cost,
            'evaluations_used': self.evaluations_used
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore GA state."""
        self.population = [ind.copy() for ind in state['population']]
        self.fitness_values = state['fitness_values'].copy()
        self.generation = state['generation']
        self.mutation_rate = state['mutation_rate']
        self.crossover_rate = state['crossover_rate']
        self.best_solution = state['best_solution'].copy() if state['best_solution'] else None
        self.best_cost = state['best_cost']
        self.evaluations_used = state['evaluations_used']