import numpy as np
from typing import Any, Dict, List
from collections import deque

class StateExtractor:
    """Extracts state features for the RL agent."""
    
    def __init__(self, problem: Any, algorithm_names: List[str], 
                sample_size: int = 50):
        """
        Initialize state extractor.
        
        Args:
            problem: Problem instance
            algorithm_names: List of algorithm names in the pool
            sample_size: Number of samples for landscape analysis
        """
        self.problem = problem
        self.algorithm_names = algorithm_names
        self.num_algorithms = len(algorithm_names)
        self.sample_size = sample_size
        
        # Track history for each algorithm
        self.algorithm_history = {
            name: {
                'cost_improvements': deque(maxlen=10),
                'selection_count': 0,
                'last_best_cost': float('inf')
            }
            for name in algorithm_names
        }
        
        # Track global history
        self.cost_history = deque(maxlen=50)
        self.best_cost_history = deque(maxlen=50)
    
    def extract_features(self, context_manager: Any, max_evaluations: int) -> np.ndarray:
        """
        Extract state features for RL agent.
        
        Args:
            context_manager: ContextManager instance
            max_evaluations: Maximum evaluations allowed
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # === Landscape Features (9 features) ===
        landscape_features = self._compute_landscape_features(context_manager, max_evaluations)
        features.extend(landscape_features)
        
        # === Algorithm History Features (2 * num_algorithms features) ===
        history_features = self._compute_algorithm_history_features(context_manager)
        features.extend(history_features)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_landscape_features(self, context_manager: Any, 
                                    max_evaluations: int) -> List[float]:
        """Compute landscape analysis features."""
        features = []
        
        best_solution, best_cost = context_manager.get_best()
        total_evals = context_manager.get_total_evaluations()
        
        # 1. Current Best Cost (normalized by initial cost if available)
        if len(self.best_cost_history) > 0:
            initial_cost = self.best_cost_history[0]
            normalized_cost = best_cost / initial_cost if initial_cost > 0 else 1.0
        else:
            normalized_cost = 1.0
        features.append(normalized_cost)
        
        # 2. Solution Diversity Metric (estimate from elite solutions)
        diversity = self._compute_diversity(context_manager)
        features.append(diversity)
        
        # 3. Improvement Rate (recent improvements)
        improvement_rate = self._compute_improvement_rate()
        features.append(improvement_rate)
        
        # 4. Local Optima Density (estimated from neighborhood sampling)
        local_optima_density = self._estimate_local_optima_density(best_solution)
        features.append(local_optima_density)
        
        # 5. Solution Quality Variance (from cost history)
        quality_variance = np.std(list(self.cost_history)) if len(self.cost_history) > 1 else 0.0
        quality_variance_normalized = quality_variance / (best_cost + 1e-6)
        features.append(quality_variance_normalized)
        
        # 6. Recent Best Cost Stagnation (no improvement counter)
        stagnation = self._compute_stagnation()
        features.append(stagnation)
        
        # 7. Search Space Coverage (estimate)
        coverage = min(1.0, total_evals / max_evaluations)
        features.append(coverage)
        
        # 8. Budget Consumption Ratio
        budget_ratio = total_evals / max_evaluations
        features.append(budget_ratio)
        
        # 9. Budget Remaining (inverse of consumption)
        budget_remaining = 1.0 - budget_ratio
        features.append(budget_remaining)
        
        return features
    
    def _compute_algorithm_history_features(self, context_manager: Any) -> List[float]:
        """Compute algorithm-specific history features."""
        features = []
        
        for alg_name in self.algorithm_names:
            history = self.algorithm_history[alg_name]
            
            # Feature 1: Average improvement when this algorithm was selected
            if len(history['cost_improvements']) > 0:
                avg_improvement = np.mean(history['cost_improvements'])
            else:
                avg_improvement = 0.0
            features.append(avg_improvement)
            
            # Feature 2: Selection frequency (normalized)
            total_selections = sum(h['selection_count'] for h in self.algorithm_history.values())
            if total_selections > 0:
                selection_freq = history['selection_count'] / total_selections
            else:
                selection_freq = 1.0 / self.num_algorithms  # Equal initially
            features.append(selection_freq)
        
        return features
    
    def _compute_diversity(self, context_manager: Any) -> float:
        """Compute diversity metric from elite solutions."""
        elite = context_manager.common_context['elite_solutions']
        
        if len(elite) < 2:
            return 0.0
        
        # Compute pairwise cost differences
        costs = [e['cost'] for e in elite]
        mean_cost = np.mean(costs)
        
        if mean_cost == 0:
            return 0.0
        
        diversity = np.std(costs) / mean_cost
        return min(1.0, diversity)  # Cap at 1.0
    
    def _compute_improvement_rate(self) -> float:
        """Compute recent improvement rate."""
        if len(self.best_cost_history) < 2:
            return 0.0
        
        recent = list(self.best_cost_history)[-10:]  # Last 10 entries
        
        if len(recent) < 2:
            return 0.0
        
        improvements = 0
        for i in range(1, len(recent)):
            if recent[i] < recent[i-1]:
                improvements += 1
        
        return improvements / (len(recent) - 1)
    
    def _estimate_local_optima_density(self, solution: Any) -> float:
        """Estimate local optima density through neighborhood sampling."""
        if solution is None:
            return 0.0
        
        current_cost = self.problem.evaluate(solution)
        
        # Sample a few neighbors
        num_samples = min(10, self.sample_size // 5)
        worse_neighbors = 0
        
        for _ in range(num_samples):
            neighbor = self._generate_random_neighbor(solution)
            neighbor_cost = self.problem.evaluate(neighbor)
            
            if neighbor_cost >= current_cost:
                worse_neighbors += 1
        
        # High density means many neighbors are worse (local optimum)
        density = worse_neighbors / num_samples if num_samples > 0 else 0.0
        return density
    
    def _generate_random_neighbor(self, solution: List[int]) -> List[int]:
        """Generate random neighbor using swap."""
        neighbor = solution.copy()
        i, j = np.random.choice(len(neighbor), 2, replace=False)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def _compute_stagnation(self) -> float:
        """Compute stagnation metric (normalized)."""
        if len(self.best_cost_history) < 2:
            return 0.0
        
        recent = list(self.best_cost_history)[-20:]  # Last 20 entries
        
        if len(recent) < 2:
            return 0.0
        
        # Count how many recent entries have same best cost
        best = recent[-1]
        stagnant_count = sum(1 for cost in recent if abs(cost - best) < 1e-6)
        
        return stagnant_count / len(recent)
    
    def update_history(self, algorithm_name: str, cost_before: float, 
                        cost_after: float, best_cost: float) -> None:
        """
        Update algorithm history after an algorithm step.
        
        Args:
            algorithm_name: Name of algorithm that just ran
            cost_before: Best cost before the step
            cost_after: Best cost after the step
            best_cost: Overall best cost
        """
        # Update algorithm-specific history
        history = self.algorithm_history[algorithm_name]
        history['selection_count'] += 1
        
        improvement = max(0.0, cost_before - cost_after)
        history['cost_improvements'].append(improvement)
        history['last_best_cost'] = cost_after
        
        # Update global history
        self.cost_history.append(cost_after)
        self.best_cost_history.append(best_cost)
    
    def reset(self) -> None:
        """Reset all history tracking."""
        for name in self.algorithm_names:
            self.algorithm_history[name] = {
                'cost_improvements': deque(maxlen=10),
                'selection_count': 0,
                'last_best_cost': float('inf')
            }
        
        self.cost_history.clear()
        self.best_cost_history.clear()
    
    def get_feature_dimension(self) -> int:
        """
        Get total feature dimension.
        
        Returns:
            Number of features
        """
        # 9 landscape features + 2 * num_algorithms history features
        return 9 + (2 * self.num_algorithms)