import numpy as np
from typing import List, Tuple
from .base_problem import BaseProblem


class TSPProblem(BaseProblem):
    """Traveling Salesman Problem implementation."""
    
    def __init__(self, distance_matrix: np.ndarray):
        """
        Initialize TSP with distance matrix.
        
        Args:
            distance_matrix: 2D numpy array where [i][j] is distance from city i to city j
        """
        super().__init__(distance_matrix)
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        
        # Validate distance matrix
        assert distance_matrix.shape[0] == distance_matrix.shape[1], "Distance matrix must be square"
        assert self.num_cities >= 3, "TSP requires at least 3 cities"
    
    def evaluate(self, solution: List[int]) -> float:
        """
        Calculate total tour distance.
        
        Args:
            solution: List of city indices representing tour order
            
        Returns:
            Total distance of the tour
        """
        if len(solution) != self.num_cities:
            raise ValueError(f"Solution must contain {self.num_cities} cities")
        
        total_distance = 0.0
        for i in range(len(solution)):
            from_city = solution[i]
            to_city = solution[(i + 1) % len(solution)]  # Wrap around to start
            total_distance += self.distance_matrix[from_city][to_city]
        
        return total_distance
    
    def random_solution(self) -> List[int]:
        """
        Generate a random tour (permutation of cities).
        
        Returns:
            Random permutation of city indices
        """
        solution = list(range(self.num_cities))
        np.random.shuffle(solution)
        return solution
    
    def get_problem_size(self) -> int:
        """
        Return number of cities.
        
        Returns:
            Number of cities in the TSP instance
        """
        return self.num_cities
    
    def is_valid(self, solution: List[int]) -> bool:
        """
        Check if solution is a valid tour.
        
        Args:
            solution: Tour to validate
            
        Returns:
            True if valid tour (permutation of all cities), False otherwise
        """
        if len(solution) != self.num_cities:
            return False
        
        # Check if it's a permutation (all cities visited exactly once)
        return set(solution) == set(range(self.num_cities))
    
    @staticmethod
    def create_random_instance(num_cities: int, seed: int = None) -> 'TSPProblem':
        """
        Create a random TSP instance with cities in 2D plane.
        
        Args:
            num_cities: Number of cities to generate
            seed: Random seed for reproducibility
            
        Returns:
            TSPProblem instance
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random city coordinates in [0, 100] x [0, 100]
        coordinates = np.random.uniform(0, 100, size=(num_cities, 2))
        
        # Compute Euclidean distance matrix
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(
                        coordinates[i] - coordinates[j]
                    )
        
        return TSPProblem(distance_matrix)
    
    @staticmethod
    def create_clustered_instance(num_cities: int, num_clusters: int = 4, 
                                    seed: int = None) -> 'TSPProblem':
        """
        Create a TSP instance with clustered cities.
        
        Args:
            num_cities: Number of cities to generate
            num_clusters: Number of clusters
            seed: Random seed for reproducibility
            
        Returns:
            TSPProblem instance
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate cluster centers
        cluster_centers = np.random.uniform(10, 90, size=(num_clusters, 2))
        
        coordinates = []
        cities_per_cluster = num_cities // num_clusters
        
        for cluster_idx in range(num_clusters):
            # Generate cities around this cluster center
            num_in_cluster = cities_per_cluster
            if cluster_idx == num_clusters - 1:
                # Last cluster gets any remaining cities
                num_in_cluster = num_cities - len(coordinates)
            
            for _ in range(num_in_cluster):
                # Add random offset around cluster center
                offset = np.random.normal(0, 5, size=2)
                city = cluster_centers[cluster_idx] + offset
                coordinates.append(city)
        
        coordinates = np.array(coordinates)
        
        # Compute distance matrix
        distance_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(
                        coordinates[i] - coordinates[j]
                    )
        
        return TSPProblem(distance_matrix)