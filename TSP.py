import numpy as np
import random
import hashlib
import time
from typing import List, Tuple, Dict, Set

class TSPPartition:
    def __init__(self, cities: List[Tuple[float, float]], distance_threshold: float = 100.0, 
                 partition_id: str = None):
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_threshold = distance_threshold
        self.distance_matrix = self._calculate_distance_matrix()
        self.partition_id = partition_id or self._generate_partition_id()
        self.best_path = None
        self.best_distance = float('inf')
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate the distance matrix between all cities."""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist_matrix[i, j] = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                                              (self.cities[i][1] - self.cities[j][1])**2)
        return dist_matrix
    
    def _generate_partition_id(self) -> str:
        """Generate a unique ID for this partition based on its content."""
        data = str(self.cities) + str(self.distance_threshold) + str(time.time())
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calculate the total distance of a path."""
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distance_matrix[path[i], path[i+1]]
        # Add distance back to starting city to complete the tour
        total_dist += self.distance_matrix[path[-1], path[0]]
        return total_dist
    
    def check_constraints(self, path: List[int]) -> List[bool]:
        """Check which distance constraints are satisfied for a path."""
        constraints = []
        for i in range(len(path)):
            next_i = (i + 1) % len(path)
            dist = self.distance_matrix[path[i], path[next_i]]
            constraints.append(dist < self.distance_threshold)
        return constraints
    
    def should_explore(self, constraint_results: List[bool]) -> bool:
        """Determine if this partition should be explored based on constraint results."""
        # If any segment is below the threshold, explore this partition
        return any(constraint_results)


class BlockchainTSPSolver:
    def __init__(self, cities: List[Tuple[float, float]], distance_threshold: float = 100.0):
        self.original_cities = cities
        self.distance_threshold = distance_threshold
        self.partitions = {}
        self.best_solution = None
        self.best_distance = float('inf')
    
    def create_initial_partitions(self, num_partitions: int = 4) -> None:
        """Create initial partitions by randomly selecting different starting points."""
        for _ in range(num_partitions):
            # Create different starting conditions for each partition
            shuffled_cities = self.original_cities.copy()
            random.shuffle(shuffled_cities)
            
            partition = TSPPartition(shuffled_cities, self.distance_threshold)
            self.partitions[partition.partition_id] = partition
    
    def create_new_partition(self, parent_partition_id: str, 
                            constraint_focus: List[bool]) -> str:
        """Create a new partition based on an existing one with modified constraints."""
        parent = self.partitions[parent_partition_id]
        
        # Create a new partition with slightly modified cities (for exploration)
        modified_cities = parent.cities.copy()
        idx1, idx2 = random.sample(range(len(modified_cities)), 2)
        modified_cities[idx1], modified_cities[idx2] = modified_cities[idx2], modified_cities[idx1]
        
        # Create new partition with potentially different threshold based on constraints
        new_threshold = parent.distance_threshold
        if sum(constraint_focus) > 0:
            # Adjust threshold slightly based on constraints
            new_threshold *= (1 + 0.1 * random.random())
        else:
            # Reduce threshold slightly to focus on tighter constraints
            new_threshold *= (1 - 0.1 * random.random())
            
        new_partition = TSPPartition(modified_cities, new_threshold)
        self.partitions[new_partition.partition_id] = new_partition
        
        return new_partition.partition_id
    
    def solve_partition(self, partition_id: str, iterations: int = 1000) -> Tuple[List[int], float]:
        """Solve a specific partition using a simple heuristic approach."""
        partition = self.partitions[partition_id]
        
        # Initialize with a random path
        current_path = list(range(partition.num_cities))
        random.shuffle(current_path)
        current_distance = partition._calculate_path_distance(current_path)
        
        best_path = current_path.copy()
        best_distance = current_distance
        
        # Simple 2-opt local search
        for _ in range(iterations):
            # Check constraints to see if we should explore this path
            constraints = partition.check_constraints(current_path)
            if not partition.should_explore(constraints):
                # If constraints not met, try a different random path
                random.shuffle(current_path)
                current_distance = partition._calculate_path_distance(current_path)
                continue
                
            # 2-opt swap
            i, j = sorted(random.sample(range(partition.num_cities), 2))
            if j - i <= 1:
                continue
                
            # Reverse the segment between i and j
            new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
            new_distance = partition._calculate_path_distance(new_path)
            
            if new_distance < current_distance:
                current_path = new_path
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance
        
        # Update partition's best solution
        partition.best_path = best_path
        partition.best_distance = best_distance
        
        # Update global best if needed
        if best_distance < self.best_distance:
            # Convert path indices back to actual cities
            self.best_solution = [partition.cities[i] for i in best_path]
            self.best_distance = best_distance
            
        return best_path, best_distance
    
    def solve(self, iterations_per_partition: int = 1000, 
              num_rounds: int = 5, partitions_per_round: int = 2) -> Tuple[List[Tuple[float, float]], float]:
        """Main solving method using blockchain-style partitioning."""
        # Create initial partitions
        self.create_initial_partitions(partitions_per_round)
        
        for round_idx in range(num_rounds):
            print(f"Round {round_idx+1}/{num_rounds}")
            
            # Process existing partitions
            results = {}
            for partition_id in list(self.partitions.keys()):
                path, distance = self.solve_partition(partition_id, iterations_per_partition)
                results[partition_id] = (path, distance)
                print(f"  Partition {partition_id}: distance = {distance:.2f}")
            
            # Create new partitions based on the most promising ones
            sorted_partitions = sorted(results.items(), key=lambda x: x[1][1])
            
            # Keep only the best partitions and create new ones
            best_partitions = [p_id for p_id, _ in sorted_partitions[:partitions_per_round]]
            
            # Remove underperforming partitions
            for p_id in list(self.partitions.keys()):
                if p_id not in best_partitions:
                    del self.partitions[p_id]
            
            # Create new partitions based on the best ones
            for parent_id in best_partitions:
                parent = self.partitions[parent_id]
                constraint_focus = parent.check_constraints(parent.best_path)
                self.create_new_partition(parent_id, constraint_focus)
        
        return self.best_solution, self.best_distance

# Example usage
def example():
    # Create a random TSP problem with 20 cities
    random.seed(42)  # For reproducibility
    num_cities = 100
    cities = [(random.uniform(0, 10000), random.uniform(0, 10000)) for _ in range(num_cities)]
    
    # Solve using blockchain-style partitioning
    solver = BlockchainTSPSolver(cities, distance_threshold=40.0)
    best_path, best_distance = solver.solve(
        iterations_per_partition=2000,
        num_rounds=5,
        partitions_per_round=4
    )
    
    print("\nFinal Solution:")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Best path (coordinates): {best_path}")
    
    # Convert coordinates back to city indices for clarity
    city_to_idx = {(x, y): i for i, (x, y) in enumerate(cities)}
    path_indices = [city_to_idx[city] for city in best_path]
    print(f"Best path (indices): {path_indices}")
    
    return cities, best_path, best_distance

if __name__ == "__main__":
    example()
