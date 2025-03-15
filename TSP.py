import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import pandas as pd
from math import radians, sin, cos, sqrt, asin

# Previously defined TSPPartition and BlockchainTSPSolver classes would be here
# Importing only the needed classes from the previous code
# For brevity, I'm not repeating the full implementation

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth specified in decimal degrees of latitude and longitude.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Real cities dataset with coordinates (latitude, longitude)
def create_real_city_dataset():
    """Create a dataset of major US cities with their coordinates."""
    cities_data = [
        {"city": "New York", "state": "NY", "lat": 40.7128, "lon": -74.0060},
        {"city": "Los Angeles", "state": "CA", "lat": 34.0522, "lon": -118.2437},
        {"city": "Chicago", "state": "IL", "lat": 41.8781, "lon": -87.6298},
        {"city": "Houston", "state": "TX", "lat": 29.7604, "lon": -95.3698},
        {"city": "Phoenix", "state": "AZ", "lat": 33.4484, "lon": -112.0740},
        {"city": "Philadelphia", "state": "PA", "lat": 39.9526, "lon": -75.1652},
        {"city": "San Antonio", "state": "TX", "lat": 29.4241, "lon": -98.4936},
        {"city": "San Diego", "state": "CA", "lat": 32.7157, "lon": -117.1611},
        {"city": "Dallas", "state": "TX", "lat": 32.7767, "lon": -96.7970},
        {"city": "San Jose", "state": "CA", "lat": 37.3382, "lon": -121.8863},
        {"city": "Austin", "state": "TX", "lat": 30.2672, "lon": -97.7431},
        {"city": "Jacksonville", "state": "FL", "lat": 30.3322, "lon": -81.6557},
        {"city": "Fort Worth", "state": "TX", "lat": 32.7555, "lon": -97.3308},
        {"city": "Columbus", "state": "OH", "lat": 39.9612, "lon": -82.9988},
        {"city": "Charlotte", "state": "NC", "lat": 35.2271, "lon": -80.8431},
        {"city": "San Francisco", "state": "CA", "lat": 37.7749, "lon": -122.4194},
        {"city": "Indianapolis", "state": "IN", "lat": 39.7684, "lon": -86.1581},
        {"city": "Seattle", "state": "WA", "lat": 47.6062, "lon": -122.3321},
        {"city": "Denver", "state": "CO", "lat": 39.7392, "lon": -104.9903},
        {"city": "Washington", "state": "DC", "lat": 38.9072, "lon": -77.0369},
        {"city": "Boston", "state": "MA", "lat": 42.3601, "lon": -71.0589},
        {"city": "Las Vegas", "state": "NV", "lat": 36.1699, "lon": -115.1398},
        {"city": "Portland", "state": "OR", "lat": 45.5051, "lon": -122.6750},
        {"city": "Miami", "state": "FL", "lat": 25.7617, "lon": -80.1918},
        {"city": "Atlanta", "state": "GA", "lat": 33.7490, "lon": -84.3880}
    ]
    
    return pd.DataFrame(cities_data)

def calculate_distance_matrix(cities_df):
    """Calculate distance matrix between all cities using Haversine formula."""
    n = len(cities_df)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i, j] = haversine_distance(
                    cities_df.iloc[i]['lat'], cities_df.iloc[i]['lon'],
                    cities_df.iloc[j]['lat'], cities_df.iloc[j]['lon']
                )
    
    return distance_matrix

def visualize_tsp_solution(cities_df, path_indices=None):
    """Visualize the cities and the TSP solution path if provided."""
    plt.figure(figsize=(12, 8))
    
    # Plot cities
    plt.scatter(cities_df['lon'], cities_df['lat'], c='blue', s=50)
    
    # Label cities
    for i, row in cities_df.iterrows():
        plt.annotate(f"{row['city']}", (row['lon'], row['lat']), fontsize=8)
    
    # Plot path if provided
    if path_indices is not None:
        path_indices = list(path_indices) + [path_indices[0]]  # Complete the loop
        path_lats = [cities_df.iloc[i]['lat'] for i in path_indices]
        path_lons = [cities_df.iloc[i]['lon'] for i in path_indices]
        plt.plot(path_lons, path_lats, 'r-', alpha=0.6)
    
    plt.title('TSP US Cities Dataset' + (' with Solution Path' if path_indices else ''))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('tsp_cities_solution.png')
    plt.close()

def prepare_for_tsp_solver(cities_df):
    """Prepare the dataset for the TSP solver."""
    # Extract coordinates for the solver
    cities = [(row['lat'], row['lon']) for _, row in cities_df.iterrows()]
    
    # Calculate average distance for thresholding
    distance_matrix = calculate_distance_matrix(cities_df)
    avg_distance = np.mean(distance_matrix[distance_matrix > 0])
    
    return cities, distance_matrix, avg_distance

# Modified version of the BlockchainTSPSolver to accept a pre-calculated distance matrix
class BlockchainTSPSolverWithMatrix:
    def __init__(self, cities, distance_matrix, city_names, distance_threshold=None):
        self.cities = cities
        self.distance_matrix = distance_matrix
        self.city_names = city_names
        self.distance_threshold = distance_threshold or np.mean(distance_matrix[distance_matrix > 0])
        self.num_cities = len(cities)
        self.partitions = {}
        self.best_solution = None
        self.best_path_indices = None
        self.best_distance = float('inf')
    
    # Implementation would continue with methods from the original BlockchainTSPSolver
    # Modified to use the pre-calculated distance matrix
    
    # For demonstration purposes, let's implement a simple 2-opt solver without partitioning
    def solve_simple(self, iterations=10000):
        """Simple 2-opt solution for demonstration."""
        # Start with a random path
        current_path = list(range(self.num_cities))
        random.shuffle(current_path)
        
        # Calculate initial path distance
        current_distance = self._calculate_path_distance(current_path)
        
        best_path = current_path.copy()
        best_distance = current_distance
        
        # Simple 2-opt improvement
        for _ in range(iterations):
            # Select two random positions
            i, j = sorted(random.sample(range(self.num_cities), 2))
            if j - i <= 1:
                continue
            
            # Create new path with the segment reversed
            new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
            new_distance = self._calculate_path_distance(new_path)
            
            # If improvement found, keep it
            if new_distance < current_distance:
                current_path = new_path
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_path = current_path.copy()
                    best_distance = current_distance
        
        self.best_path_indices = best_path
        self.best_solution = [self.cities[i] for i in best_path]
        self.best_distance = best_distance
        
        return best_path, best_distance
    
    def _calculate_path_distance(self, path):
        """Calculate the total distance of a path using the distance matrix."""
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distance_matrix[path[i], path[i+1]]
        # Add distance back to starting city to complete the tour
        total_dist += self.distance_matrix[path[-1], path[0]]
        return total_dist
    
    def print_solution(self):
        """Print the solution in a readable format."""
        if not self.best_path_indices:
            print("No solution found yet.")
            return
        
        print("\nBest TSP Route:")
        print(f"Total Distance: {self.best_distance:.2f} km")
        
        path = self.best_path_indices + [self.best_path_indices[0]]  # Complete the loop
        
        print("\nCity-by-City Itinerary:")
        print("-" * 50)
        for i in range(len(path) - 1):
            from_city = self.city_names[path[i]]
            to_city = self.city_names[path[i+1]]
            distance = self.distance_matrix[path[i], path[i+1]]
            print(f"{i+1}. {from_city} -> {to_city}: {distance:.2f} km")
        print("-" * 50)

# Main function to run the example
def main():
    # Create and prepare the dataset
    cities_df = create_real_city_dataset()
    cities, distance_matrix, avg_distance = prepare_for_tsp_solver(cities_df)
    
    # Visualize the cities
    visualize_tsp_solution(cities_df)
    
    # Create and run the solver
    city_names = [f"{row['city']}, {row['state']}" for _, row in cities_df.iterrows()]
    solver = BlockchainTSPSolverWithMatrix(cities, distance_matrix, city_names, distance_threshold=avg_distance*1.2)
    
    # For demonstration, use the simple solver
    best_path, best_distance = solver.solve_simple(iterations=20000)
    
    # Print and visualize the solution
    solver.print_solution()
    visualize_tsp_solution(cities_df, best_path)
    
    # Save the dataset to CSV
    cities_df.to_csv('us_cities_tsp_dataset.csv', index=False)
    
    # Save distance matrix
    pd.DataFrame(distance_matrix).to_csv('distance_matrix.csv', index=False)
    
    print(f"\nDataset created with {len(cities_df)} US cities")
    print("Files saved: us_cities_tsp_dataset.csv, distance_matrix.csv, tsp_cities_solution.png")
    
    return cities_df, distance_matrix, best_path, best_distance

if __name__ == "__main__":
    main()
