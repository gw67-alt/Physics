import numpy as np
import random
import hashlib
import time
from typing import List, Tuple, Dict
from math import radians, cos, sin, sqrt, atan2
import matplotlib.pyplot as plt
import mplcursors

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
        """Calculate the distance matrix between all cities using the Haversine formula."""
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    dist_matrix[i, j] = self._haversine(self.cities[i], self.cities[j])
        return dist_matrix
    
    def _haversine(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate the Haversine distance between two points on the Earth."""
        R = 6371.0  # Radius of the Earth in kilometers
        lat1, lon1 = radians(coord1[0]), radians(coord1[1])
        lat2, lon2 = radians(coord2[0]), radians(coord2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance
    
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
    # List of real places with their coordinates (latitude, longitude)
    real_places = [
        ("New York", (40.7128, -74.0060)),
         ("Los Angeles", (34.0522, -118.2437)),
        ("Chicago", (41.8781, -87.6298)),
        ("Houston", (29.7604, -95.3698)),
        ("Phoenix", (33.4484, -112.0740)),
        ("Philadelphia", (39.9526, -75.1652)),
        ("San Antonio", (29.4241, -98.4936)),
        ("San Diego", (32.7157, -117.1611)),
        ("Dallas", (32.7767, -96.7970)),
        ("San Jose", (37.3382, -121.8863)),
        ("Austin", (30.2672, -97.7431)),
        ("Jacksonville", (30.3322, -81.6557)),
        ("Fort Worth", (32.7555, -97.3308)),
        ("Columbus", (39.9612, -82.9988)),
        ("Charlotte", (35.2271, -80.8431)),
        ("San Francisco", (37.7749, -122.4194)),
        ("Indianapolis", (39.7684, -86.1581)),
        ("Seattle", (47.6062, -122.3321)),
        ("Denver", (39.7392, -104.9903)),
        ("Washington", (38.9072, -77.0369)),
        ("Boston", (42.3601, -71.0589)),
        ("El Paso", (31.7619, -106.4850)),
        ("Nashville", (36.1627, -86.7816)),
        ("Detroit", (42.3314, -83.0458)),
        ("Oklahoma City", (35.4676, -97.5164)),
        ("Portland", (45.5051, -122.6750)),
        ("Las Vegas", (36.1699, -115.1398)),
        ("Memphis", (35.1495, -90.0490)),
        ("Louisville", (38.2527, -85.7585)),
        ("Baltimore", (39.2904, -76.6122)),
        ("Milwaukee", (43.0389, -87.9065)),
        ("Albuquerque", (35.0844, -106.6504)),
        ("Tucson", (32.2226, -110.9747)),
        ("Fresno", (36.7378, -119.7871)),
        ("Sacramento", (38.5816, -121.4944)),
        ("Mesa", (33.4152, -111.8315)),
        ("Kansas City", (39.0997, -94.5786)),
        ("Atlanta", (33.7490, -84.3880)),
        ("Miami", (25.7617, -80.1918)),
        ("Raleigh", (35.7796, -78.6382)),
        ("Omaha", (41.2565, -95.9345)),
        ("Colorado Springs", (38.8339, -104.8214)),
        ("Virginia Beach", (36.8529, -75.9780)),
        ("Long Beach", (33.7701, -118.1937)),
        ("Oakland", (37.8044, -122.2711)),
        ("Minneapolis", (44.9778, -93.2650)),
        ("Tulsa", (36.1539, -95.9928)),
        ("Arlington", (32.7357, -97.1081)),
        ("New Orleans", (29.9511, -90.0715)),
        ("Wichita", (37.6872, -97.3301)),
        ("Cleveland", (41.4993, -81.6944)),
        ("Tampa", (27.9506, -82.4572)),
        ("Bakersfield", (35.3733, -119.0187)),
        ("Aurora", (39.7294, -104.8319)),
        ("Anaheim", (33.8366, -117.9143)),
        ("Honolulu", (21.3069, -157.8583)),
        ("Santa Ana", (33.7455, -117.8677)),
        ("Riverside", (33.9806, -117.3755)),
        ("Corpus Christi", (27.8006, -97.3964)),
        ("Lexington", (38.0406, -84.5037)),
        ("Henderson", (36.0395, -114.9817)),
        ("Stockton", (37.9577, -121.2908)),
        ("Saint Paul", (44.9537, -93.0900)),
        ("Cincinnati", (39.1031, -84.5120)),
        ("St. Louis", (38.6270, -90.1994)),
        ("Pittsburgh", (40.4406, -79.9959)),
        ("Greensboro", (36.0726, -79.7920)),
        ("Anchorage", (61.2181, -149.9003)),
        ("Lincoln", (40.8136, -96.7026)),
        ("Plano", (33.0198, -96.6989)),
        ("Orlando", (28.5383, -81.3792)),
        ("Irvine", (33.6846, -117.8265)),
        ("Newark", (40.7357, -74.1724)),
        ("Durham", (35.9940, -78.8986)),
        ("Chula Vista", (32.6401, -117.0842)),
        ("Toledo", (41.6528, -83.5379)),
        ("Fort Wayne", (41.0793, -85.1394)),
        ("St. Petersburg", (27.7676, -82.6403)),
        ("Laredo", (27.5306, -99.4803)),
        ("Jersey City", (40.7178, -74.0431)),
        ("Chandler", (33.3062, -111.8413)),
        ("Madison", (43.0731, -89.4012)),
        ("Lubbock", (33.5779, -101.8552)),
        ("Scottsdale", (33.4942, -111.9261)),
        ("Reno", (39.5296, -119.8138)),
        ("Glendale", (33.5387, -112.1860)),
        ("Buffalo", (42.8864, -78.8784)),
        ("North Las Vegas", (36.1989, -115.1175)),
        ("Gilbert", (33.3528, -111.7890)),
        ("Winston-Salem", (36.0999, -80.2442)),
        ("Chesapeake", (36.7682, -76.2875)),
        ("Norfolk", (36.8508, -76.2859)),
        ("Fremont", (37.5483, -121.9886)),
        ("Garland", (32.9126, -96.6389)),
        ("Irving", (32.8140, -96.9489)),
        ("Hialeah", (25.8576, -80.2781)),
        ("Richmond", (37.5407, -77.4360)),
        ("Boise", (43.6150, -116.2023)),
        ("Spokane", (47.6588, -117.4260)),
        ("Baton Rouge", (30.4515, -91.1871)),
        ("Tacoma", (47.2529, -122.4443)),
        ("San Bernardino", (34.1083, -117.2898)),
        ("Modesto", (37.6391, -120.9969)),
        ("Fontana", (34.0922, -117.4350)),
        ("Des Moines", (41.5868, -93.6250)),
        ("Moreno Valley", (33.9425, -117.2297)),
        ("Santa Clarita", (34.3917, -118.5426)),
        ("Fayetteville", (35.0527, -78.8784)),
        ("Birmingham", (33.5186, -86.8104)),
        ("Oxnard", (34.1975, -119.1771)),
        ("Rochester", (43.1566, -77.6088)),
        ("Port St. Lucie", (27.2730, -80.3582)),
        ("Grand Rapids", (42.9634, -85.6681)),
        ("Huntsville", (34.7304, -86.5861)),
        ("Salt Lake City", (40.7608, -111.8910)),
        ("Frisco", (33.1507, -96.8236)),
        ("Yonkers", (40.9312, -73.8987)),
        ("Amarillo", (35.2219, -101.8313)),
        ("Glendale", (34.1425, -118.2551)),
        ("Huntington Beach", (33.6595, -117.9988)),
        ("McKinney", (33.1972, -96.6398)),
        ("Montgomery", (32.3792, -86.3077)),
        ("Augusta", (33.4735, -82.0105)),
        ("Aurora", (41.7606, -88.3201)),
        ("Akron", (41.0814, -81.5190)),
        ("Little Rock", (34.7465, -92.2896)),
        ("Tempe", (33.4255, -111.9400)),
        ("Columbus", (32.4609, -84.9877)),
        ("Overland Park", (38.9822, -94.6708)),
        ("Grand Prairie", (32.7459, -96.9978)),
        ("Tallahassee", (30.4383, -84.2807)),
        ("Cape Coral", (26.5629, -81.9495)),
        ("Mobile", (30.6954, -88.0399)),
        ("Knoxville", (35.9606, -83.9207)),
        ("Shreveport", (32.5252, -93.7502)),
        ("Worcester", (42.2626, -71.8023)),
        ("Ontario", (34.0633, -117.6509)),
        ("Vancouver", (45.6387, -122.6615)),
        ("Sioux Falls", (43.5446, -96.7311)),
        ("Chattanooga", (35.0456, -85.3097)),
        ("Brownsville", (25.9017, -97.4975)),
        ("Fort Lauderdale", (26.1224, -80.1373)),
        ("Providence", (41.8240, -71.4128)),
        ("Newport News", (37.0871, -76.4730)),
        ("Rancho Cucamonga", (34.1064, -117.5931)),
        ("Santa Rosa", (38.4405, -122.7144)),
        ("Oceanside", (33.1959, -117.3795)),
        ("Salem", (44.9429, -123.0351)),
        ("Elk Grove", (38.4088, -121.3716)),
        ("Garden Grove", (33.7743, -117.9378)),
        ("Pembroke Pines", (26.0078, -80.2963)),
        ("Peoria", (33.5806, -112.2374)),
        ("Eugene", (44.0521, -123.0868)),
        ("Corona", (33.8753, -117.5664)),
        ("Cary", (35.7915, -78.7811)),
        ("Springfield", (37.2089, -93.2923)),
        ("Fort Collins", (40.5853, -105.0844)),
        ("Hayward", (37.6688, -122.0808)),
        ("Lancaster", (34.6868, -118.1542)),
        ("Alexandria", (38.8048, -77.0469)),
        ("Macon", (32.8407, -83.6324)),
        ("Sunnyvale", (37.3688, -122.0363)),
        ("Pomona", (34.0551, -117.7490)),
        ("Hollywood", (26.0112, -80.1495)),
        ("Clarksville", (36.5298, -87.3595)),
        ("Paterson", (40.9168, -74.1718)),
        ("Naperville", (41.7508, -88.1535)),
        ("Frisco", (33.1507, -96.8236)),
        ("Mesquite", (32.7668, -96.5992)),
        ("Savannah", (32.0835, -81.0998)),
        ("Syracuse", (43.0481, -76.1474)),
        ("Dayton", (39.7589, -84.1916)),
        ("Pasadena", (29.6911, -95.2091)),
        ("Orange", (33.7879, -117.8531)),
        ("Fullerton", (33.8704, -117.9243))
    ]

    # Extract coordinates from the list of places
    cities = [coords for name, coords in real_places]

    # Solve using blockchain-style partitioning
    solver = BlockchainTSPSolver(cities, distance_threshold=700.0)  # Adjust distance threshold for realistic travel distances
    best_path, best_distance = solver.solve(
        iterations_per_partition=2000,
        num_rounds=5,
        partitions_per_round=4
    )

    print("\nFinal Solution:")
    print(f"Best distance: {best_distance:.2f}")
    print(f"Best path (coordinates): {best_path}")

    # Convert coordinates back to city names for clarity
    coord_to_city = {coords: name for name, coords in real_places}
    path_names = [coord_to_city[coords] for coords in best_path]
    print(f"Best path (cities): {path_names}")

    # Visualization using matplotlib
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot the cities as points
    x_coords = [coord[1] for coord in cities]  # longitude
    y_coords = [coord[0] for coord in cities]  # latitude
    ax.scatter(x_coords, y_coords, color='blue', s=100)

    # Plot the path
    path_coords = best_path
    path_coords.append(best_path[0])  # to make a complete loop
    
    path_x = [coord[1] for coord in path_coords]  # longitude
    path_y = [coord[0] for coord in path_coords]  # latitude
    ax.plot(path_x, path_y, 'r-', linewidth=2)

    # Add city labels
    for i, city in enumerate([name for name, _ in real_places]):
        ax.annotate(city, (x_coords[i], y_coords[i]), fontsize=12, 
                   xytext=(5, 5), textcoords='offset points')

    plt.title('TSP Path Visualization', fontsize=16)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.grid(True)
    
    # Adjust the plot limits to show all US cities with some padding
    ax.set_xlim(min(x_coords) - 5, max(x_coords) + 5)
    ax.set_ylim(min(y_coords) - 5, max(y_coords) + 5)
    mplcursors.cursor(hover=True)
    plt.show()

    return cities, best_path, best_distance

if __name__ == "__main__":
    example()
