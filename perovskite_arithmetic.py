import numpy as np
import math

# Perovskite structural reference
class PerovskiteStructuralUnits:
    """
    Structural units and dimensions of a typical methylammonium lead iodide (CH3NH3PbI3) perovskite
    All measurements in nanometers (nm)
    """
    # Lattice constant (cubic unit cell edge length)
    LATTICE_CONSTANT = 0.63  # nm
    
    # Ionic radii (approximate)
    METHYLAMMONIUM_RADIUS = 0.17  # nm
    LEAD_RADIUS = 0.19  # nm
    IODIDE_RADIUS = 0.22  # nm
    
    # Octahedral unit size
    OCTAHEDRAL_EDGE = 0.40  # nm (typical edge length of PbI6 octahedron)

def quantize_crystal_dimension(desired_atoms, periodicity):
    """
    Quantize crystal dimension based on atomic arrangement and periodicity.
    
    Args:
    desired_atoms (int): Desired number of atoms
    periodicity (float): Periodicity factor
    
    Returns:
    dict: Detailed information about the crystal dimension
    """
    if desired_atoms < 1 or periodicity <= 0:
        raise ValueError("Invalid atoms or periodicity")
    
    # Base calculations
    base_period = 1 / periodicity
    
    # Base number of intervals between atoms
    base_intervals = desired_atoms - 1
    
    def find_max_interval_quantization(base_period, base_intervals):
        """
        Find the maximum frequency quantization for intervals with additional precision.
        """
        # Try different multipliers to increase precision
        for n in range(1, 1000):  # Limit to reasonable multipliers
            total_intervals = base_intervals + n
            
            # Start with a high maximum frequency
            max_interval_freq = 1000000  # Arbitrarily high starting point
            
            while max_interval_freq > 0:
                # Calculate interval time for this max frequency
                interval_period = 1 / max_interval_freq
                total_interval_time = interval_period * total_intervals
                
                # Check if total interval time matches the original base period
                if math.isclose(total_interval_time, base_period, rel_tol=1e-10):
                    return {
                        "max_interval_frequency": max_interval_freq,
                        "interval_period": interval_period,
                        "base_intervals": base_intervals,
                        "total_intervals": total_intervals,
                        "additional_intervals": n
                    }
                
                # Reduce frequency if not matching
                max_interval_freq -= 1
        
        raise ValueError("Could not find precise interval quantization")
    
    # Perform max frequency interval quantization
    interval_quantization = find_max_interval_quantization(base_period, base_intervals)
    
    # Calculate crystal dimension
    def calculate_crystal_dimension():
        """
        Calculate the total crystal dimension based on atomic arrangement.
        """
        # Use lattice constant as the base unit
        total_dimension = desired_atoms * PerovskiteStructuralUnits.LATTICE_CONSTANT
        
        return {
            "atoms": desired_atoms,
            "total_dimension_nm": total_dimension,
            "lattice_constant_nm": PerovskiteStructuralUnits.LATTICE_CONSTANT,
            "dimensional_components": {
                "methylammonium_radii": desired_atoms * PerovskiteStructuralUnits.METHYLAMMONIUM_RADIUS,
                "lead_radii": desired_atoms * PerovskiteStructuralUnits.LEAD_RADIUS,
                "iodide_radii": desired_atoms * PerovskiteStructuralUnits.IODIDE_RADIUS,
                "octahedral_edges": desired_atoms * PerovskiteStructuralUnits.OCTAHEDRAL_EDGE
            }
        }
    
    # Combine quantization and dimension calculation
    crystal_dimension = calculate_crystal_dimension()
    
    return {
        **interval_quantization,
        "crystal_dimension": crystal_dimension
    }

# --- Main Program ---
if __name__ == "__main__":
    try:
        # Get input for desired number of atoms
        atoms_str = input("Enter desired number of atoms (e.g., 5): ")
        desired_atoms = int(atoms_str)

        # Get input for periodicity
        periodicity_str = input("Enter periodicity (e.g., 2): ")
        periodicity = float(periodicity_str)

        # Quantize crystal dimension
        quantized_result = quantize_crystal_dimension(desired_atoms, periodicity)
        
        # Print detailed results
        print("\n" + "=" * 50)
        print("Perovskite Crystal Dimension Quantization")
        print("=" * 50)
        
        print("\nOriginal Parameters:")
        print(f"  Desired Atoms: {quantized_result['crystal_dimension']['atoms']}")
        print(f"  Periodicity: {periodicity}")
        
        print("\nCrystal Dimension:")
        print(f"  Total Dimension: {quantized_result['crystal_dimension']['total_dimension_nm']:.4f} nm")
        print(f"  Lattice Constant: {quantized_result['crystal_dimension']['lattice_constant_nm']:.4f} nm")
        
        print("\nInterval Quantization:")
        print(f"  Result: {quantized_result['max_interval_frequency']:}")

        print(f"  Total Intervals: {quantized_result['total_intervals']}")
        
        print("\n" + "=" * 50)

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
