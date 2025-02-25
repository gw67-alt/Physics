#!/usr/bin/env python3
# Quantum Chirality Simulation for Negative Time Machine
# This simulation models the core quantum optical processes

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from mpl_toolkits.mplot3d import Axes3D
import time
import argparse

class NegativeTimeMachineSimulator:
    """Simulator for the Negative Time Machine quantum optical system."""
    
    def __init__(self, grid_size=100, time_steps=200, oscillator_freq=0.05):
        """Initialize the simulation parameters.
        
        Args:
            grid_size: Dimension of the 2D grid for spatial simulation
            time_steps: Number of temporal steps to simulate
            oscillator_freq: Frequency of the mechanical blocking oscillator
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.oscillator_freq = oscillator_freq
        
        # System components
        self.photon_source_intensity = 1.0
        self.blocking_threshold = 0.5
        self.mirror_reflectivity = 0.995
        self.photodiode_sensitivity = 0.8
        
        # Simulation arrays
        self.intensity_field = np.zeros((time_steps, grid_size, grid_size))
        self.chiral_signal = np.zeros(time_steps)
        
        # Initialize photon source at center of grid
        self.source_pos = (grid_size // 2, grid_size // 2)
        
    def run_simulation(self):
        """Execute the complete simulation."""
        print("Starting quantum chirality simulation...")
        
        # Initialize first frame with a photon source
        self._initialize_source()
        
        # Step through time
        for t in range(1, self.time_steps):
            # Apply mechanical oscillator blocking
            oscillator_state = self._calculate_oscillator_state(t)
            
            # Propagate photons
            self._propagate_photons(t, oscillator_state)
            
            # Apply CO2 mirror reflection
            self._apply_mirror_reflection(t)
            
            # Calculate photodiode readings
            photodiode1_reading = self._calculate_photodiode(t, 'primary')
            photodiode2_reading = self._calculate_photodiode(t, 'secondary')
            
            # Apply Laplacian operator
            laplacian_output = self._apply_laplacian(photodiode1_reading, photodiode2_reading)
            
            # Detect chirality
            self.chiral_signal[t] = self._detect_chirality(laplacian_output, t)
            
            # Progress indicator
            if t % 20 == 0:
                print(f"Simulation progress: {t/self.time_steps*100:.1f}%")
                
        print("Simulation complete.")
    
    def _initialize_source(self):
        """Set up the initial photon source."""
        x, y = self.source_pos
        # Create a Gaussian distribution of photons around the source
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i-x)**2 + (j-y)**2)
                self.intensity_field[0, i, j] = self.photon_source_intensity * np.exp(-0.1 * dist)
    
    def _calculate_oscillator_state(self, t):
        """Calculate the mechanical blocking oscillator state."""
        # Oscillator switches between blocking and unblocking
        # Using sine function to model the oscillation
        phase_shift = np.pi/4  # Add some phase shift for complexity
        return np.sin(2*np.pi*self.oscillator_freq*t + phase_shift) > self.blocking_threshold
    
    def _propagate_photons(self, t, oscillator_state):
        """Propagate photons through the system considering oscillator state."""
        # First copy previous state
        self.intensity_field[t] = self.intensity_field[t-1].copy()
        
        # Apply diffusion to model photon propagation (simplified photon dynamics)
        self.intensity_field[t] = self._diffuse(self.intensity_field[t])
        
        # Apply oscillator blocking if in blocking state
        if not oscillator_state:
            # Mechanical blocker is active - should block central region
            blocker_size = self.grid_size // 10
            center_x, center_y = self.grid_size // 2, self.grid_size // 2
            
            self.intensity_field[t, 
                                center_x-blocker_size:center_x+blocker_size, 
                                center_y-blocker_size:center_y+blocker_size] *= 0.1
    
    def _diffuse(self, field):
        """Apply diffusion to model photon propagation."""
        # Simple diffusion model using convolution with a Gaussian kernel
        sigma = 1.0
        kernel_size = 5
        x = np.linspace(-kernel_size//2, kernel_size//2, kernel_size)
        x, y = np.meshgrid(x, x)
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)  # normalize
        
        # Apply convolution
        diffused = np.zeros_like(field)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Get region around current point
                i_min = max(0, i - kernel_size//2)
                i_max = min(self.grid_size, i + kernel_size//2 + 1)
                j_min = max(0, j - kernel_size//2)
                j_max = min(self.grid_size, j + kernel_size//2 + 1)
                
                # Adjust kernel indices
                ki_min = max(0, kernel_size//2 - i)
                ki_max = kernel_size - max(0, i + kernel_size//2 + 1 - self.grid_size)
                kj_min = max(0, kernel_size//2 - j)
                kj_max = kernel_size - max(0, j + kernel_size//2 + 1 - self.grid_size)
                
                # Apply kernel
                region = field[i_min:i_max, j_min:j_max]
                k = kernel[ki_min:ki_max, kj_min:kj_max]
                diffused[i, j] = np.sum(region * k) / np.sum(k)
        
        return diffused
    
    def _apply_mirror_reflection(self, t):
        """Apply CO2 mirror reflection effects."""
        # Simplified mirror model - reflection from left side of the grid
        mirror_position = self.grid_size // 6
        reflection_intensity = self.mirror_reflectivity
        
        # Extract the column at the mirror position
        reflected_column = self.intensity_field[t, :, mirror_position].copy()
        
        # Apply reflectivity and place reflected light
        for i in range(1, mirror_position):
            reflection_x = mirror_position - i
            if reflection_x >= 0:
                self.intensity_field[t, :, reflection_x] += reflected_column * reflection_intensity
    
    def _calculate_photodiode(self, t, diode_type):
        """Calculate photodiode readings."""
        if diode_type == 'primary':
            # Primary photodiode samples from top-right quadrant
            x_start = 3 * self.grid_size // 4
            y_start = 3 * self.grid_size // 4
            reading = np.sum(self.intensity_field[t, 
                                               x_start:x_start+self.grid_size//8, 
                                               y_start:y_start+self.grid_size//8])
        else:
            # Secondary photodiode samples from bottom-right quadrant
            x_start = 3 * self.grid_size // 4
            y_start = self.grid_size // 4
            reading = np.sum(self.intensity_field[t, 
                                               x_start:x_start+self.grid_size//8, 
                                               y_start:y_start+self.grid_size//8])
        
        # Apply photodiode sensitivity
        return reading * self.photodiode_sensitivity
    
    def _apply_laplacian(self, reading1, reading2):
        """Apply Laplacian operator to photodiode readings."""
        # Create a 2D representation of the readings
        laplacian_input = np.zeros((self.grid_size, self.grid_size))
        
        # Place readings in a pattern that will generate chirality
        center = self.grid_size // 2
        laplacian_input[center, center] = reading1
        laplacian_input[center+5, center-5] = reading2
        
        # Apply Laplacian operator
        return laplace(laplacian_input)
    
    def _detect_chirality(self, laplacian_output, t):
        """Detect chirality in the processed signal."""
        # Compute a measure of chirality by looking at asymmetry
        # between quadrants of the Laplacian output
        center = self.grid_size // 2
        
        q1 = np.sum(laplacian_output[:center, :center])  # top-left
        q2 = np.sum(laplacian_output[:center, center:])  # top-right
        q3 = np.sum(laplacian_output[center:, :center])  # bottom-left
        q4 = np.sum(laplacian_output[center:, center:])  # bottom-right
        
        # Chirality measure: cross-comparison of quadrants
        chirality = (q1*q4 - q2*q3) / (q1 + q2 + q3 + q4 + 1e-10)  # avoid division by zero
        
        # Apply sign based on unblocking ∝ intensity ∝ negative time principle
        oscillator_state = self._calculate_oscillator_state(t)
        if not oscillator_state:  # if blocking (unblocked state is False)
            chirality = -chirality
            
        return chirality
    
    def visualize_results(self, save_path=None):
        """Visualize the simulation results."""
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Intensity field snapshots
        selected_times = [0, self.time_steps//4, self.time_steps//2, 3*self.time_steps//4, self.time_steps-1]
        for i, t in enumerate(selected_times):
            plt.subplot(3, 2, i+1)
            plt.imshow(self.intensity_field[t], cmap='inferno')
            plt.title(f'Intensity Field at t={t}')
            plt.colorbar(label='Intensity')
        
        # Plot 2: Chiral signal over time
        plt.subplot(3, 2, 6)
        plt.plot(np.arange(self.time_steps), self.chiral_signal, 'g-')
        plt.title('Chiral Signal vs Time')
        plt.xlabel('Time Step')
        plt.ylabel('Chirality Measure')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Highlight regions with negative time characteristics
        negative_time_regions = self.chiral_signal < 0
        plt.fill_between(np.arange(self.time_steps), 
                         self.chiral_signal, 
                         0, 
                         where=negative_time_regions,
                         color='blue', alpha=0.3, 
                         label='Potential Negative Time Regions')
        
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Results saved to {save_path}")
        else:
            plt.show()
    
    def analyze_negative_time_effects(self):
        """Analyze the simulation for potential negative time effects."""
        # Calculate time periods where the chiral signal suggests negative time
        negative_time_periods = []
        in_negative_period = False
        start_idx = 0
        
        for t in range(self.time_steps):
            if self.chiral_signal[t] < 0 and not in_negative_period:
                # Start of a negative time period
                in_negative_period = True
                start_idx = t
            elif self.chiral_signal[t] >= 0 and in_negative_period:
                # End of a negative time period
                in_negative_period = False
                negative_time_periods.append((start_idx, t-1))
        
        # If we end in a negative period, close it
        if in_negative_period:
            negative_time_periods.append((start_idx, self.time_steps-1))
        
        # Analyze each period
        print("\nAnalysis of Potential Negative Time Effects:")
        print("-------------------------------------------")
        
        if not negative_time_periods:
            print("No significant negative time effects detected.")
            return
        
        total_negative_time = sum(end-start+1 for start, end in negative_time_periods)
        percentage = total_negative_time / self.time_steps * 100
        
        print(f"Detected {len(negative_time_periods)} periods with potential negative time characteristics.")
        print(f"Total time steps with negative time signature: {total_negative_time} ({percentage:.1f}% of simulation)")
        
        # Analyze the strongest negative time effect
        min_chirality = np.min(self.chiral_signal)
        min_chirality_idx = np.argmin(self.chiral_signal)
        
        print(f"Strongest negative time effect at t={min_chirality_idx} with chirality value {min_chirality:.4f}")
        
        # Correlation analysis to detect causal anomalies
        oscillator_states = np.array([self._calculate_oscillator_state(t) for t in range(self.time_steps)])
        correlation = np.correlate(oscillator_states, self.chiral_signal, mode='full')
        max_corr_idx = np.argmax(np.abs(correlation)) - self.time_steps + 1
        
        print(f"Temporal correlation analysis suggests a time shift of {max_corr_idx} steps")
        if max_corr_idx < 0:
            print("Negative correlation lag suggests chiral effects preceding oscillator state - potential causality anomaly!")
        else:
            print("Positive correlation lag indicates conventional causal relationship.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Negative Time Machine Quantum Simulation')
    parser.add_argument('--grid-size', type=int, default=100, help='Size of the simulation grid')
    parser.add_argument('--time-steps', type=int, default=200, help='Number of time steps to simulate')
    parser.add_argument('--oscillator-freq', type=float, default=0.05, help='Frequency of mechanical oscillator')
    parser.add_argument('--save', type=str, default=None, help='Path to save visualization results')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Measure execution time
    start_time = time.time()
    
    # Initialize and run simulation
    simulator = NegativeTimeMachineSimulator(
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        oscillator_freq=args.oscillator_freq
    )
    
    simulator.run_simulation()
    simulator.visualize_results(args.save)
    simulator.analyze_negative_time_effects()
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds.")
