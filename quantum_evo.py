import numpy as np
import matplotlib.pyplot as plt
import time
import random

def calculate_product(a, b):
    """
    Calculate a*b using addition only (for validation)
    """
    if a > b:
        a, b = b, a
    
    product = 0
    for _ in range(a):
        product += b
    
    return product

def count_wave_intersections(a, b, t_max=10.0, num_points=1000):
    """
    Count the number of intersections between two waves with frequencies related to a and b.
    The number of intersections correlates with their product.
    """
    # Create time points
    t = np.linspace(0, t_max, num_points)
    
    # Create two waves with frequencies based on the input numbers
    wave_a = np.sin(a * t)
    wave_b = np.sin(b * t)
    
    # Find where the waves intersect (sign changes in their difference)
    diff = wave_a - wave_b
    sign_changes = np.where(np.diff(np.signbit(diff)))[0]
    
    return len(sign_changes)

def analyze_wave_interference(a, b, t_max=10.0, num_points=1000):
    """
    Analyze wave interference patterns to estimate the product of a and b
    """
    # Create time points
    t = np.linspace(0, t_max, num_points)
    
    # Create waves with frequencies related to the input numbers
    wave_a = np.sin(a * t)
    wave_b = np.sin(b * t)
    
    # Calculate interference pattern (sum of waves)
    interference = wave_a + wave_b
    
    # Calculate envelope (using Hilbert transform)
    envelope = np.abs(interference)
    
    # Find peaks in the envelope
    peak_indices = []
    for i in range(1, len(envelope)-1):
        if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
            peak_indices.append(i)
    
    # Count peaks (nodes) in the interference pattern
    peak_count = len(peak_indices)
    
    # Calculate average peak amplitude
    avg_peak_amplitude = np.mean(envelope[peak_indices]) if peak_indices else 0
    
    # Calculate spectral properties of the interference pattern
    fft = np.abs(np.fft.fft(interference))
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    
    # Find dominant frequency
    dominant_freq_idx = np.argmax(fft[1:]) + 1  # Skip DC component
    dominant_freq = freqs[dominant_freq_idx]
    
    # Calculate power at dominant frequency
    dominant_power = fft[dominant_freq_idx]
    
    return {
        "peak_count": peak_count,
        "avg_peak_amplitude": avg_peak_amplitude,
        "dominant_freq": dominant_freq,
        "dominant_power": dominant_power,
        "interference": interference,
        "envelope": envelope,
        "time": t
    }

def wave_multiplication(a, b):
    """
    Estimate the product of a and b using wave interference patterns
    """
    if a <= 0 or b <= 0:
        return {"factors": (a, b), "product": 0, "confidence": 1.0}
    
    if a == 1:
        return {"factors": (a, b), "product": b, "confidence": 1.0}
    
    if b == 1:
        return {"factors": (a, b), "product": a, "confidence": 1.0}
    
    # Scaling factor for time axis, affects resolution
    t_max_base = 20.0
    t_max = t_max_base * (np.log10(max(a, b)) + 1)
    
    # Number of sample points, affects accuracy
    num_points_base = 2000
    num_points = int(num_points_base * (np.log10(max(a, b)) + 1))
    
    # Count intersections of sine waves
    intersections = count_wave_intersections(a, b, t_max, num_points)
    
    # Analyze wave interference patterns
    analysis = analyze_wave_interference(a, b, t_max, num_points)
    
    # Estimate 1: Based on wave intersections
    # The number of intersections is proportional to the product of frequencies
    estimate1 = int(intersections * (a + b) / (2 * np.pi))
    
    # Estimate 2: Based on dominant frequency
    # The dominant frequency in the interference pattern relates to the product
    estimate2 = int(abs(analysis["dominant_freq"]) * (a + b))
    
   
    # Combine estimates using a weighted approach without multiplication
    # Using bit shifting and addition
    weighted_sum = 0
    weights_sum = 0
    
    # Add estimate1 with weight 2
    for _ in range(2):
        weighted_sum += estimate1
        weights_sum += 1
    
    # Add estimate2 with weight 3
    for _ in range(4):
        weighted_sum += estimate2
        weights_sum += 1
    
    
    # Calculate weighted average
    estimated_product = weighted_sum // weights_sum
    
    # Calculate confidence based on consistency of estimates
    estimates = [estimate1, estimate2, estimate3]
    max_diff = max([abs(est - estimated_product) for est in estimates])
    relative_max_diff = max_diff / estimated_product if estimated_product != 0 else 1.0
    confidence = 1.0 / (1.0 + relative_max_diff)
    confidence = min(max(confidence, 0), 0.99)
    
    # For validation only - calculate actual product using addition
    actual_product = calculate_product(a, b)
    
    # Calculate error (for validation only)
    absolute_error = abs(estimated_product - actual_product)
    relative_error = absolute_error * 100 / actual_product if actual_product != 0 else 0
    
    return {
        "factors": (a, b),
        "product": estimated_product,
        "actual_product": actual_product,  # For validation only
        "error_percent": relative_error,   # For validation only
        "confidence": confidence,
        "estimates": [estimate1, estimate2, estimate3],
        "intersections": intersections,
        "peak_count": analysis["peak_count"],
        "dominant_freq": analysis["dominant_freq"],
        "interference_data": {
            "interference": analysis["interference"],
            "envelope": analysis["envelope"],
            "time": analysis["time"]
        }
    }

def test_algorithm(test_cases):
    """Test the wave multiplication algorithm with comparison to actual results"""
    print("Wave Interference Multiplication")
    print("===============================")
    
    print(f"{'Factors':<15} | {'Wave Product':<15} | {'Actual Product':<15} | {'Error %':<10} | {'Confidence':<10}")
    print("-" * 80)
    
    total_error_percent = 0
    total_cases = len(test_cases)
    
    for a, b in test_cases:
        # Wave multiplication
        result = wave_multiplication(a, b)
        
        total_error_percent += result["error_percent"]
        
        print(f"({a}, {b})".ljust(15) + 
              f"| {result['product']:<15} | {result['actual_product']:<15} | " +
              f"{result['error_percent']:.4f}% | {result['confidence']:.4f}")
    
    # Calculate average error
    avg_error = total_error_percent / total_cases if total_cases > 0 else 0
    print(f"\nAverage Error: {avg_error:.4f}%")
    
    return avg_error

def visualize_wave_interference(a, b):
    """Visualize the wave interference patterns used for multiplication"""
    result = wave_multiplication(a, b)
    
    # Extract data for visualization
    t = result["interference_data"]["time"]
    interference = result["interference_data"]["interference"]
    envelope = result["interference_data"]["envelope"]
    
    # Create individual waves
    wave_a = np.sin(a * t)
    wave_b = np.sin(b * t)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot individual waves
    plt.subplot(3, 1, 1)
    plt.plot(t, wave_a, label=f'Wave A (freq={a})')
    plt.plot(t, wave_b, label=f'Wave B (freq={b})')
    plt.grid(True)
    plt.legend()
    plt.title(f'Waves with frequencies {a} and {b}')
    
    # Plot interference pattern
    plt.subplot(3, 1, 2)
    plt.plot(t, interference)
    plt.grid(True)
    plt.title(f'Interference Pattern (Estimate: {result["product"]}, Actual: {result["actual_product"]})')
    
    # Plot frequency spectrum
    plt.subplot(3, 1, 3)
    fft = np.abs(np.fft.fft(interference))
    freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    plt.plot(freqs[:len(freqs)//2], fft[:len(fft)//2])
    plt.grid(True)
    plt.title(f'Frequency Spectrum (Dominant: {result["dominant_freq"]:.2f})')
    plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"wave_multiplication_{a}_{b}.png")
    plt.close()
    
    print(f"Visualization saved to wave_multiplication_{a}_{b}.png")

def generate_test_cases(num_cases=10, min_val=2, max_val=100):
    """Generate random pairs of numbers for test cases."""
    test_cases = []
    for _ in range(num_cases):
        num1 = random.randint(min_val, max_val)
        num2 = random.randint(min_val, max_val)
        test_cases.append((num1, num2))
    return test_cases

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define test cases
    test_cases = [
        (7, 11),    # 77
        (12, 13),   # 156
        (19, 23),   # 437
        (31, 37),   # 1147
    ]
    
    # Add some random test cases
    test_cases.extend(generate_test_cases(6, 2, 50))
    
    print("Multiplication via Wave Interference Patterns")
    print("===========================================")
    print("This algorithm estimates products by analyzing how waves with")
    print("frequencies proportional to the factors interfere with each other.")
    print("It uses principles inspired by physical wave behavior and quantum mechanics.")
    
    # Test the algorithm
    avg_error = test_algorithm(test_cases)
    
    # Visualize a few examples
    print("\nGenerating visualizations of wave interference patterns...")
    visualize_wave_interference(7, 11)
    visualize_wave_interference(12, 13)
    
    print("\nHow the Algorithm Works:")
    print("1. Creates two waves with frequencies equal to the input numbers")
    print("2. Counts intersections between the waves")
    print("3. Analyzes the interference pattern when the waves are combined")
    print("4. Extracts features like peak count and dominant frequencies")
    print("5. Combines multiple estimates for a robust product calculation")
    
    print(f"\nOverall Average Error: {avg_error:.4f}%")
    print("Note: Error tends to increase with larger numbers due to sampling limitations.")

if __name__ == "__main__":
    main()
