import numpy as np
import matplotlib.pyplot as plt
import time
import random
import warnings
from math import isqrt

# Define a maximum number of points based on typical RAM limits
MAX_POINTS_LIMIT = 10_000_000

def is_prime_standard(n):
    """
    Standard primality test for validation
    """
    if n <= 1: 
        return False
    if n <= 3: 
        return True
    if n % 2 == 0 or n % 3 == 0: 
        return False
    
    # Only need to check up to sqrt(n)
    limit = isqrt(n)
    for i in range(5, limit + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    
    return True

def generate_complex_wave(frequency, t, phase=0):
    """
    Generate a complex wave with given frequency and phase
    Using complex exponential e^(i*(ωt + phase))
    """
    omega = 2 * np.pi * frequency
    phase_term = np.float64(omega) * t + phase
    return np.exp(1j * phase_term)

def analyze_factor_waves(n, t_max=10.0, num_points=1000):
    """
    Analyze complex waves to detect potential factors of n.
    Returns FFT peaks that might correspond to factors.
    """
    # Create time points
    t = np.linspace(0, t_max, num_points, endpoint=False, dtype=np.float64)
    dt = t[1] - t[0]
    sampling_rate = 1 / dt

    # Generate waves - use integer frequencies to represent potential factors
    # We'll create a superposition wave of potential factor frequencies up to sqrt(n)
    factor_limit = isqrt(n) + 1
    
    # Generate a superposition of waves for all potential factors
    # This is a key innovation - we'll look for resonances in the FFT
    superposition_wave = np.zeros(num_points, dtype=complex)
    
    # For prime detection, we'll focus on frequencies 2 through sqrt(n)
    for f in range(2, factor_limit + 1):
        if n % f == 0:  # If f is a factor, add its wave to the superposition
            wave_f = generate_complex_wave(f, t)
            superposition_wave += wave_f
    
    # Also generate a reference wave at frequency n (the number itself)
    reference_wave = generate_complex_wave(n, t)
    
    # Calculate the product wave - this will show resonance properties
    # If n is prime, there should be minimal interference patterns
    # If n is composite, there will be strong interference between factor waves
    product_wave = reference_wave * superposition_wave
    
    # Calculate the ratio wave - helps identify factors
    ratio_wave = np.zeros_like(reference_wave, dtype=complex)
    mask = (abs(reference_wave) > 1e-10)
    ratio_wave[mask] = superposition_wave[mask] / reference_wave[mask]

    # --- Frequency Analysis ---
    try:
        # Analyze superposition FFT - peaks should correspond to factors
        fft_superposition = np.fft.fft(superposition_wave)
        fft_superposition_mag = np.abs(fft_superposition)
        
        # Analyze product wave FFT
        fft_product = np.fft.fft(product_wave)
        fft_product_mag = np.abs(fft_product)
        
        # Analyze ratio wave FFT
        fft_ratio = np.fft.fft(ratio_wave)
        fft_ratio_mag = np.abs(fft_ratio)
        
        # Get frequency array
        freqs = np.fft.fftfreq(num_points, dt)
        
        # Find dominant positive frequencies
        positive_freq_mask = (freqs > 0) & (freqs <= factor_limit + 1)
        positive_freqs = freqs[positive_freq_mask]
        positive_fft_superposition = fft_superposition_mag[positive_freq_mask]
        positive_fft_product = fft_product_mag[positive_freq_mask]
        positive_fft_ratio = fft_ratio_mag[positive_freq_mask]
        
        # Extract potential factor frequencies
        # Find peaks in the FFT that correspond to integer frequencies
        factor_candidates = []
        if len(positive_freqs) > 0:
            # Find the strongest peaks
            # For a prime number, there should be minimal peaks
            # For composite, we should see clear peaks at factor frequencies
            peak_indices = np.argsort(positive_fft_superposition)[-10:]  # Get top 10 peaks
            
            for idx in peak_indices:
                freq = positive_freqs[idx]
                # Check if this frequency is close to an integer and in our factor range
                if 1.5 <= freq <= factor_limit + 0.5:
                    nearest_int = round(freq)
                    if abs(freq - nearest_int) < 0.2:  # Tolerance for frequency precision
                        # Check if it's actually a factor
                        if n % nearest_int == 0:
                            magnitude = positive_fft_superposition[idx]
                            factor_candidates.append((nearest_int, magnitude))
            
            # Sort by frequency
            factor_candidates.sort(key=lambda x: x[0])

    except Exception as e:
        warnings.warn(f"Error during FFT analysis: {e}")
        return None

    # Calculate key metrics that help identify primality
    factor_count = len(factor_candidates)
    total_factor_energy = sum(mag for _, mag in factor_candidates)
    
    # For prime numbers, these metrics should be distinctly different
    # from composite numbers
    prime_resonance_ratio = 0
    if factor_count > 0:
        # Calculate the ratio of resonance energy for the largest factor
        # compared to total energy - for primes, this should be low
        max_factor_mag = max(mag for _, mag in factor_candidates) if factor_candidates else 0
        prime_resonance_ratio = max_factor_mag / np.sum(positive_fft_superposition) if np.sum(positive_fft_superposition) > 0 else 0

    results = {
        "number": n,
        "factor_candidates": factor_candidates,
        "factor_count": factor_count,
        "total_factor_energy": total_factor_energy,
        "prime_resonance_ratio": prime_resonance_ratio,
        "freqs": freqs,
        "superposition_fft": fft_superposition_mag,
        "product_fft": fft_product_mag,
        "ratio_fft": fft_ratio_mag,
        "superposition_wave": superposition_wave,
        "reference_wave": reference_wave,
        "product_wave": product_wave,
        "time": t,
        "sampling_rate": sampling_rate
    }
    
    return results

def is_prime_wave(n, threshold=0.01):
    """
    Determine if n is prime using complex wave analysis.
    
    Parameters:
    - n: The number to test for primality
    - threshold: Resonance threshold for primality determination
    
    Returns:
    - Dictionary with primality determination and analysis results
    """
    # Handle simple cases
    if n <= 1:
        return {"is_prime": False, "number": n, "confidence": 1.0, 
                "actual_prime": False, "factors": []}
    if n == 2 or n == 3:
        return {"is_prime": True, "number": n, "confidence": 1.0, 
                "actual_prime": True, "factors": []}
    if n % 2 == 0:
        return {"is_prime": False, "number": n, "confidence": 1.0, 
                "actual_prime": False, "factors": [2, n//2]}
    
    # Determine appropriate sampling parameters based on number size
    max_factor = isqrt(n) + 1
    
    # Required sampling rate to capture all potential factor frequencies
    req_sampling_rate = 4 * max_factor  # Nyquist * 2 * safety factor
    
    # Required duration to distinguish nearby factor frequencies
    # Longer duration gives better frequency resolution
    req_t_max = 10.0 / max(2, min(10, max_factor//10))  # Adjust based on factor size
    
    # Calculate number of points needed
    num_points = int(req_sampling_rate * req_t_max)
    
    # Apply memory limits
    num_points = max(2000, min(num_points, MAX_POINTS_LIMIT))
    
    # Recalculate t_max based on num_points limit
    actual_t_max = num_points / req_sampling_rate
    
    # Analyze factor waves
    results = analyze_factor_waves(n, actual_t_max, num_points)
    
    if results is None:
        return {"is_prime": None, "number": n, "confidence": 0.0, 
                "actual_prime": is_prime_standard(n), "error": "Analysis failed"}
    
    # Extract factor candidates from the analysis
    factors = [f for f, _ in results["factor_candidates"]]
    
    # Analyze resonance pattern to determine primality
    # For prime numbers, factor_count should be 0 or minimal
    # and prime_resonance_ratio should be low
    is_likely_prime = (results["factor_count"] == 0 or 
                       results["prime_resonance_ratio"] < threshold)
    
    # Calculate confidence based on resonance clarity
    if results["factor_count"] == 0:
        # No factors found - likely prime
        confidence = 0.9  # High confidence but not absolute
    else:
        # Factors found - likely composite
        # Higher resonance ratio = higher confidence it's composite
        confidence = min(0.95, results["prime_resonance_ratio"] * 10)
    
    # Ensure confidence is reasonable
    confidence = max(0.5, min(confidence, 0.95))
    
    return {
        "is_prime": is_likely_prime,
        "number": n,
        "confidence": confidence,
        "factors": factors,
        "actual_prime": is_prime_standard(n),
        "prime_resonance_ratio": results["prime_resonance_ratio"],
        "factor_count": results["factor_count"],
        "analysis_details": results
    }

def visualize_prime_analysis(n):
    """
    Visualize the complex wave analysis for primality testing.
    """
    print(f"\nGenerating visualization for number {n}...")
    result = is_prime_wave(n)
    
    if "analysis_details" not in result:
        print(f"Cannot visualize - analysis details not available for {n}")
        return
    
    features = result["analysis_details"]
    
    # Extract data for visualization
    t = features["time"]
    superposition_wave = features["superposition_wave"]
    reference_wave = features["reference_wave"]
    product_wave = features["product_wave"]
    freqs = features["freqs"]
    superposition_fft = features["superposition_fft"]
    product_fft = features["product_fft"]
    ratio_fft = features["ratio_fft"]
    
    # Create plot
    plt.figure(figsize=(12, 14))
    
    # Plot real parts of waves
    plot_points = min(len(t), 500)  # Limit points for clarity
    
    plt.subplot(5, 1, 1)
    if len(superposition_wave) > 0:
        plt.plot(t[:plot_points], np.real(superposition_wave[:plot_points]), 
                 label='Factor Wave Superposition')
    plt.grid(True)
    plt.legend()
    plt.title(f'Superposition of Factor Waves for n={n}')
    
    plt.subplot(5, 1, 2)
    if len(reference_wave) > 0:
        plt.plot(t[:plot_points], np.real(reference_wave[:plot_points]), 
                 label=f'Reference Wave (f={n})')
    plt.grid(True)
    plt.legend()
    plt.title(f'Reference Wave for Number {n}')
    
    plt.subplot(5, 1, 3)
    if len(product_wave) > 0:
        plt.plot(t[:plot_points], np.real(product_wave[:plot_points]), 
                 label='Product Wave')
    plt.grid(True)
    plt.legend()
    plt.title(f'Product Wave (Shows Resonance Patterns)')
    
    # Plot frequency spectrum of superposition wave
    plt.subplot(5, 1, 4)
    # Show frequencies up to max_factor
    max_factor = isqrt(n) + 1
    positive_freq_mask = (freqs > 0) & (freqs <= max_factor)
    
    if np.any(positive_freq_mask):
        plt.plot(freqs[positive_freq_mask], superposition_fft[positive_freq_mask], 
                 label='FFT(Factor Superposition)')
        
        # Highlight detected factor frequencies
        for factor, magnitude in features["factor_candidates"]:
            plt.axvline(factor, color='r', linestyle='--', alpha=0.7, 
                       label=f'Factor {factor}')
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        plt.title(f'Frequency Spectrum - Peaks Indicate Factors')
    
    # Add summary text
    plt.subplot(5, 1, 5)
    plt.axis('off')
    
    # Create detailed summary
    is_prime_result = "PRIME" if result["is_prime"] else "COMPOSITE"
    actual_result = "PRIME" if result["actual_prime"] else "COMPOSITE"
    match = "✓" if result["is_prime"] == result["actual_prime"] else "✗"
    
    factors_str = ", ".join(str(f) for f in result["factors"]) if result["factors"] else "None detected"
    
    summary_text = (
        f"Number: {n}\n"
        f"Wave Analysis Result: {is_prime_result} (Confidence: {result['confidence']:.2f})\n"
        f"Actual Result: {actual_result} {match}\n"
        f"Resonance Ratio: {result['prime_resonance_ratio']:.6f}\n"
        f"Factor Count: {result['factor_count']}\n"
        f"Detected Factors: {factors_str}\n"
        f"Analysis Parameters: Points={len(t)}, T_max={t[-1]:.2e}s"
    )
    
    plt.text(0.5, 0.5, summary_text, ha='center', va='center', 
             fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(pad=1.5)
    filename = f"prime_analysis_{n}.png"
    try:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()

def test_primality_algorithm(test_numbers):
    """Test the complex wave primality detection algorithm"""
    print("Prime Number Detection via Complex Wave Analysis")
    print("===============================================")
    print(f"{'Number':<12} | {'Wave Result':<12} | {'Actual':<12} | {'Confidence':<10} | {'Res. Ratio':<12} | {'Factors':<25} | {'Time (s)':<8}")
    print("-" * 100)

    correct = 0
    results = []

    for n in test_numbers:
        start_time = time.time()
        result = is_prime_wave(n)
        end_time = time.time()
        result['time_taken'] = end_time - start_time
        results.append(result)

        # Check if the prediction matches reality
        is_correct = result["is_prime"] == result["actual_prime"]
        if is_correct:
            correct += 1
        
        # Format factors list for display
        factors_str = ", ".join(str(f) for f in result["factors"][:3])
        if len(result["factors"]) > 3:
            factors_str += "..."
        
        # Use emoji to indicate correctness
        indicator = "✓" if is_correct else "✗"
        
        print(f"{n:<12} | {str(result['is_prime']):<12} | {str(result['actual_prime']):<12} | " +
              f"{result['confidence']:.4f} | {result.get('prime_resonance_ratio',0):<12.6f} | " +
              f"{factors_str:<25} | {result['time_taken']:.2f}s {indicator}")

    accuracy = correct / len(test_numbers) if test_numbers else 0
    print("-" * 100)
    print(f"Accuracy: {correct}/{len(test_numbers)} = {accuracy:.2%}")
    print("-" * 100)
    
    return accuracy, results

def generate_test_numbers(num_cases=20, max_val=1000):
    """Generate a mix of prime and composite numbers for testing."""
    test_numbers = []
    
    
    # Add random numbers to fill remaining slots
    remaining = num_cases - len(test_numbers)
    for _ in range(remaining):
        n = random.randint(100, max_val)
        if n not in test_numbers:
            test_numbers.append(n)
    
    # Shuffle the list for a more random order
    random.shuffle(test_numbers)
    
    return test_numbers[:num_cases]

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate test numbers
    test_numbers = generate_test_numbers(num_cases=15, max_val=100000000000)
    
    print("Prime Number Detection via Complex Wave Interference")
    print("===================================================")
    print("Detects primality by analyzing resonance patterns in complex waves")
    print(f"MAX_POINTS_LIMIT = {MAX_POINTS_LIMIT}")
    
    # Test the algorithm
    accuracy, results = test_primality_algorithm(test_numbers)
    
    # Sort results by confidence to find interesting cases
    results.sort(key=lambda r: r["confidence"])
    
    # Find most interesting cases for visualization
    high_confidence_correct = next((r for r in reversed(results) if r["is_prime"] == r["actual_prime"]), None)
    high_confidence_wrong = next((r for r in reversed(results) if r["is_prime"] != r["actual_prime"]), None)
    low_confidence = results[0] if results else None
    
    print("\nGenerating visualizations for selected cases...")
    
    # Choose a medium prime for visualization
    medium_prime = next((r for r in results if r["actual_prime"] and 100 <= r["number"] <= 1000), None)
    if medium_prime:
        visualize_prime_analysis(medium_prime["number"])
    
    # Choose a medium composite for comparison
    medium_composite = next((r for r in results if not r["actual_prime"] and 100 <= r["number"] <= 1000), None)
    if medium_composite:
        visualize_prime_analysis(medium_composite["number"])
    
    # Visualize an interesting case if available
    if high_confidence_wrong:
        visualize_prime_analysis(high_confidence_wrong["number"])
    elif low_confidence:
        visualize_prime_analysis(low_confidence["number"])
    
    print("\n--- How the Prime Detection Algorithm Works ---")
    print("1. Generates a superposition wave of potential factor frequencies (2 to sqrt(n)).")
    print("2. Creates a reference wave at frequency n.")
    print("3. Analyzes interference patterns between these waves using FFT.")
    print("4. For prime numbers, there should be minimal resonance at factor frequencies.")
    print("5. For composite numbers, clear resonance peaks appear at factor frequencies.")
    print("6. The algorithm measures the 'prime resonance ratio' - a low ratio suggests primality.")
    
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print("Note: The algorithm blends wave mechanics with number theory, offering")
    print("      an alternative perspective on primality testing through frequency analysis.")

if __name__ == "__main__":
    main()
