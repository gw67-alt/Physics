import numpy as np
import matplotlib.pyplot as plt
import time
import random

def calculate_product(a, b):
    """
    Calculate a*b using addition only (for validation)
    """
    # Ensure a and b are integers for this validation method
    a_int, b_int = int(round(a)), int(round(b))
    if a_int < 0 or b_int < 0: return 0 # Handle non-negatives for this simple validation

    if a_int > b_int:
        a_int, b_int = b_int, a_int

    product = 0
    for _ in range(a_int):
        product += b_int

    return product

def generate_complex_wave(frequency, t, phase=0):
    """
    Generate a complex wave with given frequency and phase
    Using complex exponential e^(i*(ωt + phase))
    """
    # Using angular frequency directly
    omega = 2 * np.pi * frequency
    return np.exp(1j * (omega * t + phase))

def analyze_complex_interference(a, b, t_max=10.0, num_points=1000):
    """
    Analyze interference of complex waves to estimate key frequencies.
    Frequencies a and b here are treated as Hertz (cycles per second).
    """
    # Create time points
    t = np.linspace(0, t_max, num_points, endpoint=False) # endpoint=False for FFT
    dt = t[1] - t[0]
    sampling_rate = 1 / dt

    # Generate complex waves with frequencies a Hz and b Hz
    wave_a = generate_complex_wave(a, t)
    wave_b = generate_complex_wave(b, t)

    # Product wave: frequency should be (a+b) IF we used angular frequencies directly
    # With e^(i*2*pi*f*t), the product wave is e^(i*2*pi*(a+b)*t)
    # Its frequency is a+b Hz
    product_wave = wave_a * wave_b

    # Sum wave: contains components at a Hz and b Hz
    sum_wave = wave_a + wave_b

    # --- Frequency Analysis ---

    # Analyze Sum Wave FFT
    fft_sum = np.fft.fft(sum_wave)
    fft_sum_mag = np.abs(fft_sum)
    freqs = np.fft.fftfreq(num_points, dt)

    # Find dominant positive frequencies (should be peaks near a and b)
    # Consider only positive frequencies, ignore DC (index 0)
    positive_freq_mask = (freqs > 0) & (freqs <= sampling_rate / 2)
    positive_freqs = freqs[positive_freq_mask]
    positive_fft_sum_mag = fft_sum_mag[positive_freq_mask]

    # Find the indices of the two largest peaks in the positive spectrum
    # Handle cases where a=b (one peak) or insufficient points
    peak_indices_sum = np.argsort(positive_fft_sum_mag)[-2:] # Get indices of 2 largest
    freq_peak1 = positive_freqs[peak_indices_sum[-1]]
    if len(positive_freqs) > 1 and len(peak_indices_sum) > 1:
         # Check if the second peak is distinct enough, could be noise otherwise
         # Or handle the case where a=b, where there's only one strong peak
        peak_ratio = positive_fft_sum_mag[peak_indices_sum[0]] / positive_fft_sum_mag[peak_indices_sum[-1]] if positive_fft_sum_mag[peak_indices_sum[-1]] > 1e-9 else 0
        if peak_ratio > 0.1: # Arbitrary threshold: second peak must be at least 10% of the first
             freq_peak2 = positive_freqs[peak_indices_sum[0]]
        else:
             freq_peak2 = freq_peak1 # Assume a=b case or second peak too small
    elif len(positive_freqs) > 0:
        freq_peak2 = freq_peak1 # Only one positive frequency found or requested
    else:
        freq_peak1 = 0 # No significant positive frequencies found
        freq_peak2 = 0

    # Assign estimated a and b based on peaks
    # This assumes the FFT correctly identifies the input frequencies
    measured_a = max(freq_peak1, freq_peak2)
    measured_b = min(freq_peak1, freq_peak2)
    measured_diff = abs(measured_a - measured_b) # This is our |a-b| estimate

    # Analyze Product Wave FFT
    fft_prod = np.fft.fft(product_wave)
    fft_prod_mag = np.abs(fft_prod)
    # Freqs are the same as for sum_wave

    # Find the dominant positive frequency (should be a+b)
    positive_fft_prod_mag = fft_prod_mag[positive_freq_mask]
    if len(positive_fft_prod_mag) > 0:
        dominant_idx_prod = np.argmax(positive_fft_prod_mag)
        measured_sum = positive_freqs[dominant_idx_prod] # This is our a+b estimate
    else:
        measured_sum = 0 # No significant positive frequency found

    # --- Other Features (can be used for confidence/debugging) ---
    phase_correlation = np.sum(wave_a * np.conj(wave_b)) / num_points # Correlation using conjugate
    envelope = np.abs(sum_wave)
    real_sum = np.real(sum_wave)
    zero_crossings = np.where(np.diff(np.signbit(real_sum)))[0]


    analysis_results = {
        "measured_a": measured_a,
        "measured_b": measured_b,
        "measured_sum": measured_sum, # From product wave FFT peak freq
        "measured_diff": measured_diff, # From difference of sum wave FFT peak freqs
        "sum_consistency_check": measured_a + measured_b, # Should ideally equal measured_sum
        "phase_correlation_magnitude": np.abs(phase_correlation),
        "phase_correlation_angle": np.angle(phase_correlation),
        "zero_crossing_count": len(zero_crossings),
        "envelope_mean": np.mean(envelope),
        "envelope_std": np.std(envelope),
        "product_wave": product_wave, # For visualization
        "sum_wave": sum_wave,         # For visualization
        "time": t,                    # For visualization
        "freqs": freqs,               # For visualization
        "fft_sum_mag": fft_sum_mag,   # For visualization
        "fft_prod_mag": fft_prod_mag, # For visualization
        "sampling_rate": sampling_rate
    }

    return analysis_results

def complex_wave_multiplication(a, b):
    """
    Estimate the product of a and b using complex wave interference based on
    4ab = (a+b)^2 - (a-b)^2 identity.
    Assumes a, b are non-negative.
    """
    # Handle edge cases
    if a <= 1e-9 or b <= 1e-9: # Use tolerance for near-zero float inputs
        # Allow multiplication by zero, but handle 1 separately for potential shortcut
         if abs(a - 1) < 1e-9: return {"factors": (a, b), "product": b, "confidence": 1.0, "actual_product": calculate_product(a,b), "error_percent": 0}
         if abs(b - 1) < 1e-9: return {"factors": (a, b), "product": a, "confidence": 1.0, "actual_product": calculate_product(a,b), "error_percent": 0}
         # If a or b is zero (or very close)
         return {"factors": (a, b), "product": 0, "confidence": 1.0, "actual_product": calculate_product(a,b), "error_percent": 0}


    # --- Adjust sampling parameters based on input frequencies ---
    # Ensure Nyquist frequency > a+b
    # Ensure t_max is long enough to resolve difference frequency |a-b|
    max_freq = a + b
    min_delta_freq = abs(a - b) if abs(a - b) > 1e-6 else max(a, b) * 0.01 # Use a small fraction if a=b

    # Required sampling rate (Nyquist * 2) - add safety factor (e.g., 4x)
    req_sampling_rate = 4 * max_freq
    # Required duration to distinguish frequencies a and b, or resolve beat
    # Need at least a few cycles of the difference frequency
    req_t_max = 5 / min_delta_freq if min_delta_freq > 1e-6 else 10 / max(a,b) # 5 cycles of beat, or 10 cycles if a=b

    # Sensible limits
    req_t_max = max(1.0, min(req_t_max, 100.0)) # Limit duration for performance
    req_sampling_rate = max(100.0, req_sampling_rate) # Minimum sampling rate

    # Number of points needed
    num_points = int(req_sampling_rate * req_t_max)
    num_points = max(2000, min(num_points, 50000)) # Limit num_points

    # Recalculate t_max based on chosen num_points and sampling rate
    actual_sampling_rate = num_points / req_t_max
    actual_t_max = req_t_max # Use the duration determined

    # print(f"Debug: a={a}, b={b}")
    # print(f"Debug: req_t_max={req_t_max:.2f}, req_sampling_rate={req_sampling_rate:.2f}, num_points={num_points}")


    # Analyze complex wave interference
    features = analyze_complex_interference(a, b, actual_t_max, num_points)

    # Extract key measured frequencies for the formula
    f_sum = features["measured_sum"]
    f_diff = features["measured_diff"]

    # --- Estimate Product ---
    # Use the identity 4ab = (a+b)^2 - (a-b)^2
    # Check if measured frequencies are reasonable
    if f_sum < 1e-9: # Product wave FFT failed or inputs were zero
        estimated_product = 0
        confidence = 0.1 # Low confidence
    else:
        # Calculate raw product estimate
        estimated_product = (f_sum**2 - f_diff**2) / 4.0

    # --- Calculate Confidence ---
    # Confidence based on consistency between frequency measurements
    # 1. Consistency of sum: How close is f_sum to (measured_a + measured_b)?
    sum_check = features["sum_consistency_check"]
    sum_error_rel = abs(f_sum - sum_check) / f_sum if f_sum > 1e-9 else 1.0
    conf1 = 1.0 / (1.0 + 10 * sum_error_rel) # Penalize deviations more heavily

    # 2. Basic check: measured_a and measured_b should be close to inputs a, b
    input_a_error = abs(features["measured_a"] - a) / a if a > 1e-9 else 0
    input_b_error = abs(features["measured_b"] - b) / b if b > 1e-9 else 0
    conf2 = 1.0 / (1.0 + 5 * (input_a_error + input_b_error)) # Penalize input deviations

    # 3. Phase correlation (optional, might not be directly indicative of product accuracy)
    # phase_corr_mag = features["phase_correlation_magnitude"]
    # conf3 = phase_corr_mag

    # Combine confidence measures (e.g., weighted average or minimum)
    confidence = (conf1 + conf2) / 2.0
    confidence = min(max(confidence, 0.0), 1.0) # Clamp to [0, 1]

    # --- Final Result ---
    # Round the estimated product if inputs were integers expected
    # For now, keep it float as inputs can be float. User can round if needed.
    # Let's round to a reasonable number of decimal places if inputs look like integers
    if abs(a - round(a)) < 1e-9 and abs(b - round(b)) < 1e-9:
         final_product = round(estimated_product)
    else:
         final_product = estimated_product # Keep float for float inputs

    # Validation part
    actual_product = calculate_product(a, b) # Uses integer logic
    absolute_error = abs(final_product - actual_product)
    relative_error = absolute_error * 100.0 / actual_product if actual_product != 0 else (0 if final_product == 0 else float('inf'))


    return {
        "factors": (a, b),
        "product": final_product,
        "actual_product": actual_product,  # For validation only
        "error_percent": relative_error,   # For validation only
        "confidence": confidence,
        "debug_info": { # Include measured values for debugging
             "measured_a": features["measured_a"],
             "measured_b": features["measured_b"],
             "measured_sum_from_prod_wave": f_sum,
             "measured_sum_from_sum_wave": sum_check,
             "measured_diff": f_diff,
             "conf1_sum_consistency": conf1,
             "conf2_input_match": conf2,
             "num_points": num_points,
             "t_max": actual_t_max,
             "sampling_rate": features["sampling_rate"]
        },
        "feature_data": features # Pass full features for visualization
    }

def test_algorithm(test_cases):
    """Test the complex wave multiplication algorithm"""
    print("Complex Wave Interference Multiplication (Improved Formula)")
    print("=========================================================")
    print(f"{'Factors':<15} | {'Est. Product':<15} | {'Actual Product':<15} | {'Error %':<12} | {'Confidence':<10}")
    print("-" * 80)

    total_error_percent = 0
    valid_cases = 0

    results = []
    for a, b in test_cases:
        start_time = time.time()
        result = complex_wave_multiplication(a, b)
        end_time = time.time()
        result['time_taken'] = end_time - start_time
        results.append(result)

        # Check if error is calculable (not inf)
        if result["error_percent"] != float('inf'):
            total_error_percent += result["error_percent"]
            valid_cases += 1

        print(f"({a}, {b})".ljust(15) +
              f"| {result['product']:<15.4f} | {result['actual_product']:<15} | " +
              f"{result['error_percent']:.4f}%".ljust(12) + f" | {result['confidence']:.4f}" +
              f" | t={result['time_taken']:.2f}s")
        # Optional: print debug info for high error cases
        # if result["error_percent"] > 5.0:
        #      print(f"  Debug Info: {result['debug_info']}")


    # Calculate average error
    avg_error = total_error_percent / valid_cases if valid_cases > 0 else 0
    print(f"\nAverage Error (on {valid_cases}/{len(test_cases)} valid cases): {avg_error:.4f}%")

    return avg_error, results

def visualize_complex_waves(a, b):
    """Visualize the complex wave interference patterns and FFTs"""
    print(f"\nGenerating visualization for factors ({a}, {b})...")
    result = complex_wave_multiplication(a, b)
    features = result["feature_data"]

    # Extract data for visualization
    t = features["time"]
    sum_wave = features["sum_wave"]
    product_wave = features["product_wave"]
    freqs = features["freqs"]
    fft_sum_mag = features["fft_sum_mag"]
    fft_prod_mag = features["fft_prod_mag"]
    sampling_rate = features["sampling_rate"]

    # Create individual waves (using the same generation function)
    wave_a = generate_complex_wave(a, t)
    wave_b = generate_complex_wave(b, t)

    # Create plot
    plt.figure(figsize=(12, 12)) # Increased height

    # Plot real parts of individual waves (limit plot points for clarity if needed)
    plot_points = min(len(t), 500) # Limit to 500 points for waveform plots
    plt.subplot(5, 1, 1)
    plt.plot(t[:plot_points], np.real(wave_a[:plot_points]), label=f'Re[Wave A] (freq={a} Hz)')
    plt.plot(t[:plot_points], np.real(wave_b[:plot_points]), label=f'Re[Wave B] (freq={b} Hz)')
    plt.grid(True)
    plt.legend()
    plt.title(f'Input Complex Waves (First {t[plot_points-1]:.2f} seconds)')

    # Plot real part of sum wave (interference)
    plt.subplot(5, 1, 2)
    plt.plot(t[:plot_points], np.real(sum_wave[:plot_points]), label='Re[A + B]')
    plt.plot(t[:plot_points], np.abs(sum_wave[:plot_points]), 'r--', label='Envelope |A + B|')
    plt.grid(True)
    plt.legend()
    plt.title(f'Sum Wave (Interference)')

    # Plot real part of product wave
    plt.subplot(5, 1, 3)
    plt.plot(t[:plot_points], np.real(product_wave[:plot_points]), label='Re[A * B]')
    plt.grid(True)
    plt.legend()
    plt.title(f'Product Wave (Frequency should be near {a+b:.2f} Hz)')

    # Plot frequency spectrum of sum wave
    plt.subplot(5, 1, 4)
    # Only plot positive frequencies up to a reasonable limit beyond expected peaks
    plot_freq_limit = 1.5 * max(a, b, 1) # Show a bit beyond a and b
    positive_freq_mask = (freqs > 0) & (freqs <= plot_freq_limit)
    if np.any(positive_freq_mask):
         plt.plot(freqs[positive_freq_mask], fft_sum_mag[positive_freq_mask], label='FFT(Sum Wave)')
         plt.axvline(features['measured_a'], color='r', linestyle='--', alpha=0.7, label=f'Peak A ~ {features["measured_a"]:.2f} Hz')
         plt.axvline(features['measured_b'], color='g', linestyle='--', alpha=0.7, label=f'Peak B ~ {features["measured_b"]:.2f} Hz')
         plt.xlabel('Frequency (Hz)')
         plt.ylabel('Magnitude')
         plt.legend()
         plt.grid(True)
         plt.title(f'Frequency Spectrum (Sum Wave) - Expect peaks near {a} & {b}')
    else:
         plt.text(0.5, 0.5, 'No positive frequencies in range', ha='center', va='center')


    # Plot frequency spectrum of product wave
    plt.subplot(5, 1, 5)
    plot_freq_limit_prod = 1.5 * (a + b) if (a + b) > 0 else 1.0 # Show a bit beyond a+b
    positive_freq_mask_prod = (freqs > 0) & (freqs <= plot_freq_limit_prod)
    if np.any(positive_freq_mask_prod):
        plt.plot(freqs[positive_freq_mask_prod], fft_prod_mag[positive_freq_mask_prod], label='FFT(Product Wave)')
        plt.axvline(features['measured_sum'], color='purple', linestyle='--', alpha=0.7, label=f'Peak Sum ~ {features["measured_sum"]:.2f} Hz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True)
        plt.title(f'Frequency Spectrum (Product Wave) - Expect peak near {a+b:.2f}')
    else:
         plt.text(0.5, 0.5, 'No positive frequencies in range', ha='center', va='center')


    plt.tight_layout(pad=1.5) # Add padding between subplots
    filename = f"complex_wave_multiplication_{a}_{b}.png"
    try:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()

    # Phase visualization remains largely the same as before (optional)
    # visualize_phase(a, b, features) # You can call a separate phase visualization if needed


def generate_test_cases(num_cases=10, min_val=2, max_val=20):
    """Generate random pairs of numbers for test cases."""
    test_cases = []
    # Add some specific cases
    test_cases.extend([(3, 5), (7, 7), (2, 10), (15, 4)])
    # Add random cases
    for _ in range(num_cases - len(test_cases)):
        num1 = random.randint(min_val, max_val)
        num2 = random.randint(min_val, max_val)
        test_cases.append((num1, num2))
    # Add a case with closer numbers
    val = random.randint(min_val + 1, max_val)
    test_cases.append((val, val - 1))
    # Add a case with floats?
    # test_cases.append((3.5, 7.2))
    return test_cases

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define test cases (using integers for easy validation with calculate_product)
    test_cases = generate_test_cases(num_cases=10, min_val=2, max_val=25) # Increased max_val slightly

    print("Multiplication via Complex Wave Interference (Improved)")
    print("====================================================")
    print("Estimates products using the identity 4ab = (a+b)^2 - (a-b)^2")
    print("where (a+b) and |a-b| are measured from complex wave FFTs.")
    print("Inputs a, b are treated as frequencies in Hertz.")

    # Test the algorithm
    avg_error, results = test_algorithm(test_cases)

    # Find best and worst cases based on error
    results.sort(key=lambda r: r['error_percent'] if r['error_percent'] != float('inf') else 1e10)
    best_case = results[0]
    worst_case = results[-1]

    print("\nGenerating visualizations for specific cases...")
    # Visualize a standard case
    visualize_complex_waves(7, 11)
    # Visualize the best performing case from tests
    if best_case:
         visualize_complex_waves(best_case['factors'][0], best_case['factors'][1])
    # Visualize the worst performing case (if error > 0.1%)
    if worst_case and worst_case['error_percent'] > 0.1:
         visualize_complex_waves(worst_case['factors'][0], worst_case['factors'][1])
    else:
         # Visualize another random case if worst case was good
         visualize_complex_waves(12, 13)


    print("\n--- How the Improved Algorithm Works ---")
    print("1. Generates complex waves: wave_a = exp(i*2*pi*a*t), wave_b = exp(i*2*pi*b*t).")
    print("2. Creates Sum Wave (A+B) and Product Wave (A*B).")
    print("3. Analyzes FFT of Sum Wave: Finds peaks corresponding to 'a' and 'b'. Calculates measured_diff = |peak_a - peak_b|.")
    print("4. Analyzes FFT of Product Wave: Finds peak corresponding to 'a+b'. Sets measured_sum = peak_(a+b).")
    print("5. Calculates product estimate: result = (measured_sum^2 - measured_diff^2) / 4.0.")
    print("6. Confidence is based on consistency of FFT measurements.")

    print(f"\nOverall Average Error: {avg_error:.4f}%")
    print("Note: Accuracy depends heavily on FFT resolution, requiring sufficient time duration (t_max)")
    print("      and sampling rate (num_points) to accurately distinguish frequencies a, b, and a+b.")


if __name__ == "__main__":
    main()
