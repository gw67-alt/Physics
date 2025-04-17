import numpy as np
import matplotlib.pyplot as plt
import time
import random
import warnings # To issue warnings if needed

# Define a maximum number of points based on typical RAM limits
# Adjust this value based on your system's available memory
# 10 million complex128 points = 10e6 * 16 bytes = 160 MB per array (wave_a, wave_b, etc.)
MAX_POINTS_LIMIT = 10_000_000 # Example: 10 million points

def calculate_product_standard(a, b):
    """
    Calculate a*b using standard multiplication (for validation with large numbers).
    Rounds result if inputs appear to be integers.
    """
    product = float(a) * float(b)
    # Round validation product if inputs look like integers
    if abs(a - round(a)) < 1e-9 and abs(b - round(b)) < 1e-9:
        return round(product)
    else:
        return product

def generate_complex_wave(frequency, t, phase=0):
    """
    Generate a complex wave with given frequency and phase
    Using complex exponential e^(i*(ωt + phase)).
    Handles potentially large frequency * t values carefully.
    """
    # Using angular frequency directly
    omega = 2 * np.pi * frequency
    # Calculate phase term - use float64 for precision
    phase_term = np.float64(omega) * t + phase
    return np.exp(1j * phase_term)

def analyze_complex_interference(a, b, t_max=10.0, num_points=1000):
    """
    Analyze interference of complex waves to estimate key frequencies.
    Frequencies a and b here are treated as Hertz (cycles per second).
    MODIFIED: Doesn't return full waves to save memory. Uses float64 time. Handles errors.
    """
    # Create time points
    # Use float64 for time vector to maintain precision with large t_max * freq
    t = np.linspace(0, t_max, num_points, endpoint=False, dtype=np.float64)
    if num_points <= 1: # Need at least 2 points for dt
        warnings.warn(f"num_points ({num_points}) too small.")
        return None # Indicate failure
    dt = t[1] - t[0]
    if dt <= 1e-15: # Check for dt being too small or zero
        warnings.warn(f"Calculated dt is too small ({dt}). Check t_max ({t_max}) and num_points ({num_points}).")
        return None # Indicate failure
    sampling_rate = 1 / dt

    # --- Generate complex waves ---
    # Use try-except for potential memory errors with huge arrays
    try:
        wave_a = generate_complex_wave(a, t)
        wave_b = generate_complex_wave(b, t)

        # Product wave
        product_wave = wave_a * wave_b

        # Sum wave
        sum_wave = wave_a + wave_b

    except MemoryError:
        warnings.warn(f"MemoryError encountered generating waves for N={num_points}. Reduce MAX_POINTS_LIMIT.")
        return None # Indicate failure
    except Exception as e:
        warnings.warn(f"Error generating waves: {e}")
        return None

    # --- Frequency Analysis ---
    try:
        # Analyze Sum Wave FFT
        fft_sum = np.fft.fft(sum_wave)
        fft_sum_mag = np.abs(fft_sum)
        freqs = np.fft.fftfreq(num_points, dt)

        # Find dominant positive frequencies (should be peaks near a and b)
        positive_freq_mask = (freqs > 1e-9) & (freqs <= sampling_rate / 2) # Exclude strict DC
        positive_freqs = freqs[positive_freq_mask]
        positive_fft_sum_mag = fft_sum_mag[positive_freq_mask]

        if len(positive_freqs) == 0:
            warnings.warn(f"No significant positive frequencies found in sum wave FFT for a={a}, b={b}. Check sampling.")
            measured_a, measured_b, measured_diff = 0, 0, 0
        else:
            num_peaks_to_find = min(2, len(positive_freqs))
            # Ensure we don't request more peaks than available frequencies
            if num_peaks_to_find > 0:
                peak_indices_sum = np.argsort(positive_fft_sum_mag)[-num_peaks_to_find:]
                freq_peak1 = positive_freqs[peak_indices_sum[-1]]

                if num_peaks_to_find > 1:
                    peak_ratio = positive_fft_sum_mag[peak_indices_sum[0]] / positive_fft_sum_mag[peak_indices_sum[-1]] if positive_fft_sum_mag[peak_indices_sum[-1]] > 1e-9 else 0
                    distinct_threshold = 0.05 if abs(a-b) < 0.1 * max(a,b,1) else 0.1
                    if peak_ratio > distinct_threshold :
                         freq_peak2 = positive_freqs[peak_indices_sum[0]]
                    else:
                         freq_peak2 = freq_peak1 # Assume a=b case or second peak too small
                else: # Only one peak found
                    freq_peak2 = freq_peak1

                measured_a = max(freq_peak1, freq_peak2)
                measured_b = min(freq_peak1, freq_peak2)
                measured_diff = abs(measured_a - measured_b)
            else: # No positive frequencies matched the mask initially
                 measured_a, measured_b, measured_diff = 0, 0, 0


        # Analyze Product Wave FFT
        fft_prod = np.fft.fft(product_wave)
        fft_prod_mag = np.abs(fft_prod)

        # Find the dominant positive frequency (should be a+b)
        positive_fft_prod_mag = fft_prod_mag[positive_freq_mask]
        if len(positive_fft_prod_mag) > 0:
            dominant_idx_prod = np.argmax(positive_fft_prod_mag)
            measured_sum = positive_freqs[dominant_idx_prod] # This is our a+b estimate
        else:
            warnings.warn(f"No significant positive frequencies found in product wave FFT for a={a}, b={b}.")
            measured_sum = 0

    except MemoryError:
        warnings.warn(f"MemoryError encountered during FFT for N={num_points}. Reduce MAX_POINTS_LIMIT.")
        return None # Indicate failure
    except Exception as e:
        warnings.warn(f"Error during FFT analysis: {e}")
        return None


    # --- Other Features (can be used for confidence/debugging) ---
    try:
        phase_correlation = np.sum(wave_a * np.conj(wave_b)) / num_points # Correlation using conjugate
        envelope = np.abs(sum_wave)
        real_sum = np.real(sum_wave)
        zero_crossings = np.where(np.diff(np.signbit(real_sum)))[0]
    except Exception as e:
        warnings.warn(f"Error calculating other features: {e}")
        phase_correlation, envelope, zero_crossings = complex(0,0), np.array([0]), []


    analysis_results = {
        "measured_a": measured_a,
        "measured_b": measured_b,
        "measured_sum": measured_sum, # From product wave FFT peak freq
        "measured_diff": measured_diff, # From difference of sum wave FFT peak freqs
        "sum_consistency_check": measured_a + measured_b, # Should ideally equal measured_sum
        "phase_correlation_magnitude": np.abs(phase_correlation),
        "phase_correlation_angle": np.angle(phase_correlation),
        "zero_crossing_count": len(zero_crossings),
        "envelope_mean": np.mean(envelope) if len(envelope)>0 else 0,
        "envelope_std": np.std(envelope) if len(envelope)>0 else 0,
        # Avoid storing large waves/FFTs - store only necessary info
        # "product_wave": product_wave, # Removed
        # "sum_wave": sum_wave,         # Removed
        # "time": t,                    # Removed (or store subset)
        # "freqs": freqs,               # Removed (or store subset)
        # "fft_sum_mag": fft_sum_mag,   # Removed (or store subset)
        # "fft_prod_mag": fft_prod_mag, # Removed (or store subset)
        "sampling_rate": sampling_rate
    }
    # Clear large intermediate arrays explicitly
    del wave_a, wave_b, sum_wave, product_wave, fft_sum, fft_prod, fft_sum_mag, fft_prod_mag, freqs, t
    return analysis_results

def complex_wave_multiplication(a, b):
    """
    Estimate the product of a and b using complex wave interference based on
    4ab = (a+b)^2 - (a-b)^2 identity.
    Assumes a, b are non-negative. Handles higher numbers with memory limits.
    MODIFIED: Sampling logic updated for high numbers. Validation uses standard multiply.
    """
    # Handle edge cases
    if a < 1e-9 or b < 1e-9: # Use tolerance for near-zero float inputs
         if abs(a - 1) < 1e-9: return {"factors": (a, b), "product": b, "confidence": 1.0, "actual_product": calculate_product_standard(a,b), "error_percent": 0, "debug_info": {"status": "Shortcut a=1"}}
         if abs(b - 1) < 1e-9: return {"factors": (a, b), "product": a, "confidence": 1.0, "actual_product": calculate_product_standard(a,b), "error_percent": 0, "debug_info": {"status": "Shortcut b=1"}}
         return {"factors": (a, b), "product": 0, "confidence": 1.0, "actual_product": calculate_product_standard(a,b), "error_percent": 0, "debug_info": {"status": "Input near zero"}}

    # --- Adjust sampling parameters based on input frequencies ---
    max_freq = float(a + b)
    min_delta_freq = abs(a - b) if abs(a - b) > 1e-9 else max(a, b, 1.0) * 0.01 # Use max(a,b,1) to avoid 0

    # Required sampling rate: Nyquist * 2 * safety_factor (e.g., 2.5 * 2 = 5)
    req_sampling_rate = max(5.0 * max_freq, 100.0) # Min 100 Hz, or 5x max frequency

    # Required duration: Aim for enough cycles for resolution
    cycles_to_resolve = 10.0
    t_res = cycles_to_resolve / min_delta_freq if min_delta_freq > 1e-9 else float('inf')
    t_base = 20.0 / max(a, b, 1.0) # ~20 cycles of base frequency

    # Choose duration, minimum 0.01s
    req_t_max = max(t_base, 0.01)
    if min_delta_freq > 1e-9:
        req_t_max = max(req_t_max, t_res)
    req_t_max = min(req_t_max, 100.0) # Limit duration to 100s max

    # Ideal number of points
    ideal_num_points = int(req_sampling_rate * req_t_max)

    # Apply cap to num_points (MEMORY/COMPUTATION LIMIT)
    num_points = max(2000, min(ideal_num_points, MAX_POINTS_LIMIT))

    # --- Adjust t_max if num_points was capped ---
    actual_t_max = num_points / req_sampling_rate

    # Check if adjusted t_max is sufficient
    min_req_t_max_for_res = (cycles_to_resolve / min_delta_freq if min_delta_freq > 1e-9 else 0)
    min_req_t_max_for_base = 10.0 / max(a, b, 1.0) # Need maybe 10 cycles?
    min_allowable_t_max = max(min_req_t_max_for_res, min_req_t_max_for_base, 0.005) # Absolute min time

    t_max_warning = ""
    time_confidence_penalty = 1.0
    if actual_t_max < min_allowable_t_max and ideal_num_points > num_points:
         t_max_warning = (f"Warning: t_max reduced to {actual_t_max:.2e}s (from req {req_t_max:.2e}s) "
                          f"due to N limit ({num_points}). May impact resolution. "
                          f"Min needed ~{min_allowable_t_max:.2e}s.")
         warnings.warn(t_max_warning)
         time_confidence_penalty = np.clip(actual_t_max / min_allowable_t_max, 0.5, 1.0)

    actual_sampling_rate = num_points / actual_t_max

    # --- Analyze complex wave interference ---
    features = analyze_complex_interference(a, b, actual_t_max, num_points)

    if features is None: # Handle analysis failure
        return {
            "factors": (a, b), "product": np.nan, "confidence": 0.0,
            "actual_product": calculate_product_standard(a,b), "error_percent": float('inf'),
            "debug_info": {"status": "Analysis failed (MemoryError or other)"}
        }

    # Extract key measured frequencies
    f_sum = features["measured_sum"]
    f_diff = features["measured_diff"]

    # --- Estimate Product ---
    estimated_product = np.nan # Default to NaN
    debug_reason = ""
    if f_sum < 1e-9 or features["sampling_rate"] < max_freq * 2:
        estimated_product = 0 # Assume zero if sum freq not found or sampling inadequate
        confidence_calc = 0.1 * time_confidence_penalty
        debug_reason = "Low f_sum or inadequate sampling"
    else:
        # Calculate raw product estimate using the identity
        estimated_product = (f_sum**2 - f_diff**2) / 4.0
        # --- Calculate Confidence ---
        sum_check = features["sum_consistency_check"]
        sum_error_rel = abs(f_sum - sum_check) / f_sum if f_sum > 1e-9 else 1.0
        conf1 = 1.0 / (1.0 + 15 * sum_error_rel) # Penalize sum inconsistency

        input_a_error = abs(features["measured_a"] - a) / max(a, 1e-9)
        input_b_error = abs(features["measured_b"] - b) / max(b, 1e-9)
        error_scale = np.log10(max(a, b, 10))
        conf2 = 1.0 / (1.0 + error_scale * 5 * (input_a_error + input_b_error)) # Penalize input match deviation

        confidence_calc = ((conf1 + conf2) / 2.0) * time_confidence_penalty
        confidence_calc = min(max(confidence_calc, 0.0), 1.0) # Clamp

    # --- Final Result ---
    final_product = estimated_product
    # Round if inputs look like integers and result is valid
    if abs(a - round(a)) < 1e-9 and abs(b - round(b)) < 1e-9 and not np.isnan(final_product):
         final_product = round(final_product)

    # --- Validation part ---
    actual_product_validation = calculate_product_standard(a, b)
    absolute_error = abs(final_product - actual_product_validation) if not np.isnan(final_product) else np.inf
    relative_error = absolute_error * 100.0 / actual_product_validation if abs(actual_product_validation) > 1e-12 else (0 if abs(final_product) < 1e-12 else float('inf'))
    # Ensure relative error is inf if final_product is nan
    if np.isnan(final_product): relative_error = float('inf')

    return {
        "factors": (a, b),
        "product": final_product,
        "actual_product": actual_product_validation,
        "error_percent": relative_error,
        "confidence": confidence_calc if 'confidence_calc' in locals() else 0.0, # Use calculated confidence or 0 if failed early
        "debug_info": {
             "status": "Completed",
             "measured_a": features.get("measured_a", np.nan),
             "measured_b": features.get("measured_b", np.nan),
             "measured_sum_from_prod_wave": features.get("measured_sum", np.nan),
             "measured_sum_from_sum_wave": features.get("sum_consistency_check", np.nan),
             "measured_diff": features.get("measured_diff", np.nan),
             "conf1_sum_consistency": locals().get("conf1", np.nan), # Use locals() to get optional vars
             "conf2_input_match": locals().get("conf2", np.nan),
             "time_penalty": time_confidence_penalty,
             "N_points": num_points,
             "T_max_actual": actual_t_max,
             "T_max_req": req_t_max,
             "Samp_Rate_actual": actual_sampling_rate,
             "T_max_warning": t_max_warning,
             "fail_reason": debug_reason
        },
        # "feature_data": features # Avoid storing features dict unless needed for deep debug
    }

def test_algorithm(test_cases):
    """Test the complex wave multiplication algorithm"""
    print("Complex Wave Interference Multiplication (High Number Capable - Based on User Provided Code)")
    print(f"(Using MAX_POINTS_LIMIT = {MAX_POINTS_LIMIT})")
    print("========================================================================================")
    print(f"{'Factors':<25} | {'Est. Product':<20} | {'Actual Product':<20} | {'Error %':<12} | {'Confidence':<10} | {'Time (s)':<8}")
    print("-" * 110)

    total_error_percent = 0
    valid_cases = 0
    results = []

    for i, (a, b) in enumerate(test_cases):
        #print(f"Running case {i+1}/{len(test_cases)}: ({a}, {b})... ", end='', flush=True)
        start_time = time.time()
        result = complex_wave_multiplication(a, b)
        end_time = time.time()
        result['time_taken'] = end_time - start_time
        results.append(result)
        #print(f"done in {result['time_taken']:.2f}s")

        is_valid = not np.isnan(result["product"]) and result["error_percent"] != float('inf')
        if is_valid:
            total_error_percent += result["error_percent"]
            valid_cases += 1
        else:
             print(f"  -> Result invalid (Product: {result['product']}, Error: {result['error_percent']})")

        # Use .g format specifier for potentially large numbers
        print(f"({a}, {b})".ljust(25) +
              f"| {result['product']:<20.4g} | {result['actual_product']:<20.4g} | " +
              f"{result['error_percent'] if is_valid else 'N/A':<12}" + f" | {result['confidence']:.4f}" +
              f" | {result['time_taken']:.2f}".ljust(8))
        # Optional: print debug info for high error or low confidence cases
      

    avg_error = total_error_percent / valid_cases if valid_cases > 0 else float('nan')
    print("-" * 110)
    print(f"Average Error (on {valid_cases}/{len(test_cases)} valid/completed cases): {avg_error:.4f}%")
    print("-" * 110)
    return avg_error, results

def visualize_complex_waves(a, b):
    """Visualize the complex wave interference patterns and FFTs.
       MODIFIED: Regenerates waves for plotting."""
    print(f"\nAttempting visualization for factors ({a}, {b})...")
    # Run analysis again to get parameters, but don't rely on stored features
    result = complex_wave_multiplication(a, b)

    if result['debug_info']['status'] != "Completed" or np.isnan(result['product']):
        print(f"Skipping visualization for ({a}, {b}) as analysis did not complete successfully.")
        return

    # Extract parameters needed for plotting
    actual_t_max = result['debug_info']['T_max_actual']
    num_points = result['debug_info']['N_points']
    sampling_rate = result['debug_info']['Samp_Rate_actual']
    measured_a = result['debug_info'].get('measured_a', np.nan)
    measured_b = result['debug_info'].get('measured_b', np.nan)
    measured_sum = result['debug_info'].get('measured_sum_from_prod_wave', np.nan)

    # Regenerate data needed for plotting
    t_plot = np.linspace(0, actual_t_max, num_points, endpoint=False, dtype=np.float64)
    if len(t_plot) == 0:
        print("Cannot generate plot - time vector is empty.")
        return

    try:
        wave_a = generate_complex_wave(a, t_plot)
        wave_b = generate_complex_wave(b, t_plot)
        sum_wave = wave_a + wave_b
        product_wave = wave_a * wave_b
        # Calculate FFTs needed for plots
        fft_sum = np.fft.fft(sum_wave)
        fft_sum_mag = np.abs(fft_sum)
        fft_prod = np.fft.fft(product_wave)
        fft_prod_mag = np.abs(fft_prod)
        freqs = np.fft.fftfreq(num_points, 1.0/sampling_rate)

    except MemoryError:
        print("MemoryError during wave generation or FFT for plotting. Skipping visualization.")
        # Clean up potentially large arrays if error occurs mid-way
        del t_plot
        if 'wave_a' in locals(): del wave_a
        if 'wave_b' in locals(): del wave_b
        # ... etc for other large arrays
        return
    except Exception as e:
        print(f"Error generating data for plot: {e}. Skipping visualization.")
        return


    # --- Create Plot ---
    plt.figure(figsize=(12, 14)) # Tall figure

    plot_points = min(len(t_plot), 1000)
    t_subset = t_plot[:plot_points]
    time_limit_str = f"First {t_subset[-1]:.2e} seconds" if len(t_subset)>0 else "Time Domain"

    # Plot individual waves (Real part)
    plt.subplot(6, 1, 1)
    plt.plot(t_subset, np.real(wave_a[:plot_points]), label=f'Re[Wave A] (f={a:.2g} Hz)')
    plt.plot(t_subset, np.real(wave_b[:plot_points]), label=f'Re[Wave B] (f={b:.2g} Hz)')
    plt.grid(True); plt.legend(); plt.title(f'Input Complex Waves ({time_limit_str})')

    # Plot sum wave (Real part + Envelope)
    plt.subplot(6, 1, 2)
    plt.plot(t_subset, np.real(sum_wave[:plot_points]), label='Re[A + B]')
    plt.plot(t_subset, np.abs(sum_wave[:plot_points]), 'r--', label='Envelope |A + B|', alpha=0.7)
    plt.grid(True); plt.legend(); plt.title('Sum Wave (Interference)')

    # Plot product wave (Real part)
    plt.subplot(6, 1, 3)
    plt.plot(t_subset, np.real(product_wave[:plot_points]), label='Re[A * B]')
    plt.grid(True); plt.legend(); plt.title(f'Product Wave (f ~ {a+b:.2g} Hz)')

    # Plot FFT of Sum Wave
    plt.subplot(6, 1, 4)
    plot_freq_limit = 1.5 * max(a, b, 1)
    positive_freq_mask = (freqs > 1e-9) & (freqs <= plot_freq_limit) & (freqs <= sampling_rate/2)
    if np.any(positive_freq_mask):
         plt.plot(freqs[positive_freq_mask], fft_sum_mag[positive_freq_mask], label='FFT(Sum Wave)', linewidth=0.8)
         if not np.isnan(measured_a): plt.axvline(measured_a, color='r', linestyle='--', alpha=0.7, label=f'Peak A ~ {measured_a:.3g} Hz')
         if not np.isnan(measured_b): plt.axvline(measured_b, color='g', linestyle='--', alpha=0.7, label=f'Peak B ~ {measured_b:.3g} Hz')
         plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude'); plt.legend(); plt.grid(True)
         plt.title(f'FFT(Sum Wave) - Expect ~{a:.2g} & {b:.2g}')
    else:
         plt.text(0.5, 0.5, 'No suitable positive frequencies found/plotted', ha='center', va='center')

    # Plot FFT of Product Wave
    plt.subplot(6, 1, 5)
    plot_freq_limit_prod = 1.5 * (a + b) if (a + b) > 0 else 1.0
    positive_freq_mask_prod = (freqs > 1e-9) & (freqs <= plot_freq_limit_prod) & (freqs <= sampling_rate/2)
    if np.any(positive_freq_mask_prod):
        plt.plot(freqs[positive_freq_mask_prod], fft_prod_mag[positive_freq_mask_prod], label='FFT(Product Wave)', linewidth=0.8)
        if not np.isnan(measured_sum): plt.axvline(measured_sum, color='purple', linestyle='--', alpha=0.7, label=f'Peak Sum ~ {measured_sum:.3g} Hz')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('Magnitude'); plt.legend(); plt.grid(True)
        plt.title(f'FFT(Product Wave) - Expect ~{a+b:.2g}')
    else:
         plt.text(0.5, 0.5, 'No suitable positive frequencies found/plotted', ha='center', va='center')

    # Add summary text box
    plt.subplot(6, 1, 6)
    plt.axis('off')
    summary_text = (
        f"Factors: ({a:.3g}, {b:.3g})\n"
        f"Est. Product: {result['product']:.4g} (Actual: {result['actual_product']:.4g})\n"
        f"Error: {result['error_percent']:.4f}% | Confidence: {result['confidence']:.4f}\n"
        f"N Points: {num_points} | T Max: {actual_t_max:.2e}s | Samp Rate: {sampling_rate:.2e} Hz\n"
        f"Measured Freqs: A~{measured_a:.3g}, B~{measured_b:.3g}, Sum~{measured_sum:.3g}\n"
        f"Debug Notes: {result['debug_info'].get('T_max_warning','')[:100]}{'...' if len(result['debug_info'].get('T_max_warning',''))>100 else ''} {result['debug_info'].get('fail_reason','')}"
    )
    plt.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=9, family='monospace')

    plt.tight_layout(pad=1.5)
    filename = f"complex_wave_multiplication_{a:.1e}_{b:.1e}.png".replace('+','').replace('e0','e')
    try:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()
    # Clean up plot arrays
    del t_plot, wave_a, wave_b, sum_wave, product_wave, freqs, fft_sum_mag, fft_prod_mag


def generate_test_cases(num_cases=10, min_val=2, max_val=20, include_large=True):
    """Generate pairs of numbers, optionally including large ones."""
    test_cases = []
    # Add specific cases from original code
    test_cases.extend([(300, 500), (7, 7), (2, 10), (15, 4)]) # From user provided version

    if include_large:
        # Add cases with larger magnitudes
        test_cases.extend([
            (100, 200),
            (987, 123),
            (5000, 5001),
            (10000, 25000),
            (75000, 1000),
        ])

    # Add random cases within standard range
    num_random = max(0, num_cases - len(test_cases))
    for _ in range(num_random):
        num1 = random.randint(min_val, max_val)
        num2 = random.randint(min_val, max_val)
        if (num1, num2) not in test_cases:
             test_cases.append((num1, num2))

    # Add a case with closer numbers in standard range
    val = random.randint(min_val + 1, max_val)
    if (val, val - 1) not in test_cases:
        test_cases.append((val, val - 1))

    return test_cases

def main():
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define test cases including larger numbers
    test_cases = generate_test_cases(num_cases=12, min_val=2, max_val=50, include_large=True)

    print("Multiplication via Complex Wave Interference (High Number Capable - Based on User Provided Code)")
    print("=======================================================================================")
    print("Estimates products using the identity 4ab = (a+b)^2 - (a-b)^2")
    print(f"NOTE: MAX_POINTS_LIMIT = {MAX_POINTS_LIMIT}. Adjust based on RAM.")

    # Test the algorithm
    avg_error, results = test_algorithm(test_cases)

    # Find best and worst valid cases
    valid_results = [r for r in results if not np.isnan(r['product']) and r['error_percent'] != float('inf')]
    if valid_results:
        valid_results.sort(key=lambda r: r['error_percent'])
        best_case = valid_results[0]
        worst_case = valid_results[-1]
    else:
        best_case, worst_case = None, None

    print("\nGenerating visualizations for selected cases...")
    # Visualize a standard case like (7, 7) or (300, 500) if they completed
    std_case = next((r for r in results if r['factors'] == (300, 500) and r['debug_info']['status'] == 'Completed'), None)
    if not std_case:
        std_case = next((r for r in results if r['factors'] == (7, 7) and r['debug_info']['status'] == 'Completed'), None)

    if std_case: visualize_complex_waves(std_case['factors'][0], std_case['factors'][1])
    else: print("Skipping standard visualization as base cases might have failed.")


    if best_case:
         visualize_complex_waves(best_case['factors'][0], best_case['factors'][1])

    if worst_case and worst_case['error_percent'] > 0.5:
         visualize_complex_waves(worst_case['factors'][0], worst_case['factors'][1])
    else: # Visualize another large case if worst case was good
         large_case = next((r for r in valid_results if r['factors'][0] > 1000), None)
         if large_case and large_case != best_case:
             visualize_complex_waves(large_case['factors'][0], large_case['factors'][1])

    print("\n--- How the Algorithm Works ---")
    # (Description remains largely the same as the previous high-number version)
    print("1. Generates complex waves: exp(i*2*pi*a*t), exp(i*2*pi*b*t).")
    print("2. Determines required sampling rate and time duration based on a, b.")
    print("3. Adjusts time duration (t_max) if num_points exceeds MAX_POINTS_LIMIT to maintain sampling rate.")
    print("4. Creates Sum Wave (A+B) and Product Wave (A*B).")
    print("5. Analyzes FFT(Sum Wave): Finds peaks 'a', 'b'. Calculates measured_diff = |peak_a - peak_b|.")
    print("6. Analyzes FFT(Product Wave): Finds peak 'a+b'. Sets measured_sum = peak_(a+b).")
    print("7. Calculates product estimate: result = (measured_sum^2 - measured_diff^2) / 4.0.")
    print("8. Confidence based on FFT measurement consistency and sufficient time duration.")

    print(f"\nOverall Average Error: {avg_error:.4f}%")


if __name__ == "__main__":
    main()
