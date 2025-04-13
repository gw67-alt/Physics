import numpy as np
import math

def quantize_peak_time(n, frequency):
    """
    Quantize peak time calculation to ensure it's a multiple of frequency and peaks.
    
    Args:
    n (int): Number of peaks
    frequency (float): Frequency of the sine wave
    
    Returns:
    dict: Detailed information about the quantized peak time calculation
    """
    if n < 1:
        print("Error: Peak number must be 1 or greater.")
        return None
    if frequency <= 0:
        print("Error: Frequency must be positive.")
        return None
    
    # Calculate the base period
    period = 1 / frequency
    
    # Quantization steps
    # 1. Ensure total time is divisible by both frequency and number of peaks
    total_peaks_time = n * period
    
    # Find the least common multiple (LCM) of frequency and number of peaks
    def lcm(a, b):
        return abs(a * b) // math.gcd(int(a), int(b))
    
    # Quantize the total time to the LCM
    lcm_value = lcm(frequency, n)
    quantized_total_time = math.ceil(total_peaks_time / (lcm_value * period)) * (lcm_value * period)
    
    # Recalculate number of peaks based on quantized time
    quantized_peaks = int(quantized_total_time / period)
    
    return {
        "original_peaks": n,
        "original_frequency": frequency,
        "original_total_time": total_peaks_time,
        "quantized_total_time": quantized_total_time,
        "quantized_peaks": quantized_peaks,
        "quantized_frequency": quantized_peaks / quantized_total_time,
        "period": period,
        "lcm_value": lcm_value
    }

def get_nth_peak_time(n, frequency, phase=0):
    """Calculates the time of the nth peak of a sine wave based on the first peak."""
    if n < 1:
        print("Error: Peak number must be 1 or greater.")
        return None  # Peaks are usually numbered starting from 1
    if frequency <= 0:
        print("Error: Frequency must be positive.")
        return None
    
    period_i = 1 / frequency
    # Calculate the time of the first peak (n=1)
    t_peak_1 = get_nth_peak_time_first_peak(frequency, phase)
    if t_peak_1 is None: # Handle potential issues in first peak calculation if needed
        return None
    # Calculate the nth peak time based on the first peak and period
    return t_peak_1 + (n - 1) * period_i

def get_nth_peak_time_first_peak(frequency, phase=0):
    """Calculates the time of the first peak (n=1) of a sine wave."""
    if frequency <= 0:
        # Already checked in calling function, but good practice here too
        return None
    # The argument of sin() for the first peak is pi/2
    # 2 * pi * f * t + phase = pi/2
    # t = (pi/2 - phase) / (2 * pi * f)
    return (np.pi / 2 - phase) / (2 * np.pi * frequency)

# --- Main Program ---
if __name__ == "__main__":
    try:
        # Get integer input from the user for the desired peak number
        nth_peak_str = input("Enter the desired peak number (e.g., 5): ")
        nth_peak = int(nth_peak_str)

        # Get integer input from the user for the frequency
        frequency_str = input("Enter the frequency in Hz (e.g., 2): ")
        frequency = int(frequency_str) # Using int() as requested

        # Initial phase (defaulting to 0)
        phase = 0.0

        # Quantize the peak calculation
        quantized_result = quantize_peak_time(nth_peak, frequency)
        
        if quantized_result is not None:
            print("-" * 50)
            print("Quantization Results:")
            print(f"Original Peaks: {quantized_result['original_peaks']}")
            print(f"Original Frequency: {quantized_result['original_frequency']} Hz")
            print(f"Original Total Time: {quantized_result['original_total_time']:.4f} s")
            print("-" * 50)
            print("Quantized Results:")
            print(f"Quantized Peaks: {quantized_result['quantized_peaks']}")
            print(f"Quantized Frequency: {quantized_result['quantized_frequency']:.4f} Hz")
            print(f"Quantized Total Time: {quantized_result['quantized_total_time']:.4f} s")
            print(f"LCM Value: {quantized_result['lcm_value']}")
            print("-" * 50)

            # Perform the calculation for the final peak time using quantized values
            peak_time = get_nth_peak_time(quantized_result['quantized_peaks'], 
                                          quantized_result['quantized_frequency'], 
                                          phase)

            if peak_time is not None:
                period_i = 1 / quantized_result['quantized_frequency']
                t_peak_1 = get_nth_peak_time_first_peak(quantized_result['quantized_frequency'], phase)
                num_intervals = quantized_result['quantized_peaks'] - 1
                duration_of_intervals = num_intervals * period_i

                print("Peak Time Calculation:")
                print(f"  First Peak Time (t1): {t_peak_1:.4f} s")
                print("Intervals Counted (Peak Times):")
                if t_peak_1 is not None:
                    for peak_num in range(1, quantized_result['quantized_peaks'] + 1):
                        current_peak_time = t_peak_1 + (peak_num - 1) * period_i
                        print(f"  Peak {peak_num}: {current_peak_time:.4f} s")
                print("-" * 50)
                print("Final Calculation:")
                print(f"  Final Calculated time: {peak_time:.4f} s")
                print("-" * 50)

    except ValueError:
        print("Invalid input. Please enter whole numbers for peak number and frequency.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
