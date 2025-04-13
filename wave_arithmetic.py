import numpy as np
import math

def quantize_peak_intervals(desired_peaks, periodic_hz):
    """
    Quantize peak intervals with maximum frequency between peaks and additional intervals.
    
    Args:
    desired_peaks (int): Desired number of peaks
    periodic_hz (float): Periodic frequency in Hz
    
    Returns:
    dict: Detailed information about the quantized peak intervals
    """
    if desired_peaks < 1 or periodic_hz <= 0:
        raise ValueError("Invalid peaks or periodic Hz")
    
    # Base calculations
    base_period = 1 / periodic_hz
    
    # Base number of intervals between peaks
    base_intervals = desired_peaks - 1
    
    def find_max_interval_quantization(base_period, base_intervals):
        """
        Find the maximum frequency quantization for intervals with additional precision.
        """
        # Try different multipliers to increase precision
        for n in range(1, 10):  # Limit to reasonable multipliers
            total_intervals = base_intervals + n
            
            # Start with a high maximum frequency
            max_interval_freq = 1000  # Arbitrarily high starting point
            
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
    
    # Final quantization parameters
    return {
        "original_peaks": desired_peaks,
        "original_periodic_hz": periodic_hz,
        "original_period": base_period,
        "max_interval_frequency": interval_quantization['max_interval_frequency'],
        "interval_period": interval_quantization['interval_period'],
        "base_intervals": interval_quantization['base_intervals'],
        "total_intervals": interval_quantization['total_intervals'],
        "additional_intervals": interval_quantization['additional_intervals']
    }

def calculate_peak_times(quantized_params, phase=0):
    """
    Calculate peak times based on quantized intervals.
    
    Args:
    quantized_params (dict): Quantized interval parameters
    phase (float): Initial phase (default 0)
    
    Returns:
    list: Peak times
    """
    peaks = quantized_params['original_peaks']
    interval_period = quantized_params['interval_period']
    
    # Calculate peak times
    peak_times = [0]  # First peak at 0
    for i in range(1, peaks):
        # Next peak time is the sum of previous peaks and intervals
        peak_times.append(peak_times[-1] + interval_period)
    
    return peak_times

# --- Main Program ---
if __name__ == "__main__":
    try:
        # Get input for desired peaks
        peaks_str = input("Enter desired number of peaks (e.g., 5): ")
        desired_peaks = int(peaks_str)

        # Get input for periodic Hz
        periodic_hz_str = input("Enter periodic Hz (e.g., 2): ")
        periodic_hz = float(periodic_hz_str)

        # Quantize peak intervals
        quantized_result = quantize_peak_intervals(desired_peaks, periodic_hz)
        
        # Calculate peak times
        peak_times = calculate_peak_times(quantized_result)
        
        # Print detailed results
        print("\n" + "=" * 50)
        print("Peak Interval Quantization Results")
        print("=" * 50)
        
        print("\nOriginal Parameters:")
        print(f"  Desired Peaks: {quantized_result['original_peaks']}")
        print(f"  Periodic Hz: {quantized_result['original_periodic_hz']} Hz")
        print(f"  Original Period: {quantized_result['original_period']:.4f} s")
        
        print("\nInterval Quantization:")
        print(f"  Result: {quantized_result['max_interval_frequency']:.4f} Hz")
        print(f"  Interval Period: {quantized_result['interval_period']:.4f} s")
        print(f"  Base Intervals: {quantized_result['base_intervals']}")
        print(f"  Additional Intervals: {quantized_result['additional_intervals']}")
        print(f"  Total Intervals: {quantized_result['total_intervals']}")
        
        print("\nPeak Times:")
        for i, time in enumerate(peak_times, 1):
            print(f"  Peak {i}: {time:.4f} s")
        
        # Verification
        print("\nVerification:")
        total_time = peak_times[-1]
        print(f"  Total Time: {total_time:.4f} s")
        print(f"  Matches Original Period: {math.isclose(total_time, quantized_result['original_period'], rel_tol=1e-10)}")
        
        print("\n" + "=" * 50)

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
