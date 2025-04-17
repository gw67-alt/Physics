import numpy as np
import matplotlib.pyplot as plt
import sys
import time # Import the time module

# Define the number of points for intersection finding - higher values increase accuracy but slow down computation
NUM_INTERSECTION_POINTS = 10000 # Keep reasonably high for accuracy

def multiply_by_wave_amplitude(a, b, num_points=NUM_INTERSECTION_POINTS): # Pass num_points as argument
    """
    Calculates a * b using the average of the product of two sine waves
    with amplitudes a and b and the same frequency. Includes performance metrics.
    """
    print(f"\nCalculating {a} * {b} using wave amplitude multiplication...")
    print(f"  Using num_points = {num_points}")

    # --- Start Timing ---
    start_time = time.perf_counter() # Use performance counter for higher resolution

    # 1. Define wave parameters
    omega = 2 * np.pi  # Angular frequency (e.g., 2*pi for frequency = 1 Hz)
    frequency = omega / (2 * np.pi)
    period = 1 / frequency
    # Simulate for exactly 2 full cycles for robust averaging
    t_max = period * 2
    # Generate time vector - use endpoint=False for cleaner average over whole cycles
    t = np.linspace(0, t_max, num_points, endpoint=False)

    # 2. Create the two waves with amplitudes a and b
    y1 = a * np.sin(omega * t)
    y2 = b * np.sin(omega * t)

    # 3. Create the product wave
    y_mult = y1 * y2
    # Theoretical product wave: (a*b/2) - (a*b/2)*np.cos(2*omega*t)

    # 4. Calculate the average value of the product wave
    # Average over the whole simulation time (multiple of period)
    avg_val = np.mean(y_mult)

    # 5. Calculate the product a * b
    product_ab = 2 * avg_val

    # --- End Timing ---
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000 # Duration in milliseconds

    # --- Results and Performance Metrics ---
    print("\n--- Results ---")
    print(f"  Wave 1: y1(t) = {a} * sin({omega:.2f}*t)")
    print(f"  Wave 2: y2(t) = {b} * sin({omega:.2f}*t)")
    print(f"  Product Wave: y_mult(t) = y1(t) * y2(t)")
    print(f"  Average value of y_mult(t) = {avg_val:.6f}") # Increased precision
    print(f"  Calculated Product (2 * Average) = {product_ab:.6f}") # Increased precision
    print(f"  Direct Calculation: {a} * {b} = {a*b}")

    print("\n--- Performance Metrics ---")
    print(f"  Calculation Time: {duration_ms:.4f} ms")
    print(f"  Number of Points (num_points): {num_points}")

    # Accuracy / Error Calculation
    direct_product = a*b
    absolute_error = abs(direct_product - product_ab)
    print(f"  Absolute Error: {absolute_error:.6e}") # Use scientific notation for small errors

    # Calculate relative error, handle division by zero
    if abs(direct_product) > 1e-12: # Avoid division by zero or near-zero
         relative_error_percent = (absolute_error / abs(direct_product)) * 100
         print(f"  Relative Error: {relative_error_percent:.6f} %")
    elif absolute_error < 1e-12: # If both are zero
         print("  Relative Error: 0.00 % (Exact)")
    else: # If direct is zero but calculated is not
         print("  Relative Error: N/A (Direct product is zero)")
    print("-" * 27)
    # --- End Performance Metrics ---


    # 6. Plotting
    print("\nGenerating plots...")
    plt.figure(figsize=(12, 9))

    # Plot y1 and y2
    plt.subplot(3, 1, 1) # 3 rows, 1 column, 1st subplot
    plt.plot(t, y1, label=f'$y_1(t) = {a} \sin({omega:.1f}t)$')
    plt.plot(t, y2, label=f'$y_2(t) = {b} \sin({omega:.1f}t)$', linestyle='--')
    plt.title('Input Waves')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tick_params(axis='x', labelbottom=False) # Hide x-axis labels for this plot

    # Plot y_mult
    plt.subplot(3, 1, 2) # 3 rows, 1 column, 2nd subplot
    plt.plot(t, y_mult, label='$y_{mult}(t) = y_1(t) \\times y_2(t)$', color='purple')
    # Plot the average value line
    plt.axhline(avg_val, color='red', linestyle=':', lw=2,
                label=f'Avg Value (Calculated) = {avg_val:.4f}')
    # Plot the theoretical DC component line
    plt.axhline(a*b/2, color='green', linestyle='-.', lw=1,
                label=f'Theoretical DC Offset = {a*b/2:.4f}')

    plt.title('Product Wave and its Average Value')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.tick_params(axis='x', labelbottom=False) # Hide x-axis labels for this plot

    # Plot y_mult and its components for verification (optional)
    plt.subplot(3, 1, 3)
    dc_comp = np.full_like(t, a*b/2)
    ac_comp = -(a*b/2) * np.cos(2*omega*t)
    plt.plot(t, y_mult, label='$y_{mult}(t)$', color='purple', lw=2, alpha=0.6)
    plt.plot(t, dc_comp, label=f'DC Component = {a*b/2:.2f}', color='green', linestyle='--')
    plt.plot(t, ac_comp, label=f'AC Component = $({-a*b/2:.2f}) \cos({2*omega:.1f}t)$', color='orange', linestyle=':')
    plt.title('Product Wave Decomposed (Theoretical)')
    plt.xlabel('Time (t)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.tight_layout() # Adjust spacing between subplots
    plt.show()

    return product_ab

# --- Main Execution ---
if __name__ == "__main__":
    # Allow setting num_points via command line argument or use default
    points = NUM_INTERSECTION_POINTS # Default
    if len(sys.argv) > 1:
        try:
            points = int(sys.argv[1])
            print(f"[ Using num_points = {points} from command line ]")
        except ValueError:
            print(f"[ Warning: Invalid command line argument '{sys.argv[1]}'. Using default num_points = {points}. ]")

    while True:
        try:
            a_str = input("Enter the first number (a): ")
            if a_str.lower() == 'exit': sys.exit()
            A = float(a_str)

            b_str = input("Enter the second number (b): ")
            if b_str.lower() == 'exit': sys.exit()
            B = float(b_str)

            # Pass the number of points to the function
            result = multiply_by_wave_amplitude(A, B, num_points=points)

        except ValueError:
            print("Invalid input. Please enter numerical values for a and b (or type 'exit').")
        except KeyboardInterrupt:
            sys.exit("\nExiting.")

        try_again = input("\nTry another multiplication? (yes/no): ").lower()
        if try_again != 'yes' and try_again != 'y':
            break

    print("\nDone.")
