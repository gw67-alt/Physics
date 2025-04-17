import numpy as np
import matplotlib.pyplot as plt
import sys

def multiply_by_wave_amplitude(a, b):
    """
    Calculates a * b using the average of the product of two sine waves
    with amplitudes a and b and the same frequency.
    """
    print(f"\nCalculating {a} * {b} using wave amplitude multiplication...")

    # 1. Define wave parameters
    omega = 2 * np.pi  # Angular frequency (e.g., 2*pi for frequency = 1 Hz)
    frequency = omega / (2 * np.pi)
    period = 1 / frequency
    t_max = period * 2 # Simulate for 2 full cycles for better visualization
    num_points = 2000  # Number of points for simulation

    # Generate time vector - use endpoint=False for cleaner average over whole cycles
    t = np.linspace(0, t_max, num_points, endpoint=False)

    # 2. Create the two waves with amplitudes a and b
    y1 = a * np.sin(omega * t)
    y2 = b * np.sin(omega * t)

    # 3. Create the product wave
    y_mult = y1 * y2
    # Theoretical product wave: (a*b/2) - (a*b/2)*np.cos(2*omega*t)

    # 4. Calculate the average value of the product wave
    # Use points covering exactly one period for the most accurate mean based on theory
    points_per_period = num_points // int(t_max / period)
    # avg_val = np.mean(y_mult[:points_per_period]) # Average over first cycle
    # Or average over the whole simulation time (should be robust if t_max is multiple of period)
    avg_val = np.mean(y_mult)

    # 5. Calculate the product a * b
    product_ab = 2 * avg_val

    print(f"  Wave 1: y1(t) = {a} * sin({omega:.2f}*t)")
    print(f"  Wave 2: y2(t) = {b} * sin({omega:.2f}*t)")
    print(f"  Product Wave: y_mult(t) = y1(t) * y2(t)")
    print(f"  Average value of y_mult(t) = {avg_val:.4f}")
    print(f"  Calculated Product (2 * Average) = {product_ab:.4f}")
    print(f"  Direct Calculation: {a} * {b} = {a*b}")
    print(f"  Difference = {abs((a*b) - product_ab):.6f}")


    # 6. Plotting
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
                label=f'Average Value = {avg_val:.4f} (Theory: {a*b/2:.4f})')
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
    while True:
        try:
            a_str = input("Enter the first number (a): ")
            if a_str.lower() == 'exit': sys.exit()
            A = float(a_str)

            b_str = input("Enter the second number (b): ")
            if b_str.lower() == 'exit': sys.exit()
            B = float(b_str)

            result = multiply_by_wave_amplitude(A, B)

        except ValueError:
            print("Invalid input. Please enter numerical values for a and b (or type 'exit').")
        except KeyboardInterrupt:
            sys.exit("\nExiting.")

        try_again = input("\nTry another multiplication? (yes/no): ").lower()
        if try_again != 'yes' and try_again != 'y':
            break

    print("\nDone.")
