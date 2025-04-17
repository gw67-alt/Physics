import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# Define the number of points for simulation
NUM_SIMULATION_POINTS = 10000 # Using a good number of points for accurate derivative estimate
PRACTICAL_FLOAT_LIMIT = 8
def multiply_by_derivative_at_origin(a, b, num_points=NUM_SIMULATION_POINTS):
    """
    Calculates a * b using the derivatives of two sine waves
    at their t=0 intersection point.

    Args:
        a (float): The first number (amplitude of the first wave).
        b (float): The second number (amplitude of the second wave).
        num_points (int): Number of points for the simulation.

    Returns:
        tuple: (calculated_product, direct_product, calculation_time_ms)
               Returns (None, None, 0) if inputs are invalid or calculation fails.
    """
    print(f"\nCalculating {a} * {b} using derivative at intersection (t=0) method...")
    print(f"  Using num_points = {num_points}")

    # --- Input Validation ---
    if not (np.isfinite(a) and np.isfinite(b)):
        print("[Error: Inputs must be finite numbers.]")
        return None, None, 0
    # We still need limits as intermediate calculations might overflow
    limit = np.finfo(float).max / 10
    if abs(a) > limit or abs(b) > limit or abs(a * b) > limit :
         print(f"[Warning: Input numbers or their product might be too large (>{limit:.1e}) for reliable float calculations. Result might be 'inf' or inaccurate.]")
    # --- End Input Validation ---

    # --- Start Timing ---
    start_time = time.perf_counter()

    # 1. Define wave parameters
    omega = 2 * np.pi  # Angular frequency (using 2pi makes period = 1)
    period = 1.0
    # We only need a small interval around t=0 to estimate the derivative accurately
    # but we'll generate a bit more for plotting context.
    t_max = period / 4 # Simulate for quarter period is enough for derivative at 0
    # Ensure at least 3 points for potential higher-order derivative estimates if needed
    if num_points < 3:
        num_points = 3
    t = np.linspace(0, t_max, num_points, endpoint=True) # Include t=0
    dt = t[1] - t[0] # Time step

    # 2. Create the two waves with amplitudes a and b
    try:
        wave1 = a * np.sin(omega * t)
        wave2 = b * np.sin(omega * t)
    except (OverflowError, ValueError) as e:
        print(f"[Error during wave generation: {e}. Inputs might be too large.]")
        return None, None, time.perf_counter() - start_time

    # 3. Calculate derivatives at t=0 using forward difference
    # Ensure we have at least 2 points
    if num_points < 2:
        print("[Error: Need at least 2 points to calculate derivative.]")
        return None, None, time.perf_counter() - start_time

    # Check for inf/nan before calculation
    if not (np.isfinite(wave1[0]) and np.isfinite(wave1[1]) and np.isfinite(wave2[0]) and np.isfinite(wave2[1])):
         print("[Error: Non-finite values detected near t=0 in input waves. Cannot calculate derivative.]")
         return None, None, time.perf_counter() - start_time

    # Note: wave1[0] and wave2[0] should be 0
    deriv1_at_0 = (wave1[1] - wave1[0]) / dt
    deriv2_at_0 = (wave2[1] - wave2[0]) / dt

    # Theoretical derivatives: a*omega and b*omega
    theoretical_deriv1 = a * omega
    theoretical_deriv2 = b * omega

    # 4. Calculate the product using the derivatives
    # a*b = (deriv1 * deriv2) / (omega*omega)
    try:
        calculated_product = (deriv1_at_0 * deriv2_at_0) / (omega**2)
    except (OverflowError, ValueError, ZeroDivisionError) as e:
        print(f"[Error during product calculation from derivatives: {e}]")
        return None, None, time.perf_counter() - start_time

    # --- End Timing ---
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    # --- Results and Performance Metrics ---
    direct_product = a * b # Calculate direct product for comparison

    print("\n--- Results ---")
    print(f"  Wave 1 Amplitude (a): {a}")
    print(f"  Wave 2 Amplitude (b): {b}")
    print(f"  Calculated derivative y1'(0): {deriv1_at_0:.6f} (Theoretical: {theoretical_deriv1:.6f})")
    print(f"  Calculated derivative y2'(0): {deriv2_at_0:.6f} (Theoretical: {theoretical_deriv2:.6f})")
    print(f"  Calculated Product (deriv1*deriv2 / omega^2): {calculated_product:.6f}")
    print(f"  Direct Calculation (a * b):                 {direct_product:.6f}")

    print("\n--- Performance Metrics ---")
    print(f"  Calculation Time: {duration_ms:.4f} ms")
    print(f"  Number of Points (num_points): {num_points}")
    print(f"  Frequency (omega / 2pi): {omega / (2 * np.pi):.2f} Hz")
    print(f"  Time step (dt): {dt:.2e} s")

    # Accuracy / Error Calculation
    if np.isfinite(calculated_product) and np.isfinite(direct_product):
        absolute_error = abs(direct_product - calculated_product)
        print(f"  Absolute Error: {absolute_error:.6e}") # Use scientific notation

        if abs(direct_product) > 1e-12: # Avoid division by zero
             relative_error_percent = (absolute_error / abs(direct_product)) * 100
             print(f"  Relative Error: {relative_error_percent:.6f} %")
        elif abs(absolute_error) < 1e-9: # Adjusted tolerance
             print("  Relative Error: 0.00 % (Essentially zero)")
        else:
             print(f"  Relative Error: N/A (Direct product is near zero, Abs Error: {absolute_error:.2e})")
    else:
        print("  Absolute Error: N/A (Result was inf or NaN)")
        print("  Relative Error: N/A")
    print("-" * 40)
    # --- End Performance Metrics ---

    return calculated_product, direct_product, duration_ms


# --- Main Execution ---
if __name__ == "__main__":
    points = NUM_SIMULATION_POINTS
    if len(sys.argv) > 1:
        try:
            points_arg = int(sys.argv[1])
            if points_arg > 1: # Need at least 2 points for derivative
                points = points_arg
                print(f"[ Using num_points = {points} from command line ]")
            else:
                print(f"Number of points must be > 1. Using default {points}.")
        except ValueError:
            print(f"[ Warning: Invalid command line argument '{sys.argv[1]}'. Using default num_points = {points}. ]")

    print("\nThis script demonstrates multiplication (a * b) using the derivatives")
    print(f"of a*sin(ωt) and b*sin(ωt) at the intersection t=0.")
    print(f"It calculates a*b = [y1'(0) * y2'(0)] / ω^2.")
    print(f"Standard float limits apply (approx {PRACTICAL_FLOAT_LIMIT:.0e}).")

    while True:
        try:
            a_str = input(f"Enter the first number (a): ")
            if a_str.lower() == 'exit': sys.exit()
            A = float(a_str) # Validate standard float conversion first

            b_str = input(f"Enter the second number (b): ")
            if b_str.lower() == 'exit': sys.exit()
            B = float(b_str) # Validate standard float conversion first

            # --- Input Validation (check for Inf/NaN from input conversion) ---
            if not (np.isfinite(A) and np.isfinite(B)):
                print("\n[Error: Inputs must be finite numbers (not Inf or NaN). Please try again.]")
                continue
            # ----------------------

            # Call the function
            result_tuple = multiply_by_derivative_at_origin(A, B, num_points=points)

            if result_tuple[0] is None: # Check if function indicated an error during calculation
                print("[Calculation failed, please check input values or potential overflow warnings.]")
                continue

            calc_prod, direct_prod, _ = result_tuple
            # Check if calculation itself resulted in inf/nan
            if not np.isfinite(calc_prod):
                 print("\n[ Note: The calculation resulted in non-finite values (inf/nan). ]")
                 print("[ This can happen if input numbers are too large for standard floats. ]")

        except ValueError:
            print("Invalid input. Please enter numerical values (e.g., 123.45 or 1.23e100) or type 'exit'.")
        except KeyboardInterrupt:
             sys.exit("\nExiting.")
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging

        try_again = input("\nTry another multiplication? (yes/no): ").lower()
        if try_again not in ['yes', 'y']:
            break

    print("\nDone.")
