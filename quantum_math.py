import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys # To exit gracefully if needed

# Define the number of points for intersection finding
NUM_INTERSECTION_POINTS = 10000  # You can adjust this value if needed

# Define the equation function outside count_intersections 
# so fsolve can access it without scope issues if needed, though nested is fine here.
def equation(t_val, constant_freq, variable_freq):
    return np.sin(constant_freq * t_val) - np.sin(variable_freq * t_val)

def count_intersections(constant_freq, variable_freq, t_max, num_points=NUM_INTERSECTION_POINTS):
    """
    Count the number of intersections between two sine waves:
    - constant sine wave: sin(constant_freq * t)
    - variable sine wave: sin(variable_freq * t)

    Returns the count and the intersection points (times).
    """
    t = np.linspace(0, t_max, num_points)
    wave1 = np.sin(constant_freq * t)
    wave2 = np.sin(variable_freq * t)

    # Find sign changes in the difference, which indicate potential intersections
    diff = wave1 - wave2
    sign_changes_indices = np.where(np.diff(np.signbit(diff)))[0]

    # Refine the intersection points using fsolve
    intersections_t = []
    # Store unique roots to avoid duplicates if fsolve converges to the same point from nearby guesses
    # Use a tolerance for uniqueness check
    tolerance = 1e-8 # Increased precision for fsolve comparison
    unique_roots_rounded = set()

    # Check t=0 explicitly as sign change might miss it
    if abs(equation(0.0, constant_freq, variable_freq)) < tolerance:
         intersections_t.append(0.0)
         unique_roots_rounded.add(round(0.0, 7)) # Use rounding for set uniqueness check

    for idx in sign_changes_indices:
        # Use the midpoint of the interval where sign change occurred as initial guess
        t_approx = (t[idx] + t[idx+1]) / 2

        try:
            # Pass additional arguments (constants) to fsolve using args=()
            t_precise = fsolve(equation, t_approx, args=(constant_freq, variable_freq), xtol=tolerance)[0]

            # Check if the root is within the desired range [0, t_max]
            # and if the function value is close to zero
            if 0 <= t_precise <= t_max and abs(equation(t_precise, constant_freq, variable_freq)) < 1e-5:
                 # Check for uniqueness using rounded values to handle floating point inaccuracies
                 t_rounded = round(t_precise, 7)
                 if t_rounded not in unique_roots_rounded:
                    intersections_t.append(t_precise)
                    unique_roots_rounded.add(t_rounded)

        except Exception as e:
            # fsolve might fail in some cases, e.g., if gradient is zero
            # print(f"fsolve warning near t={t_approx} for freq={variable_freq}: {e}")
            pass # Silently ignore fsolve failures for this example

    # Check t=t_max explicitly
    if abs(equation(t_max, constant_freq, variable_freq)) < tolerance:
        t_rounded = round(t_max, 7)
        if t_rounded not in unique_roots_rounded:
             intersections_t.append(t_max)
             unique_roots_rounded.add(t_rounded)

    # Sort the final list of unique intersections
    intersections_t.sort()

    # Filter out any points slightly outside the bounds due to numerical issues
    intersections_t = [p for p in intersections_t if 0 <= p <= t_max]

    return len(intersections_t), np.array(intersections_t) # Return as numpy array


# Visualize the waves
def plot_waves(constant_freq, variable_freq, t_max, intersections): # Pass intersections as argument
    """ Plots the two waves and their intersection points """
    t = np.linspace(0, t_max, 1000) # Use enough points for smooth plot
    wave1 = np.sin(constant_freq * t)
    wave2 = np.sin(variable_freq * t)

    plt.figure(figsize=(10, 6))
    plt.plot(t, wave1, label=f'sin({constant_freq}t)')
    plt.plot(t, wave2, label=f'sin({variable_freq}t)')

    if intersections is not None and len(intersections) > 0:
        plt.scatter(intersections, np.sin(constant_freq * intersections),
                    color='red', zorder=5, label=f'Intersections ({len(intersections)})')
    else:
         plt.text(0.5, 0.5, 'No intersections found or calculated', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


    plt.grid(True)
    plt.legend()
    plt.title(f'Intersections of sin({constant_freq}t) and sin({variable_freq}t)')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.ylim(-1.1, 1.1)
    plt.show()

# --- Main Execution ---
# Define constants
CONSTANT_FREQ = 1.0
MAX_VARIABLE_FREQ = 100 # Adjust as needed
T_MAX = 2 * np.pi

print(f"Comparing sin({CONSTANT_FREQ}t) against sin(f*t) for f = 2 to {MAX_VARIABLE_FREQ}")
print(f"Interval: t = [0, {T_MAX:.4f}]")
print(f"Number of points for intersection detection: {NUM_INTERSECTION_POINTS}")

# --- Section for Multiplication using Amplitude (from previous request) ---
# This part remains separate as it addresses a different concept

def multiply_using_amplitude_average(a, b, omega=2*np.pi, num_points=1000):
    """ Calculates a*b using the average of wave products. """
    print(f"\n--- Amplitude Multiplication Example ---")
    print(f"Calculating {a} * {b}")
    t = np.linspace(0, 2*np.pi/omega, num_points, endpoint=False)
    wave1 = a * np.sin(omega * t)
    wave2 = b * np.sin(omega * t)
    product_wave = wave1 * wave2
    average_value = np.mean(product_wave)
    calculated_product = 2 * average_value
    print(f"  Wave 1 Amplitude: {a}")
    print(f"  Wave 2 Amplitude: {b}")
    print(f"  Average of Product Wave: {average_value:.4f}")
    print(f"  Calculated Product (2 * Average): {calculated_product:.4f}")
    print(f"  Actual Product (a * b): {a * b}")
    print("-" * 40)

# Example usage for amplitude multiplication
print("\nTesting amplitude multiplication method:")
multiply_using_amplitude_average(10000000, 110679718)
