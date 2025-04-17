import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def count_intersections(constant_freq, variable_freq, t_max, num_points=10000):
    """
    Count the number of intersections between two sine waves:
    - constant sine wave: sin(constant_freq * t)
    - variable sine wave: sin(variable_freq * t)
    
    Returns the count and the intersection points
    """
    t = np.linspace(0, t_max, num_points)
    wave1 = np.sin(constant_freq * t)
    wave2 = np.sin(variable_freq * t)
    
    # Find sign changes in the difference, which indicate intersections
    diff = wave1 - wave2
    sign_changes = np.where(np.diff(np.signbit(diff)))[0]
    
    # Refine the intersection points using fsolve
    intersections = []
    for idx in sign_changes:
        t_approx = t[idx]
        # Define equation to solve: sin(constant_freq * t) - sin(variable_freq * t) = 0
        def equation(t_val):
            return np.sin(constant_freq * t_val) - np.sin(variable_freq * t_val)
        
        t_precise = fsolve(equation, t_approx)[0]
        intersections.append(t_precise)
    
    return len(intersections), intersections

def find_prime_by_intersections(constant_freq=1.0, max_variable_freq=100, t_max=2*np.pi):
    """
    Find a prime number by counting intersections between a constant sine wave
    and variable sine waves with different frequencies.
    """
    for var_freq in range(2, max_variable_freq + 1):
        count, _ = count_intersections(constant_freq, var_freq, t_max)
        
        # Check if the count is prime
        if is_prime(count):
            return count, var_freq

def is_prime(n):
    """Check if a number is prime"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Run the algorithm
prime, freq = find_prime_by_intersections()
print(f"Found prime number {prime} with variable frequency {freq}")

# Visualize the waves
def plot_waves(constant_freq, variable_freq, t_max):
    t = np.linspace(0, t_max, 1000)
    wave1 = np.sin(constant_freq * t)
    wave2 = np.sin(variable_freq * t)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, wave1, label=f'sin({constant_freq}t)')
    plt.plot(t, wave2, label=f'sin({variable_freq}t)')
    plt.scatter(intersections, np.sin(constant_freq * np.array(intersections)), 
                color='red', zorder=3, label='Intersections')
    plt.grid(True)
    plt.legend()
    plt.title(f'Intersections of sine waves: count = {len(intersections)}')
    plt.xlabel('t')
    plt.ylabel('Amplitude')
    plt.show()

# Example visualization
constant_freq = 1.0
variable_freq = freq  # Use the frequency that gave us the prime
count, intersections = count_intersections(constant_freq, variable_freq, 2*np.pi)
plot_waves(constant_freq, variable_freq, 2*np.pi)
