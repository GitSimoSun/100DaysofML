import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme() # Set a consistent visual theme for the plots

def fibonacci_search(f, a, b, n, eps=0.01, visualize=False):
    """
    Perform the Fibonacci search to find the minimum of a function.

    Parameters:
    - f: Function to minimize.
    - a, b: Interval [a, b] to search for the minimum.
    - n: Number of iterations (Fibonacci numbers).
    - eps: Tolerance for the interval contraction.
    - visualize: If True, plots the search progress on ax.

    Returns:
    - Tuple (a, b): Bracket interval around the minimum.
    """
    
    # Calculate necessary constants for the Fibonacci search
    s = (1 - np.sqrt(5)) / (1 + np.sqrt(5))  # Constant for scaling
    phi = (1 + np.sqrt(5)) / 2               # Golden ratio
    rho = 1 / (phi * (1 - s**(n+1)) / (1 - s**n))  # Scale factor for interval reduction

    # Initialize point d within the interval [a, b] based on rho
    d = rho * b + (1 - rho) * a
    yd = f(d)  # Evaluate function at d

    # Set up visualization if enabled
    if visualize:
        # Create figure and plot function curve
        fig, ax = plt.subplots()
        x_vals = np.linspace(-2, 6, 500)
        y_vals = f(x_vals)
        ax.plot(x_vals, y_vals, label="f(x)")
        
        # Initialize plot for interactive updating of points a and b
        plt.ion()
        point_a, = ax.plot(a, f(a), 'ro', label="a")
        point_b, = ax.plot(b, f(b), 'go', label="b")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Fibonacci Search for Minimum")
        plt.show()

    # Iteratively reduce the interval [a, b] over n steps
    for i in range(1, n):
        # Calculate point c as a new candidate for minimum, adjusting for the last step
        if i == n-1:
            c = eps * a + (1 - eps) * d  # Tighter interval near the end
        else:
            c = rho * a + (1 - rho) * b  # General interval division based on rho

        yc = f(c)  # Evaluate function at c

        # Update interval [a, b] and selected points based on function evaluations
        if yc < yd:
            b, d, yd = d, c, yc  # Update b to d if c yields a lower function value
        else:
            a, b = b, c  # Move a to b if d is lower than c

        # Update rho for the next iteration
        rho = 1 / (phi * (1 - s**(n-i+1)) / (1 - s**(n-i)))

        # Update visualization if enabled
        if visualize:
            point_a.set_xdata([a])  # Update point a on plot
            point_a.set_ydata([f(a)])
            point_b.set_xdata([b])  # Update point b on plot
            point_b.set_ydata([f(b)])
            plt.pause(0.3)  # Pause to show updates interactively
  
    print("The optimization process has ended.")
    # Finalize the plot if visualization is enabled
    if visualize:
        plt.ioff()
        plt.show()
        
    # Return the final interval containing the minimum
    return (a, b) if (a < b) else (b, a)



if __name__ == '__main__':
    # Define the function to minimize
    f = lambda x: (x - 2)**2

    # Call fibonacci_search with visualization on to observe the interval contraction
    print("Bracket:", fibonacci_search(f, a=5, b=0, n=10, visualize=True))
