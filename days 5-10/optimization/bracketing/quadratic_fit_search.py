import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the theme for seaborn plots
sns.set_theme()

def quadratic_fit_search(f, a, b, c, n, tolerance=1e-6, visualize=False):
    """
    Perform a quadratic fit search to find the minimum of a function.

    Parameters:
    - f: Function to minimize.
    - a, b, c: Three initial points a, b, and c for the search.
    - n: Number of iterations to refine the minimum.
    - tolerance: Tolerance for stopping criteria based on minimal x change.
    - visualize: If True, displays the search process on a plot.

    Returns:
    - Tuple (a, b, c): Final points that bracket the minimum.
    """
    
    # Initial function evaluations at points a, b, c
    ya, yb, yc = f(a), f(b), f(c)
    
    # Optional visualization setup
    if visualize:
        fig, ax = plt.subplots()
        x_vals = np.linspace(a - 1, c + 1, 500)
        y_vals = f(x_vals)
        ax.plot(x_vals, y_vals, label="f(x)")
        
        plt.ion()
        point_a, = ax.plot(a, ya, 'ro', label="a")
        point_b, = ax.plot(b, yb, 'go', label="b")
        point_c, = ax.plot(c, yc, 'bo', label="c")
        point_x, = ax.plot([], [], 'mo', label="x")
        polynomial_line, = ax.plot([], [], 'y--', label="Quadratic Fit")  
        
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Quadratic Fit Search for Minimum")
        plt.show()

    previous_x = None  # Track the previous x for stopping criteria

    # Iteratively refine the points to find the minimum
    for i in range(1, n-2):
        print(f"iter-{i}")
        
        # Calculate the x position of the minimum of the quadratic polynomial
        x = 0.5 * (ya * (b**2 - c**2) + yb * (c**2 - a**2) + yc * (a**2 - b**2)) / (ya * (b - c) + yb * (c - a) + yc * (a - b))
        yx = f(x)
        
        # Stopping condition: Check if x is changing minimally
        if previous_x is not None and abs(x - previous_x) < tolerance:
            print("Stopping early due to minimal change in x.")
            break
        previous_x = x  

        # Update the interval based on function values
        if x > b:
            if yx > yb:
                c, yc = x, yx
            else:
                a, ya, b, yb = b, yb, x, yx
        elif x < b:
            if yx > yb:
                a, ya = x, yx
            else:
                c, yc, b, yb = b, yb, x, yx

        # Update visualization if enabled
        if visualize:
            point_a.set_xdata([a])
            point_a.set_ydata([ya])
            point_b.set_xdata([b])
            point_b.set_ydata([yb])
            point_c.set_xdata([c])
            point_c.set_ydata([yc])
            point_x.set_xdata([x])
            point_x.set_ydata([yx])

            # Quadratic fit visualization
            coeffs = np.polyfit([a, b, c], [ya, yb, yc], 2)
            poly = np.poly1d(coeffs)
            
            x_poly_vals = np.linspace(min(a, b, c), max(a, b, c), 100)
            y_poly_vals = poly(x_poly_vals)
            polynomial_line.set_xdata(x_poly_vals)
            polynomial_line.set_ydata(y_poly_vals)
            
            plt.pause(0.3)  # Brief pause to visualize update

    print("The optimization process has ended.")
    
    # Finalize visualization if enabled
    if visualize:
        plt.ioff()
        plt.show()
    
    return (a, b, c)


if __name__ == '__main__':
    # Define the function to minimize
    f_rastrigin = lambda x: 10 + x**2 - 10 * np.cos(2 * np.pi * x)
    f = f_rastrigin

    # Define initial interval
    a, b, c = -20, 30, 40

    print("Points:", quadratic_fit_search(f, a, b, c, 30, visualize=True))

