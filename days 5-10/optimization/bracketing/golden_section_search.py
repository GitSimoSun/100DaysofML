import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()  # Set a consistent visual theme for the plots

def golden_section_search(f, a, b, n, visualize=False):
    """
    Perform the Golden Section search to find the minimum of a function.

    Parameters:
    - f: Function to minimize.
    - a, b: Interval [a, b] to search for the minimum.
    - n: Number of iterations to refine the interval.
    - visualize: If True, displays the search process on a plot.

    Returns:
    - Tuple (a, b): Final bracket interval around the minimum.
    """
    
    # Define constants for the golden section search
    phi = (1 + np.sqrt(5)) / 2
    rho = phi - 1  # Golden ratio (phi - 1)
    d = rho * b + (1 - rho) * a  # Initial interior point
    yd = f(d)  # Evaluate function at point d

    # Optional visualization setup
    if visualize:
        fig, ax = plt.subplots()
        x_vals = np.linspace(a - 1, b + 1, 500)
        y_vals = f(x_vals)
        ax.plot(x_vals, y_vals, label="f(x)")
        
        plt.ion()
        point_a, = ax.plot(a, f(a), 'ro', label="a")
        point_b, = ax.plot(b, f(b), 'go', label="b")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Golden Section Search for Minimum")
        plt.show()

    # Iteratively refine the interval
    for i in range(1, n):
        # Calculate a new point c based on the golden section ratio
        c = rho * a + (1 - rho) * b     
        yc = f(c)  # Evaluate function at point c
        
        # Update the interval based on function values
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c

        # Update visualization if enabled
        if visualize:
            point_a.set_xdata([a])
            point_a.set_ydata([f(a)])
            point_b.set_xdata([b])
            point_b.set_ydata([f(b)])
            plt.pause(0.3)  # Brief pause to visualize update

    print("The optimization process has ended.")
    # Finalize visualization if enabled
    if visualize:
        plt.ioff()
        plt.show()
    
    return (a, b) if (a < b) else (b, a)



if __name__ == '__main__':
    # Define the function to minimize
    f = lambda x: (x - 2)**2

    # Call the golden_section_search function with visualization disabled
    print("Bracket:", golden_section_search(f, a=0, b=4, n=10, visualize=True))
