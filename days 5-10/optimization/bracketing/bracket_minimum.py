import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()  # Set a consistent visual theme for the plots


def bracket_minimum(f, x=0, s=1e-2, k=2, max_iter=100, visualize=False):
    """
    Find an interval that brackets the minimum of a function f using a 
    simple bracketing approach.

    Parameters:
    - f: Function to minimize.
    - x: Initial starting point.
    - s: Initial step size.
    - k: Growth factor for the step size.
    - max_iter: Maximum number of iterations.
    - visualize: Whether to display visualization of the bracketing process.

    Returns:
    - (a, b): Tuple representing the interval [a, b] that brackets the minimum.
    """
    # Initialize the first two points and evaluate the function
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)

    # If the function is increasing, swap a and b to ensure we're moving toward a minimum
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya

    # Set up visualization if requested
    if visualize:
        # Create a plot with the function curve for context
        fig, ax = plt.subplots()
        x_vals = np.linspace(x - 2, x + 6, 500)
        y_vals = f(x_vals)
        ax.plot(x_vals, y_vals, label="f(x)")
        
        # Enable interactive plotting mode
        plt.ion()
        
        # Plot the initial points a and b
        point_a, = ax.plot(a, ya, 'ro', label="a")  # Red dot for point a
        point_b, = ax.plot(b, yb, 'go', label="b")  # Green dot for point b
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Bracketing the Minimum")
        plt.show()

    # Iteratively expand the interval to bracket the minimum
    i = 0
    while i < max_iter:
        # Generate a new point c to test further along the direction of b
        c, yc = b + s, f(b + s)

        # Check if we've found a bracket (yc > yb indicates the function is now increasing)
        if yc > yb:
            if visualize:
                plt.ioff()  # Turn off interactive mode
                plt.show()
            # Return the interval [a, c] that brackets the minimum
            return (a, c) if a < c else (c, a)

        # Move the interval forward: a takes b's position, b takes c's
        a, ya, b, yb = b, yb, c, yc
        s *= k  # Increase step size by factor of k for faster exploration
        i += 1

        # Update the plot if visualization is enabled
        if visualize:
            point_a.set_xdata([a])  # Update position of point a
            point_a.set_ydata([ya])
            point_b.set_xdata([b])  # Update position of point b
            point_b.set_ydata([yb])
            plt.pause(0.3)  # Pause briefly to animate

    # Finalize the plot if visualization is enabled
    if visualize:
        plt.ioff()
        plt.show()
    
    # Return the final interval [a, b] after reaching max iterations
    return (a, b)

if __name__ == '__main__':
    # Define the function to minimize
    f = lambda x: (x - 2)**2

    # Run the bracketing function without plotting
    print("Bracket without visualization:", bracket_minimum(f))

    # Run the bracketing function with plotting
    print("Bracket with visualization:", bracket_minimum(f, visualize=True))
