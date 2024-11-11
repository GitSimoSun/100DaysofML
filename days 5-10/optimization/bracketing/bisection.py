import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd 

sns.set_theme()  # Set a consistent visual theme for the plots

def bisection(f, df, a, b, eps=1e-5, max_iter=100, visualize=False):
    """
    Approximate a root of the derivative of f using the bisection method.

    Parameters:
    - f: Function for which we want to find the root of the derivative.
    - df: Derivative of the function f.
    - a, b: Initial interval [a, b] for bisection.
    - eps: Tolerance for stopping criteria.
    - max_iter: Maximum number of iterations.
    - visualize: If True, shows the process in an interactive plot.

    Returns:
    - Approximate root of df in the interval [a, b].
    """
    
    # Ensure a is the lower bound and b is the upper bound
    if a > b:
        a, b = b, a

    # Evaluate derivative at the bounds
    ya, yb = df(a), df(b)
    
    # Check if a or b is already a root of df
    if ya == 0:
        return a
    if yb == 0:
        return b

    if visualize:
        # Generate values for plotting the function f and its derivative df over the interval [a, b]
        x_vals = np.linspace(a, b, 500)
        y_vals_f = f(x_vals)
        y_vals_df = df(x_vals)

        # Set up side-by-side subplots for f(x) and f'(x)
        fig, (ax_f, ax_df) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot f(x) in the left subplot
        ax_f.plot(x_vals, y_vals_f, label="f(x)", color="blue")
        ax_f.axhline(0, color='gray', linestyle='--') # Add a horizontal line at y=0
        ax_f.set_title("Function f(x)")
        ax_f.set_xlabel("x")
        ax_f.set_ylabel("f(x)")

        # Plot f'(x) in the right subplot
        ax_df.plot(x_vals, y_vals_df, label="f'(x)", color="orange")
        ax_df.axhline(0, color='gray', linestyle='--') # Add a horizontal line at y=0
        ax_df.set_title("Derivative f'(x)")
        ax_df.set_xlabel("x")
        ax_df.set_ylabel("f'(x)")

        # Initialize markers for the bounds (a, b) and midpoint on each plot
        point_a_f, = ax_f.plot(a, f(a), 'ro', label="a (lower bound)")  # Red dot for a
        point_b_f, = ax_f.plot(b, f(b), 'go', label="b (upper bound)")  # Green dot for b
        midpoint_f, = ax_f.plot([], [], 'bo', label="Midpoint (x)")     # Blue dot for midpoint

        point_a_df, = ax_df.plot(a, ya, 'ro', label="a (lower bound)")  # Red dot for a
        point_b_df, = ax_df.plot(b, yb, 'go', label="b (upper bound)")  # Green dot for b
        midpoint_df, = ax_df.plot([], [], 'bo', label="Midpoint (x)")   # Blue dot for midpoint

        plt.legend()
        plt.ion() # Enable interactive mode for animation
        plt.show()


    #iterations counter
    i = 0
    # Bisection loop: continue until the interval [a, b] is smaller than tolerance or the maximum number of iterations max_iter is reached
    while ((b - a) > eps) and (i < max_iter):
        print(f"iter-{i}")
        x = (a + b) / 2  # Calculate the midpoint
        y = df(x)        # Evaluate the derivative at midpoint

        # Determine the new interval [a, b] based on the sign of y
        if np.sign(y) == np.sign(ya):
            a, ya = x, y  # Update a if signs match
        else:
            b, yb = x, y  # Update b if signs differ

        if visualize:
            # Update markers for the new bounds and midpoint on both subplots
            point_a_f.set_xdata([a])
            point_a_f.set_ydata([f(a)])
            point_b_f.set_xdata([b])
            point_b_f.set_ydata([f(b)])
            midpoint_f.set_xdata([x])
            midpoint_f.set_ydata([f(x)])

            point_a_df.set_xdata([a])
            point_a_df.set_ydata([ya])
            point_b_df.set_xdata([b])
            point_b_df.set_ydata([yb])
            midpoint_df.set_xdata([x])
            midpoint_df.set_ydata([y])

            plt.pause(0.3) # Pause briefly to update the plot in real-time
        
        i += 1

    print("The optimization process has ended.")
    if visualize:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Finalize the display

    # Return the midpoint as the approximate root of f'(x)
    return (a + b) / 2


if __name__ == '__main__':
    # Define the function and compute its derivative
    f = lambda x: (x - 2)**2 - 1
    df = nd.Derivative(f)

    # Set interval [a, b]
    a, b = 0, 5

    # Run the bisection method with visualization on
    root = bisection(f, df, a, b, visualize=True)
    print("Approximate root of f'(x):", root)
    print("The minimum of f is:", f(root))
