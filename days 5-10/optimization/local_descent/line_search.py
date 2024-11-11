import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from utils import bracket_minimum

sns.set_theme()

def bisection(df, a, b, eps=1e-5, visualize=False):
    """
    Performs the bisection method to find the root of the derivative function.

    Parameters:
    - df: Derivative function to find the root of.
    - a, b: Interval [a, b] to search.
    - eps: Precision threshold for stopping.
    - visualize: If True, enables visualization of the alpha progression.

    Returns:
    - Midpoint of the interval [a, b] as the approximate root.
    """
    if a > b:
        a, b = b, a

    ya, yb = df(a), df(b)
    if ya == 0:
        return a
    if yb == 0:
        return b

    alpha_values = []

    # Visualization setup if enabled
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))

    # Iterative bisection
    while (b - a) > eps:
        x = (a + b) / 2
        y = df(x)
        alpha_values.append(x)

        if visualize:
            ax.clear()
            ax.plot(alpha_values, marker='o', color='blue', label=r'$\alpha$ progression')
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"$\alpha$")
            ax.set_title(r"Progression of $\alpha$ over Iterations")
            ax.legend()
            plt.draw()
            plt.pause(0.3)

        if np.sign(y) == np.sign(ya):
            a, ya = x, y
        else:
            b, yb = x, y

    if visualize:
        plt.ioff()
        plt.show()
    
    return (a + b) / 2


def line_search(f, x, d, visualize_bisection=False):
    """
    Performs a line search to find the optimal step size alpha.

    Parameters:
    - f: Function to minimize.
    - x: Initial point.
    - d: Search direction.
    - visualize_bisection: If True, enables visualization in the bisection method.

    Returns:
    - Tuple (alpha, x_optimal): Optimal alpha and the new optimal point x.
    """
    objective = lambda alpha: f(x + alpha * d)
    a, b = bracket_minimum(objective)
    d_objective = nd.Derivative(objective)
    alpha = bisection(d_objective, a, b, visualize=visualize_bisection)
    return alpha, x + alpha * d


# Define the function, initial point, and direction
f = lambda x: np.sin(x[0] * x[1]) + np.exp(x[1] + x[2]) - x[2]
x0 = np.array([1, 2, 3])
d = np.array([0, -1, -1])

# Perform line search with optional visualization in bisection
optimal_alpha, x_optimal = line_search(f, x0, d, visualize_bisection=True)  # Set to False to skip visualization

print("Optimal α:", optimal_alpha)
print("Optimal x:", x_optimal)
