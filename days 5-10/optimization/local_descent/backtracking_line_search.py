import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def backtracking_line_search(f, df, x, d, alpha, p=0.5, beta=1e-4, visualize=False):
    """
    Performs the backtracking line search algorithm to find an appropriate step size alpha.

    Parameters:
    - f: Objective function.
    - df: Gradient of the objective function.
    - x: Current point in the search space.
    - d: Descent direction.
    - alpha: Initial step size.
    - p: Reduction factor for alpha in each iteration.
    - beta: Tolerance parameter for the Armijo condition.
    - visualize: If True, displays the progression of alpha values.

    Returns:
    - Tuple (alpha, x_new): Optimal alpha and the new point x.
    """
    y, g = f(x), df(x)
    alpha_values = [alpha]

    # Visualization setup if enabled
    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))

    # Backtracking loop
    while f(x + alpha * d) > y + beta * alpha * (g.dot(d)):
        alpha *= p
        alpha_values.append(alpha)

        if visualize:
            ax.clear()
            ax.plot(alpha_values, marker='o', color="blue", label=r"$\alpha$ progression")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"$\alpha$")
            ax.set_title(r"Backtracking Line Search: Progression of $\alpha$")
            ax.legend()
            plt.draw()
            plt.pause(0.3)

    if visualize:
        plt.ioff()
        plt.show()
    
    return alpha, x + alpha * d


# Define the objective function and its gradient
f = lambda x: x[0]**2 + x[0] * x[1] + x[1]**2
nabla_f = nd.Gradient(f)
x0 = np.array([1, 2])
d = np.array([-1, -1])
alpha0 = 10

# Perform backtracking line search with optional visualization
optimal_alpha, x_optimal = backtracking_line_search(f, nabla_f, x0, d, alpha0, visualize=True)  # Set to False to skip visualization

print("Optimal α:", optimal_alpha)
print("Optimal x:", x_optimal)
