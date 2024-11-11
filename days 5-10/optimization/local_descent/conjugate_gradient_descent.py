import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from utils import bracket_minimum, bisection


sns.set_theme()


def line_search(f, x, d):
    objective = lambda alpha: f(x + alpha * d)
    a, b = bracket_minimum(objective)  # Find a bracketing interval
    d_objective = nd.Derivative(objective)
    alpha = bisection(d_objective, a, b)
    return alpha, x + alpha * d


def conjugate_gradient_descent(f, nabla_f, x, max_iter=200, eps=1e-5, visualize=False):
    """
    Perform conjugate gradient descent to minimize a function.
    
    Parameters:
    - f: Function to minimize.
    - nabla_f: Gradient of the function.
    - x: Initial position.
    - max_iter: Maximum number of iterations.
    - eps: Convergence tolerance.
    - visualize: If True, enable real-time visualization.
    
    Returns:
    - x: The optimized position.
    """
    g = nabla_f(x)    # Initial gradient
    d = -g            # Initial direction (negative gradient)
    
    # Set up plot if visualization is enabled
    if visualize:
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        # Collect points for visualization
        path_x, path_y = [x[0]], [x[1]]
        
    i = 0

    # Conjugate gradient descent loop
    while np.linalg.norm(d) > eps and i < max_iter:
        print(f"Iteration {i}: x = {x}")
        
        gp = nabla_f(x)                          # Previous gradient
        beta = max(0, gp.dot(gp - g) / g.dot(g)) # Compute beta using Polak-Ribiere formula
        dp = -gp + beta * d                      # Update direction
        
        _, x = line_search(f, x, dp)             # Perform line search to update x
        
        d, g = dp, gp                            # Update direction and gradient

        # Update plot in real-time if visualization is enabled
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.3)  # Pause for visualization effect
        
        i += 1

    # Final plot adjustments
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Conjugate Gradient Descent Optimization Path")
        plt.show()
        
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guess
x0 = np.array([5, 5])
x_optimal = conjugate_gradient_descent(f, nabla_f, x0, visualize=True)
print("Optimal x:", x_optimal)
