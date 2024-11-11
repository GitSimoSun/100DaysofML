import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from utils import bracket_minimum, bisection

sns.set_theme()

def line_search(f, x, d):
    objective = lambda alpha: f(x + alpha * d)
    
    # Use bracketing method to identify initial bounds
    a, b = bracket_minimum(objective)
    
    # Use bisection on derivative to find optimal alpha
    d_objective = nd.Derivative(objective)
    alpha = bisection(d_objective, a, b)
    
    return alpha, x + alpha * d

def gradient_descent(f, nabla_f, x, eps=1e-5, max_iter=100, visualize=False):
    """
    Performs gradient descent optimization with optional real-time visualization.

    Parameters:
    - f: Objective function.
    - nabla_f: Gradient of the objective function.
    - x: Initial starting point.
    - eps: Convergence tolerance for the gradient norm.
    - max_iter: Maximum number of iterations.
    - visualize: If True, shows the optimization path in real-time.

    Returns:
    - x: Optimal point found by gradient descent.
    """
    d = -nabla_f(x)
    path_x, path_y = [x[0]], [x[1]]  # To store the optimization path for plotting

    if visualize:
        # Set up plot for visualization
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")

    i = 0
    while np.linalg.norm(d) > eps and i < max_iter:
        print(f"Iteration {i}: x = {x}")
        
        # Perform line search to find the optimal step size
        alpha, _ = line_search(f, x, d)
        x = x + alpha * d  # Update the current position
        d = -nabla_f(x)  # Update the gradient
        
        print("Gradient norm:", np.linalg.norm(d))
        path_x.append(x[0])
        path_y.append(x[1])
        
        # Plot the current point if visualization is enabled
        if visualize:
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.3)  # Pause for real-time effect

        i += 1
    
    # Final plot adjustments if visualization is enabled
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Gradient Descent Optimization Path")
        plt.show()
    
    return x

# Define the function and its gradient
f = lambda x: x[0]**2 + 2*x[1]**2 - .3 * np.cos(3 * np.pi * x[0]) - 0.4 * np.cos(4 * np.pi * x[1]) + 0.7
nabla_f = nd.Gradient(f)

# Initial guess for the starting point
x0 = np.array([5, 5])

# Perform gradient descent with visualization enabled
x_optimal = gradient_descent(f, nabla_f, x0, visualize=True)
print("Optimal x:", x_optimal)
