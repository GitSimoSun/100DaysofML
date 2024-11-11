import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd
from utils import line_search

sns.set_theme()

def bfgs_method(f, nabla_f, x, eps=1e-5, max_iter=100, visualize=True):
    """
    Perform optimization using the BFGS method.

    Parameters:
    - f: Function to minimize.
    - nabla_f: Gradient of the function.
    - x: Initial position (numpy array).
    - eps: Convergence threshold (default=1e-5).
    - max_iter: Maximum number of iterations (default=100).
    - visualize: Boolean to enable or disable real-time visualization.

    Returns:
    - x: Optimized position.
    """
    # Optional visualization setup
    if visualize:
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")

    # Initialize variables
    k = 0
    m = len(x)
    Q = np.eye(m)           # Initial inverse Hessian approximation
    g = nabla_f(x)          # Gradient at initial position
    path_x, path_y = [x[0]], [x[1]]  # Lists to store path for visualization

    # Optimization loop
    while np.linalg.norm(g) > eps and k < max_iter:
        print(f"Iteration {k}: x = {x}")

        # Line search step along -Q @ g direction
        _, x_new = line_search(f, x, -Q @ g)
        
        # Update gradient and compute BFGS updates
        g_new = nabla_f(x_new)
        delta = x_new - x
        gamma = g_new - g

        # BFGS Update
        rho = 1.0 / (gamma.T @ delta)
        Q = Q - rho * (Q @ np.outer(gamma, gamma) @ Q) + rho * np.outer(delta, delta) * (1 + (gamma.T @ Q @ gamma) * rho)

        # Update position and gradient
        x = np.copy(x_new)
        g = np.copy(g_new)
        
        # Store path for visualization
        path_x.append(x[0])
        path_y.append(x[1])

        # Update visualization if enabled
        if visualize:
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.5)  # Pause to display progress

        k += 1

    # Final visualization adjustments if enabled
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("BFGS Optimization Path")
        plt.show()

    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guess
x0 = np.array([5, 5])

# Call the BFGS method with visualization disabled or enabled
x_optimal = bfgs_method(f, nabla_f, x0, visualize=True)
print("Optimal x:", x_optimal)
