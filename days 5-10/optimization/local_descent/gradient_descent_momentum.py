import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def gradient_descent_momentum(f, nabla_f, x, alpha=0.5, beta=0.2, eps=1e-5, max_iter=200, visualize=False):
    """
    Perform gradient descent with momentum to find the minimum of a function.
    
    Parameters:
    - f: The function to minimize.
    - nabla_f: The gradient of the function.
    - x: Initial point for optimization.
    - alpha: Step size (learning rate).
    - beta: Momentum parameter (controls influence of previous gradient).
    - eps: Convergence tolerance.
    - max_iter: Maximum number of iterations.
    - visualize: If True, enables real-time visualization of the descent.
    
    Returns:
    - x: The optimized position.
    """
    g = nabla_f(x)        # Compute initial gradient
    v = np.zeros_like(x)   # Initialize momentum vector

    # Set up plot if visualization is enabled
    if visualize:
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        # Collect points for visualization
        path_x, path_y = [x[0]], [x[1]]    

    i = 0

    # Gradient descent with momentum loop
    while np.linalg.norm(g) > eps and i < max_iter:
        print(f"Iteration {i}: x = {x}")

        v = beta * v - alpha * g   # Update momentum vector
        x = x + v                  # Update position
        g = nabla_f(x)             # Recompute gradient

        # Update plot in real-time if visualization is enabled
        if visualize:
            # Store path for visualization
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
        plt.title("Gradient Descent with Momentum Optimization Path")
        plt.show()

    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guess
x0 = np.array([2, 2])
x_optimal = gradient_descent_momentum(f, nabla_f, x0, alpha=0.1, beta=0.8, visualize=True)

print("Optimal x:", x_optimal)
