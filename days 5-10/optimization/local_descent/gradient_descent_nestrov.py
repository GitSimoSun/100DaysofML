import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def gradient_descent_nesterov(f, nabla_f, x, alpha=0.5, beta=0.2, eps=1e-5, max_iter=200, visualize=False):
    """
    Perform gradient descent using Nesterov's accelerated gradient (NAG) method.
    
    Parameters:
    - f: Function to minimize.
    - nabla_f: Gradient of the function.
    - x: Initial position.
    - alpha: Step size.
    - beta: Momentum coefficient.
    - eps: Convergence tolerance.
    - max_iter: Maximum number of iterations.
    - visualize: If True, enables real-time visualization.
    
    Returns:
    - x: The optimized position.
    """
    v = np.zeros_like(x)  # Initialize momentum

    # Set up plot if visualization is enabled
    if visualize:
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        path_x, path_y = [x[0]], [x[1]]

    i = 0

    while np.linalg.norm(nabla_f(x)) > eps and i < max_iter:
        print(f"Iteration {i}: x = {x}")
        
        # Nesterov's accelerated gradient step
        v = beta * v - alpha * nabla_f(x + beta * v)  # Look ahead and calculate gradient
        x = x + v  # Update position
        
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
        plt.title("Nesterov Accelerated Gradient Descent Optimization Path")
        plt.show()
        
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2  # Objective function
nabla_f = nd.Gradient(f)  # Gradient of the function

# Initial guess
x0 = np.array([2, 2])
x_optimal = gradient_descent_nesterov(f, nabla_f, x0, alpha=0.1, beta=0.8, visualize=True)
print("Optimal x:", x_optimal)
