import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def adadelta(f, nabla_f, x, eps=1e-8, gamma_s=0.999, gamma_x=0.9, max_iter=1000, tol=1e-5, visualize=False):
    """
    Perform optimization using the Adadelta method.
    
    Parameters:
    - f: The function to minimize.
    - nabla_f: The gradient of the function.
    - x: Initial position.
    - eps: Small constant for numerical stability.
    - gamma_s: Decay rate for the accumulated squared gradients.
    - gamma_x: Decay rate for the accumulated squared updates.
    - max_iter: Maximum number of iterations.
    - tol: Convergence tolerance.
    - visualize: If True, enables real-time visualization.
    
    Returns:
    - x: The optimized position.
    """
    g = nabla_f(x)
    s = np.zeros_like(x)  # Accumulated squared gradients
    u = np.zeros_like(x)  # Accumulated squared updates

    # Set up plot if visualization is enabled
    if visualize:
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        path_x, path_y = [x[0]], [x[1]]
    
    i = 0
    while np.linalg.norm(g) > tol and i < max_iter:
        print(f"Iteration {i}: x = {x}")
        
        # Accumulate gradient
        s = gamma_s * s + (1 - gamma_s) * g**2
        
        # Compute update
        delta_x = - (np.sqrt(u) + eps) * g / (np.sqrt(s) + eps)
        
        # Accumulate updates
        u = gamma_x * u + (1 - gamma_x) * delta_x**2
        
        # Update position
        x = x + delta_x
        g = nabla_f(x)  # Update gradient for new position
        
        
        # Update plot in real-time if visualization is enabled
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.2)  # Pause for visualization effect
        
        i += 1

    # Final plot adjustments
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Adadelta Optimization Path")
        plt.show()
        
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2  # Objective function
nabla_f = nd.Gradient(f)  # Gradient of the function

# Initial guess
x0 = np.array([5, 5])
x_optimal = adadelta(f, nabla_f, x0, gamma_s=0.9, gamma_x=0.9, visualize=True)
print("Optimal x:", x_optimal)
