import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def adagrad(f, nabla_f, x, alpha=1e-1, eps=1e-8, tol=1e-5, max_iter=1000, visualize=False):
    """
    Perform optimization using the Adagrad method.
    
    Parameters:
    - f: The function to minimize.
    - nabla_f: The gradient of the function.
    - x: Initial position.
    - alpha: Learning rate.
    - eps: Small constant for numerical stability.
    - tol: Convergence threshold.
    - max_iter: Maximum number of iterations.
    - visualize: If True, enables real-time visualization.
    
    Returns:
    - x: The optimized position.
    """
    g = nabla_f(x)
    s = np.zeros_like(x)
    
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
        
        # Accumulate squared gradients
        s = s + g * g
        
        # Update position with Adagrad formula
        x = x - alpha * g / (np.sqrt(s) + eps)
        
        # Compute new gradient
        g = nabla_f(x)
        
        
        # Update plot in real-time if visualization is enabled
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.2)  # Pause for visualization effect
        
        i += 1

    # Final plot adjustments
    plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.legend()
    plt.title("Real-time Adagrad Optimization Path")
    plt.show()
    
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2  # Objective function
nabla_f = nd.Gradient(f)  # Gradient of the function

# Initial guess
x0 = np.array([5, 5])

# Call the adagrad function
x_optimal = adagrad(f, nabla_f, x0, alpha=1, visualize=True)
print("Optimal x:", x_optimal)
