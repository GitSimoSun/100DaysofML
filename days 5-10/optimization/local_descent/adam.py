import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def adam(f, nabla_f, x, eps=1e-8, alpha=0.001, gamma_s=0.999, gamma_v=0.9, max_iter=1000, visualize=False):
    """
    Performs Adam optimization on the given function.
    
    Parameters:
    - f: function to optimize.
    - nabla_f: gradient of the function.
    - x: initial guess for optimization.
    - eps: epsilon to prevent division by zero.
    - alpha: learning rate.
    - gamma_s: decay rate for second moment estimate.
    - gamma_v: decay rate for first moment estimate.
    - max_iter: Maximum number of iterations.
    - visualize: if True, displays the optimization path in real-time.
    
    Returns:
    - x: optimized variable values.
    """
    
    # Initialize parameters
    g = nabla_f(x)
    s = np.zeros_like(x)
    v = np.zeros_like(x)
    
    if visualize:
        # Set up plot for visualization
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        path_x, path_y = [x[0]], [x[1]]  # Collect points for visualization
    
    i = 0
    xp = np.copy(x)
    
    while np.linalg.norm(g) > 1e-5:
        print(f"Iteration {i}: x = {x}")
        
        # Update moving averages
        v = gamma_v * v + (1 - gamma_v) * g
        s = gamma_s * s + (1 - gamma_s) * (g * g)
        
        # Bias correction
        v_hat = v / (1 - gamma_v**(i + 1))
        s_hat = s / (1 - gamma_s**(i + 1))
        
        # Update x
        xp = np.copy(x)
        x = x - alpha * v_hat / (np.sqrt(s_hat) + eps)
        
        # Update gradient
        g = nabla_f(x)
        
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.2)  # Pause for real-time effect
        
        i += 1
        if i > max_iter or np.linalg.norm(x - xp) < 1e-7:  # Additional convergence check
            break
    
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Real-time Gradient Descent Optimization Path")
        plt.show()
    
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guess
x0 = np.array([5, 5])
x_optimal = adam(f, nabla_f, x0, alpha=0.1, gamma_s=0.9, gamma_v=0.9, visualize=True)
print("Optimal x:", x_optimal)
