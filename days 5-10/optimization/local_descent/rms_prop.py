import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()


def rms_prop(f, nabla_f, x, alpha=0.1, eps=1e-5, gamma=0.9, max_iter=1000, visualize=False):
    """
    Performs RMSProp optimization on the given function.
    
    Parameters:
    - f: function to optimize.
    - nabla_f: gradient of the function.
    - x: initial guess for optimization.
    - alpha: learning rate.
    - eps: convergence threshold for gradient norm.
    - gamma: decay rate for the running average of the squared gradient.
    - max_iter: Maximum number of iterations.
    - visualize: if True, displays the optimization path in real-time.
    
    Returns:
    - x: optimized variable values.
    """
    
    # Initialize parameters
    g = nabla_f(x)
    s = np.zeros_like(x)
    
    if visualize:
        # Set up plot for visualization
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        path_x, path_y = [x[0]], [x[1]]  # Collect points for visualization
    
    i = 0
    while np.linalg.norm(g) > eps and i < max_iter:
        print(f"Iteration {i}: x = {x}")
        
        # Update the squared gradient estimate
        s = gamma * s + (1 - gamma) * g**2
        
        # Update x
        x = x - alpha * g / (np.sqrt(s) + 1e-8)
        
        # Update gradient
        g = nabla_f(x)
        
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.2)  # Pause for real-time effect
        
        i += 1
    
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Real-time RMSProp Optimization Path")
        plt.show()
    
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guess
x0 = np.array([5, 5])
x_optimal = rms_prop(f, nabla_f, x0, alpha=0.5, visualize=True)
print("Optimal x:", x_optimal)
