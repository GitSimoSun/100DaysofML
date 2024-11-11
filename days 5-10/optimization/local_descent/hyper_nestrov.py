import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()


def hyper_nesterov(f, nabla_f, x, alpha=0.01, mu=1e-8, beta=0.8, eps=1e-5, max_iter=100, visualize=False):
    """
    Performs Hypergradient Nesterov optimization on the given function.
    
    Parameters:
    - f: function to optimize.
    - nabla_f: gradient of the function.
    - x: initial guess for optimization.
    - alpha: initial learning rate.
    - mu: learning rate adjustment factor.
    - beta: momentum factor.
    - eps: convergence threshold for gradient norm.
    - max_iter: maximum number of iterations.
    - visualize: if True, displays the optimization path in real-time.
    
    Returns:
    - x: optimized variable values.
    """
    
    # Initialize parameters
    g = nabla_f(x)
    g_prev = np.zeros_like(g)
    v = np.zeros_like(x)
    
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
        
        # Update learning rate based on hypergradient
        alpha = alpha + mu * (g.dot(g_prev))
        
        # Update momentum and apply to x
        v = beta * v - alpha * g
        x = x + v
        
        # Store previous gradient and calculate new gradient
        g_prev = np.copy(g)
        g = nabla_f(x + beta * v)
        
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.5)  # Pause for real-time effect
        
        i += 1
    
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Real-time Hypergradient Nesterov Optimization Path")
        plt.show()
    
    return x

# Define function and gradient
f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guess
x0 = np.array([5, 5])
x_optimal = hyper_nesterov(f, nabla_f, x0, visualize=True)
print("Optimal x:", x_optimal)
