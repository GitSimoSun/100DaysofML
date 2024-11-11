import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def newtons_method(f, nabla_f, Hf, x, eps=1e-5, max_iter=100, visualize=False):
    """Performs Newton's Method optimization with optional visualization.

    Args:
        f (callable): Function to minimize.
        nabla_f (callable): Gradient of the function.
        Hf (callable): Hessian of the function.
        x (np.ndarray): Initial guess.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        visualize (bool): If True, displays the optimization path.
    
    Returns:
        np.ndarray: Optimal x found.
    """
    if visualize:
        plt.figure(figsize=(10, 6))
        X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        Z = f([X, Y])
        plt.contour(X, Y, Z, levels=50, cmap="viridis")
        path_x, path_y = [x[0]], [x[1]]
        
        
    k = 0
    delta = np.full_like(x, np.inf)
    
    while np.linalg.norm(delta) > eps and k < max_iter:
        print(f"Iteration {k}: x = {x}")
        delta = np.linalg.inv(Hf(x)).dot(nabla_f(x))
        x = x - delta
        
        
        if visualize:
            path_x.append(x[0])
            path_y.append(x[1])
            plt.plot(path_x, path_y, 'r.-', markersize=10)
            plt.pause(0.5)  # Pause for real-time effect
            
        k += 1
    
    if visualize:
        plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.legend()
        plt.title("Newton's Method Optimization Path")
        plt.show()

    return x

# Define function and gradient
f = lambda x: 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2
nabla_f = nd.Gradient(f)
Hf = nd.Hessian(f)

# Initial guess
x0 = np.array([5, 5])
x_optimal = newtons_method(f, nabla_f, Hf, x0, visualize=True)
print("Optimal x:", x_optimal)
