import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()


def secant_method(f, nabla_f, x0, x1, eps=1e-5, max_iter=100):
    # Set up plot
    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
    Z = f([X, Y])
    plt.contour(X, Y, Z, levels=50, cmap="viridis")
    
    # Collect points for visualization
    path_x, path_y = [x1[0]], [x1[1]]
    
    k = 0
    g0 = nabla_f(x0)
    delta = np.inf
    
    while np.linalg.norm(delta) > eps and k <= max_iter:
        print(f"Iteration {k}: x1 = {x1}")
        g1 = nabla_f(x1)
        
        # Secant direction update using dot product for stability
        delta = (np.dot((x1 - x0), g1) / np.dot(g1 - g0, g1)) * g1
        
        x0 = np.copy(x1)
        g0 = g1
        x1 = x1 - delta
        
        path_x.append(x1[0])
        path_y.append(x1[1])
        
        # Plot the current point
        plt.plot(path_x, path_y, 'r.-', markersize=10)
        plt.pause(0.2)  # Pause for real-time effect
        
        k += 1
        if np.linalg.norm(g1) < eps:  # Additional convergence check on the gradient
            break
    
    plt.plot(path_x, path_y, 'r.-', markersize=10, label="Optimization Path")
    plt.xlabel("x1[0]")
    plt.ylabel("x1[1]")
    plt.legend()
    plt.title("Real-time Secant Method Optimization Path")
    plt.show()
    return x1

# Define function and gradient
f = lambda x: 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2
# f = lambda x: x[0]**2 + x[1]**2
nabla_f = nd.Gradient(f)

# Initial guesses
x0 = np.array([5, 5])
x1 = np.array([-5, -5])
x_optimal = secant_method(f, nabla_f, x0, x1)
print("Optimal x:", x_optimal)
