import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()
plt.ion()  # Turn on interactive mode

def secant_method(df, x0, x1, eps=1e-5, max_iter=100):
    g0 = df(x0)
    delta = np.inf
    
    # Set up plot
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label="Function f(x)", color="blue")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    
    # Plot the starting points
    plt.plot(x0, f(x0), 'ro', label="Initial x0")
    plt.plot(x1, f(x1), 'go', label="Initial x1")
    plt.legend()
    
    i = 0
    while (np.abs(delta) > eps) and i < max_iter:
        g1 = df(x1)
        # Secant update step
        delta = ((x1 - x0) / (g1 - g0)) * g1
        x0 = x1
        x1 = x1 - delta

        # Update plot with current points
        plt.plot(x0, f(x0), 'ro', markersize=6, label="x0" if i == 0 else "")
        plt.plot(x1, f(x1), 'go', markersize=6, label="x1" if i == 0 else "")
        plt.legend()
        
        plt.pause(0.5)  # Pause for real-time effect
        
        g0 = g1
        i += 1

    plt.plot(x1, f(x1), 'bs', markersize=5, label="Converged Point")
    plt.legend()
    plt.title("Real-time Secant Method Convergence")
    plt.show(block=True)  # Block at the end so it stays open
    return x1

# Define function and its derivative
f = lambda x: (x - 2)**2 + 5
df = nd.Derivative(f)

# Initial guesses
x0, x1 = 0, 5
x_optimal = secant_method(df, x0, x1)
print("Optimal x:", x_optimal)
