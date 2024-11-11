import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numdifftools as nd

sns.set_theme()

def strong_backtracking(f, nabla_f, x, d, alpha=1, beta=1e-4, sigma=0.1, visualize=False):
    """
    Performs the strong backtracking line search algorithm to find an appropriate step size alpha.

    Parameters:
    - f: Objective function.
    - nabla_f: Gradient of the objective function.
    - x: Current point in the search space.
    - d: Descent direction.
    - alpha: Initial step size.
    - beta: Tolerance parameter for the Armijo condition.
    - sigma: Tolerance for the curvature condition.
    - visualize: If True, displays the progression of alpha values.

    Returns:
    - alpha: Optimal step size.
    """
    y0, g0 = f(x), nabla_f(x).dot(d)
    y_prev, alpha_prev = np.nan, 0
    alpha_values = [alpha]  # Track alpha values for visualization

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))

    # Strong backtracking initial phase
    while True:
        y = f(x + alpha * d)
        alpha_values.append(alpha)

        if visualize:
            ax.clear()
            ax.plot(alpha_values, marker='o', color='blue', label=r'$\alpha$ progression')
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"$\alpha$")
            ax.set_title(r"Strong Backtracking Line Search: Progression of $\alpha$")
            ax.legend()
            plt.draw()
            plt.pause(0.3)

        if (y > y0 + beta * alpha * g0) or (not np.isnan(y_prev) and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break

        g = nabla_f(x + alpha * d).dot(d)
        if np.abs(g) <= -sigma * g0:
            if visualize:
                plt.ioff()
                plt.show()
            return alpha

        elif g >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev

        y_prev, alpha_prev = y, alpha
        alpha *= 2

    # Zoomed phase to find optimal alpha
    y_lo = f(x + alpha_lo * d)
    while True:
        alpha = (alpha_lo + alpha_hi) / 2
        y = f(x + alpha * d)
        alpha_values.append(alpha)

        if visualize:
            ax.clear()
            ax.plot(alpha_values, marker='o', color='blue', label=r"$\alpha$ progression (zoomed in)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(r"$\alpha$")
            ax.set_title(r"Strong Backtracking Line Search: Zoomed Progression of $\alpha$")
            ax.legend()
            plt.draw()
            plt.pause(0.3)

        if (y > y0 + beta * alpha * g0) or (y >= y_lo):
            alpha_hi = alpha
        else:
            g = nabla_f(x + alpha * d).dot(d)
            if np.abs(g) <= -sigma * g0:
                if visualize:
                    plt.ioff()
                    plt.show()
                return alpha
            elif g * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

# Define the objective function and its gradient
f = lambda x: x[0]**2 + x[0] * x[1] + x[1]**2
nabla_f = nd.Gradient(f)

x0 = np.array([1, 2])
d = np.array([-1, -1])
alpha0 = 1

# Perform strong backtracking with optional visualization
optimal_alpha = strong_backtracking(f, nabla_f, x0, d, alpha=alpha0, visualize=True)  # Set to False to skip visualization

print("Optimal α:", optimal_alpha)
