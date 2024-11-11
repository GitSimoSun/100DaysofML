import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"
    

def get_sp_intersection(A, B, l):
    t = ((A.y - B.y) - l * (A.x - B.x)) / (2 * l)
    return Point(A.x + t, A.y - t * l)


def shubert_piyavskii(f, a, b, l, eps=0.01, small_delta=0.01, tolerance=1e-6, visualize=False):
    """
    Performs the Shubert-Piyavskii method for finding the minimum of a function.

    Parameters:
    - f: Function to minimize.
    - a, b: Interval [a, b] to search.
    - l: Lipschitz constant for the function.
    - eps: Precision threshold for stopping.
    - small_delta: Minimum gap between interval points.
    - tolerance: Tolerance for changes in x to stop the search early.
    - visualize: If True, enables visualization of the search process.

    Returns:
    - Tuple (p_min, intervals): Minimum point and intervals containing possible minima.
    """
    
    # Optional visualization setup
    if visualize:
        x_vals = np.linspace(a, b, 500)
        y_vals = f(x_vals)
        
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="f(x)")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Shubert-Piyavskii Method for Minimum Search")
    
    # Initial search points
    m = (a + b) / 2
    A, M, B = Point(a, f(a)), Point(m, f(m)), Point(b, f(b))
    pts = [A, get_sp_intersection(A, M, l), M, get_sp_intersection(M, B, l), B]
    
    if visualize:
        search_points, = ax.plot([p.x for p in pts], [p.y for p in pts], 'ro', label="Search Points")
        plt.legend()
        plt.ion()
        plt.show()
    
    big_delta = np.inf
    previous_min_x = None

    # Iteratively refine points
    while big_delta > eps:
        i = np.argmin([p.y for p in pts])
        p = Point(pts[i].x, f(pts[i].x))
        
        big_delta = p.y - pts[i].y
        
        # Check for minimal x change
        if previous_min_x is not None and abs(p.x - previous_min_x) < tolerance:
            print("Stopping early due to minimal change in x.")
            break
        previous_min_x = p.x

        # Add new search points around p
        p_prev = get_sp_intersection(pts[i-1], p, l)
        p_next = get_sp_intersection(p, pts[i+1], l)
        
        pts.pop(i)
        pts.insert(i, p_next)
        pts.insert(i, p)
        pts.insert(i, p_prev)

        # Update visualization if enabled
        if visualize:
            search_points.set_xdata([p.x for p in pts])
            search_points.set_ydata([p.y for p in pts])
            plt.pause(0.3)  # Pause for visual effect

    print("The optimization process has ended.")
    if visualize:
        plt.ioff()
        plt.show()
    
    # Find intervals around minimum point
    intervals = []
    p_min = pts[np.argmin([p.y for p in pts])]
    y_min = p_min.y
    
    for i in range(1, len(pts) - 1):
        if pts[i].y < y_min:
            dy = y_min - pts[i].y
            x_lo = max(a, pts[i].x - dy / l)
            x_hi = min(b, pts[i].x + dy / l)
            if intervals and intervals[-1][1] + small_delta >= x_lo:
                intervals[-1] = (intervals[-1][0], x_hi)
            else:
                intervals.append((x_lo, x_hi))
    
    return (p_min, intervals)



if __name__ == '__main__':
    # Define the function and parameters
    f = lambda x: 10 + x**2 - 10 * np.cos(2 * np.pi * x)
    a, b, l = -5, 5, 20

    # Run the method with optional visualization
    print("Result:", shubert_piyavskii(f, a, b, l, visualize=True))  # Set to False to skip visualization
