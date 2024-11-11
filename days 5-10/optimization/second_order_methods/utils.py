import numpy as np
import numdifftools as nd

def bisection(df, a, b, eps=1e-5):
    if a > b:
        a, b = b, a

    ya, yb = df(a), df(b)
    if ya == 0:
        return a
    if yb == 0:
        return b
    
    while (b - a) > eps:
        x = (a + b) / 2
        y = df(x)

        if np.sign(y) == np.sign(ya):
            a, ya = x, y
        else:
            b, yb = x, y

    return (a + b) / 2

def bracket_minimum(f, x=0, s=1e-2, k=2, max_iter=100):
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)

    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya

    i = 0
    while i < max_iter:
        c, yc = b + s, f(b + s)

        if yc > yb:
            return (a, c) if a < c else (c, a)

        a, ya, b, yb = b, yb, c, yc
        s *= k
        i += 1

    return (a, b)

def line_search(f, x, d):
    objective = lambda alpha: f(x + alpha * d)
    a, b = bracket_minimum(objective)
    d_objective = nd.Derivative(objective)
    alpha = bisection(d_objective, a, b)
    return alpha, x + alpha * d