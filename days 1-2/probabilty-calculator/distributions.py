import math
import numpy as np



def uniform(a, b, i):
    return 1 / (b - a)


def binomial(n, p, i):
    return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))


def geometric(p, i):
    return p * ((1 - p) ** (i - 1))


def negative_geometric(p, r, i):
    return math.comb(i - 1, r - 1) * (p ** r) * ((1 - p) ** (i - r))


def hypergeometric(N, m, n, i):
    return math.comb(m, i) * math.comb(N - m, n - i) / math.comb(N, n)


def poisson(lam, i):
    return math.exp(-lam) * (lam ** i) / math.factorial(i)


def normal_pdf(mu, sigma_squared, x):
    return np.exp(-(x - mu) ** 2 / 2 * sigma_squared ** 2) / np.sqrt(2 * np.pi * sigma_squared)


def normal_cdf(mu, sigma_squared, x, step=0.001):
    total = 0
    t = -4 * sigma_squared + mu
    while t < x:
        total += normal_pdf(mu, sigma_squared, t) * step
        t += step
    return total


def exponential_pdf(lam, x):
    return lam * np.exp(-lam * x)


def exponential_cdf(lam, x):
    return 1 - math.exp(-lam * x)


def gamma_pdf(alpha, lam, x):
    return lam * np.exp(-lam * x) * (lam * x) ** (alpha - 1) / math.factorial(alpha - 1)


def gamma_cdf(alpha, lam, x):
    e = 0
    for k in range(alpha):
        e += ((lam * x) ** k / math.factorial(k))
    return 1 - math.exp(-lam*x) * e


def beta_pdf(a, b, x):
    return (math.factorial(a + b - 1) * (x ** (a - 1)) * ((1 - x) ** (b - 1)) ) / (math.factorial(a - 1) * math.factorial(b - 1))


def beta_cdf(a, b, x, step=0.001):
    s = 0
    for t in range(0, x, step):
        s += t**(a - 1) * (1 - t)**(b - 1)
    return (math.factorial(a + b - 1) * s ) / (math.factorial(a - 1) * math.factorial(b - 1))


def cauchy_pdf(theta, x):
    return 1 / (math.pi * 1 + (x - theta) ** 2)


def cauchy_cdf(theta, x):
    return math.atan(x - theta) / math.pi


def weibull_pdf(beta, alpha, x):
    return beta * (x / alpha) ** (beta - 1) * np.exp(- (x / alpha) ** beta) / alpha
    

def weibull_cdf(beta, alpha, x):
    return 1 - math.exp(- (x / alpha) ** beta)
    

def pareto_pdf(lam, a, x):
    return lam * (a ** lam) * (x ** -(lam + 1))


def pareto_cdf(lam, a, x):
    return 1 - (a / x) ** lam


def uniform_cdf(a, b, i):
    if i < a:
        return 0
    if i > b:
        return 1
    return (i - a) / (b - a)
