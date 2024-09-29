import eel
import math
import numpy as np
from distributions import *


eel.init('web')



@eel.expose
def compute_distribution(distribution, parameters):   
    if distribution == "Uniform Discrete":
        a = parameters["a"]
        b = parameters["b"]
        x = np.arange(a, b+1).tolist()
        y = [uniform(a, b, i) for i in x] 
        return {
            "xs": x,
            "ys": y,
            "variableType": "discrete"
        }
        
        
    if distribution == "Binomial":
        n = parameters["n"]
        p = parameters["p"]
        x = np.arange(n+1).tolist()
        y = [binomial(n, p, i) for i in x] 
        return {
            "xs": x,
            "ys": y,
            "variableType": "discrete"
        }
    
    
    if distribution == "Poisson":
        lam = parameters["lambda"]
        max_value = 3 * lam
        min_value = 0
        x = np.arange(min_value, max_value).tolist()
        y = [poisson(lam, i) for i in x]
        return {
            "xs": x,
            "ys": y,
            "variableType": "discrete"
        }
    
    
    if distribution == "Geometric":
        p = parameters["p"]
        max_value = math.ceil(1/p**2) + 3
        x = np.arange(max_value).tolist()
        y = [geometric(p, i) for i in x]
        return {
            "xs": x,
            "ys": y,
            "variableType": "discrete"
        }
    
    
    if distribution == "Negative Geometric":
        p = parameters["p"]
        r = parameters["r"]
        max_value = math.ceil(r/p**2) + 3
        x = np.arange(1, max_value).tolist()
        y = [negative_geometric(p, r, i) for i in x]
        return {
            "xs": x,
            "ys": y,
            "variableType": "discrete"
        }
 
    
    if distribution == "Hypergeometric":
        N = parameters["N"]
        m = parameters["m"]
        n = parameters["n"]
        x = np.arange(m).tolist()
        y = [hypergeometric(N, m, n, i) for i in x]
        return {
            "xs": x,
            "ys": y,
            "variableType": "discrete"
        }
    
    
    if distribution == "Uniform Continuous":
        a = parameters["a"]
        b = parameters["b"]
        x = np.linspace(a, b, int(b-a)*10).tolist()
        y = [uniform(a, b, i) for i in x] 
        return {
            "xs": x,
            "ys": y,
            "variableType": "continuous"
        }
        
    
    if distribution == "Normal":
        mu = parameters["mu"]
        sigma_squared = parameters["sigma_squared"]
        
        x = np.linspace(mu - 3.1 * sigma_squared, mu + 3.1 * sigma_squared, 1000)
        y = normal_pdf(mu, sigma_squared, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }
        
    
    if distribution == "Exponential":
        lam = parameters["lambda"]
        max_value = math.log(10 * lam)
        x = np.linspace(0, max_value, 1000)
        y = exponential_pdf(lam, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }
        
    
    if distribution == "Gamma":
        alpha = parameters["k"]
        lam = parameters["lambda"]
        max_value = 20
        x = np.linspace(0, max_value, 1000)
        y = gamma_pdf(alpha, lam, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }
        
        
    
    if distribution == "Beta":
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        x = np.linspace(0, 1, 1000)
        y = beta_pdf(alpha, beta, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }
        
    
    if distribution == "Cauchy":
        theta = parameters["theta"]
        x = np.linspace(theta - 30, theta + 30, 1000)
        y = cauchy_pdf(theta, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }
        
    
    if distribution == "Weibull":
        alpha = parameters["lambda"]
        beta = parameters["k"]
        max_value = 5
        x = np.linspace(0, max_value, 1000)
        y = weibull_pdf(alpha, beta, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }
    
    if distribution == "Pareto":
        a = parameters["a"]
        lam = parameters["lambda"]
        max_value = a + math.log(a + lam)
        x = np.linspace(a, max_value, 1000)
        y = pareto_pdf(lam, a, x)
        
        return {
            "xs": x.tolist(),
            "ys": y.tolist(),
            "variableType": "continuous"
        }



@eel.expose
def compute_probability(distribution, parameters, value):
    if distribution == "Uniform Discrete":
        a = parameters["a"]
        b = parameters["b"]
        return uniform(a, b, value)
        
        
    if distribution == "Binomial":
        n = parameters["n"]
        p = parameters["p"]
        return binomial(n, p, value)

    
    if distribution == "Poisson":
        lam = parameters["lambda"]
        return poisson(lam, value)
    
    
    if distribution == "Geometric":
        p = parameters["p"]
        return geometric(p, value) 
        
    
    if distribution == "Negative Geometric":
        p = parameters["p"]
        r = parameters["r"]
        return negative_geometric(p, r, value)
        
    
    if distribution == "Hypergeometric":
        N = parameters["N"]
        m = parameters["m"]
        n = parameters["n"]
        return hypergeometric(N, m, n, value)
    
    
    if distribution == "Uniform Continuous":
        a = parameters["a"]
        b = parameters["b"]
        return uniform_cdf(a, b, value)
        
    
    if distribution == "Normal":
        mu = parameters["mu"]
        sigma_squared = parameters["sigma_squared"]        
        return normal_cdf(mu, sigma_squared, value)
    
    
    if distribution == "Exponential":
        lam = parameters["lambda"]  
        return exponential_cdf(lam, value)
             
    
    if distribution == "Gamma":
        alpha = parameters["k"]
        lam = parameters["lambda"]    
        return gamma_cdf(alpha, lam, value)
           
    
    if distribution == "Beta":
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        return beta_cdf(alpha, beta, value)
        
    
    if distribution == "Cauchy":
        theta = parameters["theta"]
        return cauchy_cdf(theta, value)
        
    
    if distribution == "Weibull":
        alpha = parameters["lambda"]
        beta = parameters["k"]
        return weibull_cdf(alpha, beta, value)
        
    
    if distribution == "Pareto":
        a = parameters["a"]
        lam = parameters["lambda"]
        return pareto_cdf(lam, a, value)
        

eel.start('index.html', port=5000)