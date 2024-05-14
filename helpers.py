import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import scipy.interpolate as interp
import pandas as pd
import math
from scipy.stats import truncnorm


# ensure the notebook generate the same randomess
np.random.seed(1)

def index_of_sorted_elements(lst):
    sorted_indexes = [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)]
    return sorted_indexes

        
def produce_variance(mean, v):
    return 1/3 * (-abs(mean-v/2)+v/2)


def bet_Bayes_Nash(N, vi):
    return (N-1)/N*vi

def bet_Bayes_Nash_all_log(N, vi, base = None):
    if base is None:
        return (N-1)/N*vi**(math.log(N))
    else:
        return (N-1)/N*vi**(math.log(N, base))

def bet_Bayes_Nash_all_better(N,vi):
    return (2*N-2)/(2*N-1)*vi**N
    
    

def bet_Bayes_Nash_all(N,vi):
    return (N-1)/N*vi**N

def bet_Bayes_Nash_second(N, vi):
    return vi

def negative_to_zero(array, V):
    def limit_value(x):
        return max(0, min(V, x))
    
    limit_func = np.vectorize(limit_value)
    return limit_func(array)


def bet_random(v):
    return np.random.uniform(0,v)

def get_truncated_normal(mean=0, sd=1, low=0, upp=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def bet_normal(mu, sigma):
    return np.random.normal(mu, sigma)

def bet_beta(n, v):
    return np.random.beta()

def bet_uneven_Bayes_Nash(v, v_max):
    if v_max == 1:
        return 4/(3*v)*(1-(1-(3*v**2)/4)**(0.5))
    elif v_max == 2:
        return 4/(3*v)*((1+(3*v**2)/4)**(0.5)-1)
    else:
        return None

def bet_normal_uneven(v, v_max):
    return np.random.normal(bet_uneven_Bayes_Nash(v, v_max), 1/3*bet_uneven_Bayes_Nash(v, v_max))

def bet_between(N,vi, x):
    return (N-1)/N*vi**(N**x)

def bet_moderate(N,vi):
    return (N-1)/N*vi**(N**0.5)
