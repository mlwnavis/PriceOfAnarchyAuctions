import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import scipy.interpolate as interp
import pandas as pd
import math
from scipy import stats

# ensure the notebook generate the same randomess
np.random.seed(1)

def generate_nPoA(R, N, equi, dist, args_equi, args_dist):
    def make_auction(v, dist, args):
        b =  dist(*args)
        winner_index=np.argmax(b, axis=0)
        Welf = v[winner_index, np.arange(R)]
        Opt =v.max(axis = 0)
        nPoA = Welf/Opt
        return nPoA
    PoA_mean = []
    PoA_min = []
    for n in range(2,N): 
        v = np.random.uniform(0,1,(n,R))
        args_equi_full = [v,n] + args_equi
        equilibrium = equi(*args_equi_full)
        args_dist_full = [v, equilibrium] + args_dist
        nPoA_raw = make_auction(v, dist, args_dist_full)
        PoA_mean.append(np.mean(nPoA_raw))
        PoA_min.append(min(nPoA_raw))
        

    return PoA_mean, PoA_min 
        

def index_of_sorted_elements(lst):
    sorted_indexes = [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)]
    return sorted_indexes

def draw_beta_dist(v, equi, a= 4):

    mode = equi/v
    b = (a * (1-mode)-1+2*mode)/mode

    # sample from beta
    return np.random.beta(a, b) * v



def produce_variance(mean, v):
    return 1/3 * (-abs(mean-v/2)+v/2)


def bet_Bayes_Nash(vi, N):
    return (N-1)/N*vi

def bet_Bayes_Nash_all_log(vi, N, base = None):
    if base is None:
        return (N-1)/N*vi**(math.log(N))
    else:
        return (N-1)/N*vi**(math.log(N, base*N))

def bet_Bayes_Nash_all_better(vi, N):
    return (2*N-2)/(2*N-1)*vi**N
    
    

def bet_Bayes_Nash_all(vi, N):
    return (N-1)/N*vi**N

def bet_Bayes_Nash_second(vi, N):
    return vi

def negative_to_zero(V, array):
    def limit_value(x):
        return max(0, min(V, x))
    
    limit_func = np.vectorize(limit_value)
    return limit_func(array)


def bet_random(v, equi):
    return np.random.uniform(0,v)


def bet_normal(v, mu, no_overbetting = 0, var_fac = 1):

    max_sample = v**no_overbetting
    sigma = produce_variance(mu, max_sample) * var_fac
    
    sample = np.random.normal(mu, sigma)
    zeros = np.zeros_like(sigma)
    sample[sample<zeros] = 0
    sample[sample>max_sample] = max_sample[sample>max_sample]
    return sample


def bet_uneven_Bayes_Nash(v, v_max):
    if v_max == 1:
        return 4/(3*v)*(1-(1-(3*v**2)/4)**(0.5))
    elif v_max == 2:
        return 4/(3*v)*((1+(3*v**2)/4)**(0.5)-1)
    else:
        return None

def bet_normal_uneven(v, v_max):
    return np.random.normal(bet_uneven_Bayes_Nash(v, v_max), 1/3*bet_uneven_Bayes_Nash(v, v_max))

def bet_between(vi, N, x):
    return (N-1)/N*vi**(N**x)

def bet_moderate(vi, N):
    return (N-1)/N*vi**(N**0.5)
