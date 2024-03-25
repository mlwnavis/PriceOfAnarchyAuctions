import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import scipy.interpolate as interp
import pandas as pd

# ensure the notebook generate the same randomess
np.random.seed(1)

def index_of_sorted_elements(lst):
    sorted_indexes = [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)]
    return sorted_indexes

def bet_Bayes_Nash(N, vi):
    return (N-1)/N*vi
    
def negative_to_zero(array):
    return pd.DataFrame(array).applymap(lambda x: max(0, x))
    
def bet_random(v):
    return np.random.uniform(0,v)

def bet_normal(n, v):
    return np.random.normal(bet_Bayes_Nash(n,v), 1/3*bet_Bayes_Nash(n,v))

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