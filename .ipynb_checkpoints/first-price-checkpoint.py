import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import scipy.interpolate as interp

def bet_Bayes_Nash(N, vi):
    return (N-1)/N*vi

np.random.seed(1337)



N = 2
R = 100000


v = np.random.uniform(0,1,(N,R))
b =  bet_Bayes_Nash(N,v)

x=np.argmax(b, axis=0)
Welf = v[x, np.arange(R)]
Opt = v.max(axis = 0)

print(np.mean(Welf/Opt))
print(np.mean(Welf/Opt))
print(np.mean(Welf/Opt))v