import numpy as np
import math

# coupled dynamics to be solved using odeint
def pg_dyn(y, t, G, p, gamma, k, n, d):
    dydt = np.zeros(n*d)
    for i in G.nodes:
        h = 0.0
        for j in G.neighbors(i):
            h = h + k*math.sin(y[j*d+0] - y[i*d+0])
        dydt[i*d+0] = y[i*d+1]
        dydt[i*d+1] = p[i] - gamma*y[i*d+1] + h
    return dydt