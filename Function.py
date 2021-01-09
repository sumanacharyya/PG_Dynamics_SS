import numpy as np
import math

# definition of the equation
def fn(theta, G, p, k, n):
    y = np.zeros(n)
    for i in G.nodes():
        h = 0.0
        for j in G.neighbors(i):
            h = h + k*math.sin(theta[j] - theta[i])
        y[i] = p[i] + h
    return y