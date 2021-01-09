import numpy as np
import math

# definition of the steady state Jacobian
def dfn(theta, G, p, k, n):
    dy = np.zeros((n,n))
    for i in G.nodes():
        h = 0.0
        for j in G.neighbors(i):
                dy[i][j] = k*math.cos(theta[j] - theta[i])
                h = h + dy[i][j]
        dy[i][i] = -h
    return dy