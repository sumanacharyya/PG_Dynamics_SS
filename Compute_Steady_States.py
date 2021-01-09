#!/usr/bin/env python
# coding: utf-8

from scipy import optimize
from Function import fn
from Jacobian import dfn
import time
import numpy as np

def compute_steady_states(G, P, K, Nbus, theta0):
    start = time.perf_counter()
    theta = optimize.fsolve(func=fn, x0=theta0, args=(G, P, K, Nbus,), fprime=dfn,
                            full_output=0, col_deriv=0, xtol=1.49012e-05, maxfev=0, band=None, epsfcn=None, factor=100, diag=None)
    finish = time.perf_counter()
    print(f'optimize.fsolve finished in {(finish-start):.3f} seconds')
    return theta
