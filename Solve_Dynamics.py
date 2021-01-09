#!/usr/bin/env python
# coding: utf-8


from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import time
import numpy as np
from PG_Dyn import pg_dyn

def solve_dynamics(g0, y0, p, gamma, K, nbus, d, ti, tf, tint):
    start = time.perf_counter()
    tspan = np.linspace(ti,tf,num=tint,endpoint=True)
    sol_t = odeint(pg_dyn, y0, tspan, args=(g0, p, gamma, K, nbus, d))
    finish = time.perf_counter()
    print(f'odeint finished in {(finish-start):.3f} seconds')
    return tspan, sol_t