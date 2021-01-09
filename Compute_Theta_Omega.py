#!/usr/bin/env python
# coding: utf-8


import numpy as np

def compute_theta_omega(sol, nbus, d):
    row, col = sol.shape
    omega = np.zeros((row,nbus))
    theta = np.zeros((row,nbus))
    for i in range(row):
        for j in range(nbus):
            omega[i][j] = sol[i][j*d+1]
            theta[i][j] = sol[i][j*d+0]
    return theta, omega