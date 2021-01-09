#!/usr/bin/env python
# coding: utf-8

# Routine for random assignment of generator and load power

import random
import numpy as np

random.seed(a=246, version=2)
def assign_gen_load_power(pgen, pload, ngen, nload, nbus):
    power = np.zeros(nbus)
    #assign power to the generator nodes
    gidx = []
    i = ngen
    while i>0:
        j = random.randint(0,nbus-1)
        if power[j]==0.0:
            power[j] = pgen
            gidx.append(j)
            i = i-1
    #assign power to the load nodes
    lidx = []
    i = nload
    while i>0:
        j = random.randint(0,nbus-1)        
        if power[j]==0.0:
            power[j] = pload
            lidx.append(j)
            i = i-1
    return gidx, gidx, power
