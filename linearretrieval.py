#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 17:52:57 2021

@author: danieleancora
"""

import time
import numpy as np
# import scipy.ndimage
# from scipy import signal 

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########

import pyphret.backend as pyb
# import pyphret.cusignal.convolution as pyconv
from pyphret.functions import my_convolution, my_correlation, my_convcorr, my_convcorr_sqfft, my_correlation_withfft, axisflip, snrIntensity_db, my_correlation_alongaxes, my_convolution_alongaxes



def linearGerchbergSaxton(x_cpl, y_mod, power=1, iterations=10000):
    # select best device and compute problem size
    xp = pyb.get_array_module(x_cpl)

    # measurement matrix and its inverse (Moore-Penrose)
    P = x_cpl.T
    Pinv = xp.linalg.pinv(P)
        
    # make sure all the variables are cast to complex
    # P = xp.complex64(P)
    # Pinv = xp.complex64(Pinv)
    
    # move measures to torch tensor
    Y_mod = y_mod.T
    
    TMgs = xp.matmul(Pinv, Y_mod * xp.exp(1j*2*xp.pi*xp.random.rand(Y_mod.shape[0],Y_mod.shape[1])))  
    Y_out = xp.matmul(P,TMgs)
    
    # initialize metric and exponent operator
    metric = xp.zeros((iterations,))
    exponent = xp.linspace(power,1,iterations)
    
    # MAIN CICLE - TIME MONITOR ----------------------------------------------
    start = time.time()
    
    for i in range(iterations):
        
        # output field estimation, forcing the known modulus
        Y_out = (Y_mod) * xp.exp(1j*xp.angle(Y_out))
        TMgs = xp.matmul(Pinv,Y_out)
        Y_out = xp.matmul(P,TMgs)
        
        metric[i] = ((Y_mod-xp.abs(Y_out))**2).mean()
    
    end = time.time()
    # MAIN CICLE - TIME MONITOR ----------------------------------------------

    # print some useful informations
    print('Total elapsed time: ' + str(end - start))
    print('Elapsed time per step: ' + str((end - start)/iterations))
    
    return (TMgs.T), metric
