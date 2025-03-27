#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:05:06 2024

@author: daniele
"""

# LIBRARIES CALL
import time
import numpy as np

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########

# import internal to the library
import pyphret.backend as pyb
from pyphret.functions import my_convolution, my_correlation, my_convcorr, my_convcorr_sqfft, my_correlation_withfft, axisflip, snrIntensity_db, my_correlation_alongaxes, my_convolution_alongaxes



# %%
def schulzSnyder(correlation, initialization=np.float32(0), iterations=10, precision='float64', measure=True, verbose=True):
    # check modules for agnosticity and start timing for performance evaluation
    xp = pyb.get_array_module(correlation)
    start_time = time.time()

    # starting guess with a flat image
    if initialization.any()==0:
        signal_decorr = xp.full(correlation.shape,0.5) + 0.01*xp.random.rand(*correlation.shape)
    else:
        signal_decorr = initialization.copy()
        
    # useful quantities
    epsilon = 1e-7
    R_0 = signal_decorr.sum()                           # normalization factor
    # signal_decorr = signal_decorr / R_0                  # normalized image
    relative_corr = xp.zeros_like(signal_decorr)

    # cast to choosen precision
    correlation, signal_decorr, relative_corr = correlation.astype(precision), signal_decorr.astype(precision), relative_corr.astype(precision)

    # to measure the distance between the autocorrelated guess and the signal
    if measure == True: error = xp.zeros(iterations)
    else:               error = None
    
    
    ###########################################################################
    # ALGORITHM STARTS HERE ###################################################
    ###########################################################################

    for i in range(iterations):
        
        # first step of the algorithm
        estimated_corr = my_correlation(signal_decorr, signal_decorr)
        

        # this needs to be calculated before the second step so that we compare what currently reconstructed
        if measure==True:
            error[i] = xp.linalg.norm(correlation-relative_corr)

        if verbose==True and (i % 100)==0 and measure==False:
            print('Iteration ' + str(i))
        elif verbose==True and (i % 100)==0 and measure==True:
            print('Iteration ' + str(i) + ' - autocorrelation distance: ' + str(error[i]))


        # second step of the algorithm
        relative_corr = correlation / estimated_corr


        # avoid errors due to division by zero or inf
        relative_corr[xp.isinf(relative_corr)] = epsilon 
        relative_corr = xp.nan_to_num(relative_corr)
        relative_corr = xp.abs(relative_corr)


        # multiplicative update 
        signal_decorr *= my_correlation(relative_corr, signal_decorr)/ R_0
        

    print("\n\n Algorithm finished. Performance:")
    print("--- %s seconds ----" % (time.time() - start_time))
    print("--- %s sec/step ---" % ((time.time() - start_time)/iterations))

    return signal_decorr, error


# %%
def schulzSnyder_fix(correlation, fix, initialization=np.float32(0), iterations=10, precision='float64', measure=True, verbose=True):
    # check modules for agnosticity and start timing for performance evaluation
    xp = pyb.get_array_module(correlation)
    start_time = time.time()

    # starting guess with a flat image
    if initialization.any()==0:
        signal_decorr = xp.full(correlation.shape,0.5) + 0.01*xp.random.rand(*correlation.shape)
    else:
        signal_decorr = initialization.copy()
        
    # useful quantities
    epsilon = 1e-7
    R_0 = signal_decorr.sum()                           # normalization factor
    # signal_decorr = signal_decorr / R_0                  # normalized image
    relative_corr = xp.zeros_like(signal_decorr)

    # cast to choosen precision
    correlation, signal_decorr, relative_corr = correlation.astype(precision), signal_decorr.astype(precision), relative_corr.astype(precision)

    # to measure the distance between the autocorrelated guess and the signal
    if measure == True: error = xp.zeros(iterations)
    else:               error = None
    
    
    ###########################################################################
    # ALGORITHM STARTS HERE ###################################################
    ###########################################################################

    for i in range(iterations):
        
        # first step of the algorithm
        estimated_corr = my_correlation(signal_decorr, fix)
        

        # this needs to be calculated before the second step so that we compare what currently reconstructed
        if measure==True:
            error[i] = xp.linalg.norm(correlation-relative_corr)

        if verbose==True and (i % 100)==0 and measure==False:
            print('Iteration ' + str(i))
        elif verbose==True and (i % 100)==0 and measure==True:
            print('Iteration ' + str(i) + ' - autocorrelation distance: ' + str(error[i]))


        # second step of the algorithm
        relative_corr = correlation / estimated_corr


        # avoid errors due to division by zero or inf
        relative_corr[xp.isinf(relative_corr)] = epsilon 
        relative_corr = xp.nan_to_num(relative_corr)
        relative_corr = xp.abs(relative_corr)


        # multiplicative update 
        signal_decorr *= my_correlation(relative_corr, signal_decorr)/ R_0
        

    print("\n\n Algorithm finished. Performance:")
    print("--- %s seconds ----" % (time.time() - start_time))
    print("--- %s sec/step ---" % ((time.time() - start_time)/iterations))

    return signal_decorr, error



