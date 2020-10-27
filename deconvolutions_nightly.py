# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:41:18 2020

@author: Daniele Ancora
"""

# %% LIBRARIES CALL
import time

import scipy.ndimage
import cupyx.scipy.ndimage

import cupy as cp
import numpy as np

from pyphret.functions import my_convolution, my_correlation, my_convcorr, my_convcorr_sqfft, my_correlation_withfft, axisflip, snrIntensity_db


# %% DE-AUTOCORRELATION AND GAUSSIAN DECONVOLUTION ROUTINES WITH DEBLURRING
def anchorUpdateSK(signal, kernel, signal_deconv=np.float32(0), iterations=10, measure=True, clip=False, verbose=True):
    
    # for code agnosticity between Numpy/Cupy
    xp = cp.get_array_module(signal)
    xps = cupyx.scipy.get_array_module(signal)

    # for performance evaluation
    start_time = time.time()
    
    if iterations<100: 
        breakcheck = iterations
    else:
        breakcheck = 100

    # normalization
    signal /= signal.sum()
    epsilon = 1e-7

    # starting guess with a flat image
    if signal_deconv.any()==0:
        # xp.random.seed(0)
        signal_deconv = xp.full(signal.shape,0.5) + 0.01*xp.random.rand(*signal.shape)
        # signal_deconv = signal.copy()
    else:
        signal_deconv = signal_deconv #+ 0.1*prior.max()*xp.random.rand(*signal.shape)
    
    # normalization
    signal_deconv = signal_deconv/signal_deconv.sum()
        
    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        # I use this property to make computation faster
        kernel_update = xps.ndimage.gaussian_filter(signal_deconv, sigma)
        # kernel_update = xps.ndimage.fourier_gaussian(signal_deconv, sigma)
        
        kernel_mirror = (kernel_update)
        
        relative_blur = my_correlation(signal_deconv, kernel_update)
        
        # compute the measured distance metric if given
        if measure==True:
            # error[i] = xp.linalg.norm(signal/signal.sum()-relative_blur/relative_blur.sum())
            error[i] = snrIntensity_db(signal/signal.sum(), xp.abs(signal/signal.sum()-relative_blur/relative_blur.sum()))
            if (error[i] < error[i-breakcheck]) and i > breakcheck:
                break

        if verbose==True and (i % 100)==0 and measure==False:
            print('Iteration ' + str(i))
        elif verbose==True and (i % 100)==0 and measure==True:
            print('Iteration ' + str(i) + ' - noise level: ' + str(error[i]))

        relative_blur = signal / relative_blur

        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update, for the full model
        signal_deconv *= 0.5 * (my_convolution(relative_blur, kernel_mirror) + my_correlation(axisflip(relative_blur), kernel_mirror))
        # signal_deconv *= (my_convolution(relative_blur, kernel_mirror) + my_correlation(relative_blur,kernel_mirror))


        # multiplicative update, for the Anchor Update approximation
        # signal_deconv *= my_convolution(kernel_mirror, relative_blur)

        # multiplicative update, remaining term. This gives wrong reconstructions
        # signal_deconv *= my_correlation(axisflip(relative_blur), kernel_mirror)
                
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1

    print("\n\n Algorithm finished. Performance:")
    print("--- %s seconds ----" % (time.time() - start_time))
    print("--- %s sec/step ---" % ((time.time() - start_time)/iterations))
    return signal_deconv, error #,kernel_update
