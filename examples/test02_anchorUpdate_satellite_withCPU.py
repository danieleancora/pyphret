# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 16:06:04 2020

@author: Daniele Ancora
"""

import time
import numpy as np
# import cupy as cp
import matplotlib.pyplot as plt
import skimage.external.tifffile as tiff
import pyphret.functions as pf
import pyphret.deconvolutions as pd

# load the dataset
satellite = tiff.imread('..//test_images//satellite.tif')
psf_long = tiff.imread('..//test_images//psf_long.tiff')
psf_round = tiff.imread('..//test_images//psf_round.tiff')

# psf normalization
psf_long /= psf_long.sum()
psf_round /= psf_round.sum()

# noise parameters and number of iterations
lambd = 2**4
iterations = 10000
    

# %% creating the measurement described in the experiment A - if results do not converse, re-run several times until snr grows
noise = (np.random.poisson(lam=lambd, size=satellite.shape))
measureA = pf.my_autocorrelation(satellite)
measureA = (2**16) * measureA/measureA.max()
measureA_blur = pf.my_convolution(measureA, psf_long)
measureA_blur_noise = measureA_blur + noise - lambd

# running the algorithm
deconvolved_A, error_A = pd.anchorUpdateX(measureA_blur_noise, psf_long, np.zeros_like(psf_long), kerneltype='A', iterations=iterations)
deconvolved_A = pf.my_alignND(satellite, (deconvolved_A)) 

plt.figure(1)
plt.subplot(221), plt.imshow(satellite), plt.title('original')
plt.subplot(222), plt.imshow(psf_long), plt.title('psf that blurs the AUTOCORRELATION')
plt.subplot(223), plt.imshow(measureA_blur_noise), plt.title('blurred and noisy autocorrelation')
plt.subplot(224), plt.imshow(deconvolved_A), plt.title('deconvolved deautocorrelated result')


# %% creating the measurement described in the experiment B - if results do not converse, re-run several times until snr grows
noise = np.random.poisson(lam=lambd, size=satellite.shape)
measureB = pf.my_convolution(satellite, psf_round)
measureB = (2**16) * measureB/measureB.max()
measureB_noise = measureB + noise
measureB_noise_corr = pf.my_autocorrelation(measureB_noise - lambd)
    
# running the algorithm
deconvolved_B, error_B = pd.anchorUpdateX(measureB_noise_corr, psf_round, kerneltype='B', iterations=iterations)
deconvolved_B = pf.my_alignND(satellite, (deconvolved_B)) 

plt.figure(2)
plt.subplot(221), plt.imshow(satellite), plt.title('original')
plt.subplot(222), plt.imshow(psf_round), plt.title('psf that blurs the OBJECT!')
plt.subplot(223), plt.imshow(measureB_noise_corr), plt.title('blurred and noisy autocorrelation')
plt.subplot(224), plt.imshow(deconvolved_B), plt.title('deconvolved deautocorrelated result')


