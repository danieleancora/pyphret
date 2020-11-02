# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:26:55 2020

@author: Daniele Ancora
"""

import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import skimage.external.tifffile as tiff
import pyphret.functions as pf
import pyphret.deconvolutions as pd

from scipy import signal 
import pyphret.cusignal.convolution as pyconv


# load the dataset
satellite = tiff.imread('..//test_images//satellite.tif')
psf_long = tiff.imread('..//test_images//psf_long.tiff')
psf_round = tiff.imread('..//test_images//psf_round.tiff')

satellite = satellite / satellite.mean()

satellite_pad = np.zeros((256+64,256+64))
satellite_pad[:256,:256] = satellite

psf_round = pf.gaussian_psf(size=satellite_pad.shape, alpha=[3,3])

# psf normalization
psf_long /= psf_long.sum()
psf_round /= psf_round.sum()

# noise parameters and number of iterations
lambd = 2**4
iterations = 1000


# %% creating the measurement described in the experiment B - if results do not converse, re-run several times until snr grows
noise = np.random.poisson(lam=lambd, size=satellite_pad.shape)
measureB = signal.convolve(satellite_pad, psf_round, mode='same', method='fft')
measureB = (2**16) * measureB/measureB.max()
measureB_noise = measureB + noise - lambd
measureB_noise_corr = np.abs(signal.correlate(measureB_noise,measureB_noise, mode='same', method='fft'))
    
# running the algorithm
# deconvolved_B, error_B = pd.anchorUpdateX(cp.asarray(measureB_noise_corr), cp.asarray(psf_round), 
#                                           cp.asarray(0), kerneltype='B', iterations=iterations)

deconvolved_B, error_B = pd.schulzSnyder(cp.asarray(measureB_noise_corr), cp.asarray(0), iterations=iterations)
deconvolved_B, error_B = pd.schulzSnyderSK(cp.asarray(measureB_noise_corr), cp.asarray(0), iterations=iterations)

# deconvolved_B, error_B = pd.richardsonLucy(np.asarray(measureB), np.asarray(psf_round), np.asarray(0), iterations=50)


deconvolved_B, error_B = deconvolved_B.get(), error_B.get()
deconvolved_B = pf.my_alignND(satellite_pad, (deconvolved_B)) 
deconvolved_B = deconvolved_B/deconvolved_B.mean()

plt.figure(2)
plt.subplot(221), plt.imshow(satellite, vmax=satellite.max()), plt.title('original')
plt.subplot(222), plt.imshow(psf_round), plt.title('psf that blurs the OBJECT!')
plt.subplot(223), plt.imshow(measureB_noise_corr), plt.title('blurred and noisy autocorrelation')
plt.subplot(224), plt.imshow(deconvolved_B), plt.title('deconvolved deautocorrelated result')
