# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:04:49 2020

@author: Daniele Ancora
"""


import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import skimage.external.tifffile as tiff
import pyphret.deconvolutions as pd

# import correlation convolution routines
from scipy import signal
import pyphret.functions as pf
import cusignal.convolution as pyconv


# %% load the dataset
satellite = tiff.imread('..//test_images//satellite.tif')
psf_long = tiff.imread('..//test_images//psf_long.tiff')
psf_round = tiff.imread('..//test_images//psf_round.tiff')

# test = np.zeros_like(satellite)
# test[64:-64,64:-64] = satellite[64:-64,64:-64]
# satellite = test.copy()

# normalization
satellite = satellite / satellite.mean()
psf_long /= psf_long.sum()
psf_round /= psf_round.sum()

image1 = satellite.copy()
image2 = satellite.copy()

plt.subplot(121), plt.imshow(image1)
plt.subplot(122), plt.imshow(image2)


# %% test my convolutions
autoconv = pf.my_convolution(image1, image2[::-1,::-1])
autocorr = pf.my_correlation(image1, image2)
autoconv /= autoconv.max()
autocorr /= autocorr.max()

print(np.array_equal(autoconv,autocorr))

plt.subplot(131), plt.imshow(autoconv)
plt.subplot(132), plt.imshow(autocorr)
plt.subplot(133), plt.imshow(np.abs(autoconv-autocorr))


# test scipy convolutions
autoconv = signal.convolve(image1, image2[::-1,::-1], mode='same', method='fft')
autocorr = signal.correlate(image1, image2, mode='same', method='fft')
autoconv /= autoconv.max()
autocorr /= autocorr.max()

print(np.array_equal(autoconv,autocorr))


import cupyx.scipy.signal

temp = cupyx.scipy.signal.convolve(cp.asarray(image1), cp.asarray(image2[::-1,::-1]))


# test cusignal convolutions
autoconv = pyconv.convolve(image1, image2[::-1,::-1], mode='same', method='fft').get()
autocorr = pyconv.correlate(image1, image2, mode='same', method='fft').get()
autoconv /= autoconv.max()
autocorr /= autocorr.max()

print(np.array_equal(autoconv,autocorr))

plt.subplot(131), plt.imshow(autoconv)
plt.subplot(132), plt.imshow(autocorr)
plt.subplot(133), plt.imshow(np.abs(autoconv-autocorr))

