# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:45:59 2020

Example 1: here we randomly turn on few bids in a given image that have gaussian 
intensity distribution. From there, we calculate the fourier modulus of such
image and we try to look for a solution of this problem retrieving the phase.

@author: Daniele Ancora
"""

import numpy as np
import cupy as cp
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import pyphret.functions as pf
import pyphret.retrievals as pr

# we produce a test image by generating randomly distributed beads
#Read in source image
test_image = plt.imread("einstein.bmp")
test_image = np.pad(test_image, (0,150))
test_image = test_image/np.max(test_image)
# test_image = pf.my_gaussblur(test_image,4)


# calculate the magnitude of the test image
fftmagnitude = np.abs(np.fft.rfft2(test_image))
fftmagnitude = cp.asarray(fftmagnitude)

# running a phase retrieval problem
(retrieved, mask) = pr.phaseRet(fftmagnitude, rec_prior=None, phase_prior=None, masked='half',
                                    method='HIO', mode='normal',
                                    beta=0.9, steps=2000)
(retrieved, mask) = pr.phaseRet(fftmagnitude, rec_prior=retrieved, phase_prior=None, masked='half',
                                    method='ER', mode='normal',
                                    beta=0.9, steps=2000)
retrieved = retrieved.get()
mask = mask.get()

plt.figure()
plt.subplot(221),
plt.imshow(test_image)
plt.subplot(222),
plt.imshow(fftmagnitude.get())
plt.subplot(223),
plt.imshow(np.real(retrieved))
