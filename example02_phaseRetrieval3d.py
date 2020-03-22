# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:00:45 2020

Example 2: here we randomly turn on few bids in a given volume that have gaussian 
intensity distribution. From there, we calculate the fourier modulus of such
volume and we try to look for a solution of this problem retrieving the phase.

@author: Daniele Ancora
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pyphret.functions as pf
import pyphret.retrievals as pr

# we produce a test image by generating randomly distributed beads in a spherical volume
size = (256, 256, 256)
size_pad = 128
test_image = np.random.rand(*size)
test_image = (test_image > 0.99999).astype(np.float64) * pf.spherical_mask3D(size[0], size[1], size[2])
test_image = np.pad(test_image, (size_pad,size_pad), 'constant')
test_image = pf.my_gaussblur(test_image, 1.5)

# calculate the autocorrelation of the test image
test_xcorr = pf.my_autocorrelation(test_image)

# calculate the magnitude of the test image and move it to the GPU
fftmagnitude = pf.autocorrelation2fouriermod(test_xcorr)
fftmagnitude = cp.asarray(fftmagnitude)

# running a phase retrieval problem in 3D
retrieved = pr.phaseRet(fftmagnitude, rec_prior=None, phase_prior=None, masked='spherical',
                        method='HIO', mode='normal',
                        beta=0.9, steps=2000)

retrieved = retrieved.get()
retrieved = pf.my_allign3d_flip(test_image, retrieved)

# figure plot
plt.figure()
plt.subplot(221),
plt.imshow(np.mean(test_image, axis=0))
plt.subplot(222),
plt.imshow(np.mean(test_xcorr, axis=0))
plt.subplot(223),
plt.imshow(np.mean(retrieved, axis=0))

plt.figure()
plt.subplot(221),
plt.imshow(test_image[:,:,128])
plt.subplot(222),
plt.imshow(test_xcorr[:,:,128])
plt.subplot(223),
plt.imshow(retrieved[:,:,128])


