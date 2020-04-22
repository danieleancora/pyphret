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
import matplotlib.pyplot as plt
import pyphret.functions as pf
import pyphret.retrievals as pr

# we produce a test image by generating randomly distributed beads
#Read in source image
test_image = plt.imread("test_images//einstein.bmp")
test_image = np.pad(test_image, (0,300))
test_image = test_image/np.max(test_image)
# test_image = pf.my_gaussblur(test_image,4)

# calculate the autocorrelation of the test image
test_xcorr = pf.my_autocorrelation(test_image)


# calculate the magnitude of the test image
fftmagnitude = np.abs(np.fft.rfft2(test_image))
fftmagnitude = cp.asarray(fftmagnitude)

# running a phase retrieval problem
(retrieved, mask, _) = pr.phaseRet(fftmagnitude, rec_prior=None, phase_prior=None, masked='half',
                                    method='HIO', mode='classical',
                                    beta=0.9, steps=2000)
(retrieved, mask, _) = pr.phaseRet(fftmagnitude, rec_prior=retrieved, phase_prior=None, masked='half',
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





# # testing functionalities.
# mask = pf.sparsity_maskND(test_image, 0.9999)
# mask = pf.autocorrelation_maskND(test_image, 0.85)
# print(np.max(mask*test_image))
# plt.imshow(np.uint8(mask))
# np.min(test_image)



x_hat = pd.invert_autoconvolution(magnitude=(test_xcorr), prior=None, mask=None, 
                       steps=1000, mode='deautocorrelation', verbose=True)

x_hat = pf.my_alignND(test_image, x_hat, mode='flip')
plt.figure(1)
plt.subplot(121), plt.imshow(test_image)
plt.subplot(122), plt.imshow(x_hat)
