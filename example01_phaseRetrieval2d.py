# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 20:46:01 2020

@author: Daniele Ancora
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pyphret.functions as pf
import pyphret.retrievals as pr

# we produce a test image by generating randomly distributed beads
size = 256
size_pad = 128

test_image = np.random.rand(size,size)
test_image = (test_image > 0.999).astype(np.float32) * pf.circular_mask2D(size, size)
test_image = np.pad(test_image, (size_pad,size_pad), 'constant')
test_image = pf.my_gaussblur(test_image,4)

mask = pf.sparsity_maskND(test_image, 0.1)

# calculate the autocorrelation of the test image
test_xcorr = pf.my_autocorrelation(test_image)

plt.imshow(pf.autocorrelation_maskND(test_image, 0.05))

# calculate the magnitude of the test image
fftmagnitude = pf.autocorrelation2fouriermod(test_xcorr)
fftmagnitude = cp.asarray(fftmagnitude)

# running a phase retrieval problem
(retrieved, mask) = pr.phaseRet(fftmagnitude, rec_prior=None, phase_prior=None, masked='circle',
                                    method='HIO', mode='normal',
                                    beta=0.9, steps=2000)
retrieved = retrieved.get()
mask = mask.get()

plt.figure()
plt.subplot(221),
plt.imshow(test_image)
plt.subplot(222),
plt.imshow(test_xcorr)
plt.subplot(223),
plt.imshow(np.real(retrieved))
plt.subplot(224),
plt.imshow(np.uint8(mask))
