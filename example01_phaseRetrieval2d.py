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
import pyphret.deconvolutions as pd

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

# calculate the magnitude of the test image
fftmagnitude = pf.autocorrelation2fouriermod(test_xcorr)
fftmagnitude = cp.asarray(fftmagnitude)

# running a phase retrieval problem
(retrieved, mask, error) = pr.phaseRet(fftmagnitude, 
                                       rec_prior=None, phase_prior=None, 
                                       masked='autocorrelation', method='HIO', mode='shrink-wrap',
                                       measure = True,
                                       beta=0.9, steps=1000, parameters=[100, 30, 1.5, 0.9, 1])
retrieved = retrieved.get()
mask = mask.get()
error1 = error.get()

plt.figure()
plt.subplot(221),
plt.imshow(test_image)
plt.subplot(222),
plt.imshow(test_xcorr)
plt.subplot(223),
plt.imshow(np.real(retrieved*mask))
plt.subplot(224),
plt.imshow(np.uint8(mask))

retrieved = retrieved*mask
retrieved = retrieved[::-1, ::-1]


retrieved_aligned = pf.my_alignND(test_image, retrieved, mode='flip')
# retrieved_aligned = pf.my_alignND(test_image, retrieved)
plt.figure(1)
plt.subplot(221), plt.imshow(test_image)
plt.subplot(222), plt.imshow(retrieved_aligned)
plt.subplot(223), plt.imshow(test_image - retrieved_aligned)
plt.subplot(224), plt.imshow(retrieved)





(retrieved, mask, error) = pr.phaseRet(fftmagnitude, 
                                rec_prior=None, phase_prior=None, 
                                masked='circular', method='HIO', mode='normal',
                                measure = True,
                                beta=0.9, steps=2000)
retrieved = retrieved.get()
mask = mask.get()
error2 = error.get()

plt.figure()
plt.plot(error1)
plt.plot(error2)


# %%
test_xcorr.max()

noise = 1*(np.random.poisson(lam=25, size=test_xcorr.shape) - 10)
noise = 0.01 * noise / noise.max() * test_xcorr.max()
noise.max()

test_xcorr_poisson = test_xcorr + noise

plt.figure()
plt.subplot(131), plt.imshow(test_xcorr)
plt.subplot(132), plt.imshow(noise)
plt.subplot(133), plt.imshow(test_xcorr_poisson)



(x_hat, ratio) = pd.invert_autoconvolution(magnitude=test_xcorr_poisson, prior=None, mask=None, 
                       steps=420, mode='deautocorrelation', verbose=True)

x_hat = pf.my_alignND(test_image, x_hat, mode='flip')

plt.figure(1)
plt.subplot(131), plt.imshow(test_image)
plt.subplot(132), plt.imshow(x_hat)
plt.subplot(133), plt.plot(ratio[1:-1])









