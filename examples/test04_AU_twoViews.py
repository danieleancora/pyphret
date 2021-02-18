# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:50:20 2021

@author: Daniele Ancora
"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import skimage.external.tifffile as tiff
import pyphret.functions as pf
import pyphret.deconvolutions as pd

# load the dataset
satellite = tiff.imread('..//test_images//satellite.tif')
psf_long = tiff.imread('..//test_images//psf_long.tiff')
psf_round = tiff.imread('..//test_images//psf_round.tiff')

# psf normalization
satellite = satellite/satellite.mean()
psf_long /= psf_long.sum()
psf_round /= psf_round.sum()

# noise parameters and number of iterations
lambd = 2**4

psf00 = pf.gaussian_psf(size=[257,257], alpha=[1.5,4])
psf90 = pf.gaussian_psf(size=[257,257], alpha=[4,1.5])

psf00 = psf00[:256,:256]
psf90 = psf90[:256,:256]



# %% creating the measurement described in the experiment A - if results do not converse, re-run several times until snr grows
noise00 = (np.random.poisson(lam=lambd, size=satellite.shape))
noise90 = (np.random.poisson(lam=lambd, size=satellite.shape))

satellite_blur00 = pf.my_convolution(satellite, psf00)
satellite_blur90 = np.roll(pf.my_convolution(satellite, psf90), (10,-5), axis=(0,1))


measureA_00 = pf.my_autocorrelation(satellite_blur00)
measureA_90 = pf.my_autocorrelation(satellite_blur90)

measureA_00 = (2**16) * measureA_00/measureA_00.max()
measureA_90 = (2**16) * measureA_90/measureA_90.max()

apsf00 = pf.my_autocorrelation(psf00)
apsf90 = pf.my_autocorrelation(psf90)

measureA_blur00 = pf.my_convolution(measureA, apsf00)
measureA_blur90 = pf.my_convolution(measureA, apsf90)

measureA_00_blur_noise = np.abs(measureA_blur00 + noise00 - lambd)
measureA_90_blur_noise = np.abs(measureA_blur90 + noise90 - lambd)


# running the algorithm
iterations = 100000000
deconvolved_A, error_A, error_A2 = pd.anchorUpdateX(cp.asarray(measureA_00_blur_noise + measureA_90_blur_noise), 
                                          cp.asarray(apsf00 + apsf90), 
                                          cp.asarray(0), 
                                          kerneltype='A', iterations=iterations, 
                                          precision='float64')

deconvolved_A, error_A, error_A2 = deconvolved_A.get(), error_A.get(), error_A2.get()
deconvolved_A = pf.my_alignND(satellite, (deconvolved_A)) 
deconvolved_A = deconvolved_A /deconvolved_A.mean()


# %% Perfect deconvolution with RL
noise00 = (np.random.poisson(lam=lambd, size=satellite.shape))
noise90 = (np.random.poisson(lam=lambd, size=satellite.shape))

satellite_blurRL00 = pf.my_convolution(satellite, psf00)
satellite_blurRL90 = pf.my_convolution(satellite, psf90)

satellite_blurRL00 = (2**16) * satellite_blurRL00/satellite_blurRL00.max()
satellite_blurRL90 = (2**16) * satellite_blurRL90/satellite_blurRL90.max()

satellite_blurRL00 = np.abs(satellite_blurRL00 + noise00 - lambd)
satellite_blurRL90 = np.abs(satellite_blurRL90 + noise90 - lambd)

iterations = 1000
deconvolved_RL, error_RL = pd.richardsonLucy(cp.asarray(satellite_blurRL00 + satellite_blurRL90), 
                                          cp.asarray(psf00 + psf90), 
                                          cp.asarray(0), 
                                          iterations=iterations, 
                                          precision='float64')

deconvolved_RL, error_RL = deconvolved_RL.get(), error_RL.get()
deconvolved_RL = pf.my_alignND(satellite, (deconvolved_RL)) 
deconvolved_RL = deconvolved_RL /deconvolved_RL.mean()

plt.plot(error_RL)
plt.imshow(deconvolved_RL, vmax=satellite.max())


# %%
plt.figure(1)
plt.subplot(221), plt.imshow(satellite, vmax=satellite.max()), plt.title('original')
plt.subplot(222), plt.imshow(psf00), plt.title('psf that blurs the AUTOCORRELATION')
plt.subplot(223), plt.imshow(measureA_00_blur_noise + measureA_90_blur_noise), plt.title('blurred and noisy autocorrelation')
plt.subplot(224), plt.imshow(deconvolved_A, vmax=satellite.max()), plt.title('deconvolved deautocorrelated result')


plt.set_cmap('RdBu_r')


plt.figure(1, figsize=[8,4.2]),
plt.subplot(241), plt.imshow(satellite, vmax=satellite.max()), plt.title("A)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(242), plt.imshow(satellite_blur00, vmax=satellite_blur00.max()), plt.title("B)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(243), plt.imshow(satellite_blur90, vmax=satellite_blur00.max()), plt.title("C)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(244), plt.imshow(satellite_blur00+satellite_blur90, vmax=2*satellite_blur00.max()), plt.title("D)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(245), plt.imshow(pf.my_autocorrelation(satellite)), plt.title("E)", x=0.15, y=0.05, c='white'),  plt.axis('off')
plt.subplot(246), plt.imshow(measureA_00_blur_noise, vmax=measureA_00_blur_noise.max()), plt.title("F)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(247), plt.imshow(measureA_90_blur_noise, vmax=measureA_90_blur_noise.max()), plt.title("G)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(248), plt.imshow(measureA_00_blur_noise+measureA_90_blur_noise, vmax=2*measureA_90_blur_noise.max()), plt.title("H)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.tight_layout()


plt.figure(2, figsize=[8,2.3]),
plt.subplot(141), plt.imshow((psf00 + psf90)[64:-64,64:-64], vmax=(psf00 + psf90).max()),plt.title("A)", x=0.15, y=0.05, c='white'),  plt.axis('off')
plt.subplot(142), plt.imshow((apsf00 + apsf90)[64:-64,64:-64], vmax=(apsf00 + apsf90).max()), plt.title("B)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(143), plt.imshow(satellite, vmax=satellite.max()), plt.title("C)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.subplot(144), plt.imshow(deconvolved_A, vmax=satellite.max()), plt.title("D)", x=0.15, y=0.05, c='white'), plt.axis('off')
plt.tight_layout()



fig, ax1 = plt.subplots(figsize=[8,2.3])

color = 'tab:red'
ax1.set_xlabel('Iterations (t)')
ax1.set_ylabel('SNR (db)', color=color)
ax1.plot(error_A, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale('log')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Euclidean Distance (a.u.)', color=color)  # we already handled the x-label with ax1
ax2.plot(error_A2, color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')
#ax2.set_xscale('log')
plt.title("E)", x=0.03, y=0.05, c='black'), 
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()






# %% creating the measurement described in the experiment B - if results do not converse, re-run several times until snr grows
noise = np.random.poisson(lam=lambd, size=satellite.shape)
measureB = pf.my_convolution(satellite, psf_round)
measureB = (2**16) * measureB/measureB.max()
measureB_noise = measureB + noise
measureB_noise_corr = np.abs(pf.my_autocorrelation(measureB_noise - lambd))
    
# running the algorithm
deconvolved_B, error_B = pd.anchorUpdateX(cp.asarray(measureB_noise_corr), cp.asarray(psf_round), 
                                          cp.asarray(0), kerneltype='B', iterations=iterations, 
                                          precision='float32')

deconvolved_B, error_B = pd.schulzSnyder(cp.asarray(measureB_noise_corr), cp.asarray(0), iterations=iterations, precision='float32')

deconvolved_B, error_B = pd.richardsonLucy(cp.asarray(measureB), cp.asarray(psf_round), np.asarray(0), iterations=10, precision='float32')
deconvolved_B, error_B = deconvolved_B.get(), error_B.get()
deconvolved_B = pf.my_alignND(satellite, (deconvolved_B)) 
deconvolved_B = deconvolved_B/deconvolved_B.mean()

plt.figure(2)
plt.subplot(221), plt.imshow(satellite, vmax=satellite.max()), plt.title('original')
plt.subplot(222), plt.imshow(psf_round), plt.title('psf that blurs the OBJECT!')
plt.subplot(223), plt.imshow(measureB_noise_corr), plt.title('blurred and noisy autocorrelation')
plt.subplot(224), plt.imshow(deconvolved_B, vmax=satellite.max()), plt.title('deconvolved deautocorrelated result')


