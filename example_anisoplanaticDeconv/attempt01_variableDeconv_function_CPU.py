# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:47:09 2021

@author: Daniele Ancora
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.external.tifffile as tiff
from scipy import signal, ndimage
from skimage.util import view_as_windows, view_as_blocks

# from the pyphret package
import pyphret.functions as pyf
import pyphret.stackoperation as pys
import pyphret.deconvolutions as pyd
import pyphret.anisoplanatic as pya


# %% IMPORT the files
folder = 'materiale_deconvoluzione01//'
filename1 = folder + 'psf_all5x5.tif'
filename2 = folder + 'Snap_wide.tiff'
filename3 = folder + 'dark.tiff'

# image that we have to deconvolve, slight padding
imag = tiff.imread(filename2).astype('float32')
imag = np.pad(imag, (1,1))

# load the psf mapped on a grid. IT HAS TO BE A STACK, then we reshape it
psfmap = tiff.imread(filename1).astype('float32')
psfmap = psfmap.reshape(5,5,psfmap.shape[1],psfmap.shape[2])

# backgroung acquisition
dark = tiff.imread(filename3).astype('float32')
dark = np.pad(dark, (1,1))

# we subtract the background
imag = imag - dark

# visualize the psf stack
tiff.imshow(psfmap)


# %% PARAMETER DEFINITION
windowSize = 410 # it has to divede the size of the image
windowStep = 82
windowOverlap = 32


# %% FIRST METHOD - FFT DECONVOLUTION - fast
iterations = 15

# anisoplanatic
imag_rebuild, error = pya.varyingDeconvRL_fft(imag, psfmap, iterations, windowSize, windowStep, windowOverlap)

# isoplanatic, central kernel
psfcenter = np.pad(psfmap[2,2,:,:],((986,986),(986,986)))
imag_RL, error_RL = pyd.richardsonLucy(np.asarray(imag), np.asarray(psfcenter), iterations=iterations, verbose=False)

# export the results
exportName = 'testTarget03_fft'
tiff.imsave(exportName + '_original.tif', np.float32(imag/imag.mean()))
tiff.imsave(exportName + '_RL_isoplanatic_50iter.tif', np.float32(imag_RL/imag_RL.mean()))
tiff.imsave(exportName + '_RL_anisoplanatic_50iter.tif', np.float32(imag_rebuild/imag_rebuild.mean()))


# %% SECOND METHOD - DIRECT DECONVOLUTION - slow
iterations = 50

# anisoplanatic
imag_rebuild = pya.varyingDeconvRL(imag, psfmap, iterations, windowSize, windowStep, windowOverlap)

# isoplanatic, central kernel
psfcenter = psfmap[2,2,:,:]
imag_RL, error_RL = pyd.richardsonLucy_smallKernel(np.asarray(imag), np.asarray(psfcenter), iterations=iterations, verbose=False)
imag_RL, error_RL = imag_RL.get(), error_RL.get()


# export the results
exportName = 'testTarget03_direct'
tiff.imsave(exportName + '_original.tif', np.float32(imag/imag.mean()))
tiff.imsave(exportName + '_RL_isoplanatic_50iter.tif', np.float32(imag_RL/imag_RL.mean()))
tiff.imsave(exportName + '_RL_anisoplanatic_50iter.tif', np.float32(imag_rebuild/imag_rebuild.mean()))


