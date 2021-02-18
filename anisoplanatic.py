# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:44:00 2021

@author: Daniele Ancora
"""

# %% LIBRARIES CALL
import time
import numpy as np
from scipy import ndimage
from skimage.util import view_as_windows

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########

import pyphret.backend as pyb
import pyphret.deconvolutions as pyd


# %%

def varyingDeconvRL(imag, psfmap, iterations, windowSize, windowStep, windowOverlap):
    
    # split image in overlapping windows
    expanded_imag = view_as_windows(imag, (windowSize, windowSize), step=(windowStep, windowStep)).copy()

    # interpolate the number of psf mapped in the image (do not exagerate with this)
    expanded_psfmap = ndimage.zoom(psfmap, (expanded_imag.shape[0]/psfmap.shape[0],expanded_imag.shape[0]/psfmap.shape[0],1,1))
    
    # initilize useful varibles
    imag_rebuild = np.zeros_like(imag)
    imag_map = np.zeros_like(imag)
    mask = np.ones((windowSize,windowSize))
    
    for i in range(expanded_imag.shape[0]):
        for j in range(expanded_imag.shape[1]):
            print("Processing tile location = (" + str(i) + ',' + str(j) + ")")
            psf_chunk = expanded_psfmap[i,j,:,:]
            imag_chunk = expanded_imag[i,j,:,:]
            
            imag_chunk, error = pyd.richardsonLucy_smallKernel(cp.asarray(imag_chunk), cp.asarray(psf_chunk), 
                                                               iterations=iterations, verbose=False)
            # imag_chunk, error = pyd.richardsonLucy_smallKernel(np.asarray(imag_chunk), np.asarray(psf_chunk), 
                                                               # iterations=iterations, verbose=False)
            
            # difficult to read but important, places back the deconvolved tiles in the appropriate mask
            imag_rebuild[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += imag_chunk[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap].get()
            imag_map[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += mask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]
        
    # renormalize the rebuilt image by the overlapping mask
    imag_rebuild = imag_rebuild/imag_map
    imag_rebuild[np.isnan(imag_rebuild)] = 0
    
    return imag_rebuild


