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
def varyingDeconvRL(imag, psfmap, iterations, windowSize, windowStep, windowOverlap, gpu=False):
    """
    Function that aims at deconvolving different regions of the image subject 
    to a varying (anisoplanatic) point-spread function. Deconvolution is 
    accomplished in separate windows that are tiled back accordingly to a 
    masking criterion.

    Parameters
    ----------
    imag : bi-dimensional float 
        the image the we want to deconvolve.
    psfmap : four-dimensional float 
        psf that we use for deconvolution. the first two dimensions are used 
        for mapping the psf in space, the third and fourth are spatial 
        dimension of each psf.
    iterations : integer
        number of iterations to run.
    windowSize : integer, choose with care.
        size of each tile in which we carry out the deconvolution.
    windowStep : integer, choose with care.
        slide the tile by this amount in both spatial directions.
    windowOverlap : integer.
        it is used to discard a portion around each tile to avoid the inclusion
        of boundary artifacts. A value around 32 should fit any problem. 
    gpu : boolean
        choose wether to use or not GPU (cupy) computation

    Returns
    -------
    imag_rebuild : float
        the image deconvolved, tiled and normalized according to the masking 
        criterion.

    """
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
            
            if gpu == True:
                imag_chunk, error = pyd.richardsonLucy_smallKernel(cp.asarray(imag_chunk), cp.asarray(psf_chunk), iterations=iterations, verbose=False)
                imag_chunk = imag_chunk.get()
            else:
               imag_chunk, error = pyd.richardsonLucy_smallKernel(np.asarray(imag_chunk), np.asarray(psf_chunk), iterations=iterations, verbose=False)
            
            # difficult to read but important, places back the deconvolved tiles in the appropriate mask
            imag_rebuild[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += imag_chunk[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]
            imag_map[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += mask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]
        
    # renormalize the rebuilt image by the overlapping mask
    imag_rebuild = imag_rebuild/imag_map
    imag_rebuild[np.isnan(imag_rebuild)] = 0
    
    return imag_rebuild


# A LOT FASTER, BUT I AM NOT YET CONVINCED BY THE RESULT
def varyingDeconvRL_fft(imag, psfmap, iterations, windowSize, windowStep, windowOverlap, gpu=False):
    """
    Function that aims at deconvolving different regions of the image subject 
    to a varying (anisoplanatic) point-spread function. Deconvolution is 
    accomplished in separate windows that are tiled back accordingly to a 
    masking criterion. This implementation pads the psf to the same size of the
    tiled windonw in order to perform fft convolutions.

    Parameters
    ----------
    imag : bi-dimensional float 
        the image the we want to deconvolve.
    psfmap : four-dimensional float 
        psf that we use for deconvolution. the first two dimensions are used 
        for mapping the psf in space, the third and fourth are spatial 
        dimension of each psf.
    iterations : integer
        number of iterations to run.
    windowSize : integer, choose with care.
        size of each tile in which we carry out the deconvolution.
    windowStep : integer, choose with care.
        slide the tile by this amount in both spatial directions.
    windowOverlap : integer.
        it is used to discard a portion around each tile to avoid the inclusion
        of boundary artifacts. A value around 32 should fit any problem. 
    gpu : boolean
        choose wether to use or not GPU (cupy) computation

    Returns
    -------
    imag_rebuild : float
        the image deconvolved, tiled and normalized according to the masking 
        criterion.

    """

    # split image in overlapping windows
    expanded_imag = view_as_windows(imag, (windowSize, windowSize), step=(windowStep, windowStep)).copy()
    print(expanded_imag.shape)

    # interpolate the number of psf mapped in the image (do not exagerate with this)
    padlength = int((windowSize - psfmap.shape[2])/2)
    expanded_psfmap = ndimage.zoom(psfmap, (expanded_imag.shape[0]/psfmap.shape[0],expanded_imag.shape[0]/psfmap.shape[0],1,1))
    expanded_psfmap = np.pad(expanded_psfmap, ((0,0),(0,0),(padlength,padlength),(padlength,padlength)))

    # expanded_psfmap = np.abs(expanded_psfmap)

    # decovolution applied only on some directions
    if gpu == True:
        expanded_deconv, error = pyd.richardsonLucy_alongaxes(cp.asarray(expanded_imag), cp.asarray(expanded_psfmap), axes=(2,3),
                                                              iterations=iterations, verbose=True)
        expanded_deconv, error = expanded_deconv.get(), error.get()
    else:
        expanded_deconv, error = pyd.richardsonLucy_alongaxes(np.asarray(expanded_imag), np.asarray(expanded_psfmap), axes=(2,3),
                                                              iterations=iterations, verbose=True)        
    
    # recompose the image by tiling
    imag_rebuild = np.zeros_like(imag)
    imag_map = np.zeros_like(imag)
    mask = np.ones((windowSize,windowSize))
        
    for i in range(expanded_imag.shape[0]):
        for j in range(expanded_imag.shape[1]):
            imag_chunk = expanded_deconv[i,j,:,:]

            # difficult to read but important, places back the deconvolved tiles in the appropriate mask
            imag_rebuild[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += imag_chunk[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]
            imag_map[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += mask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]

    # renormalize the rebuilt image by the overlapping mask
    imag_rebuild = imag_rebuild/imag_map
    imag_rebuild[np.isnan(imag_rebuild)] = 0

    return imag_rebuild, error





