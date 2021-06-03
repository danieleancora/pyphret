# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:20:11 2021

@author: Daniele Ancora
"""
# %%
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage.util import view_as_windows
import matplotlib.pyplot as plt

from pyphret.torch.tiling import tile, untile
from pyphret.torch.functional import xcorrDepthwise, convDepthwise


# %% ISOPLANATIC DECONVOLUTION
def deconvolutionRL(signal, kernel, deconv=None, iterations=20, verbose=True):
    epsilon = 1e-7
    distance = []
    
    # set starting prior to the actual image
    if deconv is None: 
        deconv = signal.clone().detach()

    # iterative richardson lucy
    for iteration in range(iterations):
        if verbose==True:
            print('iteration = ' + str(iteration))
        
        # relative blur division
        relative_blur = convDepthwise(deconv, kernel)
        
        distance.append(torch.linalg.norm(relative_blur/relative_blur.mean()-deconv/deconv.mean()).cpu().numpy())
        
        relative_blur = signal / relative_blur
        
        # avoid errors due to division by zero or inf
        relative_blur[torch.isinf(relative_blur)] = epsilon
        relative_blur[torch.isnan(relative_blur)] = 0
        # relative_blur = torch.abs(relative_blur)

        # multiplicative update 
        deconv *= xcorrDepthwise(relative_blur, kernel)
    
    plt.figure()
    plt.plot(distance)

    return deconv


# NOT WORKING, DON'T KNOW WHY
def deconvolutionMAP(signal, kernel, deconv=None, iterations=20, verbose=True):
    epsilon = 1e-7
    
    # set starting prior to the actual image
    if deconv is None: 
        deconv = signal.clone().detach()

    # iterative richardson lucy
    for iteration in range(iterations):
        if verbose==True:
            print('iteration = ' + str(iteration))
        
        # relative blur division
        relative_blur = convDepthwise(deconv, kernel)    
        relative_blur = signal / relative_blur - 1 
        
        # avoid errors due to division by zero or inf
        # relative_blur[torch.isinf(relative_blur)] = 0
        # relative_blur[torch.isnan(relative_blur)] = 0
        # relative_blur = torch.abs(relative_blur)

        # multiplicative update 
        deconv *= torch.exp(xcorrDepthwise(relative_blur, kernel))
        
    return deconv


# NOT WORKING, IT IS MORE DIFFICULT THAN THIS
def deconvolutionRLblind(signal, kernel, deconv=None, iterations=(20,1,1), verbose=True):
    signal_deconv = signal.clone().detach()
    kernel_deconv = kernel.clone().detach()
    
    for iii in range(iterations[0]):
        signal_deconv = deconvolutionRL(signal_deconv, kernel_deconv, deconv=None, iterations=iterations[1], verbose=verbose)
        kernel_deconv = deconvolutionRL(kernel_deconv, signal_deconv, deconv=None, iterations=iterations[2], verbose=verbose)
   
    return signal_deconv, kernel_deconv


# %% ANISOPLANATIC DECONVOLUTION
def varyingDeconvRL(imag, psfmap, iterations, windowSize, windowStep, windowOverlap, mode='RL'):
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
    # expanded_imag = view_as_windows(imag, (windowSize, windowSize), step=(windowStep, windowStep)).copy()
    expanded_imag = tile(imag, windowSize, windowStep, windowOverlap)
    print('Dimension of the tiles: ' + str(expanded_imag.shape))
    originalshape_imag = expanded_imag.shape

    # interpolate the number of psf mapped in the image (do not exagerate with this)
    expanded_psfmap = ndimage.zoom(psfmap, (expanded_imag.shape[0]/psfmap.shape[0],expanded_imag.shape[0]/psfmap.shape[0],1,1))
    print('Dimension of the psf:   ' + str(expanded_psfmap.shape))
    originalshape_psf = expanded_psfmap.shape

    # linearize the anisoplanatic map
    expanded_imag = np.reshape(expanded_imag, (originalshape_imag[0]*originalshape_imag[1],originalshape_imag[2],originalshape_imag[3]))
    expanded_psfmap = np.reshape(expanded_psfmap, (originalshape_psf[0]*originalshape_psf[1],originalshape_psf[2],originalshape_psf[3]))

    # check if gpu is available, otherwise run it on the CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('\n Using device:', device, '\n') 

    # cast to float and send to the device (the GPU) 
    im = torch.from_numpy(expanded_imag).float().to(device = device)
    psf = torch.from_numpy(expanded_psfmap).float().to(device = device)

    # implementation of the deconvolution
    if mode=='RL':
        expanded_deconv = deconvolutionRL(im, 
                                          psf, 
                                          deconv=None, 
                                          iterations=iterations, 
                                          verbose=True).cpu().numpy()
    elif mode=='MAP':
        expanded_deconv = deconvolutionMAP(im, 
                                           psf, 
                                           deconv=None, 
                                           iterations=iterations, 
                                           verbose=True).cpu().numpy()
    
    # everything back to the correct shape
    expanded_imag = np.reshape(expanded_imag, originalshape_imag)
    expanded_deconv = np.reshape(expanded_deconv, originalshape_imag)
    expanded_psfmap = np.reshape(expanded_psfmap, originalshape_psf)

    # rebuild image by using the tiling rules
    imag_rebuild = untile(expanded_deconv, windowSize, windowStep, windowOverlap)

    # # preallocate memory
    # imag_rebuild = np.zeros_like(imag)
    # imag_map     = np.zeros_like(imag)
    # imag_canvas  = np.zeros_like(imag)
    # imag_mapcan  = np.zeros_like(imag)
    
    # mask = np.ones((windowSize,windowSize))
        
    # # recompose the image by tiling
    # for i in range(expanded_imag.shape[0]):
    #     for j in range(expanded_imag.shape[1]):
    #         imag_chunk = expanded_deconv[i,j,:,:]

    #         # difficult to read but important, places back the deconvolved tiles in the appropriate mask
    #         imag_rebuild[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += imag_chunk[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]
    #         imag_map[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += mask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]

    #         # this is for the canvas region windowOverlap thick
    #         imag_canvas[i*windowStep:i*windowStep+windowSize, j*windowStep:j*windowStep+windowSize] += imag_chunk
    #         imag_mapcan[i*windowStep:i*windowStep+windowSize, j*windowStep:j*windowStep+windowSize] += mask

    # # outer mask
    # outermask = np.ones_like(imag)
    # outermask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap] = 0

    # # renormalize the rebuilt image by appropriate use of overlapping mask
    # imag_rebuild = (imag_rebuild + outermask*imag_canvas)/(imag_map + outermask*imag_mapcan)
    # imag_rebuild[np.isnan(imag_rebuild)] = 0

    return imag_rebuild


# %% DEAUTOCORRELATION
def deautocorrelationSS(signal, deconv=None, iterations=20, verbose=True):
    epsilon = 1e-7
    
    # set starting prior to the actual image
    if deconv is None: 
        deconv = signal.clone().detach()

    # iterative richardson lucy
    for iteration in range(iterations):
        if verbose==True:
            print('iteration = ' + str(iteration))
        
        # relative blur division
        relative_blur = xcorrDepthwise(deconv, deconv)    
        relative_blur = signal / relative_blur
        
        # avoid errors due to division by zero or inf
        relative_blur[torch.isinf(relative_blur)] = epsilon
        relative_blur[torch.isnan(relative_blur)] = 0
        # relative_blur = torch.abs(relative_blur)

        # multiplicative update 
        deconv *= (xcorrDepthwise(relative_blur, deconv) + convDepthwise(relative_blur, deconv))
        
    return deconv



