# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:57:07 2021

@author: Daniele Ancora
"""

# %%
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from skimage.util import view_as_windows

from pyphret.torch.functional import xcorrDepthwise, convDepthwise


def tile(imag, windowSize, windowStep, windowOverlap):
    expanded_imag = view_as_windows(imag, (windowSize, windowSize), step=(windowStep, windowStep)).copy()
    return expanded_imag


def untile(expanded_deconv, windowSize, windowStep, windowOverlap):
    # calculate the final shape of the image
    iiisize = (expanded_deconv.shape[0]-1)*windowStep + windowSize
    jjjsize = (expanded_deconv.shape[1]-1)*windowStep + windowSize

    # print(iiisize), print(jjjsize)
    
    # preallocate memory
    imag_rebuild = np.zeros((iiisize, jjjsize))
    imag_map     = np.zeros((iiisize, jjjsize))
    imag_canvas  = np.zeros((iiisize, jjjsize))
    imag_mapcan  = np.zeros((iiisize, jjjsize))
    
    mask = np.ones((windowSize,windowSize))
    
    # recompose the image by tiling
    for i in range(expanded_deconv.shape[0]):
        for j in range(expanded_deconv.shape[1]):
            imag_chunk = expanded_deconv[i,j,:,:]

            # difficult to read but important, places back the deconvolved tiles in the appropriate mask
            imag_rebuild[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += imag_chunk[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]
            imag_map[i*windowStep+windowOverlap:i*windowStep+windowSize-windowOverlap, j*windowStep+windowOverlap:j*windowStep+windowSize-windowOverlap] += mask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap]

            # this is for the canvas region windowOverlap thick
            imag_canvas[i*windowStep:i*windowStep+windowSize, j*windowStep:j*windowStep+windowSize] += imag_chunk
            imag_mapcan[i*windowStep:i*windowStep+windowSize, j*windowStep:j*windowStep+windowSize] += mask

    # outer mask
    outermask = np.ones((iiisize, jjjsize))
    outermask[windowOverlap:-windowOverlap,windowOverlap:-windowOverlap] = 0

    # renormalize the rebuilt image by appropriate use of overlapping mask
    imag_rebuild = (imag_rebuild + outermask*imag_canvas)/(imag_map + outermask*imag_mapcan)
    imag_rebuild[np.isnan(imag_rebuild)] = 0

    return imag_rebuild


