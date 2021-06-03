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


# %% CREATE TILED VIEWS WITHIN A 2D IMAGE AND INVERSE
def tile(image, windowSize, windowStep, windowOverlap):
    """
    This function starts from a 2D image and select tile-regions that are 
    mapped on a 2D grid, forming a 4D array. The first two dimensions map each 
    window on a spatial grid, the last two dimensions are the spatial 
    directions of each tile.

    Parameters
    ----------
    image : 2D numpy array
        original image from which we want to extract tiled views.
    windowSize : int
        dimension of each squared tile.
    windowStep : int
        distance between one tile and the next one.
    windowOverlap : int
        not used here, eventually useful.

    Returns
    -------
    expanded_imag : 4D numpy array
        tiled image.

    """
    expanded_image = view_as_windows(image, (windowSize, windowSize), step=(windowStep, windowStep)).copy()
    return expanded_image


def untile(expanded_image, windowSize, windowStep, windowOverlap):
    """
    Inverse tile function, thus from a tiled view the functions reconstruct a
    2D image. For the moment, overlapping regions are simply averaged keeping 
    track of a contribution mask.

    Parameters
    ----------
    expanded_imag : 4D numpy array
        tiled image.
    windowSize : int
        dimension of each squared tile.
    windowStep : int
        distance between one tile and the next one.
    windowOverlap : int
        not used here, eventually useful.

    Returns
    -------
    imag_rebuild : 2D numpy array
        original image from which we want to extract tiled views.

    """
    # calculate the final shape of the image
    iiisize = (expanded_image.shape[0]-1)*windowStep + windowSize
    jjjsize = (expanded_image.shape[1]-1)*windowStep + windowSize

    # print(iiisize), print(jjjsize)
    
    # preallocate memory
    imag_rebuild = np.zeros((iiisize, jjjsize))
    imag_map     = np.zeros((iiisize, jjjsize))
    imag_canvas  = np.zeros((iiisize, jjjsize))
    imag_mapcan  = np.zeros((iiisize, jjjsize))
    
    mask = np.ones((windowSize,windowSize))
    
    # recompose the image by tiling
    for i in range(expanded_image.shape[0]):
        for j in range(expanded_image.shape[1]):
            imag_chunk = expanded_image[i,j,:,:]

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



def adddimension(image):
    return np.expand_dims(image, axis=0)


def removedimension(image):
    return np.squeeze(image)

