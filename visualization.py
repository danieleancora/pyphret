# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:55:18 2020

The functions contained in this module help visualization of data stack

@author: Daniele Ancora
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


def colorDepth(stack, axis=-1, cmap='viridis', mode='max', clipmax=1., plot=True):
    """
    This function is designed to reproduce the Temporal Color-Code hyperstack processing in FIJI
    but with more functionalities other than simple max projection.

    Example:
    max_RGB = colorDepth(stack[220:820,86:286], axis=1, cmap='jet', mode='max', clipmax=0.2)
    plt.imshow(max_RGB, vmax=[0.2,0.2,0.2])

    """
    
    viridis = cm.get_cmap(cmap, 256)
    newcolors = viridis(np.linspace(0, 1, stack.shape[axis]))

    # split the color channels to obtain separate fading
    newcolors_R = newcolors[:,0]
    newcolors_G = newcolors[:,1]
    newcolors_B = newcolors[:,2]
    
    if mode == 'max':
        # find index of where the maximum along each pixel-depth is
        depth_map = np.argmax(stack, axis=axis)

        # maximum value and normalize
        max_map = np.max(stack, axis=axis)
        max_map = (1/clipmax) * max_map/max_map.max()

        # this add a new axis to allow broadcasting
        max_map = max_map[:,:,np.newaxis]

    if mode == 'min':
        depth_map = np.argmin(stack, axis=axis)
        max_map = np.min(stack, axis=axis)
        max_map = (1/clipmax) * max_map/max_map.max()
        max_map = max_map[:,:,np.newaxis]
    
    if mode == 'mean':
        mean_map = np.mean(stack, axis=axis)
        depth_map = np.argmin(np.abs(stack - np.expand_dims(mean_map, axis=axis)), axis=axis)
        max_map = (1/clipmax) * mean_map/mean_map.max()
        max_map = max_map[:,:,np.newaxis]
    
    
    # this select the color on the lookup table according to the depth map
    depth_color = np.zeros(depth_map.shape + (3,))
    depth_color[:,:,0] = newcolors_R[depth_map]
    depth_color[:,:,1] = newcolors_G[depth_map]
    depth_color[:,:,2] = newcolors_B[depth_map]
       
    
    max_RGB = depth_color*max_map
    
    if plot==True: 
        plt.figure(),
        plt.subplot(121), plt.imshow(depth_color), plt.title('depth location')
        plt.subplot(122), plt.imshow(max_RGB), plt.title('value-weighted location')
        

    return max_RGB



