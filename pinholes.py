# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:10:25 2020

@author: Daniele Ancora
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def circular_mask2D(h, w, center=None, radius=None):
    """
    Simple function to generate a single pinhole. The function is called within random_points_mask    
    """    
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (dist_from_center < radius)
    return mask


def random_points_minDistance_slow(n, shape_min, shape_max, min_dist):
    """
    Function that generates a binary mask composed by randomly distributed pinholes within a circular
    region having radiusmask.
    
    for quick test, execute:
    plt.scatter(random_points_mask())
    
    """    
    coords = np.zeros((n,2))
    i = 0
    wrong_attempts = 0
    while i<n:
        x = np.random.uniform(low=shape_min[0], high=shape_max[0])
        y = np.random.uniform(low=shape_min[1], high=shape_max[1])
        tooclose = (np.sqrt((coords[:,0] - x) ** 2 + (coords[:,1]- y)**2) < min_dist)
        if np.any(tooclose):
           # print('change!')
           wrong_attempts += 1
        else:
            coords[i,0] = x
            coords[i,1] = y
            i+=1           
            # print('OK!')
    print('Wrong attempts: ' + str(wrong_attempts))         
    return coords


def random_points_mask(dimension = np.array([1024,1024]), radiuspinhole = 8, n = 2000, radiusmask = 256):
    """
    Function that generates a binary mask composed by randomly distributed pinholes within a circular
    region having radiusmask.
    
    for quick test, execute:
    plt.imshow(random_points_mask())


    Parameters
    ----------
    dimension : numpy array, it contains the dimensions in pixel of the whole mask
        DESCRIPTION. The default is np.array([1024,1024]), thus produces an image of 1024x1024 px.
    radiuspinhole : integer, radius of each single pinhole 
        DESCRIPTION. The default is 8.
    n : integer, number of pinholes that populates the mask
        DESCRIPTION. The default is 1000.
    radiusmask : integer, the radius of the (centered) circular region to populate with pinholes
        DESCRIPTION. The default is 256.

    Returns
    -------
    randompinhole : boolean matrix
        DESCRIPTION. This is the output mask.

    """
    mask = np.zeros(dimension)    
    center = (dimension/2)
    
    rangeX = np.rint([center[0]-radiusmask, center[0]+radiusmask]).astype(int)
    rangeY = np.rint([center[1]-radiusmask, center[1]+radiusmask]).astype(int)
    
    coords = np.zeros((n,2))
    
    i=0
    wrong_attempts = 0
    
    while i<n:
        x = np.random.randint(low=rangeX[0], high=rangeX[1])
        y = np.random.randint(low=rangeY[0], high=rangeY[1])
        
        tooclose = (np.sqrt((coords[:,0]-x)**2 + (coords[:,1]-y)**2) < radiuspinhole+4)
        distfromcenter = np.sqrt((center[0]-x)**2 + (center[1]-y)**2)
        
        if np.any(tooclose) or distfromcenter>radiusmask:
            wrong_attempts += 1
        else:
            coords[i,0] = x
            coords[i,1] = y        
            mask[x,y] = 1
            i+=1
        if wrong_attempts > 10**6:
            print('Function killed due to too high number of attempts. Consider reducing the pinhole density!')
            break
    
    pinhole = circular_mask2D((radiuspinhole)*2-1, (radiuspinhole)*2-1)
    randompinhole = (convolve2d(mask, pinhole, mode='same') > 0)

    print('Wrong attempts: ' + str(wrong_attempts))         
    return randompinhole

plt.imshow(random_points_mask())
