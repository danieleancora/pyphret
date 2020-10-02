# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:04:08 2020

@author: Daniele Ancora
"""

# LIBRARIES CALL
import time
import cupy as cp
import numpy as np
import pyphret.functions as pf


def randomGaussianBlobs_2D(dimension=512, fraction=0.999):
    size = int(dimension/2)
    size_pad = int(dimension/4)

    test_image = np.random.rand(size,size)
    test_image = (test_image > fraction).astype(np.float32) * pf.circular_mask2D(size, size)
    test_image = np.pad(test_image, (size_pad,size_pad), 'constant')
    test_image = pf.my_gaussblur(test_image,4)

    return test_image


# def randomGaussianBlobs_3D(dimension=512, fraction=0.999, center=None, radius=None):
#     size = int(dimension/2)
#     size_pad = int(dimension/4)

#     test_image = np.random.rand(size,size,size)
#     test_image = (test_image > fraction).astype(np.float32) * pf.spherical_mask3D(size, size, size, radius=radius)
#     test_image = np.pad(test_image, (size_pad,size_pad), 'constant')
#     test_image = pf.my_gaussblur(test_image,4)

#     return test_image


def randomGaussianBlobs_3D(dimension=512, fraction=0.999, center=None, radius=None):
    test_image = np.random.rand(dimension,dimension,dimension)
    test_image = (test_image > fraction).astype(np.float32) * pf.spherical_mask3D(dimension, dimension, dimension, radius=radius)
    test_image = pf.my_gaussblur(test_image,4)

    return test_image



def randomAmorphousVases_3D(intdimension=512, extdimension=1024, volumeradius=2*(64+16), maskradius1=13, maskradius2=11):
    # definition of 2 concentric masks, used for amorphous generation
    mask1 = pf.spherical_mask3D(extdimension, extdimension, extdimension, center=None, radius=maskradius1)
    mask2 = pf.spherical_mask3D(extdimension, extdimension, extdimension, center=None, radius=maskradius2)
    mask = (np.float32(mask1)-np.float32(mask2))

    # this is the mask that select the size of the phantom
    mask3 = pf.spherical_mask3D(intdimension,intdimension,intdimension, center=None, radius=volumeradius)
 
    # random phase and amplitude to associate to the mask
    phase = 2*np.pi*np.random.rand(extdimension,extdimension,extdimension)
    ampli = 1 - 0.01*np.random.rand(extdimension,extdimension,extdimension)
    
    # generation of the volumetric amorphous pattern, this is symmetric
    randomrods = np.abs(np.fft.fftn( ampli * np.exp(phase * mask)) )
    
    # select only the portion up to intdimension within mask3
    hyp = (randomrods[:intdimension,:intdimension,:intdimension]*mask3)
    hyp /= hyp.max()
    hyp_dots = hyp.copy()
    hyp = 1 - hyp
    hyp[hyp==1] = 0
    
    # creating the outer veins
    threshold_veins = 0.95
    sample = np.float32(hyp>threshold_veins)
    sample = sample*hyp
    sample = np.float32(sample) - threshold_veins
    sample[sample<0] = 0
    sample /= sample.max()
    
    # creating the inner veins
    threshold_veins = 0.93
    sample_veins = np.float32(hyp>threshold_veins)
    sample_veins = sample_veins*hyp
    sample_veins = np.float32(sample_veins) - threshold_veins
    sample_veins[sample_veins<0] = 0
    sample_veins /= sample_veins.max()
    
    # creating the vases by subtracting the veins
    sample_vase = np.abs(sample_veins - sample)
    
         
    # smooth and sharpen before returning
    sample_vase = pf.axisflip(hyp**2) * pf.my_gaussblur(sample_vase**2, 0.5)   
    sample_vase = np.abs(sample_vase) 
    sample_vase /= sample_vase.max()
    
    

    return sample_vase



def randomAmorphousVases_3Dspeckled(intdimension=512, extdimension=1024, volumeradius=2*(64+16), maskradius1=13, maskradius2=11):
    # definition of 2 concentric masks, used for amorphous generation
    mask1 = pf.spherical_mask3D(extdimension, extdimension, extdimension, center=None, radius=maskradius1)
    mask2 = pf.spherical_mask3D(extdimension, extdimension, extdimension, center=None, radius=maskradius2)
    mask = (np.float32(mask1)-np.float32(mask2))

    # this is the mask that select the size of the phantom
    mask3 = pf.spherical_mask3D(intdimension,intdimension,intdimension, center=None, radius=volumeradius)
 
    # random phase and amplitude to associate to the mask
    phase = 2*np.pi*np.random.rand(extdimension,extdimension,extdimension)
    ampli = 1 - 0.01*np.random.rand(extdimension,extdimension,extdimension)
    
    # generation of the volumetric amorphous pattern, this is symmetric
    randomrods = np.abs(np.fft.fftn( ampli * np.exp(phase * mask)) )
        
    # select only the portion up to intdimension within mask3
    hyp = (randomrods[:intdimension,:intdimension,:intdimension]*mask3)
    hyp /= hyp.max()
    hyp_dots = hyp.copy()
    hyp = 1 - hyp
    hyp[hyp==1] = 0
    
    # creating the outer veins
    threshold_veins = 0.95
    sample = np.float32(hyp>threshold_veins)
    sample = sample*hyp
    sample = np.float32(sample) - threshold_veins
    sample[sample<0] = 0
    sample /= sample.max()
    
    # creating the inner veins
    threshold_veins = 0.93
    sample_veins = np.float32(hyp>threshold_veins)
    sample_veins = sample_veins*hyp
    sample_veins = np.float32(sample_veins) - threshold_veins
    sample_veins[sample_veins<0] = 0
    sample_veins /= sample_veins.max()
    
    # creating the vases by subtracting the veins
    sample_vase = np.abs(sample_veins - sample)
    
         
    # smooth and sharpen before returning
    sample_vase = pf.axisflip(hyp**2) * pf.my_gaussblur(sample_vase**2, 0.5)   
    sample_vase = np.abs(sample_vase) 
    
    randomrods = np.abs(np.fft.fftn( ampli * np.exp(phase * mask2)) )
    randomrods = randomrods[:intdimension,:intdimension,:intdimension]

    sample_vase = sample_vase * randomrods
    
    sample_vase /= sample_vase.max()
    
    

    return sample_vase



