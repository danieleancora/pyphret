# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:02:13 2020

@author: Daniele Ancora
"""

# LIBRARIES CALL
import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


"""
Definition of useful functions that are implemented in the PR and 
deautocorrelation protocols. Both of them works on CPU and GPU and are optimized
for using real FFT protocols. 
"""

# %% alignment routines
def my_findshift3d(function1, function2):
    xcorr0 = np.correlate( np.mean(function1, axis=(1,2)), np.mean(function2, axis=(1,2)) , "full")
    xcorr1 = np.correlate( np.mean(function1, axis=(0,2)), np.mean(function2, axis=(0,2)) , "full")
    xcorr2 = np.correlate( np.mean(function1, axis=(0,1)), np.mean(function2, axis=(0,1)) , "full")
        
    shift = (- function1.shape[0] + 1 + np.argmax(xcorr0), 
             - function1.shape[1] + 1 + np.argmax(xcorr1), 
             - function1.shape[2] + 1 + np.argmax(xcorr2))
    return shift

def my_allign3d(function1, function2):    
    shift = my_findshift3d(function1, function2)
    print(shift)
    return np.roll(function2, shift, axis=(0,1,2))

def my_allign3d_flip(function1, function2):    
    shift1 = my_findshift3d(function1, function2)
    shift2 = my_findshift3d(function1, function2[::-1,::-1,::-1])
    print(shift1)
    
    noflip  = np.roll(function2, shift1, axis=(0,1,2))
    yesflip = np.roll(function2[::-1,::-1,::-1], shift2, axis=(0,1,2))
    
    product_noflip = function1 * noflip
    product_yesflip = function1 * yesflip
    
    if np.sum(product_noflip) <= np.sum(product_yesflip):
        noflip = yesflip
        print('flipped')
    
    return noflip


# %% convolution correlation routines
def my_convolution(function1, function2):
    xp = cp.get_array_module(function1)
    return xp.fft.fftshift((xp.fft.irfftn(xp.fft.rfftn(function1) * xp.fft.rfftn(function2))))

def my_correlation(function1, function2):
    xp = cp.get_array_module(function1)
    return xp.fft.fftshift((xp.fft.irfftn(xp.conj(xp.fft.rfftn(function1)) * xp.fft.rfftn(function2))))

def my_autocorrelation(x):
    return my_correlation(x, x)

def autocorrelation2fouriermod(x):
    xp = cp.get_array_module(x)
    return xp.sqrt(xp.abs(xp.fft.rfftn(x)))

def fouriermod2autocorrelation(x):
    xp = cp.get_array_module(x)
    return xp.fft.fftshift(xp.fft.irfftn(x**2))


# %% Blurring functions 
def my_gaussblur(function, sigma):
    xp = cp.get_array_module(function)
    direction = function.ndim
    normalization = (sigma*xp.sqrt(2*xp.pi))    # energy is preserved
    # loop through all dimension, the problem can be split fourier transformations
    # along each dimension of the function
    for i in range(direction):
        size = function.shape[i]
        x = xp.arange(0, size, dtype=xp.float32)
        gaussian_1d = xp.fft.fft( xp.exp(-(x-size/2)**2.0 / (2*sigma**2))/normalization )    
        reshape = np.ones_like(function.shape)
        reshape[i] = gaussian_1d.shape[0]
        temp = gaussian_1d.reshape(reshape) 
        function = xp.fft.ifft(xp.fft.fft(function, axis=i) * temp, axis=i)
    return xp.fft.fftshift(xp.real(function))    

# this is useful for OSS, alpha has opposite role than sigma
def my_gaussBlurInv(function, alpha):
    xp = cp.get_array_module(function)
    direction = function.ndim
    normalization = 1.     # energy is preserved
    # loop through all dimension, the problem can be split fourier transformations
    # along each dimension of the function
    for i in range(direction):
        size = function.shape[i]
        x = xp.arange(0, size, dtype=xp.float32)
        gaussian_1d = xp.fft.fftshift(xp.exp(-0.5*((x-size/2)**2.0) / (alpha**2))/normalization)    
        reshape = np.ones_like(function.shape)
        reshape[i] = gaussian_1d.shape[0]
        temp = gaussian_1d.reshape(reshape) 
        function = xp.fft.ifft(xp.fft.fft(function, axis=i) * temp, axis=i)
    return xp.fft.fftshift(xp.real(function))    

def gaussian_psf(size=[200,200], alpha=[10,20]):
    x = np.arange(0, size[0], dtype=np.float32)
    gaussian_1d = ( np.exp(-(x - size[0] / 2)**2.0 / (2 * alpha[0]**2)) ) 
    dim = len(alpha)
    # we use outer products to compute the psf accordingly to given dimensions
    if dim == 1:
        psf = gaussian_1d        
    if dim == 2:
        x = np.arange(0, size[1], dtype=np.float32)
        gaussian_2d = ( np.exp(-(x - size[1] / 2)**2.0 / (2 * alpha[1]**2)) ) 
        psf =  gaussian_1d.reshape(gaussian_1d.shape[0],1)*gaussian_2d        
    elif dim == 3:
        x = np.arange(0, size[1], dtype=np.float32)
        gaussian_2d = ( np.exp(-(x - size[1] / 2)**2.0 / (2 * alpha[1]**2)) ) 
        x = np.arange(0, size[2], dtype=np.float32)
        gaussian_3d = ( np.exp(-(x - size[2] / 2)**2.0 / (2 * alpha[2]**2)) ) 
        psf = (gaussian_1d.reshape(gaussian_1d.shape[0],1)*gaussian_2d).reshape(gaussian_1d.shape[0],gaussian_2d.shape[0],1)*gaussian_3d

    return psf    


# %% masking functions, first algorithm taken from 
# https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
def circular_mask2D(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (dist_from_center <= radius)
    return mask

def spherical_mask3D(h, w, d, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2), int(d/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], center[2], w-center[0], h-center[1], d-center[2])

    Y, X, Z = np.ogrid[:h, :w, :d]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)

    mask = (dist_from_center <= radius)
    return mask

def threshold_maskND(function, percent=0):
    # default value is positive thresholding
    xp = cp.get_array_module(function)
    value = xp.max(function) * percent
    mask = (function > value)

    return mask

def autocorrelation_maskND(function, percent=0):
    # default value is positive thresholding
    function = my_autocorrelation(function)
    mask = threshold_maskND(function, percent)
    
    return mask

def sparsity_maskND(function, percent=0):
    # default value is positive thresholding
    xp = cp.get_array_module(function)
    k = int(np.size(function) * percent) 
    print(k)
    linear = np.reshape(function, -1)     
   
    # find k max values within function
    idx = np.argpartition(linear, -k)[-k:]
    
    # masking out all the min-k values 
    mask = (function > np.min(linear[idx]))
    
    return mask








