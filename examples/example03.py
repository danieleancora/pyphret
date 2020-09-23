# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:54:14 2020

@author: Daniele Ancora
"""

import os
import cupy as cp
import numpy as np
from skimage import io
from natsort import natsorted
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import pyphret.functions as pf
import pyphret.retrievals as pr


# parameters of the dataset
voxelSize = 0.6500
voxelDepth = 3.2500
voxelRatio = voxelDepth / voxelSize 
threshold = 0
timepoint = 0


# load stack images and choose only one time point
im_superstack = io.imread('D://Moritz//5views//STACK_65frames_5views_17timeframes.tif')
im = im_superstack[timepoint,:,:,:,:]
    
    
# I reduce the dimension down to the nearest power of 2
result = ndimage.zoom(im, [1, 1, 0.5*1, 0.5*1], order=1)
result = ndimage.zoom(result, [0.5*voxelRatio, 1, 1, 1], order=1)
result = result[:,:,96:608,0:256]


# transpose to reorder the image stack
x = np.transpose(result, (1, 2, 3, 0))
    
    
# now I start processing x, first threshold, then pad
threshold_indices = x < threshold
x[threshold_indices] = threshold
x = x - threshold
x_pad = np.pad(x, ((0,0),(0,0),(0,0),(0,256-x.shape[3])), 'constant')
    
    
# I rotate all the projections so that we can overlap autocorrelations
x_pad[0,:,:,:] = np.rot90(x_pad[0,:,:,:], 0, axes=(1,2))
x_pad[1,:,:,:] = np.rot90(x_pad[1,:,:,:], -1, axes=(1,2))
x_pad[2,:,:,:] = np.rot90(x_pad[2,:,:,:], -2, axes=(1,2))
x_pad[3,:,:,:] = np.rot90(x_pad[3,:,:,:], -3, axes=(1,2))
x_pad[4,:,:,:] = np.rot90(x_pad[4,:,:,:], -4, axes=(1,2))
        
    
# casting to float32 to calculate correlations
x_pad = np.float32(x_pad)
correlation0 = pf.my_correlation(x_pad[0,:,:,:],x_pad[0,:,:,:])
correlation1 = pf.my_correlation(x_pad[1,:,:,:],x_pad[1,:,:,:])
correlation2 = pf.my_correlation(x_pad[2,:,:,:],x_pad[2,:,:,:])
correlation3 = pf.my_correlation(x_pad[3,:,:,:],x_pad[3,:,:,:])
correlation4 = pf.my_correlation(x_pad[4,:,:,:],x_pad[4,:,:,:])


# KEY POINT, correlation averaging
correlation_average = (correlation0 + correlation1 + correlation2 + correlation3)/4
# correlation_average = (correlation0 * correlation1 * correlation2 * correlation3) ** 1/4
# plt.plot([correlation0[256,128,128], correlation1[256,128,128] , correlation4[256,128,128] , correlation3[256,128,128]])

correlation_maximum = np.max([correlation0, correlation1, correlation2, correlation3], axis=(0))

# modulus calculatbased on the autocorrelation average
modulus = pf.autocorrelation2fouriermod(correlation_average)    


for i in range(0,5):
    
    print('processing time-stack number 0, prior from view number', i)
    prior = x_pad[i,:,:,:]
    
    # move stuff to GPU
    x_gpu = cp.asarray(modulus)
    prior = cp.asarray(prior)
    xp = cp.get_array_module(x_gpu)
    xp.random.seed()
    
    # g_k_prime = pr.phaseRet(x_gpu, rec_prior=prior, phase_prior=None, masked='full',
    #                     method='HIO', mode='normal',
    #                     beta=0.9, steps=2000)
    g_k_prime = pr.phaseRet(x_gpu, rec_prior=prior, phase_prior=None, masked='full',
                        method='ER', mode='normal',
                        beta=0.9, steps=2000)
    
    # retrieve results from GPU memory
    g_k_prime = g_k_prime.get()
    prior = prior.get()
    # mask = mask.get()
    
    # export results as tiff stacks
    export = g_k_prime
    threshold_indices = export < threshold
    export[threshold_indices] = threshold
    export = export - threshold
    
    string = "_view" + str(i) + ".tif"

    io.imsave("reconstruction"+string, np.float32(export))
    io.imsave("prior.tif"+string, prior)
    # io.imsave("mask.tif"+string, np.uint16(mask))
    