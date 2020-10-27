# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:51:51 2020

@author: Daniele Ancora
"""

import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pyphret.backend as pyb
import pyphret.functions as pf



def rotateStack(origStack, angleInitial=0, angleStep=90, rotateaxes=(0,1)):
    # pop out a warning
    print('Rotating the stack...')
    print('This functions assumes the axis=0 as the one storing different views')
    
    # create the 
    rotatedStack = np.zeros_like(origStack)
    
    # perform the rotation
    for i in range(origStack.shape[0]):
        if angleStep == 0:
            rotatedStack[i,:,:,:] = origStack[i,:,:,:]
        elif angleStep != 90:
            rotatedStack[i,:,:,:] = ndimage.rotate(origStack[i,:,:,:],   -angleStep*i, axes=rotateaxes, reshape=False)    
        else:
            rotatedStack[i,:,:,:] = np.rot90(origStack[i,:,:,:], -i, axes=rotateaxes)
        
        print('Rotating view ' + str(i) + ' - deg = ' + str(-angleStep*i))

    return rotatedStack


def alignStack(origStack, reference=0):
    # pop out a warning
    print('Aligning the stack against view ' + str(reference) + '...')
    print('This functions assumes the axis=0 as the one storing different views')

    # create the stack
    alignedStack = np.zeros_like(origStack)
    
    for i in range(origStack.shape[0]):
        if i != reference:
            alignedStack[i,:,:,:] = pf.my_alignND(origStack[reference,:,:,:], origStack[i,:,:,:])
        else:
            alignedStack[i,:,:,:] = origStack[i,:,:,:]
        print('Aligning view ' + str(i) + ' agains reference = ' + str(reference))

    return alignedStack
   
    
   
def autocorrelateStack(origStack):
    # pop out a warning
    print('Calculating the stack auto-correlation...')

    # create the stack
    spimAcorr = np.zeros_like(origStack)

    # calculating autocorrelations from spim detections
    for i in range(origStack.shape[0]):
        spimAcorr[i,:,:,:] = np.abs(pf.my_autocorrelation(origStack[i,:,:,:]))
        print('Auto-correlating view ' + str(i))

    return spimAcorr


def crosscorrelateStack(origStack):
    # pop out a warning
    print('Calculating whole stack cross-correlation...')

    # create the stack
    spimXcorr = np.zeros((origStack.shape[0], origStack.shape[0], origStack.shape[1], origStack.shape[2], origStack.shape[3]))

    # calculating autocorrelations from spim detections
    for i in range(origStack.shape[0]):
        for j in range(origStack.shape[0]):
            spimXcorr[i,j,:,:,:] = np.abs(pf.my_correlationCentered(origStack[i,:,:,:],origStack[j,:,:,:]))
            print('Cross correlating view ' + str(i) + ' with ' + str(j))
        
    return spimXcorr
    


# i'm not sure this thing is working as it should
def focusStack(origStack, sigma=2, axis=0):
    
    laplaceStack = np.zeros_like(origStack)
    
    minvalue = origStack.min()
    origStack = origStack - minvalue
    
    # calculating autocorrelations from spim detections
    for i in range(origStack.shape[0]):
        print('Calculating Laplacian of view ' + str(i))
        # laplaceStack[i,:,:,:] = np.abs(ndimage.laplace(origStack[i,:,:,:]))
        laplaceStack[i,:,:,:] = np.abs(ndimage.gaussian_filter(ndimage.laplace(origStack[i,:,:,:]), sigma))      
        # laplaceStack[i,:,:,:] = np.abs(ndimage.gaussian_laplace(origStack[i,:,:,:], sigma=2))
        
    
    index_array = np.argmax(laplaceStack, axis=axis)

    # Same as np.max(x, axis=-1, keepdims=True)
    focussedStack = np.take_along_axis(origStack, np.expand_dims(index_array, axis=axis), axis=axis)
     
    return focussedStack+minvalue
    