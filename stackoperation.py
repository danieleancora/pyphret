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
    print('This functions assumes the axis=0 as the one storing different views')
    
    # create the 
    rotatedStack = np.zeros_like(origStack)
    
    # perform the rotation
    for i in range(origStack.shape[0]):
        if angleStep != 90:
            rotatedStack[i,:,:,:] = ndimage.rotate(origStack[i,:,:,:],   -angleStep*i, axes=rotateaxes, reshape=False)    
        else:
            rotatedStack[i,:,:,:] = np.rot90(origStack[i,:,:,:], -i, axes=rotateaxes)
        print(i)

    return rotatedStack


def alignStack(origStack, reference=0):

    # pop out a warning
    print('This functions assumes the axis=0 as the one storing different views')

    # create the stack
    alignedStack = np.zeros_like(origStack)
    
    for i in range(origStack.shape[0]):
        if i != reference:
            alignedStack[i,:,:,:] = pf.my_alignND(origStack[reference,:,:,:], origStack[i,:,:,:])
            print(i)
        else:
            alignedStack[i,:,:,:] = origStack[i,:,:,:]

    return alignedStack
   
    
   
def autocorrelateStack(origStack):

    # create the stack
    spimXcorr = np.zeros_like(origStack)

    # calculating autocorrelations from spim detections
    for i in range(origStack.shape[0]):
        spimXcorr[i,:,:,:] = np.abs(pf.my_autocorrelation(origStack[i,:,:,:]))
        print(i)
        
    return spimXcorr

    