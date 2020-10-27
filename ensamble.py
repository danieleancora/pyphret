# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:32:20 2020

@author: Daniele Ancora
"""

import time
import cupy as cp
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
# import package_hiphret.PR_development_V4_forReal as pr
import pyphret.retrievals as pr


def speckle_autocorrelation(pattern, sigma = None, mask = None, threshold = None,
                           windowSize = 512, windowStep = 128, correction = None,
                           windowing = None, alpha = [0.5, 0.5]):
    # timing the execution
    t = time.time()
    
    if correction != None:
        envelope = pr.my_gaussblur(pattern, correction)
        pattern = pattern / envelope
    
    """
    Explanation of the code: the pattern is taken and divided into sliding windows
    (or rolling windows) of given size and step. This creates a 4D vector that has
    on the axis (2,3) the images of each window mapped indexed by (0,1) coordinate.
    The result is padded along the (2,3) axis and then calculate the autocorrelation
    per each window by using the fft wiener-kinchin theorem. 
    Xcorr = ifft ( conj(fft(signal))*fft(signal) ) 
    """        

    # sliding windows implemented
    expanded_input = view_as_windows(pattern - np.mean(pattern), (windowSize, windowSize), step=(windowStep, windowStep))
    expanded_input_pad = np.pad(expanded_input , [(0, 0), (0, 0), (0, windowSize), (0, windowSize)], mode='mean')

    expanded_corr = np.fft.irfft2(
            np.conj(np.fft.rfft2(expanded_input_pad, axes=(2,3))) * 
            np.fft.rfftn(expanded_input_pad, axes=(2,3)), axes=(2,3))
    expanded_fft = np.abs(np.fft.rfft2(expanded_input_pad, axes=(2,3)))
    expanded_conv = np.fft.irfft2(
            (np.fft.rfft2(expanded_input_pad, axes=(2,3))) * 
            np.fft.rfftn(expanded_input_pad, axes=(2,3)), axes=(2,3))

    enscorr = np.abs(np.fft.fftshift(np.mean(expanded_corr, axis=(0,1))))
    ensfftm = np.mean(expanded_fft, axis=(0,1))
    ensconv = np.abs(np.mean(expanded_conv, axis=(0,1)))

#    enscorr = np.abs(np.fft.fftshift(np.std(expanded_corr, axis=(0,1))))
#    ensfftm = np.std(expanded_fft, axis=(0,1))
#    ensconv = np.abs(np.std(expanded_conv, axis=(0,1)))

    enscorr = enscorr - np.min(enscorr)
    ensfftm = ensfftm - np.min(ensfftm)
    ensconv = ensconv - np.min(ensconv)

    if threshold != None:
        enscorr = enscorr - (threshold * np.max(enscorr))
        super_threshold_indices = enscorr < 0
        enscorr[super_threshold_indices] = 0
        
#        super_threshold_indices = enscorr > ((1-threshold) * np.max(enscorr))
#        enscorr[super_threshold_indices] = ((1-threshold) * np.max(enscorr))

        ensfftm = ensfftm - (threshold * np.max(ensfftm))
        super_threshold_indices = ensfftm < 0
        ensfftm[super_threshold_indices] = 0


    if windowing == "tuckey":
        windowing = np.outer(scipy.signal.tukey(2*windowSize, alpha[0]), scipy.signal.tukey(2*windowSize, alpha[1]))
        enscorr = enscorr * windowing
            
    calfftm = np.abs(np.fft.rfft2(enscorr))**0.5    
    calcorr = np.fft.fftshift(np.abs(np.fft.irfft2(ensfftm**2)))    
    calfft_ = np.abs((np.fft.rfft2(ensconv))**0.5)

    elapsed = time.time() - t
    print("elapsed time per step", elapsed)

    return (enscorr, calfftm, ensfftm, calcorr, ensconv, calfft_)




def ensambled_autocorrelation(pattern, sigma = None, mask = None, threshold = None,
                           windowSize = 512, windowStep = 128,
                           windowing = None, alpha = [0.5, 0.5], figure = False):
    """
    Explanation of the code: the pattern is taken and divided into sliding windows
    (or rolling windows) of given size and step. This creates a 4D vector that has
    on the axis (2,3) the images of each window mapped indexed by (0,1) coordinates.
    The result is padded along the (2,3) axis and then calculate the autocorrelation
    per each window by using the fft wiener-kinchin theorem. 
    Xcorr = ifft ( conj(fft(signal))*fft(signal) )
    """        
    # timing the execution
    t = time.time()  
    pattern = np.float32(pattern)
    
    # this makes a smooth boundary at the edge of the windowed speckle, if mask = 0 nothing is done
    W = np.float32(np.outer(scipy.signal.tukey(windowSize, mask), scipy.signal.tukey(windowSize, mask)))

    # sliding windows implemented
    expanded_input = view_as_windows(pattern, (windowSize, windowSize), step=(windowStep, windowStep))
    patchmean = np.mean(expanded_input, axis=(2,3))    
    expanded_input = expanded_input - patchmean[:,:, np.newaxis, np.newaxis]
        
    #expanded_input_pad = expanded_input
    expanded_input_pad = np.pad(expanded_input * W, [(0, 0), (0, 0), (0, windowSize), (0, windowSize)], mode='constant')
  
    expanded_corr = np.fft.irfft2(
                            np.conj(np.fft.rfft2(expanded_input_pad, axes=(2,3))) * 
                            np.fft.rfft2(expanded_input_pad, axes=(2,3)), axes=(2,3))

    enscorr = np.real(np.fft.fftshift(np.mean(expanded_corr, axis=(0,1))))
    # SOFI TRY:
    # enscorr = np.real(np.fft.fftshift(np.std(expanded_corr, axis=(0,1))))

    #enscorr = enscorr - np.min(enscorr)

    if threshold != None:
        enscorr = enscorr - (threshold * np.max(enscorr))
        super_threshold_indices = enscorr < 0
        enscorr[super_threshold_indices] = 0
        #super_threshold_indices = enscorr > ((1-threshold) * np.max(enscorr))
        #enscorr[super_threshold_indices] = ((1-threshold) * np.max(enscorr))

    if windowing == "tuckey":
        windowing = np.float32(np.outer(scipy.signal.tukey(enscorr.shape[0], alpha[0]), 
                                        scipy.signal.tukey(enscorr.shape[1], alpha[1])))
        enscorr = enscorr * windowing
            
    elapsed = time.time() - t; print("elapsed time per step", elapsed)
        
    if figure == True:
        plt.figure()
        plt.subplot(121), plt.imshow(pattern)
        plt.subplot(122), plt.imshow(enscorr)

    return enscorr[round(windowSize/2):round(windowSize/2+windowSize), round(windowSize/2):round(windowSize/2+windowSize)]


# smooth seems not to work
def lowthreshold(x, threshold, smooth=None):
    infer_threshold_indices = x < (threshold * np.max(x))
    if smooth != None:
        infer_threshold_indices = np.logical_not(infer_threshold_indices)
        smoothmask = pr.my_gaussblur(np.float32(infer_threshold_indices), smooth)
        smoothmask = smoothmask / np.max(smoothmask)
        x = x*smoothmask
    else:
        x[infer_threshold_indices] = 0
        
    return x

def topthreshold(x, threshold):
    super_threshold_indices = x > (threshold * np.max(x))
    x[super_threshold_indices] = (threshold * np.max(x))
    return x
