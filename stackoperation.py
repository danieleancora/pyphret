# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 12:51:51 2020

@author: Daniele Ancora
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pyphret.backend as pyb
import pyphret.functions as pf
import pyphret.volumeoperation as pv

import scipy.ndimage

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy  as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########


# %% 
def rotatelistStack(origStack, anglelist1, anglelist2, rotateaxes1=(0,1), rotateaxes2=(1,2), reference=0, gpu=True):
    # create the new stack
    rotatedStack = np.zeros_like(origStack)

    for i in range(origStack.shape[0]):
        if (i != reference) or (anglelist1[i] != 0 and anglelist2[i] != 0):
            print('Rotating view ' + str(i) + ' with phi = ' + str(anglelist1[i]) + ' and psi = ' + str(anglelist2[i]))
            if gpu == True:
                tempStack = cp.asarray(origStack[i,:,:,:])
                rotatedStack[i,:,:,:] = pv.tiltVolume(tempStack, anglelist1[i], anglelist2[i], rotateaxes1, rotateaxes2).get()
            else:
                tempStack = origStack[i,:,:,:]
                rotatedStack[i,:,:,:] = pv.tiltVolume(tempStack, anglelist1[i], anglelist2[i], rotateaxes1, rotateaxes2)
        else:
            print('No rotation for view ' + str(i))
            rotatedStack[i,:,:,:] = origStack[i,:,:,:]

    return rotatedStack


def rotatecrosStack(origStack, anglelist1, anglelist2, rotateaxes1=(0,1), rotateaxes2=(1,2), reference=0, follow=True ,gpu=True):
    # create the new stack
    rotatedStack = np.zeros_like(origStack)
    
    referenceVolume = origStack[reference,:,:,:]
    xcorr_max = np.zeros((anglelist1.shape[0],anglelist2.shape[0]))
    
    rotateAngle1 = np.zeros(origStack.shape[0])
    rotateAngle2 = np.zeros(origStack.shape[0])
    
    psi_previous = 0
    phi_previous = 0
    
    # perform the rotation
    for i in range(origStack.shape[0]):
        
        # for performance evaluation
        start_time = time.time()

        if i != reference:
            xcorr_store = 0
                
            for phi in anglelist1:
                for psi in anglelist2:
                    print('Rotating view ' + str(i) + ' with phi = ' + str(phi) + ' and psi = ' + str(psi))
                    
                    if gpu == True:
                        tempStack = cp.asarray(origStack[i,:,:,:])
                        referenceVolume = cp.asarray(referenceVolume)
                    else:
                        tempStack = origStack[i,:,:,:]
                    
                    tempVolume = pv.tiltVolume(tempStack, phi+phi_previous, psi+psi_previous, rotateaxes1, rotateaxes2)
                    xcorr = pf.my_correlation(referenceVolume, tempVolume)
                    xcorr_max = xcorr.max()

                    if gpu == True:
                        xcorr_max = xcorr_max.get()
                    
                    if xcorr.max() > xcorr_store:
                        print('Update angle!')
                        xcorr_store = xcorr.max()
                        phi_store = phi+phi_previous
                        psi_store = psi+psi_previous
                        # volume_store = tempVolume.copy()
                    # else:
                    #     break
                        
            if follow == True:
                phi_previous = phi_store.copy()
                psi_previous = psi_store.copy()

                        
            print('Best angle found: phi = ' + str(phi_store) + '  psi = ' + str(psi_store))
            
            if gpu == True:
                tempStack = cp.asarray(origStack[i,:,:,:])
                rotatedStack[i,:,:,:] = pv.tiltVolume(tempStack, phi_store, psi_store, rotateaxes1, rotateaxes2).get()
            else:
                tempStack = origStack[i,:,:,:]
                rotatedStack[i,:,:,:] = pv.tiltVolume(tempStack, phi_store, psi_store, rotateaxes1, rotateaxes2)

            rotateAngle1[i] = phi_store
            rotateAngle2[i] = psi_store
            print("\n\n Angles checked. Performance:")
            print("--- %s seconds ----" % (time.time() - start_time))
            
        else:
            rotatedStack[i,:,:,:] = origStack[i,:,:,:].copy()
            rotateAngle1[i] = 0.
            rotateAngle2[i] = 0.

    return rotatedStack, rotateAngle1, rotateAngle2


def rotateStack(origStack, angleInitial=0, angleStep=90, rotateaxes=(0,1), gpu=True):
    # pop out a warning
    print('Rotating the stack...')
    print('This functions assumes the axis=0 as the one storing different views')
    
    # create the 
    rotatedStack = np.zeros_like(origStack)
    
    if gpu == True:
        # perform the rotation
        print('Running on the GPU')
        for i in range(origStack.shape[0]):
            if angleStep == 0:
                rotatedStack[i,:,:,:] = origStack[i,:,:,:]
            elif angleStep != 90:
                tempStack = cp.asarray(origStack[i,:,:,:])
                rotatedStack[i,:,:,:] = cupyx.scipy.ndimage.rotate(tempStack,   -angleStep*i, axes=rotateaxes, reshape=False).get()    
            else:
                tempStack = cp.asarray(origStack[i,:,:,:])
                rotatedStack[i,:,:,:] = cp.rot90(tempStack, -i, axes=rotateaxes).get()
            
            print('Rotating view ' + str(i) + ' - deg = ' + str(-angleStep*i))
    
    else:
        # perform the rotation
        print('Running on the CPU')
        for i in range(origStack.shape[0]):
            if angleStep == 0:
                rotatedStack[i,:,:,:] = origStack[i,:,:,:]
            elif angleStep != 90:
                rotatedStack[i,:,:,:] = scipy.ndimage.rotate(origStack[i,:,:,:],   -angleStep*i, axes=rotateaxes, reshape=False)    
            else:
                rotatedStack[i,:,:,:] = np.rot90(origStack[i,:,:,:], -i, axes=rotateaxes)
            
            print('Rotating view ' + str(i) + ' - deg = ' + str(-angleStep*i))

    return rotatedStack


def alignStack(origStack, reference=0, gpu=True):
    # pop out a warning
    print('Aligning the stack against view ' + str(reference) + '...')
    print('This functions assumes the axis=0 as the one storing different views')

    # create the stack
    alignedStack = np.zeros_like(origStack)
    
    if gpu == True:
        # perform the rotation
        print('Running on the GPU')
        
        referenceStack = cp.asarray(origStack[reference,:,:,:])
        
        for i in range(origStack.shape[0]):
            if i != reference:
                tempStack = cp.asarray(origStack[i,:,:,:])
                alignedStack[i,:,:,:] = (pf.my_alignND(referenceStack, tempStack)).get()
            else:
                alignedStack[i,:,:,:] = origStack[i,:,:,:]
            print('Aligning view ' + str(i) + ' agains reference = ' + str(reference))

    else:
        # perform the rotation
        print('Running on the CPU')
        
        for i in range(origStack.shape[0]):
            if i != reference:
                alignedStack[i,:,:,:] = pf.my_alignND(origStack[reference,:,:,:], origStack[i,:,:,:])
            else:
                alignedStack[i,:,:,:] = origStack[i,:,:,:]
            print('Aligning view ' + str(i) + ' agains reference = ' + str(reference))

    return alignedStack
   
    
   
def autocorrelateStack(origStack, gpu=True):
    # pop out a warning
    print('Calculating the stack auto-correlation...')

    # create the stack
    spimAcorr = np.zeros_like(origStack)

    if gpu == True:
        # perform the rotation
        print('Running on the GPU')
        # calculating autocorrelations from spim detections
        for i in range(origStack.shape[0]):
            print('Auto-correlating view ' + str(i))
            tempStack = cp.asarray(origStack[i,:,:,:])
            spimAcorr[i,:,:,:] = cp.abs(pf.my_autocorrelation(tempStack)).get()
    else:
        # perform the rotation
        print('Running on the CPU')
        # calculating autocorrelations from spim detections
        for i in range(origStack.shape[0]):
            print('Auto-correlating view ' + str(i))
            spimAcorr[i,:,:,:] = np.abs(pf.my_autocorrelation(origStack[i,:,:,:]))

    return spimAcorr


def crosscorrelateStack(origStack, gpu=True):
    # pop out a warning
    print('Calculating whole stack cross-correlation...')

    # create the stack
    spimXcorr = np.zeros((origStack.shape[0], origStack.shape[0], origStack.shape[1], origStack.shape[2], origStack.shape[3]))

    if gpu == True:
        # perform the rotation
        print('Running on the GPU')
        # calculating autocorrelations from spim detections
        for i in range(origStack.shape[0]):
            for j in range(origStack.shape[0]):
                print('Cross correlating view ' + str(i) + ' with ' + str(j))
                tempStack_i = cp.asarray(origStack[i,:,:,:])
                tempStack_j = cp.asarray(origStack[j,:,:,:])
                spimXcorr[i,j,:,:,:] = cp.abs(pf.my_correlationCentered(tempStack_i,tempStack_j)).get()
    else:
        # calculating autocorrelations from spim detections
        for i in range(origStack.shape[0]):
            for j in range(origStack.shape[0]):
                print('Cross correlating view ' + str(i) + ' with ' + str(j))
                spimXcorr[i,j,:,:,:] = np.abs(pf.my_correlationCentered(origStack[i,:,:,:],origStack[j,:,:,:]))
        
    return spimXcorr
    


# i'm not sure this thing is working as it should
def focusStack(origStack, sigma=2, axis=0):
    
    laplaceStack = np.zeros_like(origStack)
    
    minvalue = origStack.min()
    origStack = origStack - minvalue
    
    # calculating autocorrelations from spim detections
    for i in range(origStack.shape[0]):
        print('Calculating Laplacian of view ' + str(i))
        # laplaceStack[i,:,:,:] = np.abs(scipy.ndimage.laplace(origStack[i,:,:,:]))
        laplaceStack[i,:,:,:] = np.abs(scipy.ndimage.gaussian_filter(scipy.ndimage.laplace(origStack[i,:,:,:]), sigma))      
        # laplaceStack[i,:,:,:] = np.abs(scipy.ndimage.gaussian_laplace(origStack[i,:,:,:], sigma=2))
        
    
    index_array = np.argmax(laplaceStack, axis=axis)

    # Same as np.max(x, axis=-1, keepdims=True)
    focussedStack = np.take_along_axis(origStack, np.expand_dims(index_array, axis=axis), axis=axis)
     
    return focussedStack+minvalue
    

def poissonreweightStack(origStack, gpu=True):
    # pop out a warning
    print('Calculating the stack reweigheted possion statistics...')

    # create the stack
    rewStack = np.zeros_like(origStack)

    if gpu == True:
        # perform the rotation
        print('Running on the GPU')
        # calculating the fft_norm
        tempStack = cp.asarray(origStack[0,:,:,:]).copy()
        fft_norm = (1/origStack.shape[0]) * (cp.sqrt(cp.abs(cp.fft.rfft(tempStack))))
        for i in range(1,origStack.shape[0]):
            print('FFT-norm contribution view ' + str(i))     
            tempStack = cp.asarray(origStack[i,:,:,:])
            fft_norm = fft_norm + (1/origStack.shape[0]) * (cp.sqrt(cp.abs(cp.fft.rfft(tempStack))))
            
        for i in range(0,origStack.shape[0]):
            print('Reweighting view ' + str(i))     
            tempStack = cp.asarray(origStack[i,:,:,:])
            rewStack[i,:,:,:] = (cp.fft.irfft((cp.sqrt(cp.abs(cp.fft.rfft(tempStack)))/fft_norm) * cp.fft.rfft(tempStack))).get()

    else:
        # perform the rotation
        print('Running on the CPU')
        # calculating the fft_norm
        tempStack = np.asarray(origStack[0,:,:,:]).copy()
        fft_norm = (1/origStack.shape[0]) * (np.sqrt(np.abs(np.fft.rfft(tempStack))))
        for i in range(1,origStack.shape[0]):
            print('FFT-norm contribution view ' + str(i))     
            tempStack = np.asarray(origStack[i,:,:,:])
            fft_norm = fft_norm + (1/origStack.shape[0]) * (np.sqrt(np.abs(np.fft.rfft(tempStack))))
            
        for i in range(0,origStack.shape[0]):
            print('Reweighting view ' + str(i))     
            tempStack = np.asarray(origStack[i,:,:,:])
            rewStack[0,:,:,:] = (np.fft.irfft((np.sqrt(np.abs(np.fft.rfft(tempStack)))/fft_norm) * np.fft.rfft(tempStack)))

    return rewStack


# def poissonAverageStack(origStack, gpu=True):
#     # pop out a warning
#     print('Calculating the stack reweigheted possion statistics...')

#     # create the stack
#     rewStack = np.zeros_like(origStack[0,:,:,:])

#     if gpu == True:
#         # perform the rotation
#         print('Running on the GPU')
#         # calculating the fft_norm

#         fft_norm = (1/origStack.shape[0]) * np.sqrt(np.abs(np.fft.rfft(origStack[0,:,:,:])))
#         for i in range(1,origStack.shape[0]):
#             print(i)
#             fft_norm = fft_norm + (1/origStack.shape[0]) * np.sqrt(np.abs(np.fft.rfft(origStack[i,:,:,:])))
            
        
#         mean_init = (np.sqrt(np.abs(np.fft.rfft(origStack[0,:,:,:])))/fft_norm) * np.fft.rfft(origStack[0,:,:,:])
        
#         for i in range(1,origStack.shape[0]):
#             print(i)
#             mean_init = mean_init + (np.sqrt(np.abs(np.fft.rfft(origStack[i,:,:,:])))/fft_norm) * np.fft.rfft(origStack[i,:,:,:])
        
#         mean_real = np.fft.irfft(mean_init)


#     else:
#         # perform the rotation
#         print('Running on the CPU')
#         # calculating the fft_norm
#         tempStack = np.asarray(origStack[0,:,:,:]).copy()
#         fft_norm = (1/origStack.shape[0]) * (np.sqrt(np.abs(np.fft.rfft(tempStack))))
#         for i in range(1,origStack.shape[0]):
#             print('FFT-norm contribution view ' + str(i))     
#             tempStack = np.asarray(origStack[i,:,:,:])
#             fft_norm = fft_norm + (1/origStack.shape[0]) * (np.sqrt(np.abs(np.fft.rfft(tempStack))))
            
#         for i in range(0,origStack.shape[0]):
#             print('Reweighting view ' + str(i))     
#             tempStack = np.asarray(origStack[i,:,:,:])
#             rewStack[0,:,:,:] = (np.fft.irfft((np.sqrt(np.abs(np.fft.rfft(tempStack)))/fft_norm) * np.fft.rfft(tempStack)))

#     return rewStack



