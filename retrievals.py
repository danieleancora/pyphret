# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:11:39 2020

@author: Daniele Ancora
"""

import time
import cupy as cp
import numpy as np
import pyphret.functions as pf

# %% GENERAL UTILITIES
# verbose status update and timing
def algorithmStatus(t, k, steps):
    if k % 100 == 0:
        elapsed = time.time() - t
        print("step", k, "of", steps, "- elapsed time per step", elapsed/100)
        t = time.time()
    return t


def generateInitialGuess(fftmagnitude, rec_prior, phase_prior):
    xp = cp.get_array_module(fftmagnitude)

    # random phase if prior is None, otherwise start with the prior Fourier
    if (rec_prior is None) and (phase_prior is None):
        xp.random.seed()
        g_k = 2*(xp.pi) * xp.random.rand(*fftmagnitude.shape)
        g_k = fftmagnitude * xp.exp(1j*g_k)
        g_k =  xp.fft.irfftn(g_k)
        # g_k =  xp.real(xp.fft.irfftn(fftmagnitude*xp.exp(1j*2*xp.pi*xp.random.rand(*fftmagnitude.shape))))
    elif (rec_prior is not None) and (phase_prior is None):
        g_k = rec_prior
    elif (rec_prior is None) and (phase_prior is not None):
        g_k = fftmagnitude*xp.exp(1j*phase_prior)
        g_k = xp.fft.irfftn(g_k)
        
    # free memory
    rec_prior = None
    phase_prior = None
    
    return g_k


def generateMask(masked, size, xp = None):
    # object support constraint
    if (masked is None) or (masked =="full"):
        print('Uniform full mask')
        mask = True

    # squared masking
    elif (masked == "half"):
        print('Uniform mask on half space')
        mask = np.zeros(size, dtype=bool)
        if len(size)==1:
            mask[0:int(mask.shape[0]/2)] = True
        elif len(size)==2:
            mask[0:int(mask.shape[0]/2), 0:int(mask.shape[1]/2)] = True
        elif len(size)==3:
            mask[0:int(mask.shape[0]/2), 0:int(mask.shape[1]/2), 0:int(mask.shape[2]/2)] = True
        elif len(size)==4:
            mask[0:int(mask.shape[0]/2), 0:int(mask.shape[1]/2), 0:int(mask.shape[2]/2), 0:int(mask.shape[3]/2)] = True
        else:
            print('Dimensions > 4 are nor currently supported, we set no mask')
            mask = True

    # circular and spherical masking
    elif (masked == "circular") and (len(size) == 2):
        print('Circular mask')
        mask = pf.circular_mask2D(size[0], size[1])
    elif (masked == "spherical") and (len(size) == 3):
        print('Spherical mask')
        mask = pf.spherical_mask3D(size[0], size[1], size[2])

    # elif masked == "autocorrelation":
    #     mask = (autocorrelation > (xp.max(autocorrelation)*0.05))

    # not supported decisions return a full mask
    else:
        print('Currently not supported mask decision, we set no mask')
        mask = True
        
    if xp is not None:
        print('Sending the mask to GPU')
        mask = xp.asarray(mask)
        
    return mask
    

# %% PHASE RETRIEVAL IMPLEMENTATIONS
# This is a pedagogical implementation for the error reduction protocol
def ER_pedantic(fftmagnitude, g_k, mask, steps):
    print('Running Phase-Retrieval iterations using a pedantic implementation of Error Reduction')
    xp = cp.get_array_module(fftmagnitude)      # to save agnosticity of the code
    t = time.time()    
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)
 
        # phase retrieval four-iterations
        G_k = xp.fft.rfftn(g_k)
        Phi_k = xp.angle(G_k)
        Gp_k = fftmagnitude * xp.exp(1j * Phi_k)
        gp_k = xp.fft.irfftn(Gp_k)
            
        # find elements that violate object domain constraints or are not masked
        # satisfied = xp.logical_and(gp_k>0, mask) 
        satisfied = (gp_k>0) 
        violated = xp.logical_not(satisfied)

        # 4th step - updates for elements that violate object domain constraints
        g_k[satisfied] = gp_k[satisfied]
        g_k[violated] = 0
    
    return (g_k, mask)


# ONGOING DEVELOPMENT - supports volumes up to 1024x1024x512 double
def ER(fftmagnitude, g_k, mask, steps):
    print('Running Phase-Retrieval iterations using Error Reduction method')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)
        
        # phase retrieval four-iterations, this sequence minimize memory usage
        g_k = xp.fft.rfftn(g_k)                    # alias for G_k
        g_k = xp.angle(g_k)                        # alias for Phi_k
        g_k = fftmagnitude * xp.exp(1j * g_k)      # alias for Gp_k
        g_k = xp.fft.irfftn(g_k)                   # alias for gp_k
            
        # 4th step - updates for elements that violate object domain constraints
        index = (g_k<0)
        g_k[index] = 0

    return (g_k, mask)


# ONGOING DEVELOPMENT - it requires more memory than ER implementation
def HIO(fftmagnitude, g_k, mask, beta, steps):
    print('Running Phase-Retrieval iterations using Hybrid Input-Output method')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
        gp_k = xp.angle(gp_k)                        # alias for Phi_k
        gp_k = xp.exp(1j * gp_k)                     # alias for Gp_k
        gp_k = fftmagnitude * gp_k                   # alias for Gp_k
        gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
                
        # 4th step - updates for elements that violate object domain constraints
        index = xp.logical_and(gp_k>0, mask) 
        g_k[index] = gp_k[index]
        index = xp.logical_not(index)
        g_k[index] = g_k[index] - (beta*gp_k[index])

    return (g_k, mask)


# ONGOING DEVELOPMENT - it requires more memory than ER implementation
def HIO_mode(fftmagnitude, g_k, mask, beta, steps, mode):
    print('Running Phase-Retrieval iterations using Hybrid Input-Output method')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    
    if mode == 'shrink-wrap':
        maskupdate = 20
        nupdates = -int(-steps//maskupdate)
        sigma = xp.linspace(3, 1.5, nupdates)
        counter = -1
        # prior 
        g_k = pf.fouriermod2autocorrelation(fftmagnitude)
        mask = pf.autocorrelation_maskND(g_k,0.04)
        
    if mode == 'sparsity':
        maskupdate = 100

        
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
        gp_k = xp.angle(gp_k)                        # alias for Phi_k
        gp_k = xp.exp(1j * gp_k)                     # alias for Gp_k
        gp_k = fftmagnitude * gp_k                   # alias for Gp_k
        gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
                
        # 4th step - updates for elements that violate object domain constraints
        index = xp.logical_and(gp_k>0, mask) 
        g_k[index] = gp_k[index]
        index = xp.logical_not(index)
        g_k[index] = g_k[index] - (beta*gp_k[index])

        if mode == 'shrink-wrap':
            # 5th step - Shrink-wrap implementation
            if (k+1) % maskupdate == 0:
                counter = counter+1
                print("smoothed mask with sigma = ", sigma[counter])   

            gp_k = pf.my_gaussblur(g_k, sigma[counter])
            mask = pf.threshold_maskND(gp_k, 0.20)

        if mode == 'sparsity' and (k+1) % maskupdate == 0:
            mask = pf.sparsity_maskND(g_k, 0.10)


    return (g_k, mask)


# ONGOING DEVELOPMENT - it requires more memory than ER implementation
def HIO_shrinkwrap(fftmagnitude, g_k, mask, beta, steps):
    print('Running Phase-Retrieval iterations using Hybrid Input-Output method')
    print('The mask evolves using shrink-wrap rules every 20 steps')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()  
    
    maskupdate = 20
    nupdates = -int(-steps//maskupdate)
    sigma = xp.linspace(3, 1.5, nupdates)
    counter = -1

    g_k = pf.fouriermod2autocorrelation(fftmagnitude)
    # mask = pf.autocorrelation_maskND(g_k,0.04)

    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
        gp_k = xp.angle(gp_k)                        # alias for Phi_k
        gp_k = xp.exp(1j * gp_k)                     # alias for Gp_k
        gp_k = fftmagnitude * gp_k                   # alias for Gp_k
        gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
                
        # 4th step - updates for elements that violate object domain constraints
        index = xp.logical_and(gp_k>0, mask) 
        g_k[index] = gp_k[index]
        index = xp.logical_not(index)
        g_k[index] = g_k[index] - (beta*gp_k[index])

        # 5th step - Shrink-wrap implementation
        if (k+1) % maskupdate == 0:
            counter = counter+1
            print("smoothed mask with sigma = ", sigma[counter])   
        
        gp_k = pf.my_gaussblur(g_k, sigma[counter])
        mask = pf.threshold_maskND(gp_k, 0.20)

    return (g_k, mask)


# APPARENTLY WORKING - it requires more memory than ER implementation
def OSS(fftmagnitude, g_k, mask, beta, steps):
    print('Running Phase-Retrieval iterations using Oversampling Smoothness method')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    
    if xp.all(mask) == True:
        print('WARNING: there is no mask support, OSS won\'t do anything more than HIO...')
    
    N = g_k.shape[0]
    alpha = xp.linspace(N, 1/N, 10)
    counter = -1
    
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
        gp_k = xp.angle(gp_k)                        # alias for Phi_k
        gp_k = xp.exp(1j * gp_k)                     # alias for Gp_k
        gp_k = fftmagnitude * gp_k                   # alias for Gp_k
        gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
                
        # 4th step - equal to HIO
        index = xp.logical_and(gp_k>0, mask) 
        g_k[index] = gp_k[index]
        index = xp.logical_not(index)
        g_k[index] = g_k[index] - (beta*gp_k[index])
        
        if (k-1) % (steps/10) == 0:
            counter = counter+1
            print("change alpha to ", alpha[counter])   
        
        # 5th step, applying a smoothing only outside the support area
        gp_k = pf.my_gaussBlurInv(g_k, alpha[counter])
        index = xp.logical_not(mask) 
        g_k[index] = gp_k[index]

    return (g_k, mask)


def OO(fftmagnitude, g_k, mask, beta, steps):
    print('Running Phase-Retrieval iterations using Output-Output')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
        gp_k = xp.angle(gp_k)                        # alias for Phi_k
        gp_k = xp.exp(1j * gp_k)                     # alias for Gp_k
        gp_k = fftmagnitude * gp_k                   # alias for Gp_k
        gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
                
        # 4th step - updates for elements that violate object domain constraints
        index = xp.logical_and(gp_k>0, mask) 
        g_k[index] = gp_k[index]
        index = xp.logical_not(index)
        g_k[index] = gp_k[index] - (beta*gp_k[index])

    return (g_k, mask)


def II(fftmagnitude, g_k, mask, beta, steps):
    print('Running Phase-Retrieval iterations using Input-Input')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
        gp_k = xp.angle(gp_k)                        # alias for Phi_k
        gp_k = xp.exp(1j * gp_k)                     # alias for Gp_k
        gp_k = fftmagnitude * gp_k                   # alias for Gp_k
        gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
                
        # 4th step - updates for elements that violate object domain constraints
        index = xp.logical_and(gp_k>0, mask) 
        g_k[index] = g_k[index]
        index = xp.logical_not(index)
        g_k[index] = g_k[index] - (beta*gp_k[index])

    return (g_k, mask)


# Experimental implementation to try to save memory working in place
def ER_inplaceFFT(fftmagnitude, g_k, mask, steps):
    print('Running Error Reduction iterations using pedantic implementation')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    

    # only cupyx.scipy supports in place fft. but I don't see memory consumption difference 
    from cupyx.scipy import fft
    import cupyx.scipy as yp

    for k in range(0,steps):
        t = algorithmStatus(t, k, steps)

        # phase retrieval four-iterations, this sequence minimize memory usage
        g_k = yp.fft.rfftn(g_k, overwrite_x=True)  # alias for G_k
        g_k = xp.angle(g_k)                        # alias for Phi_k
        g_k = fftmagnitude * xp.exp(1j * g_k)      # alias for Gp_k
        g_k = yp.fft.irfftn(g_k, overwrite_x=True)                   # alias for gp_k
            
        # 4th step - updates for elements that violate object domain constraints
        index = (g_k<0)
        g_k[index] = 0

    return (g_k, mask)


# %% MAIN ALGORITHM - every functions above is called within this
def phaseRet(fftmagnitude, 
             rec_prior=None, phase_prior=None, masked=None,
             method='ER', mode='normal',
             beta=0.9, steps=200):

    # start computing time from the first call 
    t = time.time()    

    # agnostic code, xp is either numpy or cupy depending on the magnitude array module
    xp = cp.get_array_module(fftmagnitude)

    # check if input values are allowed for the simulation
    assert beta > 0, 'the value for beta should be a positive'
    assert steps >= 0, 'number of steps should be a positive number'    
    assert (mode=='normal') or (mode=='shrink-wrap') or (mode=='sparsity'), 'the mode should be \'normal\' or \'shrink-wrap\''
    assert (method=='ER') or (method=='II') or (method=='OO') or (method=='HIO') or (method=='HIO_shrinkwrap') or (method=='OSS') or (method=='ER_pedantic') or (method=='ER_inplaceFFT'),\
        'the available methods are: \'ER\', \'II\', \'OO\', \'HIO\', \'OSS\', \'ER_pedantic\', \'ER_inplaceFFT\''

    # the initial guess is computed with the prior information
    g_k  = generateInitialGuess(fftmagnitude, rec_prior, phase_prior)
    mask = generateMask(masked, size=g_k.shape, xp=xp)

    # calls to Phase Retrieval functions. They take an estimate g_k and throw the next one g_k after steps iterations
    if method=='ER_pedantic': 
        (g_k, mask) = ER_pedantic(fftmagnitude, g_k, mask, steps)
    elif method=='ER':
        (g_k, mask) = ER(fftmagnitude, g_k, mask, steps)
    elif method=='II':
        (g_k, mask) = II(fftmagnitude, g_k, mask, beta, steps)
    elif method=='OO':
        (g_k, mask) = OO(fftmagnitude, g_k, mask, beta, steps)
    elif method=='HIO' and mode!='sparsity':
        (g_k, mask) = HIO(fftmagnitude, g_k, mask, beta, steps)
    elif method=='HIO' and mode=='sparsity':
        (g_k, mask) = HIO_mode(fftmagnitude, g_k, mask, beta, steps, mode)
        
    elif method=='HIO_shrinkwrap':
        (g_k, mask) = HIO_shrinkwrap(fftmagnitude, g_k, mask, beta, steps)
    elif method=='OSS':
        (g_k, mask) = OSS(fftmagnitude, g_k, mask, beta, steps)
    elif method=='ER_inplaceFFT':
        (g_k, mask) = ER_inplaceFFT(fftmagnitude, g_k, mask, steps)
            
    return (g_k * mask, mask)
        
        


# %% TEMP CHUNK OF WORKING CODE
    # # This is a pedagogical implementation for the error reduction protocol
    # if method=='ER_pedantic': 
    #     g_k = ER_pedantic(fftmagnitude, g_k, mask, steps)
            
    # # ONGOING DEVELOPMENT - supports volumes up to 1024x1024x512 double
    # elif method=='ER':
    #     g_k = ER(fftmagnitude, g_k, mask, steps)
    #     # print('Running Error Reduction iterations')
    #     # for k in range(0,steps):
    #     #     if k % 100 == 0:
    #     #         elapsed = time.time() - t
    #     #         print("step", k, "of", steps, "- elapsed time per step", elapsed/100)
    #     #         t = time.time()
 
    #     #     # phase retrieval four-iterations, this sequence minimize memory usage
    #     #     g_k = xp.fft.rfftn(g_k)                    # alias for G_k
    #     #     g_k = xp.angle(g_k)                        # alias for Phi_k
    #     #     g_k = fftmagnitude * xp.exp(1j * g_k)      # alias for Gp_k
    #     #     g_k = xp.fft.irfftn(g_k)                   # alias for gp_k
            
    #     #     # 4th step - updates for elements that violate object domain constraints
    #     #     index = (g_k<0)
    #     #     g_k[index] = 0
    
    # # ONGOING DEVELOPMENT - it requires more memory than ER implementation
    # elif method=='HIO':
    #     g_k = HIO(fftmagnitude, g_k, mask, beta, steps)
    #     # print('Running Hybrid Input-Output iterations')
    #     # for k in range(0,steps):
    #     #     if k % 100 == 0:
    #     #         elapsed = time.time() - t
    #     #         print("step", k, "of", steps, "- elapsed time per step", elapsed/100)
    #     #         t = time.time()

    #     #     # phase retrieval four-iterations, this sequence minimize memory usage
    #     #     gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
    #     #     gp_k = xp.angle(gp_k)                        # alias for Phi_k
    #     #     gp_k = xp.exp(1j * gp_k)      # alias for Gp_k
    #     #     gp_k = fftmagnitude * gp_k      # alias for Gp_k
    #     #     gp_k = xp.fft.irfftn(gp_k)                   # alias for gp_k
            
    #     #     # 4th step - updates for elements that violate object domain constraints
    #     #     index = (gp_k>0) 
    #     #     g_k[index] = gp_k[index]
    #     #     index = xp.logical_not(index)
    #     #     g_k[index] = g_k[index] - (beta*gp_k[index])

    # # Experimental implementation to try to save memory
    # elif method=='ER_inplaceFFT':
    #     g_k = ER_inplaceFFT(fftmagnitude, g_k, mask, steps)
        
    #     # # only cupyx.scipy supports in place fft. but I don't see memory consumption difference 
    #     # from cupyx.scipy import fft
    #     # import cupyx.scipy as yp
        
    #     # for k in range(0,steps):
    #     #     if k % 100 == 0:
    #     #         elapsed = time.time() - t
    #     #         print("step", k, "of", steps, "- elapsed time per step", elapsed/100)
    #     #         t = time.time()
           
    #     #     # phase retrieval four-iterations, this sequence minimize memory usage
    #     #     g_k = yp.fft.rfftn(g_k, overwrite_x=True)  # alias for G_k
    #     #     g_k = xp.angle(g_k)                        # alias for Phi_k
    #     #     g_k = fftmagnitude * xp.exp(1j * g_k)      # alias for Gp_k
    #     #     g_k = yp.fft.irfftn(g_k, overwrite_x=True)                   # alias for gp_k
            
    #     #     # 4th step - updates for elements that violate object domain constraints
    #     #     index = (g_k<0)
    #     #     g_k[index] = 0

