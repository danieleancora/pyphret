# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:11:39 2020

@author: Daniele Ancora
"""

import time
import numpy as np
import pyphret.functions as pf
import matplotlib.pyplot as plt
from pyphret.functions import snrIntensity_db

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy  as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########


# %% GENERAL UTILITIES
# verbose status update and timing
def algorithmStatus(t, k, steps):
    if k % 100 == 0:
        elapsed = time.time() - t
        print("step", k, "of", steps, "- elapsed time per step", elapsed/100)
        t = time.time()
    return t


def fourierDistance(fftmagnitude, g_k):
    xp = cp.get_array_module(fftmagnitude)
    normalization = 1 / (fftmagnitude.size * xp.linalg.norm(fftmagnitude))
    gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
    gp_k = xp.abs(gp_k)                        # alias for Phi_k
    distance = xp.linalg.norm(fftmagnitude - gp_k) * normalization
    del gp_k

    return distance


def generateInitialGuess(fftmagnitude, rec_prior, phase_prior, attempts=10):
    xp = cp.get_array_module(fftmagnitude)

    # random phase if prior is None, otherwise start with the prior Fourier
    if (rec_prior is None) and (phase_prior is None):
        xp.random.seed()

        # g_k = 2*(xp.pi) * xp.random.rand(*fftmagnitude.shape)
        # g_k = fftmagnitude * xp.exp(1j*g_k)
        # g_k =  xp.fft.irfftn(g_k)
        # distance_best = fourierDistance(fftmagnitude, g_k)

        distance_best = xp.inf

        for attempt in range(attempts+1):
            seed = int(xp.random.randint(2**16))
            xp.random.seed(seed)
            
            g_k = 2*(xp.pi) * xp.random.rand(*fftmagnitude.shape)
            g_k = fftmagnitude * xp.exp(1j*g_k)
            g_k =  xp.fft.irfftn(g_k)
            distance = fourierDistance(fftmagnitude, g_k)
            print('Phase Attempt: ' + str(attempt) + ' error ' + str(distance))
            
            if distance < distance_best:
                distance_best = distance
                bestseed = seed
                print('Updated!!!')

        # generate again the guess based on the best seed found
        xp.random.seed(bestseed)
        g_k = 2*(xp.pi) * xp.random.rand(*fftmagnitude.shape)
        g_k = fftmagnitude * xp.exp(1j*g_k)
        g_k =  xp.fft.irfftn(g_k)
        distance = fourierDistance(fftmagnitude, g_k)
        print('Phase choosen: ' + str(attempt) + ' error ' + str(distance))

    elif (rec_prior is not None) and (phase_prior is None):
        g_k = rec_prior
    elif (rec_prior is None) and (phase_prior is not None):
        g_k = fftmagnitude*xp.exp(1j*phase_prior)
        g_k = xp.fft.irfftn(g_k)
        
    # free memory
    rec_prior = None
    phase_prior = None
    # gp_k = None
    distance = None
    distance_best = None
    
    return g_k


def generateMask(g_k, masked, size, xp = None):
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

    elif (masked == "autocorrelation"):
        mask = pf.autocorrelation_maskND(g_k, 0.04)

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
def HIO(fftmagnitude, g_k, mask, beta, steps, measure=False):
    print('Running Phase-Retrieval iterations using Hybrid Input-Output method')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    error = None

    if measure == True:
        normalization = 1 / (fftmagnitude.size * xp.linalg.norm(fftmagnitude))
        error = xp.zeros(steps)

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

        # measure the solution distance
        if measure == True:
            gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
            gp_k = xp.abs(gp_k)                        # alias for Phi_k
            error[k] = xp.linalg.norm(fftmagnitude - gp_k) * normalization

    return (g_k, mask, error)


# ONGOING DEVELOPMENT - it requires more memory than ER implementation
def HIO_mode(fftmagnitude, g_k, mask, beta, steps, mode, measure=False, parameters=[100, 30, 1.5, 0.9, 1]):
    print('Running Phase-Retrieval iterations using Hybrid Input-Output method with options')
    xp = cp.get_array_module(fftmagnitude)
    t = time.time()    
    error = xp.zeros(steps)
    normalization = 1 / (fftmagnitude.size * xp.linalg.norm(fftmagnitude))
    epsilon = parameters[4]
    
    if epsilon == 0:
        print('WARNING: you are actually running ER method, if you want to get back to HIO set parammeters[4]=1')
    
    if mode == 'normal':
        print('Normal HIO implementation with epsilon = ' + str(epsilon))

    if mode == 'shrink-wrap':
        counter = -1
        maskupdate = parameters[0]
        nupdates = -int(-steps//maskupdate)
        start_sigma = parameters[1]
        stop_sigma = parameters[2]
        print('The mask evolves using shrink-wrap rules every ' + str(maskupdate) + ' steps')
        print('Starting smoothing with sigma: ' + str(start_sigma))
        print('Ending smoothing with sigma: ' + str(stop_sigma))
        sigma = xp.linspace(start_sigma, stop_sigma, nupdates)
        # mask = pf.autocorrelation_maskND(g_k, 0.04)
        # g_k = pf.fouriermod2autocorrelation(fftmagnitude)
        
    if mode == 'sparsity':
        counter = -1
        maskupdate = parameters[0]
        nupdates = -int(-steps//maskupdate)
        # mask = pf.autocorrelation_maskND(g_k, 0.04)
        start_sparsity = xp.sum(mask)/mask.size
        end_sparsity = 0.1
        print('The mask evolves using sparsity rules every ' + str(maskupdate) + ' steps')
        print('Starting sparsity: ' + str(start_sparsity))
        print('End sparsity: ' + str(end_sparsity))
        sparsity = xp.linspace(start_sparsity, end_sparsity, nupdates)
        
    if mode == 'exponential-average':
        alpha = parameters[3]
        g_exp = g_k
        print('The solution evolves using exponentially weighted average with alpha ' + str(alpha))
        print('... it means the exponential average is done on past ' + str(1/(1-alpha)) + 'estimates')


    # iteration starts here
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
        g_k[index] = epsilon * (g_k[index] - (beta*gp_k[index]))

        # 5th step - Shrink-wrap implementation
        if mode == 'shrink-wrap':
            if (k+1) % maskupdate == 0:
                counter = counter+1
                print("smoothed mask with sigma = ", sigma[counter])   
                gp_k = pf.my_gaussblur(g_k, sigma[counter])
                mask = pf.threshold_maskND(gp_k, 0.01)
                
        # 5th step - Shrink-wrap implementation
        if mode == 'sparsity':
            if (k+1) % maskupdate == 0:
                counter = counter+1
                print("smoothed mask with sigma = ", sparsity[counter])
                mask = pf.sparsity_maskND(g_k, 0.1)
                
        # update process following exponential average rules
        if mode == 'exponential-average':
            g_exp = alpha * g_exp + (1-alpha) * g_k
          
        # measure the solution distance
        if measure == True:
            gp_k = xp.fft.rfftn(g_k)                     # alias for G_k
            gp_k = xp.abs(gp_k)                        # alias for Phi_k
            # error[k] = xp.linalg.norm(fftmagnitude - gp_k) * normalization

            error[k] = snrIntensity_db(fftmagnitude/fftmagnitude.sum(), xp.abs(fftmagnitude/fftmagnitude.sum()-gp_k/gp_k.sum()))


    if mode == 'exponential-average':
        g_k = g_exp

    return (g_k, mask, error)


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
             rec_prior=None, phase_prior=None, attempts=10,
             masked='full', method='ER', mode='normal',
             measure = False,
             beta=0.9, steps=200, parameters=[100, 30, 1.5, 0.9, 1]):
    """
    This is the implementation of the Phase Retrieval algorithm. It relies on 
    the functions above in order to accomplish the task of recovery the phase 
    connected with a given modulus.

    Parameters
    ----------
    fftmagnitude : TYPE
        modulus of the recorded Fourier transform.
    rec_prior : TYPE, optional
        starting guess for the object. The default is None.
    phase_prior : TYPE, optional
        starting guess for the Foruier phase. The default is None.
    attempts : TYPE, optional
        DESCRIPTION. The default is 10.
    masked : TYPE, optional
        'full', 'half', 'circular', 'spherical'. The default is 'full'.
    method : TYPE, optional
        'ER', 'II', 'OO', 'HIO', 'OSS', 'ER_pedantic', 'ER_inplaceFFT', 'HIO_mode', 'HIO_shrinkwrap'. The default is 'ER'.
    mode : TYPE, optional
        'classical', 'normal', 'shrink-wrap', 'sparsity'. The default is 'normal'.
    measure : TYPE, optional
        DESCRIPTION. The default is False.
    beta : TYPE, optional
        feedback parameter as defined in HIO. The default is 0.9.
    steps : TYPE, optional
        number of phase retrieval steps. The default is 200.
    parameters : TYPE, optional
        DESCRIPTION. The default is [100, 30, 1.5, 0.9].

    Returns
    -------
    g_k : TYPE
        phase retrieved reconstruction.
    mask : TYPE
        final output binary mask.
    error : TYPE
        error during iteration.

    """

    # start computing time from the first call 
    t = time.time()    

    # agnostic code, xp is either numpy or cupy depending on the magnitude array module
    xp = cp.get_array_module(fftmagnitude)

    # check if input values are allowed for the simulation
    assert beta > 0, 'the value for beta should be a positive'
    assert steps >= 0, 'number of steps should be a positive number'    
    assert (masked=='full') or (masked=='half') or (masked=='circular') or (masked=='spherical') or (masked=='autocorrelation')
    assert (mode=='classical') or (mode=='normal') or (mode=='shrink-wrap') or (mode=='sparsity') or (mode=='exponential-average'),\
        'the mode should be \'normal\' or \'shrink-wrap\''
    assert (method=='ER') or (method=='II') or (method=='OO') or (method=='HIO') or (method=='HIO_shrinkwrap') or (method=='OSS') or (method=='ER_pedantic') or (method=='ER_inplaceFFT'),\
        'the available methods are: \'ER\', \'II\', \'OO\', \'HIO\', \'OSS\', \'ER_pedantic\', \'ER_inplaceFFT\''

    # the initial guess is computed with the prior information
    g_k  = generateInitialGuess(fftmagnitude, rec_prior, phase_prior, attempts)
    mask = generateMask(g_k, masked, size=g_k.shape, xp=xp)
    error = None
    gp_k = None

    # calls to Phase Retrieval functions. They take an estimate g_k and throw the next one g_k after steps iterations
    if method=='ER_pedantic': 
        (g_k, mask) = ER_pedantic(fftmagnitude, g_k, mask, steps)

    elif method=='ER':
        (g_k, mask) = ER(fftmagnitude, g_k, mask, steps)

    elif method=='II':
        (g_k, mask) = II(fftmagnitude, g_k, mask, beta, steps)

    elif method=='OO':
        (g_k, mask) = OO(fftmagnitude, g_k, mask, beta, steps)

    elif method=='HIO' and mode=='classical':
        (g_k, mask, error) = HIO(fftmagnitude, g_k, mask, beta, steps, measure)

    elif (method=='HIO' and mode=='normal') or (method=='HIO' and mode=='sparsity') or (method=='HIO' and mode=='shrink-wrap') or (method=='HIO' and mode=='exponential-average'):
        (g_k, mask, error) = HIO_mode(fftmagnitude, g_k, mask, beta, steps, mode, measure, parameters)

    elif method=='OSS':
        (g_k, mask) = OSS(fftmagnitude, g_k, mask, beta, steps)

    elif method=='ER_inplaceFFT':
        (g_k, mask) = ER_inplaceFFT(fftmagnitude, g_k, mask, steps)     
   
    # measure the solution distance
    if measure == False:
        error = xp.zeros(1,)
        normalization = 1 / (fftmagnitude.size * xp.linalg.norm(fftmagnitude))
        gp_k = xp.fft.rfftn(g_k)                        # alias for G_k
        gp_k = xp.abs(gp_k)                             # alias for Phi_k
        error[0] = xp.linalg.norm(fftmagnitude - gp_k) * normalization

    del gp_k        

    print('P.S. --> Algorithm has finished, the correct solution is g_k * mask!')
    return (g_k, mask, error)
        
        
