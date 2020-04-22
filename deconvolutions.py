# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:13:31 2020

@author: Daniele Ancora
"""

# %% LIBRARIES CALL
import scipy
import cupy as cp
import numpy as np
import cupyx.scipy
from cupyx.scipy import ndimage

from pyphret.functions import my_convolution, my_correlation, axisflip


# %% DENOISE ROUTINES
# this function was imported directly from skimage.restoration, the np calls were replaced with xp
def _denoise_tv_chambolle_nd(image, weight=0.1, eps=2.e-4, n_iter_max=200):
    """Perform total-variation denoising on n-dimensional images.

    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight. The greater `weight`, the more denoising (at
        the expense of fidelity to `input`).
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array of floats.

    Notes
    -----
    Rudin, Osher and Fatemi algorithm.
    """

    xp = cp.get_array_module(image)
    ndim = image.ndim
    p = xp.zeros((image.ndim, ) + image.shape, dtype=image.dtype)
    g = xp.zeros_like(p)
    d = xp.zeros_like(image)
    i = 0
    while i < n_iter_max:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = xp.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = xp.sqrt((g ** 2).sum(axis=0))[xp.newaxis, ...]
        E += weight * norm.sum()
        tau = 1. / (2.*ndim)
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if xp.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


# %% DECONVOLUTION ROUTINES
# PAY ATTENTION: maybe i cannot implement this procedure using only realfft, 
# since there might non trivial phase connecting them.
def wiener_deconvolution(signal, kernel, snr):
    xp = cp.get_array_module(signal)
    # clever way to create a tuple of tuple containing the difference between two lists
    difference = tuple((0, x-y) for x,y in zip(signal.shape,kernel.shape))
    kernel = np.pad(kernel, difference, mode='constant' )
    
    # wiener deconvolution starts here, snr is the signal to noise ratio
    H = xp.fft.fftn(kernel)
    
#    G = ( xp.conj(H) / (H*xp.conj(H) + snr**2) )
#    deconvolved = xp.fft.fftshift( xp.real( xp.fft.ifft(xp.fft.fftn(signal) * G) ) )    
    deconvolved = xp.fft.fftshift( xp.real( xp.fft.ifftn( (xp.fft.fftn(signal)*xp.conj(H)) / (H*xp.conj(H) + snr**2) ) ) )
    return deconvolved


# %% here the deconvolution routines with small kernels
def richardsonLucy_smallKernel(image, psf, iterations=10, clip=True, verbose=False):
    print('This procedure may be very slow! -> Use it with small psf!!!')
    xp = cp.get_array_module(image)
    xps = cupyx.scipy.get_array_module(image)

    image = image.astype(xp.float)
    psf = psf.astype(xp.float)
    im_deconv = xp.full(image.shape, 0.5)
    psf_flip = axisflip(psf)
    
    for i in range(iterations):
        if verbose==True:
            print('Iteration ' + str(i))
            
        relative_blur = xps.ndimage.convolve(im_deconv, psf)
        relative_blur = image / relative_blur
        im_deconv *= xps.ndimage.convolve(relative_blur, psf_flip)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv

def maxAPosteriori_smallKernel(image, psf, iterations=10, clip=True, verbose=False):
    print('This procedure may be very slow! -> Use it with small psf!!!')
    xp = cp.get_array_module(image)
    xps = cupyx.scipy.get_array_module(image)

    image = image.astype(xp.float)
    psf = psf.astype(xp.float)
    im_deconv = xp.full(image.shape, 0.5)
    psf_flip = axisflip(psf)
    
    for i in range(iterations):
        if verbose==True:
            print('Iteration ' + str(i))

        relative_blur = xps.ndimage.convolve(im_deconv, psf)
        relative_blur = image / relative_blur - 1 
            
        im_deconv *= xp.exp(xps.ndimage.convolve(relative_blur, psf_flip))
        
    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv



# %% big kernel implemented with FFT, 
# Here i call signal the image and kernel the psf, to differenciate from nonFFT implementations


def richardsonLucy(signal, kernel, iterations=10, measure=True, clip=True, verbose=False):
    xp = cp.get_array_module(signal)
    signal_deconv = signal 
    # starting guess with a flat image
    signal_deconv = xp.full(signal.shape, 0.5)

    epsilon = 1e-7

    kernel_mirror  = axisflip(kernel)
    
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        if verbose==True:
            print('Iteration ' + str(i))

        relative_blur = my_convolution(signal_deconv, kernel)
        if measure==True:
            error[i] = xp.linalg.norm(signal-relative_blur)
        relative_blur = signal / relative_blur
        
        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update 
        signal_deconv *= my_convolution(relative_blur, kernel_mirror)
        
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1

    return signal_deconv, error
    

def maxAPosteriori(signal, kernel, iterations=10, measure=True, clip=True, verbose=False):
    xp = cp.get_array_module(signal)
    signal_deconv = signal 
    # starting guess with a flat image
    signal_deconv = xp.full(signal.shape, 0.5)

    epsilon = 1e-7

    kernel_mirror = axisflip(kernel)
    
    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        if verbose==True:
            print('Iteration ' + str(i))

        relative_blur = my_convolution(signal_deconv, kernel)
        if measure==True:
            error[i] = xp.linalg.norm(signal-relative_blur)
        relative_blur = signal / relative_blur

        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update 
        signal_deconv *= xp.exp(my_convolution(relative_blur - 1, kernel_mirror))
        
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1
        
    return signal_deconv, error


def anchorUpdate(signal, kernel, prior=np.float32(0), iterations=10, measure=True, clip=False, verbose=True):
    xp = cp.get_array_module(signal)
    signal_deconv = signal
    signal = signal / signal.sum()
    kernel = kernel / kernel.sum()
    epsilon = 1e-7

    # starting guess with a flat image
    if prior.any()==0:
        signal_deconv = xp.full(signal.shape,0.5) + 0.01*xp.random.rand(*signal.shape)
    else:
        signal_deconv = prior #+ 0.1*prior.max()*xp.random.rand(*signal.shape)

    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        if verbose==True and (i % 100)==0:
            print('Iteration ' + str(i))

        kernel_update = my_convolution(signal_deconv, kernel)
        kernel_update = my_correlation(kernel_update, kernel)
        kernel_update = kernel_update / kernel_update.sum()

        kernel_mirror = axisflip(kernel_update)

        relative_blur = my_convolution(signal_deconv, kernel_update)
        if measure==True:
            error[i] = xp.linalg.norm(signal-relative_blur)
        relative_blur = signal / relative_blur

        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update 
        signal_deconv *= my_convolution((relative_blur), kernel_mirror)
                
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1

    return signal_deconv, error


def anchorUpdate2(signal, kernel, prior=np.float32(0), iterations=10, measure=True, clip=True, verbose=False):
    xp = cp.get_array_module(signal)
    signal_deconv = signal
    signal = signal / signal.sum()
    kernel = kernel / kernel.sum()
    epsilon = 1e-7

    # starting guess with a flat image
    if prior.any()==0:
        signal_deconv = xp.full(signal.shape,0.5) + 0.01*xp.random.rand(*signal.shape)
    else:
        signal_deconv = prior #+ 0.1*prior.max()*xp.random.rand(*signal.shape)
    signal_deconv = signal_deconv/signal_deconv.max()


    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        if verbose==True:
            print('Iteration ' + str(i))

        # kernel_update = my_convolution(signal_deconv, kernel)
        # kernel_update = my_correlation(kernel_update, kernel)
        # kernel_update = kernel_update / kernel_update.sum()

        kernel_update = my_convolution(kernel, axisflip(kernel))
        kernel_update = my_convolution( (kernel_update), signal_deconv)
        kernel_update = kernel_update / kernel_update.sum()

        kernel_mirror = axisflip(kernel_update)

        relative_blur = my_convolution(signal_deconv, kernel_update)
        if measure==True:
            error[i] = xp.linalg.norm(signal-relative_blur)
        relative_blur = signal / relative_blur

        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update 
        signal_deconv *= my_convolution(axisflip(relative_blur), kernel_mirror)
                
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1

    return signal_deconv, error



def schulzSnyder(correlation, kernel, prior=np.float32(0), iterations=10, measure=True, clip=True, verbose=True):
    xp = cp.get_array_module(correlation)
    epsilon = 1e-7

    # starting guess with a flat image
    if prior.any()==0:
        signal_decorr = xp.full(correlation.shape,0.5) + 0.01*xp.random.rand(*correlation.shape)
    else:
        signal_decorr = prior #+ 0.1*prior.max()*xp.random.rand(*signal.shape)
        
    R_0 = signal_decorr.sum()
    signal_decorr = signal_decorr / R_0


    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        if verbose==True and (i % 100)==0:
            print('Iteration ' + str(i))

        relative_corr = my_correlation(signal_decorr, signal_decorr)
        
        if measure==True:
            error[i] = xp.linalg.norm(correlation-relative_corr)

        # relative_corr = 0.5*(correlation + axisflip(correlation)) / relative_corr
        relative_corr = (correlation) / relative_corr

        # avoid errors due to division by zero or inf
        relative_corr[xp.isinf(relative_corr)] = epsilon 
        relative_corr = xp.nan_to_num(relative_corr)

        # multiplicative update 
        # signal_decorr *= my_correlation(axisflip(signal_decorr), (relative_corr)) / R_0
        signal_decorr *= my_correlation((relative_corr), (signal_decorr)) / R_0
        # signal_decorr *= (my_correlation((relative_corr),(signal_decorr)) + my_convolution((signal_decorr), (relative_corr))) / R_0
        
    if clip:
        signal_decorr[signal_decorr > +1] = +1
        signal_decorr[signal_decorr < -1] = -1

    return signal_decorr, error



def anchorUpdate_MAP(signal, kernel, prior=0, iterations=10, measure=True, clip=True, verbose=False):
    xp = cp.get_array_module(signal)
    signal_deconv = signal
    signal = signal / signal.sum()
    kernel = kernel / kernel.sum()
    epsilon = 1e-7

    # starting guess with a flat image
    if prior.any()==0:
        signal_deconv = xp.full(signal.shape,0.5) + 0.01*xp.random.rand(*signal.shape)
    else:
        signal_deconv = prior #+ 0.1*prior.max()*xp.random.rand(*signal.shape)

    # to measure the distance between the guess convolved and the signal
    error = None    
    if measure == True:
        error = xp.zeros(iterations)

    for i in range(iterations):
        if verbose==True:
            print('Iteration ' + str(i))

        kernel_update = my_convolution(signal_deconv, kernel)
        kernel_update = my_correlation(kernel_update, kernel)
        kernel_update = kernel_update / kernel_update.sum()

        kernel_mirror = axisflip(kernel_update)

        relative_blur = my_convolution(signal_deconv, kernel_update)
        if measure==True:
            error[i] = xp.linalg.norm(signal-relative_blur)
        relative_blur = signal / relative_blur

        # avoid errors due to division by zero or inf
        relative_blur[xp.isinf(relative_blur)] = epsilon
        relative_blur = xp.nan_to_num(relative_blur)

        # multiplicative update 
        signal_deconv *= xp.exp(my_convolution(relative_blur - 1 , kernel_mirror))
                
    if clip:
        signal_deconv[signal_deconv > +1] = +1
        signal_deconv[signal_deconv < -1] = -1

    return signal_deconv, error


# def lucyRichardson_TV(signal, kernel, lambd, iterations=10):
#     xp = cp.get_array_module(signal)
#     u = signal 
#     # starting guess with a flat image
#     u = xp.full(signal.shape, 0.5)
#     # kernel_mirror = kernel[::-1,::-1,::-1]

#     for i in range(0,iterations):
#         # print('Iteration ' + str(i))
#         relative_blur = my_convolution(u, kernel)
#         relative_blur = signal / relative_blur
        
#         tv_ratio = (1 - divGrad(u))
        
#         u = (u / tv_ratio) * my_convolution(relative_blur, kernel[::-1,::-1,::-1])
#         # u = u * my_correlation(relative_blur, kernel)
        
#     return u    



# %% DEAUTOCONVOLUTION and DEAUTOCORRELATION
"""
Iterative implementation of the deautoconvolution and deautocorrelation protocols.

It seems to work both methods but when including the mask constraints it leads 
to wrong reconstruction results. Still investigating the reason.

N.B. It should be possible to simplify the deautocorrelation procedure by
using symmetry of the measured autocorrelation, but it is yet to be implemented.
"""
def invert_autoconvolution(magnitude, prior=None, mask=None, measure=True,
                           steps=200, mode='deautocorrelation', verbose=True):
    
    # agnostic code, xp is either numpy or cupy depending on the magnitude array module
    xp = cp.get_array_module(magnitude)

    # object support constraint
    if mask is None:
        mask = xp.ones(magnitude.shape)

    # assert magnitude.shape == mask.shape, 'mask and magnitude should have same shape'
    assert steps > 0, 'steps should be a positive number'
    assert mode == 'deautoconvolution' or mode == 'deautocorrelation',\
            'mode should be \'deautoconvolution\' or \'deautocorrelation\''

    # random phase if prior is None, otherwise start with the prior Fourier
    if prior is None:
        x_hat = 1 + 0.01*xp.random.rand(*magnitude.shape)
    else:
        x_hat = prior
        

    if measure ==True:
        ratio = xp.zeros(steps)
    else:
        ratio = None
    

    x_hat = x_hat * mask
    y_mes = 0.5*(magnitude + magnitude[::-1,::-1])
    
    # normalization for energy preservation
    y0 = (xp.sum(x_hat))**2
    # y0 = (xp.sum(x_hat))
    x_hat = xp.divide(x_hat, xp.sqrt(y0))
    
    # monitoring the convergence of the solution
    # convergence = xp.zeros(steps)

    # loop for the minimization, I guess there can be an analogue for the autocorrelation
    if mode == "deautoconvolution":
        for i in range(0, steps):
            y = my_convolution(x_hat, x_hat)
            # u_hat = y_mes / y
            # zero divided by zero is equal to zero
            u_hat = xp.divide(y_mes, y, out=xp.zeros_like(y_mes), where=y!=0)
            
            if measure==True:
                ratio[i] = u_hat.mean()
            
            # convergence[i] = xp.mean(u_hat)
            r_hat = 1/xp.sqrt(y0) * my_convolution(u_hat, x_hat)
            x_hat = x_hat * r_hat
    
    # not ready yet
    elif mode == "deautocorrelation":
        for i in range(0, steps):
            y = my_correlation(x_hat, x_hat)
            
            # if measure==True:
            #     ratio[i] = xp.linalg.norm(y_mes - y)

            u_hat = xp.divide(y_mes, y, out=xp.zeros_like(y_mes), where=y!=0)        
            
            if measure==True:
                ratio[i] = u_hat.mean()
       
            r_hat = (0.5/xp.sqrt(y0)) * ( my_correlation(x_hat, u_hat) + (my_convolution(x_hat, u_hat)) )
            # r_hat = (0.5/(y0)) * ( my_correlation(x_hat[::-1,::-1], u_hat) + my_convolution(x_hat, u_hat) )
            x_hat = x_hat * r_hat            

            # r_hat = (1/xp.sqrt(y0)) * my_correlation(x_hat[::-1,::-1], u_hat)
            # x_hat = x_hat * r_hat            

    
    return (x_hat, ratio)
    
