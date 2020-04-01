# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:13:31 2020

@author: Daniele Ancora
"""

# %% LIBRARIES CALL
import cupy as cp
import numpy as np
from pyphret.functions import my_convolution, my_correlation
import cupyx.scipy
from cupyx.scipy import ndimage
import scipy


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


def lucyRichardson_deconvolution(signal, kernel, iterations=10):
    xp = cp.get_array_module(signal)
    u = signal 
    # starting guess with a flat image
    u = xp.full(signal.shape, 0.5)
    # kernel_mirror = kernel[::-1,::-1,::-1]

    for i in range(0,iterations):
        # print('Iteration ' + str(i))
        relative_blur = my_convolution(u, kernel)
        relative_blur = signal / relative_blur
        u = u * my_convolution(relative_blur, kernel[::-1,::-1,::-1])
        # u = u * my_correlation(relative_blur, kernel)
        
    return u    
    

# I could think on using this functions to use smaller kernels 
# https://docs-cupy.chainer.org/en/stable/reference/generated/cupyx.scipy.ndimage.convolve.html#cupyx.scipy.ndimage.convolve
# https://github.com/scikit-image/scikit-image/blob/master/skimage/restoration/deconvolution.py#L140
def lucyRichardson_deconvolutionSmall(image, psf, iterations=10, clip=False):
    print('This procedure may be very slow!')
    xp = cp.get_array_module(image)
    xps = cupyx.scipy.get_array_module(image)

    image = image.astype(xp.float)
    psf = psf.astype(xp.float)
    im_deconv = xp.full(image.shape, 0.5)
    # psf_mirror = xp.flip(psf, axis=(0,1,2))
    psf_mirror = psf[::-1,::-1,::-1]
    
    for i in range(iterations):
        # print('Iteration ' + str(i))
        relative_blur = xps.ndimage.convolve(im_deconv, psf)
        relative_blur = image / relative_blur
        im_deconv *= xps.ndimage.convolve(relative_blur, psf_mirror)

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv



# %% DEAUTOCONVOLUTION and DEAUTOCORRELATION
"""
Iterative implementation of the deautoconvolution and deautocorrelation protocols.

It seems to work both methods but when including the mask constraints it leads 
to wrong reconstruction results. Still investigating the reason.

N.B. It should be possible to simplify the deautocorrelation procedure by
using symmetry of the measured autocorrelation, but it is yet to be implemented.
"""
def invert_autoconvolution(magnitude, prior=None, mask=None,  
                           steps=200, mode='deautoconvolution', verbose=True):
    
    # agnostic code, xp is either numpy or cupy depending on the magnitude array module
    xp = cp.get_array_module(magnitude)

    # object support constraint
    if mask is None:
        mask = xp.ones(magnitude.shape)

#    assert magnitude.shape == mask.shape, 'mask and magnitude should have same shape'
    assert steps > 0, 'steps should be a positive number'
    assert mode == 'deautoconvolution' or mode == 'deautocorrelation',\
    'mode should be \'deautoconvolution\' or \'deautocorrelation\''

    # random phase if prior is None, otherwise start with the prior Fourier
    if prior is None:
        x_hat = 1 + 0.01*xp.random.rand(*magnitude.shape)
    else:
        x_hat = prior

    x_hat = x_hat * mask
    y_mes = magnitude
    
    # normalization for energy preservation
    y0 = (xp.sum(x_hat))**2
    x_hat = xp.divide(x_hat, xp.sqrt(y0))
    
    # monitoring the convergence of the solution
    # convergence = xp.zeros(steps)

    # loop for the minimization, I guess there can be an analogue for the autocorrelation
    if mode == "deautoconvolution":
        for i in range(1, steps+1):
            y = my_convolution(x_hat, x_hat)
            # u_hat = y_mes / y
            # zero divided by zero is equal to zero
            u_hat = xp.divide(y_mes, y, out=xp.zeros_like(y_mes), where=y!=0)        
            # convergence[i] = xp.mean(u_hat)
            r_hat = 1/xp.sqrt(y0) * my_convolution(u_hat, x_hat)
            x_hat = x_hat * r_hat
    
    # not ready yet
    elif mode == "deautocorrelation":
        for i in range(1, steps+1):
            y = my_correlation(x_hat, x_hat)
            u_hat = xp.divide(y_mes, y, out=xp.zeros_like(y_mes), where=y!=0)        
            
            r_hat = (0.5/xp.sqrt(y0)) * ( my_correlation(x_hat, u_hat) + my_convolution(x_hat, u_hat) )
            x_hat = x_hat * r_hat            
    
    return x_hat
    
