# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:27:08 2021

@author: Daniele Ancora
"""

import time
import torch
import numpy as np
from scipy import ndimage
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage.util import view_as_windows

import torch.nn.functional as F
import matplotlib.pyplot as plt

import pyphret.torch.restoration as pytr
import pyphret.torch.tiling as pytt


# %% IMPORT the files
folder = 'materiale_deconvoluzione01//'
filename1 = folder + 'psf_all5x5.tif'
filename2 = folder + 'Snap_wide.tiff'
filename3 = folder + 'dark.tiff'

# image that we have to deconvolve, slight padding
imag = tiff.imread(filename2).astype('float32')
imag = np.pad(imag, (1,1))

# load the psf mapped on a grid. IT HAS TO BE A STACK with ODD DIMENSIONS, then we reshape it
psfmap = tiff.imread(filename1).astype('float32')
psfmap = psfmap.reshape(5,5,psfmap.shape[1],psfmap.shape[2])
psfmap = psfmap[:,:,:-1,:-1]

# crop the psfmap further to fit within less memory, ideally croppete=0
croppete = 25
psfmap = psfmap[:,:,croppete:-croppete,croppete:-croppete]
tiff.imshow(psfmap)

# backgroung acquisition
dark = tiff.imread(filename3).astype('float32')
dark = np.pad(dark, (1,1))

# we subtract the background
imag = imag - dark
imag /= imag.mean()


# %% PARAMETER DEFINITION
deconv = pytr.varyingDeconvRL(imag, 
                              psfmap, 
                              iterations=30, 
                              windowSize=410,  # it has to divede the size of the image
                              windowStep=82,   # same as above
                              windowOverlap=32, # crop out a region on the canvas of each tile
                              mode='RL'
                              )


# %% VISUALIZE THE RESULTS
plt.figure(1)
plt.subplot(121), plt.imshow(imag)
plt.subplot(122), plt.imshow(deconv)

tiff.imshow(np.stack([imag, deconv], axis=0), cmap='hot')











