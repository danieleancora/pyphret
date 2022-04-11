# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:22:32 2021

ISOPLANATIC DECONVOLUTION:
With this demo, we use the depth-wise deconvolution routine to perform an 
isoplanatic deconvolution of the image. This is not the correct usage, since it 
takes the central PSF and uses it to deconvolve over the entire image plane.


@author: Daniele Ancora
"""

import torch
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

import pyphret.torch.restoration as pytr
import pyphret.torch.tiling as pytt
import pyphret.torch.functional as pytf


# %% IMPORT the files ABERRATED FOLDER
number = 9
croppete = 10
psfmapname = str(number) + 'x' + str(number)
folder = psfmapname + '//'

aberration_high = 'Aberrated//'

filename1 = folder + aberration_high + 'psf_all' + psfmapname + '.tif'
filename2 = folder + aberration_high + 'Snap_wide.tiff'
filename3 = folder + aberration_high + 'dark.tiff'

# image that we have to deconvolve, with a slight padding to fix shapes
imag1 = tiff.imread(filename2).astype('float32')
imag1 = np.pad(imag1, (1,1))

# load the psf mapped on a grid. IT HAS TO BE A STACK with ODD DIMENSIONS, then we reshape it
psfmap = tiff.imread(filename1).astype('float32')
psfmap = psfmap.reshape(number,number,psfmap.shape[1],psfmap.shape[2])

# crop the psfmap further to fit within less memory, ideally croppete=0
psfmap = psfmap[:,:,croppete:-croppete,croppete:-croppete]

# visualize the PSF map
index = 0
plt.figure()
for i in range(number):
    for j in range(number):
        index += 1;
        plt.subplot(number,number,index)
        plt.imshow(psfmap[i,j,:,:])
        plt.axis('off')
plt.tight_layout()

# backgroung image acquisition
dark = tiff.imread(filename3).astype('float32')
dark = np.pad(dark, (1,1))

# we subtract the background
imag1 = imag1 - dark
imag1 /= imag1.mean()


# %% SET UP THE STAGE
# check if gpu is available, otherwise run it on the CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('\n Using device:', device, '\n') 

# cast to float and send to the device (the GPU) 
imag2 = pytt.adddimension(imag1)
psf2 = pytt.adddimension(psfmap[4,4,:,:])

im = torch.from_numpy(imag2).float().to(device = device)
psf = torch.from_numpy(psf2).float().to(device = device)

blur = pytf.xcorrDepthwise(im, psf)
blur = blur.cpu().numpy().squeeze()


# %% PARAMETER DEFINITION
imag2 = pytt.adddimension(imag1)
psf2 = pytt.adddimension(psfmap[4,4,:,:])
im = torch.from_numpy(imag2).float().to(device = device)
psf = torch.from_numpy(psf2).float().to(device = device)

# this is the call for the deconvolution function
deconv = pytr.deconvolutionRL(im, psf, deconv=None, iterations=20, verbose=True)

# plotting the results
plt.figure()
plt.subplot(121), plt.imshow(blur), plt.title('Weakly aberrated measurement'), plt.axis('off')
plt.subplot(122), plt.imshow(deconv.cpu().numpy().squeeze(), vmax=3), plt.title('Anisoplanatic deconvolution'), plt.axis('off')
plt.tight_layout()


# tiff.imsave(folder + 'notaberrated_deconvol_isoplanatic.tif', np.float32(deconv.cpu().numpy().squeeze()))
