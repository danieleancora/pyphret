# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 15:21:55 2021

ANISOPLANATIC DECONVOLUTION:
This is the correct usage of the algorithm. It takes a map of PSF measured in 
different regions of the image plane and uses them to deconvolve locally. The 
local tiles in the image plane are choosen to overlap one to another, to avoid
artifacts at the boundaries. Once the tiles are deconvolved, we use them to 
rebuild the entire deconvolved image.
    

@author: Daniele Ancora
"""

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

import pyphret.torch.restoration as pytr
import pyphret.torch.tiling as pytt


# %% IMPORT the files from STRONG ABERRATION EXPERIMENT
number = 9
croppete = 10
psfmapname = str(number) + 'x' + str(number)
folder = psfmapname + '//'

aberration_high = 'Aberrated//'

filename1 = folder + aberration_high + 'psf_all' + psfmapname + '.tif'
filename2 = folder + aberration_high + 'Snap_wide.tiff'
filename3 = folder + aberration_high + 'dark.tiff'

# image that we have to deconvolve, slight padding
imag1 = tiff.imread(filename2).astype('float32')
imag1 = np.pad(imag1, (1,1))

# load the psf mapped on a grid. IT HAS TO BE A STACK with ODD DIMENSIONS, then we reshape it
psfmap = tiff.imread(filename1).astype('float32')
psfmap = psfmap.reshape(number,number,psfmap.shape[1],psfmap.shape[2])

# crop the psfmap further to fit within less memory, ideally croppete=0
psfmap = psfmap[:,:,croppete:-croppete,croppete:-croppete]

# visualize the PSF map
index = 0
plt.figure(1)
for i in range(number):
    for j in range(number):
        index += 1;
        plt.subplot(number,number,index)
        plt.imshow(psfmap[i,j,:,:])
        plt.axis('off')
plt.tight_layout()

# backgroung acquisition
dark = tiff.imread(filename3).astype('float32')
dark = np.pad(dark, (1,1))

# we subtract the background
imag1 = imag1 - dark
imag1 /= imag1.mean()


# %% PARAMETER DEFINITION
deconv1 = pytr.varyingDeconvRL(imag1,           # the actual image to be deconvolved
                              psfmap,           # the psf map over the image plane
                              iterations=200,   # the number of iterations to be carried out, do not exceed 
                              windowSize=410,   # it has to divede the size of the image
                              windowStep=82,    # same as above
                              windowOverlap=32, # crop out a region on the canvas of each tile
                              mode='RL'         # RL stands for Richardson Lucy deconvolution
                              )


# %% IMPORT the files from LOW ABERRATION MEASUREMENT
aberration_low  = 'Not Aberrated//'

filename1 = folder + aberration_low + 'psf_all' + psfmapname + '.tif'
filename2 = folder + aberration_low + 'Snap_wide.tiff'
filename3 = folder + aberration_low + 'dark.tiff'

# image that we have to deconvolve, slight padding
imag2 = tiff.imread(filename2).astype('float32')
imag2 = np.pad(imag2, (1,1))

# load the psf mapped on a grid. IT HAS TO BE A STACK with ODD DIMENSIONS, then we reshape it
psfmap = tiff.imread(filename1).astype('float32')
psfmap = psfmap.reshape(number,number,psfmap.shape[1],psfmap.shape[2])

# crop the psfmap further to fit within less memory, ideally croppete=0
psfmap = psfmap[:,:,croppete:-croppete,croppete:-croppete]
tiff.imshow(psfmap)

# backgroung acquisition
dark = tiff.imread(filename3).astype('float32')
dark = np.pad(dark, (1,1))

# we subtract the background
imag2 = imag2 - dark
imag2 /= imag2.mean()


# %% PARAMETER DEFINITION
deconv2 = pytr.varyingDeconvRL(imag2, 
                              psfmap, 
                              iterations=200, 
                              windowSize=410,   # it has to divide the size of the image
                              windowStep=82,    # same as above
                              windowOverlap=32, # crop out a region on the canvas of each tile
                              mode='RL'
                              )


# %% VISUALIZE THE RESULTS
plt.figure(1)
plt.subplot(121), plt.imshow(imag1)
plt.subplot(122), plt.imshow(deconv1)

plt.figure(2)
plt.subplot(121), plt.imshow(imag2)
plt.subplot(122), plt.imshow(deconv2)



# tiff.imshow(np.stack([[imag1, deconv1],[imag2, deconv2]], axis=0), cmap='hot')


# tiff.imsave(folder + 'aberrated_original.tif', np.float32(imag1))
# tiff.imsave(folder + 'aberrated_deconvol.tif', np.float32(deconv1))

# tiff.imsave(folder + 'notaberrated_original.tif', np.float32(imag2))
# tiff.imsave(folder + 'notaberrated_deconvol.tif', np.float32(deconv2))




# psfmapspace = pytt.untile(psfmap, windowSize=53,windowStep=53,windowOverlap=1)
# tiff.imsave(folder + 'psfmapspace.tif', np.float32(psfmapspace))

# imagtiled = pytt.tile(imag1[25:-25,25:-25],   windowSize=400, windowStep=400, windowOverlap=0)
# decotiled = pytt.tile(deconv1[25:-25,25:-25], windowSize=400, windowStep=400, windowOverlap=0)



# tiff.imshow(deconv1)

# plt.imshow(psfmapspace)


# import matplotlib
# for i in range(5):
#     matplotlib.image.imsave(folder + 'imagetiled' + str(i) + '.png', imagtiled[i,i,:,:], cmap='magma')
#     matplotlib.image.imsave(folder + 'decotiled' + str(i) + '.png', decotiled[i,i,:,:], cmap='magma')
#     matplotlib.image.imsave(folder + 'psfmap5x5' + str(i) + '.png', psfmap[i,i,:,:], cmap='magma')

#     tiff.imsave(folder + 'imagetiled' + str(i) + '.tif', np.float32(imagtiled[i,i,:,:]))
#     tiff.imsave(folder + 'decotiled' + str(i) + '.tif', np.float32(decotiled[i,i,:,:]))
#     tiff.imsave(folder + 'psfmap5x5' + str(i) + '.tif', np.float32(psfmap[i,i,:,:]))









# plt.figure(2)
# plt.imshow(np.abs(imag - deconv))

# tiff.imsave(export_original, imag)
# tiff.imsave(export_deconvol, np.float32(deconv))

# imtile = pytt.tile(imag, windowSize=410, windowStep=410, windowOverlap=32)
# imtile = imtile.reshape((5*5,410,410))

# deconv = pytt.tile(deconv, windowSize=410, windowStep=410, windowOverlap=32)
# deconv = deconv.reshape((5*5,410,410))

# psfmap = psfmap.reshape((5*5,37,37))
# psfmap = psfmap.reshape((5,5,37,37))

# psfmapspace = pytt.untile(psfmap, windowSize=37,windowStep=37,windowOverlap=1)

# fig = plt.figure(1)
# tiff.imshow(imtile, figure=fig, subplot=121)
# tiff.imshow(deconv, figure=fig, subplot=122)

# tiff.imsave('imtile.tif', imtile)
# tiff.imsave('deconv.tif', np.float32(deconv))
# tiff.imsave('psfmap.tif', psfmap)
# tiff.imsave('psfmapspace.tif', np.float32(psfmapspace))


