#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:51:59 2024

@author: daniele
"""

import pyphret.functions as pyf
import matplotlib.pyplot as plt 
import numpy as np
import skimage.data as data
import matisse.generate

psf = pyf.gaussian_psf(size=[256,256], alpha=[3,3])


brain = data.brain()[5,:,:]
brain = matisse.generate.randomtranscript(channels=1, density=0.001).squeeze()

brainblur = pyf.my_convolution(brain, psf)


# %% SUM projections (for purely absolute values is L1-norm)

psf_sumproj = psf.sum(axis=0)
brain_sumproj = brain.sum(axis=0)

conv_sumproj = pyf.my_convolution(psf_sumproj, brain_sumproj)

brainblur_sumproj = brainblur.sum(axis=0)


# %% L-norm projection
p = 10
psf_l2proj = ((psf**p).sum(axis=0))**(1/p)
brain_l2proj = ((brain**p).sum(axis=0))**(1/p)

brainblur_l2proj = ((brainblur**p).sum(axis=0))**(1/p)

conv_l2proj = pyf.my_convolution(psf_l2proj, brain_l2proj)


# %% MAX projections
psf_maxproj = psf.max(axis=0)
brain_maxproj = brain.max(axis=0)

conv_maxproj = pyf.my_convolution(psf_maxproj, brain_maxproj)

brainblur_maxproj = brainblur.max(axis=0)


plt.figure()
plt.subplot(141)
plt.imshow(brainblur)

plt.subplot(142)
plt.plot(conv_sumproj)
plt.plot(brainblur_sumproj,'.')

plt.subplot(143)
plt.plot(conv_l2proj/conv_l2proj.sum())
plt.plot(brainblur_l2proj/brainblur_l2proj.sum(),':')

plt.subplot(144)
plt.plot(conv_maxproj/conv_maxproj.sum())
plt.plot(brainblur_maxproj/brainblur_maxproj.sum(),':')







