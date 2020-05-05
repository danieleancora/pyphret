# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:04:08 2020

@author: Daniele Ancora
"""

# LIBRARIES CALL
import time
import cupy as cp
import numpy as np
import pyphret.functions as pf


def randomGaussianBlobs_2D(dimension=512, fraction=0.999):
    size = int(dimension/2)
    size_pad = int(dimension/4)

    test_image = np.random.rand(size,size)
    test_image = (test_image > fraction).astype(np.float32) * pf.circular_mask2D(size, size)
    test_image = np.pad(test_image, (size_pad,size_pad), 'constant')
    test_image = pf.my_gaussblur(test_image,4)

    return test_image


def randomGaussianBlobs_3D(dimension=512, fraction=0.999):
    size = int(dimension/2)
    size_pad = int(dimension/4)

    test_image = np.random.rand(size,size,size)
    test_image = (test_image > fraction).astype(np.float32) * pf.spherical_mask3D(size, size, size)
    test_image = np.pad(test_image, (size_pad,size_pad), 'constant')
    test_image = pf.my_gaussblur(test_image,4)

    return test_image
