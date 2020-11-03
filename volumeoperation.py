# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:10:14 2020

@author: Daniele Ancora
"""

import time
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import pyphret.backend as pyb
import pyphret.functions as pf

import cupyx.scipy.ndimage
import scipy.ndimage



def tiltVolume(volume, angle1, angle2, rotateaxes1=(0,1), rotateaxes2=(0,2)):
    
    xps = pyb.get_array_module_scipy(volume)
    
    volume_rot = xps.ndimage.rotate(volume,     angle1, axes=rotateaxes1, reshape=False)    
    volume_rot = xps.ndimage.rotate(volume_rot, angle2, axes=rotateaxes2, reshape=False)    

    return volume_rot
