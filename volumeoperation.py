# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 18:10:14 2020

@author: Daniele Ancora
"""

import time
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import pyphret.backend as pyb
import pyphret.functions as pf

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy  as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########


# %% 
def tiltVolume(volume, angle1, angle2, rotateaxes1=(0,1), rotateaxes2=(0,2)):
    
    xps = pyb.get_array_module_scipy(volume)
    
    volume_rot = xps.ndimage.rotate(volume,     angle1, axes=rotateaxes1, reshape=False)    
    volume_rot = xps.ndimage.rotate(volume_rot, angle2, axes=rotateaxes2, reshape=False)    

    return volume_rot
