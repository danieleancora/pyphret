# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:44:00 2021

@author: Daniele Ancora
"""

# %% LIBRARIES CALL
import time
import numpy as np
# import scipy.ndimage
# from scipy import signal 

######### import cupy only if installed #########
from importlib import util
cupy_enabled = util.find_spec("cupy") is not None
if cupy_enabled:
    import cupy as cp
    import cupyx.scipy.ndimage
######### ----------------------------- #########

import pyphret.backend as pyb
# import pyphret.cusignal.convolution as pyconv
from pyphret.functions import my_convolution, my_correlation, my_convcorr, my_convcorr_sqfft, my_correlation_withfft, axisflip, snrIntensity_db


# %%


