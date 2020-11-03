# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:48:44 2020

@author: Daniele Ancora
"""

# import cupy  as cp
import numpy as np
from importlib import util
import cupyx.scipy
import scipy 


# taken from sigpy
# https://github.com/mikgroup/sigpy/blob/master/sigpy/config.py
cupy_enabled = util.find_spec("cupy") is not None

if cupy_enabled:
    import cupy  as cp


# this function is taken from SIGPY package from the link
# https://github.com/mikgroup/sigpy/blob/5bd25cdfda5b72c2728993ad5e6f7288f274ddc4/sigpy/backend.py
def get_array_module(array):
    """Gets an appropriate module from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module` and here it is 
    ment to replace it. The difference is that this function can be used even 
    if cupy is not available.

    Args:
        array: Input array.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on input.
    """
    if cupy_enabled:
        return cp.get_array_module(array)
    else:
        return np


def get_array_module_scipy(array):
    """Gets an appropriate module from :mod:`numpy` or :mod:`cupy`.

    This is almost equivalent to :func:`cupy.get_array_module` and here it is 
    ment to replace it. The difference is that this function can be used even 
    if cupy is not available.

    Args:
        array: Input array.

    Returns:
        module: :mod:`cupy` or :mod:`numpy` is returned based on input.
    """
    if cupy_enabled:
        return cupyx.scipy.get_array_module(array)
    else:
        return scipy






# this is taken from the CUPY package
# def my_get_array_module(*args):
#     """Returns the array module for arguments.
#     This function is used to implement CPU/GPU generic code. If at least one of
#     the arguments is a :class:`cupy.ndarray` object, the :mod:`cupy` module is
#     returned.
#     Args:
#         args: Values to determine whether NumPy or CuPy should be used.
#     Returns:
#         module: :mod:`cupy` or :mod:`numpy` is returned based on the types of
#         the arguments.
#     .. admonition:: Example
#        A NumPy/CuPy generic function can be written as follows
#        >>> def softplus(x):
#        ...     xp = cupy.get_array_module(x)
#        ...     return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))
#     """
#     for arg in args:
#         if isinstance(arg, (ndarray, _cupyx.scipy.sparse.spmatrix,
#                             cupy.core.fusion._FusionVarArray,
#                             cupy.core.new_fusion._ArrayProxy)):
#             return _cupy
#     return numpy


