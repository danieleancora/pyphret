# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:16:59 2021

@author: Daniele Ancora
"""


# %%
import time
import torch
import numpy as np
import torch.nn.functional as F



# %% FUNCTION DEFINITION
def xcorrDepthwise(signal, kernel):
    """
    Function capable of performing 2D convolution along the last two axis of a 
    3D stack depthwise.

    Parameters
    ----------
    signal : torch 3D Tensor
        the stack of 2D images that we want to convolve depthwise.
    kernel : torch 3D Tensor
        depthwise filter. It has to have odd shape.

    Returns
    -------
    conv : torch 3D Tensor
        each plane of the signal convolved with each correspondent filter.

    """
    depth = kernel.size(0)
    
    signal = signal.view(1, depth, signal.size(1), signal.size(2))
    kernel = kernel.view(depth, 1, kernel.size(1), kernel.size(2))

    conv = F.conv2d(signal, 
                    kernel, 
                    bias=None, 
                    stride=1,
                    padding=(kernel.size(2)//2, kernel.size(3)//2),
                    groups=depth
                    )
    
    conv = conv.view(depth, conv.size(2), conv.size(3)).detach()
    return conv


def convDepthwise(signal, kernel):
    return xcorrDepthwise(signal, kernel.flip([1, 2]))






