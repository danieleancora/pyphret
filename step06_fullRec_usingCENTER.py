#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:52:53 2024

In this script we run a reconstruction based on the autocorrelation of the 
central SPAD element (the perfectly confocal image) with high the highest NA 
but with low signal to noise level. 
We initialize the algorithm using APR, SUM or smoothed start and we explore how
the reconstruction look like after running our optimization

@author: daniele
"""

import os
import h5py
import scipy
import cupy as cp
import numpy as np
import skimage.restoration
import matplotlib.pyplot as plt
from pyphret import functions as pyf
from pyphret import deconvolutions as pyd

import brighteyes_ism.dataio.mcs as mcs
import brighteyes_ism.analysis.FRC_lib as frc
import brighteyes_ism.analysis.APR_lib as apr
import brighteyes_ism.analysis.Tools_lib as tools

import napari

plt.close('all')


#%% load raw data
path = r'/home/daniele/Desktop/[Projects]/SPAD_deautocorr/'
datafolder = 'datasets/'
recfolder = 'reconstructions/'
filename = 'data-06-12-2022-20-45-30.h5'

data, metadata = mcs.load(path+datafolder+filename)

APRdata = np.load(path+recfolder+'time_seriesAPR.npz')
APR = APRdata['APR']


#%% Crop a central region to avoid boundaries
raw = tools.CropEdge(data, npx = 20, edges = 'ru')
raw = np.squeeze( raw )


#%% calculate physical parameters
sz = np.asarray( raw.shape )

szt = sz.copy()
szt[-2] /= 2

dx = metadata.dx * 17.303/metadata.calib_x
dt = metadata.dt


#%% do RAW temporal integration to produce the SUM
raw_odd  = raw[:, :, 1::2, :]
raw_even = raw[:, :, 0::2, :]

sum_odd = np.empty( szt )
sum_even = np.empty( szt )

for t in range( szt[-2] ):
    print( f'time = {t}' )
    sum_odd[:,:,t,:] = np.sum(raw_odd[...,:t+1,:], axis = -2)
    sum_even[:,:,t,:] = np.sum(raw_even[...,:t+1,:], axis = -2)        

SUM = []

# integrating over the SPAD arrays
SUM.append( np.sum(sum_odd, axis = -1) )
SUM.append( np.sum(sum_even, axis = -1) )


# %% calculate autocorrelation
acorr_odd = np.zeros_like(raw_odd)
acorr_even = np.zeros_like(raw_even)

for t in range(acorr_odd.shape[-2]):
    print( f'time = {t}' )
    for i in range(acorr_odd.shape[-1]):
        acorr_odd[:,:,t,i] = pyf.my_autocorrelation(raw_odd[:,:,t,i])
        acorr_even[:,:,t,i] = pyf.my_autocorrelation(raw_even[:,:,t,i])


acorr_sum_odd = np.empty( szt )
acorr_sum_even = np.empty( szt )

for t in range( szt[-2] ):
    print( f'time = {t}' )
    acorr_sum_odd[:,:,t,:] = np.sum(acorr_odd[...,:t+1,:], axis = -2)
    acorr_sum_even[:,:,t,:] = np.sum(acorr_even[...,:t+1,:], axis = -2)        


# integrating over the SPAD arrays
ACORR = []
ACORR.append(np.squeeze(acorr_sum_odd[:,:,:,12]))
ACORR.append(np.squeeze(acorr_sum_even[:,:,:,12]))


#%% masking the autocorrelation peak in the center
mask = np.zeros((730,730))
mask[365,365] = 1


# %% Parameters set
attempts = 50
iterations = 1000
tot_iterations = np.arange(iterations, attempts*iterations+1, iterations)

ss_odd_store = np.zeros_like(ACORR[0])
ss_even_store = np.zeros_like(ACORR[1])
iterations_store = np.zeros(ACORR[0].shape[-1])
FRC_store = np.zeros(ACORR[0].shape[-1])
FRCmap_store = np.zeros((ACORR[0].shape[-1], attempts))


# %% Tune the FRC analysis - Starting with APR
# for t in range(0,ACORR[0].shape[-1]):
    t = 1

    print('Processing dwell time ', t)
    
    # set storing variable for each dwell time
    ss_odd = np.zeros((730,730,attempts))
    ss_even = np.zeros((730,730,attempts))
    resolution_priorAPR = np.zeros((attempts,))
    
    # pick correct starting points
    acorr_odd = ACORR[0][:,:,t]
    acorr_odd = skimage.restoration.inpaint_biharmonic(acorr_odd, mask)
    prior_odd = APR[0][:,:,t].squeeze()
    prior_odd = scipy.ndimage.gaussian_filter(APR[0][:,:,t].squeeze(), .4) 
    # prior_odd = scipy.ndimage.gaussian_filter(SUM[0][:,:,t].squeeze(), .4) 

    
    acorr_even = ACORR[1][:,:,t]
    acorr_even = skimage.restoration.inpaint_biharmonic(acorr_even, mask)
    prior_even = APR[1][:,:,t].squeeze()
    prior_even = scipy.ndimage.gaussian_filter(APR[1][:,:,t].squeeze(), .4) 
    # prior_even = scipy.ndimage.gaussian_filter(SUM[1][:,:,t].squeeze(), .4) 


    # Calculate FRC value of the APR reconstruction 
    frc_results = frc.FRC_resolution(prior_odd, prior_even, px=dx)
    print('Starting FRC ', frc_results[0])

    # Move everything to GPU
    acorr_odd = cp.asarray(acorr_odd)
    acorr_even = cp.asarray(acorr_even)
    prior_odd = cp.asarray(prior_odd)
    prior_even = cp.asarray(prior_even)
    
    # iteration monitoring FRC results
    for i in range(attempts):
        print('\nTested iteration ', iterations*i, "dwell time ", t)
        
        # Reconstruction on the ODD part
        prior_odd, error0 = pyd.schulzSnyder(acorr_odd, prior=prior_odd, iterations=iterations, precision='float32', measure=False, clip=False, verbose=False)
        ss_odd[:,:,i] = prior_odd.get()
        
        # Reconstruction on the EVEN part
        prior_even, error0 = pyd.schulzSnyder(acorr_even, prior=prior_even, iterations=iterations, precision='float32', measure=False, clip=False, verbose=False)
        ss_even[:,:,i] = prior_even.get()

        # Calculate FRC at a given iteration number
        frc_results = frc.FRC_resolution(prior_odd.get(), prior_even.get(), px=dx)
        resolution_priorAPR[i] = frc_results[0]
        FRCmap_store[t,i] = resolution_priorAPR[i]
    
        print('FRC = ', resolution_priorAPR[i])
        
    # Pick the optimal reconstructions based on FRC value
    bestidx = np.argmin(resolution_priorAPR)
    ss_odd_store[:,:,t] = ss_odd[:,:,bestidx]
    ss_even_store[:,:,t] = ss_even[:,:,bestidx]
    iterations_store[t] = tot_iterations[bestidx]
    FRC_store[t] = resolution_priorAPR[bestidx]
    
    
    
# %%
plt.plot(resolution_priorAPR)
t=1

cmap='magma'

viewer = napari.Viewer()
center_layer = viewer.add_image(sum_odd[:,:,t,12], colormap=cmap)
center_layer = viewer.add_image(sum_odd[:,:,-1,12], colormap=cmap)
# aco_layer = viewer.add_image(ss_odd_store[:,:,t], colormap=cmap)
aco_layer = viewer.add_image(ss_odd[:,:,4], colormap=cmap)
apr_layer = viewer.add_image(APR[0][:,:,t], colormap=cmap)
sum_layer = viewer.add_image(SUM[0][:,:,t], colormap=cmap)


no = viewer.add_image(ss_odd/ss_odd.mean(axis=(0,1),keepdims=True), colormap=cmap)

# %% Save the data
SS = []
SS.append(ss_odd_store)
SS.append(ss_even_store)

np.savez_compressed(path+recfolder+'time_seriesSS_optimal_APRstart_usingCENTER.npz', SS = SS, dt = dt, dx = dx, iterations=iterations_store, FRC=FRC_store, FRCmap=FRCmap_store)

