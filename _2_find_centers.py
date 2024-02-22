# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:40:02 2021

@author: Kevin
"""
import os

import h5py

import numpy as np

from scipy.ndimage.filters import median_filter

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Ellipse

from quagmire.fit import fit2d as f2d
from quagmire.utils.image_mask import circular_mask

import _0_common as com

SAVE_DATA = True

###############################################################################
# Load full-frame atom-per-pixel data from h5 
###############################################################################

DATADIR = 'E:\\Persistent Currents\\insitu_ring_images\\2020_07_09_insitu_ring_images'
os.chdir(DATADIR)
h5f = h5py.File(DATADIR + "\\processed\\insitu_ring_processed.h5", 'a')

atoms_pix = h5f["atom_per_pix"]
shot_ids = atoms_pix.attrs.get('shot_ids')

atoms_avg = h5f['atom_per_pix_avg']
bfield_mT = atoms_avg.attrs.get('bfield_mT')

##############################################################################
# Check variation of ring and halo center in each average image by 2D fit
##############################################################################

#######################################################
# Find the ring center in each group-averaged image using a parabolic ring fit
ring_guess = com.parab2d_ring_guess

ringfits = []
for i in range(4):
    ringfits.append(f2d.fit2D(atoms_avg[i], f2d.parab_ring, ring_guess))
    
ring_centers = np.array([rf[0:2] for rf in ringfits])

ring_center = np.average(np.array(ring_centers), axis=0)

shifts = [ rc - ring_center for rc in ring_centers]
rms_shift = np.sqrt(np.average(np.array(shifts)**2, axis=0))
# The rms shifts from (231, 297) are (0.2, 0.4) pixels. Pretty negligible!

# Create masks forselecting/omitting the ring region for each average image
ringmasks = []
for xc, yc in ring_centers:
    mask_s = circular_mask(radius = 23, array_shape = (512, 512), center=(yc, xc), inside_is=False)
    mask_b = ~circular_mask(radius = 36, array_shape = (512, 512), center=(yc, xc), inside_is=False)
    ringmasks.append( mask_s*mask_b )

#######################################################
# de-weighting the data in the ring region, fit the halo with a 2D gaussian

halo_guess = com.gauss2d_halo_guess

halofits = []
for i in range(4):
    halofits.append(f2d.fit2D(atoms_avg[i], f2d.gaussian2D, halo_guess, ~ringmasks[i]))

halo_centers = np.array([hf[0:2] for hf in halofits])

# Generate a figure showing how much the halo shifts in each of the images
plt.figure(21, clear=True, figsize=(8,2), dpi=200)
for i in range(4):
    plt.subplot(1,4,i+1)
    filtimg = median_filter(atoms_avg[i], 5)
    plt.imshow(filtimg * ~ringmasks[i])
    ax = plt.gca()
    ax.add_patch(Circle(ring_centers[i], 2, color='red'))
    ax.add_patch(Circle(halo_centers[i], 2, color='blue'))
    ax.add_patch(Ellipse(halo_centers[i], halofits[i][2], halofits[i][3], halofits[i][6], color='blue', fill=None))
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
 
plt.savefig("results//halo_shifts")
    
# The halo shifts are NOT negligible!

if SAVE_DATA:
    # create h5 file for processed images and info extracted from shotfiles
    with h5py.File(com.DATADIR + "\\processed\\insitu_ring_processed.h5", 'a') as h5f:
       
        com.to_h5(h5f, 'ring_centers', ring_centers)   
        com.to_h5(h5f, 'halo_centers', halo_centers)          


