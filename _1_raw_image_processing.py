# -*- coding: utf-8 -*-
"""
This script is for performing first-stage processing of in-situ ring images
from 2020-07-09.
"""
SAVE_DATA = False

"""
If true, saves the following:
processed/od_imgs.npy
processed/od_avg.npy
    
processed/atom_per_pix.npy
processed/atom_per_pix_avg.npy
    
processed/insitu_ring_processed.h5
    /od_imgs
    /od_grouped
    /od_avg    
    /atom_per_pix
    /atom_per_pix_avg
"""
SAVE_FIGURES = True
"""
If true, saves the following:
    
results//atom_image_background_check.png    
results//atomnumber_plot
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d as gfilt1d
import quagmire.analysis.atomnumber as anum

# Import paramaters and utility functions from the common module
import _0_common as com

# Load raw data from file
shots = com.load_shotlist()
shot_ids = [s.id for s in shots]
bcs_field_global = [i.shotfile['globals'].attrs['BCS_field'] for i in shots]
abs_sets = [shot.get_absorption_set() for shot in shots]

##############################################################################
# Processing of raw data from individual shots to OD omages
##############################################################################

print("Correcting offsets and normalizing raw absorption data")
corr_flat_sets = [com.correct_raw_imgdata(aset) for aset in abs_sets]

print("Converting raw absorption image data to optical depth")
od_imgs = [com.corr_flats_to_od(cfs) for cfs in corr_flat_sets]

od_grouped = com.group(od_imgs)

print("Computing average of optical depths for each group of shots")
od_then_groupav = [np.mean(odg, axis=0) for odg in od_grouped]

##############################################################################
# Processing of group-averaged raw data to OD images for each group
##############################################################################

corr_flat_sets_grouped = com.group(corr_flat_sets)

print("Computing average of the raw data for each group and calculating ODs")
corr_flat_sets_groupav = []
for group in corr_flat_sets_grouped:       
    avatom = np.mean([cfs[0] for cfs in group], axis=0)
    avprob = np.mean([cfs[1] for cfs in group], axis=0)
    avdark = np.mean([cfs[2] for cfs in group], axis=0)
    cfsga = [np.ma.masked_array(data, fill_value=0) for data in [avatom,avprob,avdark]]
    corr_flat_sets_groupav.append(cfsga)

groupav_before_od = []
for cfsga in corr_flat_sets_groupav:
    final_norm = com._calc_probe_norm(cfsga[0], cfsga[1])
    groupav_before_od.append(com.corr_flats_to_od(cfsga, final_norm))

##############################################################################
# Display comparison of results of averaging OD from each shot in a group 
# against the result of averaging the raw data before computing the OD
##############################################################################
    
plt.figure(11, clear=True, figsize=(8,2), dpi=200)
for n in range(4):
    plt.subplot(1,4,n+1)
    imdata = od_then_groupav[n]
    plt.imshow(imdata, cmap='jet', vmin=0, vmax = 4,  interpolation=None)
    offset = np.average(com.av_bacg_select * imdata)
    plt.text(50,50, "Calc OD then average", color='white', fontsize=6)
    plt.text(50,100,"Background OD: %.4f" % offset, color='white', fontsize=6)
    plt.text(50,150,"Peak OD: %.2f" % np.max(imdata), color='white', fontsize=6)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
    
plt.figure(12, clear=True, figsize=(8,2), dpi=200)
for n in range(4):    
    plt.subplot(1,4,n+1)
    imdata = groupav_before_od[n]
    plt.imshow(imdata, cmap='jet', vmin=0, vmax = 4, interpolation=None)
    offset = np.average(com.av_bacg_select * imdata)
    plt.text(50,50, "Average flats then calc OD", color='white', fontsize=6)
    plt.text(50,100,"Background OD: %.4f" % offset, color='white', fontsize=6)
    plt.text(50,150,"Peak OD: %.2f" % np.max(imdata), color='white', fontsize=6)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()

if SAVE_FIGURES:
    plt.savefig("results//atom_image_background_check")


##############################################################################
# Convert from optical depth to atoms per pixel and save to files      
        
atom_per_pix = []
for odimg in od_imgs:
    atom_per_pix.append(anum.od_to_atoms( odimg, com.pixsize, com.sigma_eff, com.mag_factor))
atom_per_pix_grouped = com.group(atom_per_pix)

atom_per_pix_avg = []
for od_avg in groupav_before_od:
    atom_per_pix_avg.append(anum.od_to_atoms( od_avg, com.pixsize, com.sigma_eff, com.mag_factor))

##############################################################################
# Validate probe normalization / background elimination
##############################################################################

n2d_av_r = []
for image in atom_per_pix_avg:
    n2d_av_r.append(com.circleav(image, (com.av_xc, com.av_yc)))
    
plt.figure(13, clear=True, figsize=(5,4), dpi=200)
fsz = 9
plt.cla()
"""
clist = ['blue', 'green', 'purple', 'red']
for n, n2d in enumerate(n2d_av_r):
    plt.plot(n2d[0],n2d[1], color=clist[n], linewidth = 1)
"""
clist = ['blue', 'green', 'purple', 'red']
for n, n2d in enumerate(n2d_av_r):
    plt.plot(n2d[0], gfilt1d(n2d[1]*10,2), color=clist[n], linewidth = 1)
    
    
plt.ylim(-0.1, 2)
plt.xlim(0, np.max(n2d_av_r[0][0]))
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)#MISTAKE: atom number is by pixel, not per um^2
plt.legend(['68.0 mT', '85.3 mT', '97.0 mT', '107.4 mT'], fontsize=fsz)
plt.hlines(0,60,110, colors='black')
plt.xticks(fontsize=fsz)

plt.tight_layout()

##############################################################################
# Analyze the total number of atoms and atoms in the ring
##############################################################################

def plotvals(data, label):
    avg = []
    std = []
    for i in range(4):
        subset = data[i*10:(i+1)*10]
        avg.append(np.average(subset))
        std.append(np.std(subset))
    plt.errorbar(range(4), avg, yerr=std, label=label)

atom_total = np.array([np.sum(x*com.av_atom_select) for x in atom_per_pix])
ring_total = np.array([np.sum(x*com.av_ring_select) for x in atom_per_pix])
halo_total = np.array([np.sum(x*com.av_halo_select) for x in atom_per_pix])
bacg_total = np.array([np.sum(x*com.av_bacg_select) for x in atom_per_pix])

bacg_avg = bacg_total/com.bacg_area

atom_bgcorr = atom_total-bacg_avg * com.atom_area
ring_bgcorr = ring_total-bacg_avg * com.ring_area
halo_bgcorr = halo_total-bacg_avg * com.halo_area
bacg_bgcorr = bacg_total-bacg_avg * com.bacg_area

plt.figure(14, clear=True)
plt.subplot(121)
plotvals(atom_total, 'total')
plotvals(ring_total, 'ring' )
plotvals(halo_total, 'halo' )
plotvals(bacg_total, 'bacg' )
plt.legend()

plt.ylim(0,12000)
plt.title("Atom number by region (Uncorr.)")

plt.subplot(122)
plotvals(atom_bgcorr, 'total')
plotvals(ring_bgcorr, 'ring' )
plotvals(halo_bgcorr, 'halo' )
plotvals(bacg_bgcorr, 'bacg' )
plt.legend()

plt.ylim(0,12000)
plt.title("Atom number by region (Backg Subtracted)")

if SAVE_FIGURES:
    plt.savefig("results//atomnumber_plot")


##############################################################################
if SAVE_DATA:
    
    # save processed image data as numpy files for convenience
    np.save('processed//od_imgs', od_imgs)
    np.save('processed//od_avg', groupav_before_od)
    
    np.save('processed//atom_per_pix', atom_per_pix)
    np.save('processed//atom_per_pix_avg', atom_per_pix_avg)
    
    # create h5 file for storage of comlpete complete processing results
    with h5py.File(com.DATADIR + "\\processed\\insitu_ring_processed.h5", 'a') as h5f:

        # Save optical depth data
        odimg_grp = com.to_h5(h5f, 'od_imgs', od_imgs, attrs = {'shot_ids':shot_ids})
        com.to_h5(h5f, 'od_grouped', com.group(odimg_grp), attrs={'shot_ids':com.group(shot_ids)})
        com.to_h5(h5f, 'od_avg', np.array(groupav_before_od), attrs = {'bfield_mT':com.bfield_mT})      
        
        # save atom per pixel data
        com.to_h5(h5f, 'atom_per_pix', np.array(atom_per_pix), attrs = {'shot_ids':shot_ids})
        com.to_h5(h5f, 'atom_per_pix_avg', atom_per_pix_avg, attrs = {'bfield_mT': com.bfield_mT})
        
