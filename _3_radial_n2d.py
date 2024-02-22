# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:40:02 2021

@author: Kevin
"""
import os
import h5py

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gfilt1d
import scipy.constants as sc
from matplotlib import pyplot as plt
import pandas as pd

from quagmire.fit import fit1d as f1d
from quagmire.utils.image_mask import circular_mask
from quagmire.atomic import Li6

import _0_common as com

SAVE_DATA = True
SAVE_FIGURES = True

###############################################################################
# Load full-frame atom-per-pixel data from h5 
###############################################################################

# DATADIR = 'H:\\Persistent Currents\\insitu_ring_images\\2020_07_09_insitu_ring_images'
#os.chdir(DATADIR)
DATADIR = os.getcwd()
h5f = h5py.File(DATADIR + "\\processed\\insitu_ring_processed.h5", 'a')

atoms_pix = h5f["atom_per_pix"]#just plain OD like image for every shot
shot_ids = atoms_pix.attrs.get('shot_ids')

atoms_avg = h5f['atom_per_pix_avg']#just plain OD like image for every shot, avgd for each label
bfield_mT = atoms_avg.attrs.get('bfield_mT')

ring_centers = h5f['ring_centers']
halo_centers = h5f['halo_centers']

##############################################################################
# Convert atom number per pixel to 2D column density (atoms per um^2)
############################################################################## 

mag_factor = 28.65
pixsize = 12E-6
um_per_pix = pixsize/mag_factor*1E6  # 0.4188 um/pix
um2_per_pix = um_per_pix**2

n2d = []
for atimg in atoms_pix:
    n2d.append(atimg / um2_per_pix)#pixel wise atom density per um^2

n2d_av = []
for atimg in atoms_avg:
    n2d_av.append(atimg / um2_per_pix)#pixel wise atom density  per um^2
    
if SAVE_DATA:
    # create h5 file for processed images and info extracted from shotfiles
    with h5py.File(com.DATADIR + "\\processed\\insitu_ring_processed.h5", 'a') as h5f:
       
        com.to_h5(h5f, 'n2d', n2d, attrs={'shot_ids':shot_ids})   
        com.to_h5(h5f, 'n2d_av', n2d_av, attrs={'bfield_mT': com.bfield_mT} )          


###########################################################################
# Extract atomic column density (/um^2) versus radius at each Bfield value
#Using the averaged image per um^2 for this
# We'll do this twice: centered on the ring, and centered on the halo.
###########################################################################

pixvals = np.linspace(1,260,500)  
# NOTE: This goes off the edge of the image, but we're only averaging
#       pixels that are included in the mask

rvals =  pixvals * com.um_per_pix
dr = rvals[1]-rvals[0]

two_pi_r = 2 * np.pi * rvals #circumference of each slice
two_pi_r_dr = 2 * np.pi * rvals * dr #area of each slice

d = 0.5 # width of the mask sampling interval

labels = ['68.0', '85.4', '97.6', '107.4'] 

ringc_data = {'68.0':{}, '85.4':{}, '97.6':{}, '107.4':{} }# a new dictionary to store all the quantities
for n, l in enumerate(labels):
    dl = ringc_data[l]
    dl['rvals'] = rvals #radial values
    dl['N'] = np.zeros(500) #empty arrays
    dl['avg'] = np.zeros(500)
    dl['std'] = np.zeros(500)
    for i, R in enumerate(pixvals):
            x0, y0 = ring_centers[n]
            o_mask = circular_mask(R+d, (512, 512), (y0, x0), inside_is=False)
            i_mask = circular_mask(R-d, (512, 512),  (y0, x0), inside_is=True)
            exclusion_mask = i_mask + o_mask
            Rbindata = np.ma.masked_array(n2d_av[n], mask=exclusion_mask, fill_value=np.nan).compressed()
            #dl['bindata'].append(Rbindata)
            dl['N'][i] = len(Rbindata) #number of pixels at this radius
            dl['avg'][i] = np.average(Rbindata) #Average number of atoms per pixel
            dl['std'][i] = np.std(Rbindata) #std of number of atoms per pixel
    dl['err'] = dl['std']/np.sqrt(dl['N']) #SEM at each radius
    dl['n1d_r'] = ringc_data[l]['avg'] *  two_pi_r#Total number of atoms per radial slice

##########################################################
# Again, but this time centered on the halos

# Convert the dictionary of results to a Pandas dataframe    
ringc_data=pd.DataFrame(ringc_data)
 
haloc_data = {'68.0':{}, '85.4':{}, '97.6':{}, '107.4':{} }           
for n, l in enumerate(labels):
    dl = haloc_data[l]
    dl['rvals'] = rvals
    dl['N'] = np.zeros(500)
    dl['avg'] = np.zeros(500)
    dl['std'] = np.zeros(500)
    for i, R in enumerate(pixvals):
            x0, y0 = halo_centers[n]
            o_mask = circular_mask(R+d, (512, 512), (y0, x0), inside_is=False)
            i_mask = circular_mask(R-d, (512, 512),  (y0, x0), inside_is=True)
            exclusion_mask = i_mask + o_mask
            Rbindata = np.ma.masked_array(n2d_av[n], mask=exclusion_mask, fill_value=np.nan).compressed()
            #dl['bindata'].append(Rbindata)
            dl['N'][i] = len(Rbindata)
            dl['avg'][i] = np.average(Rbindata)
            dl['std'][i] = np.std(Rbindata)
    dl['err'] = dl['std']/np.sqrt(dl['N'])  
    dl['n1d_r'] = haloc_data[l]['avg'] *  two_pi_r

# Convert the dictionary of results to a Pandas dataframe
haloc_data = pd.DataFrame(haloc_data)

###############################################################################

if SAVE_DATA:
    ringc_data.to_hdf("plotdata\\n2d(r).h5", key="ring_centered")
    haloc_data.to_hdf("plotdata\\n2d(r).h5", key="halo_centered")   

##############################################################################
##############################################################################



##############################################################################
# The rest below is just plots and validation
###############################################################################
# Plot the ring-centered and halo-centered radial column density  (BEC limit) - gauss_peak_1D

# Exclude data points for small and large radius
weights = np.where(rvals > 25, 1, 0) * np.where(110 > rvals, 1, 0)

#########################################
# First, the ring-centered column density
plt.figure(31, clear=True, figsize=(5,5), dpi=200)
plt.subplot(2,1,1)
plt.title(" Ring-centered radial column density (BEC limit)")

plt.plot(rvals, ringc_data['68.0']['avg'], color="blue", linewidth = 1)

# fit 1D gaussian to the halo
becinitp =  [0,  100, 1.1, 0]
becbounds =((-0.001, 50, 0.5, -0.001),
         ( 0.001, 150, 2.0,  0.001))

becfit = f1d.fit1D( ringc_data['68.0']['avg'],  f1d.gauss_peak_1D, initp=becinitp, bounds=becbounds, xdata=rvals, weights=weights, return_all=True) 
bec_gauss_fitfunc = f1d.gauss_peak_1D(becfit['x'])
bec_gauss_fit_data = bec_gauss_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bec_gauss_fit_data, 'k:')

# Add labeling etc.
plt.ylim(0,1.5)
plt.xlim(0,np.max(rvals))
fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['68.0 mT'], fontsize=fsz)
plt.xticks(fontsize=fsz)
rbec = becfit['x'][1]
Tbec = (rbec * 1E-6) **2 * Li6.mass * (2 * np.pi * 37)**2 / (2 * sc.Boltzmann)
plt.text(50, 0.8, "Gaussian $1/e^2$ = %.1f $\mu$m" % rbec)
plt.text(50, 0.6, "$T_{BEC}$ = %.1f nK" % (Tbec * 1E9))

#######################################
# Then the halo-centered column density
plt.subplot(2,1,2)
plt.title(" Halo-centered radial column density (BEC limit)")

plt.plot(rvals, haloc_data['68.0']['avg'], color="blue", linewidth = 1)

# fit 1D gaussian to the halo
becinitp =  [0,  100, 1.1, 0]
becbounds =((-0.001, 50, 0.5, -0.001),
         ( 0.001, 200, 2.0,  0.001))

becfit = f1d.fit1D( haloc_data['68.0']['avg'], f1d.gauss_peak_1D, initp=becinitp, bounds=becbounds, xdata=rvals, weights=weights, return_all=True) 
bec_gauss_fitfunc = f1d.gauss_peak_1D(becfit['x'])
bec_gauss_fit_data = bec_gauss_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bec_gauss_fit_data, 'k:')

# Add labeling etc.
plt.ylim(0,1.5)
plt.xlim(0,np.max(rvals))
fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['68.0 mT'], fontsize=fsz)
plt.xticks(fontsize=fsz)
rbec = becfit['x'][1]
Tbec = (rbec * 1E-6) **2 * Li6.mass * (2 * np.pi * 37)**2 / (2 * sc.Boltzmann)
plt.text(50, 0.8, "Gaussian $1/e^2$ = %.1f $\mu$m" % rbec)
plt.text(50, 0.6, "$T_{BEC}$ = %.1f nK" % (Tbec * 1E9))

plt.tight_layout()

###############################################################################

if SAVE_FIGURES:
    plt.savefig("results//radial_profile_68mT")

###############################################################################
# Plot the ring-centered and halo-centered radial column density (Unitary 85.3 mT) fermi_dirac_dist_peak_1D

#########################################
# First, the ring-centered column density
plt.figure(32, clear=True, figsize=(5,5), dpi=200)
plt.subplot(2,1,1)
plt.cla()
plt.title("Ring-centered column density near unitarity") 
ydata = ringc_data['85.4']['avg']
plt.plot(rvals, ydata, color='green', linewidth=1)


# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    1.14]
bcsbounds =((-0.001,    50,    0.5,  -0.001,    1),
            ( 0.001,   100,    2.0,   0.001,    10))

bcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
bcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(bcsfit['x'])
bcs_fit_data = bcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.5)
plt.xlim(0,np.max(rvals))

fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['85.3 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(50, 0.8, "width = %.1f $\mu$m" % bcsfit['x'][1])
plt.text(50, 0.6, "beta mu = %.2f" % bcsfit['x'][4])

#########################################
# Then, the halo-centered column density
plt.subplot(2,1,2)
plt.cla()
plt.title("Halo-centered column density near unitarity") 
ydata = haloc_data['85.4']['avg']
plt.plot(rvals, ydata, color='green', linewidth=1)


# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    1.14]
bcsbounds =((-0.001,    50,    0.5,  -0.001,    1),
            ( 0.001,   100,    2.0,   0.001,    10))

bcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
bcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(bcsfit['x'])
bcs_fit_data = bcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.5)
plt.xlim(0,np.max(rvals))

fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['85.3 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(50, 0.8, "width = %.1f $\mu$m" % bcsfit['x'][1])
plt.text(50, 0.6, "beta mu = %.2f" % bcsfit['x'][4])
plt.tight_layout()

###############################################################################

if SAVE_FIGURES:  
    plt.savefig("results//radial_profile_85mT")

###############################################################################
# Plot the ring-centered and halo-centered radial column density  (BCS 97 mT) - fermi_dirac_dist_peak_1D

#########################################
# First, the ring-centered column density
plt.figure(33, clear=True, figsize=(5,5), dpi=200)
plt.subplot(2,1,1)
plt.cla()
plt.title(" Ring-centered column density in the BCS limit") 
ydata = ringc_data['97.6']['avg']
plt.plot(rvals, ydata, color='purple', linewidth=1)


# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    1.14]
bcsbounds =((-0.001,    50,    0.5,  -0.001,    1),
            ( 0.001,   125,    2.0,   0.001,    10))

bcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
bcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(bcsfit['x'])
bcs_fit_data = bcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.5)
plt.xlim(0,np.max(rvals))

fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['97.5 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(60, 0.8, "width = %.1f $\mu$m" % bcsfit['x'][1])
plt.text(60, 0.6, "beta mu = %.2f" % bcsfit['x'][4])

#########################################
# Then, the halo-centered column density
plt.subplot(2,1,2)
plt.cla()
plt.title("Halo-centered column density in the BCS limit") 

ydata = haloc_data['97.6']['avg']
plt.plot(rvals, ydata, color='purple', linewidth=1)


# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    1.14]
bcsbounds =((-0.001,    75,    0.5,  -0.001,    1),
            ( 0.001,   125,    2.0,   0.001,    10))

bcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
bcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(bcsfit['x'])
bcs_fit_data = bcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.5)
plt.xlim(0,np.max(rvals))


fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['97.5 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(60, 0.8, "width = %.1f $\mu$m" % bcsfit['x'][1])
plt.text(60, 0.6, "beta mu = %.2f" % bcsfit['x'][4])

plt.tight_layout()
##############################################################################

if SAVE_FIGURES:  
    plt.savefig("results//radial_profile_97mT")

###############################################################################
# Plot the ring-centered and halo-centered radial column density  (BCS 107 mT) - fermi_dirac_dist_peak_1D

#########################################
# First, the ring-centered column density
plt.figure(34, clear=True, figsize=(5,5), dpi=200)
plt.subplot(2,1,1)
plt.cla()
plt.title(" Ring-centered column density in the BCS limit") 

ydata = ringc_data['107.4']['avg']
plt.plot(rvals, ydata, color='red', linewidth=1)


# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    1.14]
bcsbounds =((-0.001,    50,    0.5,  -0.001,    1),
            ( 0.001,   125,    2.0,   0.001,    10))


bcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
bcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(bcsfit['x'])
bcs_fit_data = bcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.5)
plt.xlim(0,np.max(rvals))


fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['107.4 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(60, 0.8, "width = %.1f $\mu$m" % bcsfit['x'][1])
plt.text(60, 0.6, "beta mu = %.2f" % bcsfit['x'][4])

#########################################
# Then, the halo-centered column density
plt.subplot(2,1,2)
plt.cla()
plt.title("Halo-centered column density in the BCS limit") 
ydata = haloc_data['107.4']['avg']
plt.plot(rvals, ydata, color='red', linewidth=1)

# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    1.14]
bcsbounds =((-0.001,    75,    0.5,  -0.001,    1),
            ( 0.001,   125,    2.0,   0.001,    10))

bcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
bcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(bcsfit['x'])
bcs_fit_data = bcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.5)
plt.xlim(0,np.max(rvals))


fsz = 9
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['107.4 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(60, 0.8, "width = %.1f $\mu$m" % bcsfit['x'][1])
plt.text(60, 0.6, "beta mu = %.2f" % bcsfit['x'][4])

plt.tight_layout()

if SAVE_FIGURES:  
    plt.savefig("results//radial_profile_107mT")

