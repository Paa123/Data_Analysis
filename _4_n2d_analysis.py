# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 13:40:02 2021

@author: Kevin
"""
import os

import h5py

import numpy as np
from numpy.ma import masked_array

import scipy.constants as sc
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd

from matplotlib import pyplot as plt

from quagmire.fit import fit1d as f1d
from quagmire.utils.image_mask import circular_mask
from quagmire.atomic import Li6

import _0_common as com

SAVE_FIGURES = False

###############################################################################
# Load full-frame atom-per-pixel data from h5 (created in raw_to_processed.py)
###############################################################################

DATADIR = 'H:\\Persistent Currents\\Data Analysis 2021\\2020_07_09_insitu_ring_images'
os.chdir(DATADIR)

ringc_data = pd.read_hdf("plotdata\\n2d(r).h5", key="ring_centered")
haloc_data = pd.read_hdf("plotdata\\n2d(r).h5", key="halo_centered")   

###########################################################################
# Extract atomic column density (/um^2) versus radius at each Bfield value
# We'll do this twice: centered on the ring, and centered on the halo.
###########################################################################

pixvals = np.linspace(1,260,500)  
# NOTE: This goes off the edge of the image, but we're only averaging
#       pixels that are included in the mask

rvals =  ringc_data['68.0']['rvals']
dr = rvals[1]-rvals[0]

labels = ringc_data.keys() # ['68.0', '85.4', '97.6', '107.4'] 
##############################################################################
# Plot the (ring-centered) radial column densities for comparison

plt.figure(41, clear=True, figsize=(5,4), dpi=200)
fsz = 9
#plt.subplot(2,1,1)
plt.cla()

clist = ['blue', 'green', 'purple', 'red']
for n in range(4):
    dset = ringc_data[labels[n]]
    plt.plot(rvals, dset['avg'], color=clist[n], linewidth = 1)

for n in range(4):
    dset = ringc_data[labels[n]]
    plt.plot(rvals[100:], dset['avg'][100:]*10, color=clist[n], linewidth = 1)
    
    
plt.ylim(0, np.max(ringc_data['68.0']['avg']))
plt.xlim(0,np.max(rvals))
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['68.0 mT', '85.3 mT', '97.0 mT', '107.4 mT'], fontsize=fsz)
plt.xticks(fontsize=fsz)

plt.tight_layout()

if SAVE_FIGURES: 
    plt.savefig("results//ring_centered_radial_column_density")

###############################################################################
# Perform fits to the radial column densities at each field
###############################################################################
# Exclude data points for small and large radius
weights = np.where(rvals > 25, 1, 0) * np.where(110 > rvals, 1, 0)

###############################################################################

# Fit the radial column density in the BEC limit
plt.figure(42, clear=True, figsize=(8,6), dpi=150)
fsz = 9 # font size
plt.subplot(2,2,1)

ydata = haloc_data['68.0']['avg']
plt.plot(rvals, ydata, color="blue", linewidth = 1)

# fit 1D gaussian to the halo
becinitp =  [0,  70, 0.8, 0]
becbounds =((-0.001, 50, 0.5, -0.001),
         ( 0.001, 100, 1.5,  0.001))

becfit = f1d.fit1D( ydata, f1d.gauss_peak_1D, initp=becinitp, bounds=becbounds, xdata=rvals, weights=weights, return_all=True) 
bec_gauss_fitfunc = f1d.gauss_peak_1D(becfit['x'])
bec_gauss_fit_data = bec_gauss_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, bec_gauss_fit_data, 'k--')

# Add labeling etc.
plt.ylim(0, 1.25)
plt.xlim(0,np.max(rvals))
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['68.0 mT'], fontsize=fsz)
plt.xticks(fontsize=fsz)
rbec = becfit['x'][1]
Tbec = (rbec * 1E-6) **2 * Li6.mass * (2 * np.pi * 37)**2 / (2 * sc.Boltzmann)
plt.text(50, 0.9, "Gaussian $1/e^2$ = %.1f $\mu$m" % rbec, fontsize=fsz)
plt.text(50, 0.6, "$T_{BEC}$ = %.1f nK" % (Tbec * 1E9), fontsize=fsz)

##################################################
# Plot and fit for unitary 85.3 mT

plt.subplot(2,2,2)
plt.cla()
ydata = haloc_data['85.4']['avg']
plt.plot(rvals, ydata, color='green', linewidth=1)

# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
uniinitp =  [ 0.000,    80,    1.0,   0.000,    1.14]
unibounds =((-0.001,    55,    0.5,  -0.001,    1),
            ( 0.001,   120,    2.0,   0.001,    10))

unifit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=uniinitp, bounds=unibounds, xdata=rvals,  weights=weights, return_all=True) 
uni_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(unifit['x'])
uni_fit_data = uni_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, uni_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.25)
plt.xlim(0,np.max(rvals))

plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['85.3 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(50, 0.8, "width = %.1f $\mu$m" % unifit['x'][1], fontsize=fsz)
plt.text(50, 0.6, " $ \\beta \cdot \mu$ = %.2f" % unifit['x'][4], fontsize=fsz)

###############################################################################
# Plot and fit for near BCS 97 mT

plt.subplot(2,2,3)
plt.cla()

ydata =  haloc_data['97.6']['avg']
plt.plot(rvals, ydata, color='purple', linewidth=1)


# Fit 3D Fermi-Dirac column density function to the halo
#             r0,    TF width   n0    n(inf),   beta*mu
bcsinitp =  [ 0.000,    100,    1.0,   0.000,    2]
bcsbounds =((-0.001,    75,    0.5,  -0.001,    1),
            ( 0.001,   125,    2.0,   0.001,    10))


nbcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
nbcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(nbcsfit['x'])
nbcs_fit_data = nbcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, nbcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.25)
plt.xlim(0,np.max(rvals))

plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['97.5 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(60, 0.8, "width = %.1f $\mu$m" % nbcsfit['x'][1], fontsize=fsz)
plt.text(70, 0.6, " $ \\beta \cdot \mu$ = %.2f" % nbcsfit['x'][4], fontsize=fsz)

###############################################################################
# Plot and fit for far BCS 107 mT

plt.subplot(2,2,4)
plt.cla()

ydata =  haloc_data['107.4']['avg']
plt.plot(rvals, ydata, color='red', linewidth=1)

fbcsfit = f1d.fit1D(ydata, f1d.fermi_dirac_dist_peak_1D, initp=bcsinitp, bounds=bcsbounds, xdata=rvals,  weights=weights, return_all=True) 
fbcs_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(fbcsfit['x'])
fbcs_fit_data = fbcs_fd_fitfunc(rvals)

# add the fit to the plot
plt.plot(rvals, fbcs_fit_data, 'k:', label="fermi-dirac peak fit")

plt.ylim(0, 1.25)
plt.xlim(0, np.max(rvals))

plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['107.4 mT', '3D F-D fit'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.text(60, 0.8, "width = %.1f $\mu$m" % fbcsfit['x'][1], fontsize=fsz)
plt.text(70, 0.6, " $ \\beta \cdot \mu$ = %.2f" % fbcsfit['x'][4], fontsize=fsz)

plt.tight_layout()

if SAVE_FIGURES: 
    plt.savefig("results//radial_density_fits")


############################
#Plot the column density in log scale
fsz=14
plt.figure(43, clear=True, figsize=(6,3), dpi=200)

plt.semilogy(rvals, ringc_data['68.0']['avg'], "b-", linewidth = 1.5)
#plt.semilogy(rvals, bec_gauss_fit_data, 'k--')

plt.semilogy(rvals, ringc_data['85.4']['avg'], "g-", linewidth = 1.5)
#plt.semilogy(rvals, bec_gauss_fit_data, 'k--')

plt.semilogy(rvals,ringc_data['107.4']['avg'], "r-", linewidth = 1.5)
#plt.semilogy(rvals, fbcs_fit_data, 'k:', label="fermi-dirac prak fit")

plt.ylim(0.01, 25)
plt.xlim(0,np.max(rvals))

plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('$n_{2D}$ (atom/$\mu$m$^{2})$', fontsize=fsz)

# plt.legend(['BEC (68 mT)', 'BEC G Fit','BCS (107 mT)', 'BCS FD Fit'], fontsize=fsz)

plt.xticks(fontsize=fsz)
plt.yticks([0.1, 0.2, 0.5, 1, 2, 5, 10], fontsize=16)
# plt.xticks([0,10,20,30,40,50,60, 70, 80], fontsize=16)
plt.grid()
plt.tight_layout()

if SAVE_FIGURES: 
    plt.savefig("results//radial_profile_logscale")

##################
# Plot the number of atoms found at each radius

plt.figure(44, clear=True, figsize=(6,3), dpi=200)

plt.plot(rvals, ringc_data['68.0']['n1d_r'], "b-", linewidth = 1.5)
plt.plot(rvals, ringc_data['107.4']['n1d_r'], "r-", linewidth = 1.5)

plt.ylim(0,700)
plt.xlim(0,np.max(rvals))

plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('$2 \pi r \cdot n_{2D}$ ($\mu$m$^{-1})$', fontsize=fsz)

plt.tight_layout()
plt.legend(['BEC (68 mT)','BCS (107 mT)'], fontsize=fsz)

if SAVE_FIGURES: 
    plt.savefig("results//number_versus_radius")


##############################################################################
# Plot the BEC and BCS radial column densities with uncertainties
plt.figure(45, clear=True, figsize=(3,2), dpi=300)
fsz = 8
sigma = 4

ax1 = plt.subplot(1,1,1)
ax1.cla()
dset = ringc_data['68.0']
upper = dset['avg'] + 2*dset['err']
lower = dset['avg'] - 2*dset['err']
ax1.fill_between(rvals, gaussian_filter1d(lower, sigma), gaussian_filter1d(upper, sigma), color='lightgray')
ax1.plot(rvals, gaussian_filter1d(dset['avg'], sigma), color='blue', linewidth=1)


dset = ringc_data['107.4']
upper = dset['avg'] + 2*dset['err']
lower = dset['avg'] - 2*dset['err']
ax1.fill_between(rvals, gaussian_filter1d(lower, sigma), gaussian_filter1d(upper, sigma), color='lightgray')
ax1.plot(rvals, gaussian_filter1d(dset['avg'], sigma), color='red', linewidth=1)

plt.ylim(0,10.5)
plt.xlim(0,np.max(rvals))
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('$N_{\sigma}$/$\mu$m$^{2}$', fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)

from mpl_toolkits.axes_grid.inset_locator import (InsetPosition)
fsz=7
ax2 = plt.axes([0,0,1,1])
ax2.cla()
ip = InsetPosition(ax1, [0.3, 0.25, 0.67,0.7])
ax2.set_axes_locator(ip)

dset = ringc_data['68.0']
upper = dset['avg'] + 2*dset['err']
lower = dset['avg'] - 2*dset['err']
ax2.fill_between(rvals, gaussian_filter1d(lower, sigma), gaussian_filter1d(upper, sigma), color='lightgray')
ax2.plot(rvals, gaussian_filter1d(dset['avg'], sigma), color='blue', linewidth=1)
ax2.plot(rvals, bec_gauss_fit_data, 'k:', label="Gauss. fit", linewidth=1)

dset = ringc_data['107.4']
upper = dset['avg'] + 2*dset['err']
lower = dset['avg'] - 2*dset['err']
ax2.fill_between(rvals, gaussian_filter1d(lower, sigma), gaussian_filter1d(upper, sigma), color='lightgray')
ax2.plot(rvals, gaussian_filter1d(dset['avg'], sigma), color='red', linewidth=1)
ax2.plot(rvals, fbcs_fit_data, 'k:', label="F-D fit", linewidth=1)

plt.ylim(0,1)
plt.xlim(25,np.max(rvals))

plt.xticks(fontsize=fsz)
plt.yticks([0.0, 0.5, 1.0],fontsize=fsz)

plt.legend(['68.0 mT', 'BEC Fit', '107.4 mT', 'F-D Fit'], fontsize=5)
plt.tight_layout()

##############################################################################
# Plot the BCS radial column density, with zoomed-in view
plt.figure(46, clear=True, figsize=(3,1.5), dpi=300)
fsz = 7
sigma = 1

ax1 = plt.subplot(1,1,1)
ax1.cla()
dset = ringc_data['107.4']
avg = dset['avg']
upper = dset['avg'] + 2*dset['err']
lower = dset['avg'] - 2*dset['err']

avgf = gaussian_filter1d(avg, sigma)
upperf = gaussian_filter1d(upper, sigma)
lowerf = gaussian_filter1d(lower, sigma)

ax1.fill_between(rvals, lowerf, upperf, color='gray')
ax1.plot(rvals, avgf, color='red', linewidth=1)

mult = 10
ax1.fill_between(rvals[100:], lowerf[100:] * mult, upperf[100:] * mult, color='gray')
ax1.plot(rvals[100:], avgf[100:] * mult, color='red', linewidth=1)
ax1.plot(rvals, fbcs_fit_data * mult, 'k:', linewidth=1)

plt.ylim(0, np.max(upper))
plt.xlim(0,np.max(rvals))
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('$n_{2D}^\sigma$ (atoms/$\mu$m$^{2}$)', fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks([2,4,6,8,10], fontsize=fsz)
plt.text(80, 2, "$\\times 10$", fontsize=fsz)

# plt.legend(['68.0 mT', 'x5', 'F-D Fit'], fontsize=5)
plt.tight_layout()

if SAVE_FIGURES: 
    plt.savefig("results//density_x10")
    
    
# Plot the (ring-centered) radial column densities for comparison

plt.figure(49, clear=True, figsize=(8,3), dpi=250)
fsz = 18
#plt.subplot(2,1,1)
plt.cla()

clist = ['blue', 'green', 'purple','red']
for n in [0,1,2,3]:
    dset = ringc_data[labels[n]]
    plt.plot(rvals, dset['avg'], color=clist[n], linewidth = 2)

for n in [0,1,2,3]:
    dset = ringc_data[labels[n]]
    vals = gaussian_filter1d(dset['avg'], 4)[95:] * 10
    plt.plot(rvals[95:], vals, color=clist[n], linewidth = 2)
    
    
plt.ylim(0, np.max(ringc_data['68.0']['avg']))
plt.xlim(0,np.max(rvals))
plt.xlabel('r ($\mu$m)', fontsize=fsz)
plt.ylabel('atoms/$\mu$m$^{2}$', fontsize=fsz)
plt.legend(['68.3 mT', '85.7 mT', '97.6 mT', '107.4 mT'], fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)

plt.tight_layout()

if SAVE_FIGURES: 
    plt.savefig("results//radial_column_density_comparison")