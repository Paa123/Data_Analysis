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



SAVE_DATA = True
SAVE_FIGURES = True

###############################################################################
# Load full-frame atom-per-pixel data from h5 
###############################################################################

DATADIR = 'E:\\Persistent Currents\\insitu_ring_images\\2020_07_09_insitu_ring_images'
os.chdir(DATADIR)
h5f = h5py.File(DATADIR + "\\processed\\insitu_ring_processed.h5", 'a')

import _0_common as com

n2d = h5f['n2d']
n2d_av = h5f['n2d_av']
bfield_mT = n2d_av.attrs.get('bfield_mT')

ring_centers = h5f['ring_centers']
halo_centers = h5f['halo_centers']




###########################################################################
# Adding code for extracting radial profile of rings at different B
###########################################################################

# The averaged images for each magnetic field are in n2d_av
# Image arrays are 300x300 pixels
# Looks like the data is dead centered on [150,150]

# Options for azimuthal averaging?


from scipy.ndimage.filters import gaussian_filter1d
from quagmire.fit.sample import gsamp_az, gsamp_rad
from quagmire.fit.fit1d import fit1D, gauss_peak_1D, peakguess1D, check_1Dfit



N=200
thetavals = np.linspace(0, 2*np.pi, N)

R = 29

xc, yc = ring_centers[2]
d2 = np.array(gsamp_az(n2d_av[2], xc, yc, 0, 2*np.pi, R, N, 1)[0])
fd2 = gaussian_filter1d(d2, 1)

xc, yc = ring_centers[3]
d3 = np.array(gsamp_az(n2d_av[3], xc, yc, 0, 2*np.pi, R, N, 1)[0])
fd3 = gaussian_filter1d(d3, 1)

t = gsamp_rad(n2d_av[3], xc, yc, theta=0.1 , r1=0, r2=60, N=120, rs=1)[0]
fitguess = peakguess1D(t)
f = fit1D(t, gauss_peak_1D, fitguess)
check_1Dfit(t, fitguess, f, gauss_peak_1D)

plt.figure(500, clear=True)
plt.plot(thetavals, fd2, color='red')
plt.plot(thetavals, fd3, color='gray')

plt.fill_between(thetavals, fd2, fd3, color='lightgray')

d23 = (d2+d3)/2
fd23 = gaussian_filter1d(d23, 1)
davg = np.average(d23)
# plt.plot(thetavals, d23, 'k.')
plt.plot(thetavals, fd23, 'k-')

plt.xlabel("Angle (radians)")
plt.ylabel("$n_{2D}$ atoms/$\mu$m$^2$")

dmax = np.max(fd23)
dmin = np.min(fd23)


plt.hlines(davg, 0, 2*np.pi, color='black', linestyle='dotted')

plt.plot(2.51, dmin,'bo' )

deltamin = davg-dmin
deltamax = dmax-davg



plt.text(2,4,'$n_{2D}=%.1f^{+%.1f}_{-%.1f}$ atoms/$\mu$m$^2$' % (davg, deltamax, deltamin), fontsize=12)

plt.ylim(0, 11)
plt.xlim(0, 2 * np.pi)


##################################################################
# Peak 3D density around the ring
##################################################################
from scipy.constants import hbar, h
from scipy.constants import Boltzmann as kB

a_z = np.sqrt(hbar/(Li6.mass*2*np.pi*1500))

def n3d(n2d): 

    return (512/(81*np.pi**5))**(1/4)*(n2d/a_z**2)**(3/4)

corr = 1.75

n3dav = n3d( (fd2 + fd3) / 2 * 1E12) / 1E18 * corr

n3drms = n3d(np.sqrt((fd2-fd3)**2) * 1E12) / 1E18 * corr
n3drmsav =np.average(n3drms)
n3dperr = gaussian_filter1d(n3dav + n3drmsav, 3)
n3dmerr = gaussian_filter1d(n3dav - n3drmsav,3)
   
plt.figure(501, clear=True)
plt.plot(thetavals, n3dperr, color='gray')
plt.plot(thetavals, n3dmerr, color='gray')
plt.fill_between(thetavals, n3dperr, n3dmerr, color='lightgray')
plt.plot(thetavals, n3dav, 'k-')

plt.xlabel("Angle (radians)")
plt.ylabel("$n_{3D}$ atoms/$\mu$m$^3$")

n3dmax = np.max(n3dav)
n3dmin = np.min(n3dav)

n3d_azavg = np.average(n3dav)

plt.hlines(n3d_azavg, 0, 2*np.pi, color='black', linestyle='dotted')

plt.plot(2.505, n3dmin,'bo' )

n3d_deltamin = n3d_azavg-n3dmin
n3d_deltamax = n3dmax-n3d_azavg



plt.text(2,1,'$n_{3D}=%.1f^{+%.1f}_{-%.1f}$ atoms/$\mu$m$^2$' % (n3d_azavg, n3d_deltamax, n3d_deltamin), fontsize=12)

plt.ylim(0, 3.5)
plt.xlim(0, 2 * np.pi)

##################################################################
# Local Fermi temperature around the ring (uK)
##################################################################


def EF(n3d):
    return hbar**2/(2*Li6.mass)*(3*np.pi**2 *n3d)**(2/3)

def kF(EF):
    return np.sqrt(2 * Li6.mass * EF)/hbar

Tfav = EF(n3dav * 1E18) / kB * 1E6

Tfrmsav = 0.077
   
plt.figure(501, clear=True)

Tf_azavg = np.average(Tfav)

Tfperr = np.ones(N)*np.max(Tfav) # gaussian_filter1d(Tfav + Tfrmsav, 6)
Tfmerr = np.ones(N)*np.min(Tfav)  # gaussian_filter1d(Tfav - Tfrmsav, 6)


plt.plot(thetavals, Tfperr, color='gray', linestyle='dotted')
plt.plot(thetavals, Tfmerr, color='gray', linestyle='dotted')
plt.fill_between(thetavals, Tfperr, Tfmerr, color='lightgray')
plt.plot(thetavals, Tfav, 'k.')

plt.hlines(Tf_azavg, 0, 2*np.pi, color='gray', linestyle='dashed')

plt.xlabel("$\phi$ (radians)")
plt.ylabel("$T_F( \phi )$ ($\mu$K)")

Tfmax = np.max(Tfav)
Tfmin = np.min(Tfav)

plt.plot(2.505, Tfmin,'bo' )

plt.text(2.5, 0.65,"$T_F^{(-)}=%.2f$ $\mu$k" % (np.min(Tfav)), fontsize=12)
plt.text(2.5, 0.92,"$T_F^{(+)}=%.2f$ $\mu$k" % (np.max(Tfav)), fontsize=12)

plt.ylim(0, 1)
plt.xlim(0, 2 * np.pi)

##################################################################
# Local Fermi energy around the ring, frequency units kHz
##################################################################

Efav = EF(n3dav * 1E18) / h / 1E3

Efrmsav = 1
Efperr = gaussian_filter1d(Efav + Efrmsav, 2)
Efmerr = gaussian_filter1d(Efav - Efrmsav, 2)
   
plt.figure(502, clear=True)
plt.plot(thetavals, Efperr, color='gray')
plt.plot(thetavals, Efmerr, color='gray')
plt.fill_between(thetavals, Efperr, Efmerr, color='lightgray')
plt.plot(thetavals, Efav, 'k-')

plt.xlabel("Angle (radians)")
plt.ylabel("$E_F/h$ (kHz)")

Efmax = np.max(Efav)
Efmin = np.min(Efav)

Ef_azavg = np.average(Efav)

plt.hlines(Ef_azavg, 0, 2*np.pi, color='black', linestyle='dotted')

plt.plot(2.505, Efmin,'bo' )

Ef_deltamin = Ef_azavg-Efmin
Ef_deltamax = Efmax-Ef_azavg



plt.text(2, 7,'$E_F=%.1f^{+%.1f}_{-%.1f}$ kHz' % (Ef_azavg, Ef_deltamax, Ef_deltamin), fontsize=12)

plt.ylim(0, 19)
plt.xlim(0, 2 * np.pi)

############################################################################
def kF(EF):
    return np.sqrt(2 * Li6.mass * EF)/hbar
