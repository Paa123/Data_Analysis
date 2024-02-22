# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 10:05:04 2021

@author: Kevin
"""
# If running from DIRAC data directory

import os
import numpy as np
import h5py


import quagmire as qg
try:
    from quagmire.utils.image_mask import circlemask
except:
    from quagmire.utils.image_mask import circular_mask
from quagmire.process.absorption import subtract_dark, calc_probe_norm, flats_to_od
# from quagmire.analysis.atomnumber import calc_sigma_eff, od_to_atoms
from quagmire.process.absorption import abs_to_OD
import quagmire.analysis.atomnumber as anum
from quagmire.fit import fit1d as f1d
from quagmire.fit import fit2d as f2d
import matplotlib.pyplot as plt
from quagmire.feature.cloud import find_peak_coords
from quagmire. feature.roi import ROI
from scipy.ndimage import median_filter, gaussian_filter
from scipy.special import iv as bessel
from numpy.fft import fft,fftfreq,fftshift



##############################################################################
# Load data from file, and create an h5 file for saving processed images etc
##############################################################################
# DATADIR = 'E:\\Persistent Currents\\insitu_ring_images\\2020_07_09_insitu_ring_images'
# =============================================================================
# DATADIR = os.getcwd()
# RAWDIR = os.path.join(DATADIR,"raw")
# 
# def load_shotlist():
#     os.chdir(DATADIR)
#     shotlist = qg.QGShotList(RAWDIR, camera='cell_v')
#     print('Sorting shots by increasing bfield value')
#     shotlist.sort(key=lambda shot:shot.shotfile['globals'].attrs['BCS_field'])
#     return shotlist
# 
# magcoil_current_control = np.array([680/121.975,7,8,8.8])
# bfield_G = np.array(magcoil_current_control * 121.975)
# bfield_mT = bfield_G/10
# 
# =============================================================================

###############################################################################
# Utility functions
###############################################################################

# =============================================================================
# def group(set_of_all):
#     return [ set_of_all[10*i:10*(i+1)] for i in range(4)]
# 
# def ungroup(grouped):
#     return grouped[0] + grouped[1] + grouped[2] + grouped[3]
# 
# =============================================================================
def int_pattern_2D(inpars):
    # xc,yc,R,w(sigma),,amp,bac,t
    if len(inpars) != 7:
        raise f2d.Fit2DError('gaussian2D: missing parameters')
    inpars = [float(par) for par in inpars]
    xc, yc, R, w, amp, bac,t = inpars
    factor=w**2-2*(1j)*t
    if bac == None:
        def gr2D(x, y):
            rho = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            gr = amp*np.real(np.exp((-rho**2-R**2)/2/factor)*bessel(0,rho*R/factor)/factor)
            return gr
    else:
        def gr2D(x, y):
            rho = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            gr = amp*np.real(np.exp((-rho**2-R**2)/2/factor)*bessel(0,rho*R/factor)/factor)+bac
            return gr
    return gr2D


def masked_array_mean(ma_list):
    result = np.ma.masked_array.mean(np.ma.masked_array(ma_list), axis = 0)
    result.fill_value = ma_list[0].fill_value
    return result

#There is a similar function in quagmire.utils, but that is not boolean, anti-aliased
# and centre-fixed. can't turn of anti-aliasing. Otherwise, it is mostly inspored from
# that function, and by circular_mask.

def sector_mask(rmin, rmax, pmin, pmax, center, array_shape,shift=None):
    '''
    Parameters
    ----------
    rmin : float or int
        Inner radius of sector.
    rmax : float or int
        Outer radius of sector.
    pmin : float
        Inner angle of sector.
    pmax : float
        Outer angle of sector.
    center : (y0: int, x0: int)
        location of the center of the circle (distance from upper left corner)
        If no value is given the default is the middle of the array
    array_shape : either tuple, (int,int) or int
        Dimensions of the array (y height, x width). Assumed square if only a
        single integer is given, made (2*rmax+1, 2*rmax+1) if no shape is specified
    shift : tuple/list, optional
        Shift of the center of the circle from the middle of the array. Only
        works if no center location is explicitly specified. The default is None.

    Raises
    ------
    ValueError
        Raises an error if Array shape is not a tuple, list, int or '2r+1'.

    Returns
    -------
    C : numpy.ndarray
        Creates a boolean mask array for a sector of a ring, with
        inner and outer radii specified by rmin and rmax, and between angles
        specified by pmin and pmax (in radians). The returned mask is square, of
        dimensions given by the parameter 'size'.

    '''
    # Determine dimensions of the output array
    if isinstance(array_shape, (tuple, list)):
        ymax, xmax = array_shape
    elif isinstance(array_shape, int):
        ymax, xmax = array_shape, array_shape
    elif array_shape == '2r+1':
        s = 2*int(rmax)+1
        ymax, xmax = s, s
    else:
        raise ValueError("array_shape not valid: %s" % str(array_shape))

    # determine the location of the center of the circle
    if isinstance(center, (tuple, list)):
        y0 = center[0]
        x0 = center[1]
    elif center == 'mid':
        y0 = int(ymax - 1) / 2.
        x0 = int(xmax - 1) / 2.
        if isinstance(shift, (tuple, list)):
            y0 = y0 + shift[0]
            x0 = x0 + shift[1]
    X, Y = np.meshgrid(range(xmax), range(ymax))
    X=X-x0
    Y=Y-y0
    pmin=pmin%(2*np.pi)
    pmax=pmax%(2*np.pi)
    R = np.sqrt(X**2 + Y**2)
    P = np.arctan2(Y, X) + np.pi
    if pmin < pmax:
        C = np.where((rmin < R) & (R < rmax) & (pmin < P) & (P <= pmax), True, False)
    else:
        C = np.where((rmin < R) & (R < rmax) & ((pmin < P) | (P <= pmax)), True, False)
    return C


def dict_to_h5(h5fpath,dict1,grouppara):
    for keys in dict1:
        for prop in dict1[keys]:
            for n,i in enumerate(dict1[keys][prop]):
                if type(dict1[keys][prop][n])!=str:
                    print(prop)
                    to_h5(h5fpath,'/'+grouppara+'/'+keys+'/'+prop,dict1[keys]['fileid'][n],i)
                else:
                    print(prop+' skipped.')

def to_h5(h5fpath, group_name, dset_name, data, attrs=None):
        h5f=h5py.File(h5fpath, 'a')
        adata = np.array(data)
        g=h5f.require_group(group_name)
        d=g.require_dataset(dset_name, adata.shape, adata.dtype)       
        d[...] = adata
        if attrs:
            for k, v in attrs.items():
                d.attrs[k] = v
            print('Saving %s: %s, %s\n' %(dset_name, str(adata.shape), str(adata.dtype)))
        h5f.close()  

def truncate_title(title):
    if len(title)>15:
        title=title[0:5]+'...'+title[-7:]
    return title

###############################################################################
# Core processing from raw images to optical depth, accounting for polarization
##############################################################################

min_signal = 0
filter_dark = True
# norm_type='adaptive' # this script uses a custom normalization routine.
pol_corr = 0.65     # fraction of probe light that is sigma-minus polarized
# Isat = 290
###############################################################################
# Convenience functions to convert from from OD to atoms/pixel and atoms/um^2
##############################################################################
mag_factor = 28.65
pixsize = 12E-6
um_per_pix = pixsize/mag_factor*1E6  # 0.4188 um/pix
um2_per_pix = um_per_pix**2
sigma_eff = anum.calc_sigma_eff(0)
od_to_atoms_conv=anum.od_to_atoms(1,pixsize, sigma_eff,mag_factor)

###############################################################################
# Region selection masks (based on average of all ring images)
# Bfield group averaged centers are defined below.
# Individual image centers are determined in "find_centers" script
###############################################################################
imsize = (512,512)
ring_xc, ring_yc = [233, 250] # typical center point of ring for all images

# region cutoff radii
ring_i_cut = 12
ring_o_cut = 34
bacg_i_cut = 230
bacg_o_cut = 260

#tof cutoff time - from ring like to peak like

# =============================================================================
# def _calc_probe_norm(atom_flat, prob_flat):
#     atom_bg = np.ma.masked_array(atom_flat, mask=~av_bacg_select).compressed()
#     prob_bg = np.ma.masked_array(prob_flat, mask=~av_bacg_select).compressed()
#     
#     atom_bg_av = np.average(atom_bg)
#     prob_bg_av = np.average(prob_bg)
#     
#     normfactor = atom_bg_av/prob_bg_av
#     return normfactor
# 
# def correct_raw_imgdata(raw_abs_set):
#     atom_flat, prob_flat, dark_flat = subtract_dark(raw_abs_set, min_signal, filter_dark)
#     norm_factor = _calc_probe_norm(atom_flat, prob_flat) 
#     normalized_prob_flat = prob_flat * norm_factor
#     
#     # This is the key, we're subtracting the "wrong polarization" light as additional dark background
#     sigmaplus_light = normalized_prob_flat * (1 - pol_corr)
#     corr_flat_set = [atom_flat - sigmaplus_light,
#                      normalized_prob_flat - sigmaplus_light,
#                      dark_flat]
#     return corr_flat_set
# 
# def corr_flats_to_od(corr_flat_set, probe_norm=1):
#     return np.array(flats_to_od(corr_flat_set, probe_norm, Isat))
# 
# =============================================================================
def circleav(image, center):
    x0, y0 = center
    pixvals = np.linspace(1,260,500)  
    d = 0.5
    rvals =  pixvals * um_per_pix
    n2d_r = np.zeros(500)
    for i, R in enumerate(pixvals):
        o_mask = circular_mask(R+d, (512, 512), (y0, x0), inside_is=False)
        i_mask = circular_mask(R-d, (512, 512),  (y0, x0), inside_is=True)
        exclusion_mask = i_mask + o_mask
        Rbindata = np.ma.masked_array(image, mask=exclusion_mask, fill_value=np.nan).compressed()
        n2d_r[i] = np.average(Rbindata)
        # print(np.average(Rbindata))
        # print(np.sum(Rbindata)/np.sum(~exclusion_mask))
    return rvals, n2d_r

def radialav(image, center,rmin,rmax):
    '''
    The return angular values have 0 on the left of the horizontal, and increases
    clockwise.
    '''
    x0, y0 = center
    d = 0.1
    angvals = np.linspace(0,2*np.pi,64)  
    n2d_r = np.zeros(64)
    for i, ang in enumerate(angvals):
        exclusion_mask = ~sector_mask(rmin,rmax,ang-d,ang+d,(y0,x0),512)
        Rbindata = np.ma.masked_array(image, mask=exclusion_mask, fill_value=np.nan).compressed()
        n2d_r[i] = np.average(Rbindata)
        # print(np.average(Rbindata))
        # print(np.sum(Rbindata)/np.sum(~exclusion_mask))
    return angvals, n2d_r

def calc_quant_save_ring(flats,centre_av,procparams,dict1,key,fileid,immode):
    av_xc,av_yc=centre_av
    print('calculating abs_to_od'+key)
    od          =   abs_to_OD(flats, **procparams)
    print(fileid)
    ring_guess=[av_xc,av_yc,(ring_o_cut+ring_i_cut)/2,(ring_o_cut-ring_i_cut)/2,2,0.5]
    halo_guess=[av_xc,av_yc,160,160,0.2,0,0]
    uniinitp =  [ 0.000,    80,    1.0,   0.000,    1.14]
    unibounds =((-0.001,    55,    0.5,  -0.001,    1),
                ( 0.001,   120,    2.0,   0.001,    10))
    av_ring_select =  circular_mask(ring_i_cut, imsize, (av_yc, av_xc), inside_is=False) *  \
                     ~circular_mask(ring_o_cut, imsize, (av_yc, av_xc), inside_is=False)
    x_ring,y_ring,R,w,amp_ring,bac_ring=f2d.fit2D(od, f2d.parab_ring, ring_guess)
    print(x_ring,y_ring,R,w,amp_ring,bac_ring)
    ring_centre=[x_ring,y_ring]
    w_actual=np.sqrt(1+bac_ring/amp_ring)*w
    # av_ring_select_corr =  circular_mask(R-w_actual, imsize, (y_ring, x_ring), inside_is=False) *  \
    #               ~circular_mask(R+w_actual, imsize, (y_ring, x_ring), inside_is=False)
    x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot=f2d.fit2D(od, f2d.gaussian2D, halo_guess,circular_mask(R+w_actual, imsize, (y_ring, x_ring), inside_is=False))
    print(x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot)
    halo_centre=[x_halo,y_halo]
    av_ring_select =  circular_mask(R-w_actual, imsize, (y_ring, x_ring), inside_is=False) *  \
                     ~circular_mask(R+w_actual, imsize, (y_ring, x_ring), inside_is=False)
    av_atom_select = ~circular_mask(bacg_i_cut, imsize, (av_yc, av_xc), inside_is=False)
    av_halo_select = ~av_ring_select * av_atom_select
    av_bacg_select =  circular_mask(bacg_i_cut, imsize, (av_yc, av_xc), inside_is=False) *  \
                     ~circular_mask(bacg_o_cut, imsize, (av_yc, av_xc), inside_is=False)
    atom_area   =   np.sum(av_atom_select)
    ring_area   =   np.sum(av_ring_select)
    halo_area   =   np.sum(av_halo_select)
    bacg_area   =   np.sum(av_bacg_select)
    offset      =   np.mean(np.array(od),where=av_bacg_select)
    anum1       =   anum.od_to_atoms(od, pixsize, sigma_eff, mag_factor)
    atom_total  =   np.sum(anum1*av_atom_select)
    ring_total  =   np.sum(anum1*av_ring_select)
    halo_total  =   np.sum(anum1*av_halo_select)
    bacg_total  =   np.sum(anum1*av_bacg_select)
    bacg_avg    =   bacg_total/bacg_area
    atom_bgcorr =   atom_total-bacg_avg * atom_area
    ring_bgcorr =   ring_total-bacg_avg * ring_area
    halo_bgcorr =   halo_total-bacg_avg * halo_area
    bacg_bgcorr =   bacg_total-bacg_avg * bacg_area
        # x,y=np.meshgrid(np.arange(512),np.arange(512))
        # plt.figure()
        # # plt.imshow(f2d.parab_ring([x_ring,y_ring,R,w,amp,bac])(x,y))
        # plt.imshow(f2d.gaussian2D([x_halo,y_halo,wmaj,wmin,amp,bac,rot])(x,y))
        # plt.figure()
        # plt.imshow(od*~av_ring_select_corr,vmax=0.8,cmap='seismic')
    '''
    Parabolic_ring
    xc   : x coordinate of the ring center
    yc   : y coordinate of the ring center
    R    : radius of the wing
    w    : 1/e^2 radius of the annulus
    amp  : max amplitude of the of the ring (defaults to 1 if not specified)
    bac  : background level (defaults to 0 if not specified)
    Gaussian 2D
    Parameter list must contain:
    xc   : x coordinate of the gaussian peak
    yc   : y coordinate of the gaussian peak
    wmaj : peak width along major axis
    wmin : peak width along minor axis
    amp  : amplitude of the of the peak (defaults to 1 if not specified)
    bac  : background level (defaults to 0 if not specified)
    rot  : angle of rotation (DEGREES!) of major axis relative to +x axis
    Fermi Dirac fitting
    xc  : x coordinate of the center
    xF  : Thomas-Fermi distance
    amp : amplitude of the of the peak
    bac : background level (defaults to 0 if None)
    q  : shape factor mu/(k_B * T)
    xc, xF, amp, bac, q
    '''
    n2d_r       =   circleav(anum1, centre_av)
    n2d_ring    =   circleav(anum1, ring_centre)
    n2d_halo    =   circleav(anum1, halo_centre)
    n2d_az      =   radialav(anum1, ring_centre, R-w_actual,  R+w_actual)

    # if immode=='ring_insitu':
    weights=np.where(n2d_halo[0] > 30, 1, 0) * np.where(110 > n2d_halo[0], 1, 0)
    try:
        unifit = f1d.fit1D((n2d_halo[1]-offset*od_to_atoms_conv)/um2_per_pix, f1d.fermi_dirac_dist_peak_1D, initp=uniinitp, bounds=unibounds, xdata=n2d_halo[0],  weights=weights, return_all=True) 
        print(unifit['x'])
        dict1[key]['uni_fit_params'].append(unifit['x'])
    except:
        dict1[key]['uni_fit_params'].append([0]*5)
    # uni_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(unifit['x'])
    # uni_fit_data = uni_fd_fitfunc(n2d_ring[0])
    dict1[key]['fileid'].append(fileid)
    dict1[key]['imaging_mode'].append(str(immode))
    dict1[key]['od'].append(od)
    dict1[key]['anum'].append(anum1)
    dict1[key]['offset'].append(offset)
    dict1[key]['n2d_r'].append(n2d_r)
    dict1[key]['n2d_ring'].append(n2d_ring)
    dict1[key]['n2d_halo'].append(n2d_halo)
    dict1[key]['n2d_az'].append(n2d_az)
    dict1[key]['ring_centre'].append(ring_centre)
    dict1[key]['halo_centre'].append(halo_centre)
    dict1[key]['ring_params'].append([R,w,amp_ring,bac_ring])
    dict1[key]['halo_params'].append([wmaj,wmin,amp_halo,bac_halo,rot])
    dict1[key]['centre_av'].append(centre_av)
    dict1[key]['atom_total'].append(atom_total)
    dict1[key]['ring_total'].append(ring_total)
    dict1[key]['halo_total'].append(halo_total)
    dict1[key]['bacg_total'].append(bacg_total)
    dict1[key]['atom_bgcorr'].append(atom_bgcorr)
    dict1[key]['ring_bgcorr'].append(ring_bgcorr)
    dict1[key]['halo_bgcorr'].append(halo_bgcorr)
    dict1[key]['bacg_bgcorr'].append(bacg_bgcorr)

def calc_quant_save_halo(flats,centre_av,procparams,dict1,key,fileid,immode):
    av_xc,av_yc=centre_av
    print('calculating abs_to_od'+key)
    od          =   abs_to_OD(flats, **procparams)
    print(fileid)
    # ring_guess=[av_xc,av_yc,(ring_o_cut+ring_i_cut)/2,(ring_o_cut-ring_i_cut)/2,2,0.5]
    halo_guess=[av_xc,av_yc,160,160,0.2,0,0]
    uniinitp =  [ 0.000,    80,    1.0,   0.000,    1.14]
    unibounds =((-0.001,    55,    0.5,  -0.001,    1),
                ( 0.001,   120,    2.0,   0.001,    10))
    # av_ring_select =  circular_mask(ring_i_cut, imsize, (av_yc, av_xc), inside_is=False) *  \
    #                  ~circular_mask(ring_o_cut, imsize, (av_yc, av_xc), inside_is=False)
    # x_ring,y_ring,R,w,amp_ring,bac_ring=f2d.fit2D(od, f2d.parab_ring, ring_guess)
    # print(x_ring,y_ring,R,w,amp_ring,bac_ring)
    # ring_centre=[x_ring,y_ring]
    # w_actual=np.sqrt(1+bac_ring/amp_ring)*w
    # av_ring_select_corr =  circular_mask(R-w_actual, imsize, (y_ring, x_ring), inside_is=False) *  \
    #               ~circular_mask(R+w_actual, imsize, (y_ring, x_ring), inside_is=False)
    x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot=f2d.fit2D(od, f2d.gaussian2D, halo_guess)#consider entire image for halo fitting 
    print(x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot)
    halo_centre=[x_halo,y_halo]
    # av_ring_select =  circular_mask(R-w_actual, imsize, (y_ring, x_ring), inside_is=False) *  \
    #                  ~circular_mask(R+w_actual, imsize, (y_ring, x_ring), inside_is=False)
    av_atom_select = ~circular_mask(bacg_i_cut, imsize, (av_yc, av_xc), inside_is=False)
    # av_halo_select = ~av_ring_select * av_atom_select
    av_bacg_select =  circular_mask(bacg_i_cut, imsize, (av_yc, av_xc), inside_is=False) *  \
                     ~circular_mask(bacg_o_cut, imsize, (av_yc, av_xc), inside_is=False)
    atom_area   =   np.sum(av_atom_select)
    # ring_area   =   np.sum(av_ring_select)
    # halo_area   =   np.sum(av_halo_select)
    bacg_area   =   np.sum(av_bacg_select)
    offset      =   np.mean(np.array(od),where=av_bacg_select)
    anum1       =   anum.od_to_atoms(od, pixsize, sigma_eff, mag_factor)
    atom_total  =   np.sum(anum1*av_atom_select)
    # ring_total  =   np.sum(anum1*av_ring_select)
    # halo_total  =   np.sum(anum1*av_halo_select)
    bacg_total  =   np.sum(anum1*av_bacg_select)
    bacg_avg    =   bacg_total/bacg_area
    atom_bgcorr =   atom_total-bacg_avg * atom_area
    # ring_bgcorr =   ring_total-bacg_avg * ring_area
    # halo_bgcorr =   halo_total-bacg_avg * halo_area
    bacg_bgcorr =   bacg_total-bacg_avg * bacg_area
        # x,y=np.meshgrid(np.arange(512),np.arange(512))
        # plt.figure()
        # # plt.imshow(f2d.parab_ring([x_ring,y_ring,R,w,amp,bac])(x,y))
        # plt.imshow(f2d.gaussian2D([x_halo,y_halo,wmaj,wmin,amp,bac,rot])(x,y))
        # plt.figure()
        # plt.imshow(od*~av_ring_select_corr,vmax=0.8,cmap='seismic')
    '''
    Parabolic_ring
    xc   : x coordinate of the ring center
    yc   : y coordinate of the ring center
    R    : radius of the wing
    w    : 1/e^2 radius of the annulus
    amp  : max amplitude of the of the ring (defaults to 1 if not specified)
    bac  : background level (defaults to 0 if not specified)
    Gaussian 2D
    Parameter list must contain:
    xc   : x coordinate of the gaussian peak
    yc   : y coordinate of the gaussian peak
    wmaj : peak width along major axis
    wmin : peak width along minor axis
    amp  : amplitude of the of the peak (defaults to 1 if not specified)
    bac  : background level (defaults to 0 if not specified)
    rot  : angle of rotation (DEGREES!) of major axis relative to +x axis
    Fermi Dirac fitting
    xc  : x coordinate of the center
    xF  : Thomas-Fermi distance
    amp : amplitude of the of the peak
    bac : background level (defaults to 0 if None)
    q  : shape factor mu/(k_B * T)
    xc, xF, amp, bac, q
    '''
    n2d_r       =   circleav(anum1, centre_av)
    # n2d_ring    =   circleav(anum1, ring_centre)
    n2d_halo    =   circleav(anum1, halo_centre)
    weights=np.where(n2d_halo[0] > 30, 1, 0) * np.where(110 > n2d_halo[0], 1, 0)
    unifit = f1d.fit1D((n2d_halo[1]-offset*od_to_atoms_conv)/um2_per_pix, f1d.fermi_dirac_dist_peak_1D, initp=uniinitp, bounds=unibounds, xdata=n2d_halo[0],  weights=weights, return_all=True) 
    print(unifit['x'])
    # uni_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(unifit['x'])
    # uni_fit_data = uni_fd_fitfunc(n2d_ring[0])
    dict1[key]['fileid'].append(fileid)
    dict1[key]['imaging_mode'].append(str(immode))
    dict1[key]['od'].append(od)
    dict1[key]['anum'].append(anum1)
    dict1[key]['offset'].append(offset)
    dict1[key]['n2d_r'].append(n2d_r)
    # dict1[key]['n2d_ring'].append(n2d_ring)
    dict1[key]['n2d_halo'].append(n2d_halo)
    # dict1[key]['ring_centre'].append(ring_centre)
    dict1[key]['halo_centre'].append(halo_centre)
    # dict1[key]['ring_params'].append([R,w,amp_ring,bac_ring])
    dict1[key]['halo_params'].append([wmaj,wmin,amp_halo,bac_halo,rot])
    dict1[key]['uni_fit_params'].append(unifit['x'])
    dict1[key]['centre_av'].append(centre_av)
    dict1[key]['atom_total'].append(atom_total)
    # dict1[key]['ring_total'].append(ring_total)
    # dict1[key]['halo_total'].append(halo_total)
    dict1[key]['bacg_total'].append(bacg_total)
    dict1[key]['atom_bgcorr'].append(atom_bgcorr)
    # dict1[key]['ring_bgcorr'].append(ring_bgcorr)
    # dict1[key]['halo_bgcorr'].append(halo_bgcorr)
    dict1[key]['bacg_bgcorr'].append(bacg_bgcorr)

def calc_quant_save_tof(flats,centre_av,procparams,dict1,key,fileid,immode,tof):
    av_xc,av_yc=centre_av
    print('calculating abs_to_od'+key)
    od          =   abs_to_OD(flats, **procparams)
    print(fileid)
    od_pk = find_peak_coords(od, smoothing_radius=15)
    roi = ROI(center=od_pk, size=(101,101))
    od_roi = od[roi.slices]
    # odC = od[ROI(center=od_pk, size=(51,51)).slices]
    odW=(np.exp(50 * (0.5*np.amax(od_roi) - od_roi)) + 1)**(-1)
    com1 = f2d.center_of_mass(odW)
    av_yc,av_xc=(od_pk[0]+com1[1]-50,od_pk[1]+com1[0]-50)
    roi.recenter((av_yc,av_xc))
    od_roi = od[roi.slices]
    
    #Taking a second run with a better slice
    odW=(np.exp(50 * (0.5*np.amax(od_roi) - od_roi)) + 1)**(-1)
    com1 = f2d.center_of_mass(odW)
    av_yc,av_xc=(av_yc+com1[1]-50,av_xc+com1[0]-50)
    roi.recenter((av_yc,av_xc))
    od_roi = od[roi.slices]
    # pk_yc,pk_xc=find_peak_coords(od, smoothing_radius=3)
    ring_guess=[av_xc,av_yc,(ring_o_cut+ring_i_cut)/2,(ring_o_cut-ring_i_cut)/2,2,0.5]
    x_ring,y_ring,R,w,amp_ring,bac_ring=f2d.fit2D(od, f2d.parab_ring, ring_guess)
    if tof<1000:
        av_xc,av_yc=x_ring,y_ring
    print(x_ring,y_ring,R,w,amp_ring,bac_ring)
    '''
# =============================================================================
#     # print(x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot)
#     # plt.figure()
#     # plt.imshow()
#     plt.figure()
#     plt.imshow(od,vmax=1.5,cmap='seismic')
#     plt.title(tof)
#     # plt.scatter(x_ring,y_ring,s=100,c='orange')
#     plt.scatter(av_xc,av_yc,s=100,c='black')
#     # plt.scatter(x_halo,y_halo,s=100,c='green')
#     # plt.scatter(pk_xc,pk_yc,s=100,c='green')
#     # plt.scatter(50,50,s=100,c='orange')
#     # plt.scatter(com1[0],com1[1],s=100,c='orange')
#     halo_guess=[av_xc,av_yc,80,80,0.2,0,0]
#     x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot=f2d.fit2D(od, f2d.gaussian2D, halo_guess)
#     print(x_halo,y_halo,wmaj,wmin,amp_halo,bac_halo,rot)
#     # halo_centre=[x_halo,y_halo]
#     roi = ROI(center=[y_halo,x_halo], size=(101,101))
#     od_roi = od[roi.slices]
#     plt.figure()
#     plt.imshow(od_roi,vmax=1.5,cmap='seismic')
#     plt.scatter(50,50,s=100,c='orange')
#     # plt.scatter(com1[0],com1[1],s=100,c='orange')
# =============================================================================
    '''
    av_atom_select = ~circular_mask(bacg_i_cut, imsize, (av_yc, av_xc), inside_is=False)
    av_bacg_select =  circular_mask(bacg_i_cut, imsize, (av_yc, av_xc), inside_is=False) *  \
                     ~circular_mask(bacg_o_cut, imsize, (av_yc, av_xc), inside_is=False)
    atom_area   =   np.sum(av_atom_select)
    bacg_area   =   np.sum(av_bacg_select)
    offset      =   np.mean(np.array(od),where=av_bacg_select)
    anum1       =   anum.od_to_atoms(od, pixsize, sigma_eff, mag_factor)
    atom_total  =   np.sum(anum1*av_atom_select)
    bacg_total  =   np.sum(anum1*av_bacg_select)
    bacg_avg    =   bacg_total/bacg_area
    atom_bgcorr =   atom_total-bacg_avg * atom_area
    bacg_bgcorr =   bacg_total-bacg_avg * bacg_area
    n2d_r       =   circleav(anum1, centre_av)
    n2d_tof     =   circleav(anum1, (av_xc,av_yc))
    n2d_az      =   radialav(anum1-offset, (av_xc,av_yc), 8,40)
    #Taking fourier transform of the radial density average
    p_spec_az   =   fftshift(np.abs(fft(n2d_az[1]-offset*od_to_atoms_conv)))
    shape       =   np.shape(n2d_az[1])[-1]
    p_spec_azfreq   =   fftshift(fftfreq(shape,2*np.pi/shape))
    dict1[key]['fileid'].append(fileid)
    dict1[key]['imaging_mode'].append(str(immode))
    dict1[key]['od'].append(od)
    dict1[key]['od_roi'].append(od_roi)
    dict1[key]['anum'].append(anum1)
    dict1[key]['offset'].append(offset)
    dict1[key]['n2d_r'].append(n2d_r)
    dict1[key]['n2d_tof'].append(n2d_tof)
    dict1[key]['tof'].append(tof)
    dict1[key]['tof_centre'].append([av_xc,av_yc])
    dict1[key]['n2d_az'].append(n2d_az)
    dict1[key]['p_spec_az'].append([p_spec_azfreq,p_spec_az])
    dict1[key]['centre_av'].append(centre_av)
    dict1[key]['atom_total'].append(atom_total)
    dict1[key]['bacg_total'].append(bacg_total)
    dict1[key]['atom_bgcorr'].append(atom_bgcorr)
    dict1[key]['bacg_bgcorr'].append(bacg_bgcorr)
    
# def od_to_atoms_per_pix(od):
#     return od_to_atoms(od, pixsize, sigma_eff, mag_factor)

# def od_to_atoms_per_um2(od):
#     return od_to_atoms(od, pixsize, sigma_eff, mag_factor) * um2_per_pix #incorrect:,should be a divide.



###############################################################################
# Good starting points for individual fitting for finding centers
###############################################################################

# =============================================================================
# parab2d_ring_guess = [231, 298, 28, 5, 2, 0.15]
# 
# gauss2d_halo_guess = [231,298, 160, 160, 0.2, 0, 0]                 
# 
# =============================================================================
##############################################################################
# 2D Fitted centers of ring and halo in group-averaged images
##############################################################################

# =============================================================================
# ring_centers = np.array([[230.49739721, 297.77615181],
#                         [231.11974744, 297.3005564 ],
#                         [231.00738079, 296.89414808],
#                         [230.76479179, 296.73463551]])
# 
# halo_centers = np.array([[233.0052278 , 310.27885947],
#                          [230.30937852, 307.32858402],
#                          [225.38676555, 313.89529168],
#                          [224.96788922, 313.56728653]])
# =============================================================================
