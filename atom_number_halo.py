# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:53:18 2024

@author: parth
"""

from lyse import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as col
import matplotlib.patches as ptch
try:
    from quagmire.utils.image_mask import circlemask
except:
    from quagmire.utils.image_mask import circular_mask
from quagmire.process.absorption import abs_to_OD
import quagmire.analysis.atomnumber as anum
import _0_common as com
from scipy.ndimage.filters import gaussian_filter1d as gfilt1d
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy import constants as const
from quagmire.fit import fit1d as f1d
# from quagmire.fit import fit2d as f2d
import quagmire.constants as qgconst
import copy
import os
import h5py
import timeit
start = timeit.default_timer()

cmap        =   col.get_cmap('jet')
plt.rcParams.update({'font.family':'sans-serif','mathtext.fontset':'stix','font.size': 12,'axes.labelsize':'large','lines.linewidth':5,'savefig.bbox':'tight','legend.loc' : 'upper right','legend.fontsize':'small'})
ncol        =   4 #Number of columns in the plots
ndigits     =   10 #Number of digits in plot titles       
# =============================================================================
# lyse variables
# =============================================================================
n_sequences =   None #number of sequences to import from loaded shots
df          =   data(n_sequences=n_sequences) #Pandas Dataframe imported from lyse
fi          =   df['filepath'] #filepaths of loaded shots
grouppara   =   'load_time' #parameter to group by
grouppara1  =   None#second parameter to group by
# grouppara1  =   'DMD_sequence_dir'#second parameter to group by
label       =   'temp'
off         =   [0,0] #should get rid of it, it was used check the accuracy of halo centre
imaging_mode=   os.path.commonprefix(list(df['imaging_mode'])) #common imaging mode for the shots
use_DMD     =   os.path.commonprefix([str(i) for i in list(df['use_DMD'])]) #common imaging mode for the shots
print(use_DMD)
if imaging_mode!='ring_insitu' or use_DMD!='False':
    # print('Mixed imaging modes found, be sure include imaging_mode in grouping parameters')
    raise Exception("Incorrect or mixed imaging modes found, please pick files of same imaging_mode")

# =============================================================================
# Data saving and retrieving settings
# =============================================================================
SAVE        =   False
RET         =   True            #Short for Retrieve from an existing file
SAVEFIG     =   True
RET         =   (not SAVE) and RET
SAVEFIG     =   SAVE or (SAVEFIG and RET)  #Savefig is set to False if neither Saving to HDF files or retieving results
print(SAVE, RET,SAVEFIG)
# =============================================================================
# if SAVE= True and RET is True, RET is set to False, new quants are saved, retrieve is not attempted
# if SAVE= False and RET is True, RET is set to True, a retrieve of quantities is attempted, failing which a save is not attempted
# if SAVE= True and RET is False, RET is set to False, new qunatities are saved
# if SAVE= False and RET is False, RET is set to False, new quantities are calculated and but not saved
# Only scenario in which anything is saved is when Save is True, Retrieve is ignored in that case.
# If Save is False, Retrieve is attempted based on choice, but nothing is saved.
# If either is True, the correct SAVE_DIR is calculated, 
# a folder named processed id calculated, if one doesn't exists, it's created, but RET is abandoned
# Best course of action to get desired results is to have only one of these to be True.
# Savefig Follows Save unless, both savefig and ret are true
'''
SAVE	RET	SAVEFIG		    SAVE	RET	    SAVEFIG
TRUE	TRUE	TRUE	-	TRUE	FALSE	TRUE
FALSE	TRUE	TRUE	-	FALSE	TRUE	TRUE
TRUE	FALSE	TRUE	-	TRUE	FALSE	TRUE
FALSE	FALSE	TRUE	-	FALSE	FALSE	FALSE
TRUE	TRUE	FALSE	-	TRUE	FALSE	TRUE
FALSE	TRUE	FALSE	-	FALSE	TRUE	FALSE
TRUE	FALSE	FALSE	-	TRUE	FALSE	TRUE
FALSE	FALSE	FALSE	-	FALSE	FALSE	FALSE
'''
# =============================================================================
SAVE_DIR    =   None
if SAVE is True or RET is True:
    if SAVE_DIR is None:
        SAVE_DIR=   os.path.commonpath(list(fi))
    print(SAVE_DIR)
    processed_folder_location=os.path.join(SAVE_DIR,"processed")
    print(processed_folder_location)
    if os.path.exists(processed_folder_location):
        print("Processed folder exists")
    else:
        print('Creating Processed folder.')
        os.mkdir(processed_folder_location)
        RET = False
    h5fpath=processed_folder_location + "\\insitu_ring_processed.h5"
    figures_folder_location=os.path.join(SAVE_DIR,"figures")
    print(figures_folder_location)
    if os.path.exists(figures_folder_location):
        print("Figures folder exists")
    else:
        print('Creating Figures folder.')
        os.mkdir(figures_folder_location)
  
# h5py.File(SAVE_DIR + "\\processed\\insitu_ring_processed.h5", 'a')
# =============================================================================
# Declaring empty arrays, common constants and file locations
# =============================================================================
shotlist    =   {}  #Results of a first pass over the data, shots are grouped by keys
ind         =   {} #individual shots and their properties
# indav       =   {}
keyav       =   {} #properties of averaged shot
infodict    =   {'od':[],'anum':[],'offset':[],'n2d_r':[],'centre_av':[],'fileid':[],'imaging_mode':[],
                 'halo_centre':[],'halo_params':[],'uni_fit_params':[],'n2d_halo':[],
                 'atom_total':[],'bacg_total':[],'atom_bgcorr':[],'bacg_bgcorr':[]}
try:
    centre_av   =   np.mean(df['cell_v','center'],axis=0)
except:
    centre_av   =   (com.ring_xc, com.ring_yc)
# centre_av   =   (233, 250) #initial guess for ring centre. Ring centres are pulled agian fro each image from the saved result from the cell_v script.

try:
    cmask   =   circlemask(radius = 50, arraysize = 512, dx=centre_av[0],dy=centre_av[1], invert=False) 
except:
    cmask   =   circular_mask(radius = 50, array_shape = 512, center=[centre_av[1],centre_av[0]], inside_is=True) 
procparams  =   {'min_signal'     : 0,
                'filter_dark'    : True,
                'Isat'           : 100,
                'norm_type'      : 'automatic',
                'exclusion_mask' : cmask}

# =============================================================================
# Getting raw data from all the shots, looping over them and assigning a key to every shot
# =============================================================================

for i in list(fi):
    run     =   Run(i)
    ser     =   data(i)
    print(i)
    key     =   str(ser[grouppara]).replace("/", "")
    #It is important to remove "/" from the key or else the when saving, the hdf5 
    #file creates series of a subfolders, if the key is a filepath
    if grouppara1 is not None:
        key+=   str(ser[grouppara1]).replace("/", "")
    print(key)
    atom_abs =  np.array(run.get_image('side','cell_v','MOT_abs atom_abs'), dtype='float')
    prob_refs=  np.array(run.get_image('side','cell_v','MOT_abs prob_ref'), dtype='float')
    dark_refs=  np.array(run.get_image('side','cell_v','MOT_abs dark_ref'), dtype='float')
    centre  =   ser['cell_v','center']
    # centre  =   np.array([233, 250]) #another place you can hard code the centre
    # fileid  =   str(ser['filepath'])[-35:-22]+str(ser['run number'])
    meta    =   run.get_attrs(group='/') #metedata of the hdf5 file. Has the following attributes:
    # {'n_runs': 36, 'run number': 8, 'run time': '20240203T044252', 'script_basename': 'master_script_V2', 'sequence_date': '2024-02-03', 'sequence_id': '20240203T043530_master_script_V2', 'sequence_index': 15}
    fileid  =   str(meta['sequence_id']).split("_",)[0]+str(meta['run number'])
    immode  =   str(ser['imaging_mode'])
    print(fileid)
    #creates a dict with key and raw images
    shotlist.setdefault(key,[]).append([atom_abs,prob_refs,dark_refs,centre,fileid,immode])

#The grouppara can be combined at this stage
if grouppara1 is not None:
    grouppara   =   grouppara+grouppara1
keyslist    =   list(shotlist.keys())
print(keyslist)
length_keys =   len(keyslist)

# =============================================================================
# Check if a set of values with the same key and shot id exists in the indicated hdf5 file.
# If that fails, Processing of raw data from individual shots to OD images, and other quantities
# like centre, atom numbers,offset
# =============================================================================

for keys in shotlist:
    print(keys)
    abs_av,probe_av,dark_av,centre_av =   np.mean(np.array(shotlist[keys])[:,0:4],axis=0)
    immode  =   np.unique(np.array(shotlist[keys])[:,5])
    keyav.setdefault(keys,copy.deepcopy(infodict))
    ind.setdefault(keys,copy.deepcopy(infodict))
    if RET is True:
        try:
            print('Attempting retrieve for key:'+str(keys))
            h5f=h5py.File(h5fpath, 'a')
            for prop in h5f[grouppara][keys].keys():
                keyav[keys][prop].append(h5f[grouppara][keys][prop]['av'][()])
                for n,i in enumerate(shotlist[keys]):
                    ind[keys][prop].append(h5f[grouppara][keys][prop][i[4]][()])
            #Strings are not saved in the hdf5 file, and have to be manually retrieved.
            keyav[keys]['fileid'].append('av')
            ind[keys]['fileid']=np.array(shotlist[keys])[:,4]
        except Exception as error:
            RET==False #If RET fails, the retrieve attempt is abandoned for the rest of the code and fresh calculations are performed
            keyav[keys]=copy.deepcopy(infodict)
            ind[keys]=copy.deepcopy(infodict)
            print("An exception occurred:", type(error).__name__) 
            print('Retrieve failed during key'+str(keys))
    if RET is False:
        com.calc_quant_save_halo((abs_av,probe_av,dark_av),centre_av,procparams,keyav,keys,'av',immode)
        # indav.setdefault(key,infodict.copy())
        for n,i in enumerate(shotlist[keys]):
            com.calc_quant_save_halo (i[0:3],centre_av,procparams,ind,keys,i[4],i[5])
            # com.calc_quant_save_halo([i[0],probe_av,dark_av],centre_av,procparams,indav,key)
if RET is True:
    try:
        h5f.close()
    except:
        "Failed in closing hdf5 file."
        

# =============================================================================
# Validate probe normalization / background elimination by taking radial average 
# =============================================================================
fig1=plt.figure(101, clear=True, dpi=200) ;plt.suptitle('Average then OD')
fig2=plt.figure(102, clear=True, dpi=200) ;plt.suptitle('OD then Average')
# fig3=plt.figure(103, clear=True, dpi=200) ;plt.suptitle('OD then Average')
fig4=plt.figure(104, clear=True, dpi=200) ;plt.title('Atom numbers')
fig5=plt.figure(105, clear=True, dpi=200) ;plt.title('Atoms per radial slice')
fig6=plt.figure(106, clear=True, dpi=200) ;plt.title('Azimuthally averaged density (halo)')
fig7=plt.figure(107, clear=True, dpi=200) ;plt.title('Peak 3D density')
fig8=plt.figure(108, clear=True, dpi=200) ;plt.suptitle('Peak 3D density (Image)')
fig9=plt.figure(109, clear=True, dpi=200) ;plt.suptitle('Fermi Energy (Image)')
fig10=plt.figure(110, clear=True, dpi=200) ;plt.suptitle('Critical Temp (Image)')
fig11=plt.figure(111, clear=True, dpi=200) ;plt.suptitle('Critical Temp (Image) (Zoomed In)')
fig12=plt.figure(112, clear=True, dpi=200) ;plt.title('Critical Temp')
for n,keys in enumerate(keyslist):
    print(keys)
    plt.figure(101)
    # plt.subplot(1,length_keys,n+1)
    plt.subplot(int(np.ceil(length_keys/ncol)),ncol,n+1)
    plt.gca().title.set_text(com.truncate_title(keys))
    # plt.gca().title.set_text(grouppara+'='+keys)
    imdata=keyav[keys]['od'][0]
    # R,w,amp_ring,bac_ring=keyav[keys]['ring_params'][0]
    # w_actual=np.sqrt(1+bac_ring/amp_ring)*w
    # ring_mask=(circular_mask(R-w_actual, (512,512), (keyav[keys]['ring_centre'][0][1], keyav[keys]['ring_centre'][0][0]), inside_is=True)+circular_mask(R+w_actual, (512,512), (keyav[keys]['ring_centre'][0][1], keyav[keys]['ring_centre'][0][0]),  inside_is=False))
    plt.imshow(imdata, cmap='seismic', vmin=0, vmax = 0.7,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    # plt.scatter(x=keyav[keys]['ring_centre'][0][0],y=keyav[keys]['ring_centre'][0][1], color='red')
    plt.scatter(x=keyav[keys]['halo_centre'][0][0]+off[0],y=keyav[keys]['halo_centre'][0][1]+off[1], color='yellow')
    ax=plt.gca()
    halofit = ptch.Ellipse(keyav[keys]['halo_centre'][0]+off,keyav[keys]['halo_params'][0][0],keyav[keys]['halo_params'][0][1],keyav[keys]['halo_params'][0][4], color='yellow', fill=False,ls='--')
    ax.add_patch(halofit)
    ringfitinner = ptch.Circle(keyav[keys]['halo_centre'][0], 35/com.um_per_pix, color='red', fill=False,ls='--')
    ringfitouter = ptch.Circle(keyav[keys]['halo_centre'][0], 45/com.um_per_pix, color='red', fill=False,ls='--')
    ax.add_patch(ringfitinner)
    ax.add_patch(ringfitouter)

    # plt.imshow(circular_mask(45/0.41,(512,512),(keyav[keys]['halo_centre'][0][1],keyav[keys]['ring_centre'][0][0]),inside_is=False)*
    #            circular_mask(55/0.41,(512,512),(keyav[keys]['halo_centre'][0][1],keyav[keys]['ring_centre'][0][0]),inside_is=True),cmap='Greys',alpha=0.5)
    
    plt.text(50,100,"Background OD: %.4f" % keyav[keys]['offset'][0], color='white')
    plt.text(50,150,"Peak OD: %.2f" % np.max(imdata), color='white')
    plt.xticks([]);plt.yticks([])
    
    plt.figure(102)
    plt.subplot(int(np.ceil(length_keys/ncol)),ncol,n+1)
    plt.gca().title.set_text(com.truncate_title(keys))
    imdata=np.mean(ind[keys]['od'],axis=0)
    plt.imshow(imdata, cmap='seismic', vmin=-1, vmax = 2.5,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.scatter(x=np.mean(ind[keys]['ring_centre'],axis=0)[0],y=np.mean(ind[keys]['ring_centre'],axis=0)[1], color='red')
    plt.scatter(x=np.mean(ind[keys]['halo_centre'],axis=0)[0],y=np.mean(ind[keys]['halo_centre'],axis=0)[1], color='yellow')
    # R,w,amp_ring,bac_ring=np.mean(ind[keys]['ring_params'],axis=0)
    # w_actual=np.sqrt(1+bac_ring/amp_ring)*w
    ax=plt.gca()
    halofit = ptch.Ellipse(np.mean(ind[keys]['halo_centre'],axis=0),np.mean(ind[keys]['halo_params'],axis=0)[0],np.mean(ind[keys]['halo_params'],axis=0)[1],np.mean(ind[keys]['halo_params'],axis=0)[4], color='yellow', fill=False,ls='--')
    ax.add_patch(halofit)

    plt.text(50,100,"Background OD: %.4f" % np.mean(ind[keys]['offset'],axis=0), color='white')
    plt.text(50,150,"Peak OD: %.2f" % np.max(imdata), color='white')
    plt.xticks([]);plt.yticks([])

# =============================================================================
#     plt.figure(103)
#     plt.subplot(1,2,1)
#     plt.imshow(keyav[keys]['od'])
#     plt.subplot(1,2,2)
#     plt.imshow()
# =============================================================================

    plt.figure(106)

    # =============================================================================
    # Attempting to fit to the the fermi dirac distribution
    # =============================================================================
    uniinitp =  [ 0.000,    80,    1.0,   0.000,    5]
    unibounds =((-0.001,    -10,    0.5,  -0.00001,    1),
                ( 0.001,   120,    2.0,   0.00001,    10))
    # xc  : x coordinate of the center
    # xF  : Thomas-Fermi distance
    # amp : amplitude of the of the peak
    # bac : background level (defaults to 0 if None)
    # q  : shape factor mu/(k_B * T)
    # xc, xF, amp, bac, q
    rvals=np.array(keyav[keys]['n2d_halo'])[0][0]
    data=(np.mean(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))
    unifit = f1d.fit1D(data/com.um2_per_pix, f1d.fermi_dirac_dist_peak_1D, initp=uniinitp, bounds=unibounds, xdata=np.array(ind[keys]['n2d_halo'])[0][0], weights= np.where(110 > rvals, 1, 0)*np.where(rvals > 42, 1, 0), return_all=True) # The output of the fit1D func saved in unifit is a set of output of return variables (same as output for least_squares func) out of which only values of 'x' are important and so we print and save them for future use.
    print(['centre', 'radius', 'amp','bac','q'])
    print(unifit['x']) # 'x' includes xc  : x coordinate of the center, xF  : Thomas-Fermi distance, q  : shape factor mu/(k_B * T), amp : amplitude of the of the peak, and bac : background level (defaults to 0 if None).
    uni_fd_fitfunc = f1d.fermi_dirac_dist_peak_1D(unifit['x'])  # Returns the FD function using the values of arguments fed and saves the callable function in uni_fd_fitfunc
    uni_fit_data = uni_fd_fitfunc(rvals) # Applying the saved function to rvals which are the x values and save the resultant y values in uni_fit_data
    plt.plot(rvals,uni_fit_data,ls='--',color=cmap(n/length_keys),label='fit',lw=2)
    # plt.plot(rvals,10*uni_fit_data,ls='--',color='blue',label='fitzoomed',lw=2)
    plt.text(60, 1-n/10, "width = %.1f $\mu$m" % unifit['x'][1],color=cmap(n/length_keys))
    plt.text(70, 1-n/10, " $ \\beta \cdot \mu$ = %.2f" % unifit['x'][4],color=cmap(n/length_keys))
    # plt.plot(rvals, 10*data/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1)
    # plt.fill_between(rvals,
    #                  10*(data+np.std(np.array(ind[keys]['n2d_halo'])[:,1],axis=0))/com.um2_per_pix, 
    #                  10*(data-np.std(np.array(ind[keys]['n2d_halo'])[:,1],axis=0))/com.um2_per_pix, 
    #                  color='grey',alpha=0.5, linewidth = 1)       # Zoomed in area of the plot of interest

    plt.plot(np.array(keyav[keys]['n2d_halo'])[0][0], (np.array(keyav[keys]['n2d_halo'])[0][1]-com.od_to_atoms_conv*keyav[keys]['offset'][0])/com.um2_per_pix, color=cmap(n/length_keys), linewidth = 1,label=com.truncate_title(keys)+'keyav')   # Original n2d_halo plot
    # plt.plot(np.array(ind[keys]['n2d_halo'])[0][0], (np.mean(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=com.truncate_title(keys)+'ind')
    plt.fill_between(np.array(ind[keys]['n2d_halo'])[0][0],
                     (np.mean(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)+np.std(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))/com.um2_per_pix, 
                     (np.mean(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)-np.std(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))/com.um2_per_pix, 
                     color='grey',alpha=0.5, linewidth = 1)
    plt.xlabel('r ($\mu$m)')
    plt.ylabel('atoms/$\mu$m$^{2}$')
    plt.axhline(y=0,ls='--',lw=1)
    plt.xticks()
    
# =============================================================================
#     plt.figure(107)
#     # plt.subplot(length_keys,1,n+1)
#     # x,y=np.meshgrid(np.arange(512),np.arange(512))
#     # plt.imshow(f2d.parab_ring([x_ring,y_ring,R,w,amp,bac])(x,y))
#     # plt.imshow(f2d.gaussian2D(list(keyav[keys]['halo_centre'][0])+list(keyav[keys]['halo_params'][0]))(x,y))
#     # plt.plot(rvals, np.fft.fft(data))
# =============================================================================
# =============================================================================
#     plt.figure(107)
#     n2d1=[]
#     for j,i in enumerate(ind[keys]['fileid']):
#         imdata=ind[keys]['anum'][j]
#         halo_centre=np.array(keyav[keys]['halo_centre'][0])+off
#         rvals,n2d=com.circleav(imdata, halo_centre)
#         n2d=n2d-keyav[keys]['offset'][0]
#         n2d1.append(n2d)
#         plt.plot(rvals, n2d,ls='dotted',label=i)
#     plt.plot(rvals,np.mean(n2d1,axis=0))
#     plt.plot(rvals[140:],gfilt1d(np.mean(n2d1,axis=0)[140:]*10,3))
# =============================================================================
    plt.figure(107)
    rvals=np.mean(np.array(ind[keys]['n2d_halo'])[:,0],axis=0)
    data=(np.mean(np.array(ind[keys]['n2d_halo'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))*1e12/com.um2_per_pix
    mmol=qgconst.m6
    sheet_power=40e-3#Watts
    omega=1.6e3*np.sqrt(sheet_power/50e-3)
    a=np.sqrt(const.hbar/mmol/omega)
    n3d=((512/81/(np.pi**5))**0.25)*((data**(0.75))/(a**(1.5)))/1e18
    plt.plot(rvals, n3d, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=com.truncate_title(keys)+'ind')
    plt.xlabel('r ($\mu$m)')
    plt.ylabel('atoms/$\mu$m$^{3}$')
    
    plt.figure(108)
    plt.subplot(int(np.ceil(length_keys/ncol)),ncol,n+1)
    plt.gca().title.set_text(com.truncate_title(keys))
    imdata=np.clip(keyav[keys]['anum'][0]-com.od_to_atoms_conv*keyav[keys]['offset'][0],0,None)*1e12/com.um2_per_pix
    a=np.sqrt(const.hbar/mmol/omega)
    imdata=((512/81/(np.pi**5))**0.25)*((imdata**(0.75))/(a**(1.5)))/1e18
    plt.imshow(imdata, cmap='seismic', vmin=0, vmax = 1.5,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.xlabel('r ($\mu$m)')
    plt.ylabel('atoms/$\mu$m$^{3}$')
    
    plt.figure(109)
    plt.subplot(int(np.ceil(length_keys/ncol)),ncol,n+1)
    plt.gca().title.set_text(com.truncate_title(keys))
    EFermi=(const.hbar**2/2/mmol)*((3*(np.pi**2)*imdata*1e18)**(2/3))#T = 0 ideal (scattering length a = 0) fermion equation of state from Yanping Thesis supplemental material
    kF=np.sqrt(2*mmol*EFermi)/const.hbar
    scat_length=-20516.4
    kFa=kF*scat_length
    plt.imshow(-1*kFa, cmap='seismic', vmin=0, vmax = 7e10,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.xlabel('r ($\mu$m)')
    plt.ylabel('-k$_F$a')
    
    plt.figure(110)
    plt.subplot(int(np.ceil(length_keys/ncol)),ncol,n+1)
    plt.gca().title.set_text(com.truncate_title(keys))
    TFermi=EFermi/const.Boltzmann
    Tc=0.277*TFermi*np.exp(-1*np.pi/kFa)
    # where_are_NaNs = np.isnan(Tc)
    # Tc[where_are_NaNs] = 0
    plt.imshow(Tc, cmap='seismic', vmin=0, vmax = 6.5e-8,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.xlabel('r ($\mu$m)')
    plt.ylabel('T$_c$')
    
    plt.figure(111)
    plt.subplot(int(np.ceil(length_keys/ncol)),ncol,n+1)
    plt.gca().title.set_text(com.truncate_title(keys))
    TFermi=EFermi/const.Boltzmann
    Tc=0.277*TFermi*np.exp(-1*np.pi/kFa)
    # where_are_NaNs = np.isnan(Tc)
    # Tc[where_are_NaNs] = 0
    roi=(slice(int(keyav[keys]['halo_centre'][0][0])-100,int(keyav[keys]['halo_centre'][0][0])+100),slice(int(keyav[keys]['halo_centre'][0][1])-100,int(keyav[keys]['halo_centre'][0][1])+100))
    plt.imshow(Tc[roi], cmap='jet', vmin=0, vmax = 6.5e-8,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    # plt.xlabel('r ($\mu$m)')
    plt.ylabel('T$_c$')
    
    plt.figure(112)
    EFermi=(const.hbar**2/2/mmol)*((3*(np.pi**2)*n3d*1e18)**(2/3))#T = 0 ideal (scattering length a = 0) fermion equation of state from Yanping Thesis supplemental material
    kF=np.sqrt(2*mmol*EFermi)/const.hbar
    kFa=kF*scat_length
    TFermi=EFermi/const.Boltzmann
    Tc=0.277*TFermi*np.exp(-1*np.pi/kFa)
    plt.plot(rvals, Tc*1e9, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=com.truncate_title(keys)+'ind')
    plt.xlabel('r ($\mu$m)')
    plt.ylabel('T$_c$ (nK)')
    
    plt.figure(105)
    plt.plot(np.array(keyav[keys]['n2d_halo'])[0][0], (np.array(keyav[keys]['n2d_halo'])[0][1]-com.od_to_atoms_conv*keyav[keys]['offset'][0])*(2*np.pi)*np.array(keyav[keys]['n2d_halo'])[0][0]/com.um2_per_pix, color=cmap(n/length_keys), linewidth = 1,label=com.truncate_title(keys)+'keyav')
    # plt.plot(np.array(ind[keys]['n2d_ring'])[0][0], (np.mean(np.array(ind[keys]['n2d_ring'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))*(2*np.pi)*np.array(ind[keys]['n2d_ring'])[0][0]/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=com.truncate_title(keys)+'ind')
    plt.xlabel('r ($\mu$m)')
    plt.ylabel('2$\pi$r n$_{2D}$($\mu$m$^{-1}$)')
    plt.axhline(y=0,ls='--',lw=1)
    plt.xticks()
    
    plt.figure(104, clear=True)
    plt.subplot(1,3,1)
    plt.scatter(n,keyav[keys]['atom_bgcorr'][0],label=com.truncate_title(keys)+'keyav')
    plt.errorbar(x=n,y=np.mean(ind[keys]['atom_bgcorr']),yerr=np.std(ind[keys]['atom_bgcorr']))
    plt.title('Atom numbers')
    plt.legend(title=grouppara)

fig5.legend(title=grouppara);fig6.legend(title=grouppara)
fig7.legend();fig11.legend()
plt.tight_layout()

if SAVEFIG:
    fig1.savefig(os.path.join(figures_folder_location,"Averaged OD Images"),dpi=200)
    fig2.savefig(os.path.join(figures_folder_location,"OD then Average Images"),dpi=200)
    # fig3.savefig(os.path.join(figures_folder_location,"Azimuthally averaged density"),dpi=200)
    fig4.savefig(os.path.join(figures_folder_location,"Atom numbers"),dpi=200)
    fig5.savefig(os.path.join(figures_folder_location,"Atoms per radial slice"),dpi=200)
    fig6.savefig(os.path.join(figures_folder_location,"Azimuthally averaged density (halo)"),dpi=200)
    fig7.savefig(os.path.join(figures_folder_location,"Peak 3D density"),dpi=200)
    fig8.savefig(os.path.join(figures_folder_location,"Peak 3D density (Image)"),dpi=200)
    fig9.savefig(os.path.join(figures_folder_location,"Fermi Energy (Image)"),dpi=200)
    fig10.savefig(os.path.join(figures_folder_location,"Critical Temp (Image)"),dpi=200)
    fig11.savefig(os.path.join(figures_folder_location,"Critical Temp (Image) (Zoomed In)"),dpi=200)
    fig12.savefig(os.path.join(figures_folder_location,"Critical Temp"),dpi=200)
if SAVE:
    #saving the figures
    com.dict_to_h5(h5fpath, keyav, grouppara)
    com.dict_to_h5(h5fpath, ind, grouppara)
stop = timeit.default_timer()
print('Time: ', stop - start)
plt.show()
# =============================================================================
# Find the ring center in each group-averaged image using a parabolic ring fit
# =============================================================================

# =============================================================================
# de-weighting the data in the ring region, fit the halo with a 2D gaussian
# =============================================================================

# =============================================================================
# Convert atom number per pixel to 2D column density (atoms per um^2)
# =============================================================================

# =============================================================================
# Extract atomic column density (/um^2) versus radius at each Bfield value
# We'll do this twice: centered on the ring, and centered on the halo.
# =============================================================================