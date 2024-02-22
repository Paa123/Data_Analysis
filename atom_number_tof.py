# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:08:44 2024

@author: parth

A new file has been created for TOF images. I could just mod the ring file, but 
I'll have to change the infodict, change the figures, which is the whole file anyways.

Goals from the tof analysis script:
    1.extract atom numbers, offset, centre? (need to think), radial/azimuthal average
    2.strip out the temperature estimation bit
    3.some way of removing the initial density distribution
    4. find power spectrum of the azimuthal density distribution
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
max_figs    =   12       
# =============================================================================
# lyse variables
# =============================================================================
n_sequences =   None #number of sequences to import from loaded shots
df          =   data(n_sequences=n_sequences) #Pandas Dataframe imported from lyse
fi          =   df['filepath'] #filepaths of loaded shots
grouppara   =   'load_time' #parameter to group by
grouppara1  =   'sheet_tof'#second parameter to group by
# label       =   'temp'
off         =   [0,0] #should get rid of it, it was used check the accuracy of halo centre
imaging_mode=   os.path.commonprefix(list(df['imaging_mode'])) #common imaging mode for the shots
print(imaging_mode)
if imaging_mode!='jump_field':
    # print('Mixed imaging modes found, be sure include imaging_mode in grouping parameters')
    raise Exception("Incorrect or mixed imaging modes found, please pick files of same imaging_mode")

# =============================================================================
# Data saving and retrieving settings
# =============================================================================
SAVE        =   False
RET         =   True            #Short for Retrieve from an existing file
SAVEFIG     =   False
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
infodict    =   {'od':[],'anum':[],'offset':[],'n2d_r':[],'n2d_tof':[],'n2d_az':[],'centre_av':[],'fileid':[],'imaging_mode':[],'tof':[],
                 'tof_centre':[],'atom_total':[],'bacg_total':[],'atom_bgcorr':[],'bacg_bgcorr':[],'od_roi':[],'p_spec_az':[]}
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
    key     =   str(ser[grouppara])
    if grouppara1 is not None:
        key+=   str(int(ser[grouppara1]*1e6)).replace("/", "")
    print(key)
    atom_abs =  np.array(run.get_image('side','cell_v','MOT_abs atom_abs'), dtype='float')
    prob_refs=  np.array(run.get_image('side','cell_v','MOT_abs prob_ref'), dtype='float')
    dark_refs=  np.array(run.get_image('side','cell_v','MOT_abs dark_ref'), dtype='float')
    centre  =   ser['cell_v','center']
    tof     =   int(ser['sheet_tof']*1e6)
    # centre  =   np.array([233, 250]) #another place you can hard code the centre
    fileid  =   str(ser['filepath'])[-35:-22]+str(ser['run number'])
    immode  =   str(ser['imaging_mode'])
    print(fileid)
    #creates a dict with key and raw images
    shotlist.setdefault(key,[]).append([atom_abs,prob_refs,dark_refs,centre,tof,fileid,immode])

#Once more complex grouping becomes necessary, te grouppara can be combined at this stage
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
    abs_av,probe_av,dark_av,centre_av,tof =   np.mean(np.array(shotlist[keys])[:,0:5],axis=0)
    immode  =   np.unique(np.array(shotlist[keys])[:,5])
    keyav.setdefault(keys,copy.deepcopy(infodict))
    ind.setdefault(keys,copy.deepcopy(infodict))
    if RET is True:
        # try:
        print('Attempting retrieve for key:'+str(keys))
        h5f=h5py.File(h5fpath, 'a')
        for prop in h5f[grouppara][keys].keys():
            keyav[keys][prop].append(h5f[grouppara][keys][prop]['av'][()])
            for n,i in enumerate(shotlist[keys]):
                ind[keys][prop].append(h5f[grouppara][keys][prop][i[5]][()])
        #Strings are not saved in the hdf5 file, and have to be manually retrieved.
        keyav[keys]['fileid'].append('av')
        ind[keys]['fileid']=np.array(shotlist[keys])[:,5]
        # except Exception as error:
        #     RET==False #If RET fails, the retrieve attempt is abandoned for the rest of the code and fresh calculations are performed
        #     keyav[keys]=copy.deepcopy(infodict)
        #     ind[keys]=copy.deepcopy(infodict)
        #     print("An exception occurred:", type(error).__name__) 
        #     print('Retrieve failed during key'+str(keys))
    if RET is False:
        com.calc_quant_save_tof((abs_av,probe_av,dark_av),centre_av,procparams,keyav,keys,'av',immode,tof)
        # indav.setdefault(key,infodict.copy())
        for n,i in enumerate(shotlist[keys]):
            com.calc_quant_save_tof (i[0:3],centre_av,procparams,ind,keys,i[5],i[6],i[4])
            # com.calc_quant_save_ring([i[0],probe_av,dark_av],centre_av,procparams,indav,key)
if RET is True:
    try:
        h5f.close()
    except:
        "Failed in closing hdf5 file."
    
# =============================================================================
# Validate probe normalization / background elimination by taking radial average 
# =============================================================================
fig1=[];fig2=[]
for i in range(int(np.ceil((length_keys)/max_figs))):
    x=plt.figure(100+i, clear=True, dpi=200)
    print(f'opening image {100+i}')
    fig1.append(x);plt.suptitle('OD'+str(i))
for i in range(int(np.ceil((length_keys)/max_figs))):
    print(f'opening image {200+i}')
    x=plt.figure(200+i, clear=True, dpi=200)
    fig2.append(x);plt.suptitle('OD zoomed in centred'+str(i))
# fig2=plt.figure(200, clear=True, dpi=200) ;plt.suptitle('OD zoomed in centred')
fig3=plt.figure(300, clear=True, dpi=200) ;plt.title('Azimuthally averaged density')
fig4=plt.figure(400, clear=True, dpi=200) ;plt.title('Atom numbers')
fig5=plt.figure(500, clear=True, dpi=200) ;plt.title('Atoms per radial slice')
fig6=plt.figure(600, clear=True, dpi=200) ;plt.title('Radially averaged density')
fig7=plt.figure(700, clear=True, dpi=200) ;plt.title('Radially averaged power spectrum')
for n,keys in enumerate(keyslist):
    print(keys)
    plt.figure(100+int(np.floor(n/max_figs)))
    plt.subplot(min(int(np.ceil((length_keys)/ncol)),int(max_figs/ncol)),ncol,(n)%max_figs+1)
    plt.gca().title.set_text(grouppara+'='+keys)
    imdata=keyav[keys]['od'][0]
    plt.imshow(imdata, cmap='seismic', vmin=0, vmax = 1.5,  interpolation=None)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.scatter(x=keyav[keys]['tof_centre'][0][0],y=keyav[keys]['tof_centre'][0][1], color='yellow')
    ax=plt.gca()
    ringfitouter = ptch.Circle(keyav[keys]['tof_centre'][0], 40, color='red', fill=False,ls='--')
    ringfitinner = ptch.Circle(keyav[keys]['tof_centre'][0], 8, color='red', fill=False,ls='--')
    ax.add_patch(ringfitinner)
    ax.add_patch(ringfitouter)
    plt.text(50,100,"Background OD: %.4f" % keyav[keys]['offset'][0], color='white')
    plt.text(50,150,"Peak OD: %.2f" % np.max(imdata), color='white')
    plt.xticks([]);plt.yticks([])
    
    plt.figure(200+int(np.floor((n)/max_figs)))
    plt.subplot(min(int(np.ceil((length_keys)/ncol)),int(max_figs/ncol)),ncol,(n)%max_figs+1)
    plt.gca().title.set_text(grouppara+'='+keys)
    imdata=keyav[keys]['od_roi'][0]
    plt.imshow(imdata, cmap='seismic', vmin=0, vmax = 1.5,  interpolation=None)
    plt.scatter(x=50,y=50, color='yellow')

    plt.figure(300)
    rvals=np.array(keyav[keys]['n2d_tof'])[0][0]  #List of radial values at which we want to plot the azimuthally averaged density
    # data=(np.mean(np.array(ind[keys]['n2d_tof'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))  # data = average of all n2d_r values of ind shots - average of offset OD * atoms per OD. Offset OD is calculated by the averaging the OD near the edge of an OD image. It should be zero and so if it is not, then there is clearly an offset which need to ber incorporated in actual data.
    plt.plot(rvals, (np.array(keyav[keys]['n2d_tof'])[0][1]-com.od_to_atoms_conv*keyav[keys]['offset'][0])/com.um2_per_pix, color=cmap(n/length_keys), linewidth = 1,label=keys+'keyav') # plotting y vs x where x=rvals of keyav shots in um and y= (n2d_r values of keyav shots- offset OD values * atoms per OD)/um^2 per pixel. n2d_r is in atoms/pixel and OD is in OD/pixel and so y is in atoms/um^2.
    # Using color map in such a way so that we have control over it and it doesn't repeat itself. The extremes change between 0 and 1 so any fraction will give a different shade of color as long as it is less than 1.
    # plt.plot(rvals, data/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=keys+'ind')
    # plt.fill_between(rvals,
    #                  (data+np.std(np.array(ind[keys]['n2d_tof'])[:,1],axis=0))/com.um2_per_pix, 
    #                  (data-np.std(np.array(ind[keys]['n2d_tof'])[:,1],axis=0))/com.um2_per_pix, 
    #                  color='grey',alpha=0.5, linewidth = 1)     #Basically error bars
    plt.xlabel('r ($\mu$m)')
    plt.ylabel('atoms/$\mu$m$^{2}$')
    plt.axhline(y=0,ls='--',lw=1)  #adding a horizontal dotted line along the x axis
    plt.xticks()

    plt.figure(500)
    plt.plot(np.array(keyav[keys]['n2d_tof'])[0][0], (np.array(keyav[keys]['n2d_tof'])[0][1]-com.od_to_atoms_conv*keyav[keys]['offset'][0])*(2*np.pi)*np.array(keyav[keys]['n2d_tof'])[0][0]/com.um2_per_pix, color=cmap(n/length_keys), linewidth = 1,label=keys+'keyav')
    # plt.plot(np.array(ind[keys]['n2d_tof'])[0][0], (np.mean(np.array(ind[keys]['n2d_tof'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))*(2*np.pi)*np.array(ind[keys]['n2d_tof'])[0][0]/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=keys+'ind')
    plt.xlabel('r ($\mu$m)')
    plt.ylabel('2$\pi$r n$_{2D}$($\mu$m$^{-1}$)')
    plt.axhline(y=0,ls='--',lw=1)
    plt.xticks()
    
    plt.figure(400, clear=True)
    # plt.subplot(1,3,1)
    plt.scatter(n,keyav[keys]['atom_bgcorr'][0],label=keys+'keyav')
    # plt.errorbar(x=n,y=np.mean(ind[keys]['atom_bgcorr']),yerr=np.std(ind[keys]['atom_bgcorr']),label=keys+'ind')
    plt.title('Total Atom Number')
    plt.legend(title=grouppara)
    
    plt.figure(600)
    rvals=np.array(keyav[keys]['n2d_az'])[0][0]  #List of azimuthal values at which we want to plot the azimuthally averaged density
    # check=np.array(ind[keys]['n2d_az'])[:,1]
    data=(np.mean(np.array(ind[keys]['n2d_az'])[:,1],axis=0)-com.od_to_atoms_conv*np.mean(ind[keys]['offset'],axis=0))  # data = average of all n2d_r values of ind shots - average of offset OD * atoms per OD. Offset OD is calculated by the averaging the OD near the edge of an OD image. It should be zero and so if it is not, then there is clearly an offset which need to ber incorporated in actual data.
    plt.plot(rvals, (np.array(keyav[keys]['n2d_az'])[0][1]-com.od_to_atoms_conv*keyav[keys]['offset'][0])/com.um2_per_pix, color=cmap(n/length_keys), linewidth = 1,label=com.truncate_title(keys)+'keyav') # plotting y vs x where x=rvals of keyav shots in um and y= (n2d_r values of keyav shots- offset OD values * atoms per OD)/um^2 per pixel. n2d_r is in atoms/pixel and OD is in OD/pixel and so y is in atoms/um^2.
    # Using color map in such a way so that we have control over it and it doesn't repeat itself. The extremes change between 0 and 1 so any fraction will give a different shade of color as long as it is less than 1.
    # plt.plot(rvals, data/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1,label=com.truncate_title(keys)+'ind')
    plt.fill_between(rvals,
                      (data+np.std(np.array(ind[keys]['n2d_az'])[:,1],axis=0))/com.um2_per_pix, 
                      (data-np.std(np.array(ind[keys]['n2d_az'])[:,1],axis=0))/com.um2_per_pix, 
                      color='grey',alpha=0.5, linewidth = 1)     #Basically error bars
    # plt.plot(rvals[100:], 10*gfilt1d(data[100:],2)/com.um2_per_pix, color=cmap((2*n+1)/2/length_keys), linewidth = 1)  # Using gaussian filter to smooth out the plot.
    # plt.fill_between(rvals[100:],
    #                  10*(data[100:]+np.std(np.array(ind[keys]['n2d_az'])[:,1],axis=0)[100:])/com.um2_per_pix, 
    #                  10*(data[100:]-np.std(np.array(ind[keys]['n2d_az'])[:,1],axis=0)[100:])/com.um2_per_pix,   
    #                  color='grey',alpha=0.5, linewidth = 1)    #Zooming out the area of interest
    plt.xlabel('Angle (radians)')
    plt.ylabel('atoms/$\mu$m$^{2}$')
    plt.axhline(y=0,ls='--',lw=1)  #adding a horizontal dotted line along the x axis
    plt.xticks()

    
    plt.figure(700+int(keys[0]),clear=True)
    plt.plot(np.array(keyav[keys]['p_spec_az'])[0][0], (np.array(keyav[keys]['p_spec_az'])[0][1]), linewidth = 1,label=keys+'keyav')
    # plt.fill_between(rvals,
    #                   (data+np.std(np.array(ind[keys]['n2d_az'])[:,1],axis=0))/com.um2_per_pix, 
    #                   (data-np.std(np.array(ind[keys]['n2d_az'])[:,1],axis=0))/com.um2_per_pix, 
    #                   color='grey',alpha=0.5, linewidth = 1)     #Basically error bars
    plt.xlabel('Angle (radians)')
    plt.ylabel('Absolute Value of power spectrum')
    plt.axhline(y=0,ls='--',lw=1)
    plt.ylim(top=2)
    plt.xticks()
    plt.legend()

fig3.legend(title=grouppara);fig5.legend(title=grouppara);fig6.legend(title=grouppara);fig7.legend(title=grouppara)
plt.tight_layout()

if SAVEFIG:
    for n,i in enumerate(fig1):
        i.savefig(os.path.join(figures_folder_location,"Averaged OD Images"+str(n)),dpi=200)
    for n,i in enumerate(fig2):
        i.savefig(os.path.join(figures_folder_location,"OD zoomed in centred"+str(n)),dpi=200)
    # fig1.savefig(os.path.join(figures_folder_location,"Averaged OD Images"),dpi=200)
    # fig2.savefig(os.path.join(figures_folder_location,"OD zoomed in centred"),dpi=200)
    fig3.savefig(os.path.join(figures_folder_location,"Azimuthally averaged density"),dpi=200)
    fig4.savefig(os.path.join(figures_folder_location,"Atom numbers"),dpi=200)
    fig5.savefig(os.path.join(figures_folder_location,"Atoms per radial slice"),dpi=200)
    fig6.savefig(os.path.join(figures_folder_location,"Radially averaged density"),dpi=200)
    fig7.savefig(os.path.join(figures_folder_location,"Radially averaged power spectrum"),dpi=200)

if SAVE:
    #saving the figures
    com.dict_to_h5(h5fpath, keyav, grouppara)
    com.dict_to_h5(h5fpath, ind, grouppara)
stop = timeit.default_timer()
print('Time: ', stop - start)
plt.show()
