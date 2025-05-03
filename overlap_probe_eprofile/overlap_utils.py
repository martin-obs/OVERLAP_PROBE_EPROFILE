#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions called in by other modules

@author martin osborne: martin.osborne@metoffice.gov.uk
"""

import numpy as np
import pandas as pd
import scipy.signal as ss
import netCDF4 as nc
import os
import datetime
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import colors

def conv2d ( x , direction = None ) :

    """Finds signal gradient along the stated direction using convolution with a 
    Sobel operator. Matlab's conv2 and Python's scipy.signal.convolve2d behave 
    slightly differently. This function reproduces the behaviour of the Matlab 
    function. Used in 'check_grads' to match the Matlab code results.
    
    Parameters
    ----------
    
    x : 2D array of floats
        input signal 
    direction : str
        direction in which grdient is to be calculated. 'x' or 'y'
    
    Returns
    -------
    gradient : 2D array of floats
        gradient of input signal along "direction""
        
    See also
    --------
    overlap_probe_eprofile.pre_checks.check_grads
    

    """
    
    if direction == 'y' :
        
        grad = np.asarray ( [ 1 , 0 , -1 , 2 , 0 , -2 , 1 , 0 , -1 ] ).reshape ( ( 3 , 3 ) )
        
    elif direction == 'x' :

        grad = np.asarray ( [ 1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1 ] ).reshape ( ( 3 , 3 ) )

    return np.rot90 ( ss.convolve2d ( np.rot90 ( x , 2 ) , np.rot90 ( grad , 2 ) , mode = 'same' ) , 2 )

def simple_linear_fit ( n , x , y ) :
    
    """
    
    Calculates the intercepts (alphas) and slopes (betas) of a simple linear fit 
    to the unmasked values in each column of masked array y. 
        
    Parameters
    ----------
    
    n : 1D array
        total lengths of each unmasked window
    x : 2D masked array
        altitude array with masked values defining fitting windows
    y : 2D masked array
        data array - regression is done on columns
              
    Returns
    -------
    
    alpha : 1D array of floats
        intercepts of regression
    beta : 1D array of floats
        slopes of regression
        
    """

    x = np.ma.masked_invalid ( x ) 
    
    y = np.ma.masked_invalid ( y )
       
    beta = (   n * np.ma.sum ( ( x * y ) , axis = 0 ) - np.ma.sum ( x , axis = 0 ) * np.ma.sum ( y , axis = 0 ) ) / ( n * np.ma.sum (  ( x * x ) , axis = 0 ) - ( np.ma.sum ( x , axis = 0  ) * np.ma.sum ( x, axis = 0  ) ) )
    
    alpha = ( ( 1 / n ) * np.ma.sum ( y , axis = 0 ) ) - ( ( 1 / n ) * beta * np.ma.sum ( x , axis = 0 ) )
    
    return  alpha , beta

def conv3d ( x , direction = None ) :

    """Finds the signal gradient along a stated direction for each layer of a 3D 
    array using convolution with a Sobel operator. Matlab's convolve and Python's 
    scipy.signal.convolve behave slightly differently. This function reproduces 
    the behaviour of the Matlab function. Called by 'check_temporal_spatial_homogeneity' 
    to match the Matlab code results.
    
    Parameters
    ----------
    
    x : 3D array of floats
        input signal 
    direction : str
        direction in which grdient is to be calculated. 'x' or 'y'
    
    Returns
    -------
    gradient : 3D array of floats
        gradient of input signal along "direction""
        
    See also
    --------
    overlap_probe_eprofile.process_checks.check_temporal_spatial_homogeneity
    

    """

    if direction == 'x' :
        
        grad = np.asarray ( [ 1 , 0 , -1 , 2 , 0 , -2 , 1 , 0 , -1 ] ).reshape ( ( 3 , 3 ) )
        
    elif direction == 'y' :

        grad = np.asarray ( [ 1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1 ] ).reshape ( ( 3 , 3 ) )

    return np.rot90 ( ss.convolve ( np.rot90 ( x , 2 ) , np.rot90 ( grad [ :, : , None ] , 2 ) , mode = 'same' ) , 2 )

def quantile ( x , q ) :
    
    n = len(x)
    
    y = np.sort(x)
    
    return ( np.interp ( q , np.linspace ( 1 / ( 2 * n ) , ( 2 * n - 1 ) / ( 2 * n ), n ), y ) )

def prctile ( x , p ) :
      
    return ( quantile ( x ,  p  / 100 ) )  

def create_results_df ( rng , p , poly , resid , resid_whole_zone , ov , ovp_fc , val_max , relgrad_max , relgrad_mean , val_min_slope , index_min_slope , top_mask , bottom_mask , internal_temperature , conditionals , config ) :
   
    """writes the results of quality checks to a Pandas dataframe.
    
    Parameters
    ----------
    
    rng :
    p :
    poly :
    resid :
    resid_whole_zone :
    ov  :
    ovp_fc :
    val_max :
    relgrad_max :
    relgrad_mean :
    val_min_slope :
    index_min_slope : 
    top_mask :
    bottom_mask :
    internal_temperature :
    conditionals :
    config :
    
    Returns
    -------
    results_for_this_interval : Pandas dataframe
        results of checks for the current time interval, including corrected
        overlap functions
        
    See also
    --------
    overlap_probe_eprofile.process_checks.check_temporal_spatial_homogeneity
       

    """

    
    columns = ['rng_start' ,
               'rng_end' ,
               'fit_length' ,
               'slope' ,
               'slope_limits',
               'intercept' ,
               'intercept_limits' ,
               'residual',
               'residual_threshold',
               'residual_whole_zone' ,
               'resid_whole_zon_thresh',
               'max_ov' ,
               'max_ov_thresh',
               'overlap_relative_error' ,
               'ov_rel_err_thresh' ,
               'rel_grad_mag_max',
               'rel_grad_max_thresh',
               'rel_grad_mag_mean',
               'rel_grad_mean_thresh',
               'val_min_slope',
               'val_min_slope_thresh',
               'index_min_slope',
               'index_min_slope_max',
               'pass_all',
               'internal_temperature' ]
            
    results_for_this_interval = pd.DataFrame ( data = np.zeros ( ( np.shape ( top_mask ) [ 1 ] , len(columns) ) ) , columns = columns )
    
    corrected_overlap = pd.DataFrame ( data = ovp_fc.T )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ], np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    index_begin = np.argmin ( bottom_mask , axis = 0 )
    
    index_stop = np.argmax ( top_mask , axis = 0 )
    
    fit_begin = deep_rng [ index_begin -1 , np.arange ( 0 , np.shape ( top_mask ) [ 1 ] ) ]
    
    fit_stop = deep_rng [ index_stop -1 , np.arange ( 0 , np.shape ( top_mask ) [ 1 ] ) ]
    
    fit_length = fit_stop - fit_begin
    
    results_for_this_interval ['rng_start' ] = fit_begin
    
    results_for_this_interval [ 'rng_end'  ] = fit_stop
    
    results_for_this_interval [ 'fit_length' ] = fit_length
    
    results_for_this_interval [ 'slope' ] = p [ 1 ]
    
    results_for_this_interval ['slope_limits'] = str ( config [ 'min_expected_slope' ].values [ 0 ] )  + ' <= slope <= ' + str  ( config [ 'max_expected_slope' ].values [ 0 ] )  
    
    results_for_this_interval [ 'intercept' ] = p [ 0 ]
    
    results_for_this_interval [ 'intercept_limits'] = str ( config [ 'min_expected_zero_fit_value' ].values [ 0 ] )  + ' <= intercept >= ' + str  ( config [ 'max_expected_zero_fit_value' ].values [ 0 ] ) 
    
    results_for_this_interval [ 'residual' ] = resid
    
    results_for_this_interval [ 'residual_thresh'] =  config [ 'thresh_resid_rel' ].values [ 0 ] * np.ma.mean ( poly , axis = 0) 
    
    results_for_this_interval [ 'residual_whole_zone' ] = resid_whole_zone
    
    results_for_this_interval [ 'residual_whole_zone_thresh'] =config [ 'thresh_resid_whole_zone' ].values [ 0 ]
    
    results_for_this_interval [ 'max_ov' ] = np.nanmax ( ovp_fc , axis = 0 )
    
    results_for_this_interval [ 'max_ov_thresh' ] = config [ 'max_overlap_value' ].values [ 0 ]  * np.nanmax ( ov )
    
    results_for_this_interval [ 'overlap_relative_error' ] = val_max 
    
    results_for_this_interval [ 'ov_rel_err_thresh' ] = config [ 'thresh_overlap_valid_rel_error' ].values [ 0 ]
    
    results_for_this_interval [ 'rel_grad_mag_max' ] = relgrad_max 
    
    results_for_this_interval [ 'rel_grad_max_thresh' ] = config [ 'max_relgrad' ].values [ 0 ]
    
    results_for_this_interval [ 'rel_grad_mag_mean' ] = relgrad_mean
    
    results_for_this_interval [ 'rel_grad_mean_thresh' ] =  config [ 'max_relgrad_mean' ].values [ 0 ]
    
    results_for_this_interval [ 'val_min_slope' ] = val_min_slope
    
    results_for_this_interval [ 'val_min_slope_thresh' ] = config [ 'min_slope' ].values [ 0 ]
    
    results_for_this_interval [ 'index_min_slope' ] =  index_min_slope
    
    results_for_this_interval [ 'index_min_slope_max'] = np.argmax ( top_mask , axis = 0 ) - 1
    
    results_for_this_interval [ 'pass_all'] = conditionals
    
    results_for_this_interval [ 'internal_temperature'] = np.nanmean ( internal_temperature )
       
    results_for_this_interval = pd.concat ( [ results_for_this_interval , corrected_overlap ] , axis = 1 , ignore_index=False )
    
    return results_for_this_interval  


def write_temp_model_to_netcdf ( path_for_results , obj ) :
    
    site = obj.site_location.split ( ',') [ 0 ]
    
    opt_mod =  obj.opt_mod_number
    
    rng_res = str ( obj.rng_res [ 0 ] )
    
    nc_name = 'Overlap_correction_model_' + site + '_' + opt_mod + '.nc'
       
    if not os.path.exists( path_for_results ) :
        
        os.mkdir ( path_for_results )
        
    path_for_results = os.path.join ( path_for_results , nc_name )
    
    print ('Writing temperature model to ' , path_for_results )
    
    if os.path.isfile ( path_for_results ) :  
        
        os.remove ( path_for_results )
                  
    ncfile = nc.Dataset ( path_for_results , mode='w' , format = 'NETCDF4_CLASSIC' ) 
   
    ncfile.site_location = obj.site_location

    ncfile.insturment_id = obj.instrument_id
    
    ncfile.wigos_station_id = obj.wigos_station_id 

    ncfile.serlom = obj.opt_mod_number
    
    ncfile.available_time_range_when_model_created = 'From ' + obj.all_available_dates [ 0 ] + ' to ' + obj.all_available_dates [ -1 ]
    
    ncfile.Days_selected_for_model_creation = 'From ' + datetime.datetime.strftime ( obj.available_dts [ 0 ] , '%Y-%m-%d' ) + ' to ' + datetime.datetime.strftime ( obj.available_dts [ -1 ] , '%Y-%m-%d'  )
    
    ncfile.number_of_days_selected = len ( obj.relative_difference )
    
    ncfile.range_resolution = rng_res + 'm'
    
    ncfile.date_of_model_creation = datetime.datetime.strftime ( datetime.datetime.now ( ) , '%Y-%m-%d' )
    
    ncfile.method = 'Based on Hervo et al. 2016'
    
    ncfile.description = 'Output for overlap artefact correction. Use: Dif (z) = a (z)* T + b (z) and Overlap_corrected (z) = 1. / (Dif (z) / 100 / overlap_ref (z) + 1. / overlap_ref (z) )'

    ncfile.createDimension('range', len ( obj.rng ) )     
    
    rng = ncfile.createVariable('range', 'f4', ('range', ) )
    
    rng.standard_name = 'range'
    
    rng.long_name = 'Altitude above ground (m)' 
    
    rng.units = 'm'
    
    rng [ : ] = obj.rng
    
    a = ncfile.createVariable('a', 'f4', ('range', ) )
    
    a.standard_name = 'a'
    
    a.long_name = 'Results of fit (difference = a *Temperature +b )'
    
    a [ : ] = obj.alpha_2
    
    b = ncfile.createVariable('b', 'f4', ('range', ) )
    
    b.standard_name = 'b'
    
    b.long_name = 'Results of fit (difference = a *Temperature +b )'
    
    b [ : ] = obj.beta_2

    overlap_ref = ncfile.createVariable('overlap_ref', 'f4', ('range', ) )
    
    overlap_ref.standard_name = 'overlap_ref'
    
    overlap_ref.long_name = 'Reference overlap function'
    
    overlap_ref [ : ] = obj.ref_ov

    ncfile.close ( )
    



def create_ceilo_plot ( L1 , vdr = None , mass = None , instrument = None , savepath = None , location = None ) :

    RVal = [255, 212, 209, 207, 205, 202, 199, 196, 193, 189, 186, 183, 179, 176, 172, 169, 166, 163, 159, 156, 153, 149, 146, 143, 139, 136, 132, 128, 124, 121, 117, 113, 109, 105, 102,  98,  93,  89,  85,  81,  77,  73,  70,  66,  61,  57,  53,
            49,  45,  41,  38,  34,  30,  26,  22,  19,  16,  13,  11,   9,   7,   5,   3,   2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   2,   4,   7,  10,  14,  18,  23,  28,  34,  41,  48,  55,  63,  72,  80,  88,  95, 103, 112,
            121, 129, 137, 145, 153, 161, 168, 176, 184, 192, 201, 208, 215, 221, 227, 232, 236, 240, 244, 246, 249, 251, 252, 253, 253, 253, 253, 253, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253,
            253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 252, 252, 252, 251, 251, 251, 252, 252, 252, 252, 252, 251, 251, 251, 250, 249, 248, 246, 244, 242, 240, 237, 234, 230, 226, 223, 219, 215, 212,
            208, 204, 199, 195, 191, 188, 184, 180, 176, 172, 167, 163, 159, 155, 151, 147, 144, 141, 138, 136, 134, 132, 131, 130]

    GVal = [255, 216, 212, 209, 206, 203, 200, 197, 194, 190, 187, 184, 180, 176, 173, 169, 166, 163, 159, 156, 153, 149, 146, 142, 139, 135, 132, 128, 124, 121, 117, 113, 110, 106, 102,  98,  94,  90,  86,  82,  78,  74,  70,  67,  62,  58,  54,
            50,  46,  42,  39,  35,  31,  27,  23,  20,  18,  16,  15,  15,  15,  15,  16,  18,  21,  24,  28,  32,  37,  43,  48,  53,  58,  63,  68,  72,  78,  83,  88,  92,  97, 102, 107, 111, 116, 122, 127, 132, 136, 141, 145, 149, 153,
            157, 161, 165, 168, 171, 175, 178, 181, 184, 187, 190, 193, 196, 200, 203, 206, 209, 212, 215, 218, 221, 225, 228, 231, 234, 236, 239, 241, 243, 244, 246, 247, 249, 250, 251, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253,
            253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 252, 251, 249, 247, 245, 243, 241, 238, 235, 232, 229, 225, 221, 217, 213, 209, 205, 201, 196, 192, 188, 184, 180, 176, 172, 168, 164, 161, 156, 152, 148, 144, 141,
            137, 133, 128, 124, 120, 116, 111, 107, 103,  99,  95,  91,  88,  84,  80,  76,  72,  68,  65,  60,  56,  53,  49,  45,  40,  36,  32,  27,  23,  20,  17,  14,  11,   8,   6,   4,   3,   2,   1,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]

    BVal = [255, 225, 226, 227, 228, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 248, 249, 249, 250, 251, 251, 251, 251, 251, 252, 252, 252, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251,
            251, 251, 252, 252, 252, 252, 251, 251, 251, 250, 250, 249, 248, 247, 245, 243, 241, 239, 236, 234, 231, 229, 226, 223, 220, 217, 214, 211, 208, 205, 202, 199, 196, 193, 191, 188, 185, 182, 179, 175, 172, 168, 164, 161, 157, 153,
            149, 145, 141, 137, 133, 128, 123, 118, 113, 108, 102,  97,  92,  86,  81,  76,  71,  66,  61,  56,  51,  46,  41,  36,  32,  28,  24,  20,  17,  14,  12,  10,   8,   6,   4,   3,   2,   2,   1,   1,   1,   1,   1,   1,   1,   1,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,
            0,   1,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   1,   2,   3,   5,   7,   9,  11,  13,  16,  19,  22,  26,  30,  33,  36,  40,
            44,  47,  52,  55,  59,  62,  66,  70,  74,  78,  83,  87,  91,  95,  99, 102, 105, 108, 111, 113, 115, 117, 118, 120]

    colours = np.transpose ( np.asarray ( ( RVal , GVal , BVal ) ) )

    fctab= colours / 255.0

    my_cmap = colors.ListedColormap ( fctab , name = 'Cloudnet' , N = None )

    my_cmap.set_bad('white')

    params = {'legend.fontsize': 20,
              'figure.figsize': (15, 5),
              'axes.labelsize': 20,
              'axes.titlesize':20,
              'axes.linewidth':2,
              'xtick.labelsize':30,
              'ytick.labelsize':30,
              'xtick.major.size': 3.5,
              'xtick.minor.size': 2}

    plt.rcParams.update(params)

    date = datetime.datetime.strftime ( L1.dt [ 0 ] , '%Y%m%d' )

    beta = L1.rcs_0

    cbh = L1.cbh

    m_time = L1.Time

    range1 = L1.rng / 1000

    elastic = np.log10(beta)

    Time = np.asarray(m_time)

    if instrument.upper() == 'CL61':

        VDR = np.log10(vdr)

    fig = plt.figure(num=None, facecolor='w', edgecolor='k')

    fig.set_size_inches(15,11)

    LABEL_SIZE = 15

    gs = gridspec.GridSpec(nrows=3, ncols=2 , width_ratios=[1,0.01])

    ax1 = fig.add_subplot(gs[0,0])

    plt.suptitle ( location + ' ' + instrument.upper() + ' ' + date , x = 0.125, y = 0.92,fontsize = LABEL_SIZE, color = 'r', ha = 'left')

    p1 = plt.imshow(np.flipud(np.transpose(elastic)), vmin = 4, vmax = 6, extent=[Time[0],Time[-1],range1[0],range1[-1]],cmap = my_cmap,interpolation='none', aspect = 'auto')

    ax1.xaxis_date()

    cbh_symbols = ['x' , '^' , '*' , 's' , 'o']

    for n in range ( 0, np.shape ( cbh ) [ 1 ] ) :

        ax1.plot ( Time , cbh[:,n]/1000 , cbh_symbols [ n ] , color = 'k', ms = 2 , zorder = 20 )

    date_format = matplotlib.dates.DateFormatter('%H:%M')

    ax1.xaxis.set_major_formatter(date_format)

    plt.title(r'Log$_{10}$ Attenuated Backscatter', fontsize = LABEL_SIZE-4, pad = 10)

    plt.ylabel('Range [km]', fontsize = LABEL_SIZE)

    cax = fig.add_subplot(gs[0,1])

    cbar = matplotlib.colorbar.Colorbar(cax,mappable = p1  , cmap =  my_cmap,orientation='vertical')

    cbar.set_label(r'[m$^{-1}$sr$^{-1}$]', rotation=90, labelpad=20, y=0.45, fontsize = LABEL_SIZE)

    cbar.ax.tick_params(labelsize=15)

    plt.clim(4,6)

    ax1.tick_params(labelsize=LABEL_SIZE-5)

    ax1.set_ylim([0,15])

    if instrument.upper() != 'CL61':

        ax1.set_xlabel('Time [UTC]', fontsize = LABEL_SIZE)

    if instrument.upper() == 'CL31':

        ax1.set_ylim([0,8])

    if instrument.upper() == 'CL61':

        ax2 = plt.subplot(gs[1,0])

        p2 = plt.imshow(np.flipud(np.transpose(VDR)), vmin = -2.5, vmax = 0 , extent=[Time[0],Time[-1],range1[0],range1[-1]],cmap = my_cmap,interpolation='none', aspect = 'auto')

        ax2.xaxis_date()

        date_format = matplotlib.dates.DateFormatter('%H:%M')

        ax2.xaxis.set_major_formatter(date_format)

        plt.title(r'Log$_{10}$VDR', fontsize = LABEL_SIZE-4, pad = 10)

        plt.ylabel('Range [km]', fontsize = LABEL_SIZE)

        cax = fig.add_subplot(gs[1,1])

        cbar = matplotlib.colorbar.Colorbar(cax,mappable = p2  , cmap =  my_cmap,orientation='vertical')

        cbar.set_label(r'[AU]', rotation=90, labelpad=20, y=0.45, fontsize = LABEL_SIZE)

        cbar.ax.tick_params(labelsize=15)

        plt.clim(-2.5,0)

        ax2.tick_params(labelsize=LABEL_SIZE-5)

        ax2.set_ylim([0,15])

        ax3 = plt.subplot(gs[2,0])

        plt.imshow(np.flipud(np.transpose(mass)), extent=[Time[0],Time[-1],range1[0],range1[-1]],cmap = matplotlib.cm.get_cmap('Reds'),interpolation='none', aspect = 'auto')

        ax3.xaxis_date()

        date_format = matplotlib.dates.DateFormatter('%H:%M')

        ax3.xaxis.set_major_formatter(date_format)

        plt.title(r'Mass concentration', fontsize = LABEL_SIZE-4, pad = 10)

        plt.ylabel('Range [km]', fontsize = LABEL_SIZE)

        plt.xlabel('Time [UTC]', fontsize = LABEL_SIZE)

        cax = fig.add_subplot(gs[2,1])

        cbar = matplotlib.colorbar.ColorbarBase(cax,matplotlib.cm.get_cmap('Reds'),orientation='vertical')

        cax.yaxis.set_ticks_position('left')

        tks = [y/200 for y in range ( 0 , 200 ,25) ]

        cax.set_yticks(tks)

        fac = 0.68/0.38

        tk_lab = [ int(y*fac) for y in range ( 0 , 240 ,40) ]

        cax.set_yticklabels(tk_lab)

        cax.set_ylabel(r'Ash [$\mu$gm$^{-3}$]', rotation=90, labelpad=30, y=0.45, fontsize = LABEL_SIZE-5)

        clone = cax.twinx()

        clone.tick_params(axis='both', which='major', labelsize=10)

        clone.set_ylim([0,200])

        clone.set_ylabel(r'Dust [$\mu$gm$^{-3}$]', rotation=90, labelpad=-70, y=0.45, fontsize = LABEL_SIZE-5)

        cbar.ax.tick_params(labelsize=LABEL_SIZE-5)

        plt.clim(0,200)

        ax3.tick_params(labelsize=LABEL_SIZE-5)

        ax3.set_ylim([0,15])

    fig.subplots_adjust(wspace=0.11)

    fig.savefig ( savepath + '/' + instrument.upper ( ) + '_' + date + '.png' , bbox_inches = 'tight' , format = 'png' , dpi = 300 )
