#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions called in by other modules

@author martin osborne: martin.osborne@metoffice.gov.uk
"""

import numpy as np
import pandas as pd
import scipy.signal as ss



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