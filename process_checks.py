# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
Started December 2022

functions to perform processing and checks on ceilometer data to be used

in calculating an overlap function.  This is a translation /

refactoring of Matlab code written by Maxime Hervo, Rolf Ruefenacht 

and Melania Van Hove.

@author martin osborne: martin.osborne@metoffice.gov.uk

'''
import numpy as np
import scipy.signal as ss
from copy import deepcopy
import pandas as pd

def _make_mask ( fl , fb , rng ) :
    
    '''
    
    Takes in 'fit_length' and 'fit_begin' ranges for the windows within 
    
    which a simple linear regressesion is to be performed on the mean log signal
    
    being checked. These are worked up into two masks that can be applied to an 
    
    array consisting of the repeated mean log signal.
    
    
    '''
    
    wbmax = fb [ -1 ]
       
    wbegin = np.repeat ( fb  , len (fl ) )
       
    wlen = np.tile ( fl  , len ( fb )  )
       
    wstop = wbegin + wlen

    wbegin = np.delete ( wbegin , np.where ( wstop  > wbmax  ) )
        
    wstop = np.delete ( wstop , np.where ( wstop  > wbmax  ) )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ] , len ( wstop ) , axis = 1 )
    
    bottom_mask = ( deep_rng < wbegin ) 
    
    top_mask =  ( deep_rng >  wstop )
    
    n =  len ( rng ) - np.sum ( bottom_mask + top_mask , axis = 0 )
               
    return top_mask , bottom_mask , n 
   
def _simple_linear_fit ( n , x , y ) :
    
    '''
    
    Calculates the intercepts (alphas) and slopes (betas) of a simple linear fit
    
    to the unmasked values in each column of masked array y
        
    '''

    x = np.ma.masked_invalid ( x ) 
    
    y = np.ma.masked_invalid ( y )
       
    beta = (   n * np.ma.sum ( ( x * y ) , axis = 0 ) - np.ma.sum ( x , axis = 0 ) * np.ma.sum ( y , axis = 0 ) ) / ( n * np.ma.sum (  ( x * x ) , axis = 0 ) - ( np.ma.sum ( x , axis = 0  ) * np.ma.sum ( x, axis = 0  ) ) )
    
    alpha = ( ( 1 / n ) * np.ma.sum ( y , axis = 0 ) ) - ( ( 1 / n ) * beta * np.ma.sum ( x , axis = 0 ) )
    
    return  alpha , beta



def conv3d ( x , direction = None ) :

    '''

    Matlab's convolve and Python's scipy.signal.convolve behave slightly differently

    This function reproduces the behaviour of the Matlab function. Used in 
    
    'find_gradient_of_corrected_signal' to match the Matlab code results

    '''

    if direction == 'x' :
        
        grad = np.asarray ( [ 1 , 0 , -1 , 2 , 0 , -2 , 1 , 0 , -1 ] ).reshape ( ( 3 , 3 ) )
        
    elif direction == 'y' :

        grad = np.asarray ( [ 1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1 ] ).reshape ( ( 3 , 3 ) )

    return np.rot90 ( ss.convolve ( np.rot90 ( x , 2 ) , np.rot90 ( grad [ :, : , None ] , 2 ) , mode = 'same' ) , 2 )

def _check_regression ( p , masked_signal , masked_signal_whole_zone , masked_rng , masked_rng_whole_zone , n ) :
    
    poly = p [ 0 ] + p [ 1 ] * masked_rng
       
    poly_whole_zone = p [ 0 ] + p [ 1 ] * masked_rng_whole_zone
                    
    resid = np.sqrt ( ( 1 /  n  ) * np.ma.sum ( ( masked_signal - poly ) ** 2 , axis = 0) )
      
    resid_whole_zone = np.ma.max ( abs ( masked_signal - poly_whole_zone ) / abs ( poly_whole_zone ) , axis = 0 )
        
    return poly , resid , resid_whole_zone
    
def _make_ovp_fc ( signal_all , p , ov , rng , top_mask , config ) :
        
    signal_all = np.repeat ( signal_all [ : , np.newaxis ], np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ], np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    deep_ov = np.repeat ( ov [ : , np.newaxis ], np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    poly_all = p [ 0 ] + p [ 1 ] * deep_rng

    diff = signal_all - poly_all
    
    overlap_corr_factor = 10 ** diff
    
    overlap_corr_factor [ ( top_mask == 1 ) ] = 1
    
    ovp_fc = deep_ov * overlap_corr_factor
    
    min_overlap_valid = config [ 'min_overlap_valid' ].values [ 0 ]
    
    rel_err =  abs ( deep_ov [ rng >= min_overlap_valid , : ] - ovp_fc [ rng >= min_overlap_valid , : ] )  / abs ( deep_ov [ rng >= min_overlap_valid , : ] ) 
    
    valmax = np.nanmax ( rel_err , axis = 0 ) 
    
    return overlap_corr_factor , ovp_fc , valmax 

def find_gradient_of_corrected_signal ( rcs_0 , rng , overlap_corr_factor , top_mask , max_available_fit_range , condition1 , config ):
    
    np.seterr(divide='ignore')
       
    index_range_for_grad = ( rng >= config [ 'min_range_std_over_mean' ].values [ 0 ] ) * ( rng < max_available_fit_range )
     
    deep_rcs_0 = np.repeat ( rcs_0.T [ index_range_for_grad , : , np.newaxis ], np.shape ( top_mask ) [ 1 ] , axis = 2 )
    
    deep_rcs_0 = deep_rcs_0 [ : , : , condition1 ]
    
    deep_overlap_corr_factor = np.repeat ( overlap_corr_factor [ index_range_for_grad , np.newaxis , : ], np.shape ( rcs_0.T ) [ 1 ] , axis = 1 )
    
    deep_overlap_corr_factor = deep_overlap_corr_factor [ : , : , condition1 ]
    
    signal_for_grad_check =   np.log10 ( abs ( deep_rcs_0 ) / deep_overlap_corr_factor  ) 
    
    gradY = conv3d (  signal_for_grad_check , direction =  'y' )

    gradX = conv3d (  signal_for_grad_check , direction = 'x' )

    relgradmagn = np.sqrt ( gradX  ** 2 + gradY  ** 2 ) / abs ( signal_for_grad_check )

    new_elements_to_mask = np.argmax  ( top_mask , axis = 0 ) 
    
    top_mask = np.asarray(top_mask , dtype = 'bool')
    
    top_mask_temp = deepcopy ( top_mask )
    
    top_mask_temp [ new_elements_to_mask -1, np.arange(0,np.shape(top_mask)[1]) ] = True
    
    deep_top_mask = np.repeat ( top_mask_temp [ index_range_for_grad , np.newaxis , : ] , np.shape ( rcs_0.T ) [ 1 ] , axis = 1 )
    
    deep_top_mask = deep_top_mask [ : , : , condition1 ]
  
    relgradmagn = np.ma.masked_array ( relgradmagn , mask=deep_top_mask )
            
    relgradmagn = relgradmagn [ 1 : -1 , 1 : -1 , : ]
      
    relgrad_max_small , relgrad_mean_small , condition2_small = _check_conditions_2 ( relgradmagn , config )  
    
    relgrad_max = np.zeros ( len ( condition1 )  )
    
    relgrad_mean = np.zeros ( len ( condition1 ) )
    
    condition2 = np.zeros ( len ( condition1 ) , dtype = bool )
    
    relgrad_max [ condition1 ] = relgrad_max_small
    
    relgrad_mean [ condition1 ] = relgrad_mean_small
        
    condition2 [ condition1 ] = condition2_small
    
    return relgradmagn , relgrad_max , relgrad_mean , condition2

def find_savgol_slope ( ovp_fc , rng , top_mask,  config ) : 
   
    slope = ss.savgol_filter ( ovp_fc [ : 167 , : ] , window_length = int ( config [ 'sgolay_width' ].values [ 0 ] ) , polyorder = int ( config [ 'sgolay_ord' ].values [ 0 ] ) , axis = 0 , deriv = 1 , delta = rng [ 1 ] - rng [ 0 ] )
   
    val_min_slope = np.nanmin  ( slope , axis = 0)
    
    index_min_slope = np.argmin ( slope , axis = 0 ) 
    
    index_range_stop_correction = np.argmax ( top_mask , axis = 0 ) - 1
    
    condition3 = _check_condtions_3 ( val_min_slope , index_min_slope , index_range_stop_correction , config ) 
    
    return val_min_slope , index_min_slope , condition3


def _check_conditions_1 ( p , poly , resid , resid_whole_zone , ovp_fc , ov , valmax , config ) :
       
    con1 = ( p [ 1 ] >= config [ 'min_expected_slope' ].values [ 0 ] )
    
    con2 = ( p [ 1 ] <= config [ 'max_expected_slope' ].values [ 0 ] ) 
                    
    con3 = ( p [ 0 ] >= config [ 'min_expected_zero_fit_value' ].values [ 0 ] )
      
    con4 = ( p [ 0 ] <= config [ 'max_expected_zero_fit_value' ].values [ 0 ] )
    
    con5 =  ( resid < config [ 'thresh_resid_rel' ].values [ 0 ] * np.ma.mean ( poly , axis = 0) )
    
    con6 = ( resid_whole_zone < config [ 'thresh_resid_whole_zone' ].values [ 0 ] )
    
    con7 = ( ( np.nanmax ( ovp_fc , axis = 0 ) ) <= config [ 'max_overlap_value' ].values [ 0 ]  * np.nanmax ( ov , axis = 0) )
     
    con8 = ( valmax < config [ 'thresh_overlap_valid_rel_error' ].values [ 0 ] )
    
    condition1 = con1 * con2 * con3 * con4 * con5 * con6 * con7 * con8
    
    #print ('cons1 = ' , con1 , con2 , con3 , con4 , con5 , con6 , con7 , con8 )
    
    return condition1

def _check_conditions_2 ( relgradmagn  , config ) :
    
    relgradmagn = np.ma.masked_invalid ( relgradmagn )
    
    relgrad_max = np.ma.max ( np.ma.max ( relgradmagn , axis = 1 ) , axis = 0 )

    relgrad_mean = np.ma.mean ( np.ma.mean ( relgradmagn , axis = 1 ) , axis = 0 )
    
    con9 = ( relgrad_max  <=  config [ 'max_relgrad' ].to_numpy ( ) )
       
    con10 = ( relgrad_mean <= config [ 'max_relgrad_mean' ].to_numpy ( ) ) 
                        
    condition2 = con9 * con10
    
    #print ('cons2 = ' , con9 , con10 )
    
    return relgrad_max , relgrad_mean , condition2

def _check_condtions_3 ( val_min_slope , index_min_slope , index_range_stop_correction , config ) :
       
    val_min_slope = np.ma.masked_invalid ( val_min_slope ) 
    
    index_min_slope = np.ma.masked_invalid ( index_min_slope )
    
    con11 = ( val_min_slope >= config [ 'min_slope' ].to_numpy ( ) ) 
    
    con12 =  ( index_min_slope > index_range_stop_correction )

    condition3 = con11 + con12
    
    #print ('cons3 = ' ,  con11 , con12 )
    
    return condition3

def do_regresion ( rcs_0 , rng , max_available_fit_range , config , ov ) :
    
    '''
    
    Returns the coeffficients p of a simple linear regression within sliding 
    
    windows of different legths as defined by 'min_fit_length', 'max_fit_legth'
    
    'd_fit_length' , 'min_fit_range' and 'd_fit_range'
    
    '''
    
    signal_all = np.nanmean ( np.log10 ( abs ( rcs_0 ) ) , axis = 0 )
       
    fl = np.asarray ( np.arange( config [ 'min_fit_length' ].values [ 0 ] , config [ 'max_fit_length' ].values [ 0 ] , config [ 'd_fit_length' ].values [ 0 ] ) )
       
    fb = np.asarray (  np.arange ( config [ 'min_fit_range' ].values [ 0 ] , float ( max_available_fit_range ) , config [ 'd_fit_range' ].values [ 0 ] ) )
    
    top_mask , bottom_mask , n  = _make_mask ( fl , fb , rng )
    
    mask = np.array ( top_mask + bottom_mask, dtype = bool ) 
    
    deep_signal = np.repeat ( signal_all [ : , np.newaxis ], np.shape ( mask ) [ 1 ] , axis = 1 )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ], np.shape ( mask ) [ 1 ] , axis = 1 )
    
    masked_signal = np.ma.masked_array ( deep_signal , mask=mask )
    
    mask_for_whole_zone = ( top_mask == 1 ) & (deep_rng <= config [ 'min_fit_range' ].values [ 0 ] )
    
    masked_signal_whole_zone = np.ma.masked_array ( deep_signal , mask=mask_for_whole_zone )
    
    masked_rng = np.ma.masked_array ( deep_rng , mask=mask )
    
    masked_rng_whole_zone = np.ma.masked_array ( deep_rng , mask=mask_for_whole_zone )
    
    p = _simple_linear_fit ( n , masked_rng , masked_signal )  
    
    poly , resid , resid_whole_zone = _check_regression ( p , masked_signal , masked_signal_whole_zone , masked_rng , masked_rng_whole_zone , n )
    
    overlap_corr_factor , ovp_fc , valmax = _make_ovp_fc ( signal_all , p , ov , rng , top_mask , config )  
    
    condition1 = _check_conditions_1 ( p , poly , resid , resid_whole_zone , ovp_fc , ov , valmax , config)

    return p , poly , resid , resid_whole_zone , overlap_corr_factor , ov , ovp_fc , valmax , top_mask, bottom_mask , condition1 
    

def create_results_df ( rng , p , poly , resid , resid_whole_zone , ov , ovp_fc , val_max , relgrad_max , relgrad_mean , val_min_slope , index_min_slope , top_mask , bottom_mask , internal_temperature , conditionals , config ) :

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
    
    results_for_this_interval [ 'residual_thresh'] =  config [ 'thresh_resid_rel' ].values [ 0 ] * np.ma.mean ( poly )  
    
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
    
def check_polyfit ( rcs_0 , rng , internal_temperature , max_available_fit_range , config , ov ) :
    
    condition1 = 0
    condition2 = 0
    condition3 = 0
    
    p , poly , resid , resid_whole_zone , overlap_corr_factor , ov ,  ovp_fc , valmax , top_mask , bottom_mask,  condition1 = do_regresion ( rcs_0 , rng , max_available_fit_range , config , ov )
       
    if np.sum(condition1) > 0  :
        
        relgradmagn , relgrad_max , relgrad_mean , condition2 = find_gradient_of_corrected_signal ( rcs_0 , rng , overlap_corr_factor , top_mask , max_available_fit_range , condition1 , config )    

        val_min_slope , index_min_slope , condition3 = find_savgol_slope ( ovp_fc , rng , top_mask , config ) 
        
        conditionals = condition1 * condition2 * condition3
    
    
        results_df = create_results_df ( rng , p , poly ,  resid , resid_whole_zone , ov , ovp_fc , valmax , relgrad_max , relgrad_mean , val_min_slope , index_min_slope , top_mask , bottom_mask , internal_temperature , conditionals, config )
    else:
        
        fill = np.zeros ( len ( resid ) )
        
        conditionals = condition1 * condition2 * condition3
        
        results_df  = create_results_df ( rng , p , poly ,  resid , resid_whole_zone , ov , ovp_fc , fill , fill , fill , fill , fill , top_mask , bottom_mask , internal_temperature , conditionals , config )

    
    
    
    return results_df





