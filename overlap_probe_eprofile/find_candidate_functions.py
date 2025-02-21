# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""This module contans functions to perform the checks and processing decribed in 
section 2 of the appendix of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
titled "Quality check of fits and determination of a set of overlap correction candidates"

As descibed in the linked paper, the checks are repetedly applied to small, shifting sections of the data 
defined by shifting altitude windows. This is computationally expensive
and when done using loops in Python takes a long time - much longer than in Matlab. 
To speed things up the functions have been vectorised - the 1D mean input signal is repeated into 
a second dimension to be the total number of shifting windows wide. The resulting 2D array is 
then masked using Numpy's masking function. The mask is constructed so that the n\ :sup:`th` column 
of the 2D array contains unmasked values only within the n\ :sup:`th` shifting altitude window. 
The checks are then performed on this masked array in one operation. 

@author martin osborne: martin.osborne@metoffice.gov.uk
"""

import numpy as np
import scipy.signal as ss
from copy import deepcopy
from overlap_probe_eprofile.overlap_utils import create_results_df, conv3d , simple_linear_fit

np.seterr(divide = 'ignore') 
np.seterr(invalid = 'ignore')


def check_fits ( rcs_0 , rng , max_available_fit_range , config ) :
    
    """Calls make_mask and simple_linear_fit to make straight line fits to the 
    mean log signal as in Eqn. (8) of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_ 
    within the altitude ranges defined by 'fit_length' and 'fit_begin' and checks the plausibility of the fits. 
    
    The checks applied in this function are described in Appendix 4 and 5 
    of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
    
    
    The following checks must be passed:
    
    The slopes must be between 'min_expected_slope' and 'max_expected_slope', 
    and the intercepts must be between 'min_expected_zero_fit_value' and 'max_expected_zero_fit_value'.
    
    The residuals between the fits and the mean log signal within the relevant altitude 
    window must be less than 'thresh_resid_rel', and the residuals betwen the fits and 
    the mean log signal within the altitude ranges extended down to 'min_fit_range' 
    must be less than 'thresh_resid_whole_zone'
    
    Parameters
    ----------
    
    rcs_0 : 2D array 
        range corrected signal within current time window 
    rng : array
        range array for CHM15k
    max_available_fit_range : float
        maximum altitude at which all pre_checks were passed
    config : pandas data frame
        thresholds and setting        
    
    Returns
    -------
    p : array object
        object returned by simple_linear_fit - slopes and intercepts of regressions
        returned by simple_linear_fit
    poly : 2D array 
        regression lines
    resid : array of floats
        sum of the squares of the residuals between poly and data
    resid_whole_zone : array of floats
        max of the absolute value of the residuals between poly and data
    top_mask : 2D array of bools
        mask defining the upper bounds of the the altitude windows
    bottom_mask : 2D array of bools
        mask defining the lower bounds of the altitude windows
    condition1 : boolean array
        results of checks on the plausability of the linear regressions, the 
        goodness of the fit and the maximum values of the corrected overlap
        functions. True for pass and False for fail
            
    See also
    --------
    overlap_probe_eprofile.process_checks.make_mask
    overlap_probe_eprofile.overlap_utils.simple_linear_fit
    overlap_probe_eprofile.process_checks.get_regression_residuals    
    
    """
    
    signal_all = np.nanmean ( np.log10 ( abs ( rcs_0 ) ) , axis = 0 )
       
    fl = np.asarray ( np.arange ( config [ 'min_fit_length' ].values [ 0 ] , config [ 'max_fit_length' ].values [ 0 ] , config [ 'd_fit_length' ].values [ 0 ] ) )
    
    fb = np.asarray (  np.arange ( config [ 'min_fit_range' ].values [ 0 ] , float ( max_available_fit_range ) , config [ 'd_fit_range' ].values [ 0 ] ) )
    
    top_mask , bottom_mask , n  = make_mask ( fl , fb , rng )
    
    mask = np.array ( top_mask + bottom_mask, dtype = bool ) 
    
    deep_signal = np.repeat ( signal_all [ : , np.newaxis ], np.shape ( mask ) [ 1 ] , axis = 1 )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ], np.shape ( mask ) [ 1 ] , axis = 1 )
    
    masked_signal = np.ma.masked_array ( deep_signal , mask=mask )
    
    mask_for_whole_zone = ( top_mask == 1 ) & (deep_rng <= config [ 'min_fit_range' ].values [ 0 ] )
    
    masked_signal_whole_zone = np.ma.masked_array ( deep_signal , mask=mask_for_whole_zone )
    
    masked_rng = np.ma.masked_array ( deep_rng , mask=mask )
    
    masked_rng_whole_zone = np.ma.masked_array ( deep_rng , mask=mask_for_whole_zone )
    
    p = simple_linear_fit ( n , masked_rng , masked_signal )  
    
    poly , resid , resid_whole_zone = get_regression_residuals ( p , masked_signal , masked_signal_whole_zone , masked_rng , masked_rng_whole_zone , n )
    
    condition1 = _check_conditions_1 ( p , poly , resid , resid_whole_zone , config)
    
    return p , poly , resid , resid_whole_zone , top_mask, bottom_mask ,  condition1 

def check_relative_error ( rcs_0 , p , ov , rng , top_mask , config , condition1 ) :
    
    """Calls make_ovp_fc to create corrected overlap functions and then applies the checks
    described in Appendix 6 and 7 of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_  

    The ratio of the maximum of each corrected overlap function to the max of the 
    referenec overlap function must be less than 'max_overlap_value'
    
    The max of relative error between the corrected overlap functions and the reference
    overlap function must be less than 'thresh_overlap_valid_rel_error'
    
    Parameters
    ----------
    
    rcs_0 : 2D array 
        range corrected signal within current time window       
    p : array object
        object returned by simple_linear_fit - slopes and intercepts of regressions
        returned by simple_linear_fit
    ov : array
        reference overlap signal
    rng : array
        range array for CHM15k
    top_mask : 2D array of bools
        mask defining the upper bounds of the the altitude windows
    config : pandas data frame
        thresholds and setting  
    condition1 : boolean array
        results of checks performed by check_fits
  
    Returns
    -------
    
    ovp_fc : 2D array
        candidate corrected overlap functions   
    overlap_corr_factor : 2D array 
        array of overlap correction factors
    valmax : array 
        max of the relative error between the reference overlap function and the 
        candidate corrected overlap functions  
    condition2 : boolean array
        results of checks on the maximum values of the corrected overlap
        functions. True for pass and False for fail
            
    See also
    --------
    overlap_probe_eprofile.process_checks.make_ovp_fc
    
    
    """
    
    if np.sum ( condition1 ) > 0 :
    
        signal_all = np.nanmean ( np.log10 ( abs ( rcs_0 ) ) , axis = 0 )
        
        overlap_corr_factor , ovp_fc , valmax = make_ovp_fc ( signal_all , p , ov , rng , top_mask , config )  
        
        condition2 = _check_conditions_2 ( ovp_fc , ov , valmax , config ) * condition1
    
        return  ovp_fc , overlap_corr_factor , valmax ,  condition2
    
    else:
        
        fill = np.zeros ( len ( condition1 ) )
        
        return fill , fill , fill  , fill

def check_temporal_spatial_homogeneity ( rcs_0 , rng , overlap_corr_factor , top_mask , max_available_fit_range , config, condition2 ) :
    
    """Calls conv3d to find the spatio-temporal gradients of the corrected
    signal found using the candidate corrected overlap functions, and then checks 
    the maximum and mean are less than 'max_relgrad' and  'max_relgrad_mean' as
    defined in config.
    
    The checks applied in this function are described in Appendix 8 
    of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
    
    
    Parameters
    ----------
    
    rcs_0 : 2D array 
        range corrected signal within current time window 
    rng : array
        range array for CHM15k
    overlap_corr_factor : 2D array 
        array of overlap correction factors
    top_mask : 2D array of bools
        mask defining the upper bounds of the the altitude windows
    max_available_fit_range : float
        maximum altitude at which all pre_checks were passed
    condition1 : array of bools
        profiles that have passed checks so far
    config : pandas data frame
        thresholds and setting 
    
    Returns
    -------
    relgradmagn : 3D array of floats
        gradient of input signal along "direction""
    relgrad_max : 3D array of floats
        gradient of input signal along "direction""
    relgrad_mean : 3D array of floats
        gradient of input signal along "direction""
    condition3 : 3D array of floats
        gradient of input signal along "direction""        
    See also
    --------
    overlap_probe_eprofile.overlap_utils.conv3d
    
    """
    if np.sum ( condition2 ) > 0 :
        
        np.seterr(divide='ignore')
       
        index_range_for_grad = ( rng >= config [ 'min_range_std_over_mean' ].values [ 0 ] ) * ( rng < max_available_fit_range )
         
        deep_rcs_0 = np.repeat ( rcs_0.T [ index_range_for_grad , : , np.newaxis ] , np.shape ( top_mask ) [ 1 ] , axis = 2 )
        
        deep_rcs_0 = deep_rcs_0 [ : , : , condition2 ]
        
        deep_overlap_corr_factor = np.repeat ( overlap_corr_factor [ index_range_for_grad , np.newaxis , : ], np.shape ( rcs_0.T ) [ 1 ] , axis = 1 )
        
        deep_overlap_corr_factor = deep_overlap_corr_factor [ : , : , condition2 ]
        
        signal_for_grad_check =   np.log10 ( abs ( deep_rcs_0 ) / deep_overlap_corr_factor  ) 
        
        gradY = conv3d (  signal_for_grad_check , direction =  'y' )
    
        gradX = conv3d (  signal_for_grad_check , direction = 'x' )
    
        relgradmagn = np.sqrt ( gradX  ** 2 + gradY  ** 2 ) / abs ( signal_for_grad_check )
    
        new_elements_to_mask = np.argmax  ( top_mask , axis = 0 ) 
        
        top_mask = np.asarray(top_mask , dtype = 'bool')
        
        top_mask_temp = deepcopy ( top_mask )
        
        top_mask_temp [ new_elements_to_mask -1, np.arange(0,np.shape(top_mask)[1]) ] = True
        
        deep_top_mask = np.repeat ( top_mask_temp [ index_range_for_grad , np.newaxis , : ] , np.shape ( rcs_0.T ) [ 1 ] , axis = 1 )
        
        deep_top_mask = deep_top_mask [ : , : , condition2 ]
    
        relgradmagn = np.ma.masked_array ( relgradmagn , mask=deep_top_mask )
                
        relgradmagn = relgradmagn [ 1 : -1 , 1 : -1 , : ]
          
        relgrad_max_small , relgrad_mean_small , condition3_small = _check_conditions_3 ( relgradmagn , config )  
        
        relgrad_max = np.zeros ( len ( condition2 )  )
        
        relgrad_mean = np.zeros ( len ( condition2 ) )
        
        condition3 = np.zeros ( len ( condition2 ) , dtype = bool )
        
        relgrad_max [ condition2 ] = relgrad_max_small
        
        relgrad_mean [ condition2 ] = relgrad_mean_small
            
        condition3 [ condition2 ] = condition3_small
        
        return relgradmagn , relgrad_max , relgrad_mean , condition3
    
    else:
        
        fill = np.zeros ( len ( condition2 ) )
        
        return fill , fill , fill  , fill

def check_monotonic ( ovp_fc , rng , top_mask,  config , condition3 ) :     
    
    """
    Checks that the candidate corrected overlap functions are monotonicicaly increasing up 
    to the range of full overlap. This check is performed by using a Savitzky–Golay 
    filter (scipy.signal.savgol_filter) to find the running gradients of each function, and 
    checking that they are near positive. The filter length is given by 'sgolay_width', 
    and the filter order by 'sgolay_ord' - both defined in config. The minimum gradients 
    within the fitting windows must not be less than 'min_slope' defined in config. 
    
    The checks applied in this function are described in Appendix 9 
    of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
    
    Parameters
    ----------
    
    ovp_fc : 2D array
        candidate corrected overlap functions
    rng : array
        range array for CHM15k
    top_mask : 2D array of bools
        mask defining the upper bounds of the the fitting windows
    config : pandas data frame
        thresholds and setting 
    
    Returns
    -------
    val_min_slope : array
        minumum gradients of each candidate corrected overlap function
    index_min_slope : array
        indexes of minumum gradients of each candidate corrected overlap function
    condition3 : boolean array
        results of check that min gradient is above 'min_slope', True for pass
        False for fail   

    """
    
    if np.sum ( condition3 ) > 0 :
   
        slope = ss.savgol_filter ( ovp_fc [ : 167 , : ] , window_length = int ( config [ 'sgolay_width' ].values [ 0 ] ) , polyorder = int ( config [ 'sgolay_ord' ].values [ 0 ] ) , axis = 0 , deriv = 1 , delta = rng [ 1 ] - rng [ 0 ] )
       
        val_min_slope = np.nanmin  ( slope , axis = 0)
        
        index_min_slope = np.argmin ( slope , axis = 0 ) 
        
        index_range_stop_correction = np.argmax ( top_mask , axis = 0 ) - 1
        
        condition4 = _check_conditions_4 ( val_min_slope , index_min_slope , index_range_stop_correction , config ) 
        
        return val_min_slope , index_min_slope , condition4  
   
    else:
        
        fill = np.zeros ( len ( condition3 ) )
        
        return fill , fill , fill  

def make_mask ( fit_length , fit_begin , rng ) :
    
    """Takes in arrays of 'fit_length' and 'fit_begin' that define the shiting 
    altitude windows within which a linear regressesion is to be performed. These 
    are worked up into a mask that can be applied to an array consisting of 
    the repeated mean log signal.
    
    The mask is returned in two parts that can be appled to the top and bottom
    of the 2D data array as some checks require the altitude window to start from
    a constant altitude bin. 
        
    Parameters
    ----------
    
    fit_length : 1D array
        number of altitude bins in each fitting window
    fit_begin  : 1D array   
        altitude bins to start each fitting window
    rng : 1D array
        altitude array  
        
    Returns'fit_length' and 'fit_begin'
    -------
    
    top_mask : 2D array of bools
        mask defining the upper bounds of the the altitude windows
    bottom_mask : 2D array of bools
        mask defining the lower bounds of the altitude windows
    n : 1D array
        total lengths of each unmasked window
        
    """

    wbmax = fit_begin [ -1 ]
       
    wbegin = np.repeat ( fit_begin  , len ( fit_length ) )
       
    wlen = np.tile ( fit_length  , len ( fit_begin )  )
       
    wstop = wbegin + wlen

    wbegin = np.delete ( wbegin , np.where ( wstop  > wbmax  ) )
        
    wstop = np.delete ( wstop , np.where ( wstop  > wbmax  ) )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ] , len ( wstop ) , axis = 1 )
    
    bottom_mask = ( deep_rng < wbegin ) 
    
    top_mask =  ( deep_rng >  wstop )
    
    n =  len ( rng ) - np.sum ( bottom_mask + top_mask , axis = 0 )
               
    return top_mask , bottom_mask , n 

def get_regression_residuals ( p , masked_signal , masked_signal_whole_zone , masked_rng , masked_rng_whole_zone , n ) :
    
    """Returns the sum of the squares of the residuals and the max of the abs. values of the 
    residuals between the regression line returned by simple_linear_fit and the mean log signal. 
    
    Parameters
    ----------
    
    p : array object
        object returned by simple_linear_fit - slopes and intercepts of regressions 
    masked_signal : 2D masked array
        the repeated mean log signal for the current time window masked where values
        are outside the altitude windows defined by fit_begin and fit_length
    masked_signal_whole_zone : 2D masked array
        the repeated mean log signal for the current time window masked where values
        are outside the altitude windows defined by fit_begin and fit_length but 
        extended downwards so all windows begin at 'min_fit_range'
    masked_rng : 2D array
        the repeated range array masked to match masked_signal
    masked_rng__whole_zone : 2D array
        the repeated range array maked to match masked_signal_whole_zone
    n : array of int
        total lengths of each unmasked windows in masked_signal
        
    Returns
    -------
    poly : 2D array 
        regression lines
    resid : array of floats
        sum of the squares of the residuals between poly and data
    resid_whole_zone : array of floats
        max of the absolute value of the residuals between poly and data
        
    See also@author martin osborne: martin.osborne@metoffice.gov.uk
    --------
    overlap_probe_eprofile.process_checks.simple_linear_fit
    

    """
    
    poly = p [ 0 ] + p [ 1 ] * masked_rng
       
    poly_whole_zone = p [ 0 ] + p [ 1 ] * masked_rng_whole_zone
                    
    resid = np.sqrt ( ( 1 /  n  ) * np.ma.sum ( ( masked_signal - poly ) ** 2 , axis = 0 ) )
      
    resid_whole_zone = np.ma.max ( abs ( masked_signal - poly_whole_zone ) / abs ( poly_whole_zone ) , axis = 0 )
        
    return poly , resid , resid_whole_zone
    
def make_ovp_fc ( signal_all , p , ov , rng , top_mask , config ) :
    
    """Calculates candidate corrected overlap functions using Eqs. (2) and (9) of 
    `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_.
    Also returns the maximum of the relative error between each candidate functions 
    and the reference overlap function
    
    
    Parameters
    ----------
    
    signal_all : 2D array
        unmasked array of repeated mean log signal 
    p : array object
        object returned by simple_linear_fit - slopes and intercepts of regressions
        returned by simple_linear_fit
    ov : array
        reference overlap signal
    rng : array
        range array for CHM15k
    top_mask : 2D array of bools
        mask defining the upper bounds of the the altitude windows
    config : pandas data frame
        thresholds and setting 
        
    Returns
    -------
    overlap_corr_factor : 2D array 
        array of overlap correction factors
    ovp_fc : 2D array
        candidate corrected overlap functions
    valmax : array 
        max of the relative error between the reference overlap function and the 
        candidate corrected overlap functions
    
    See also
    --------
    overlap_probe_eprofile.process_checks.check_relative_error
        
    """
        
    signal_all = np.repeat ( signal_all [ : , np.newaxis ] , np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    deep_rng = np.repeat ( rng [ : , np.newaxis ] , np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    deep_ov = np.repeat ( ov [ : , np.newaxis ] , np.shape ( top_mask ) [ 1 ] , axis = 1 )
    
    poly_all = p [ 0 ] + p [ 1 ] * deep_rng

    diff = signal_all - poly_all
    
    overlap_corr_factor = 10 ** diff
    
    overlap_corr_factor [ ( top_mask == 1 ) ] = 1
    
    ovp_fc = deep_ov * overlap_corr_factor
    
    min_overlap_valid = config [ 'min_overlap_valid' ].values [ 0 ]
    
    rel_err =  abs ( deep_ov [ rng >= min_overlap_valid , : ] - ovp_fc [ rng >= min_overlap_valid , : ] )  / abs ( deep_ov [ rng >= min_overlap_valid , : ] ) 
    
    valmax = np.nanmax ( rel_err , axis = 0 ) 
    
    return overlap_corr_factor , ovp_fc , valmax 


def _check_conditions_1 ( p , poly , resid , resid_whole_zone , config ) :
       
    con1 = ( p [ 1 ] >= config [ 'min_expected_slope' ].values [ 0 ] )
    
    con2 = ( p [ 1 ] <= config [ 'max_expected_slope' ].values [ 0 ] ) 
                    
    con3 = ( p [ 0 ] >= config [ 'min_expected_zero_fit_value' ].values [ 0 ] )
      
    con4 = ( p [ 0 ] <= config [ 'max_expected_zero_fit_value' ].values [ 0 ] )
    
    con5 =  ( resid < config [ 'thresh_resid_rel' ].values [ 0 ] * np.ma.mean ( poly , axis = 0) )
    
    con6 = ( resid_whole_zone < config [ 'thresh_resid_whole_zone' ].values [ 0 ] )
    
    condition1 = con1 * con2 * con3 * con4 * con5 * con6 
    
    return condition1

def _check_conditions_2 ( ovp_fc , ov , valmax , config ) :
    
    con7 = ( ( np.nanmax ( ovp_fc , axis = 0 ) ) <= config [ 'max_overlap_value' ].values [ 0 ]  * np.nanmax ( ov , axis = 0) )
     
    con8 = ( valmax < config [ 'thresh_overlap_valid_rel_error' ].values [ 0 ] )
    
    condition2 =  con7 * con8
    
    return condition2

def _check_conditions_3 ( relgradmagn  , config ) :
    
    relgradmagn = np.ma.masked_invalid ( relgradmagn )
    
    relgrad_max = np.ma.max ( np.ma.max ( relgradmagn , axis = 1 ) , axis = 0 )

    relgrad_mean = np.ma.mean ( np.ma.mean ( relgradmagn , axis = 1 ) , axis = 0 )
    
    con9 = ( relgrad_max  <=  config [ 'max_relgrad' ].to_numpy ( ) )
       
    con10 = ( relgrad_mean <= config [ 'max_relgrad_mean' ].to_numpy ( ) ) 
                        
    condition3 = con9 * con10
       
    return relgrad_max , relgrad_mean , condition3

def _check_conditions_4 ( val_min_slope , index_min_slope , index_range_stop_correction , config ) :
       
    val_min_slope = np.ma.masked_invalid ( val_min_slope ) 
    
    index_min_slope = np.ma.masked_invalid ( index_min_slope )
    
    con11 = ( val_min_slope >= config [ 'min_slope' ].to_numpy ( ) ) 
    
    con12 =  ( index_min_slope > index_range_stop_correction )

    condition4 = con11 + con12
    
    return condition4
    
def do_quality_checks ( rcs_0 , rng , internal_temperature , max_available_fit_range , config , ov ) :

    p , poly , resid , resid_whole_zone, top_mask , bottom_mask,  condition1 = check_fits ( rcs_0 , rng , max_available_fit_range , config ) 
    
    ovp_fc , overlap_corr_factor , valmax ,  condition2 = check_relative_error ( rcs_0  , p , ov , rng , top_mask , config , condition1 )
        
    relgradmagn , relgrad_max , relgrad_mean , condition3 = check_temporal_spatial_homogeneity ( rcs_0 , rng , overlap_corr_factor , top_mask , max_available_fit_range ,  config , condition2 )    

    val_min_slope , index_min_slope , condition4 = check_monotonic ( ovp_fc , rng , top_mask , config , condition3 ) 
    
    conditionals = condition2 * condition3 * condition4

    results_df = create_results_df ( rng , p , poly ,  resid , resid_whole_zone , ov , ovp_fc , valmax , relgrad_max , relgrad_mean , val_min_slope , index_min_slope , top_mask , bottom_mask , internal_temperature , conditionals, config )

    return results_df

   


