# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""This module contans functions to perform the checks and processing decribed in 
section 1 of the appendix of `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
titled "Determination of the fitting intervals"

This is a translation / refactoring of Matlab code written by Maxime Hervo, Yann Poltera,
Rolf Ruefenacht and Melania Van Hove.

    @author martin osborne: martin.osborne@metoffice.gov.uk
"""

import numpy as np
import datetime
from overlap_probe_eprofile.overlap_utils import conv2d

def at_least_one_profile ( flag ) :

    """Part 1 of the checks described in section 1 of the appendix of 
    `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
    Check there is at least one profile within the current time window. 
    
    Parameters
    ----------
    
    flag : array of bools
        Boolean array indicating which profiles are to be ignored as filled
        missing data
        
    Returns
    -------
    
    Result : bool
        True or False for pass or fail
        
        
    See also
    --------
    overlap_probe_eprofile.process_L1.Eprofile_Reader.fill_gaps

    """

    return ~np.all ( flag )

def all_clear_sky ( check , sci ) :

    """Part 2 of the checks described in section 1 of the appendix of 
    `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_
    
    Check all profiles in current time window come with a clear sky condition. 
    
    Parameters
    ----------
    
    check : bool
        result of first check ( at_least_one_profile )
    
    sci : list of ints
        sky conditions for each profile
        
    Returns
    -------
    
    Result : bool
        True or False for pass or fail
        
        
    See also
    --------
    overlap_probe_eprofile.pre_checks.at_least_one_profile

    """

    if check :

        return np.all ( sci == 0)

    else :

        return False

def enough_clear_range_cbi ( check ,  cbi , config ) :

    """Checks described in section 2 of the appendix of 
    `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_

    Checks that the minimum cloud base height within the current time window leaves 
    enough range bins below it to allow for a large enough fitting window 
    as defined by 'min_fit_range' + 'min_fit_length' in config. If this test 
    is passed, the lowest cloud base height is returned as the max_available_fit_range
    
    Parameters
    ----------
    
    check : bool
        result of all_clear_sky test
    cbi : array of floats
        cloud base heights for current time window  
    config : class attribute
        contains settings and thresholds
    
    Returns
    -------
    Result: bool
        True or False for pass or fail
    max_available_fit_range : float
        maximum altitude at which test is passed
        
    See also
    --------
    overlap_probe_eprofile.pre_checks.all_clear_sky

    """

    if check:

        cbi [ cbi < 0 ] = 15000

        max_fit_range = config ['max_fit_range'].to_numpy()

        lower_cbs = np.nanmin ( cbi , axis = 1 )

        result = np.all ( lower_cbs >= (  config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) )

        lower_cbs [ np.where ( lower_cbs >= max_fit_range ) ] = max_fit_range

        max_available_fit_range = np.min ( lower_cbs )

        return  result , max_available_fit_range

    else :

        return False , 0

def running_variance ( check ,  rcs_0 , rng , dt , config , max_available_fit_range ) :
    
    """Checks described in section 3.1.1 of the appendix of 
    `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_

    Calculates the variance for a subset of profiles within the current time window and 
    within a defined altitude range. The subset is defined by 'dt_sliding_variance'
    in config, and is that number of profiles wide. The subset is moved by one profile until 
    the end of the curent time widow is reached. The altitude range is defined by 
    'min_range_std_over_mean' (from config) and 'max_available_fit_range' from the previous
    tests. The first altitude at which the variance exceeds 'max_std_over_mean', as defined in 
    config, is returned. This is the new 'max_available_fit_range'. If this leaves enough bins 
    in the available fitting window (defined by 'min_fit_range' + 'min_fit_length' in config) 
    then the test is passed.
    

    
    Parameters
    ----------
    
    check : bool
        result of enough_clear_range_cbi test
    rcs_0 : 2D array of floats
        range corrected signal for current time window
    rng : array of floats
        range array for rcs_0
    dt : array of datetimes
        time array for current time window
    config : class attribute
        contains settings and thresholds
    max_available_fit_range : float
        maximum altitude at which all tests so far have been passed
    
    Returns
    -------
    Result: bool
        True or False for pass or fail
    max_available_fit_range : float
        maximum altitude at which test is passed
    variance : float
        variance at max_available_fit_range
        
    See also
    --------
    overlap_probe_eprofile.pre_checks.enough_clear_range_cbi
    
    
    """

    if check:

        min_r = np.where ( rng <= config [ 'min_range_std_over_mean' ].to_numpy ( ) ) [ 0 ] [ -1 ]

        max_r = np.where ( rng <= max_available_fit_range ) [ 0 ] [ -1 ]

        dt_sliding_variance=  int ( config [ 'dt_sliding_variance' ].iloc [ 0 ] )

        max_std_over_mean = config [ 'max_std_over_mean' ].to_numpy()

        variance_times = dt + datetime.timedelta ( minutes = dt_sliding_variance )

        start_inds = [ *range ( 0 , len ( dt ) ) ]

        end_inds = [ np.where ( np.asarray ( dt ) <= f ) [ 0 ] [ -1 ] for f in variance_times ]

        std_over_mean = np.zeros ( max_r - min_r )

        signal1 = np.abs ( rcs_0 [  :  , min_r : max_r  ] )

        for s , f in zip ( start_inds , end_inds ) :

            if dt [ s ] <= ( dt [ -1 ] -  datetime.timedelta ( minutes = dt_sliding_variance ) ) :

                signal2 = np.abs ( rcs_0 [ s : f , min_r : max_r  ] )

                std_over_mean_tmp = np.nanstd ( np.emath.log10 ( signal2 ) , axis = 0 ) / np.nanmedian ( np.emath.log10 ( signal1 )  , axis = 0 )

                std_over_mean = np.maximum ( std_over_mean , std_over_mean_tmp )

        i_first_over_thresh_std_over_mean = np.argwhere ( std_over_mean >= max_std_over_mean )

        if np.shape (i_first_over_thresh_std_over_mean ) [ 0 ] != 0:

                ind = i_first_over_thresh_std_over_mean [ 0 ] [ 0 ]

                max_available_fit_range = rng [ min_r + ind  ]

                return max_available_fit_range  >=  (config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) , max_available_fit_range , std_over_mean [ ind ] 

        else :

            return max_available_fit_range >=  (config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) , max_available_fit_range , 0.0

    else :

        return False , max_available_fit_range , 0.0





def check_grads ( check ,  rcs_0 , rng , config , max_available_fit_range ) :
    
    """Checks described in section 3.1.2, 3.2 and 3.4 of the appendix of 
    `amt-9-2947-2016 <https://amt.copernicus.org/articles/9/2947/2016/amt-9-2947-2016.pdf>`_

    Calls conv2d to find the range bin at which the relative gradients of log10 ( rcs_0 ) along 
    the altitude and time directions are below 'max_relgrad' and checks that this leaves 
    enough bins in the fitting window. Also find the highest range bin where the 
    max and mean of the magnitide of the relative gradients from 'first_range_gradY' 
    are less than 'max_relgrad' and 'max_relgrad_mean'
    

    
    Parameters
    ----------
    
    check : bool
        result of running_variance test
    rcs_0 : 2D array of floats
        range corrected signal for current time window
    rng : array of floats
        range array for rcs_0
    dt : array of datetimes
        time array for current time window
    config : class attribute
        contains settings and thresholds
    max_available_fit_range : float
        maximum altitude at which all tests so far have been passed
    
    Returns
    -------
    Result: bool
        True or False for pass or fail
    max_available_fit_range : float
        maximum altitude at which test is passed
    variance : float
        variance at max_available_fit_range
        
    See also
    --------
    overlap_probe_eprofile.pre_checks.running_variance test   
    overlap_probe_eprofile.pre_checks.conv2
    
    """

    if check :

        X , Y = 0.0 , 0.0 

        m1 = m2 = m3 = max_available_fit_range 

        min_r = np.where ( rng >= config [ 'min_range_std_over_mean' ].to_numpy ( ) ) [ 0 ] [ 0 ]

        max_r = np.where ( rng <= max_available_fit_range ) [ 0 ] [ -1 ] 

        rconv = rng [ min_r + 1 : max_r - 1 ]

        lRCS = np.log10 ( abs ( rcs_0 [ : , min_r : max_r ] ) )

        gradY = conv2d ( lRCS , direction =  'y' )

        gradX = conv2d ( lRCS, direction = 'x' )

        relgradY = abs ( gradY ) / abs (lRCS)

        relgradX = abs ( gradX ) / abs (lRCS)

        relgradmagn = np.sqrt ( gradX  ** 2 + gradY  ** 2 ) / abs ( lRCS )

        relgradYmax = np.max ( relgradY [ 1 : -1 , 1 : -1] , axis = 0  )

        relgradXmax = np.max ( relgradX [ 1 : -1 , 1 : -1 ] , axis = 0  )

        relgradmagn_sub = relgradmagn [ 1 : -1 , 1 : -1 ]

        irgradmagn  = np.where ( ( rconv >= config [ 'first_range_gradY' ].to_numpy ( ) ) * ( rconv <= max_available_fit_range ) ) [ 0 ]

        i_first_over_thresh_relgradY = np.where ( relgradYmax [ np.where ( rconv >= config [ 'first_range_gradY' ].to_numpy ( ) ) ] >= config [ 'max_relgrad' ].to_numpy ( ) )

        i_first_over_thresh_relgradX = np.where ( relgradXmax [ np.where ( rconv >= config [ 'min_range_std_over_mean' ].to_numpy ( ) ) ] >= config [ 'max_relgrad' ].to_numpy ( ) )

        if np.shape( i_first_over_thresh_relgradY ) [ 1 ] > 1 :

            Y = relgradYmax [ i_first_over_thresh_relgradY [ 0 ] [ 0 ] ]

            m1 = rconv [ 10 + i_first_over_thresh_relgradY [ 0 ] [ 0 ] ]

        if np.shape( i_first_over_thresh_relgradX ) [ 1 ]  > 1 :

            X = relgradXmax [ i_first_over_thresh_relgradX [ 0 ] [ 0 ] ]

            m2 = rconv [ i_first_over_thresh_relgradX [ 0 ] [ 0 ] ]

        for k in range ( np.shape(irgradmagn)[0]-1 , 1 , -1 ) :

            chunk = ( relgradmagn_sub [ : , irgradmagn [ 0 ] : irgradmagn [ k ] ] )

            if ( np.nanmax(chunk)  <= config [ 'max_relgrad' ].to_numpy( ) ) and (np.nanmean(chunk ) <= config [ 'max_relgrad_mean' ].to_numpy( ) ) :

                m3 = rconv [ irgradmagn [ k ] -1 ]

                break

        max_available_fit_range = np.min ( [ max_available_fit_range , m1 , m2 , m3 ] )

        return max_available_fit_range >=  (config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) , max_available_fit_range , X , Y , m1 , m2 , m3 

    else :

        return False , max_available_fit_range , 0.0 , 0.0 , 0.0 , 0.0, 0.0


