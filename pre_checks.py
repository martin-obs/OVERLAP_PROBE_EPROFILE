# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
    Started November 2022

    functions to perform pre-checks on ceilometer data to be used

    in calculating an overlap function.  This is a translation /

    refactoring of Matlab code written by Maxime Hervo, Rolf Ruefenacht 

    and Melania Van Hove.

    @author martin osborne: martin.osborne@metoffice.gov.uk

'''

import numpy as np
import datetime
import scipy.signal as ss

def at_least_one_profile ( flag ) :

    '''

    Check at least one profile in sliding window

    '''

    return ~np.all ( flag )

def all_clear_sky ( check , sci ) :

    '''

    Check all profiles in sliding window come with a clear 

    sky condition

    '''

    if check :

        return np.all ( sci == 0)

    else :

        return False

def enough_clear_range_cbi ( check ,  cbi , config ) :

    '''

    Check all minimum cloud base heights leave enough 

    bins in the fitting window as defined as 'min_fit_range' + 'min_fit_length'

    and return lowest as new max range

    '''

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

    '''

    Calculate variance for each profile within altitude ranges

    'min_range_std_over_mean' and 'max_available_fit_range' and

    find the first altitude at which this is above 'max_std_over_mean'.

    This is the new 'max_available_fit_range'. Checks that this leaves 

    enough bins in the fitting window.

    '''

    if check:

        min_r = np.where ( rng <= config [ 'min_range_std_over_mean' ].to_numpy ( ) ) [ 0 ] [ -1 ]

        max_r = np.where ( rng <= max_available_fit_range ) [ 0 ] [ -1 ]

        dt_sliding_variance=  int(config [ 'dt_sliding_variance' ])

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

                max_available_fit_range = rng [ min_r + ind ]

                return max_available_fit_range  >=  (config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) , max_available_fit_range , std_over_mean [ ind ] 

        else :

            return max_available_fit_range >=  (config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) , max_available_fit_range , 0.0

    else :

        return False , max_available_fit_range , 0.0


def conv2 ( x , direction = None ) :

    '''

    Matlab's conv2 and Python's scipy.signal.convolve2d behave slightly differently

    This function reproduces the behaviour of the Matlab function. Used in 'check_grads'

    to match the Matlab code results

    '''
    
    if direction == 'y' :
        
        grad = np.asarray ( [ 1 , 0 , -1 , 2 , 0 , -2 , 1 , 0 , -1 ] ).reshape ( ( 3 , 3 ) )
        
    elif direction == 'x' :

        grad = np.asarray ( [ 1 , 2 , 1 , 0 , 0 , 0 , -1 , -2 , -1 ] ).reshape ( ( 3 , 3 ) )

    return np.rot90 ( ss.convolve2d ( np.rot90 ( x , 2 ) , np.rot90 ( grad , 2 ) , mode = 'same' ) , 2 )


def check_grads ( check ,  rcs_0 , rng , config , max_available_fit_range ) :

    '''

    Finds the range bin at which the relative gradients of the Log10 ( rcs ) in the X and 

    Y directions are below 'max_relgrad' and checks that this leaves enough bins in the 

    fitting window. Also find the highest range bin where the max and mean of the magnitide 

    of the relative gradients from 'first_range_gradY' are less than 'max_relgrad' and

    'max_relgrad_mean'

    '''

    if check :

        X , Y = 0.0 , 0.0 

        m1, m2 , m3 = max_available_fit_range , max_available_fit_range , max_available_fit_range

        min_r = np.where ( rng >= config [ 'min_range_std_over_mean' ].to_numpy ( ) ) [ 0 ] [ 0 ]

        max_r = np.where ( rng <= max_available_fit_range ) [ 0 ] [ -1 ] 

        rconv = rng [ min_r + 1 : max_r - 1 ]

        lRCS = np.log10 ( abs ( rcs_0 [ : , min_r : max_r ] ) )

        gradY = conv2 ( lRCS , direction =  'y' )

        gradX = conv2 ( lRCS, direction = 'x' )

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

            if (  np.nanmax(chunk)  <= config [ 'max_relgrad' ].to_numpy( ) ) and (np.nanmean(chunk ) <= config [ 'max_relgrad_mean' ].to_numpy( ) ) :

                m3 = rconv [ irgradmagn [ k ] -1 ]

                break

        max_available_fit_range = np.min ( [ max_available_fit_range , m1 , m2 , m3 ] )

        return max_available_fit_range >=  (config [ 'min_fit_range' ]  + config [ 'min_fit_length' ] ).to_numpy ( ) , max_available_fit_range , X , Y

    else :

        return False , max_available_fit_range , 0.0 , 0.0 


