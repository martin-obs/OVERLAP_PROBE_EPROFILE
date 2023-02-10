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
import datetime
import scipy.signal as ss
#import numpy.polynomial.polynomial as pol

def get_ov_ok_info ( A_dict ) :
    
    '''
    
    Takes in a dictionary of results from process checks, loops through and 
    
    constructs lists of time intervals containing good samples, and corresponding 
    
    litst of upper and lower range limits and corrected overlap functions. 
    
    '''
    
    time_intervals = [ key for key in A_dict.keys ( ) if bool ( A_dict [ key ] [ 'corrected_ov' ] )  ]
    
    starts = [ datetime.datetime.strptime( t [ 8:16 ] + ' ' + t [ 27:37 ] , '%H:%M:%S %Y-%m-%d' ) for t in time_intervals ]
    
    ends = [ datetime.datetime.strptime( t [ 18:27 ] + ' ' + t [ 27:37 ] , '%H:%M:%S %Y-%m-%d' ) for t in time_intervals ]
    
    times = list ( zip ( starts , ends ) )
    
    ranges = [ list ( A_dict [ key ] [ 'corrected_ov' ].keys ( ) ) for key in A_dict.keys ( ) if bool ( A_dict [ key ] [ 'corrected_ov' ] )  ]
    
    lower = [ [ float ( item [ 0:6 ] ) for item  in r ] for r in ranges ]
    
    upper = [ [ float ( item [ 22:28 ] ) for item  in r ] for r in ranges ]
    
    ov_fcs = [ list ( A_dict [ key ] [ 'corrected_ov' ].values(  ) )  for key in A_dict.keys ( ) if bool ( A_dict [ key ] [ 'corrected_ov' ] )  ]
    
    return times , lower , upper , ov_fcs
    
def check_ov_fcs_in_time_ranges ( results_dict , dt , rng , rcs , ov , config ) :
    
    '''
    
    Takes each corrected ov_fc that passed all checks so far, and 
    
    applies this to the RCS (truncated to the max range of the oc_fc being
                           
    tested) within each time range that contains any valid corrected 
    
    ov_fc. Finds the relative gradient of this corrected RCS and check 
    
    it is below a threshold
    
    '''
    
    times , lower , upper , ov_fcs = get_ov_ok_info ( results_dict  )
    
    times = np.asarray(times)
    
    max_ranges = [ np.max ( item ) for item in upper ] 
    
    passed_inds = []
    
    for i , test_functions in enumerate (ov_fcs) :
        
        max_r = max_ranges [ i ]
        
        window_inds = ( dt >= times [ i , 0 ] ) * ( dt <= ( times [ i , 1 ] ) )
        
        range_inds = ( rng >= config [ 'min_range_std_over_mean' ].values [ 0 ]  ) * ( rng <= max_r )
        
        RCSc = rcs * ov 
        
        for test_function in test_functions:

            signal = np.log10 ( abs ( RCSc / test_function ) ) [ : , range_inds ] [ window_inds , : ]
           
            gradY = conv2 (  signal , direction =  'y' )

            gradX = conv2 (  signal , direction = 'x' )
            
            relgradmagn = ( np.sqrt ( gradX ** 2 + gradY ** 2 ) / abs ( signal ) ) [ 1 : -1 , 1 : -1 ]
                        
            condition1 = ( np.nanmax ( np.nanmax ( relgradmagn , axis = 1 ) ) <=  config [ 'max_relgrad' ].to_numpy ( ) )
                
            condition2 = ( np.nanmean ( np.nanmean ( relgradmagn , axis = 1 ) ) <= config [ 'max_relgrad_mean' ].to_numpy ( ) )
            
            if all ( [ condition1 , condition2 ] ) :
                
                condition3 = check_standard_over_mean ( times , max_ranges , test_function ,  RCSc , dt , rng ,  config )
                
                if condition3 :
                               
                    passed_inds.append ( True )
                
                else:
                    
                    passed_inds.append ( False )
                
    return passed_inds
                
            
def check_standard_over_mean ( times , max_ranges , test_function ,  RCSc , dt , rng ,  config ) :
     
    '''
    
    Take each corrected ov_fc that passed all checks so far, and 
    
    apply this to the RCS (truncated to the max range of the oc_fc being
                           
    tested) withing each time range that contains any valid corrected 
    
    ov_fc. Check that the standard over mean of the corrected signal within 
    
    a sliding time window it is below a threshold
    
    '''

    print ('checking standard over mean')    
    
    for i in range (np.shape(times)[0]):
    
        window_inds = ( dt >= times [ i , 0 ] ) * ( dt <= ( times [ i , 1 ] ) )
        
        dt_window = dt [ window_inds ]
        
        max_r = max_ranges [ i ] 
        
        range_inds = ( rng >= config [ 'min_range_std_over_mean' ].values [ 0 ]  ) * ( rng <= max_r )
          
        dt_sliding_variance=  int ( config [ 'dt_sliding_variance' ] )
        
        max_std_over_mean = config [ 'max_std_over_mean' ].to_numpy ( )
        
        variance_times =  dt_window + datetime.timedelta ( minutes = dt_sliding_variance )

        start_inds = [ *range ( 0 , len (  dt_window ) ) ]

        end_inds = [ np.where ( np.asarray (  dt_window ) <= f ) [ 0 ] [ -1 ] for f in variance_times ]
        
        np.seterr(divide='ignore')
        
        signal1 = np.log10 ( abs ( RCSc / test_function ) ) [ : , range_inds ] [ window_inds , : ]
        
        std_over_mean = np.zeros ( np.shape ( signal1 ) [ 1 ] )

        for s , f in zip ( start_inds , end_inds ) :

            if  dt_window [ s ] <= (  dt_window [ -1 ] -  datetime.timedelta ( minutes = dt_sliding_variance ) ) :

                signal2 = np.abs ( signal1  [ s : f , :  ] )

                std_over_mean_tmp = np.nanstd ( np.emath.log10 ( signal2 ) , axis = 0 ) / np.nanmedian ( np.emath.log10 ( signal1 )  , axis = 0 )

                std_over_mean = np.maximum ( std_over_mean , std_over_mean_tmp )
                
                if np.nanmax ( std_over_mean > max_std_over_mean ):
                    
                    return False
                
                else :
                    
                    pass
                
        if np.nanmax ( std_over_mean < max_std_over_mean ):
                               
            return True
                    
def remove_failed ( A_dict, passed_inds , rng  , config ) :
    
    passed_inds = passed_inds
    
    ovs = np.zeros(len (rng))
    
    intervals = np.array ([0 , 0] , dtype = float )
    
    times , lower , upper , ov_fcs = get_ov_ok_info ( A_dict )
    
    lower = [ item for sublist in lower for item in sublist]
    
    upper = [ item for sublist in upper for item in sublist]
    

        
    for t ,  f in enumerate (ov_fcs) :
        
        ts = np.asarray ( times ) [ t ,  0 ].timestamp()
        
        tf = np.asarray ( times )  [ t , 1 ].timestamp()
        
        intervaltmp = np.zeros ( ( np.shape(np.asarray ( f ) ) [ 0 ] , 2 ) )
        
        intervaltmp [ : , 0 ] = ts
        
        intervaltmp [ : , 1 ] = tf
        
        intervals  = np.vstack ( ( intervals , intervaltmp ) )    
        
        ovs = np.vstack ( ( ovs , np.asarray ( f ) ) )
        
    intervals = intervals [1:,:]
                             
    ovs = ovs [ 1 : , : ]
    
    ovs = ovs [ passed_inds , : ]
    
    intervals = intervals [ passed_inds , : ]
    
    rng_intervals = np.asarray(list(zip (lower,upper)))
 
    print ( np.shape ( ovs ) , np.shape (intervals ) , np.shape(rng_intervals))
    
    final_ovs , final_ov  ,  outlier_pass_inds = remove_outliers (ovs , rng , config )
    
    print (np.shape ( final_ovs ) )
    
    intervals = intervals [ outlier_pass_inds, :]
    
    rng_intervals = rng_intervals [ outlier_pass_inds, :]
    
    return intervals , rng_intervals , final_ovs , final_ov
    
def remove_outliers (  ovs , rng , config ) :
    
    whiskers = config [ 'whiskers_length' ].to_numpy()
    
    pc_25 = []
    pc_75 = []
    pc_50 = []
    
    for idx in range( 0 , len(rng) ) :
           
        pc_25.append( prctile ( np.asarray( ovs [ : , idx ] ), 25 ) )
        
        pc_75.append(prctile ( np.asarray(ovs [ : , idx ] ), 75 ))
        
        pc_50.append ( prctile ( np.asarray(ovs [ : , idx ]) , 50  ))
        
    pc_25 = np.asarray (  pc_25 )
    pc_75 = np.asarray (  pc_75 )
    pc_50 = np.asarray (  pc_50 )   
   

    outliers_plus = pc_50 + whiskers*(pc_75-pc_25)
        
    outliers_minus = pc_50 - whiskers*(pc_75-pc_25)
    
    ov_final = np.zeros ( len ( rng ) ) 
    
    outlier_pass_inds = [ ]

    for r , ov_func in enumerate ( ovs ) :

        if  ( all (ovs[r,:]  >= outliers_minus  ) ) and ( all (ovs[r,:]   <= outliers_plus ) ) :
            
            ov_final = np.vstack ( ( ov_final , ov_func ) )
            
            outlier_pass_inds.append ( True )
        
        else:
            
            outlier_pass_inds.append ( False )
    
    return ov_final [ 1 : , : ] , np.nanmean ( ov_final , axis = 0 ) , outlier_pass_inds
        

def quantile ( x , q ) :
    
    n = len(x)
    
    y = np.sort(x)
    
    return ( np.interp ( q , np.linspace ( 1 / ( 2 * n ) , ( 2 * n - 1 ) / ( 2 * n ), n ), y ) )

def prctile ( x , p ) :
      
    return ( quantile ( x ,  p  / 100 ) )   
      
    
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