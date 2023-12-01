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
import gc

from overlap_probe_eprofile.process_checks import conv3d


def get_ov_ok_info_df ( A_dict , rng ) :
    
    '''
    
    Takes in a dictionary of results from process checks, loops through and 
    
    constructs lists of time intervals containing good samples, and corresponding 
    
    lists of upper and lower range limits and overlap functions. 
    
    '''
        
    time_intervals  = [ np.repeat ( key , np.shape (  A_dict [ key ] [ 'data_frame' ] [ A_dict [ key ] ['data_frame'] [ 'pass_all'] == True ] ) [ 0 ] )  for key in A_dict.keys ( ) if  bool ( any (A_dict [ key ] ['data_frame'] ['pass_all'] ) ) ] 
        
    #print (time_intervals)
    
    time_intervals = [ item for sublist in time_intervals for item in sublist ]
     
    starts = [ datetime.datetime.utcfromtimestamp ( ( float ( t.split ( ' ' ) [ 1 ] ) ) ) for t in time_intervals ]
    
    ends =  [ datetime.datetime.utcfromtimestamp ( ( float ( t.split ( ' ' ) [ 3 ] ) ) ) for t in time_intervals ]
    
    time_ints = np.asarray( list ( zip ( starts , ends ) ) )

    ts = [ key for key in A_dict.keys ( ) if  bool ( any ( A_dict [ key ] ['data_frame'] ['pass_all'] ) ) ]
    
    ov_fcs = [ item for sublist in [  list ( A_dict [ key ] [ 'data_frame' ] [ A_dict [ key ] ['data_frame'] [ 'pass_all'] == True ].to_numpy ( ) [ : , 27 : ] )  for key in ts ] for item in sublist ]
    
    temperatures = [ item for sublist in [ list ( A_dict [ key ] [ 'data_frame' ] [ A_dict [ key ] ['data_frame'] [ 'pass_all'] == True][ 'internal_temperature'] ) for key in ts ] for item in sublist ]
    
    starts = [ datetime.datetime.utcfromtimestamp ( ( float ( t.split ( ' ' ) [ 1 ] ) ) ) for t in ts ]
    
    ends =  [ datetime.datetime.utcfromtimestamp ( ( float( t.split ( ' ' ) [ 3 ] ) ) ) for t in ts ]
    
    times = np.asarray( list ( zip ( starts , ends ) ) )
 
    lower = [ list ( A_dict [ key ] [ 'data_frame' ] [ A_dict [ key ] ['data_frame'] [ 'pass_all'] == True][ 'rng_start'] ) for key in ts ]
    
    upper = [ list ( A_dict [ key ] [ 'data_frame' ] [ A_dict [ key ] ['data_frame'] [ 'pass_all'] == True][ 'rng_end'] ) for key in ts ]
    
    max_rng_for_interval = [ item for sublist in [ np.repeat( np.max ( r ) , len ( r ) ) for r in upper ] for item in sublist ]
    
    #print (len(ts))
    
    print (len(ov_fcs) , len(temperatures))
    
    #print (len(max_rng_for_interval))
         
    return time_ints , times , lower , upper , max_rng_for_interval , ov_fcs , temperatures


def make_deep_signal ( dt , times , RCSc ) :
    
     '''
     
     Takes the RCS already truncated to altitude fitting window 
     
     and pulls out time windows corresponding to 'times' and 
     
     stacks them into third dimension. Returns array of dimension
     
     ( time_interval_length , altitude_window , number_of_time_intervals )
     
     '''
          
     deep_time = np.repeat ( dt [ : , np.newaxis ] , np.shape ( times ) [ 0 ] , axis = 1 )

     unsorted_time_window_inds =  np.argwhere ( (  deep_time >= times [ : , 0 ] ) * ( deep_time <= times [ : , 1 ] ) )
     
     profiles_per_time_window  = int ( np.shape ( unsorted_time_window_inds ) [ 0 ] / np.shape ( times ) [ 0 ] )
     
     time_window_inds = unsorted_time_window_inds [ unsorted_time_window_inds [ : , 1 ].argsort ( ) ] [ : , 0 ]
     
     time_window_inds = np.sort ( time_window_inds.reshape ( np.shape ( times ) [ 0 ] , profiles_per_time_window  ) ) 
     
     deep_signal = np.rollaxis ( RCSc [ time_window_inds , : ] , 0 , 3 ).astype('float32')

     return deep_signal
 
    
def make_variance_windows ( dt , times , config ) :
   
    '''
    
    Takes the RCS already truncated to altitude fitting window 
    
    and pulls out time windows corresponding to 'times' and 
    
    stacks them into third dimension. Returns array of dimension
    
    ( time_interval_length , altitude_window , number_of_time_intervals )
    
    '''
    
    dt_sliding_variance=  int ( config [ 'dt_sliding_variance' ].iloc [ 0 ] )
    
    deep_time = np.repeat ( dt [ : , np.newaxis ] , np.shape ( times ) [ 0 ] , axis = 1 )
    
    unsorted_time_window_inds =  np.argwhere ( (  deep_time >= times [ : , 0 ] ) * ( deep_time < times [ : , 1 ] ) )
    
    profiles_per_time_window  = int ( np.shape ( unsorted_time_window_inds ) [ 0 ] / np.shape ( times ) [ 0 ] ) 
    
    time_window_inds = unsorted_time_window_inds [ unsorted_time_window_inds [ : , 1 ].argsort ( ) ] [ : , 0 ]
    
    time_window_inds = np.sort ( time_window_inds.reshape ( np.shape ( times ) [ 0 ] , profiles_per_time_window  ) ) 
    
    profiles_per_sliding_window = int ( datetime.timedelta ( minutes = dt_sliding_variance ) / ( dt [ 1 ] - dt [ 0 ] ))
    
    no_sliding_windows = 1 + profiles_per_time_window - profiles_per_sliding_window
    
    sliding_inds = np.repeat (  np.arange ( 0 , profiles_per_sliding_window , 1 ) [ np.newaxis , : ] , no_sliding_windows , axis = 0) + np.arange ( 0 , no_sliding_windows , 1 ) [ : , None ]
    
    return sliding_inds


def do_sort_checks ( results_dict , dt , rng , rcs , ov , config ) :
    
    time_intervals , times , lower , upper , max_rng , ov_fcs , temperatures = get_ov_ok_info_df ( results_dict , rng  )
    
    max26 = [ np.max ( r )  for r in upper ]
    
    if np.shape ( ov_fcs ) [ 0 ] == 0 :
        
        return False
    
    if np.shape ( ov_fcs ) [ 0 ] <= config [ 'min_nb_samples_for_skipping_good_test' ].values [ 0 ] :
      
        RCSc = rcs * ov 
           
        range_index = ( rng >= config [ 'min_range_std_over_mean' ].values [ 0 ] ) * ( rng <= np.max ( max_rng ) )
        
        RCSc = RCSc [ : , range_index ]
        
        rng = rng [ range_index ]
        
        ov_fcs = np.asarray(ov_fcs) [ : , range_index ]     
           
        deep_signal = make_deep_signal ( dt , times , RCSc )
        
        condition1 = check_variance (  deep_signal  , ov_fcs , times , dt  , config )
        
        condition2 = check_rel_grad_magn ( rng  , max26 , ov_fcs  , times , deep_signal , config )
        
        print ('After good tests = ' , sum(condition1*condition2))
          
        return condition1 * condition2
    
    else:
        
        return np.ones( np.shape ( ov_fcs ) [ 0 ] ).astype(bool)
    
      
def check_rel_grad_magn ( rng  , max26, ov_fcs , times , deep_signal , config ):
    
    stack_size = np.shape ( ov_fcs ) [ 0 ]
      
    deep_signal = np.tile ( deep_signal , stack_size  ).astype('float32')
    
    ovs_to_test = np.repeat ( ov_fcs , np.shape ( times ) [ 0 ] , axis = 0).astype('float32')
    
    deep_signal = np.asarray ( abs ( deep_signal / ovs_to_test.T ) ).astype('float32')
      
    del ovs_to_test
    
    gc.collect ( )
    
    lcc = np.log10 ( deep_signal  ).astype('float32')
               
    gradY = ( conv3d (  lcc , direction =  'y' ) ).astype('float32')

    gradX = ( conv3d ( lcc , direction = 'x' ) ).astype('float32')
    
    gradY =  ( ( np.sqrt ( gradX ** 2 + gradY ** 2 ) / abs ( lcc ) ) ).astype('float32')
    
    del gradX
    
    del lcc
    
    gc.collect ( ) 
    
    gradmax = np.nanmax ( gradY [ 1:-1 , : , :] , axis = 0 )
    
    a = np.asarray(np.where(rng[:,None] == np.asarray(max26).astype('float32')))

    a = a[:,np.argsort(a[1,:])]

    a = a [0,:]
    
    a = np.tile( a, stack_size )
    
    for i , j in enumerate (a) :
        
        gradmax [j-1:,i] = np.nan
                
    con1 =  (np.nanmax ( gradmax [  1:-1 , : ]  , axis = 0 )  <=  config [ 'max_relgrad' ].to_numpy ( ) ).astype(bool)
  
    con2 = ( np.nanmean ( np.nanmean ( gradY [ 1:-1 , 1:-1 , : ] , axis = 1 ),axis = 0 ) <= config [ 'max_relgrad_mean' ].to_numpy ( ) ).astype(bool)
   
    con1 = np.multiply.reduceat ( con1 , np.arange ( 0 , len ( con1 ) , np.shape ( times ) [ 0 ] ) ).astype(bool)
    
    con2 = np.multiply.reduceat ( con2 , np.arange ( 0 , len ( con2 ) , np.shape ( times ) [ 0 ] ) ).astype(bool)
    
    #condition  = np.multiply.reduceat ( con12 , np.arange ( 0 , len ( con12 ) , np.shape ( times ) [ 0 ] ) ).astype(bool)

     
    condition = con1 * con2
      
    return condition 
    
   
def check_variance ( signal  , ov_fcs , times , dt  , config )  :
    
    stack_size = np.shape ( ov_fcs ) [ 0 ]
      
    signal = np.tile ( signal , stack_size  ).astype('float32')
    
    ovs_to_test = np.repeat ( ov_fcs , np.shape ( times ) [ 0 ] , axis = 0).astype('float32')
    
    signal = np.log10 ( np.asarray ( abs ( signal / ovs_to_test.T ) ) ).astype('float32')
    
    sliding_window_inds = make_variance_windows ( dt , times , config ).astype('int32')
    
    denomenator =  np.nanmedian ( signal , axis = 0 ).astype('float32')
            
    variance = stdomean (sliding_window_inds , signal , denomenator , np.shape ( ov_fcs ) [ 0 ] ,  np.shape ( times ) [ 0 ] ).astype('float32')  
       
    condition = np.nanmax ( variance , axis = 0 )  <  config [ 'max_std_over_mean' ].to_numpy ( )
    
    return condition
    
   
def stdomean (sliding_window_inds , deep_signal , denomenator , ov_shape ,  times_shape  ):
    
    variance = np.zeros( ( np.shape ( sliding_window_inds ) [ 0 ] ,  ov_shape ) ).astype('float32')
    
    for sw in range ( np.shape ( sliding_window_inds ) [ 0 ]  ) :
        
        v = np.nanstd (deep_signal [ sliding_window_inds [ sw ] , : , : ] ,  axis = 0 ) / denomenator
             
        v = np.maximum.reduceat ( v.T , np.arange ( 0 , np.shape ( v ) [ 1 ] , times_shape ) )
        
        variance [ sw , : ] = np.max ( v , axis = 1 )
                    
    return variance    

def remove_failed ( results_dict , passed_inds , rng  , config ) :
    
    
    passed_inds = passed_inds
    
    ovs = np.zeros ( len ( rng ) )
    
    intervals = np.array ( [ 0 , 0 ] , dtype = float )
    
    intervals , times , lower , upper ,max_rng_for_interval  , ov_fcs , temperatures = get_ov_ok_info_df ( results_dict , rng  )
        
    lower = [ item for sublist in lower for item in sublist]
    
    upper = [ item for sublist in upper for item in sublist]
                                 
    ovs = np.asarray(ov_fcs)
    
    ovs = ovs [ passed_inds , : ]
    
    intervals = intervals [ passed_inds , : ]
    
    temperatures = np.asarray ( temperatures ) [ passed_inds ]
    
    rng_intervals = np.asarray ( list ( zip ( lower , upper ) ) ) [ passed_inds , : ]
 
    #print ( np.shape ( ovs ) , np.shape (intervals ) , np.shape ( rng_intervals ) )
    
    final_ovs , final_ov  ,  outlier_pass_inds = remove_outliers ( ovs , rng , config )
    
    #print ('final ovs b4 outlier removal= ' , np.shape ( final_ovs ) )
    
    intervals = intervals [ outlier_pass_inds, :]
    
    temperatures = temperatures [ outlier_pass_inds ]
    
    rng_intervals = rng_intervals [ outlier_pass_inds, : ]
    
    return intervals , rng_intervals , final_ovs , temperatures , final_ov
    
def remove_outliers (  ovs , rng , config ) :
    
    whiskers = config [ 'whiskers_length' ].to_numpy()
    
    pc_25 = []
    pc_75 = []
    pc_50 = []
    
    for idx in range( 0 , len(rng) ) :
           
        pc_25.append ( prctile ( np.asarray ( ovs [ : , idx ] ,dtype='float64') , 25 ) )
        
        pc_75.append ( prctile ( np.asarray ( ovs [ : , idx ] ,dtype='float64') , 75 ) )
        
        pc_50.append ( prctile ( np.asarray ( ovs [ : , idx ] ,dtype='float64') , 50 ) )
        
    pc_25 = np.asarray (  pc_25 )
    pc_75 = np.asarray (  pc_75 )
    pc_50 = np.asarray (  pc_50 )   
   

    outliers_plus = pc_50 + whiskers*(pc_75-pc_25)
            
    outliers_minus = pc_50 - whiskers*(pc_75-pc_25)
    
    ov_final = np.zeros ( len ( rng ) ) 
    
    outlier_pass_inds = [ ]

    for r , ov_func in enumerate ( ovs ) :

        if  not (  ( any ( ovs [ r , : ]  < outliers_minus ) ) | ( any ( ovs [ r , : ] > outliers_plus ) ) ) :
            
            ov_final = np.vstack ( ( ov_final , ov_func ) )
            
            outlier_pass_inds.append ( True )
        
        else:
            
            outlier_pass_inds.append ( False )
            
    print ('after outlier removal = ' , np.shape(ov_final[1:,:]))
    
    return ov_final [ 1 : , : ] , np.nanmean ( ov_final , axis = 0 ) , outlier_pass_inds
        

def quantile ( x , q ) :
    
    n = len(x)
    
    y = np.sort(x)
    
    return ( np.interp ( q , np.linspace ( 1 / ( 2 * n ) , ( 2 * n - 1 ) / ( 2 * n ), n ), y ) )

def prctile ( x , p ) :
      
    return ( quantile ( x ,  p  / 100 ) )   
      
    