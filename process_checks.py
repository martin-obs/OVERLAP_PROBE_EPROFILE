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
import numpy.polynomial.polynomial as pol

def check_polyfit ( rcs_0 , rng , max_available_fit_range , config , ov ) :

    candidates = { }
    
    corrected_ov = {}
    
    post_fit_results = { }

    n_tryouts = 0
    
    n_polyfit_params_passed = 0
    
    results_dict = { }
    
    signal_all = np.nanmean ( np.log10 ( abs ( rcs_0 ) ) , axis = 0 )
    
    ov_fc_ok = np.empty_like ( ov )
    
    ov_fc_ok [ : ] = 0
    
    ov_ranges = np.asarray([ 1 , 2 ])

    ov_ranges.shape = (2,)
    
    ov_ranges [ : ] = 0

    for fit_length in np.arange( config [ 'min_fit_length' ].values [ 0 ] , config [ 'max_fit_length' ].values [ 0 ] , config [ 'd_fit_length' ].values [ 0 ] ) :

        for fit_begin in np.arange ( config [ 'min_fit_range' ].values [ 0 ] , int ( max_available_fit_range ) , config [ 'd_fit_range' ].values [ 0 ] ) :

            if fit_begin + fit_length <= max_available_fit_range : 
                
                n_tryouts = n_tryouts + 1
                
                index_range = ( rng >= fit_begin ) * ( rng <= ( fit_begin + fit_length ) )
                
                window_str = str ( fit_begin ) + 'm to ' + str ( fit_begin+fit_length  ) + 'm'
               
                idx = np.nansum(index_range)
                
                index_whole_zone = ( rng >= config [ 'min_fit_range' ].values [ 0 ] ) * ( rng <= fit_begin + fit_length )
                
                mean_log_signal = np.nanmean ( np.log10 ( abs ( rcs_0 [ : , index_range ] ) ) , axis = 0 )
                
                s_hape = np.shape (rcs_0 [ : , index_range ] )
                
                whole_zone_signal = np.nanmean ( np.log10 ( abs ( rcs_0 [ : , index_whole_zone ] ) ) , axis = 0 )
                                  
                p = pol.polyfit ( rng [ index_range ] ,  mean_log_signal  , 1 )
                
                poly = pol.polyval ( rng [ index_range ] , p )
                
                poly_whole_zone = pol.polyval ( rng [ index_whole_zone ] , p )
                
                poly_all = pol.polyval (  rng , p )
                                
                resid = np.sqrt ( ( 1 / np.nansum ( idx ) ) * np.nansum ( ( mean_log_signal - poly ) **2 ) )
                
                resid_whole_zone = np.nanmax ( abs ( whole_zone_signal - poly_whole_zone ) / abs ( poly_whole_zone ) )  
                                
                diff = signal_all - poly_all
                
                overlap_corr_factor = 10 ** diff
                
                overlap_corr_factor [ rng >= fit_begin+fit_length ] = 1
                
                ovp_fc = ov * overlap_corr_factor
                
                min_overlap_valid = config [ 'min_overlap_valid' ].values [ 0 ]
                
                rel_err =  abs ( ov [ rng >= min_overlap_valid ] - ovp_fc [ rng >= min_overlap_valid ] )  / abs ( ov [ rng >= min_overlap_valid ] ) 
                
                valmax = np.nanmax ( rel_err ) 
                
                valmax_r_st = np.where ( rng >= min_overlap_valid ) [ 0 ] [ 0 ]
                
                valmax_r = valmax_r_st +  np.where ( rel_err == valmax )
                
                np.seterr(divide='ignore')
                
                index_range_for_grad = ( rng >= config [ 'min_range_std_over_mean' ].values [ 0 ] ) * ( rng <= fit_begin + fit_length )
                
                l = rng [ index_range_for_grad ] [ 0 ]
                
                h = rng [ index_range_for_grad ] [ -1 ]
                
                signal_for_grad_check =   np.log10 ( abs ( rcs_0  [ : , index_range_for_grad ] )  / overlap_corr_factor [ index_range_for_grad ] ) 
                
                shape_sig = np.shape (signal_for_grad_check)
                
                gradY = conv2 (  signal_for_grad_check , direction =  'y' )

                gradX = conv2 (  signal_for_grad_check , direction = 'x' )

                relgradmagn = np.sqrt ( gradX  ** 2 + gradY  ** 2 ) / abs ( signal_for_grad_check )
                
                relgradmagn = relgradmagn [ 1 : -1 , 1 : -1 ]
                
                slope = ss.savgol_filter ( ovp_fc [ :167 ] , window_length = int ( config [ 'sgolay_width' ].values [ 0 ] ) , polyorder = int ( config [ 'sgolay_ord' ].values [ 0 ] ) , deriv = 1 , delta = rng [ 1 ] - rng [ 0 ] )
                
                r_min_slope = rng [ :167 ]
                
                index_range_stop_correction = np.where ( r_min_slope <= ( fit_begin + fit_length ) ) [ 0 ] [ -1 ]
                
                val_min_slope = np.nanmin  ( slope )
                
                index_min_slope = np.where ( slope == val_min_slope )
                
                condition1 = ( p [ 1 ] >= config [ 'min_expected_slope' ].values [ 0 ] )
                
                condition2 = ( p [ 1 ] <= config [ 'max_expected_slope' ].values [ 0 ] )
                
                condition3 = ( p [ 0 ] >= config [ 'min_expected_zero_fit_value' ].values [ 0 ] )
                
                condition4 = ( p [ 0 ] <= config [ 'max_expected_zero_fit_value' ].values [ 0 ] )
                
                condition5 = ( resid < config [ 'thresh_resid_rel' ].values [ 0 ] * np.nanmean ( poly ) )
                
                condition6 = ( resid_whole_zone < config [ 'thresh_resid_whole_zone' ].values [ 0 ] )
                
                condition7 = ( np.nanmax ( ovp_fc ) ) <= config [ 'max_overlap_value' ].values [ 0 ]  * np.nanmax ( ov )
                
                condition8 = valmax < config [ 'thresh_overlap_valid_rel_error' ].values [ 0 ]
                
                condition9 = ( np.nanmax ( np.nanmax ( relgradmagn , axis = 1 ) ) <=  config [ 'max_relgrad' ].to_numpy ( ) )
                
                condition10 = ( np.nanmean ( np.nanmean ( relgradmagn , axis = 1 ) ) <= config [ 'max_relgrad_mean' ].to_numpy ( ) )
                
                condition11 = ( val_min_slope >= config [ 'min_slope' ].to_numpy ( ) ) 
                
                condition12 =  ( index_min_slope > index_range_stop_correction )
                
                if  all ( [ condition1 , condition2 , condition3 , condition4 ] ):
                    
                    n_polyfit_params_passed = n_polyfit_params_passed + 1     
                    
                    if condition5 :

                        if condition6 :
                            
                            line = { 'fit begin' : fit_begin , 'fit length ' : fit_length , 'residual' : resid , 'whole zone residual' : resid_whole_zone , 'p[0]' : p [ 1 ] , 'p[1]' : p [ 0 ] , 'mean polly' : np.nanmean( poly ) }

                            candidates [ window_str ] = line
                            
                            results_dict [ window_str ] =  ' passed ' + str (  p[1] / 1e-6  ) + '*1e-6 ' + str ( p[0] )
                            
                            if condition7 :
                            
                                if condition8 :
                                    
                                    if all ( [ condition9 , condition10 ] ):
                                        
                                        if condition11 or condition12:
                                            
                                            post_fit_results [ window_str ] = 'passed all ' + ' post grad magn failed: shape = ' + str ( shape_sig ) + 's:f = ' + str(l) + ' ' + str(h) + ' max =  '  +  str ( np.nanmax ( np.nanmax ( relgradmagn , axis = 1) ) )  + ' and mean = ' + str ( np.nanmean ( np.nanmean ( relgradmagn , axis = 1 ) ) ) 
                                            
                                            print ('passed all!')

                                            corrected_ov [ window_str ] = ovp_fc 
                                            
                                        else:
                                                                                      
                                            results_dict [ window_str ]  = ' slope of overlap fct failed: min=' + str (val_min_slope) + ' @r=' + str ( r_min_slope ) + 'm  should be >=' + str ( config [ 'min_slope' ].to_numpy ( ) ) 
                                            
                                            pass
                                        
                                    else: 
                                        
                                        results_dict [ window_str ]  =  ' post grad magn failed: shape = ' + str ( shape_sig ) + 's:f = ' + str(l) + ' ' + str(h) + ' max =  '  +  str ( np.nanmax ( np.nanmax ( relgradmagn , axis = 1) ) )  + ' and mean = ' + str ( np.nanmean ( np.nanmean ( relgradmagn , axis = 1 ) ) ) 
                                        
                                        pass
                                    
                                else: 
                                    
                                    results_dict [ window_str ] =  'overlap valid test failed: max rel error @r>=' + str ( min_overlap_valid ) + 'm is ' + str ( valmax ) + ' @r=' + str ( rng [ valmax_r ] )  + 'm'
    
                                    pass
                                
                            else:
 
                                results_dict [ window_str ] = 'resids = ' + str ( resid ) + 'should be < ' + str (config [ 'thresh_resid_rel' ].values [ 0 ] * np.nanmean ( poly )) + ' too high overlap fct value : ' + str ( np.nanmax ( ovp_fc ) ) + '@ ' + str ( rng [ ovp_fc == np.nanmax ( ovp_fc ) ] )
                                
                                pass
                            
                        else:

                            results_dict [ window_str ] = 'resids = ' + str ( resid ) + 'should be < ' + str (config [ 'thresh_resid_rel' ].values [ 0 ] * np.nanmean ( poly )) + ' failed whole zone residual' + str ( resid_whole_zone ) + ' should be < ' + str (config [ 'thresh_resid_whole_zone' ].values [ 0 ]) + '( vs fit zone resid = ' + str (resid ) 
                            
                            pass
                        
                    else:

                        results_dict [ window_str ] =  ' failed residuals ' + str ( resid ) + 'should be < ' + str (config [ 'thresh_resid_rel' ].values [ 0 ] * np.nanmean ( poly ))
                        
                        pass

                else:
                    
                    results_dict [ window_str ] =  ' failed p[0] = ' + str (  p[1] / 1e-6  ) + '*1e-6 , p[1] = ' + str ( p[0] ) + 'shape was: ' + str(s_hape)
                    
                    pass

    return results_dict , candidates , post_fit_results , corrected_ov

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



