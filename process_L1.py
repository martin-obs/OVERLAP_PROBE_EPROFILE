# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
    Started December 2022

    Class to hold a bunch of celometer processing tools. This is 

    a translation / refactoring of code written by Maxime Hervo

    Rolf Ruefenacht and Melania Van Hove in Matlab

    @author martin osborne: martin.osborne@metoffice.gov.uk

'''

import numpy as np
import netCDF4 as nc
import datetime
import matplotlib.dates as mdate
import pandas as pd
from scipy import stats


import pre_checks as pc
import process_checks as proc
import sort_samples as srt


#-------------------------------------------------------------------------------

class Eprofile_Reader ( object ) :
    
    '''

    Class to read in netCDF of celometer data. Makes methods available to

    allow processing through to products

    '''

    def __init__( self , data_file  ):

        '''

        Class to hold ceilometer data for use with processing / plotting functions

        Keyword arguments:

            data_file1 = full path to a netCDF file containing L1 data on day 1

            data_file12 = full path to a netCDF file containing L1 data on day +1

        OUTPUT:

           Class containing L1 ceilometer data for use with lidar function methods

        '''

        L_nc = nc.Dataset( data_file )

        self.time = 24.0 * 60.0 * 60.0 * np.asarray ( L_nc.variables [ 'time' ] [ : ] )

        self.rng = np.asarray ( L_nc.variables [ 'range' ] [ : ] )

        self.rcs_0 = np.asarray ( L_nc.variables [ 'rcs_0' ] [ : , : ] )

        self.cbh = np.asarray ( L_nc.variables [ 'cloud_base_height' ] [ : , : ] )

        self.sci = np.asarray ( L_nc.variables [ 'sci' ] [ : ] )

        self.rng_res = np.asarray ( L_nc.variables [ 'range_resol' ] [ : ] )


    def get_constants ( self , config , ov ) :

        config_df = pd.read_csv ( config , sep = ',', skiprows = 1 , header = None )

        config_df = config_df.transpose ( )

        config_df.columns = config_df.iloc [ 0 ]

        config_df = config_df.drop ( config_df.index [ 0 ] )

        self.ov = np.asarray ( pd.read_csv ( ov , sep = '\t' , skiprows = 1 , header = None , nrows = 1 ) ) [ 0 ]

        config_df [ 'd_fit_range' ] = np.rint ( self.rng_res )

        config_df [ 'min_fit_range' ] = self.rng [ np.where ( self.ov > 0.6 ) [ 0 ] [ 0 ] ]

        config_df [ 'min_overlap_valid' ] = self.rng [ np.where ( self.ov >= 1 ) [ 0 ] [ 0 ] ]

        config_df [ 'max_fit_length' ] = config_df [ 'max_fit_range' ] - config_df [ 'min_fit_range' ]

        config_df [ 'd_fit_length' ] = config_df [ 'd_fit_range' ]

        config_df [ 'min_expected_slope' ] = -2 * 1 / np.log ( 10 ) * 10 * 1e-6

        config_df [ 'max_expected_slope' ] = -2 * 1 / np.log ( 10 ) * 0.1 * 1e-6

        config_df [ 'first_range_gradY' ] = config_df [ 'min_fit_range' ]

        config_df [ 'min_nb_good_samples' ] = np.floor ( config_df ['good_samples_proportion'] *\

                                                        config_df ['min_nb_samples'] )

        config_df [ 'min_slope' ] = -2.5*1e-4

        self.config = config_df


    def remove_some (self ) :

        li = [ *range (5, 100 ) , *range (1020,2010) ]

        li = [*range (4000,4001)]

        self.time = np.delete ( self.time , li , 0 )

        self.rcs_0 = np.delete (self.rcs_0 , li , 0 )

        self.cbh = np.delete (self.cbh , li , 0 )

        self.sci = np.delete ( self.sci , li , 0 )

    def create_time ( self ):

        '''

        The time stamp that comes with the data is in days ( now seconds )

        since epoch and it is convenient to convert this into datetime

        objects and  matplotlib.date objects ( for plotting at some point )

        '''

        dt = []

        Time = []

        for i in range ( len ( self.time ) ) :

            dt.append ( datetime.datetime.utcfromtimestamp ( self.time [ i ] ) )

            Time.append ( mdate.date2num ( dt [ i ] ) )

        self.Time = np.asarray ( Time )

        self.dt = np.asarray ( dt )


    def fill_gaps ( self ) :

        '''

        fill in any missing profiles within the start and end time of data_file

        with NaNs

        '''

        tdelta = np.ediff1d ( self.time )

        mode_delta = stats.mode ( tdelta ) [ 0 ]

        gaps  = np.rint ( ( tdelta  / mode_delta ) )

        if len (np.where ( gaps > 1 )[0]) > 0 :

            ind_list =  np.cumsum ( gaps , dtype = int )

            ind_list = np.insert ( ind_list , 0 , 0 )

            self.time = self.__make_fill_times ( self.time , gaps , mode_delta , ind_list )

            rcs_0 = self.__find_and_fill( self.rcs_0 , gaps , ind_list)

            cbh = self.__find_and_fill ( self.cbh , gaps , ind_list )

            cbh [ self.missing_flag ] = -999

            self.cbh = cbh

            self.sci = self.__find_and_fill ( self.sci , gaps , ind_list )

            rcs_0 [ self.missing_flag , : ] = np.nan

            self.rcs_0 = rcs_0

        else:

            missing_flag = np.zeros ( np.shape ( self.rcs_0 ) [ 0 ] )

            missing_flag [ : ] = False

            self.missing_flag = missing_flag.astype ( bool )

    def __make_fill_times ( self , signal , gaps , mode_delta , ind_list ) :

        '''

        having found gaps, make time stamps to fill and so make a 

        continuous time array

        '''

        gap_inds = np.where ( gaps > 1 )

        gap_len = gaps [ gap_inds ] - 1

        sts = np.insert (signal , 0 , 0 ) [ gap_inds ]

        ends = sts + gap_len * mode_delta

        t_fill_array = np.concatenate ( [ np.arange ( s , e , mode_delta ) for s , e in zip ( sts , ends ) ] , axis = None )

        filled_signal = np.zeros ( int ( sum ( gaps ) ) + 1 )

        filled_signal [ ind_list ] = signal

        missing_flag = np.zeros ( int ( sum ( gaps ) ) + 1 )

        missing_flag [ : ] = True 

        missing_flag [ ind_list ] = False

        self.missing_flag = missing_flag.astype ( bool )

        fill_inds = np.where ( filled_signal == 0 )

        filled_signal [ fill_inds ] = t_fill_array

        return filled_signal


    def __find_and_fill ( self , signal , gaps , ind_list ) :

        '''

        This does the actuall filling and is called with in "fill_missing_profiles" 

        Inputs:

            signal - 2d array of profiles

            gaps - list of sizes for gaps

            ind_list - list of indices where gaps are to be inserted

        Returns:

            filled_signal - 2d array of signal profiles with blank profiles 

            inserted where data is missing 

        '''

        if signal.ndim == 1 :

            filled_signal = np.zeros ( int ( sum ( gaps ) ) + 1  )

            filled_signal [ ind_list ] = signal

        elif signal.ndim == 2 :

            filled_signal = np.zeros( ( int ( sum ( gaps ) ) + 1 ,  np.shape ( signal ) [ 1 ] ) ) 

            filled_signal [ ind_list , : ] = signal

        return filled_signal


    def catch_errors ( self , checks , max_fit_ranges , dt , config , variance , X , Y  ) :

        if ~checks [ 0 ] :

            return 'contains no data' 

        if ~checks [ 1 ] :

            return  'at least one sci~=0'

        if ~checks [ 2 ] :

            return  'lowest cloud base is ' + str ( round ( max_fit_ranges[0] , 1 ) ) , 'm should be ' + str ( self.config_df [ 'min_fit_range' ].values [ 0 ] ) + 'm'

        if ~checks [ 3 ] :

            return 'failed variance check ' + str ( round ( variance , 3 ) ) +  ' at ' , str ( round ( max_fit_ranges[1] , 1 ) ) + 'm'

        if ~checks [ 4 ] :

            return 'failed grad check: X = ' + str ( round ( X , 3 ) ) + ' Y = ' + str ( round ( Y, 3 ) ) + ' at ' + str ( round( max_fit_ranges [ 2 ] , 1 ) ) + 'm'
        
    def write_result_to_csv ( self ) : 
    
        times_df = pd.DataFrame(self.intervals, columns = ['start' , 'end'])
        
        ov_columns = [ str ( r ) for r in self.rng ]
        
        self.ov_df = pd.DataFrame ( self.final_ovs , columns = ov_columns )
        
        rng_df = pd.DataFrame ( self.rng_intervals , columns = ['rng lower','rng upper'])
        
        results_df = pd.concat ( [ times_df , rng_df , self.ov_df ] , axis = 1)
               
        results_df.to_csv('results.csv',index=False)
        
        print (np.shape(results_df))

          
    def get_final_overlapfunction (self) :
        
        self.intervals , self.rng_intervals , self.final_ovs , self.final_ov = srt.remove_failed ( self.results , self.passed_inds , self.rng , self.config)
        
        self.write_result_to_csv()
      

    def loop_over_time ( self , start = None , stop = None ) :
        
        results = {}

        if start == None or stop == None:

            start = 0

            stop = -1

        dt = self.dt [ start : stop ] 
        
        time_interval_length = self.config [ 'time_interval_length' ].values [ 0 ]

        d_fit_time = int ( self.config [ 'd_fit_time' ] )

        d_fit_time_str = str ( d_fit_time ) + 'min'

        five_time = pd.date_range ( start = dt [ 0 ] , end = dt [ -1 ] , freq = d_fit_time_str )

        thirty_time = five_time + datetime.timedelta ( minutes = time_interval_length)

        start_inds = [ np.where ( np.asarray ( dt ) <= f ) [ 0 ] [-1] for f in five_time ]

        end_inds = [ np.where ( np.asarray ( dt ) <= f ) [ 0 ] [ -1 ]   for f in thirty_time ]
        
        overlap_functions = np.empty_like (self.ov)
        
        overlap_functions [ : ] = 0

        for s , f in zip ( start_inds , end_inds ) :
            
            print (s , f)

            if dt [ s ] <= ( dt [ -1 ] -  datetime.timedelta ( minutes = time_interval_length ) ) :
                
                int_str = 'Interval' + str( dt [ s ].time ( ) ) + 'to' + str ( dt [ f ].time ( ) ) + ' ' + str ( dt [ s ].date ( ) )
                
                results [ int_str ] = {}

                check1 = pc.at_least_one_profile ( self.missing_flag [ s : f ] )

                check2 = pc.all_clear_sky ( check1 ,  self.sci [ s : f ] ) 

                check3 , max_available_fit_range1 = pc.enough_clear_range_cbi ( check2 , self.cbh [ s : f , : ] , self.config )

                check4 , max_available_fit_range2 , variance = pc.running_variance ( check3 , self.rcs_0 [ s : f , : ] , self.rng , dt [ s : f ] , self.config , max_available_fit_range1 )

                check5 , max_available_fit_range , X , Y  = pc.check_grads ( check4 ,  self.rcs_0 [ s : f , : ]  , self.rng , self.config , max_available_fit_range2 )

                max_fit_ranges = [ max_available_fit_range1 , max_available_fit_range2 , max_available_fit_range ]

                if check1 and check2 and check3 and check4 and check5:

                    results [ int_str ] [ 'pre-check results' ] = 'passed pre-checks. Max range is = ' + str ( round ( max_available_fit_range , 1  ) )  + 'm' 

                else:

                    checks = [ check1 , check2 , check3 , check4 , check5 ]
                    
                    results [ int_str ] [ 'pre-check results' ] = self.catch_errors ( checks , max_fit_ranges , dt [ s : f ] , self.config , variance , X , Y  )

                poly_results , candidates , post_fit_results , corrected_ov = proc.check_polyfit ( self.rcs_0 [ s : f , : ] , self.rng , max_available_fit_range , self.config , self.ov )
                
                results [ int_str ] [ 'poly results' ] = poly_results
                
                results [ int_str ] [ 'candidates' ] = candidates
                
                results [ int_str ] [ 'post_fit_results' ] = post_fit_results
                
                results [ int_str ] [ 'corrected_ov' ] = corrected_ov
                      
        passed_inds = srt.check_ov_fcs_in_time_ranges ( results , self.dt , self.rng , self.rcs_0 , self.ov , self.config )
        
        self.passed_inds = passed_inds
        
        self.results = results
        


        




