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
import pytz
import pandas as pd
from scipy import stats
import os


import overlap_probe_eprofile.find_fitting_windows as ffw
import overlap_probe_eprofile.find_candidate_functions as fcf
import overlap_probe_eprofile.final_selection as fs


#-------------------------------------------------------------------------------

class Eprofile_Reader ( object ) :
    """Class to read in netCDF of celometer data. Makes methods available
    to allow processing through to products

    :param data_file: full path to a netCDF file containing an L1 file of
    CHM15k data
    :type data_file:  str    
    """
    def __init__( self , data_file  ):
        """Constructor method
        """
        
        L_nc = nc.Dataset( data_file )

        self.raw_time =  np.asarray ( L_nc.variables [ 'time' ] [ : ] ) 

        self.rng = np.asarray ( L_nc.variables [ 'range' ] [ : ] , dtype = 'float64')

        self.rcs_0 = np.asarray ( L_nc.variables [ 'rcs_0' ] [ : , : ] )

        self.cbh = np.asarray ( L_nc.variables [ 'cloud_base_height' ] [ : , : ] )

        self.sci = np.asarray ( L_nc.variables [ 'sci' ] [ : ] )

        self.rng_res = np.asarray ( L_nc.variables [ 'range_resol' ] [ : ] )
        
        self.internal_temperature = np.asarray( L_nc.variables [ 'temp_int' ] [ : ] )
        
        self.opt_mod_number = getattr ( L_nc , 'optical_module_id' )
        
        self.site_location = getattr ( L_nc , 'site_location' )
        
        self.wigos_station_id = getattr ( L_nc , 'wigos_station_id' )
        
        self.instrument_id = getattr ( L_nc , 'instrument_id' ) 

        self.instrument_serial_number = getattr ( L_nc , 'instrument_serial_number' ) 
        
    def get_constants ( self , config , ov ) :
        
        """Reads in a text file of settings / threshold values. Those that can 
        be are computed from a combination of other settings / data file 
        dimentions. Aslo reads in a default ovelap function from a text file
        Currently saved as a Pandas dataframe, this will be changed
        to a named tuple in future for easier access
        
        :param config: path to text file containing settings and thresholds
        :type config: str
        :param ov: path to test file containing a default overlap function. The
        resolution can be different to the data contained in data_file, but 
        the function will be interpolated into a native resolution
        :type ov : str
        """

        config_df = pd.read_csv ( config , sep = ',', skiprows = 1 , header = None )

        config_df = config_df.transpose ( )

        config_df.columns = config_df.iloc [ 0 ]

        config_df = config_df.drop ( config_df.index [ 0 ] )

        self.ov = np.asarray ( pd.read_csv ( ov , sep = '\t' , skiprows = 1 , header = None , nrows = 1 ) ) [ 0 ]
        
        self._check_ov_range_res_same_as_data ( )

        config_df [ 'd_fit_range' ] = np.rint ( self.rng_res )

        config_df [ 'min_fit_range' ] = self.rng [ np.where ( self.ov > 0.6 ) [ 0 ] [ 0 ] ]

        config_df [ 'min_overlap_valid' ] = self.rng [ np.where ( self.ov >= 1 ) [ 0 ] [ 0 ] ]

        config_df [ 'max_fit_length' ] = config_df [ 'max_fit_range' ] - config_df [ 'min_fit_range' ]

        config_df [ 'd_fit_length' ] = config_df [ 'd_fit_range' ]

        config_df [ 'min_expected_slope' ] = -2 * 1 / np.log ( 10 ) * 10 * 1e-6

        config_df [ 'max_expected_slope' ] = -2 * 1 / np.log ( 10 ) * 0.1 * 1e-6

        config_df [ 'first_range_gradY' ] = config_df [ 'min_fit_range' ]

        config_df [ 'min_nb_good_samples' ] = np.floor ( config_df ['good_samples_proportion'] * config_df ['min_nb_samples'] )

        config_df [ 'min_slope' ] = -2.5*1e-4

        self.config = config_df


    def _check_ov_range_res_same_as_data (self):
        """Checks that the range resolution of the default overlap function is
        the same as the data file
        """
        
        if len ( self.ov ) != len ( self.rng ) :
            
            ov_native_rng = np.arange ( 14.985 ,  15344.64+14.985 , 14.985 )
            
            self.ov = np.interp ( self.rng ,  ov_native_rng , self.ov )

    def remove_some (self ) :
        
        """This is a utility used during code development to ensure that the code 
        works with data files with missing profiles - please ignore
        """

        li = [ *range (5, 100 ) , *range (1020,2010) ]

        li = [*range (4000,4001)]

        self.time = np.delete ( self.time , li , 0 )

        self.rcs_0 = np.delete (self.rcs_0 , li , 0 )

        self.cbh = np.delete (self.cbh , li , 0 )

        self.sci = np.delete ( self.sci , li , 0 )

    def create_time ( self ):

        """Creates more convinient time stamps. The time stamp that comes with
        the data is in fractional days since epoch and it is convenient to 
        seconds since epoch (UNIX time) and the convert this into datetime 
        objects
        """
       
        self.dt_raw = [ datetime.datetime ( 1970 , 1 , 1 ) + datetime.timedelta ( t ) for t in self.raw_time ]
        
        self.dt = np.asarray ( [ pd.to_datetime(t).round('1s') for t in self.dt_raw ]  )
        
        self.time = np.asarray ( [ t.timestamp() for t in self.dt ] ) 
        

    def fill_gaps ( self ) :

        """Fills in any missing profiles within the start and end time of data_file
        with NaNs
        """

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

        """Having found gaps, make time stamps to fill and so make a 
        continuous time        
        :param signal: data with gaps to be filled
        :type signal: array
        :param gaps: sizes of any gaps in data 
        :type gaps: list
        :param mode_delta: mode lenght of time step in data 
        :type mode_delta: float
        :param ind_list: indices where nans are to be inserted
        :type ins_list: list 
        :return: filled_signal : data with nans inserted to 
        where data is missing 
        :rtype: array
        """

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

        """This does the actuall filling and is called within 
        fill_missing_profiles 
        :param signal: data with gaps to be filled
        :type signal: array
        :param gaps: sizes of any gaps in data 
        :type gaps: list
        :param ind_list: indices where nans are to be inserted
        :type ins_list: list 
        
        :return: filled_signal : data with nans inserted to 
        where data is missing 
        :rtype: array
        """

        if signal.ndim == 1 :

            filled_signal = np.zeros ( int ( sum ( gaps ) ) + 1  )

            filled_signal [ ind_list ] = signal

        elif signal.ndim == 2 :

            filled_signal = np.zeros( ( int ( sum ( gaps ) ) + 1 ,  np.shape ( signal ) [ 1 ] ) ) 

            filled_signal [ ind_list , : ] = signal

        return filled_signal


    def catch_errors ( self , checks , max_fit_ranges , dt , config , variance , X , Y  ) :
        """Sorts though the results of the pre-checks and returns the reason for 
        any failiures
        :param signal: data with gaps to be filled
        :type signal: array
        :param gaps: sizes of any gaps in data 
        :type gaps: list
        :param ind_list: indices where nans are to be inserted
        :type ins_list: list 
        
        :return: filled_signal : data with nans inserted to 
        where data is missing 
        :rtype: array
        """
        

        if ~checks [ 0 ] :

            return 'contains no data' 

        if ~checks [ 1 ] :

            return  'at least one sci~=0'

        if ~checks [ 2 ] :

            return  'lowest cloud base is ' + str ( round ( max_fit_ranges[0] , 1 ) ) , 'm should be ' + str ( self.config [ 'min_fit_range' ].values [ 0 ] ) + 'm'

        if ~checks [ 3 ] :

            return 'failed variance check ' + str ( round ( variance , 3 ) ) +  ' at ' , str ( round ( max_fit_ranges[1] , 1 ) ) + 'm'

        if ~checks [ 4 ] :

            return 'failed grad check: X = ' + str ( round ( X , 3 ) ) + ' Y = ' + str ( round ( Y, 3 ) ) + ' at ' + str ( round( max_fit_ranges [ 2 ] , 1 ) ) + 'm'
        
    def write_result_to_csv ( self , save_path ) : 
        
        '''
        
        Create results cvs and save to file
        
        '''
    
        times_df = pd.DataFrame ( self.intervals , columns = [ 'start' , 'end' ] )
        
        ov_columns = [ str ( r ) for r in self.rng ]
        
        self.ov_df = pd.DataFrame ( self.final_ovs , columns = ov_columns )
        
        rng_df = pd.DataFrame ( self.rng_intervals , columns = [ 'rng lower','rng upper' ] )
        
        temps = pd.DataFrame ( self.temperatures , columns = [ 'internal_temperature' ] )
        
        temps [ 'range_resolution' ] = self.rng_res
        
        results_df = pd.concat ( [ times_df , rng_df , temps , self.ov_df  ] , axis = 1 )
              
        location =  self.site_location.split ( ',' ) [ 0 ]
        
        #complete_path = '/'.join ( ( save_path , location ) )
        
        complete_path = '/'.join ( ( save_path , self.wigos_station_id ) )   
        
        if not os.path.exists ( complete_path )  :
            
            os.makedirs( complete_path )
               
        results_name = 'ov_results_' + location + '_' + self.opt_mod_number + '_' + str ( self.dt [ 0 ].date ( ) ) + '.csv'
               
        meta_data = [ 'opt_mod_number = ' + self.opt_mod_number ,'site_location = ' + self.site_location ,
                     'wigos_station_id = ' + self.wigos_station_id , 'instrument_id = ' + self.instrument_id ,
                     'instrument_serial_number = ' +  self.instrument_serial_number ]
        
        path_n_name = '/'.join ( ( complete_path , results_name ) )
              
        with open (  path_n_name, 'w+' ) as f :
        
            content = f.read ( )
            
            for l in meta_data :
            
                f.write ( l.rstrip  ('\r\n' ) + '\n' + content )
        
        results_df.to_csv (  path_n_name , mode = 'a' , index = False )
          
    def get_final_overlapfunction ( self , save_path ) :
        
        if np.sum(self.passed_inds) != 0: 
        
            self.intervals , self.rng_intervals , self.final_ovs , self.temperatures , self.final_ov = fs.remove_failed ( self.results , self.passed_inds , self.rng , self.config )
            
            self.write_result_to_csv ( save_path  )
                

      

    def loop_over_time ( self , start = None , stop = None ) :
        
        '''
        
        Loop through data with a window 'time_interval_length' wide and 
        
        shifting by 'd_fit_time' each loop. Because of issues comparing 
        
        datetimes, time deltas and timestamps created from fractional days
        
        since midnight, and also because the time stamp in the L1 file 
        
        wanders ( e.g. upto a second either side of 15s ) sometimes the window 
        
        length ends up one profile too short or too long. Therefore the 
        
        mode width is used to ensure all time windows are the same width 
        
        ( for example 120 profiles wide if 'time_interval_length' = 30 min 
         
         and time resolution is 15s )
                        
        '''
                                          
        results = {}

        if start == None or stop == None :

            start = 0

            stop = -1

        dt = self.dt [ start : stop ] 
         
        time_interval_length = int ( self.config [ 'time_interval_length' ].values [ 0 ] )

        d_fit_time = int ( self.config [ 'd_fit_time' ].iloc [ 0 ] )

        d_fit_time_str = str ( d_fit_time ) + 'min'

        five_time = pd.date_range ( start = dt [ 0 ] , end = dt [ -1 ] , freq = d_fit_time_str )

        thirty_time = five_time + datetime.timedelta ( minutes = time_interval_length )

        start_inds = [ np.where ( np.asarray ( dt ) >= f ) [ 0 ] [ 0 ] for f in five_time ]

        end_inds = [ np.where ( np.asarray ( dt ) <= f ) [ 0 ] [ -1 ]   for f in thirty_time ]
        
        mode_diff = stats.mode ( np.asarray ( end_inds ) - np.asarray ( start_inds ) )
        
        end_inds = start_inds + mode_diff.mode
        
        overlap_functions = np.empty_like ( self.ov )
        
        overlap_functions [ : ] = 0

        for s , f in zip ( start_inds , end_inds ) :

            if dt [ s ] <= ( dt [ -1 ] -  datetime.timedelta ( minutes = time_interval_length ) ) :
                
                int_str = 'Interval' + datetime.datetime.strftime ( dt [ s ] , '%H:%M:%S%f' ) + '-' + datetime.datetime.strftime ( dt [ f ] , '%H:%M:%S%f' ) + ' ' +  str ( self.time [ s ] ) + ' to ' + str ( self.time [ f ] ) + ' ' + str ( dt [ s ].date ( ) )
  
                results [ int_str ] = {}

                check1 = ffw.at_least_one_profile ( self.missing_flag [ s : f ] )

                check2 = ffw.all_clear_sky ( check1 ,  self.sci [ s : f ] ) 

                check3 , max_available_fit_range1 = ffw.enough_clear_range_cbi ( check2 , self.cbh [ s : f , : ] , self.config )

                check4 , max_available_fit_range2 , variance = ffw.running_variance ( check3 , self.rcs_0 [ s : f , : ] , self.rng , dt [ s : f ] , self.config , max_available_fit_range1 )

                check5 , max_available_fit_range , X , Y  = ffw.check_grads ( check4 ,  self.rcs_0 [ s : f , : ]  , self.rng , self.config , max_available_fit_range2 )

                max_fit_ranges = [ max_available_fit_range1 , max_available_fit_range2 , max_available_fit_range ]

                if check1 and check2 and check3 and check4 and check5:

                    results [ int_str ] [ 'pre-check results' ] = 'passed pre-checks. Max range is = ' + str ( round ( max_available_fit_range , 1  ) )  + 'm' 
                    
                    poly_results = fcf.do_quality_checks_meteoswiss ( self.rcs_0 [ s : f , : ] , self.rng , self.internal_temperature [ s : f ] , max_available_fit_range , self.config , self.ov )
                    
                    results [ int_str ] [ 'data_frame' ] = poly_results
                    

                else:
                    
                    checks = [ check1 , check2 , check3 , check4 , check5 ]
                    
                    results [ int_str ] [ 'pre-check results' ] = self.catch_errors ( checks , max_fit_ranges , dt [ s : f ] , self.config , variance , X , Y  )
                    
                    results [ int_str ] [ 'data_frame' ] = pd.DataFrame(data = [False], columns = ['pass_all'])
                    
        self.results = results
       
        passed_inds = fs.do_sort_checks ( results , self.dt , self.rng , self.rcs_0 , self.ov , self.config )
        
        self.passed_inds = passed_inds
    


        




