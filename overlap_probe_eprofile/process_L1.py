# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""Class to hold processing tools for CHM15k ceilometer data. The Class methods 
   are used to read in a single CHM15k data file, prepare it for processing
   and then loop over defined time windows applying various checks contained in
   overlap_probe_eprofile.pre_checks, overlap_probe_eprofile.process_checks, and
   overlap_probe_eprofile.sort_samples
   
   A reference overalp function and a configuration file are necessary to start 
   the processing. This module was developed using a specific overlap function, 
   that is "TUB120011_20121112_1024", for which empirical thresholds and settings
   are included in the config.txt file. Using a different reference overlap would 
   probably require different values for these thresholds.  
   
   This is a refactoring of code written by Maxime Hervo, Yann Poltera, Rolf Ruefenacht and 
   Melania Van Hove in Matlab
   
   @author martin osborne: martin.osborne@metoffice.gov.uk
   
   
   See also
   --------
   overlap_probe_eprofile.pre_checks
   overlap_probe_eprofile.process_checks
   overlap_probe_eprofile.sort_samples
   
"""

import numpy as np
import netCDF4 as nc
import datetime
import pandas as pd
from scipy import stats
from scipy.interpolate import interpn
import os
import traceback
import pyfiglet
from termcolor import colored

import overlap_probe_eprofile.find_fitting_windows as ffw
import overlap_probe_eprofile.find_candidate_functions as fcf
import overlap_probe_eprofile.final_selection as fs


class Eprofile_Reader ( object ) :
    """Class to read CHM15k data file and hold the class methods needed to 
    produce corrected overlap functions. Data file should be a standard format
    L1 E-PROFILE file.
    
    Parameters
    ----------
    
    data_file : string
        full path to an L1 file of CHM15k data.
  
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
        
        """Reads in and sorts settings / threshold values. Further threshold
        values are computed from a combination of these values and data_file 
        dimentions. The default overlap function is also read in from a text file.
        Currently saved as a Pandas dataframe, this will be changed
        to a named tuple in future for easier access.
        
        Parameters
        ----------
        
        config : string 
            path to text file containing settings and thresholds
        ov : sting
            path to text file containing a default overlap function. The
            resolution can be different to the data contained in data_file, but 
            the data will be interpolated into a native resolution. This module 
            was developed using a specific overlap function, that is 
            "TUB120011_20121112_1024", for which empirical thresholds and settings
            are included in the config.txt file. Using a different reference 
            overlap would probably need different values for these thresholds. 
        """

        config_df = pd.read_csv ( config , sep = ',', skiprows = 1 , header = None )

        config_df = config_df.transpose ( )

        config_df.columns = config_df.iloc [ 0 ]

        config_df = config_df.drop ( config_df.index [ 0 ] )

        self.ov = np.asarray ( pd.read_csv ( ov , sep = '\t' , skiprows = 1 , header = None , nrows = 1 ) ) [ 0 ]
        
        self.check_ov_range_res_same_as_data ( )

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
        



    def check_ov_range_res_same_as_data ( self ) :
        """Checks that the data range resolution is the same as the reference overlap 
        function . If not the data is interpolated onto the grid of the reference 
        overlap function. This has the effect of smothing the data slightly if it 
        was originally on a finer grid. This matches the behaviour of the original 
        Matlab code.
        """
        
        if len ( self.ov ) != len ( self.rng ) :
                   
            ov_native_rng = np.arange ( 0, 15344, 14.984999 )
            
            t_mesh, r_mesh = np.meshgrid (  self.raw_time , ov_native_rng )
            
            rcs_0 = interpn ( ( np.asarray ( self.raw_time ), np.asarray ( self.rng ) ) , self.rcs_0 , ( t_mesh , r_mesh ), bounds_error=False , fill_value = 0 ,  method = 'linear' )
            
            self.rcs_0 = np.transpose(rcs_0)
      
            self.rng = ov_native_rng 

    def remove_some ( self ) :
        
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
        the data is in fractional days since epoch and it is convenient to work
        into seconds since epoch (UNIX time) and then convert this into datetime 
        objects
        """
       
        self.dt_raw = [ datetime.datetime ( 1970 , 1 , 1 ) + datetime.timedelta ( t ) for t in self.raw_time ]
        
        self.dt = np.asarray ( [ pd.to_datetime(t).round('1s') for t in self.dt_raw ]  )
        
        self.time = np.asarray ( [ t.timestamp() for t in self.dt ] ) 
        

    def fill_gaps ( self ) :

        """Fills in any missing profiles within the start and end time of data_file
        with nans by calling make_fill_times and find_and_fill. Any time steps in
        the data_file time array that are bigger than 1.5 x the mode time step are 
        assumed to be gaps and are filled with nans
        
        See also
        --------
        
        overlap_probe_eprofile.process_L1.Eprofile_Reader.make_fill_times
        
        overlap_probe_eprofile.process_L1.Eprofile_Reader.find_and_fill
        
        """

        tdelta = np.ediff1d ( self.time )

        mode_delta = stats.mode ( tdelta ) [ 0 ]

        gaps  = np.rint ( ( tdelta  / mode_delta ) )

        if len (np.where ( gaps > 1 )[0]) > 0 :

            ind_list =  np.cumsum ( gaps , dtype = int )

            ind_list = np.insert ( ind_list , 0 , 0 )

            self.time = self.make_fill_times ( self.time , gaps , mode_delta , ind_list )

            rcs_0 = self.find_and_fill( self.rcs_0 , gaps , ind_list)

            cbh = self.find_and_fill ( self.cbh , gaps , ind_list )

            cbh [ self.missing_flag ] = -999

            self.cbh = cbh

            self.sci = self.find_and_fill ( self.sci , gaps , ind_list )

            rcs_0 [ self.missing_flag , : ] = np.nan

            self.rcs_0 = rcs_0

        else:

            missing_flag = np.zeros ( np.shape ( self.rcs_0 ) [ 0 ] )

            missing_flag [ : ] = False

            self.missing_flag = missing_flag.astype ( bool )
            

    def make_fill_times ( self , signal , gaps , mode_delta , ind_list ) :

        """Having found any gaps in the data, this function is called by 
        fill_gaps to make a continuous time variable with datetimes inserted 
        where data was missing
        
        Parameters
        ----------
        
        signal : 2D array
            data with gaps to be filled
        gaps : list 
            sizes of any gaps in data 
        mode_delta : float
            mode lenght of time step in data 
        ind_list : list
            indices where nans are to be inserted 
            
        Returns
        -------
        
        filled_time_signal : 2D array
            datetime array with appropriate datetimes inserted where data is missing 
            
        """
        
        

        gap_inds = np.where ( gaps > 1 )

        gap_len = gaps [ gap_inds ] - 1

        sts = np.insert (signal , 0 , 0 ) [ gap_inds ]

        ends = sts + gap_len * mode_delta

        t_fill_array = np.concatenate ( [ np.arange ( s , e , mode_delta ) for s , e in zip ( sts , ends ) ] , axis = None )

        filled_time_signal = np.zeros ( int ( sum ( gaps ) ) + 1 )

        filled_time_signal [ ind_list ] = signal

        missing_flag = np.zeros ( int ( sum ( gaps ) ) + 1 )

        missing_flag [ : ] = True 

        missing_flag [ ind_list ] = False

        self.missing_flag = missing_flag.astype ( bool )

        fill_inds = np.where ( filled_time_signal == 0 )

        filled_time_signal [ fill_inds ] = t_fill_array

        return filled_time_signal


    def find_and_fill ( self , signal , gaps , ind_list ) :

        """This does the actuall filling and is called within 
        fill_gaps

        Parameters
        ----------
        
        signal : array  
            data with gaps to be filled
        gaps : list 
            sizes of any gaps in data 
        ind_list : list
            indices where nans are to be inserted

        Returns
        -------        

        filled_signal : array
            data with columns of nans inserted where data is missing 
            
        See also
        --------
        overlap_probe_eprofile.process_L1.Eprofile_Reader.fill_gaps
        

        """

        if signal.ndim == 1 :

            filled_signal = np.zeros ( int ( sum ( gaps ) ) + 1  )

            filled_signal [ ind_list ] = signal

        elif signal.ndim == 2 :

            filled_signal = np.zeros( ( int ( sum ( gaps ) ) + 1 ,  np.shape ( signal ) [ 1 ] ) ) 

            filled_signal [ ind_list , : ] = signal

        return filled_signal


    def catch_errors ( self , checks , max_fit_ranges , dt , config , variance , X , Y  ) :
        """Sorts though the results of the pre-checks module and returns the 
        reason for failiure as a string.
        
        Parameters
        ----------
        checks : list of bools 
            resuts of pre-checks
        max_fit_ranges : list 
            list of max fitting range after each pre-check
        dt : list of datetime objects
            datetimes for current time window
        config : class object 
            object containing varios config thresholds
        variance : float 
            variance at max fitting range after variance check
        X : float 
            signal gradient in X direction at max fitting range after grad check
        Y : float 
            signal gradient in Y direction at max fitting range after grad check
            
        Returns
        -------
        
        reason for failiure with details : string


        See also
        --------
        overlap_probe_eprofile.pre_checks module
            Returns True of False plus some details for various pre-checks.

        """
        

        if ~checks [ 0 ] :

            return 'contains no data' 

        if ~checks [ 1 ] :

            return  'at least one sci~=0'

        if ~checks [ 2 ] :

            return  'lowest cloud base is ' + str ( round ( max_fit_ranges[0] , 1 ) ) + 'm should be ' + str ( self.config [ 'min_fit_range' ].values [ 0 ] ) + 'm'

        if ~checks [ 3 ] :

            return 'failed variance check ' + str ( round ( variance , 3 ) ) +  ' at ' + str ( round ( max_fit_ranges[1] , 1 ) ) + 'm'

        if ~checks [ 4 ] :

            return 'failed grad check: X = ' + str ( round ( X , 3 ) ) + ' at ' + str ( round( max_fit_ranges [ 3 ] , 1 ) )  + ' Y = ' + str ( round ( Y, 3 ) ) + ' at ' + str ( round( max_fit_ranges [ 4 ] , 1 ) ) + ' finaly ' + str ( round( max_fit_ranges [ 2 ] , 1 ) ) + 'm'
        
    def write_result_to_csv ( self , save_path ) : 
        
        '''
        
        Create a results data frame and save to file
        :param save_path: path to location results csv will be save in
        :type save_path: string
        
        '''
    
        times_df = pd.DataFrame ( self.intervals , columns = [ 'start' , 'end' ] )
        
        ov_columns = [ str ( r ) for r in self.rng ]
        
        self.ov_df = pd.DataFrame ( self.final_ovs , columns = ov_columns )
        
        rng_df = pd.DataFrame ( self.rng_intervals , columns = [ 'rng lower','rng upper' ] )
        
        temps = pd.DataFrame ( self.temperatures , columns = [ 'internal_temperature' ] )
        
        temps [ 'range_resolution' ] = self.rng_res
        
        results_df = pd.concat ( [ times_df , rng_df , temps , self.ov_df  ] , axis = 1 )
              
        location =  self.site_location.split ( ',' ) [ 0 ]
        
        complete_path = '/'.join ( ( save_path , location ) )
        
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
                
        """Loop through data with a window 'time_interval_length' wide and 
        shifting by 'd_fit_time' each loop, as defined in config.txt. Because
        of issues comparing datetimes, time deltas and timestamps created from 
        fractional days since midnight, and also because the time stamp in the 
        L1 file wanders ( e.g. upto a second either side of 15s ) sometimes the 
        window length ends up one profile too short or too long. Therefore the 
        mode width is used to ensure all time windows are the same width (for 
        example 120 profiles wide if 'time_interval_length' = 30 min and time resolution is 15s )
        
        Parameters
        ----------
        
        start : int
            index to start from if we are not using whole file
        end : int 
            index to stop at
        """

                                        
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
        
        end_inds = np.asarray(start_inds) + mode_diff.mode
        
        end_inds = end_inds.tolist()
        
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

                check5 , max_available_fit_range , X , Y , m1 , m2 , m3  = ffw.check_grads ( check4 ,  self.rcs_0 [ s : f , : ]  , self.rng , self.config , max_available_fit_range2 )

                max_fit_ranges = [ max_available_fit_range1 , max_available_fit_range2 , max_available_fit_range , m1 , m2 , m3 ]

                if check1 and check2 and check3 and check4 and check5:

                    results [ int_str ] [ 'pre-check results' ] = 'passed pre-checks. Max range is = ' + str ( round ( max_available_fit_range , 1  ) )  + 'm' 
                    
                    poly_results = fcf.do_quality_checks ( self.rcs_0 [ s : f , : ] , self.rng , self.internal_temperature [ s : f ] , max_available_fit_range , self.config , self.ov )
                    
                    results [ int_str ] [ 'data_frame' ] = poly_results
                    

                else:
                    
                    checks = [ check1 , check2 , check3 , check4 , check5 ]
                    
                    results [ int_str ] [ 'pre-check results' ] = self.catch_errors ( checks , max_fit_ranges , dt [ s : f ] , self.config , variance , X , Y )
                    
                    results [ int_str ] [ 'data_frame' ] = pd.DataFrame(data = [False], columns = ['pass_all'])
                    
        self.results = results
        
        self.passed_inds = fs.do_sort_checks ( results , self.dt , self.rng , self.rcs_0 , self.ov , self.config )

#Entry point#

NAME = "OVERLAP PROBE EPROFILE"

__version__ = "1.0.0"

def welcome_msg():
    """
    print a welcome message in the terminal (Try EU colours!)

    """
    ascii_art1 = pyfiglet.figlet_format ( "OVERLAP" )
    
    ascii_art2 = pyfiglet.figlet_format( "PROBE" )
    
    ascii_art3 = pyfiglet.figlet_format ( "E-PROFILE" )
    
    print(r"")
    print(colored("---------------------------------------------------", 'yellow'))
    print(NAME)
    print(r"")
    print(colored(ascii_art1, 'blue'))
    print(colored(ascii_art2, 'yellow'))
    print(colored(ascii_art3, 'blue'))
    print(r"version: " + __version__)
    print(r"MeteoSwiss , Met Office, PROBE Cost Action CA18235 ")
    print(colored("---------------------------------------------------", 'yellow'))
    print(r"")

    return None#

def daily_processing (data_file , config , ov_ref , save_path ) :
    
    try :            
        
        L1 = Eprofile_Reader ( data_file )
        
        L1.get_constants ( config , ov_ref )
        
        L1.create_time ( )
        
        L1.fill_gaps ( ) 
        
        L1.loop_over_time ( start = 0 )
        
        L1.get_final_overlapfunction ( save_path  )
    
    except Exception: 
        
        print ('passing ', data_file [-11:-3 ] )
        
        traceback.print_exc()
        
        pass    
    
    
def process_L1 ( input_data  , config , ov_ref , save_path ) :
    
    welcome_msg()
    
    if os.path.isdir(input_data) :
    
        file_list = get_list_of_data_files ( input_data ) 
        
        for data_file in file_list :
            
            print ('Working on ' , data_file [ -11 : -3 ] )
        
            daily_processing (data_file , config , ov_ref , save_path )
            
    else :
        
        print ('Working on ' , input_data [ -11 : -3 ] )
    
        daily_processing ( input_data , config , ov_ref , save_path )
        

def get_list_of_data_files (base_path) :
    
    file_list = []
    
    for root, dirs, files in os.walk(base_path, topdown=False):
        
       for name in files:
           
          file_list.append(os.path.join(root, name))

    return np.sort (  file_list )    

def L1_CHM15k_daily ():
    
    """ Processing entry point

    Example usage when installed in a virtualenv (see also setup.py):

        L1_CHM15k_daily -i <inputfilelist> -o <output_directory> -c <configurations_file -f <ref_overlap> 
    """
    
    import argparse
    parser = argparse.ArgumentParser(description='Overlap Porbe Eprofile CHM15k daily corrected overlap')
    parser.add_argument('-i', '--input_data', help='L1 eprofile CHM15k netCDF file or directory', required=True)
    parser.add_argument('-c', '--configuration_file', help='File contianing fornatted settings and thresholds', required=True)
    parser.add_argument('-f', '--reference_overlap', help='reference overlap function', required=True)
    parser.add_argument('-o', '--output_directory', help='path to output directory', required=True)

    args = parser.parse_args()
    
    process_L1 ( args.input_data , args.configuration_file , args.reference_overlap , args.output_directory )



    
    
    
