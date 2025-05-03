# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
    Started March 2023

    Class to hold a information needed to calculate a temperature model
    
    for overlap correction. This is a translation / refactoring of code 
    
    written by Maxime Hervo, Rolf Ruefenacht and Melania Van Hove in Matlab

    @author martin osborne: martin.osborne@metoffice.gov.uk

'''

import numpy as np
import datetime
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import more_itertools as mit

import overlap_probe_eprofile.overlap_utils as w2nc

#-------------------------------------------------------------------------------

class Temperature_model_builder ( object ) :
    
    '''

    Class to read in csvs of corrected overlap functions. Makes methods available to

    allow processing through to temperature model

    '''

    def __init__( self , date_start , date_stop , ref_ov , path_to_csvs , config ):

        '''
        Args:
            
            date_start (str): date to start looking for csvs of corrected functions (format %Y/%m/%d )

            date_stop (str): date to stop looking for csvs of corrected functions (format %Y/%m/%d )
            
            ref_ov (str) = full path to text file containing reference overlap function
            
            path_to_csvs (str) = full path to directory containing csvs of corrected functions
            
            config (str) = full path to config file containing thresholds and settings

        Returns:

            class object containing temperature model and associated variables

        '''

        self.all_available_files = [ f for f in sorted ( os.listdir ( path_to_csvs ) , key = lambda d : d.split ( '_' ) [ -1 ] )  if os.path.isfile ( path_to_csvs + f ) ]
        
        self.dt_start = datetime.datetime.strptime ( date_start , '%Y/%m/%d' ).date ( )
        
        self.dt_stop = datetime.datetime.strptime ( date_stop , '%Y/%m/%d' ).date ( )
        
        self.path_to_csvs = path_to_csvs
        
        self.ref_ov = ref_ov
        
        self.ov_native_rng = np.arange ( 14.985 ,  15344.64+14.985 , 14.985 )
            
        self._get_constants ( config )
        
        self.number_samples_flag = False
        
        self.end_ind = 0
              
        
    def _get_constants ( self , config ) :
        
        '''
        
        Read in a text file of settings / threshold values.  Currently 
        
        saved as a Pandas dataframe, this will be changed to a named tuple in future
        
        for easier access
        
        '''

        config_df = pd.read_csv ( config , sep = ',', skiprows = 1 , header = None )

        config_df = config_df.transpose ( )

        config_df.columns = config_df.iloc [ 0 ]

        config_df = config_df.drop ( config_df.index [ 0 ] )  
        
        self.ref_ov = np.asarray ( pd.read_csv ( self.ref_ov , sep = '\t' , skiprows = 1 , header = None , nrows = 1 ) ) [ 0 ]
        
        self.config = config_df
   
        
    def check_dates_available ( self ) :
        
        '''
        Check there are enough days with corrected overlap functions within the specified
        
        date range. 
        
        '''
        
        self.all_available_dates = [ d  [ -14 :-4 ] for d in self.all_available_files ] 
               
        self.all_available_dts = np.asarray ( [ datetime.datetime.strptime ( d , '%Y-%m-%d' ).date ( ) for d in self.all_available_dates ] )
        
        start_dt_ind = (np.where ( self.all_available_dts >= self.dt_start ) [ 0 ] [ 0 ] )
               
        stop_dt_ind = (np.where ( self.all_available_dts <= self.dt_stop ) [ 0 ] [ -1 ] + 1 )
        
        number_avaialble = stop_dt_ind - start_dt_ind
        
        if number_avaialble > 0 :
            
            self.start_ind = start_dt_ind
            
            self.stop_ind = stop_dt_ind
            
            self.available_files =  self.all_available_files [ self.start_ind : self.stop_ind ]             
            
            self.available_dts = self.all_available_dts [ self.start_ind : self.stop_ind ]
        
        else:
            
            print ('no overlap files available for chosen dates, temperature model will not be made' )
            
            sys.exit()
                           
    def get_last_optical_module ( self ) :
        
        '''
        List the number of different optical modules within selected date range and 
        
        the dates on which the change happened. Keep only files for the most 
        
        recent optical module
        
        '''
        
        op_mods = [ TUB.split ( '_' ) [ -2 ]  for TUB in self.available_files ] 
       
        changes = np.where ( np.asarray ( op_mods [ : -1 ] ) != np.asarray ( op_mods ) [ 1 : ] ) [ 0 ] + 1
        
        date_changed = [  d.strftime('%d/%m/%Y') for d in  self.available_dts [ changes ] ]
        
        if len(changes) > 1 :
            
            print ( str (len(changes)) , 'changes in optical module found within date range. Optical module changed on ')
                   
            [ print (d , ' ' ) for d in date_changed ]
            
            print ('Attempting to create temperature model for the most recent module' , op_mods [ -1 ] )
            
            print ('starting from', date_changed[-1] )
            
            print ('To make model for earlier optical module adjust requested dates accordingly' )
            
            self.available_files =  self.all_available_files [ changes [ -1 ] : ]             
            
            self.available_dts = self.all_available_dts [ changes [ -1 ]  : ]

        else :
             
            print ('One optical module found within date range. Attempting to create temperature model for module' , op_mods [ -1 ] )
                   
    def get_meta_data_from_first_file ( self ) :
            
            with open ( self.path_to_csvs + self.available_files [ 0 ] , 'r+' ) as f :
            
                self.meta_data = [ row for row in f ]
                
            self.opt_mod_number = str ( self.meta_data [ 0 ].split ( ' ' ) [ 2 ] ).rstrip()
          
            self.site_location = str ( self.meta_data [ 1 ].split ( ' ' ) [ 2 ] ).rstrip()
            
            self.wigos_station_id = str ( self.meta_data [ 2 ].split ( ' ' ) [ 2 ] ).rstrip()
            
            self.instrument_id = str ( self.meta_data [ 3 ].split ( ' ' ) [ 2 ] ).rstrip()
            
            self.instrument_serial_number = str ( self.meta_data [ 4 ].split ( ' ' ) [ 2 ] ).rstrip()

    def check_resolution_n_get_range ( self ) :

       self.rng_res = np.asarray ( [ self._get_range_resolotion ( self.path_to_csvs + f ) for f in self.available_files ] )
       
       if np.diff ( self.rng_res ).any ( ) :
           
           self.rng_res_change_ind = np.where ( np.diff ( self.rng_res ) !=  0 )

           date_changes = []
           
           for d in list ( self.rng_res_change_ind [ 0 ] ) :

               date_changes.append ( self.available_files [ d ] [ -14 :-4 ] )
           
           print ('Range resoloution changes on date(s) ' , *date_changes , sep = ', ' )
           
           print ('daily files will be interpolated to resoluton of reference overlap function')
           
           self.rng = self.ov_native_rng
 
       else :
  
           print ( 'Range resolution consistent within date range')
           
           self._get_rng ( )
           
           if len ( self.rng ) != len ( self.ov_native_rng ) :
               
               self.ref_ov = np.interp ( self.rng , self.ov_native_rng , self.ref_ov)
                       
               self.ov_native_rng = self.rng                
        
    def _get_range_resolotion ( self ,  overlap_csv ) : 
        
        '''
        
        Get range resolution from overlap file - should be the same
        
        for every row hence read only first row
        
        '''
        
        return pd.read_csv ( overlap_csv , sep = ',' , header = 0 , skiprows = 5 , nrows = 1 , usecols = [ 'range_resolution' ] ).iloc [ 0 ] [ 'range_resolution' ]
    
    
    def _get_rng ( self ) :
        
        self.rng = np.asarray ( pd.read_csv ( self.path_to_csvs + self.available_files [ 0 ] , sep = ',' , header = 0  , skiprows = 5 , index_col=0 , nrows=0 ).columns.tolist() [ 5 : ] , dtype = 'float')
        
           
    def get_daily_medians ( self , use_matlab = False ) :

        print ("Getting median functions for days with enough samples") 
        
        day_ov = np.empty_like ( self.rng )
        
        day_temp = []
        
        plt_date = []
        
        for i , f in enumerate ( self.available_files ):
            
            df = pd.read_csv ( self.path_to_csvs + f , skiprows = 5 , sep = ',' , header = 0 )
            
            d = self.available_dts [ i ]
                       
            if np.shape ( df ) [ 0 ] >= self.config [ 'min_nb_good_samples_after_outliers_removal' ].to_numpy()  :
            
                ov , t = self._create_daly_median ( df )
            
                day_ov = np.vstack ( ( day_ov , ov ) )

                day_temp.append ( t )
                
                plt_date.append ( d )
            
        self.daily_ovs = day_ov [ 1 : , : ]

        self.daily_temp = np.asarray ( day_temp ) [ : ]
        
        self.plt_dates = plt_date [ : ]
   
    def _create_daly_median ( self , df ) :

        print ( "CHECK MEDIAN OF DAILY FUNCTION" )
        
        ov = np.median ( np.asarray ( df.iloc [ : , 6: ] ) , axis = 0 )

        print ( ov [ : 15 ] ) 
        
        t = np.median ( np.asarray ( df.iloc [ : , 4 ] ) )

        print ( t )
        
        if len (ov) != len (self.ov_native_rng) :
            
            rng_this_file = np.asarray ( df.columns.tolist() [ 6 : ] , dtype = 'float')
            
            #ov = np.intep ( self.rng , rng_this_file , ov )
            
            ov = np.interp ( self.rng , rng_this_file , ov )
        return ov , t
        
    def get_relative_diff ( self ) :
        
        with np.errstate(divide='ignore'):
              
            self.relative_difference =  ( self.ref_ov - self.daily_ovs ) / self.daily_ovs
               
    def do_regression_1 ( self ):
        
        print ("DO REGRESSION 1")

        self._make_regression_signals_1 ( )
                    
        self.alpha_1 , self.beta_1 , self.r2_1 = self._simple_linear_fit ( self.n_1 , self.A_1  , self.sum_rel_diff , axis = 0 )   

        print ( self.alpha_1, self.beta_1, self.r2_1 )
                   
    def _make_regression_signals_1 ( self ) :
        
        sum_rel_diff = np.sqrt ( np.trapz ( self.relative_difference [ : , self.ref_ov >= 0.05  ] ** 2 , axis = 1 ) )
        
        sum_rel_diff = np.repeat ( sum_rel_diff [ : , np.newaxis ] , len ( sum_rel_diff ) , axis = 1 )
        
        mask = np.ones_like ( sum_rel_diff , dtype= bool)
        
        mask [ np.triu_indices_from ( mask ) ] = False
        
        self.sum_rel_diff = np.ma.masked_array ( sum_rel_diff , mask = mask)
        
        self.n_1 = np.count_nonzero ( self.sum_rel_diff , axis = 0 )

        T = self.daily_temp-273.15

        A_1 = np.repeat (  ( T )  [ : , np.newaxis ] , len ( T ) , axis = 1)
        
        self.A_1 = np.ma.masked_array ( A_1 , mask = mask )
        
    
    def choose_n_check_r2_diff_window ( self ) :
        
        #print (self.r2_1)

        self.diff_r2 = np.ma.diff ( self.r2_1 )

        self.diff_r2 [ 0 ] = 0
        
        self.bool_run_len = list ( mit.run_length.encode ( abs ( self.diff_r2 )  < self.config ['thrsh_diff_r2'].values [ 0 ] ) )

        #print (self.bool_run_len)

        max_true_count = -1
        
        max_true_idx  = -1
        
        for idx , ( val , count ) in enumerate ( self.bool_run_len ) :
            
            if val and max_true_count < count:
                
                max_true_count = count
                
                max_true_idx = idx
  
        self.max_true_count = max_true_count
               
        if self.max_true_count > self.config ['number_samples'].values [ 0 ] :
            
            self.number_samples_flag = True
            
            self.end_ind = int ( sum ( np.asarray ( self.bool_run_len ) [ : max_true_idx , 1 ] ) + max_true_count ) - 1
                        
            self._if_last_diff_negative_step_forwards ( )
            
        else :

            print ("Not enough data to trust model, model wont be created")
            
            self.number_samples_flag = False
        
    def _if_last_diff_negative_step_forwards ( self ):
        
        idx = self.end_ind
        
        while ( self.diff_r2 [ idx ] < 0 ) and ( idx <= ( len ( self.diff_r2 ) - 2 ) ) :
            
            idx = idx + 1
            
        self.end_ind = idx  
 
    def _make_regresions_signals_2 ( self ) :
         
         self.A_2 = np.repeat (  ( self.daily_temp-273.15 )  [  : ,  np.newaxis  ] , len ( self.rng ) , axis = 1 ) [ : self.end_ind   , : ]
         
         self.B_2 = self.relative_difference  [ : self.end_ind  , : ] * 100
         
         self.n_2 = np.shape ( self.A_2 ) [ 0 ]
         
         
    def _remove_abberant_regression_results ( self ) :

        print ("Checking for aberrations in regression")
                
        abberations = np.where( ( abs ( self.alpha_2 )  > 10  ) | ( abs ( self.beta_2 ) > 200 ) ) #120
        
        if any ( abberations [ 0 ] ) : 
        
            abberation_ind = np.max ( abberations )
        
            self.alpha_2 [ : abberation_ind + 1 ] = 0
            
            self.beta_2 [ : abberation_ind + 1 ] = 0

            #print ( self.alpha_2 [ : 50 ] ) 

            #print ( self.beta_2 [ : 50 ] )

        
    def _check_for_artefacts ( self ) :
            
        if not any ( self.alpha_2 [ ( self.rng >= 160 ) * ( self.rng <= 700 ) ] == 0 ):
        
            self.artefact = False
            
        else :
            
            print ( 'Warning: abberant coefficients, applying daily progressive quality control' ) #mvh
                        
    def do_regression_2 ( self ) :
            
        self.artefact = True 
        
        if self.number_samples_flag :
        
            self._make_regresions_signals_2 ( )
            
            self.alpha_2 , self.beta_2 , self.r2_2 = self._simple_linear_fit ( self.n_2 , self.A_2 , self.B_2 , axis = 0 )
            
            self._remove_abberant_regression_results ( )
            
            self._check_for_artefacts ( )
            
            self.end_ind = len ( self.A_2 ) 
            
        else:
            
            print ( 'Warning: not enough data, artifact progressive detection' )  
        
        while self.artefact :
            
            self.end_ind = self.end_ind - 1

            self._make_regresions_signals_2 ( )
            
            self.alpha_2 , self.beta_2 , self.r2_2 = self._simple_linear_fit ( self.n_2 , self.A_2 , self.B_2 , axis = 0 )
            
            self._remove_abberant_regression_results ( )
            
            self._check_for_artefacts ( )
            
        print ('Success!')    
        
    def do_final_checks ( self ) :
        
        if self.alpha_2.mask.any ( ) :
            
            print ('MASKED VALUES')
            
            err1 =  'Warning - there are masked values in regression 2 for ' +  self.site_location + ' ' + \
            self.wigos_station_id + ' ' + self.instrument_id + ' ' + self.opt_mod_number 
            
            err2 =  '. Temperature model will not be created'
            
            sys.exit ( err1 + err2 )
            
        if np.isnan(self.alpha_2).any():
            
            print ('Warning - there are NaNs in regression 2 for ', self.site_location , self.wigos_station_id , self.instrument_id, self.opt_mod_number )
                      
        if len ( self.relative_difference ) <= 15 :

            err3 = 'Warning - len relative_differece is only ' + str ( len ( self.relative_difference) ) + '. Temperature model will not be created'
            
            sys.exit ( err3 ) 
        
    
    def plot_regression_1 ( self ) :
        
        params = {'legend.fontsize': 8,
    			  'axes.titlepad':10,
    			  'figure.figsize': (15, 5),
    			  'axes.labelsize': 10,
    			  'axes.titlesize':10,
    			  'axes.linewidth':2,
    			  'xtick.labelsize':10,
    			  'ytick.labelsize':10,
    			  'ytick.major.size': 5,
    			  'xtick.major.size': 5,
    			  'xtick.minor.size': 3}
        plt.rcParams.update(params)
        fig = plt.figure(num=None, facecolor='w', edgecolor='k')
        fig.set_size_inches(7,4)
        ax = plt.subplot(111)
        ax.plot(self.plt_dates ,  self.r2_1 )
        date_format = DateFormatter('%d/%m')
        ax.xaxis.set_major_formatter(date_format) 
        ax.grid()
        ax.tick_params(direction="in",which="both")
        ax.set_xlabel('Date')
        ax.set_ylabel('R$^2$ values')
        fig.savefig('test.png',format='png', dpi=300)      
               
    def plot_regression_2 ( self ) :
        
        params = {'legend.fontsize': 8,
    			  'axes.titlepad':10,
    			  'figure.figsize': (15, 5),
    			  'axes.labelsize': 10,
    			  'axes.titlesize':10,
    			  'axes.linewidth':2,
    			  'xtick.labelsize':10,
    			  'ytick.labelsize':10,
    			  'ytick.major.size': 5,
    			  'xtick.major.size': 5,
    			  'xtick.minor.size': 3}
        plt.rcParams.update(params)
        fig = plt.figure(num=None, facecolor='w', edgecolor='k')
        fig.set_size_inches(7,4)
        ax = plt.subplot(111)
        ax.plot( self.alpha_2 , self.rng , '-o')       
        ax.grid()
        ax.tick_params(direction="in",which="both")
        ax.set_xlabel('alpha')
        ax.set_ylabel('Range [m]')
        ax.set_ylim ( [ 0 , 700 ])
        ax.set_xlim ( [ -4 , 4 ])
        fig.savefig('test_alpha.png',format='png', dpi=300)


        
    def _simple_linear_fit ( self , n , x , y , axis ) :
        
        '''
        
        Calculates the slope (alpha) and intercept (beta) and correlation 
        
        coefficient ( r2 ) of a simple linear fit to the values in each column
        
        ( axis = 0 ) or row ( axis = 1 ) of array y. Works for masked arrays 
        
        unlike the various Python polyfit type functions
            
        '''
        
        x = np.ma.masked_invalid ( x ) 
        
        y = np.ma.masked_invalid ( y )
        
        Sxy = np.ma.sum ( ( x * y ) , axis = axis  )
        
        Sxx =  np.ma.sum ( ( x * x ) , axis = axis  )
        
        Syy = np.ma.sum ( ( y * y ) , axis = axis )
        
        Sx = np.ma.sum ( x , axis = axis  )
        
        Sy = np.ma.sum ( y , axis = axis  )
           
        alpha = ( n * Sxy - Sx * Sy ) / ( n * Sxx - Sx ** 2  )
        
        beta =  ( 1 / n ) * Sy - ( ( 1 / n ) * alpha * Sx )
        
        r2 = ( ( n * Sxy - Sx * Sy  ) ** 2 )  / ( ( n * Sxx - Sx **2 ) * ( n * Syy - Sy **2  ) )
              
        return  alpha , beta , r2        
    
    
#entry point#

def make_temperature_model ( start , end , ref_ov , path_to_csvs , config ,  path_for_result , plot = False , write = True ) :
    
    TM = Temperature_model_builder ( start , end , ref_ov ,  path_to_csvs  , config )
    
    TM.plot = plot
    
    TM.write = write
    
    TM.check_dates_available ( )
    
    TM.get_last_optical_module ( )
    
    TM.get_meta_data_from_first_file ( )
    
    TM.check_resolution_n_get_range ( )
    
    TM.get_daily_medians ( )
    
    TM.get_relative_diff ( )
    
    TM.do_regression_1 ( )
    
    TM.choose_n_check_r2_diff_window ( )
    
    TM.do_regression_2 ( )
    
    TM.do_final_checks ( )
    
    if TM.plot :
    
        TM.plot_regression_1 ( )
        
        TM.plot_regression_2 ( )
        
    if TM.write :
    
        w2nc.write_temp_model_to_netcdf ( path_for_result , TM )
        
def CHM15k_temperature_model (): 
    
    """ Processing entry point

    Example usage when installed in a virtualenv (see also setup.py):

        CHM15k_temperature_model -s start_date -e end_date -f reference_overlap -i path_to_csvs -c config_file 
                
                                    -o output_directory -p plot -w write_to_file
    """
    
    import argparse
    parser = argparse.ArgumentParser(description='Overlap Porbe Eprofile CHM15k daily corrected overlap')
    parser.add_argument('-s', '--start_date', help='L1 eprofile CHM15k netCDF file', required=False)
    parser.add_argument('-e', '--end_date', help='directory containing L1 eprofile CHM15k netCDF files', required=False)
    parser.add_argument('-f', '--reference_overlap', help='reference overlap function', required=True)
    parser.add_argument('-i', '--path_to_csvs', help='path_to_csvs files containing corrected overlap functions', required=True)
    parser.add_argument('-c', '--configuration_file', help='File contianing fornatted settings and thresholds', required=True)
    parser.add_argument('-o', '--output_directory', help='path to output directory', required=True)
    parser.add_argument('-p', '--make_plots', help='make plots or not', required=False, default = False)
    parser.add_argument('-w', '--write_to_file', help='path to output directory', required=False, default = True)

    args = parser.parse_args()
    
    make_temperature_model ( args.start_date ,
                            args.end_date ,
                            args.reference_overlap ,
                            args.path_to_csvs ,
                            args.configuration_file ,
                            args.output_directory ,
                            args.make_plots,
                            args.write_to_file ) 

        
  
