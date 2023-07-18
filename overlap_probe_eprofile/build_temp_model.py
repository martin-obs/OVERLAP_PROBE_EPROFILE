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
from collections import Counter
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import more_itertools as mit

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

        self.all_available_files = [ f for f in np.sort ( os.listdir ( path_to_csvs ) ) if os.path.isfile ( path_to_csvs + f )  ]
        
        self.dt_start = datetime.datetime.strptime ( date_start , '%Y/%m/%d' ).date ( )
        
        self.dt_stop = datetime.datetime.strptime ( date_stop , '%Y/%m/%d' ).date ( )
        
        self.path_to_csvs = path_to_csvs
        
        self.ref_ov = ref_ov
            
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
               
        self.all_available_dates = np.sort ( [ d  [ -14 :-4 ] for d in self.all_available_files ] )
               
        self.all_available_dts = np.asarray ( [ datetime.datetime.strptime ( d , '%Y-%m-%d' ).date ( ) for d in self.all_available_dates ] )
                
        start_dt_ind = (np.where ( self.all_available_dts >= self.dt_start ) [ 0 ] [ 0 ] )
               
        stop_dt_ind = (np.where ( self.all_available_dts <= self.dt_stop ) [ 0 ] [ -1 ] +1 )
        
        number_avaialble = stop_dt_ind - start_dt_ind
        
        if number_avaialble > 0 :
            
            print ( number_avaialble , ' overlap files available for chosen dates' )
            
            self.start_ind = start_dt_ind
            
            self.stop_ind = stop_dt_ind
            
            self.available_files =  self.all_available_files [ self.start_ind : self.stop_ind ] 
            
            self.available_dts = self.all_available_dts [ self.start_ind : self.stop_ind ]
        
        else:
            
            print ('no overlap files available for chosen dates, temperature model will not be made' )
            
            sys.exit()
                   

    def get_meta_data_from_first_file ( self ) :
            
            with open ( self.path_to_csvs + self.available_files [ 0 ] , 'r+' ) as f :
            
                self.meta_data = [ row for row in f ]
                
            self.opt_mod_number = str ( self.meta_data [ 0 ].split ( ' ' ) [ 2 ] ).rstrip()
          
            self.site_location = str ( self.meta_data [ 1 ].split ( ' ' ) [ 2 ] ).rstrip()
            
            self.wigos_station_id = str ( self.meta_data [ 2 ].split ( ' ' ) [ 2 ] ).rstrip()
            
            self.instrument_id = str ( self.meta_data [ 3 ].split ( ' ' ) [ 2 ] ).rstrip()
            
            self.instrument_serial_number = str ( self.meta_data [ 4 ].split ( ' ' ) [ 2 ] ).rstrip()
        
    def check_optical_module ( self ) :

        op_mods =  [ d.split ( '_' ) [ 3 ] for d in self.available_files ] 
        
        self.op_mods_list = list ( Counter ( op_mods ).keys ( ) )
        
        self.op_mod_dict = Counter ( op_mods )
        
        print ( 'containing ' ,  len ( self.op_mods_list ) , ' optical module(s) ' , self.op_mods_list ,', a temperature model will be made for each optical module')
        
        
    def check_resolution_n_get_range ( self ) :

       self.rng_res = np.asarray ( [ self._get_range_resolotion ( self.path_to_csvs + f ) for f in self.available_files ] )
       
       if np.diff ( self.rng_res ).any ( ) :
           
           self.rng_res_change_ind = np.where ( np.diff ( self.rng_res ) !=  0 )
           
           date_changes = [ d [ -14 :-4 ] for d in self.available_files [ self.rng_res_change_ind ] ]
           
           print ('Range resoloution changes on date(s) ' , *date_changes , sep = ', ' )
           
       else :
  
           print ( 'Range resolution consistent within date range')
           
           self._get_rng ( )
                 
        
    def _get_range_resolotion ( self ,  overlap_csv ) : 
        
        '''
        
        Get range resolution from overlap file - should be the same
        
        for every row hence read only first row
        
        '''
        
        return pd.read_csv ( overlap_csv , sep = ',' , header = 0 , skiprows = 5 , nrows = 1 , usecols = [ 'range_resolution' ] ).iloc [ 0 ] [ 'range_resolution' ]
    
    
    def _get_rng ( self ) :
        
        self.rng = np.asarray ( pd.read_csv ( self.path_to_csvs + self.available_files [ 0 ] , sep = ',' , header = 0  , skiprows = 5 , index_col=0 , nrows=0 ).columns.tolist() [ 5 : ] , dtype = 'float')
        
           
    def get_daily_medians ( self , use_matlab = False ) :
        
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
        
        ov = np.median ( np.asarray ( df.iloc [ : , 6: ] ) , axis = 0 )
        
        t = np.median ( np.asarray ( df.iloc [ : , 4 ] ) )
             
        return ov , t
        
    def get_relative_diff ( self ) :
              
        self.relative_difference =  ( self.ref_ov - self.daily_ovs ) / self.daily_ovs
               
    def do_regression_1 ( self ):
        
        self._make_regression_signals_1 ( )
                    
        self.alpha_1 , self.beta_1 , self.r2_1 = self._simple_linear_fit ( self.n_1 , self.A_1  , self.sum_rel_diff , axis = 0 )   
                   
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
        
        
    def _simple_linear_fit ( self , n , x , y , axis ) :
        
        '''
        
        Calculates the intercepts (alpha) and slopes (beta) and correlation 
        
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
           
        beta = ( n * Sxy - Sx * Sy ) / ( n * Sxx - Sx ** 2  )
        
        alpha =  ( 1 / n ) * Sy - ( ( 1 / n ) * beta * Sx )
        
        r2 = ( ( n * Sxy - Sx * Sy  ) ** 2 )  / ( ( n * Sxx - Sx **2 ) * ( n * Syy - Sy **2  ) )
              
        return  alpha , beta , r2
    
    def choose_n_check_r2_diff_window ( self ) :
        
        self.diff_r2 = np.diff ( self.r2_1 )
        
        print ( 'len diff_r2 = ' , len ( self.diff_r2 ) )

        self.bool_run_len = list ( mit.run_length.encode ( self.diff_r2  < self.config ['thrsh_diff_r2'].values [ 0 ] ) )

        max_true_count = -1
        
        max_true_idx  = -1
        
        for idx , ( val , count ) in enumerate ( self.bool_run_len ) :
            
            if val and max_true_count < count:
                
                max_true_count = count
                
                max_true_idx = idx
  
        self.max_true_count = max_true_count
        
        if self.max_true_count >= self.config ['number_samples'].values [ 0 ] :
            
            self.number_samples_flag = True
            
            self.end_ind = int ( sum ( np.asarray ( self.bool_run_len ) [ : max_true_idx , 1 ] ) + max_true_count ) - 1 
            
        self._if_last_diff_negative_step_backwards ( )
        
        #self.end_ind = -2
        
        
    def _if_last_diff_negative_step_backwards ( self ):
        
        idx = self.end_ind
        
        while ( self.diff_r2 [ idx ] < 0 ) and ( idx >= 0 ) :
            
            idx = idx - 1
            
        self.end_ind = idx  
                
    def do_regression_2 ( self ) :
        
        self._make_regresions_signals_2 ( )
        
        self.beta_2 , self.alpha_2 , self.r2_2 = self._simple_linear_fit ( self.n_2 , self.A_2 , self.B_2 , axis = 0 )
 
    def _make_regresions_signals_2 ( self ) :
         
         self.B_2 = self.relative_difference  [ : self.end_ind  , : ] * 100
         
         self.A_2 = np.repeat (  ( self.daily_temp-273.15 )  [  : ,  np.newaxis  ] , len ( self.rng ) , axis = 1 ) [   : self.end_ind   , : ]
         
         self.n_2 = np.shape ( self.A_2 ) [ 0 ]
    
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
        ax.scatter( self.alpha_2 , self.rng )       
        ax.grid()
        ax.tick_params(direction="in",which="both")
        ax.set_xlabel('alpha')
        #ax.set_ylabel('R$^2$ values')
        ax.set_ylim ( [ 0 , 600 ])
        ax.set_xlim ( [ -1 , 6 ])
        fig.savefig('test_alpha.png',format='png', dpi=300)


        
        
        
        
        

        
  