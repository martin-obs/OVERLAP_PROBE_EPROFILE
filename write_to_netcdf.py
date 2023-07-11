#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Started May 2023

functions to write results of temperature model to netcdf   This is a 

translation / refactoring of Matlab code written by Maxime Hervo, 

Rolf Ruefenacht and Melania Van Hove.

@author martin osborne: martin.osborne@metoffice.gov.uk

'''
import os
import netCDF4 as nc
import datetime

def write_temp_model_to_netcdf ( path_for_results , obj ) :
    
    site = obj.site_location.split ( ',') [ 0 ]
    
    opt_mod =  obj.opt_mod_number
    
    rng_res = str ( obj.rng_res [ 0 ] )
    
    nc_name = 'Overlap_correction_model_' + site + '_' + opt_mod + '.nc'
       
    if not os.path.exists( path_for_results ) :
        
        os.mkdir ( path_for_results )
    
    elif os.path.isfile ( path_for_results + nc_name ) :  
        
        os.remove ( path_for_results + nc_name )
                  
    ncfile = nc.Dataset ( path_for_results + nc_name , mode='w' , format = 'NETCDF4_CLASSIC' ) 
   
    ncfile.site_location = obj.site_location

    ncfile.insturment_id = obj.instrument_id
    
    ncfile.wigos_station_id = obj.wigos_station_id 

    ncfile.serlom = obj.opt_mod_number
    
    ncfile.available_time_range_when_model_created = 'From ' + obj.all_available_dates [ 0 ] + ' to ' + obj.all_available_dates [ -1 ]
    
    ncfile.Days_selected_for_model_creation = 'From ' + datetime.datetime.strftime ( obj.available_dts [ 0 ] , '%Y-%m-%d' ) + ' to ' + datetime.datetime.strftime ( obj.available_dts [ -1 ] , '%Y-%m-%d'  )
    
    ncfile.number_of_days_selected = len ( obj.relative_difference )
    
    ncfile.range_resolution = rng_res + 'm'
    
    ncfile.date_of_model_creation = datetime.datetime.strftime ( datetime.datetime.now ( ) , '%Y-%m-%d' )
    
    ncfile.method = 'Based on Hervo et al. 2016'
    
    ncfile.description = 'Output for overlap artefact correction. Use: Dif (z) = a (z)* T + b (z) and Overlap_corrected (z) = 1. / (Dif (z) / 100 / overlap_ref (z) + 1. / overlap_ref (z) )'

    ncfile.createDimension('range', len ( obj.rng ) )     
    
    rng = ncfile.createVariable('range', 'f4', ('range', ) )
    
    rng.standard_name = 'range'
    
    rng.long_name = 'Altitude above ground (m)' 
    
    rng.units = 'm'
    
    rng [ : ] = obj.rng
    
    a = ncfile.createVariable('a', 'f4', ('range', ) )
    
    a.standard_name = 'a'
    
    a.long_name = 'Results of fit (difference = a *Temperature +b )'
    
    a [ : ] = obj.alpha_2
    
    b = ncfile.createVariable('b', 'f4', ('range', ) )
    
    b.standard_name = 'b'
    
    b.long_name = 'Results of fit (difference = a *Temperature +b )'
    
    b [ : ] = obj.beta_2

    overlap_ref = ncfile.createVariable('overlap_ref', 'f4', ('range', ) )
    
    overlap_ref.standard_name = 'overlap_ref'
    
    overlap_ref.long_name = 'Reference overlap function'
    
    overlap_ref [ : ] = obj.ref_ov

    ncfile.close ( )
