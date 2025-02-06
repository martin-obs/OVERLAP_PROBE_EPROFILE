#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:22:04 2023

@author: mosborne
"""
import numpy as np

import overlap_probe_eprofile.build_temp_model as btm

import overlap_probe_eprofile.overlap_utils as w2nc

from datetime import datetime, timedelta

import os


def compute_temp_model_for_all(daily_dir, wigos, output_dir, ov_ref, config):
    """"
    Refactoring of the original script running at MeteoSwiss in order to loop over all WIGOS stations and compute overlap models.
    
    It simply avoids calling this script through bash commands and instead calls the Python functions directly.
    
    @author: eric
    
    Parameters
    ----------
    daily_dir : str
        Path to the daily functions directory.
    wigos : str
        WIGOS station code.
    output_dir : str
        Path to the output directory.
    ov_ref : str
        Path to the reference overlap file.
    config : str
        Path to the configuration file
    """
    
    path_to_csvs = f'{daily_dir}/{wigos}/'
    print (path_to_csvs)

    path_for_result =output_dir
    print (path_for_result)

    ref_ov = ov_ref

    config = config

    time_to_apply = (datetime.today() - timedelta(days = 1)).date()
    year = str ( time_to_apply ) [:4]
    month = str ( time_to_apply ) [5:7]
    day = str ( time_to_apply ) [8:10]
    print ( f'{year}/{month}/{day}')

    TM = btm.Temperature_model_builder ( '2021/01/01' , f'{year}/{month}/{day}' , ref_ov ,  path_to_csvs  , config )

    TM.check_dates_available ( )

    #TM.get_meta_data_from_first_file ( ) #mvh

    TM.check_optical_module ( )

    # select daily files for current opt
    #n = len(TM.op_mods_list) - 1

    TM.select_dates_for_op_mods ( )

    TM.get_meta_data_from_first_file ( )

    TM.check_resolution_n_get_range ( )

    TM.get_daily_medians ( )

    TM.get_relative_diff ( )

    TM.do_regression_1 ( )

    TM.plot_regression_1 ( )

    TM.choose_n_check_r2_diff_window ( )

    TM.do_regression_2 ( )

    TM.plot_regression_2 ( ) 

    print ("check if nan in alpha values : ", np.isnan(TM.alpha_2).any())

    print (TM.alpha_2)
    print (TM.alpha_2[:20])
    print (TM.alpha_2[10])
    print (type(TM.alpha_2[10]))
    if TM.alpha_2.mask.any() :
        print ('there are masked values, model wont be created : ', TM.alpha_2.mask.any())


    if ( len ( TM.relative_difference ) > 15 ) & ( not TM.alpha_2.mask.any() ) : 

        print ( "enough data because len (TM.relative_difference) = ", len (TM.relative_difference))
        print ( " no nan detected in alpha_2, in fact TM.alpha_2.mask.any() = ", TM.alpha_2.mask.any()) 

        w2nc.write_temp_model_to_netcdf ( path_for_result , TM )
        
if __name__ == "__main__" :
    daily_dir = '/data/pay/REM/ACQ/E_PROFILE_ALC/Overlap/DAILY_FUNCTIONS'
    output_dir = '/data/pay/REM/ACQ/E_PROFILE_ALC/Overlap/TEMP_MODELS/'
    list_wigos = os.listdir(daily_dir)
    #list_wigos = ["0-20000-0-06215"]

    ov_ref = '/proj/pay/E-PROFILE/Overlap_codes/version_for_implementation/OVERLAP_PROBE_EPROFILE-master/TUB120011_20121112_1024.cfg'
    config = '/proj/pay/E-PROFILE/Overlap_codes/version_for_implementation/OVERLAP_PROBE_EPROFILE-master/config.txt'

    for wigos in list_wigos :
        try:
            print ( f" ============ Processing WIGOS {wigos} ============ ")
            compute_temp_model_for_all(daily_dir, wigos, output_dir, ov_ref, config)
        except Exception as e:
            print (f"Error processing WIGOS {wigos} : {e}")
            continue


