#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:22:04 2023

@author: mosborne, refactored by sae
"""
from overlap_probe_eprofile.build_temp_model import make_temperature_model

from datetime import datetime, timedelta

import os


def compute_temp_model_for_all(daily_dir, wigos, output_dir, ov_ref, config):
    """"
    Refactoring of the original script running at MeteoSwiss in order to loop over all WIGOS stations and compute overlap models.
    
    It simply avoids calling this script through bash commands and instead calls the Python functions directly.
    It seems to run fine on the MeteoSwiss server and produce similar results as the operational script currently running at MeteoSwiss
    but we need to test it more thoroughly and check why there has been changes compared to the original model builder.
    
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

    make_temperature_model ( '2021/01/01' , f'{year}/{month}/{day}' , ref_ov ,  path_to_csvs  , config , path_for_result , plot = True , write = True, generate_dummy_if_fail=True)

        
if __name__ == "__main__" :
    daily_dir = '/data/pay/REM/ACQ/E_PROFILE_ALC/Overlap/DAILY_FUNCTIONS'
    output_dir = '/data/pay/REM/ACQ/E_PROFILE_ALC/Overlap/TEMP_MODELS/'
    list_wigos = os.listdir(daily_dir)
    #list_wigos = ["0-20000-0-06215"]

    ov_ref = '/proj/pay/E-PROFILE/Overlap_codes/dev/OVERLAP_PROBE_EPROFILE/TUB120011_20121112_1024.cfg'
    config = '/proj/pay/E-PROFILE/Overlap_codes/dev/OVERLAP_PROBE_EPROFILE/config.txt'

    for wigos in list_wigos :
        try:
            print ( f" ============ Processing WIGOS {wigos} ============ ")
            compute_temp_model_for_all(daily_dir, wigos, output_dir, ov_ref, config)
        except Exception as e:
            print (f"Error processing WIGOS {wigos} : {e}")
            continue


