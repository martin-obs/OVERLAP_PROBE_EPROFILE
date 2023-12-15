#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:22:04 2023

@author: mosborne
"""

import overlap_probe_eprofile.build_temp_model as btm

import overlap_probe_eprofile.overlap_utils as w2nc

path_to_csvs = '/scratch/mosborne/overlap_results/LINDENBERG_non_fixed_window/'

path_for_result = '/scratch/mosborne/overlap_results/LINDENBERG_non_fixed_window/temp_model/'

config = 'config.txt'

ref_ov = 'TUB120011_20121112_1024.cfg'

TM = btm.Temperature_model_builder ( '2021/01/01' , '2021/08/14' , ref_ov ,  path_to_csvs  , config )

TM.check_dates_available ( )

TM.get_meta_data_from_first_file ( )

TM.check_optical_module ( )

TM.check_resolution_n_get_range ( )

TM.get_daily_medians ( )

TM.get_relative_diff ( )

TM.do_regression_1 ( )

TM.plot_regression_1 ( )

TM.choose_n_check_r2_diff_window ( )

TM.do_regression_2 ( )

TM.plot_regression_2 ( )

w2nc.write_temp_model_to_netcdf ( path_for_result , TM )



