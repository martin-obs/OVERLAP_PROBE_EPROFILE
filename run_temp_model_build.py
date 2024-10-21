#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 10:22:04 2023

@author: mosborne
"""
import sys

import overlap_probe_eprofile.build_temp_model as btm

import overlap_probe_eprofile.write_to_netcdf as w2nc

#path_to_csvs = '/home/vam/OVERLAP/DAILY_FUNCTIONS/HOHENPEISSENBERG/'
path_to_csvs = sys.argv[1]

#path_for_result = '/home/vam/OVERLAP/TEMP_MODEL/'
path_for_result = sys.argv[2]

config = '/home/vam/OVERLAP/OVERLAP_PROBE_EPROFILE-master/config.txt'

ref_ov = '/home/vam/OVERLAP/OVERLAP_PROBE_EPROFILE-master/TUB120011_20121112_1024.cfg'

#TM = btm.Temperature_model_builder ( '2022/01/01' , '2023/12/31' , ref_ov ,  path_to_csvs  , config )

TM = btm.Temperature_model_builder ( '2000/01/01' , '2023/12/31' , ref_ov ,  path_to_csvs  , config )

TM.check_dates_available ( )

#TM.get_meta_data_from_first_file ( ) #mvh

TM.check_optical_module ( )


for n in range(len(TM.op_mods_list)) :

	TM.select_dates_for_op_mods ( n )  #mvh

	TM.get_meta_data_from_first_file ( ) #mvh

	TM.check_resolution_n_get_range ( )

	TM.get_daily_medians ( )

	TM.get_relative_diff ( )

	TM.do_regression_1 ( )

	TM.plot_regression_1 ( )

	TM.choose_n_check_r2_diff_window ( )

	TM.do_regression_2 ( )

	TM.plot_regression_2 ( )

	w2nc.write_temp_model_to_netcdf ( path_for_result , TM )

	TM.check_dates_available ( )

        


