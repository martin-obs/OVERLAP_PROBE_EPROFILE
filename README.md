# OVERLAP_PROBE_EPROFILE
Repository for the development of code to correct the overlap functions for CHM15K ceilometers and others. 

This code is a translation and refactoring of Matlab code written by Maxime Hervo, Rolf Ruefenacht and Melania Van Hove.

The code will take a list of EPROFILE L1 CHM15K data files and calculate coeficients that can be used to create an overlap function to correct for temperature effects. 
It is based on the techniques presented in:

Hervo, M., Poltera, Y. and Haefele, A. (2016) An empirical method to correct for temperature dependent variations in the overlap function of CHM15k ceilometers. 
Atmospheric Measurement Techniques, 7, 2947– 2959. https://doi.org/10.5194/amt-9-2947-2016.

Step 1
Users should use "run_process_L1.py" as the basis for a wrapper script to select the directory and date range for the L1 files they wish to use, as well as the config file 
and the reference overlap function. One year of L1 files will take around 5 hours to process and the results will be a csv file for each day for which at least one successful 
overlap sample was found. 

Step2
Users should then use "run_temp_model_build.py" as the basis for a wrapper script to select the directory and date range for the csv files resulting from step 1 they
wish to use in calculating a temperature model. The result is a netCDF file containg the temperature model, the reference overlap function used, and associated meta data. 
