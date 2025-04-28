# OVERLAP PROBE EPROFILE

Top level python package OVERLAP_PROBE_EPROFILE which provides code to correct the overlap functions for CHM15K ceilometers. 

This code is a translation and refactoring of Matlab code written by Maxime Hervo, Rolf Ruefenacht and Melania Van Hove.

The code will take EPROFILE L1 CHM15K data files and calculate coeficients that can be used to create an overlap function to correct for temperature effects. 
It is based on the techniques presented in: Hervo, M., Poltera, Y. and Haefele, A. (2016) An empirical method to correct for temperature dependent variations in the 
overlap function of CHM15k ceilometers. Atmospheric Measurement Techniques, 7, 2947– 2959. https://doi.org/10.5194/amt-9-2947-2016.

Creating a virtualenv
----------------------------

To use the code please create a Python virtual environment and then install this package using pip. 

To create a virtual environment:

    $ python -m venv "path to new environment"

Installing into a virtualenv
----------------------------

Activate the venv:

    $ . path to new environment/bin/activate
    
and then within the directory containing setup.py:

    $ pip install .

This will install any dependencies into the virtualenv if necessary.

Using the code
--------------

After installation:

Step 1
------
Command line entry point is:

L1_CHM15k_daily -i {data file or data directory} -c {config file} -f {reference overlap} -o {output file}

Alternatively 'process_L1_example.py' shows how the entry point module can be omprted and called from a python script.

One year of L1 files will take around 5 hours to process and the results will be a csv file for each day for which at least one successful 
overlap sample was found. 

Step2
------
Users should then use "run_temp_model_build.py" as the basis for a wrapper script to select the directory and date range for the csv files resulting from step 1 they
wish to use in calculating a temperature model. The result is a netCDF file containg the temperature model, the reference overlap function used, and associated meta data. 
