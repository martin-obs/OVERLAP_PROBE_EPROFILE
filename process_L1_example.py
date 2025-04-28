from overlap_probe_eprofile.process_L1 import process_L1

#your input file or directory here (if directory all files will be processed) 
base_path = '/scratch/mosborne/L1_for_overlap/0-20008-0-UGR-GRANADA/2022/04/L1_0-20008-0-UGR_A20220401.nc' 

#your output directory here - a subdirectory with site location name will be created
save_path = '/scratch/mosborne/overlap_results/' 

#your reference ov function here
ov_ref = 'TUB120011_20121112_1024.cfg' 

# your config file here
config = 'config.txt' 

process_L1 ( base_path, config , ov_ref , save_path )
                                 
