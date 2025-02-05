import overlap_probe_eprofile.process_L1 as pro
import os
import sys
import numpy as np
import time
import traceback

#print ("run_process_L1 started")

f = sys.argv[1]

print ( f"====== Processing file {f} =======" )

save_path = sys.argv[2]

ov_ref = sys.argv[3] #'/proj/pay/E-PROFILE/Overlap_codes/version_for_implementation/OVERLAP_PROBE_EPROFILE-master/TUB120011_20121112_1024.cfg' #your reference ov function here

config = sys.argv[4] #'/proj/pay/E-PROFILE/Overlap_codes/version_for_implementation/OVERLAP_PROBE_EPROFILE-master/config.txt' # your config file here

total_start_time = time.time ( )
                
print ('working on ' , f [ -11 : -3 ] )
            
start_time = time.time()

try :            
    
    L1 = pro.Eprofile_Reader ( f )
    
    L1.get_constants ( config , ov_ref )
    
    L1.create_time ( )
    
    L1.fill_gaps ( ) 
    
    L1.loop_over_time ( start = 0 )
    
    L1.get_final_overlapfunction ( save_path )

except Exception: 
    
    print ('passing ', f [-11:-3 ] )
    
    traceback.print_exc()
    
    pass

print ("--- %s seconds ---" % ( time.time ( ) - start_time ) )
             
print ("--- %s Total for year seconds ---" % ( time.time ( ) - total_start_time ) )