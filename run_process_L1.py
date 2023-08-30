import overlap_probe_eprofile.process_L1 as pro
import os
import numpy as np
import time
import traceback

base_path = '/scratch/mosborne/L1_for_overlap/0-20000-0-10393/2021/' #your L1 directory here



base_path = '/scratch/mosborne/L1_for_overlap/0-20000-0-10395/2021/'

save_path = '/scratch/mosborne/overlap_results/' #your output directory here - a subdirectory with site location name will be created

ov_ref = 'TUB120011_20121112_1024.cfg' #your reference ov function here

config = 'config.txt' # your config file here

months_to_pro = np.sort ( os.listdir ( base_path ) )

print ( months_to_pro )

total_start_time = time.time ( )

for m in  months_to_pro [ : ] :
    
    file_names =  os.listdir ( base_path + m )
    
    files_to_pro = np.sort (  [ base_path + m + '/' + f for  f in file_names ] )
    
    print ( files_to_pro )
    
    for f in files_to_pro [ : ]  :
                
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






