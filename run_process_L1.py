import overlap_probe_eprofile.ov_functions.process_L1 as pro
import os
import numpy as np
import time
import traceback

base_path = '/scratch/mosborne/L1_for_overlap/0-20008-0-UGR/2022/'

ov_ref = 'TUB120011_20121112_1024.cfg'

months_to_pro = np.sort(os.listdir( base_path ))

print (months_to_pro)

total_start_time = time.time()

for m in  months_to_pro [ : ] :
    
    file_names =  os.listdir ( base_path + m )
    
    files_to_pro = np.sort (  [ base_path + m + '/' + f for  f in file_names ] )
    
    print ( files_to_pro )
    
    for f in files_to_pro [ : ]  :
                
        print ('working on ' , f [ -11 : -3 ] )
                    
        start_time = time.time()
        
        config = 'config.txt'
        
        ov = ov_ref 
        
        try :            
            
            L1 = pro.Eprofile_Reader ( f )
            
            L1.get_constants ( config , ov )
            
            L1.create_time ( )
            
            L1.fill_gaps ( ) 
            
            L1.loop_over_time ( start = 0 )
            
            L1.get_final_overlapfunction ()
       
        except Exception: 
            
            print ('passing ', f [-11:-3 ] )
            
            traceback.print_exc()
            
            pass

        print ("--- %s seconds ---" % ( time.time ( ) - start_time ) )
             
print ("--- %s Total for year seconds ---" % ( time.time ( ) - total_start_time ) )






