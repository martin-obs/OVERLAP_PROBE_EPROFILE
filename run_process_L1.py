import process_L1 as pro
import os
import numpy as np
import time
import gc

import ceilo_plotter as cp

import traceback



#import ceilo_plotter as cp


# =============================================================================
# 0-20000-0-06784: Davos. Mountain site at 1600 m
# 0-20000-0-01492: Oslo. Boreal site with probably quite some cloudy episodes. There are more extreme ones though, if you wish I can search for those, just let me know
# 0-20008-0-UGR: Granada. Site in Andalucia with plenty of sun and aerosols
# 0-20000-0-10393: Lindenberg. Typical continental Europe
# =============================================================================



#L1_file = '/scratch/mosborne/CHM15k/eprofile_1a_LqualairIchm15k_v01_20220702_000011_1440.nc'

#L1_file = '/scratch/mosborne/CHM15k/eprofile_1a_LqualairIchm15k_v01_20220701_000011_1440.nc'

base_path = '/scratch/mosborne/L1_for_overlap/0-20008-0-UGR/2022/'

months_to_pro = np.sort(os.listdir( base_path ))

print (months_to_pro)

total_start_time = time.time()

for m in  months_to_pro [ 0:1 ] :
    
    file_names =  os.listdir ( base_path + m )
    
    files_to_pro = np.sort (  [ base_path + m + '/' + f for  f in file_names ] )
    
    print ( files_to_pro )
    
    for f in files_to_pro [ : ]  :
                
        print ('working on ' , f [ -11 : -3 ] )
                    
        start_time = time.time()
        
        config = 'config.txt'
        
        ov = 'TUB120011_20121112_1024.cfg'
        
        try :            
     
            #f = '/scratch/mosborne/L1_for_overlap/0-20008-0-UGR/2022/12/L1_0-20008-0-UGR_A20221203.nc'
            
            L1 = pro.Eprofile_Reader ( f )
            
            L1.get_constants ( config , ov )
            
            L1.create_time ( )
            
            #L1.remove_some ( )
            
            L1.fill_gaps ( ) 
            
            L1.loop_over_time ( start = 0 )
            
            L1.get_final_overlapfunction ()

        
        except Exception: 
            
            print ('passing ', f [-11:-3 ] )
            
            traceback.print_exc()
            
            pass


        #cp.create_ceilo_plot( L1 , location = 'Granada' , instrument = 'Nimbus' , savepath = '.' )
        print ("--- %s seconds ---" % ( time.time ( ) - start_time ) )
        
        
print ("--- %s Total for year seconds ---" % ( time.time ( ) - total_start_time ) )






