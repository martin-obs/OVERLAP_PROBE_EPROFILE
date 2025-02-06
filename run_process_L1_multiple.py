import overlap_probe_eprofile.process_L1 as pro
import os
import numpy as np
import time
import traceback
import pandas as pd
from datetime import datetime, timedelta


def run_process_L1(f, save_path, ov_ref, config):
    print ( f"====== Processing file {f} =======" )

    #total_start_time = time.time ( )
                    
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
                
    #print ("--- %s Total for year seconds ---" % ( time.time ( ) - total_start_time ) )

if __name__ == "__main__" :
    df = pd.read_excel ('/proj/pay/E-PROFILE/Instruments_list.xlsx')
    list_wigos = df ['WIGOS ID'][ df ['Instrument type'] == 'CHM15k' ].values
    list_id = df ['Identifier'][ df ['Instrument type'] == 'CHM15k' ].values

    ov_ref = 'TUB120011_20121112_1024.cfg'
    config = 'config.txt'

    l1_dir = '/data/zue/E_PROFILE/ALC/L1_FILES'
    output_dir = '/data/pay/REM/ACQ/E_PROFILE_ALC/Overlap/DAILY_FUNCTIONS/'

    time_to_apply = (datetime.today() - timedelta(days = 1)).date()
    year = time_to_apply.year
    month =  time_to_apply.month
    day = time_to_apply.day
    
    for wigos, id in zip(list_wigos, list_id) :
        try : 
            input_file = f"{l1_dir}/{wigos}/{year}/{month}/L1_{wigos}_{id}{year}{month}{day}.nc"
            print(input_file)
            run_process_L1(input_file, output_dir, ov_ref, config)               
        except :
            print(f"Warning : Issue with file {input_file}") 
	    