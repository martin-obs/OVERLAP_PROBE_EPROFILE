import pandas as pd
from datetime import datetime, timedelta

from overlap_probe_eprofile.process_L1 import process_L1

if __name__ == "__main__" :
    df = pd.read_excel ('/proj/pay/E-PROFILE/Instruments_list.xlsx')
    list_wigos = df ['WIGOS ID'][ df ['Instrument type'] == 'CHM15k' ].values
    list_id = df ['Identifier'][ df ['Instrument type'] == 'CHM15k' ].values

    ov_ref = '/proj/pay/E-PROFILE/Overlap_codes/dev/OVERLAP_PROBE_EPROFILE/TUB120011_20121112_1024.cfg'
    config = '/proj/pay/E-PROFILE/Overlap_codes/dev/OVERLAP_PROBE_EPROFILE/config.txt'

    l1_dir = '/data/zue/E_PROFILE/ALC/L1_FILES'
    output_dir = '/data/pay/REM/ACQ/E_PROFILE_ALC/Overlap/DAILY_FUNCTIONS_DEV/'

    time_to_apply = (datetime.today() - timedelta(days = 1)).date()
    year = str ( time_to_apply ) [:4]
    month = str ( time_to_apply ) [5:7]
    day = str ( time_to_apply ) [8:10]
    
    for wigos, id in zip(list_wigos, list_id) :
        try : 
            input_file = f"{l1_dir}/{wigos}/{year}/{month}/L1_{wigos}_{id}{year}{month}{day}.nc"
            print(input_file)

            process_L1 ( input_file, config , ov_ref , output_dir )
        except :
            print(f"Warning : Issue with file {input_file}") 
	    