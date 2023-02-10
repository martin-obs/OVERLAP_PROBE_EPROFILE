import process_L1 as pro

#import ceilo_plotter as cp

L1_file = '/scratch/mosborne/CHM15k/eprofile_1a_LqualairIchm15k_v01_20220702_000011_1440.nc'

L2_file = 'L2.nc'

config = 'config.txt'

ov = 'TUB120011_20121112_1024.cfg'

L1 = pro.Eprofile_Reader ( L1_file )

L1.get_constants ( config , ov )

#L1.remove_some ( )

L1.fill_gaps ( ) 

L1.create_time ( )

L1.loop_over_time ( start = 0 )

#%%

L1.get_final_overlapfunction ()
#cp.create_ceilo_plot( L1 , location = 'Test' , instrument = 'Nimbus' , savepath = '.' )

#%%
