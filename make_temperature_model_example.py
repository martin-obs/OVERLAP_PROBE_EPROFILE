from overlap_probe_eprofile.build_temp_model import make_temperature_model

path_to_csvs = '/scratch/mosborne/overlap_results/GRANADA/'

path_for_result = '.'

config = 'config.txt'

ref_ov = 'TUB120011_20121112_1024.cfg'

make_temperature_model ( '2021/01/01' , '2022/12/30' , ref_ov ,  path_to_csvs  , config ,path_for_result , plot = True )