#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:44:01 2023

@author: mosborne
"""

import os

import numpy as np

base_path = '/scratch/mosborne/L1_for_overlap/0-20008-0-UGR/2021/'

files = np.sort ( [ f for f in os.listdir ( base_path ) if '.nc' in f  ] )

days = np.linspace ( 1 , 31 , 31 )

months = np.linspace ( 1 , 12 , 12 )


for f in files: 
    
    fdate = f.split ( '_' ) [ -1 ] [ 1 : 9 ]

    
    m = fdate [ 4 :6 ]
                 

    
    comline = 'mv ' + base_path + f + ' ' + base_path + m 
    
    print ( comline) 
    
    os.system (comline)
                   
                   