import numpy as np 
import pandas as pd
from glob2 import glob  
from seidart.routines.fabricsynth import Fabric 

# https://tc.copernicus.org/articles/15/303/2021/

plot = True
sp_strongnorm = {
    'distribution': 'normal',
    'multimodal': True,
    'npts': [155, 320, 80, 150, 110, 110, 110],
    'strike': [275, 80, 295, 115, 115, 195, 230],
    'strike_std': [5, 7, 10, 5, 35, 10, 10],
    'dip': [75, 75, 65, 75, 50, 65, 75],
    'dip_std': [5, 5, 5, 5, 8, 5, 5],
    'skew_strike': [10, 0, 0, 0, 35, 20, -20],
    'skew_dip': [0, 0, 0, 0, -15, 0, 0]
}

sps = Fabric(
    sp_strongnorm, plot = False
)
sps.cmap = 'cool'
sps.alpha = 0.2 
sps.marker_size = 1 
sps.projection_plot()

# sps_unif = {
#     'distribution': 'uniform',
#     'npts': 100,
#     'strike': 90,
#     'strike_std': 15,
#     'dip': 30,
#     'dip_std': 10,
#     'skew_strike': 10,
#     'skew_dip': 5
# }

# mps_unif = {
#     'distribution': 'uniform',
#     'npts': 100,
#     'strike': 90,
#     'strike_std': 15,
#     'dip': 30,
#     'dip_std': 10,
#     'skew_strike': 10,
#     'skew_dip': 5
# }

# mps_norm = {
#     'distribution': 'normal',
#     'npts': [100,220,400],
#     'strike': 90,
#     'strike_std': 15,
#     'dip': 30,
#     'dip_std': 10,
#     'skew_strike': 10,
#     'skew_dip': 5
# }