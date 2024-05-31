import numpy as np 
import pandas as pd
from glob2 import glob  
from seidart.routines.fabricsynth import Fabric 

# This recipe replicates a couple fabrics from Figure 7 from the following paper:
# https://tc.copernicus.org/articles/15/303/2021/
#
# The parameters were tuned somewhat heuristically to match the figure

multipole1 = {
    'distribution': 'normal',
    'multimodal': True,
    'npts': [160, 175, 160, 90, 90],
    'strike': [110, 290, 255, 75, 50],
    'strike_std': [10, 10, 8, 5, 6],
    'dip': [80, 90, 80, 65, 72],
    'dip_std': [5, 20, 25, 5, 5],
    'skew_strike': [0, 15, -8, 0, 0],
    'skew_dip': [0, -20, -8, 0, 0]
}

multipole2 = {
    'distribution': 'normal',
    'multimodal': True,
    'npts': [155, 320, 85, 150, 110, 110, 110],
    'strike': [275, 80, 295, 115, 115, 195, 230],
    'strike_std': [5, 7, 10, 5, 35, 10, 10],
    'dip': [75, 75, 65, 75, 50, 65, 75],
    'dip_std': [5, 5, 5, 5, 8, 5, 5],
    'skew_strike': [10, 0, 0, 0, 35, 20, -20],
    'skew_dip': [0, 0, 0, 0, -15, 0, 0]
}

# We can plot now or we can plot later. 
mp1 = Fabric(
    multipole1, plot = True
)

mp2 = Fabric(
    multipole2, plot = True
)

mp3 = Fabric(
    multipole2, plot = True
)
# mp1.cmap_name = 'gist_heat_r'
# mp1.custom_cmap()
# mp1.alpha = 0.3 
# mp1.marker_size = 1 
# mp1.projection_plot()



