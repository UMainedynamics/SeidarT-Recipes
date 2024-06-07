import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
from seidart.routines.fabricsynth import Fabric 

multipole1 = {
    'distribution': 'normal',
    'npts': [160, 175, 160, 90, 90],
    'strike': [110, 290, 255, 75, 50],
    'strike_std': [10, 10, 8, 5, 6],
    'skew_strike': [0, 15, -8, 0, 0],
    'dip': [80, 90, 80, 65, 72],
    'dip_std': [5, 20, 25, 5, 5],
    'skew_dip': [0, -20, -8, 0, 0]
}

multipole2 = {
    'distribution': 'normal',
    'npts': [155, 320, 85, 150, 110, 110, 110],
    'strike': [275, 80, 295, 115, 115, 195, 230],
    'strike_std': [5, 7, 10, 5, 35, 10, 10],
    'skew_strike': [10, 0, 0, 0, 35, 20, -20],
    'dip': [75, 75, 65, 75, 50, 65, 75],
    'dip_std': [5, 5, 5, 5, 8, 5, 5],
    'skew_dip': [0, 0, 0, 0, -15, 0, 0]
}

multipole3 = {
    'distribution': 'normal',
    'npts': [190, 150, 110, 60, 180, 120],
    'strike': [83, 128, 147, 200, 240, 265],
    'strike_std': [8, 10, 10, 7, 13, 5],
    'skew_strike': [0, -10, 0, 0, -20, 0],
    'dip': [80, 87, 45, 50, 85, 75],
    'dip_std': [5, 5, 5, 5, 7, 4],
    'skew_dip': [0, 0, 0, 0, 0, 0]
}

# We can plot now or we can plot later. 
mp1 = Fabric(
    multipole1, output_filename = 'multipole1.csv', plot = False
)
mp1.cmap_name = 'nipy_spectral'
mp1.custom_cmap(n_colors = 20)
mp1.alpha = 0.3 
mp1.marker_size = 1 
mp1.projection_plot()

# Change the name of euler_angles.csv 
mp2 = Fabric(
    multipole2, output_filename = 'multipole2.csv', plot = False
)
mp2.cmap_name = 'nipy_spectral'
mp2.custom_cmap(n_colors = 6)
mp2.alpha = 0.3 
mp2.marker_size = 1 
mp2.projection_plot()

mp3 = Fabric(
    multipole3, output_filename = 'multipole3.csv', plot = False
)
mp3.cmap_name = 'nipy_spectral'
mp3.custom_cmap(n_colors = 10)
mp3.alpha = 0.3 
mp3.marker_size = 1 
mp3.projection_plot()




# Combine the figure and axes objects into a subplot  
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (10, 15) ) 
for original_axis, new_ax in zip([mp1.ax, mp2.ax, mp3.ax], [ax1, ax2, ax3]):
    for artist in original_axis.get_children():
        artist.remove() 
        new_ax.add_artist(artist) 
    # new_ax.set_xlim(original_axis.get_xlim())
    # new_ax.set_ylim(original_axis.get_ylim())
    # new_ax.set_aspect('equal')
    # new_ax.set_azimuth_ticks(np.arange(0, 360, 10))
    # new_ax.set_longitude_grid(10)
    # new_ax.set_rotation(0)
    # new_ax.grid(True)



# Suppress tick labels
for ax in [new_ax1, new_ax2, new_ax3]:
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Show the combined plot
plt.tight_layout()
plt.show()
    

