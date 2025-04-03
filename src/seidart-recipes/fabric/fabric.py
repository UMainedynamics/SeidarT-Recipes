import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
from seidart.routines.fabricsynth import Fabric 


# ------------------
isotropic1 = {
    'distribution': 'uniform',
    'npts': [50000],
    'trend_low': [0],
    'trend_high': [360],
    'plunge_low': [0],
    'plunge_high': [180],
    'orientation_low': [0],
    'orientation_high': [360],
}

# We can plot now or we can plot later. 
is1 = Fabric(
    isotropic1, output_filename = 'isotropic.csv', plot = False
)
# is1.cmap_name = 'nipy_spectral'
# is1.custom_cmap(n_colors = 30)
# is1.contour_levels = 30
# is1.alpha = 0.3 
# is1.marker_size = 1 
# is1.projection_plot(vmin = 1.0, colorbar_location = 'right')

# ------------------
multipole1 = {
    'distribution': 'normal',
    'npts': [160, 175, 160, 90, 90],
    'trend': [110, 290, 255, 75, 50],
    'trend_std': [10, 10, 8, 5, 6],
    'skew_trend': [0, 15, -8, 0, 0],
    'plunge': [80, 90, 80, 65, 72],
    'plunge_std': [5, 20, 25, 5, 5],
    'skew_plunge': [0, -20, -8, 0, 0],
    'orientation': [200, 80, 10, 90, 105],
    'orientation_std': [10, 15, 20, 5, 15],
    'skew_orientation': [-5, 0, 5, 3, 3]
}

# We can plot now or we can plot later. 
mp1 = Fabric(
    multipole1, output_filename = 'multipole1.csv', plot = False
)
mp1.cmap_name = 'nipy_spectral'
mp1.custom_cmap(n_colors = 30)
mp1.contour_levels = 30
mp1.alpha = 0.3 
mp1.marker_size = 1 
mp1.projection_plot(vmin = 1.0, colorbar_location = 'right')
mp1.fig.savefig('multipole1.png', transparent = True, dpi = 400)
# Create the a-axis plot. This will overwrite the c-axis plot.
mp1.projection_plot(vmin = 1.5, a_axes = True, colorbar_location = 'right')
mp1.fig.savefig('multipole1a.png', transparent = True, dpi = 400)

# ------------------
multipole2 = {
    'distribution': 'normal',
    'npts': [190, 300, 100, 240, 90, 90, 130, 150, 50],
    'trend': [275, 74, 295, 115, 105, 135, 205, 225, 235],
    'trend_std': [5, 8, 10, 5, 25, 15, 20, 10, 30],
    'skew_trend': [10, 0, 0, 0, 25, 45, 15, -20, -20],
    'plunge': [75, 72, 65, 80, 50, 30, 70, 80, 30],
    'plunge_std': [5, 5, 5, 5, 10, 3, 3, 8, 5],
    'skew_plunge': [0, 0, 0, 0, -5, 0, 0, 0, 0],
    'orientation': [95, 330, 205, 95, 50, 50, 150, 150, 100], # 1, 2, 3, 4
    'orientation_std': [20, 10, 20, 15, 20, 15, 8, 8, 5],
    'skew_orientation': [-20, 0, 0, -40, 0, 0, 0, 0, 0]
}
# Change the name of euler_angles.csv 
mp2 = Fabric(
    multipole2, output_filename = 'multipole2.csv', plot = False
)
mp2.cmap_name = 'nipy_spectral'
mp2.custom_cmap(n_colors = 6)
mp2.contour_levels = 30
mp2.alpha = 0.3 
mp2.marker_size = 1 
mp2.projection_plot(vmin = 3.3, colorbar_location = 'right')
mp2.fig.savefig('multipole2.png', transparent = True, dpi = 400)
mp2.projection_plot(vmin = 1.2, a_axes = True, colorbar_location = 'right')
mp2.fig.savefig('multipole2a.png', transparent = True, dpi = 400)

# ------------------
multipole3 = {
    'distribution': 'normal',
    'npts': [250, 170, 130, 60, 190, 140, 50],
    'trend': [83, 128, 147, 200, 240, 265, 0],
    'trend_std': [8, 10, 10, 7, 13, 5, 5],
    'skew_trend': [0, -10, 0, 0, -20, 0, 0],
    'plunge': [80, 87, 45, 50, 85, 75,20],
    'plunge_std': [5, 5, 5, 5, 7, 4, 5],
    'skew_plunge': [0, 0, 0, 0, 0, 0, 0],
    'orientation': [80, 30, 65, 150, 120, 80, 105],
    'orientation_std': [20, 30, 20, 20, 15, 20, 20],
    'skew_orientation': [10, -15, 0, 0, 0, 0, 0]
}

mp3 = Fabric(
    multipole3, output_filename = 'multipole3.csv', plot = False
)
mp3.cmap_name = 'nipy_spectral'
mp3.custom_cmap(n_colors = 10)
mp3.contour_levels = 30
mp3.alpha = 0.3 
mp3.marker_size = 1 
mp3.projection_plot(vmin = 0.8, colorbar_location = 'right')
mp3.fig.savefig('multipole3.png', transparent = True, dpi = 400)
mp3.projection_plot(vmin = 1.0, a_axes = True, colorbar_location = 'right')
mp3.fig.savefig('multipole3a.png', transparent = True, dpi = 400)


