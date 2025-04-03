import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import mplstereonet
from seidart.routines.fabricsynth import Fabric 

# ------------------
isotropic1 = {
    'distribution': 'uniform',
    'npts': [5000],
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

is1.cmap_name = 'nipy_spectral'
is1.custom_cmap(n_colors = 30)
is1.contour_levels = 30
is1.alpha = 0.3 
is1.marker_size = 1 
is1.projection_plot(vmin = 10.0, colorbar_location = 'right')

# ------------------

# ------------------
weak_horizontal_c = {
    'distribution': 'normal',
    'npts': [8000],
    'trend': [90],
    'trend_std': [30],
    'skew_trend': [0],
    'plunge': [90],
    'plunge_std': [30],
    'skew_plunge': [0],
    'orientation': [0],
    'orientation_std': [0],
    'skew_orientation': [0]
}

# We can plot now or we can plot later. 
whc = Fabric(
    weak_horizontal_c, output_filename = 'weak_horizontal_c.csv', plot = False
)

whc.cmap_name = 'nipy_spectral'
whc.custom_cmap(n_colors = 30)
whc.contour_levels = 30
whc.alpha = 0.3 
whc.marker_size = 1 
whc.projection_plot(vmin = 5.0, colorbar_location = 'right')


# ------------------
strong_horizontal_c = {
    'distribution': 'normal',
    'npts': [5000],
    'trend': [90],
    'trend_std': [5],
    'skew_trend': [0],
    'plunge': [90],
    'plunge_std': [5],
    'skew_plunge': [0],
    'orientation': [0],
    'orientation_std': [0],
    'skew_orientation': [0]
}

# We can plot now or we can plot later. 
shc = Fabric(
    strong_horizontal_c, output_filename = 'strong_horizontal_c.csv', plot = False
)

shc.cmap_name = 'nipy_spectral'
shc.custom_cmap(n_colors = 30)
shc.contour_levels = 30
shc.alpha = 0.3 
shc.marker_size = 1 
shc.projection_plot(vmin = 5.0, colorbar_location = 'right')



# ------------------
weak_vertical_c = {
    'distribution': 'normal',
    'npts': [8000],
    'trend': [0],
    'trend_std': [90],
    'skew_trend': [0],
    'plunge': [0],
    'plunge_std': [45],
    'skew_plunge': [0],
    'orientation': [0],
    'orientation_std': [0],
    'skew_orientation': [0]
}

# We can plot now or we can plot later. 
wvc = Fabric(
    weak_vertical_c, output_filename = 'weak_vertical_c.csv', plot = False
)

wvc.cmap_name = 'nipy_spectral'
wvc.custom_cmap(n_colors = 30)
wvc.contour_levels = 30
wvc.alpha = 0.3 
wvc.marker_size = 1 
wvc.projection_plot(vmin = 3.0, colorbar_location = 'right')

# ------------------
strong_vertical_c = {
    'distribution': 'normal',
    'npts': [5000],
    'trend': [0],
    'trend_std': [90],
    'skew_trend': [0],
    'plunge': [0],
    'plunge_std': [5],
    'skew_plunge': [0],
    'orientation': [0],
    'orientation_std': [0],
    'skew_orientation': [0]
}

# We can plot now or we can plot later. 
svc = Fabric(
    strong_vertical_c, output_filename = 'strong_vertical_c.csv', plot = False
)

svc.cmap_name = 'nipy_spectral'
svc.custom_cmap(n_colors = 30)
svc.contour_levels = 30
svc.alpha = 0.3 
svc.marker_size = 1 
svc.projection_plot(vmin = 80.0, colorbar_location = 'right')
