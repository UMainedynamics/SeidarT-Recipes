import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 
from scipy.ndimage import gaussian_filter

project_file = 'five_layer.json' 
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)
seis.sourcefunction(seis)

pklfiles = glob('*-Vz-*.pkl')
f = open(pklfiles[0], 'rb')
array_vz = pickle.load(f)

# array_vz.timeseries = array_vz.timeseries[0:4700,:]
array_vz.multichannel_analysis(
    1, 
    velocity_limits = (0, 1200), 
    frequency_limits = (5, 40), 
    contour_levels = 400, 
    nxfft = 2*array_vz.timeseries.shape[1], 
    ntfft = 2*array_vz.timeseries.shape[0], 
    n_tapers = 10,
    vmin = 00,
    colormap = 'inferno'
)

# Get the p and s wave velocities
h = 96*dom.dz # Thickness of the first layer 
T = array_vz.seismic.stiffness_coefficients
rho = seis.stiffness_coefficients['rho']
__, __, __, (vs, vs0, vp_air) = seis.get_christoffel_matrix(0, 'z')
__, __, __, (vs, vs_snow, vp_snow) = seis.get_christoffel_matrix(1, 'z')
__, __, __, (vs, vs_ice, vp_ice) = seis.get_christoffel_matrix(2, 'z')

# Calculate the poisson ratio
nu1 = (vp_snow**2 - 2*vs_snow**2)/(2*(vp_snow**2 - vs_snow**2))
nu2 = (vp_ice**2 - 2*vs_ice**2)/(2*(vp_ice**2 - vs_ice**2))
# Calculate the rayleigh wave velocity
vr1 = vs_snow * (0.862 + 1.14*nu1)/(1+nu1)
vr2 = vs_ice * (0.862 + 1.14*nu2)/(1+nu2)

# Calculate the crossover distance and time
x_c = 2*h*np.sqrt( (vp_snow+vp_ice)/(vp_ice-vp_snow) )
t_c = x_c / vp_snow
tp0 = 2*h*np.sqrt(vp_ice**2 - vp_snow**2)/(vp_snow*vp_ice)
ts0 = 2*h*np.sqrt(vs_ice**2 - vs_snow**2)/(vs_snow*vs_ice)

# Calculate travel times
array_vz.srcrcx_distance()
dist = array_vz.distances
receiver_indices = np.arange(0,len(dist))
air_wave_tt = dist/vp_air
direct_p_tt = dist/vp_snow 
direct_s_tt = dist/vs_snow
rayleigh_tt = dist/vr1

pp_twtt = 2 * np.sqrt((h**2) + (dist/2)**2) / vp_snow
ss_twtt = 2 * np.sqrt((h**2) + (dist/2)**2) / vs_snow

# Correct travel_times to section plot extent
m,n = array_vz.timeseries.shape 
air_wave_tt_c = air_wave_tt/array_vz.dt
direct_p_tt_c = direct_p_tt/array_vz.dt
direct_s_tt_c = direct_s_tt/array_vz.dt
rayleigh_tt_c = rayleigh_tt/array_vz.dt

pp_twtt_c = pp_twtt/array_vz.dt
ss_twtt_c = ss_twtt/array_vz.dt

# ------------------------------------------------------------------------------
# Create the section
array_vz.exaggeration = 0.1
array_vz.agc_gain_window = 501
array_vz.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC')


# # Add the expected travel time curves
# array_vz.ax_section.plot(receiver_indices, direct_p_tt_c, c='#bd5200')
# # array_vz.ax_section.plot(receiver_indices, air_wave_tt_c, c='#00bd97') #mint green
# # array_vz.ax_section.plot(receiver_indices, direct_s_tt_c, c='#ebb734') #mustard yellow
# array_vz.ax_section.plot(receiver_indices, rayleigh_tt_c, c='#e234eb') #magenta
# array_vz.ax_section.plot(receiver_indices, pp_twtt_c, c='#ebb734') #mustard yellow
# array_vz.ax_section.plot(receiver_indices, ss_twtt_c, c='#e234eb') #magenta
# array_vz.fig_section.show()

n = len(dist)
m = seis.time_steps
time_vals = np.arange(0, 35, 2)*1e-2
time_locs = np.round(time_vals / array_vz.dt)
rcx_locs = np.arange(0, n, 60)
rcx_vals = np.round(dist[rcx_locs]). astype(int) #np.round(rcx_locs*array_vz.domain.dx).astype(int)

norm = TwoSlopeNorm(
    vmin=-np.max(array_vz.agc_corrected_timeseries), 
    vmax=np.max(array_vz.agc_corrected_timeseries), 
    vcenter=0
)

fig, ax = plt.subplots(figsize = (5,7), constrained_layout = True )
ax.imshow(
    array_vz.agc_corrected_timeseries, 
    cmap = 'seismic', aspect = 'auto', norm = norm,
    extent = [0, n, m, 0]
)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_xticks(rcx_locs)
ax.set_xticklabels(rcx_vals)
ax.set_ylabel(r'Time (s)')
ax.set_xlabel(r'Source-Receiver Distance (m)')
ax.set_yticks(time_locs)
ax.set_yticklabels(time_vals)
ax.set_aspect(aspect = 0.08)
ax.set_ylim(4650, 0)
fig.tight_layout()
# fig.savefig('two_layer_seismic_phases.png', transparent = True, dpi = 400)
# fig.close()
fig.show()


# ------------------------------------------------------------------------------
# array_vz.fk_analysis(
#     0.25, 
#     frequency_limits = (0, 160), 
#     wavenumber_limits = (0, 0.5),
#     colormap = 'turbo',
#     contour_levels = 200,
#     figure_size = (5,4)
# )

# before computing spec = rfft(...)


dr = 1

array_vz.dispersion_analysis(dr, frequency_limits = (5, 100), velocity_limits = (0, 3500))

# Filter then taper 
tv = np.arange(0, seis.time_steps) * seis.dt
flow = 20 
fhigh = 80
fs = 1/seis.dt 

array_vz.butterworth_filter('bandpass', lowcut = flow, highcut = fhigh)
array_vz.agc_gain_window = 1001
array_vz.sectionplot(use_filtered = True, colormap = 'seismic', amplitude_correction_type = 'AGC')
# plot with log scale, smoothing, and overlay direct/refracted lines
mode_lines = {
  r'P': (vp_snow, '#294c8c'),
  r'P$_{ref}$': (vp_ice, '#0e254f'),
  r'S': (vs_snow, '#5d4191'),
  r'S$_{ref}$': (vs_ice, '#2a184a'),
  r'Rayleigh':    (vr1,  '#8c2929'),
}

array_vz.fk_analysis(
    dr,
    ntfft = 2*array_vz.timeseries.shape[0],
    nxfft = 2*array_vz.timeseries.shape[1],
    taper = 'both',
    wavenumber_limits = (-0.15,0.15), 
    frequency_limits = (0, 100), 
    use_filtered = False
)