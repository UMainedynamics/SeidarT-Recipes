import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

from seidart.routines.definitions import *
from seidart.routines import prjbuild, prjrun, sourcefunction
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 

prjfile = 'two_layer.prj' 
dom, mat, seis, em = prjrun.domain_initialization(prjfile)
timevec, fx, fy, fz, srcfn = sourcefunction(seis, 1e8, 'gaus1')

pklfiles = glob('*.pkl')
f = open('two_layer.vz.pkl', 'rb')
array_vz = pickle.load(f)
array_vz.dt = array_vz.dt*0.7 
array_vz.seismic.dt = array_vz.seismic.dt * 0.7

# Get the p and s wave velocities
h = 20 # Thickness of the first layer 
T = array_vz.seismic.tensor_coefficients[:,1:-1].astype(float)
rho = array_vz.seismic.tensor_coefficients[:,-1].astype(float)
vp0, vph, vs0, vsh = mf.tensor2velocities(T[0,:], rho[0])
vp1, vph, vs1, vsh = mf.tensor2velocities(T[1,:], rho[1])
vp2, vph, vs2, vsh = mf.tensor2velocities(T[2,:], rho[2])

# Calculate the poisson ratio
nu1 = (vp1**2 - 2*vs1**2)/(2*(vp1**2 - vs1**2))
nu2 = (vp2**2 - 2*vs2**2)/(2*(vp2**2 - vs2**2))
# Calculate the rayleigh wave velocity
vr1 = vs1 * (0.862 + 1.14*nu1)/(1+nu1)
vr2 = vs2 * (0.862 + 1.14*nu2)/(1+nu2)

# Calculate the crossover distance and time
x_c = 2*h*np.sqrt( (vp1+vp2)/(vp2-vp1) )
t_c = x_c / vp1
tp0 = 2*h*np.sqrt(vp2**2 - vp1**2)/(vp1*vp2)
ts0 = 2*h*np.sqrt(vs2**2 - vs1**2)/(vs1*vs2)

# Calculate travel times
dist = array_vz.distances
receiver_indices = np.arange(0,len(dist))
air_wave_tt = dist/vp0
direct_p_tt = dist/vp1 
direct_s_tt = dist/vs1
rayleigh_tt = dist/vr1

pp_twtt = 2 * np.sqrt((h**2) + (dist/2)**2) / vp1
ss_twtt = 2 * np.sqrt((h**2) + (dist/2)**2) / vs1

# Correct travel_times to section plot extent
m,n = array_vz.timeseries.shape 
air_wave_tt_c = air_wave_tt/array_vz.dt
direct_p_tt_c = direct_p_tt/array_vz.dt
direct_s_tt_c = direct_s_tt/array_vz.dt
rayleigh_tt_c = rayleigh_tt/array_vz.dt

pp_twtt_c = pp_twtt/array_vz.dt
ss_twtt_c = ss_twtt/array_vz.dt

# ------------------------------------------------------------------------------
array_vz.exaggeration = 0.1
array_vz.agc_gain_window = 1301
array_vz.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC')
# array_vz.sectionwiggleplot(
#     receiver_indices, 
#     scaling_factor = 0.25,
#     receiver_distance = array_vz.distances.round(),
#     positive_fill_color = '#1f2282',
#     negative_fill_color = '#821f24', 
#     amplitude_correction_type = 'AGC',
#     plot_vertical = True,
#     figure_size = (7,9)
# )

# # Add the expected travel time curves
# array_vz.ax_section.plot(receiver_indices, direct_p_tt_c, c='#bd5200')
# # array_vz.ax_section.plot(receiver_indices, air_wave_tt_c, c='#00bd97') #mint green
# # array_vz.ax_section.plot(receiver_indices, direct_s_tt_c, c='#ebb734') #mustard yellow
# array_vz.ax_section.plot(receiver_indices, rayleigh_tt_c, c='#e234eb') #magenta
# array_vz.ax_section.plot(receiver_indices, pp_twtt_c, c='#ebb734') #mustard yellow
# array_vz.ax_section.plot(receiver_indices, ss_twtt_c, c='#e234eb') #magenta
# array_vz.fig_section.show()

time_vals = np.arange(0, 22, 2)*1e-2
time_locs = np.round(time_vals / array_vz.dt)
rcx_locs = np.arange(0, n, 60)
rcx_vals = np.round(rcx_locs*array_vz.domain.dx).astype(int)

norm = TwoSlopeNorm(
    vmin=-np.max(array_vz.agc_corrected_timeseries), 
    vmax=np.max(array_vz.agc_corrected_timeseries), 
    vcenter=0
)

fig, ax = plt.subplots(figsize = (5,6), constrained_layout = True )
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
ax.set_aspect(aspect = 0.1)
plt.tight_layout()
plt.savefig('two_layer_seismic_phases.png', transparent = True, dpi = 400)
plt.close()
# plt.show()


# ------------------------------------------------------------------------------
# array_vz.fk_analysis(
#     0.25, 
#     frequency_limits = (0, 160), 
#     wavenumber_limits = (0, 0.5),
#     colormap = 'turbo',
#     contour_levels = 200,
#     figure_size = (5,4)
# )
