import numpy as np 
import pandas as pd 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from seidart.routines.definitions import * 
from seidart.routines import prjrun

'''
This is a custom figure maker script. Much of it has been copied from functions
that can be found in the Array object. See Array.sectionplot and 
Array.sectionwiggleplot if you would like to do something similar in other 
model outputs. 
'''

# Load saved objects
zop1 = pickle.load( open('zop1.ez.pkl', 'rb') )
zop2 = pickle.load( open('zop2.ez.pkl', 'rb') )

zop1.srcrcx_distance() 
zop2.srcrcx_distance() 

# This step isn't necessary because the domain, model, and material objects are
# wrapped in the zop objects. 
prjfile = 'zop.prj'
domain, material, seismic, electromag = prjrun.domain_initialization(prjfile)

png_dpi = 400

# Get the profiles of temperature, lwc, em velocity,
_, temp_profile = parameter_profile_1d(
    domain, material, electromag, 300, parameter = 'temperature'
)
_, lwc_profile = parameter_profile_1d(
    domain, material, electromag, 300, parameter = 'lwc'
)
profile_depth, vz_profile = parameter_profile_1d(
    domain, material, electromag, 300, parameter = 'velocity'
)

# Get only the profile values that are in the depth range of the survey
rcx_depth = (zop2.receiver_xyz_all[:,2] - 65)*zop2.domain.dz

min_depth = rcx_depth.min()
max_depth = rcx_depth.max() 
range_ind = (profile_depth >= min_depth) * (profile_depth <= max_depth)
vz_profile = vz_profile[range_ind,:]
lwc_profile = lwc_profile[range_ind]
temp_profile = temp_profile[range_ind] 
profile_depth = profile_depth[range_ind]

depth = profile_depth[::10]
vz = vz_profile[10:][::10]

# ------------------------------------------------------------------------------
exaggeration = 0.03
rcx_tick_label = rcx_depth[::15]
rcx_tick_locs = np.arange(0, len(rcx_depth))[::15]
timevals = np.arange(0, zop1.electromag.time_steps)
timeval_locs = timevals[::600]
timeval_labels = np.round(timeval_locs * zop1.electromag.dt / 1e-9).astype(int)


norm = TwoSlopeNorm(vmin=-np.max(zop1.timeseries), vmax=np.max(zop1.timeseries), vcenter=0)

fig0, ax0 = plt.subplots(figsize = (3,3), constrained_layout = True)
# ax0 = plt.subplot(gs[0]) 
im = ax0.imshow(
    zop1.timeseries, cmap = 'seismic', norm = norm, 
    aspect = 'auto', 
    extent = [0, 75, zop1.electromag.time_steps, 0],
    origin = 'upper'
)
ax0.set_xlabel(r'Depth (m)')

ax0.xaxis.tick_top()
ax0.xaxis.set_label_position('top')
ax0.set_xticks(rcx_tick_locs)
ax0.set_xticklabels(rcx_tick_label)
ax0.set_ylabel(r'Two-way Travel Time (ns)')
ax0.set_yticks(timeval_locs)
ax0.set_yticklabels(timeval_labels)
ax0.set_ylim([2400, 600])      
# ax0.set_aspect(aspect = exaggeration)
# ax0.text(0, m + 0.03*m, 'x $10^{-6}$')
 # Enable x-tick labels on the top subplot
ax0.tick_params(
    axis='x', which='both', 
    bottom=False, top=True, 
    labelbottom=False, labeltop=True
)

# ax0.scatter(depth-5, travel_times/zop1.electromag.dt, marker = '.', s = 10, c='#42eff5')

plt.tight_layout()
plt.savefig('zop1.amplitude_colored.png', transparent = True, dpi = png_dpi)
plt.close()

# ------------------------------------------------------------------------------
norm = TwoSlopeNorm(vmin=-np.max(zop2.timeseries), vmax=np.max(zop2.timeseries), vcenter=0)

fig1, ax1 = plt.subplots(figsize = (3, 3), constrained_layout = True)
im = ax1.imshow(
    zop2.timeseries, 
    cmap = 'seismic', norm=norm,
    aspect = 'auto', 
    extent = [0, 75, zop1.electromag.time_steps, 0],
    origin = 'upper'
)
ax1.set_xlabel(r'Depth (m)')
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_xticks(rcx_tick_locs)
ax1.set_xticklabels(rcx_tick_label)
ax1.set_ylabel(r'Two-way Travel Time (ns)')
ax1.set_yticks(timeval_locs)
ax1.set_yticklabels(timeval_labels)
ax1.set_ylim([2400, 600])      
# Enable x-tick labels on the top subplot
ax1.tick_params(
    axis='x', which='both', 
    bottom=False, top=True, 
    labelbottom=False, labeltop=True
)
plt.tight_layout()
plt.savefig('zop2.amplitude_colored.png', transparent = True, dpi = png_dpi)
plt.close()

# ------------------------------------------------------------------------------
positive_fill_color = '#999999'
negative_fill_color = None
trace_indice = np.arange(0,30, dtype = int)
n_traces_to_plot = len(trace_indice)

# Source receiver distance is constant for zop2
travel_time = np.ones([len(profile_depth)]) * 25 
travel_time = travel_time / vz_profile[:,2] /1e-9
travel_time_depth = profile_depth[::10]
travel_time = travel_time[::10]
travel_time = travel_time[2:]

data = zop2.timeseries.copy()
m,n = data.shape 
time = np.arange(m)*zop1.electromag.dt/1e-9
# Plotting the section plot
fig2, ax2 = plt.subplots(figsize=(4, 7.1))

scaling_factor = 2.0e5
# Loop over each depth to plot each time series
for i in range(0,n_traces_to_plot):
    indice = trace_indice[i]
    vz = vz_profile[profile_depth == rcx_depth[indice]][:,2]
    ax2.plot(
        time, data[:, indice]*scaling_factor + rcx_depth[indice], 
        'k', linewidth=0.8
    )  # Plotting the waveform in black
    if positive_fill_color:
        ax2.fill_between(
            time, rcx_depth[indice], 
            data[:, indice]*scaling_factor + rcx_depth[indice], 
            where=(data[:, indice] > 0), color=positive_fill_color
        )  # Shaded positive regions
    if negative_fill_color:
        ax2.fill_between(
            time, rcx_depth[indice], 
            data[:, indice]*scaling_factor + rcx_depth[indice], 
            where=(data[:, indice] < 0), color=negative_fill_color
        )
    ax2.scatter(
        travel_time[indice], 
        data[0,indice]*scaling_factor + travel_time_depth[indice], 
        color = '#f500cc', marker = '|', s=150, linewidth = 1.5
    )

# Invert the y-axis so increasing depth goes downward
ax2.invert_yaxis()
ax2.grid(
    True, which = 'both', axis = 'x', 
    linestyle = '--', color = '#c7c7c7', linewidth = 0.7
)
# Labeling
ax2.set_xlabel('Time (ns)')
ax2.set_ylabel('Depth (m)')
ax2.set_xlim(1.2e2, 2.0e2)
# Show the plot
plt.tight_layout()
plt.savefig('zop2.section_wiggleplot.png', transparent = True, dpi = png_dpi)
plt.close()


# ------------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize = (2.968,2) )

ax3.plot(rcx_depth-5, zop1.distances, '.', color = '#4152a6', lw = 0.5,  )
ax3.plot(rcx_depth-5, zop2.distances, '.', color = '#a84040', lw = 0.5,  )
ax3.set_xlabel('Depth (m)')
ax3.set_ylabel('Source-Receiver \n Distance (m)')
ax3.set_xticks(rcx_tick_locs)
ax3.set_xticklabels(rcx_tick_label)
ax3.set_ylim(20,40)
plt.tight_layout()
plt.margins(0)
plt.savefig('zop.source_receiver_distance.png', transparent = True, dpi = png_dpi)
# plt.show()
plt.close()
