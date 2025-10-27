import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 

import numpy as np

# ------------------------------------------------------------------------------
def compute_refracted_crossovers(h, vp, source_depth):
    """
    Compute crossover distance and time where the refracted wave overtakes
    the direct wave, assuming the source is in Layer 2.
    
    Parameters:
        h (array): layer thicknesses (len = N)
        vp (array): velocities (len = N+1)
        source_depth (float): depth of source in meters
    
    Returns:
        List of (interface_index, crossover_x [m], crossover_t [s])
    """
    results = []
    
    # Identify which layer source is in
    cumulative_depths = np.cumsum([0] + list(h))
    source_layer = np.searchsorted(cumulative_depths, source_depth, side='right') - 1
    
    # Redefine top layer to be the source layer
    v_top = vp[source_layer]
    
    for i in range(source_layer + 1, len(vp)):
        v_refract = vp[i]
        t_vert = 0.0
        
        for j in range(source_layer, i):
            theta_j = np.arcsin(vp[j] / v_refract)
            
            if j == source_layer:
                h_adj = cumulative_depths[j+1] - source_depth
            else:
                h_adj = h[j]
            
            t_vert += h_adj / (vp[j] * np.cos(theta_j))
        
        t_vert *= 2
        x_c = t_vert / (1/v_top - 1/v_refract)
        t_c = x_c / v_top
        
        results.append((i, x_c, t_c))
    return results

def compute_ray_theory_crossovers(h, vp, source_depth):
    """
    Compute crossover distance and ray-theory-based travel time for refracted waves.
    
    Parameters:
        h (array): layer thicknesses [m]
        vp (array): velocities [m/s] (len = len(h) + 1)
        source_depth (float): source depth [m]
    
    Returns:
        List of tuples: (interface_index, crossover_x [m], t_ray_theory [s])
    """
    results = []
    
    # Get depth boundaries for layer interfaces
    depths = np.cumsum([0] + list(h))
    source_layer = np.searchsorted(depths, source_depth, side='right') - 1
    v_top = vp[source_layer]
    
    for i in range(source_layer + 1, len(vp)):
        v_refract = vp[i]
        t_vert = 0.0
        
        for j in range(source_layer, i):
            theta_j = np.arcsin(vp[j] / v_refract)
            
            if j == source_layer:
                h_adj = depths[j+1] - source_depth
            else:
                h_adj = h[j]
            
            t_vert += h_adj / (vp[j] * np.cos(theta_j))
        
        t_vert *= 2  # down and up
        x_c = t_vert / (1/v_top - 1/v_refract)
        
        # ⚠️ Here's the difference: ray-theory arrival time
        t_ray = t_vert + x_c / v_refract
        
        results.append((i, x_c, t_ray))
    
    return results

# ------------------------------------------------------------------------------
h = np.array([4, 16, 30, 50, 95])
vp = np.array([1000, 1500, 2000, 2500, 3000, 3500])
vs = np.array([500, 800, 1100, 1400, 1700, 2000])

# pcrossovers = compute_refracted_crossovers(h, vp, source_depth = 8)
# scrossovers = compute_refracted_crossovers(h, vs, source_depth = 8)

pcrossovers = compute_ray_theory_crossovers(h, vp, source_depth = 3)
scrossovers = compute_ray_theory_crossovers(h, vs, source_depth = 3)

for i, x, t_c in pcrossovers:
    print(f"Refracted from Layer {i+1}: crossover at {x:.1f} m, time = {t_c:.3f} s")

for i, x, t_c in scrossovers:
    print(f"Refracted from Layer {i+1}: crossover at {x:.1f} m, time = {t_c:.3f} s")
 
# ------------------------------------------------------------------------------



## Initiate the model and domain objects
project_file = 'six_layer.json' 
receiver_file = 'receivers4.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

zpklfiles = glob('*-Vz-*.pkl')
xpklfiles = glob('*-Vx-*.pkl')
f = open(zpklfiles[0], 'rb')
array_vz = pickle.load(f)
f.close() 
f = open(xpklfiles[0], 'rb')
array_vx = pickle.load(f)
f.close() 


array_vz.exaggeration = 0.11
array_vz.agc_gain_window = 1201 
array_vz.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC')


array_vz.butterworth_filter(
    'lowpass', lowcut = 20, highcut = 60, pad_samples = 2*seis.time_steps, order = 8
)

# array_vz.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC', use_filtered = True)

# ------------------------------------------------------------------------------
mode_lines = {
    r'P1' : (1200, '#397afa'),
    r'P2' : (1600, '#397afa'),
    r'P3' : (2000, '#397afa'),
    r'P4' : (2500, '#397afa'),
    r'P5' : (3000, '#397afa'),
    r'P6' : (3500, '#397afa'),
    r'S1' : (750,  '#ff4040'),
    r'S2' : (900,  '#ff4040'),
    r'S3' : (1100, '#ff4040'),
    r'S4' : (1400, '#ff4040'),
    r'S5' : (1700, '#ff4040'),
    r'S6' : (2000, '#ff4040')
}

array_vz.fk_analysis(
    1, 
    ntfft = 2*array_vz.timeseries.shape[0],
    nxfft = 2*array_vz.timeseries.shape[1],
    taper = 'both',
    wavenumber_limits = (-0.0,0.035), 
    frequency_limits = (0, 60), 
    use_filtered = False,
    colormap = 'pink_r',
    mask_db = -40,
    mode_lines = mode_lines
)



array_vx.agc_gain_window = 701
array_vx.exaggeration = 0.09
array_vx.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC')




array_vz.srcrcx_distance()
m, n = array_vz.timeseries.shape 
tv = np.arange(0, m) * seis.dt 
r = array_vz.distances.copy()

# Create the mask
vzstd = np.std(array_vz.agc_corrected_timeseries, axis = 0, keepdims = True)
vxstd = np.std(array_vx.agc_corrected_timeseries, axis = 0, keepdims = True)

std_threshhold = 0.25
maskx = np.abs(array_vx.agc_corrected_timeseries) < (std_threshhold * vxstd)
maskz = np.abs(array_vz.agc_corrected_timeseries) < (std_threshhold * vzstd)

vxmasked = np.ma.masked_where(maskx, array_vx.agc_corrected_timeseries) 
vzmasked = np.ma.masked_where(maskz, array_vz.agc_corrected_timeseries) 


time_vals = np.arange(0, 60, 3)*1e-2
time_locs = np.round(time_vals / array_vz.dt)
rcx_locs = np.arange(0, n, 60)
rcx_vals = np.round(array_vz.distances[rcx_locs]). astype(int) #np.round(rcx_locs*array_vz.domain.dx).astype(int)

norm = TwoSlopeNorm(
    vmin=-np.max(array_vz.agc_corrected_timeseries), 
    vmax=np.max(array_vz.agc_corrected_timeseries), 
    vcenter=0
)

# ------------------------------------------------------------------------------

fig = plt.figure(figsize=(7, 6))
gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[ 3, 0.25, 2], wspace=0.07)
gs_fk = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs[2], height_ratios=[1,1,0.9, 0.2, 0.2, 0.2], wspace=0.02)

# Main plots
ax1 = fig.add_subplot(gs[0])  # Vx shot gather
ax2 = fig.add_subplot(gs_fk[0:3])  # fk spectrum

# Plot Vx
pcm_x = ax1.pcolormesh(r, tv, vzmasked.data, cmap='seismic', shading='auto', norm=norm)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_xlabel("Source-Receiver Distance (m)")
ax1.set_ylabel("Two-way Travel Time (s)")
ax1.set_ylim(tv[7250], tv[0])
ax1.yaxis.set_inverted(True)
ax1.grid(True, color='black', alpha=0.15, linestyle = ':')

# ----------------------------------- fk plot ----------------------------------
# Plot f–k
mask_db = -110
frequency_limits = (0, 60)
wavenumber_limits = (-0.07, 0.07)

mask = array_vz.fk_spectrum < mask_db
fk_spectrum_masked = np.ma.masked_where(mask, array_vz.fk_spectrum)
contour_levels = 200 

contour = ax2.contourf(
    array_vz.fk_wavenumbers, array_vz.fk_freqs, fk_spectrum_masked, 
    cmap='Greys', levels=contour_levels
)
ax2.set_xlabel(r'Wavenumber (m$^{-1}$)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_ylim(frequency_limits)
ax2.set_xlim(wavenumber_limits)


wavenumber_units = 'cycles'
for label, spec in mode_lines.items():
    if isinstance(spec, (tuple, list)):
        v, col = spec
    else:
        v, col = spec, 'white'
    # compute k_line in correct units
    if wavenumber_units == 'radians':
        kline = 2 * np.pi * array_vz.fk_freqs / v
    else:
        kline = array_vz.fk_freqs / v
    
    if 'S' in label:
        linetype = ':'
    else:
        linetype = '--'
    ax2.plot(kline, array_vz.fk_freqs, linetype, color=col, lw=2, label=label, alpha = 0.4)
    ax2.plot(-kline, array_vz.fk_freqs, linetype, color=col, lw=2, alpha = 0.6)

# ---------------------------------- Colorbar ----------------------------------
cax = fig.add_subplot(gs_fk[4])
cbar = fig.colorbar(contour, cax=cax, orientation='horizontal')
cbar.set_label("Gain (dB)")
cax.tick_params(axis='x', labelsize=8)
cax.xaxis.set_ticks_position('bottom')
cax.set_xticks(cbar.ax.get_xticks())  # sync ticks just in case
cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

for label in cbar.ax.get_xticklabels():
    label.set_rotation(-45)  # Clockwise
    label.set_ha('left')    # Align for readability
    label.set_va('top')
    label.set_x(label.get_position()[0] + 0.01)

fig.subplots_adjust(
    left=0.09, right = 0.95, bottom = 0.03, top = 0.92, wspace=0.05
)
# fig.tight_layout()
fig.savefig('bodywave_shotgather_fk.png', dpi = 200)
fig.show()


