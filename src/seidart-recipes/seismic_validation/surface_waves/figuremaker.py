import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from scipy.signal import welch

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 

from disba import PhaseDispersion

def normalize_2d(data, axis=0, eps=1e-12):
  norm = np.max(np.abs(data), axis=axis, keepdims=True)
  norm[norm < eps] = 1.0
  return data / norm

def bidirectional_normalization(data):
  # Normalize each trace (receiver)
  norm_by_trace = normalize_2d(data, axis=0)
  # Then normalize each time sample (row)
  norm_by_time = normalize_2d(norm_by_trace, axis=1)
  return norm_by_time
  
def compute_distance_weights(n_receivers, dx, source_index=0, power=0.5, eps=1e-3):
  offsets = np.arange(n_receivers) - source_index
  distances = np.abs(offsets * dx) + eps  # avoid zero division
  weights = 1 / distances**power
  weights /= np.max(weights)  # normalize
  return weights
  
def normalize_per_trace(data, eps=1e-12):
  norm = np.max(np.abs(data), axis=0, keepdims=True)
  norm[norm < eps] = 1.0
  return data / norm

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'surfacewave_validation.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

pklfiles = glob('*-Vz-*.pkl')
f = open(pklfiles[0], 'rb')
array_vz = pickle.load(f)
f.close() 


pklfiles = glob('*-Vx-*.pkl')
f = open(pklfiles[0], 'rb')
array_vx = pickle.load(f)
f.close()

array_vz.srcrcx_distance()
m, n = array_vz.timeseries.shape 
tv = np.arange(0, m) * seis.dt 
r = array_vz.distances.copy()

seis.use_multimodal = True
seis.source_n_octaves = 3
seis.sourcefunction(seis) 
fs = 1.0/seis.dt 
# freq_psd, Pxx = welch(
#   seis.sourcefunction_z, 
#   fs = fs, window = 'tukey', nperseg = 2**13, noverlap = 2**12, 
#   scaling = 'density'
# )
# ------------------------------------------------------------------------------
n_receivers = array_vz.timeseries.shape[1]
source_index = 14
weights = compute_distance_weights(n_receivers, dom.dx, source_index = source_index)

freqs, vs, image, picks = compute_dispersion_image(
  normalize_per_trace(array_vz.timeseries), 
  dx = 1, dt = seis.dt, 
  fmin = 0, fmax = 50, 
  vmin = 10, vmax = 1000,
  nv = 1000, nfreq = 400, 
)

array_vz.timeseries = bidirectional_normalization(array_vz.timeseries)
# array_vz.sectionplot(colormap = 'seismic')
# Create the mask
vzstd = np.std(array_vz.timeseries, axis = 0, keepdims = True)
std_threshhold = 0.05
maskz = np.abs(array_vz.timeseries) < (std_threshhold * vzstd)
vzmasked = np.ma.masked_where(maskz, array_vz.timeseries) 

norm = TwoSlopeNorm(
    vmin=-np.max(array_vz.timeseries), 
    vmax=np.max(array_vz.timeseries), 
    vcenter=0
)

rho = seis.stiffness_coefficients['rho']
velocity_model = np.zeros([5, 4])
velocity_model[:,-1] = rho[[0,1,3,4,5]]/1000
velocity_model[:,0] = np.array([0.005, 0.010, 0.020, 0.060, 0.300])
velocity_model[:,1] = np.array([500, 800, 1200, 1800, 2400])/1000
velocity_model[:,2] = np.array([150, 250, 400, 700, 1000])/1000
pd = PhaseDispersion(*velocity_model.T, algorithm="dunkin", dc=0.001)

# Periods must be sorted starting with low periods
f = np.linspace(2.5, 50.0, 100)
t = 1.0 / np.linspace(50, 2.5, 100)
fundamental = pd(t, mode=0, wave="rayleigh")
higher1 = pd(t, mode=1, wave="rayleigh")
higher2 = pd(t, mode=2, wave="rayleigh")
higher3 = pd(t, mode=3, wave="rayleigh")
period0 = fundamental.period
period1 = higher1.period
period2 = higher2.period
period3 = higher3.period 
velocity0 = fundamental.velocity
velocity1 = higher1.velocity
velocity2 = higher2.velocity
velocity3 = higher3.velocity

# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=[ 1,0.2, 1], wspace=0.05)
gs_masw = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[2], height_ratios=[1,1, 0.1, 0.1, 0.1], wspace=0.03)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs_masw[0:2])

# ------------------------------------------------------------------------------
# Plot Vz
pcm_z = ax1.pcolormesh(r, tv, vzmasked, cmap='seismic', shading='auto', norm = norm)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_xlabel("Source-Receiver Distance (m)")
ax1.set_ylabel("Two-way Travel Time (s)")
# ax1.set_ylim(tv[7250], tv[0])
ax1.yaxis.set_inverted(True)
ax1.grid(True, color='black', alpha=0.15, linestyle = ':')

# ------------------------------------------------------------------------------
contour_levels = 200
contour = ax2.contourf(
    freqs, vs, image.T, 
    cmap='Greys', levels=contour_levels
)
ax2.set_xlabel(r'Wavenumber (m$^{-1}$)')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_ylim(0, 1000)
ax2.set_xlim(0, 50)
ax2.plot(1/period0, -45+velocity0*1000, '-', c = '#940f03', lw = 3, alpha = 0.5)
ax2.plot(1/period1, -45+velocity1*1000, '--',c = '#940f03', lw = 3, alpha = 0.5)
ax2.plot(1/period2, -45+velocity2*1000, '-.',c = '#940f03', lw = 3, alpha = 0.5)
ax2.plot(1/period3, -45+velocity3*1000, ':', c = '#940f03', lw = 3, alpha = 0.5)

# ------------------------------------------------------------------------------
cax = fig.add_subplot(gs_masw[3])
cbar = fig.colorbar(contour, cax=cax, orientation='horizontal')
cbar.set_label("Normalized Energy")
cax.tick_params(axis='x', labelsize=8)
cax.xaxis.set_ticks_position('bottom')
cax.set_xticks(cbar.ax.get_xticks())  # sync ticks just in case
cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
# ------------------------------------------------------------------------------
fig.subplots_adjust(
    left=0.09, right = 0.95, bottom = 0.3, top = 0.92, wspace=0.05
)
fig.savefig('surfacewave_shotgather_masw.png', dpi = 200)
fig.show()
