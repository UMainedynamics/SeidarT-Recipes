import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 

from scipy.signal import correlate

# ------------------------------------------------------------------------------
def adaptive_scale_stack(trace1, trace2, t0, t1, dt):
    """
    Computes adaptive scaling factor alpha such that:
        stacked = trace1 - alpha * trace2
    minimizes energy in the time window [t0, t1].
    
    Parameters
    ----------
    trace1 : np.ndarray
        First time series (e.g., +45° source)
    trace2 : np.ndarray
        Second time series (e.g., -45° source)
    t0 : float
        Start time of the P-wave window (in seconds)
    t1 : float
        End time of the P-wave window (in seconds)
    dt : float
        Sampling interval (in seconds)
    
    Returns
    -------
    stacked : np.ndarray
        Resulting trace with minimized P-wave energy
    alpha : float
        Scaling factor applied to trace2
    """
    i0 = int(t0 / dt)
    i1 = int(t1 / dt)
    
    x = trace1[i0:i1]
    y = trace2[i0:i1]
    
    # Least-squares scaling factor to minimize ||x - alpha*y||
    num = np.dot(x, y)
    denom = np.dot(y, y) + 1e-12  # prevent division by zero
    alpha = num / denom
    
    stacked = trace1 - alpha * trace2
    return stacked, alpha


def matched_filter(trace, source_wavelet, normalize=True):
    """
    Applies a matched filter to a 1D trace using a known source waveform.
    
    Parameters
    ----------
    trace : np.ndarray
        1D array of the signal (e.g., receiver trace)
    source_wavelet : np.ndarray
        Known wavelet or template to match
    normalize : bool
        Normalize source wavelet to unit energy before filtering
    
    Returns
    -------
    mf_output : np.ndarray
        Matched filter output (same length as trace)
    """
    s = source_wavelet[::-1]  # time-reversed wavelet for correlation
    if normalize:
        s = s / (np.linalg.norm(s) + 1e-12)
    
    # Full cross-correlation, then center output to match trace length
    corr_full = correlate(trace, s, mode='full')
    start = (len(corr_full) - len(trace)) // 2
    mf_output = corr_full[start:start + len(trace)]
    return mf_output


# ------------------------------------------------------------------------------
## Initiate the model and domain objects
project_file = 'bw_validation.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

f = open('bw_validation-Vx-10.0-1.0-10.0.pkl', 'rb')
array_stack = pickle.load(f)
f.close() 

f = open('shot1.pkl', 'rb')
array1 = pickle.load(f) 
f.close() 

f = open('shot2.pkl', 'rb')
array2 = pickle.load(f) 
f.close() 

__, srcx1, srcy1, srcz1, srcfn1 = array1.seismic.sourcefunction(array1.seismic)
__, srcx2, srcy2, srcz2, srcfn2 = array2.seismic.sourcefunction(array2.seismic)

array_stack.exaggeration = 0.2
array1.exaggeration = 0.2 
array2.exaggeration = 0.2 
array_stack.agc_gain_window = 501 
array1.agc_gain_window = 501 
array2.agc_gain_window = 501 

ts1 = array1.timeseries[:,300]
ts2 = array2.timeseries[:,300]
n = array1.timeseries.shape[1]

array1.timeseries = array1.timeseries/array1.timeseries.max(axis = 0)
array2.timeseries = array2.timeseries/array2.timeseries.max(axis = 0)

for ind in range(n):
    mf1 = matched_filter(array1.timeseries[:,ind],  srcx1)
    mf2 = matched_filter(array2.timeseries[:,ind],  srcx1)
    array_stack.timeseries[:,ind] = mf1 + mf2
   
ts1_norm = ts1/ts1.max() 
ts2_norm = ts2/ts2.max() 
stack = ts1 + ts2 
stack = stack/stack.max() 
stack_norm = (ts1_norm + ts2_norm)/2
tv = np.arange(0, seis.time_steps) * seis.dt

gain = tv**1.5
fig, ax = plt.subplots() 
ax.plot(tv, mf1+mf2, 'b')
# ax.plot(tv, ts2_norm, 'r')
# ax.plot(tv, ts1_norm, 'b')
# ax.plot(tv, ts1_norm + ts2_norm, 'k--')
plt.show()

array_vz.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC')

array_vz.fk_analysis(
    2, 
    ntfft = 2*array_vz.timeseries.shape[0],
    nxfft = 2*array_vz.timeseries.shape[1],
    taper = 'both',
    wavenumber_limits = (-0.04,0.04), 
    frequency_limits = (0, 120), 
    use_filtered = True,
    colormap = 'inferno',
    mask_db = -160,
)
