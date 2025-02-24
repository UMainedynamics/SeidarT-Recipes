import numpy as np
import os
from glob2 import glob 
import subprocess 

from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

from scipy.signal.windows import hann
import matplotlib.pyplot as plt 

def moving_window_phase_magnitude_tapered(
        A, B, fs, window_len, overlap=0.5, fft_len=None
    ):
    """
    Moving-window phase & magnitude response with optional tapering and zero-padding.
    """
    m, n_receivers = A.shape
    
    # Calculate step
    step = int(window_len * (1 - overlap))
    n_windows = max(0, (m - window_len) // step + 1)
    
    # Create a taper (Hann window)
    taper = hann(window_len)
    
    # FFT length
    if fft_len is None:
        fft_len = window_len  # no zero-padding by default
    else:
        # ensure fft_len >= window_len
        if fft_len < window_len:
            raise ValueError("fft_len must be >= window_len.")
    
    # Frequency array for rFFT of size fft_len
    f_arr = np.fft.rfftfreq(fft_len, d=1.0/fs)
    n_freqs = len(f_arr)
    
    mag = np.zeros((n_windows, n_freqs, n_receivers), dtype=np.float64)
    phase = np.zeros((n_windows, n_freqs, n_receivers), dtype=np.float64)
    
    eps = 1e-20
    
    for w in range(n_windows):
        start = w * step
        end = start + window_len
        
        for i in range(n_receivers):
            # Extract segment
            a_seg = A[start:end, i]
            b_seg = B[start:end, i]
            
            # Apply taper
            a_seg_tapered = a_seg * taper
            b_seg_tapered = b_seg * taper
            
            # Zero-padding if needed
            if fft_len > window_len:
                a_seg_pad = np.zeros(fft_len, dtype=np.float64)
                b_seg_pad = np.zeros(fft_len, dtype=np.float64)
                
                a_seg_pad[:window_len] = a_seg_tapered
                b_seg_pad[:window_len] = b_seg_tapered
            else:
                a_seg_pad = a_seg_tapered
                b_seg_pad = b_seg_tapered
            
            # FFT
            A_fft = np.fft.rfft(a_seg_pad)
            B_fft = np.fft.rfft(b_seg_pad)
            
            # Transfer function: B / A
            H = B_fft / (A_fft + eps)
            
            mag[w, :, i] = np.abs(H)
            phase[w, :, i] = np.angle(H)
    
    t_arr = (np.arange(n_windows) * step + window_len / 2.0) / fs
    
    return t_arr, f_arr, mag, phase

def plot_median_variance(
        f_arr,
        mag,
        phase,
        receiver_index=0,
        use_std=True,
        fmin=None,
        fmax=None,
        log_freq=False,
        ylim_mag=None,
        ylim_phase=None
    ):
    """
    Plots a statistical summary (median and shaded spread) of magnitude and phase
    vs. frequency for a given receiver. The data is aggregated across the
    time-window dimension.

    Parameters
    ----------
    f_arr : np.ndarray of shape (n_freqs,)
        Array of frequency bins.
    mag : np.ndarray of shape (n_windows, n_freqs, n_receivers)
        Magnitude response.
    phase : np.ndarray of shape (n_windows, n_freqs, n_receivers)
        Phase response (in radians).
    receiver_index : int, optional
        Which receiver to plot (default=0).
    use_std : bool, optional
        If True, shades +/- 1 standard deviation around the median.
        If False, shades +/- 1 variance around the median.
    fmin : float, optional
        Minimum frequency for x-axis limit. If None, uses f_arr.min().
    fmax : float, optional
        Maximum frequency for x-axis limit. If None, uses f_arr.max().
    log_freq : bool, optional
        If True, sets x-axis to a log scale (default=False).
    ylim_mag : tuple or list, optional
        (ymin, ymax) for the magnitude subplot. If None, it auto-scales.
    ylim_phase : tuple or list, optional
        (ymin, ymax) for the phase subplot. If None, it auto-scales.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : np.ndarray of matplotlib.axes.Axes
        Subplot axes [ax_mag, ax_phase].
    """    
    # Validate shapes
    n_windows, n_freqs, n_receivers = mag.shape
    if not (0 <= receiver_index < n_receivers):
        raise ValueError(f"receiver_index={receiver_index} out of range [0, {n_receivers-1}]")
    
    # Default frequency range if not provided
    if fmin is None:
        fmin = f_arr.min()
    if fmax is None:
        fmax = f_arr.max()
    
    # Extract data for the chosen receiver
    mag_data = mag[:, :, receiver_index]   # shape (n_windows, n_freqs)
    phase_data = phase[:, :, receiver_index]
    
    # Median across time windows
    mag_median = np.median(mag_data, axis=0)    # shape (n_freqs,)
    phase_median = np.median(phase_data, axis=0)
    
    # Spread (std or variance)
    if use_std:
        mag_spread = np.std(mag_data, axis=0)
        phase_spread = np.std(phase_data, axis=0)
        spread_label = "±1 Std. Dev."
    else:
        mag_spread = np.var(mag_data, axis=0)
        phase_spread = np.var(phase_data, axis=0)
        spread_label = "±1 Variance"
    
    # Create figure/subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
    ax_mag, ax_phase = axes
    
    # Plot Magnitude
    ax_mag.plot(f_arr, mag_median, color='blue', label='Median Magnitude')
    ax_mag.fill_between(
        f_arr,
        mag_median - mag_spread,
        mag_median + mag_spread,
        color='blue',
        alpha=0.2,
        label=spread_label
    )
    ax_mag.set_ylabel("Magnitude")
    ax_mag.set_title(f"Magnitude vs. Frequency (Receiver = {receiver_index})")
    ax_mag.legend(loc="best")
    
    # Vertical grid lines (dashed, light gray)
    # 'axis="x"' to show only vertical lines; 
    # 'which="major"' to only show grid at major ticks
    ax_mag.grid(True, which='major', axis='x', linestyle='--', color='gray', alpha=0.4)
    
    # Plot Phase
    ax_phase.plot(f_arr, phase_median, color='red', label='Median Phase')
    ax_phase.fill_between(
        f_arr,
        phase_median - phase_spread,
        phase_median + phase_spread,
        color='red',
        alpha=0.2,
        label=spread_label
    )
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (radians)")
    ax_phase.set_title(f"Phase vs. Frequency (Receiver = {receiver_index})")
    ax_phase.legend(loc="best")
    ax_phase.grid(True, which='major', axis='x', linestyle='--', color='gray', alpha=0.4)
    
    # Set frequency limits
    ax_mag.set_xlim(fmin, fmax)
    ax_phase.set_xlim(fmin, fmax)
    
    # Set y-limits for each subplot if provided
    if ylim_mag is not None:
        ax_mag.set_ylim(*ylim_mag)
    if ylim_phase is not None:
        ax_phase.set_ylim(*ylim_phase)
    
    # Log scale on the frequency axis if requested
    if log_freq:
        ax_mag.set_xscale('log')
        ax_phase.set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes

# ------------------------------------------------------------------------------

receiver_file1 = 'receivers1.xyz'
receiver_file2 = 'receivers2.xyz'
project_file = 'quads.json'

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model() 
)

seis.build(material, domain)
seis.kband_check(domain)
seis.run() 
# array_vx = Array('Vx', project_file, receiver_file)
array_vx1 = Array('Vx', project_file, receiver_file1)
array_vx2 = Array('Vx', project_file, receiver_file2)
array_vz1 = Array('Vz', project_file, receiver_file1)
array_vz2 = Array('Vz', project_file, receiver_file2)

frame_delay = 5 
frame_interval = 10 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
)

build_animation(
        project_file, 
        'Vx', frame_delay, frame_interval, alpha_value, 
)
   
# ------------------------------------------------------------------------------

fs = 1/seis.dt
window_length = 2**7 #seis.time_steps-1 #2**14


t_arrz, f_arrz, mag_isz, phase_isz = moving_window_phase_magnitude_tapered(
    array_vz1.timeseries[:,0],
    array_vz1.timeseries[:,-1],
    fs,
    window_len=window_length,
    overlap=0.5,
    fft_len=window_length*2  # zero-pad to 512 for improved freq resolution
)

rcx = 10
plot_median_variance(
    f_arrz, mag_isz, phase_isz, 
    receiver_index=rcx, use_std=True, 
    fmin = 1.0,
    fmax = 500,
    # ylim_mag = [-50, 50],
    # ylim_phase = [-0.5,0.75],
    log_freq = False
)

# ------------------------------------------------------------------------------
vz_iso = array_vz1.timeseries[:,0]
vz_iso_atten = array_vz1.timeseries[:,-1]

vz_aniso = array_vz2.timeseries[:,3]
vz_aniso_atten = array_vz2.timeseries[:,-4]

timevector = np.arange(seis.time_steps) * seis.dt

fig, ax = plt.subplots() 
ax.plot(timevector, vz_iso_atten, 'r')
ax.plot(timevector, vz_iso, 'b')
plt.show()

