import numpy as np
import pandas as pd
from glob2 import glob 
import pickle

import matplotlib.pyplot as plt



# ------------------------------------------------------------------------------
import numpy as np
from scipy.signal.windows import hann

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

# ------------------------------------------------------------------------------
fn_iso_x = glob('ex.isotropic.*.csv')
fn_weak_x = glob('ex.weak_vertical*.*.csv')
fn_strong_x = glob('ex.strong_vertical*.*.csv')

fn_iso_z = glob('ez.isotropic.*.csv')
fn_weak_z = glob('ez.weak_vertical*.*.csv')
fn_strong_z = glob('ez.strong_vertical*.*.csv')

n = len(fn_iso_x)

# stack each set of receivers for a given frequency
iso_x = pd.read_csv(fn_iso_x[0], header = None)
weak_x = pd.read_csv(fn_weak_x[0], header = None)
strong_x = pd.read_csv(fn_strong_x[0], header = None)

iso_z = pd.read_csv(fn_iso_z[0], header = None)
weak_z = pd.read_csv(fn_weak_z[0], header = None)
strong_z = pd.read_csv(fn_strong_z[0], header = None)

for ind in range(1,n):
    iso_x = iso_x + pd.read_csv(fn_iso_x[ind], header = None)
    weak_x = weak_x + pd.read_csv(fn_weak_x[ind], header = None)
    strong_x = strong_x + pd.read_csv(fn_strong_x[ind], header = None)
    iso_z = iso_z + pd.read_csv(fn_iso_z[ind], header = None)
    weak_z = weak_z + pd.read_csv(fn_weak_z[ind], header = None)
    strong_z = strong_z + pd.read_csv(fn_strong_z[ind], header = None)


# Now we need to attach the waveforms to the meta data. Each array object has 
# the sourcefunction attached to it, but the domain and material objects should 
# be the same for the 3 categories 
iso_meta = pickle.load( open(glob('ex.isotropic.*.pkl')[0], 'rb') )
weak_meta = pickle.load( open(glob('ex.weak_vertical*.*.pkl')[0], 'rb') )
strong_meta = pickle.load( open(glob('ex.strong_vertical*.*.pkl')[0], 'rb') )

fs = 1/iso_meta.electromag.dt
window_length = 2**8


t_arrx, f_arrx, mag_isx, phase_isx = moving_window_phase_magnitude_tapered(
    iso_x.to_numpy(),
    strong_x.to_numpy(),
    fs,
    window_len=window_length,
    overlap=0.9,
    fft_len=window_length*2  # zero-pad to 512 for improved freq resolution
)

t_arrz, f_arrz, mag_isz, phase_isz = moving_window_phase_magnitude_tapered(
    iso_z.to_numpy(),
    strong_z.to_numpy(),
    fs,
    window_len=window_length,
    overlap=0.9,
    fft_len=window_length*2  # zero-pad to 512 for improved freq resolution
)

t_arrx, f_arrx, mag_isx, phase_isx = moving_window_phase_magnitude_tapered(
    iso_x.to_numpy(),
    weak_x.to_numpy(),
    fs,
    window_len=window_length,
    overlap=0.9,
    fft_len=window_length*2  # zero-pad to 512 for improved freq resolution
)

t_arrz, f_arrz, mag_isz, phase_isz = moving_window_phase_magnitude_tapered(
    iso_z.to_numpy(),
    weak_z.to_numpy(),
    fs,
    window_len=window_length,
    overlap=0.9,
    fft_len=window_length*2  # zero-pad to 512 for improved freq resolution
)
# ------------------------------------------------------------------------------

def plot_timewindow_lines(
        f_arr, mag, phase, receiver_index=0, fmin = None , fmax = None
    ):
    """
    Plots line graphs for each time window in 'mag' and 'phase' arrays
    for a single receiver.

    Parameters
    ----------
    f_arr : np.ndarray, shape (n_freqs,)
        Frequency axis for plotting.
    mag : np.ndarray, shape (n_windows, n_freqs, n_receivers)
        Magnitude data.
    phase : np.ndarray, shape (n_windows, n_freqs, n_receivers)
        Phase data (in radians).
    receiver_index : int, optional
        Which receiver to plot. Default is 0.
    """
    n_windows, n_freqs, n_receivers = mag.shape
    
    # Make sure receiver_index is valid
    if receiver_index < 0 or receiver_index >= n_receivers:
        raise ValueError(f"receiver_index={receiver_index} is out of range [0, {n_receivers-1}].")
    
    if not fmin:
        fmin = f_arr.min() 
    
    if not fmax:
        fmax = f_arr.max()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    ax_mag, ax_phase = axes
    
    for w in range(n_windows):
        # Extract the line data for the current window and receiver
        mag_line = mag[w, :, receiver_index]
        phase_line = phase[w, :, receiver_index]
        
        # Plot magnitude on top subplot
        ax_mag.plot(f_arr, mag_line, label=f"Window {w}")
        
        # Plot phase on bottom subplot
        ax_phase.plot(f_arr, phase_line, label=f"Window {w}")
    
    # Aesthetics
    ax_mag.set_ylabel("Magnitude")
    ax_mag.set_title(f"Magnitude vs. Frequency (Receiver = {receiver_index})")
    # If you want a legend, uncomment the line below (could be large if many windows)
    # ax_mag.legend(loc='best')
    
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.set_ylabel("Phase (radians)")
    ax_phase.set_title(f"Phase vs. Frequency (Receiver = {receiver_index})")
    # ax_phase.legend(loc='best')  # Also optional
    
    plt.tight_layout()
    plt.show()


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

rcx = 60 
mag = mag_is[:,:,rcx] 
phase = phase_is[:,:,rcx]

# plot_timewindow_lines(f_arr, mag_is, phase_is, receiver_index = rcx)

plot_median_variance(
    f_arrx, mag_isx, phase_isx, 
    receiver_index=rcx, use_std=True, 
    fmin = 1.0e7,
    fmax = 1.0e9,
    ylim_mag = [0.5, 1.5],
    ylim_phase = [-0.5,0.75],
    log_freq = False
)
plot_median_variance(
    f_arrz, mag_isz, phase_isz, 
    receiver_index=rcx, use_std=True, 
    fmin = 1.0e7
    fmax = 1.0e9,
    ylim_mag = [0.5, 1.5],
    ylim_phase = [-0.5,0.75],
    log_freq = False
)
# plot_median_variance(f_arr, mag_iw, phase_iw, receiver_index=rcx, use_std=True, fmax = 1.0e9)
