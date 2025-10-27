import numpy as np 
import pandas as pd
import pickle
from glob2 import glob 
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.signal.windows import tukey


# ------------------------------------------------------------------------------\
def compute_transfer_function(A, B, dt_a, dt_b, fft_len=None, eps = 1e-20):
    """
    Moving-window phase & magnitude response with optional tapering and zero-padding.
    """
    fs_a = 1.0/dt_a 
    fs_b = 1.0/dt_b 
    
    if fs_a > fs_b:
        # Downsample A to match B
        ratio = Fraction(fs_b / fs_a).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator
        A = resample_poly(A, up, down)
        fs = fs_b
    elif fs_b > fs_a:
        # Downsample B to match A
        ratio = Fraction(fs_a / fs_b).limit_denominator(1000)
        up = ratio.numerator
        down = ratio.denominator
        B = resample_poly(B, up, down)
        fs = fs_a
    else:
        fs = fs_a  # already equal
    
    # Ensure same length now
    ma = len(A)
    mb = len(B)
    M = ma + mb
    a_pad = np.zeros(M, dtype=np.float64)
    b_pad = np.zeros(M, dtype=np.float64)
    
    # Taper and pad
    a_pad[:ma] = A * tukey(ma, alpha=0.25)
    b_pad[:mb] = B * tukey(mb, alpha=0.25)
    
    # FFT length
    if fft_len is None:
        fft_len = M
    
    # Frequency array
    f_arr = np.fft.rfftfreq(fft_len, d=1.0/fs)
    
    # FFT
    A_fft = np.fft.rfft(a_pad, n=fft_len)
    B_fft = np.fft.rfft(b_pad, n=fft_len)
    
    # Transfer function
    H = B_fft / (A_fft + eps)
    
    mag = np.abs(H)
    phase = np.angle(H, deg = True)
    
    return f_arr, mag, phase


def plot_transfer_function(
        f_arr, mag, phase, 
        fmin = 0.0, fmax = None, figsize=(10, 6)
    ):
    """
    Plot the magnitude and phase of a transfer function.
    
    Parameters:
        f_arr : array-like
            Frequency array (Hz)
        mag : array-like
            Magnitude response
        phase : array-like
            Phase response (radians)
        f_source : float
            Center frequency of interest (e.g., source freq)
        band : float
            +/- bandwidth around f_source to auto-zoom in Hz
        fmax : float
            Max frequency to show (Hz), if not using banded zoom
        log_mag : bool
            Whether to show magnitude in dB
        figsize : tuple
            Size of figure
    """
    phase_deg = np.rad2deg(phase)
    
    if fmax is None:
        fmax = f_arr.max() 
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.plot(f_arr, mag, lw=2)
    ax1.set_ylabel("Magnitude")
    
    ax1.grid(True, which='both', ls='--', alpha=0.4)
    
    ax2.plot(f_arr, phase_deg, lw=2, color='orange')
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    
    ax1.set_xlim(fmin, fmax)
    ax2.set_xlim(fmin, fmax)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# pklfiles = glob('*.pkl')

f = open('Vx-05.pkl', 'rb')
vx05 = pickle.load(f) 
f.close()
f = open('Vx-1.pkl', 'rb')
vx1 = pickle.load(f) 
f.close()
f = open('Vx-2.pkl', 'rb')
vx2 = pickle.load(f)
f.close()
f = open('Vx-3.pkl', 'rb')
vx3 = pickle.load(f)
f.close()
f = open('Vx-4.pkl', 'rb')
vx4 = pickle.load(f)
f.close()
f = open('Vx-6.pkl', 'rb')
vx6 = pickle.load(f)
f.close()

f = open('Vy-05.pkl', 'rb')
vy05 = pickle.load(f) 
f.close()
f = open('Vy-1.pkl', 'rb')
vy1 = pickle.load(f) 
f.close()
f = open('Vy-2.pkl', 'rb')
vy2 = pickle.load(f)
f.close()
f = open('Vy-3.pkl', 'rb')
vy3 = pickle.load(f)
f.close()
f = open('Vy-4.pkl', 'rb')
vy4 = pickle.load(f)
f.close()
f = open('Vy-6.pkl', 'rb')
vy6 = pickle.load(f)
f.close()

f = open('Vz-05.pkl', 'rb')
vz05 = pickle.load(f) 
f.close()
f = open('Vz-1.pkl', 'rb')
vz1 = pickle.load(f) 
f.close()
f = open('Vz-2.pkl', 'rb')
vz2 = pickle.load(f)
f.close()
f = open('Vz-3.pkl', 'rb')
vz3 = pickle.load(f)
f.close()
f = open('Vz-4.pkl', 'rb')
vz4 = pickle.load(f)
f.close()
f = open('Vz-6.pkl', 'rb')
vz6 = pickle.load(f)
f.close()


vz6.get_christoffel_equations

tv05 = np.arange(0, vx05.timeseries.shape[0]) * vx05.seismic.dt
tv1 = np.arange(0, vx1.timeseries.shape[0]) * vx1.seismic.dt
tv2 = np.arange(0, vx2.timeseries.shape[0]) * vx2.seismic.dt
tv3 = np.arange(0, vx3.timeseries.shape[0]) * vx3.seismic.dt
tv4 = np.arange(0, vx4.timeseries.shape[0]) * vx4.seismic.dt
tv6 = np.arange(0, vx6.timeseries.shape[0]) * vx6.seismic.dt

tsx05=vx05.timeseries[:,1]
tsx1 = vx1.timeseries[:,1]
tsx2 = vx2.timeseries[:,1]
tsx3 = vx3.timeseries[:,1]
tsx4 = vx4.timeseries[:,1]
tsx6 = vx6.timeseries[:,1]

tsy05=vy05.timeseries[:,1]
tsy1 = vy1.timeseries[:,1]
tsy2 = vy2.timeseries[:,1]
tsy3 = vy3.timeseries[:,1]
tsy4 = vy4.timeseries[:,1]
tsy6 = vy6.timeseries[:,1]

tsz05=vz05.timeseries[:,1]
tsz1 = vz1.timeseries[:,1]
tsz2 = vz2.timeseries[:,1]
tsz3 = vz3.timeseries[:,1]
tsz4 = vz4.timeseries[:,1]
tsz6 = vz6.timeseries[:,1]

tsx05 = tsx05[tv05 < 0.2]
tsy05 = tsy05[tv05 < 0.2]
tsz05 = tsz05[tv05 < 0.2]
tv05 = tv05[tv05 < 0.2]

tsx1 = tsx1[tv1 < 0.2]
tsy1 = tsy1[tv1 < 0.2]
tsz1 = tsz1[tv1 < 0.2]
tv1 = tv1[tv1 < 0.2] 

tsx2 = tsx2[tv2 < 0.2]
tsy2 = tsy2[tv2 < 0.2]
tsz2 = tsz2[tv2 < 0.2]
tv2 = tv2[tv2 < 0.2] 


tsx3 = tsx3[tv3 < 0.2]
tsy3 = tsy3[tv3 < 0.2]
tsz3 = tsz3[tv3 < 0.2]
tv3 = tv3[tv3 < 0.2] 

tsx4 = tsx4[tv4 < 0.2]
tsy4 = tsy4[tv4 < 0.2]
tsz4 = tsz4[tv4 < 0.2]
tv4 = tv4[tv4 < 0.2] 

tsx6 = tsx6[tv6 < 0.2]
tsy6 = tsy6[tv6 < 0.2]
tsz6 = tsz6[tv6 < 0.2]
tv6 = tv6[tv6 < 0.2] 

# ------------------------------------------------------------------------------
# Plot time series 



fmin = vx1.seismic.f0/2
fmax = vx1.seismic.f0*2
f1x, mag1x, phase1x = compute_transfer_function(tsx05, tsx1, vx05.seismic.dt, vx1.seismic.dt)#, 
f2x, mag2x, phase2x = compute_transfer_function(tsx05, tsx2, vx05.seismic.dt, vx2.seismic.dt)#, 
f3x, mag3x, phase3x = compute_transfer_function(tsx05, tsx3, vx05.seismic.dt, vx3.seismic.dt)#, 
f4x, mag4x, phase4x = compute_transfer_function(tsx05, tsx4, vx05.seismic.dt, vx4.seismic.dt)
f6x, mag6x, phase6x = compute_transfer_function(tsx05, tsx6, vx05.seismic.dt, vx6.seismic.dt)

f1y, mag1y, phase1y = compute_transfer_function(tsy05, tsy1, vy05.seismic.dt, vy1.seismic.dt)#, 
f2y, mag2y, phase2y = compute_transfer_function(tsy05, tsy2, vy05.seismic.dt, vy2.seismic.dt)#, 
f3y, mag3y, phase3y = compute_transfer_function(tsy05, tsy3, vy05.seismic.dt, vy3.seismic.dt)#, 
f4y, mag4y, phase4y = compute_transfer_function(tsy05, tsy4, vy05.seismic.dt, vy4.seismic.dt)
f6y, mag6y, phase6y = compute_transfer_function(tsy05, tsy6, vy05.seismic.dt, vy6.seismic.dt)

f1z, mag1z, phase1z = compute_transfer_function(tsz05, tsz1, vz05.seismic.dt, vz1.seismic.dt)#, 
f2z, mag2z, phase2z = compute_transfer_function(tsz05, tsz2, vz05.seismic.dt, vz2.seismic.dt)#, 
f3z, mag3z, phase3z = compute_transfer_function(tsz05, tsz3, vz05.seismic.dt, vz3.seismic.dt)#, 
f4z, mag4z, phase4z = compute_transfer_function(tsz05, tsz4, vz05.seismic.dt, vz4.seismic.dt)
f6z, mag6z, phase6z = compute_transfer_function(tsz05, tsz6, vz05.seismic.dt, vz6.seismic.dt)


# plot_transfer_function(f2z, mag2z, phase2z, fmin = fmin, fmax = fmax)
# plot_transfer_function(f3z, mag3z, phase3z, fmin = fmin, fmax = fmax)


# ------------------------------------------------------------------------------
fig_mag, (ax_magx, ax_magy, ax_magz) = plt.subplots(3, 1, figsize = (8,8) )
ax_magx.plot(f1x, 20*np.log10(mag1x), c = 'r', lw = 2)
ax_magx.plot(f2x, 20*np.log10(mag2x), c = '#02b32b', lw = 2) 
ax_magx.plot(f3x, 20*np.log10(mag3x), c = '#04002e', lw = 2) 
ax_magx.plot(f4x, 20*np.log10(mag4x), c = '#ab1f94', lw = 2) 
ax_magx.plot(f6x, 20*np.log10(mag6x), c = '#9c860b', lw = 2) 
ax_magx.set_xlim(fmin, fmax) 
# ax_magx.set_ylim(-0.05, 0.22)
ax_magx.set_ylabel(r"Gain$_x$ (dB)")
ax_magx.set_xlabel("Frequency (Hz)")
ax_magx.grid(True, which='both', ls='--', alpha=0.4)

ax_magy.plot(f1y, 20*np.log10(mag1y), c = 'r', lw = 2)
ax_magy.plot(f2y, 20*np.log10(mag2y), c = '#02b32b', lw = 2) 
ax_magy.plot(f3y, 20*np.log10(mag3y), c = '#04002e', lw = 2) 
ax_magy.plot(f4y, 20*np.log10(mag4y), c = '#ab1f94', lw = 2) 
ax_magy.plot(f6y, 20*np.log10(mag6y), c = '#9c860b', lw = 2) 
ax_magy.set_xlim(fmin, fmax) 
# ax_magy.set_ylim(-0.1, 0.18)
ax_magy.set_ylabel(r"Gain$_y$ (dB)")
ax_magy.set_xlabel("Frequency (Hz)")
ax_magy.grid(True, which='both', ls='--', alpha=0.4)

ax_magz.plot(f1z, 20*np.log10(mag1z), c = 'r', lw = 2)
ax_magz.plot(f2z, 20*np.log10(mag2z), c = '#02b32b', lw = 2) 
ax_magz.plot(f3z, 20*np.log10(mag3z), c = '#04002e', lw = 2) 
ax_magz.plot(f4z, 20*np.log10(mag4z), c = '#ab1f94', lw = 2) 
ax_magz.plot(f6z, 20*np.log10(mag6z), c = '#9c860b', lw = 2) 
ax_magz.set_xlim(fmin, fmax) 
# ax_magz.set_ylim(-0.05, 0.18)
ax_magz.set_ylabel(r"Gain$_z$ (dB)")
ax_magz.set_xlabel("Frequency (Hz)")
ax_magz.grid(True, which='both', ls='--', alpha=0.4)

fig_mag.show() 



# ------------------------------------------------------------------------------
fig_phase, (ax_phasex, ax_phasey, ax_phasez) = plt.subplots(3, 1, figsize = (8,8) )
ax_phasex.plot(f1x, phase1x, c = 'r', lw = 2)
ax_phasex.plot(f2x, phase2x, c = '#02b32b', lw = 2) 
ax_phasex.plot(f3x, phase3x, c = '#04002e', lw = 2) 
ax_phasex.plot(f4x, phase4x, c = '#ab1f94', lw = 2) 
ax_phasex.plot(f6x, phase6x, c = '#9c860b', lw = 2) 
ax_phasex.set_xlim(fmin, fmax) 
# ax_phasex.set_ylim(-0.05, 0.22)
ax_phasex.set_ylabel(r"Phase$_x$ (deg)")
ax_phasex.set_xlabel("Frequency (Hz)")
ax_phasex.grid(True, which='both', ls='--', alpha=0.4)

ax_phasey.plot(f1z, phase1y, c = 'r', lw = 2)
ax_phasey.plot(f2y, phase2y, c = '#02b32b', lw = 2) 
ax_phasey.plot(f3y, phase3y, c = '#04002e', lw = 2) 
ax_phasey.plot(f4y, phase4y, c = '#ab1f94', lw = 2) 
ax_phasey.plot(f6y, phase6y, c = '#9c860b', lw = 2) 
ax_phasey.set_xlim(fmin, fmax) 
# ax_phasey.set_ylim(-0.1, 0.18)
ax_phasey.set_ylabel(r"Phase$_y$ (deg)")
ax_phasey.set_xlabel("Frequency (Hz)")
ax_phasey.grid(True, which='both', ls='--', alpha=0.4)

ax_phasez.plot(f1z, phase1z, c = 'r', lw = 2)
ax_phasez.plot(f2z, phase2z, c = '#02b32b', lw = 2) 
ax_phasez.plot(f3z, phase3z, c = '#04002e', lw = 2) 
ax_phasez.plot(f4z, phase4z, c = '#ab1f94', lw = 2) 
ax_phasez.plot(f6z, phase6z, c = '#9c860b', lw = 2) 
ax_phasez.set_xlim(fmin, fmax) 
# ax_phasez.set_ylim(-0.05, 0.18)
ax_phasez.set_ylabel(r"Phase$_z$ (deg)")
ax_phasez.set_xlabel("Frequency (Hz)")
ax_phasez.grid(True, which='both', ls='--', alpha=0.4)

fig_phase.show() 

# ax_phase.plot(f12, phase12, c = '#02b32b') 
# ax_phase.plot(f13, phase13, c = '#04002e') 
# ax_phase.plot(f14, phase14, c = '#ab1f94') 
# ax_phase.plot(f16, phase16, c = '#9c860b') 
# ax_phase.set_xlim(fmin, fmax) 
# ax_phase.set_ylabel("Phase (degrees)")
# ax_phase.set_xlabel("Frequency (Hz)")
# ax_phase.grid(True, which='both', ls='--', alpha=0.4)

# ------------------------------------------------------------------------------
norm05x = np.abs(tsx05[tv05<0.16]).max()
norm1x = np.abs(tsx1[tv1<0.16]).max()
norm2x = np.abs(tsx2[tv2<0.16]).max()
norm3x = np.abs(tsx3[tv3<0.16]).max()
norm4x = np.abs(tsx4[tv4<0.16]).max()
norm6x = np.abs(tsx6[tv6<0.16]).max()

norm05y = np.abs(tsy05[tv05<0.16]).max()
norm1y = np.abs(tsy1[tv1<0.16]).max()
norm2y = np.abs(tsy2[tv2<0.16]).max()
norm3y = np.abs(tsy3[tv3<0.16]).max()
norm4y = np.abs(tsy4[tv4<0.16]).max()
norm6y = np.abs(tsy6[tv6<0.16]).max()

norm05z = np.abs(tsz05[tv05<0.16]).max()
norm1z = np.abs(tsz1[tv1<0.16]).max()
norm2z = np.abs(tsz2[tv2<0.16]).max()
norm3z = np.abs(tsz3[tv3<0.16]).max()
norm4z = np.abs(tsz4[tv4<0.16]).max()
norm6z = np.abs(tsz6[tv6<0.16]).max()


fig, (axx, axy, axz) = plt.subplots(3, 1, figsize = (8,6) )

axx.plot(tv05, tsx05/norm05x, '-', lw = 2)
axx.plot(tv1, tsx1/norm1x, c='#524042', lw = 3, alpha = 0.5)
axx.plot(tv2, tsx2/norm2x, c='#524042', lw = 2.75, alpha = 0.6)
axx.plot(tv3, tsx3/norm3x, c='#524042', lw = 2.5, alpha = 0.7)
axx.plot(tv4, tsx4/norm4x, c='#524042', lw = 2.25, alpha = 0.8)
axx.plot(tv6, tsx6/norm6x, c='#524042', lw = 2, alpha = 0.9)
axx.set_xlim(0.12, .194)
axx.set_ylim(-2.5, 2.5)
axx.set_ylabel(r'V$_x$')

axy.plot(tv05, tsy05/np.abs(tsy05).std(), '-', lw = 2)
axy.plot(tv1, tsy1/norm1y, c='#524042', lw = 3, alpha = 0.5)
axy.plot(tv2, tsy2/norm2y, c='#524042', lw = 2.75, alpha = 0.6)
axy.plot(tv3, tsy3/norm3y, c='#524042', lw = 2.5, alpha = 0.7)
axy.plot(tv4, tsy4/norm4y, c='#524042', lw = 2.25, alpha = 0.8)
axy.plot(tv6, tsy6/norm6y, c='#524042', lw = 2, alpha = 0.9)
axy.set_xlim(0.12, .194)
axy.set_ylim(-2.5, 2.5)
axy.set_ylabel(r'V$_y$')

axz.plot(tv1, tsz1/norm1z, c='#524042', lw = 3, alpha = 0.5)
axz.plot(tv2, tsz2/norm2z, c='#524042', lw = 2.75, alpha = 0.6)
axz.plot(tv3, tsz3/norm3z, c='#524042', lw = 2.5, alpha = 0.7)
axz.plot(tv4, tsz4/norm4z, c='#524042', lw = 2.25, alpha = 0.8)
axz.plot(tv6, tsz6/norm6z, c='#524042', lw = 2, alpha = 0.9)
axz.plot(tv05, tsz05/norm05z, '-', lw = 4, alpha = 0.4)
axz.set_xlim(0.12, .194)
axz.set_ylim(-2.5, 2.5)
axz.set_xlabel('Time (s)')
axz.set_ylabel(r'V$_z$')

fig.show()