import numpy as np 
import pandas as pd
import pickle
from glob2 import glob 
from fractions import Fraction
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from scipy.signal.windows import tukey


# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# There is a lot of energy coupling in the shear modes so we don't need the 
# individual horizontal models just the 3C. 
f = open('Vx05p.pkl', 'rb')
vx05p = pickle.load(f) 
f.close()
# f = open('Vx05s.pkl', 'rb')
# vx05s = pickle.load(f) 
# f.close()
f = open('Vy05p.pkl', 'rb')
vy05p = pickle.load(f)
f.close()
# f = open('Vy05s.pkl', 'rb')
# vy05s = pickle.load(f)
# f.close()
f = open('Vz05p.pkl', 'rb')
vz05p = pickle.load(f)
f.close()
# f = open('Vz05s.pkl', 'rb')
# vz05s = pickle.load(f)
# f.close()

f = open('Vx1p.pkl', 'rb')
vx1p = pickle.load(f) 
f.close()
# f = open('Vx1s.pkl', 'rb')
# vx1s = pickle.load(f) 
# f.close()
f = open('Vy1p.pkl', 'rb')
vy1p = pickle.load(f)
f.close()
# f = open('Vy1s.pkl', 'rb')
# vy1s = pickle.load(f)
# f.close()
f = open('Vz1p.pkl', 'rb')
vz1p = pickle.load(f)
f.close()
# f = open('Vz1s.pkl', 'rb')
# vz1s = pickle.load(f)
# f.close()


f = open('Vx2p.pkl', 'rb')
vx2p = pickle.load(f) 
f.close()
# f = open('Vx2s.pkl', 'rb')
# vx2s = pickle.load(f) 
# f.close()
f = open('Vy2p.pkl', 'rb')
vy2p = pickle.load(f)
f.close()
# f = open('Vy2s.pkl', 'rb')
# vy2s = pickle.load(f)
# f.close()
f = open('Vz2p.pkl', 'rb')
vz2p = pickle.load(f)
f.close()
# f = open('Vz2s.pkl', 'rb')
# vz2s = pickle.load(f)
# f.close()

f = open('Vx6p.pkl', 'rb')
vx6p = pickle.load(f) 
f.close()
# f = open('Vx6s.pkl', 'rb')
# vx6s = pickle.load(f) 
# f.close()
f = open('Vy6p.pkl', 'rb')
vy6p = pickle.load(f)
f.close()
# f = open('Vy6s.pkl', 'rb')
# vy6s = pickle.load(f)
# f.close()
f = open('Vz6p.pkl', 'rb')
vz6p = pickle.load(f)
f.close()
# f = open('Vz6s.pkl', 'rb')
# vz6s = pickle.load(f)
# f.close()

# ------------------------------------------------------------------------------
__, __, __, (vs1, vs2, vp) = vz1p.seismic.get_christoffel_matrix(0, 'z')
vz1p.srcrcx_distance() 
distance = vz1p.distances[1]
tts1 = distance/vs1
tts2 = distance/vs2
ttp = distance/vp 

# vz6s.get_christoffel_equations

tv05 =np.arange(0, vx05p.timeseries.shape[0]) * vx05p.seismic.dt
tv1 = np.arange(0, vx1p.timeseries.shape[0]) * vx1p.seismic.dt
tv2 = np.arange(0, vx2p.timeseries.shape[0]) * vx2p.seismic.dt
# tv3 = np.arange(0, vx3.timeseries.shape[0]) * vx3.seismic.dt
# tv4 = np.arange(0, vx4.timeseries.shape[0]) * vx4.seismic.dt
tv6 = np.arange(0, vx6p.timeseries.shape[0]) * vx6p.seismic.dt

tsx05p = vx05p.timeseries[:,1]
# tsx05s = vx05s.timeseries[:,1]
tsy05p = vy05p.timeseries[:,1]
# tsy05s = vy05s.timeseries[:,1]
tsz05p = vz05p.timeseries[:,1]
# tsz05s = vz05s.timeseries[:,1]

tsx1p = vx1p.timeseries[:,1]
# tsx1s = vx1s.timeseries[:,1]
tsy1p = vy1p.timeseries[:,1]
# tsy1s = vy1s.timeseries[:,1]
tsz1p = vz1p.timeseries[:,1]
# tsz1s = vz1s.timeseries[:,1]

tsx2p = vx2p.timeseries[:,1]
# tsx2s = vx2s.timeseries[:,1]
tsy2p = vy2p.timeseries[:,1]
# tsy2s = vy2s.timeseries[:,1]
tsz2p = vz2p.timeseries[:,1]
# tsz2s = vz2s.timeseries[:,1]

tsx6p = vx6p.timeseries[:,1]
# tsx6s = vx6s.timeseries[:,1]
tsy6p = vy6p.timeseries[:,1]
# tsy6s = vy6s.timeseries[:,1]
tsz6p = vz6p.timeseries[:,1]
# tsz6s = vz6s.timeseries[:,1]

# ------------------------------------------------------------------------------
# Plot time series 

fig, (axx, axy, axz) = plt.subplots(nrows=3, figsize=(10,8) )
# axx.plot(tv05, tsx05p, lw=2, c='#9138a1')
axx.plot(tv2, tsx2p, lw=2, c='#692525')
axy.plot(tv2, tsy2p, lw=2, c='#692525')
axz.plot(tv2, tsz2p, lw=2, c='#692525')
# axx.plot(tv2, tsx2p, lw=2, c='#375239') 
# axx.plot(tv6, tsx6p, lw=2, c='#253069')
# axx.set_xlim(0, 0.20)
fig.show() 


fmin = vx1.seismic.f0/2
fmax = vx1.seismic.f0*2
f1x, mag1x, phase1x = compute_transfer_function(tsx05, tsx1, tsx1.seismic.dt, tsx1.seismic.dt)#, 
f2x, mag2x, phase2x = compute_transfer_function(tsx05, tsx2, tsx1.seismic.dt, tsx2.seismic.dt)#, 
f3x, mag3x, phase3x = compute_transfer_function(tsx05, tsx3, tsx1.seismic.dt, tsx3.seismic.dt)#, 
f4x, mag4x, phase4x = compute_transfer_function(tsx05, tsx4, tsx1.seismic.dt, tsx4.seismic.dt)
f6x, mag6x, phase6x = compute_transfer_function(tsx05, tsx6, tsx1.seismic.dt, tsx6.seismic.dt)

f1y, mag1y, phase1y = compute_transfer_function(tsy05, tsy1, tsy1.seismic.dt, tsy1.seismic.dt)#, 
f2y, mag2y, phase2y = compute_transfer_function(tsy05, tsy2, tsy1.seismic.dt, tsy2.seismic.dt)#, 
f3y, mag3y, phase3y = compute_transfer_function(tsy05, tsy3, tsy1.seismic.dt, tsy3.seismic.dt)#, 
f4y, mag4y, phase4y = compute_transfer_function(tsy05, tsy4, tsy1.seismic.dt, tsy4.seismic.dt)
f6y, mag6y, phase6y = compute_transfer_function(tsy05, tsy6, tsy1.seismic.dt, tsy6.seismic.dt)

f1z, mag1z, phase1z = compute_transfer_function(tsz05, tsz1, tsz1.seismic.dt, tsz1.seismic.dt)#, 
f2z, mag2z, phase2z = compute_transfer_function(tsz05, tsz2, tsz1.seismic.dt, tsz2.seismic.dt)#, 
f3z, mag3z, phase3z = compute_transfer_function(tsz05, tsz3, tsz1.seismic.dt, tsz3.seismic.dt)#, 
f4z, mag4z, phase4z = compute_transfer_function(tsz05, tsz4, tsz1.seismic.dt, tsz4.seismic.dt)
f6z, mag6z, phase6z = compute_transfer_function(tsz05, tsz6, tsz1.seismic.dt, tsz6.seismic.dt)


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