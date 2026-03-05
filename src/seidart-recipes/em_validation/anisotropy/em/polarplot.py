import numpy as np 


import pickle 




import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.routines.materials import rotator_zxz, ice_permittivity, snow_conductivity

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = ['strong_aniso.json', 'mod_aniso.json', 'aniso_ice.json']
 
receiver_file = 'receivers.xyz'

dt = np.zeros([3])
for ind in range(3):
    domain, material, seis, em = loadproject(
        project_file[ind], Domain(), Material(), Model(), Model()
    )
    dt[ind] = em.dt 

# ------------------------------------------------------------------------------
with open('strong_aniso-Ex-2.0-2.0-0.75.pkl', 'rb') as sf:
    strongx = pickle.load(sf)

with open('strong_aniso-Ey-2.0-2.0-0.75.pkl', 'rb') as sf:
    strongy = pickle.load(sf)

with open('strong_aniso-Ez-2.0-2.0-0.75.pkl', 'rb') as sf:
    strongz = pickle.load(sf)

with open('mod_aniso-Ex-2.0-2.0-0.75.pkl', 'rb') as sf:
    modx = pickle.load(sf)

with open('mod_aniso-Ey-2.0-2.0-0.75.pkl', 'rb') as sf:
    mody = pickle.load(sf)

with open('mod_aniso-Ez-2.0-2.0-0.75.pkl', 'rb') as sf:
    modz = pickle.load(sf)

with open('aniso_ice-Ex-2.0-2.0-0.75.pkl', 'rb') as sf:
    icex = pickle.load(sf)

with open('aniso_ice-Ey-2.0-2.0-0.75.pkl', 'rb') as sf:
    icey = pickle.load(sf)

with open('aniso_ice-Ez-2.0-2.0-0.75.pkl', 'rb') as sf:
    icez = pickle.load(sf)

exaggeration = 0.01

strongx.exaggeration = exaggeration
strongy.exaggeration = exaggeration
strongz.exaggeration = exaggeration

modx.exaggeration = exaggeration
mody.exaggeration = exaggeration
modz.exaggeration = exaggeration

icex.exaggeration = exaggeration
icey.exaggeration = exaggeration
icez.exaggeration = exaggeration



Exs = strongx.timeseries[:,-1]
Eys = strongy.timeseries[:,-1]
Ezs = strongz.timeseries[:,-1]

Exm = modx.timeseries[:,-1]
Eym = mody.timeseries[:,-1]
Ezm = modz.timeseries[:,-1]

Exi = icex.timeseries[:,-1]
Eyi = icey.timeseries[:,-1]
Ezi = icez.timeseries[:,-1]


# ------------------------------------------------------------------------------
angles = np.linspace(0, 2*np.pi, 360)

fs = 0.5/dt
fc = 1.1*em.f0
to_db = True
corrected = False

energy_xzs = compute_polar_energy(
    lowpass_filter(Exs, fc, fs[0]), lowpass_filter(Ezs, fc, fs[0]),
    dt[0], angles, to_db = to_db, corrected = corrected
)
energy_yzs = compute_polar_energy(
    lowpass_filter(Eys, fc, fs[0]), lowpass_filter(Ezs, fc, fs[0]), 
    dt[0], angles, to_db = to_db, corrected = corrected
)
energy_xys = compute_polar_energy(
    lowpass_filter(Exs, fc, fs[0]), lowpass_filter(Eys, fc, fs[0]), 
    dt[0], angles, to_db = to_db, corrected = corrected
)

energy_xzm = compute_polar_energy(
    lowpass_filter(Exm, fc, fs[1]), lowpass_filter(Ezm, fc, fs[1]), 
    dt[1], angles, to_db = to_db, corrected = corrected
)
energy_yzm = compute_polar_energy(
    lowpass_filter(Eym, fc, fs[1]), lowpass_filter(Ezm, fc, fs[1]), 
    dt[1], angles, to_db = to_db, corrected = corrected
)
energy_xym = compute_polar_energy(
    lowpass_filter(Exm, fc, fs[1]), lowpass_filter(Eym, fc, fs[1]), 
    dt[1], angles, to_db = to_db, corrected = corrected
)

energy_xzi = compute_polar_energy(
    lowpass_filter(Exi, fc, fs[2]), lowpass_filter(Ezi, fc, fs[2]), 
    dt[2], angles, to_db = to_db, corrected = corrected
)
energy_yzi = compute_polar_energy(
    lowpass_filter(Eyi, fc, fs[2]), lowpass_filter(Ezi, fc, fs[2]), 
    dt[2], angles, to_db = to_db, corrected = corrected
)
energy_xyi = compute_polar_energy(
    lowpass_filter(Exi, fc, fs[2]), lowpass_filter(Eyi, fc, fs[2]), 
    dt[2], angles, to_db = to_db, corrected = corrected
)

# ------------------------------------------------------------------------------
str_color = '#455c35'
mod_color = '#ff2b2b'
ice_color = '#727c9e'

fig_polar, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(2.5,6),
    subplot_kw={'projection':'polar'},
    constrained_layout=False
)

ax1.plot(angles-np.pi/2, energy_xys, color = str_color, alpha = 0.8, lw = 2)
ax1.plot(angles-np.pi/2, energy_xym, color = mod_color, alpha = 0.8, lw = 2)
ax1.plot(angles-np.pi/2, energy_xyi, color = ice_color, alpha = 0.8, lw = 2)

ax2.plot(angles-np.pi/2, energy_xzs, color = str_color, alpha = 0.8, lw = 2) 
ax2.plot(angles-np.pi/2, energy_xzm, color = mod_color, alpha = 0.8, lw = 2) 
ax2.plot(angles-np.pi/2, energy_xzi, color = ice_color, alpha = 0.8, lw = 2) 

ax3.plot(angles-np.pi/2, energy_yzs, color = str_color, alpha = 0.8, lw = 2)
ax3.plot(angles-np.pi/2, energy_yzm, color = mod_color, alpha = 0.8, lw = 2)
ax3.plot(angles-np.pi/2, energy_yzi, color = ice_color, alpha = 0.8, lw = 2)

for ax in (ax1, ax2, ax3):
    ax.grid(True, which = 'both', color = '#a3a3a3', linestyle = '--', lw = 0.5)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_rlim(-40, 0) 
    ax.set_rlabel_position(90) 

fig_polar.tight_layout()
fig_polar.show()
fig_polar.savefig('aniso_polar_plot.png', transparent = True, dpi = 400)


# ------------------------------------------------------------------------------
tss = np.stack(
    [lowpass_filter(Exs, fc, fs[0]), 
    lowpass_filter(Eys, fc, fs[0]), 
    lowpass_filter(Ezs, fc, fs[0])]
).T
tsi = np.stack(
    [lowpass_filter(Exi, fc, fs[2]), 
    lowpass_filter(Eyi, fc, fs[2]), 
    lowpass_filter(Ezi, fc, fs[2])]
).T
tsm = np.stack(
    [lowpass_filter(Exm, fc, fs[1]), 
    lowpass_filter(Eym, fc, fs[1]), 
    lowpass_filter(Ezm, fc, fs[1])]
).T
tvs = np.arange(0, em.time_steps) * dt[0]
tvm = np.arange(0, em.time_steps) * dt[1]
tvi = np.arange(0, em.time_steps) * dt[2]

R = rotator_zxz(np.array([np.pi/4, np.pi/2, 0]))
tss_rot = (R @ tss.T).T  
tsm_rot = (R @ tsm.T).T 
tsi_rot = (R @ tsi.T).T 

# Calculate arrival times for the ordinary and extra ordinary waves. 
no = np.sqrt([
    2,#strongx.electromag.permittivity_coefficients['e11'][0],
    2,#modx.electromag.permittivity_coefficients['e11'][0],
    3.1793,#icex.electromag.permittivity_coefficients['e11'][0]
])

ne = np.sqrt([
    10,#strongx.electromag.permittivity_coefficients['e33'][0],
    4,#modx.electromag.permittivity_coefficients['e33'][0],
    3.2049,#icex.electromag.permittivity_coefficients['e33'][0]
])
strongx.srcrcx_distance() 
dist = strongx.distances[-1] 

co = clight / no 
ce = clight / ne

ordinary_arrival_time = dist / co
extraordinary_arrival_time = dist / ce



xlims = (1.0e-07, 4.0e-07)

fig_ts, (axe, axo, axz) = plt.subplots(3, 1, figsize = (6,6), sharex = True)

axe.plot(tvm, tsm_rot[:,0], lw = 2, color = mod_color, alpha = 0.8)
axe.plot(tvi, tsi_rot[:,0], lw = 2, color = ice_color, alpha = 0.8)
axe.plot(tvs, tss_rot[:,0], lw = 2, color = str_color, alpha = 0.8)
axe.set_ylabel(r'E$_{∥}$ (V/m)', fontsize = 12)
axe.grid(True, linestyle = '--', alpha = 0.5)
axe.set_xlim(xlims)
for t in extraordinary_arrival_time:
    axe.axvline(float(t), color = '#3b000d', linestyle = '--', lw = 1.2)

axo.plot(tvm, tsm_rot[:,2], lw = 2, color = '#ff2b2b', alpha = 0.8)
axo.plot(tvi, tsi_rot[:,2], lw = 2, color = ice_color, alpha = 0.8)
axo.plot(tvs, tss_rot[:,2], lw = 2, color = str_color, alpha = 0.8)
axo.set_ylabel(r'E$_{⊥}$ (V/m)', fontsize = 12)
axo.grid(True, linestyle = '--', alpha = 0.5)
axo.set_xlim(xlims)
for t in ordinary_arrival_time:
    axo.axvline(float(t), color = '#3b0200', linestyle = '--', lw = 1.2)

axz.plot(tvm, tsm_rot[:,1], lw = 2, color = mod_color, alpha = 0.8)
axz.plot(tvi, tsi_rot[:,1], lw = 2, color = ice_color, alpha = 0.8)
axz.plot(tvs, tss_rot[:,1], lw = 2, color = str_color, alpha = 0.8)
axz.grid(True, linestyle = '--', alpha = 0.5)
axz.set_xlim(xlims)

axz.set_ylabel( r'E$_{z}$ (V/m)', fontsize = 12)
axz.set_xlabel("Time (s)", fontsize = 12)
fig_ts.tight_layout() 
fig_ts.show()
fig_ts.savefig('aniso_time_series.png', transparent = True, dpi = 400)
