import numpy as np
import os
from glob2 import glob 
import subprocess 
import pickle

from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

from scipy.signal.windows import hann
from scipy.signal import hilbert, correlate
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

with open('vx1.pkl', 'rb') as f:
    array_vx1 = pickle.load(f)

with open('vx2.pkl', 'rb') as f:
    array_vx2 = pickle.load(f)

with open('vy1.pkl', 'rb') as f:
    array_vy1 = pickle.load(f)

with open('vy2.pkl', 'rb') as f:
    array_vy2 = pickle.load(f)

with open('vz1.pkl', 'rb') as f:
    array_vz1 = pickle.load(f)

with open('vz2.pkl', 'rb') as f:
    array_vz2 = pickle.load(f)

project_file = 'quads.json'

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model() 
)

seis.sourcefunction(seis)
# ------------------------------------------------------------------------------
def find_zero_crossing(timevec, wavelet_vals, noise_region=slice(0, 100), k=3):
    # Estimate noise level from a noise-only segment
    sigma = np.std(wavelet_vals[noise_region])
    tol = k * sigma
    
    # Set values within tolerance to exactly zero
    filtered_vals = np.where(np.abs(wavelet_vals) < tol, 0, wavelet_vals)
    
    # Look for sign changes in the filtered signal
    sign_changes = np.where(np.diff(np.sign(filtered_vals)) != 0)[0]
    if len(sign_changes) == 0:
        return None  # No zero crossing found
    i0 = sign_changes[0]
    
    # Linear interpolation for a more accurate crossing time
    t0_approx = timevec[i0] + (timevec[i0+1] - timevec[i0]) * \
                (0 - filtered_vals[i0]) / (filtered_vals[i0+1] - filtered_vals[i0])
    return t0_approx


# ------------------------------------------------------------------------------
# compute the direction vector to get the christoffel matrix
direction = array_vx1.receiver_xyz - array_vx1.source_xyz
direction1 = direction[0,:] / np.linalg.norm(direction[0,:]) # isotropic
direction2 = direction[-1,:] / np.linalg.norm(direction[-1,:]) # isotropic with attenuation
direction = array_vx2.receiver_xyz - array_vx2.source_xyz 
direction3 = direction[0,:] / np.linalg.norm(direction[0,:]) # anisotropic
direction4 = direction[-1,:] / np.linalg.norm(direction[-1,:]) # anisotropic with attenuation

gamma, eigs, eigv, velocities1 = seis.get_christoffel_matrix(0, direction1)
gamma, eigs, eigv, velocities2 = seis.get_christoffel_matrix(1, direction2)
gamma, eigs, eigv, velocities3 = seis.get_christoffel_matrix(2, direction3)
gamma, eigs, eigv, velocities4 = seis.get_christoffel_matrix(3, direction4)

array_vx1.srcrcx_distance()
dist = array_vx1.distances[0]

tt1 = dist / velocities2
tt2 = dist / velocities3
tt3 = dist / velocities1
tt4 = dist / velocities4


N = domain.nmats
vx = np.zeros([seis.time_steps, N])

vx[:,0] = array_vx1.timeseries[:,0]
vx[:,1] = array_vx1.timeseries[:,-1]
vx[:,2] = array_vx2.timeseries[:,0]
vx[:,3] = array_vx2.timeseries[:,-1]

vy = np.zeros([seis.time_steps, 4])
vy[:,0] = array_vy1.timeseries[:,0]
vy[:,1] = array_vy1.timeseries[:,-1]
vy[:,2] = array_vy2.timeseries[:,0]
vy[:,3] = array_vy2.timeseries[:,-1]

vz = np.zeros([seis.time_steps, 4])
vz[:,0] = array_vz1.timeseries[:,0]
vz[:,1] = array_vz1.timeseries[:,-1]
vz[:,2] = array_vz2.timeseries[:,0]
vz[:,3] = array_vz2.timeseries[:,-1]


# do the rotation 
iso = np.array([vx[:,0], vy[:,0], vz[:,0]]).T
iso_a = np.array([vx[:,1], vy[:,1], vz[:,1]]).T
aniso = np.array([vx[:,2], vy[:,2], vz[:,2]]).T
aniso_a = np.array([vx[:,3], vy[:,3], vz[:,3]]).T

zrt_iso = rotate_to_zrt(iso, direction = direction1)
zrt_isoa = rotate_to_zrt(iso_a, direction = direction2)
zrt_aniso = rotate_to_zrt(aniso, direction = direction3)

timevector = np.arange(seis.time_steps) * seis.dt
t0_approx = find_zero_crossing(timevector, seis.sourcefunction_z, noise_region=slice(0, int(seis.f0)), k=3)
arrival_labels1 = {'P': r'$P_{iso}$', 'S1': r'$S_{1,iso}$', 'S2': r'$S_{2,iso}$'}
arrival_labels2 = {'P': r'$P_{aniso}$', 'S1': r'$S_{1,aniso}$', 'S2': r'$S_{2,aniso}$'}
arrival_colors1 = {'P': '#bfaa32', 'S1': '#bfaa32', 'S2': '#bfaa32'}
arrival_colors2 = {'P': '#9e775f', 'S1': '#9e775f', 'S2': '#9e775f'}

# fig_iso, ax_iso = plot_3c(
#     timevector, zrt_iso, 
#     data2 = zrt_isoa,
#     source_wavelet = None, #seis.sourcefunction_z,
#     arrivals1 = tt2+t0_approx,
#     arrival_labels1 = arrival_labels1, arrival_colors1 = arrival_colors1,
#     color_data1 = '#db3939', 
#     envelope_color1='#c9c9c9', 
#     show_legend = False
# )

fig_all, ax_all = plot_3c(
    timevector, zrt_iso, 
    data2 = zrt_isoa,
    data3 = zrt_aniso,
    data_label1 = 'Isotropic',
    data_label2 = 'Iso. w/ attenuation',
    data_label3 = 'Anisotropic',
    source_wavelet = None, #seis.sourcefunction_z,
    arrivals1 = tt1+t0_approx, arrivals2 = tt3+t0_approx,
    arrival_labels1 = arrival_labels1, arrival_labels2 = arrival_labels2,
    arrival_colors1 = arrival_colors1, arrival_colors2 = arrival_colors2,
    color_data1 = '#b02525', color_data2 = '#d46161', color_data3 = '#3345a6',
    envelope_color1='#c9c9c9', envelope_color2='#969696', envelope_color3 = '#969696',
    show_legend = False,
    xlims = (.006, .0245)
)
fig_all.savefig('seismic25d_anisotropy_attenuation.eps', transparent=True, dpi = 300)

