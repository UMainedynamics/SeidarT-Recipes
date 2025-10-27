import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 
from scipy.ndimage import gaussian_filter


project_file1 = 'surface_waves/validation.json'
project_file2 = 'body_waves/bw_validation.json'

dom1, mat1, seis1, __ = loadproject(
    project_file1, Domain(), Material(), Model(), Model()
)

dom2, mat2, seis2, z__ = loadproject(
    project_file2, Domain(), Material(), Model(), Model()
)


z1, v1 = seis1.parameter_profile_1d(dom1, mat1, int(dom1.nx/2), parameter = 'velocity')
z1, rho1 = seis1.parameter_profile_1d(dom1, mat1, int(dom1.nx/2), parameter = 'density')
z2, v2 = seis2.parameter_profile_1d(dom2, mat2, int(dom2.nx/2), parameter = 'velocity')
z2, rho2 = seis2.parameter_profile_1d(dom2, mat2, int(dom2.nx/2), parameter = 'density')

# Depth should be below surface not relative to 0,0 in the model domain. 
z1 = z1 - 30

fig, ax = plt.subplots(figsize = (4, 10)) 

ax.plot(v1[:,0], z1, 'r-')
ax.plot(v1[:,2], z1, 'r--')
ax.plot(v2[:,0], z2, 'b-')
ax.plot(v2[:,2], z2, 'b--')
ax.set_ylim(z2.max(), z2.min() )
ax.set_ylabel('Depth (m)')
ax.set_xlabel(r'Velocity m$\cdot$s$^{-1}$')
fig.show()