import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import copy 
import seaborn as sns
import os 
from glob2 import glob

from seidart.routines import prjrun, sourcefunction
from seidart.routines.arraybuild import Array 
from seidart.routines.definitions import * 
from seidart.routines.fabricsynth import Fabric

# ------------------------------------------------------------------------------
# 
generate_fabric = False 
if generate_fabric:
    import fabric_gen

# ------------------------------------------------------------------------------
# Run a set of seismic simulations for the different fabrics generated above

# Load the project 
prjfile = 'split_material.prj'
domain, material, seismic,__ = prjrun.domain_initialization(prjfile)
timevec, fx, fy, fz, srcfn = sourcefunction(seismic, 10000, 'gaus1')
angfiles = np.array(
    [
        'single_pole1.txt',
        'single_pole2.txt',
        'single_pole3.txt',
        'single_pole4.txt',
        'single_pole5.txt'
    ]
)
output_basefiles = np.array(['sp1', 'sp2', 'sp3', 'sp4', 'sp5'])
# By default, the material list is 11 character unicode which means that the 
# filenames of the fabric files will get truncated.
# When we run prjrun.status_check(...) the function 
# material.sort_material_list() is also called which will overwrite values using 
# the material_list variable so we need to change the material_list and not the 
# individual parameters afterwards.
material.material_list = material.material_list.astype('U16')
material.material_list[1,-2] = 'True' 

n = len(angfiles)
theta = np.array([90,0,0])
phi = np.array([0,90,0])
channels = np.array(['x','y','z'])

for ii in range(len(theta)):
    for jj in range(n):
        print('Seismic modeling of single pole fabric ' + str(jj+1) )   
        material.material_list[1,-1] = angfiles[jj]
        seis = copy.deepcopy(seismic)
        prjrun.status_check(
            seis, material, domain, prjfile
        )
        # P waves
        print(
            f'Computing model for source \
                theta = {theta[ii]}, phi = {phi[ii]}; P waves'
        )
        print()
        seis.theta = theta[jj]
        seis.phi = phi[jj]
        prjrun.runseismic(
            seis, material, domain
        )
        for channel in channels:
            print(f'Creating the array objects for channel {channel}.')
            print()        
            arr = Array(f'V{channel}', prjfile, 'receivers.xyz')        
            arr.seismic.theta = seis.theta 
            arr.seismic.phi = seis.phi
            print(f'Saving the array objects for the {channel}-component.')
            print()
            arr.save(
                output_basefile = output_basefiles[jj] + \
                    f'.theta{theta[ii]}.phi{phi[iI]}.{channel}'
                )
    
        print(
            f'Deleting model outputs for \
                theta = {theta[ii]} and phi = {phi[ii]}.'
        )
        print()
        datfiles = glob('V*.dat')
        for file_path in datfiles:
            if os.path.exists(file_path):
                os.remove(file_path)
    


# ==============================================================================
# Make some plots 
patterns = np.array([
    '*.theta90.phi0.*.pkl',
    '*.theta0.phi0.*.pkl', 
    '*.theta0.phi90.*.pkl'
])

# Allocate space. We will load the homogeneous 
m = int(seismic.time_steps)
p = len(patterns)

x = np.zeros([m, n, 3])
y = x.copy() 
z = x.copy()
homoxyz = np.zeros([m, 3, p])


for jj in range(p):
    files = glob(patterns[jj])
    xi = 0
    yi = 0
    zi = 0
    for file in files:
        if '.x.' in file:
            with open(file, 'rb') as f:
                tempx = pickle.load(f)
            x[:,xi,jj] = tempx.timeseries[:,1]
            xi += 1
        
        if '.y.' in file:
            with open(file, 'rb') as f:
                tempy = pickle.load(f)
            y[:,yi,jj] = tempy.timeseries[:,1]
            yi += 1
        
        if '.z.' in file:
            with open(file, 'rb') as f:
                tempz = pickle.load(f)
            z[:,zi,jj] = tempz.timeseries[:,1]
            zi += 1
    
    homoxyz[:,0,jj] = tempx.timeseries[:,0]
    homoxyz[:,1,jj] = tempy.timeseries[:,0]
    homoxyz[:,2,jj] = tempz.timeseries[:,0]
    


    
# Sort the files into their

labels = [
    'Homogeneous', 
    'Modeled Homogeneous', 
    'Weak', 
    'Moderately Weak', 
    'Moderately Strong', 
    'Strong'
]
# 'Source-x', 'Source-y', 'Source-z'
# There will be a 3c set of subplots for each of the 3 sources
# ------------------------------------------------------------------------------
