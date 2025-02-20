import numpy as np
import os
from glob2 import glob 
import subprocess 

from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

project_file = 'cmp_anisotropic1.json' 
receiver_file = 'receivers.xyz'

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model() 
)

## Compute the permittivity coefficients
emfreqs = np.array([1e7, 2e7, 3e7, 4e7, 5e7, 6e7, 7e7, 8e7, 9e7, 1e8, 1.5e8, 2e8])
seisfreqs = np.array([16.0]) #np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]) #np.array([1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 40.0, 80.0, 120.0, 150.0, 180.0])
angfiles = np.array(['isotropic.csv'])#, 'weak_vertical_c.csv', 'strong_vertical_c.csv'])

run_electromag = False

if run_electromag:
    for file in angfiles:
        material.angfile[np.where(material.is_anisotropic)] = file
        for ind in range(len(emfreqs)):
            em.f0 = emfreqs[ind]
            em.build(material, domain)
            em.kband_check(domain)
            em.run() 
            array_ex = Array('Ex', project_file, receiver_file)
            # Copy over the sourcefunction for some bookkeeping
            array_ex.electromag.sourcefunction_x = em.sourcefunction_x.copy()
            array_ex.electromag.sourcefunction_y = em.sourcefunction_y.copy()
            array_ex.electromag.sourcefunction_z = em.sourcefunction_z.copy()
            # Do the same for zede
            array_ez = Array('Ez', project_file, receiver_file)
            array_ez.electromag.sourcefunction_x = em.sourcefunction_x.copy()
            array_ez.electromag.sourcefunction_y = em.sourcefunction_y.copy()
            array_ez.electromag.sourcefunction_z = em.sourcefunction_z.copy()
            array_ex.save(output_basefile = f'ex.{file.split('.')[0]}.{str(emfreqs[ind])}')
            array_ez.save(output_basefile = f'ez.{file.split('.')[0]}.{str(emfreqs[ind])}')
else:
    # Do the same for seismic
    for file in angfiles:
        material.angfile[np.where(material.is_anisotropic)] = file
        for ind in range(len(seisfreqs)):
            seis.f0 = seisfreqs[ind]
            seis.build(material, domain)
            seis.kband_check(domain)
            seis.run() 
            array_vx = Array('Vx', project_file, receiver_file)
            array_vz = Array('Vz', project_file, receiver_file)
            # Copy over the sourcefunction for some bookkeeping
            array_vx.seismic.sourcefunction_x = seis.sourcefunction_x.copy()
            array_vx.seismic.sourcefunction_y = seis.sourcefunction_y.copy()
            array_vx.seismic.sourcefunction_z = seis.sourcefunction_z.copy()
            # Do the same for zede
            array_vz.seismic.sourcefunction_x = seis.sourcefunction_x.copy()
            array_vz.seismic.sourcefunction_y = seis.sourcefunction_y.copy()
            array_vz.seismic.sourcefunction_z = seis.sourcefunction_z.copy()
            array_vx.save(output_basefile = f'vx.{file.split('.')[0]}.{str(seisfreqs[ind])}')
            array_vz.save(output_basefile = f'vz.{file.split('.')[0]}.{str(seisfreqs[ind])}')


    #     # Create the GIF animation so we can 
frame_delay = 5 
frame_interval = 25 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
)
   

