import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.routines.materials import rotator_zxz, ice_permittivity, snow_conductivity

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# ------------------------------------------------------------------------------

project_file = 'single_medium.json'
receiver_file = 'receivers2.xyz'

domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)
seis.CFL = seis.CFL/1.0 
seis.density_method = 'geometric'
seis.build(material, domain)
seis.kband_check(domain)
seis.run()

array_vz = Array('Vz', project_file, receiver_file) 

frame_delay = 10 
frame_interval = 40
alpha_value = 0.3
build_animation(
    project_file, 
    'Vz', frame_delay, frame_interval, alpha_value, 
    is_single_precision = True
)

