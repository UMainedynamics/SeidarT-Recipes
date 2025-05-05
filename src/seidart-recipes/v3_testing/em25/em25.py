import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.visualization.slice25d import slicer
from seidart.routines.materials import rotator_zxz, ice_permittivity, snow_conductivity

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'em25.json' 
receiver_file = 'receivers.xyz'
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

em.CFL = em.CFL # We aren't worried too much about stability, but this helps us add more time steps to allow for better Fourier analysis
em.build(material, domain)
em.kband_check(domain)
em.run()


# frame_delay = 15 
# frame_interval = 17
# alpha_value = 0.3

# build_animation(
#     project_file, 
#     'Ex', frame_delay, frame_interval, alpha_value, 
#     is_single_precision = True
# )

slicer(
    project_file, 'Ex', indslice, num_steps, delay,
            plane = 'xz', alpha = 0.3, is_single = True
        )