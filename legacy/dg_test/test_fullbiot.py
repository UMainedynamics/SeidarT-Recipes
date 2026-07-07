import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model, Biot
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'surfacewave_validation_homo.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Biot(), Model()
)

seis.use_multimodal = False
seis.source_n_octaves = 3
# seis.CFL = seis.CFL/5.0
seis.build(mat, dom, recompute_tensors = True) 
seis.run()


frame_delay = 5 
frame_interval = 20 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vx', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True, numerical_method = 'dg'
)

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True, numerical_method = 'dg'
)

build_animation(
        project_file, 
        'Qx', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True, numerical_method = 'dg'
)

build_animation(
        project_file, 
        'Qz', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True, numerical_method = 'dg'
)

build_animation(
        project_file, 
        'Pp', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True, numerical_method = 'dg'
)
