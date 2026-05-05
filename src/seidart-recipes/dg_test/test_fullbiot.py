import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model, Biot
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'surfacewave_validation.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Biot(), Model()
)

seis.use_multimodal = True
seis.source_n_octaves = 3
seis.build(mat, dom) 
seis.run()


frame_delay = 5 
frame_interval = 20 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True
)

build_animation(
        project_file, 
        'Vx', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True
)

