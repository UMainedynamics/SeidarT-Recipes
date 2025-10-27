import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'surfacewave_validation.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

seis.use_multimodal = True
seis.source_n_octaves = 3
# seis.density_method = 'arithmetic'
# seis.density_method = 'harmonic'
seis.density_method = 'geometric' 
seis.build(mat, dom, recompute_tensors = False) 
seis.kband_check(dom)
seis.run()

array_vx = Array('Vx', project_file, receiver_file)
array_vx.exaggeration = 0.01 
array_vx.save()
array_vz = Array('Vz', project_file, receiver_file) 
array_vz.exaggeration = 0.01
array_vz.save() 
if dom.dim == 2.5:
        array_vy = Array('Vy', project_file, receiver_file) 
        array_vy.exaggeration = 0.01
        array_vy.save() 

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

