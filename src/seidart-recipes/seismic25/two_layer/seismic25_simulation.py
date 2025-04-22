import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'two_layer.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)
# seis.sourcefunction(seis)
seis.CFL = seis.CFL/2
seis.density_method = 'geometric' #'harmonic'
seis.build(mat, dom, recompute_tensors = False) 
seis.kband_check(dom)
seis.run()


# Create the GIF animation so we can
if dom.dim == 2: 
    frame_delay = 5 
    frame_interval = 30 
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


array_vx = Array('Vx', project_file, receiver_file)
array_vx.exaggeration = 0.025 
array_vx.save()
array_vz = Array('Vz', project_file, receiver_file) 
array_vz.exaggeration = 0.025
array_vz.save() 