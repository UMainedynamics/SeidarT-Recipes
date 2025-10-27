import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'five_layer.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)
# seis.sourcefunction.
# seis.CFL = seis.CFL/1.0
# seis.broadband_fmin = 2 
# seis.broadband_fmax = seis.f0
# seis.use_broadband = True
# seis.use_multimodal = True
# seis.source_n_octaves = 3
# seis.density_method = 'arithmetic'
# seis.density_method = 'harmonic'
seis.density_method = 'geometric' 
seis.build(mat, dom, recompute_tensors = True) 
seis.kband_check(dom)
seis.run()


array_vx = Array('Vx', project_file, receiver_file)
array_vx.exaggeration = 0.05 
array_vx.save()
array_vz = Array('Vz', project_file, receiver_file) 
array_vz.exaggeration = 0.05
array_vz.save() 

# Create the GIF animation so we can
if dom.dim == 2: 
    frame_delay = 5
    frame_interval = 25 
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


