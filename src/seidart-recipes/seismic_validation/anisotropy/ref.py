import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------
## Initiate the model and domain objects
project_file = 'ref.json' 
receiver_file = 'receivers1.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

seis.velocity_scaling_factor = 3.0
seis.CFL = seis.CFL/1.0
seis.density_method = 'geometric' 
seis.build(mat, dom, recompute_tensors = False) 
sigma, kappa, alpha, __, __ = cpmlcompute(seis, dom, velocity_scaling_factor = seis.velocity_scaling_factor)
print(sigma.max(), alpha.max() )
seis.kband_check(dom)
seis.run()

frame_delay = 5 
frame_interval = 25 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True
)

# build_animation(
#         project_file, 
#         'Vx', frame_delay, frame_interval, alpha_value, 
#         is_single_precision = True
# )

