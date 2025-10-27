import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.visualization.slice25d import slicer

# ------------------------------------------------------------------------------
## Initiate the model and domain objects
project_file = 'ref1.json' 
receiver_file = 'receivers_ref.xyz'

dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

cfl_scalar = 2.0
if dom.dx == 0.5:
    dx_str = '05'
elif dom.dx == 0.25:
    dx_str = '025'
else:
    dx_str = int(dom.dx)
    cfl_scalar = 2.0 * cfl_scalar

receiver_file = f'receivers{dx_str}.xyz'
# seis.velocity_scaling_factor = 2.5
# seis.CFL = 0.3
# seis.CFL = seis.CFL/2.0
seis.CFL = seis.CFL/cfl_scalar 

seis.density_method = 'geometric' 
seis.build(mat, dom, recompute_tensors = False) 
seis.kband_check(dom)
seis.run()

array_vx = Array('Vx', project_file, receiver_file)
array_vx.exaggeration = 0.01 
array_vx.save(output_basefile = f'Vx-{dx_str}')
array_vy = Array('Vy', project_file, receiver_file) 
array_vy.exaggeration = 0.01
array_vy.save(output_basefile = f'Vy-{dx_str}') 
array_vz = Array('Vz', project_file, receiver_file) 
array_vz.exaggeration = 0.01
array_vz.save(output_basefile = f'Vz-{dx_str}') 

