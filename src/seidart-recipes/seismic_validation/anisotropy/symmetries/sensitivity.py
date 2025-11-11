import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.visualization.slice25d import slicer

# ------------------------------------------------------------------------------
## Initiate the model and domain objects
# project_file = 'isotropic.json' 
project_file = 'orthorhombic.json' 

receiver_file = 'receivers_ref.xyz'

dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

seis.density_method = 'geometric' 
seis.build(mat, dom, recompute_tensors = False) 
seis.kband_check(dom)
seis.run()

