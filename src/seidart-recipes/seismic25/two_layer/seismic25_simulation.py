import numpy as np
from seidart.routines.definitions import *
from seidart.routines import prjbuild, prjrun, sourcefunction
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

prjfile = 'two_layer.prj' 

## Initiate the model and domain objects
dom, mat, seis, em = prjrun.domain_initialization(prjfile)

## Compute the permittivity coefficients
# prjrun.status_check(em, mat, dom, prjfile)
prjrun.status_check(
    seis, mat, dom, prjfile, append_to_prjfile = True
)

timevec, fx, fy, fz, srcfn = sourcefunction(seis, 1e8, 'gaus1')
kband_check(seis, dom)
prjrun.runseismic(seis, mat, dom)

# Create the GIF animation so we can 
# frame_delay = 5 
# frame_interval = 30 
# alpha_value = 0.3

# build_animation(
#         prjfile, 
#         'Vz', frame_delay, frame_interval, alpha_value, 
#         is_single_precision = True
# )

# build_animation(
#         prjfile, 
#         'Vx', frame_delay, frame_interval, alpha_value, 
#         is_single_precision = True
# )
