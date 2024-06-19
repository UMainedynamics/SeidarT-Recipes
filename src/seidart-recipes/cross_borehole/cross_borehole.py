import numpy as np 

from seidart.routines import prjbuild, sourcefunction 
from seidart.routines.arraybuild import Array 
from seidart.visualization.slice25d import slice

# ------------------------------------------------------------------------------
prjfile = 'cross_borehole.prj'
rcxfile = 'receivers.xyz'
is_complex = False

# ------------------------------------------------------------------------------
# Prep and run the model
domain, material, __, em = prjrun.domain_initialization(prjfile)
prjrun.status_check(em, material, domain, prjfile)
tv, fx, fy, fz, srcfn = sourcefunction(em, 1e6, 'gaus1')

prjrun.runelectromag(
    em, material, domain, use_complex_equations = is_complex
)

# ------------------------------------------------------------------------------
# Visualize the wavefield
plane = 'xz' 
indslice = 25 #Inline with the source

# GIF parameters 
num_steps = 10
alpha = 0.3 
delay = 5 

slice(prjfile, 'Ex', indslice, num_steps, plane, alpha, delay)
slice(prjfile, 'Ey', indslice, num_steps, plane, alpha, delay)
slice(prjfile, 'Ez', indslice, num_steps, plane, alpha, delay)

