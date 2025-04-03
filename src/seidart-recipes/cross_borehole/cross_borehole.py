import numpy as np 

from seidart.routines import prjrun, sourcefunction 
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
indslice = 25 + domain.cpml #Inline with the source and corrected for the CPML

# GIF parameters 
num_steps = 10
alpha = 0.3 
delay = 10 

slice(prjfile, 'Ex', indslice, num_steps, plane, alpha, delay)
slice(prjfile, 'Ey', indslice, num_steps, plane, alpha, delay)
slice(prjfile, 'Ez', indslice, num_steps, plane, alpha, delay)

# ------------------------------------------------------------------------------
# Visualize the receiver data
arr_x = Array('Ex', prjfile, rcxfile) 
arr_y = Array('Ey', prjfile, rcxfile)
arr_z = Array('Ez', prjfile, rcxfile)

arr_x.gain = 31 
arr_y.gain = 31
arr_z.gain = 31
arr_x.exaggeration = 0.1 
arr_y.exaggeration = 0.1
arr_z.exaggeration = 0.1    

arr_x.sectionplot()
arr_y.sectionplot()
arr_z.sectionplot()

