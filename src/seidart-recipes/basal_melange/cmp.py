import numpy as np 

from seidart.routines import prjrun, sourcefunction
from seidart.routines.arraybuild import Array 
from seidart.visualization.im2anim import build_animation 

prjfile = 'basal_melange.prj'
receiver_file = 'receivers.xyz' 

# Load the project file
domain, material, seismic, electromag = prjrun.domain_initialization(prjfile) 
prjrun.status_check(seismic, material, domain, prjfile, append_to_prjfile=True)
tv, fx, fy, fz, srcfn = sourcefunction(seismic, 1e7, 'gaus1')


prjrun.runseismic(seismic, material, domain, use_complex_equations = False)

# Visualize the wavefield
step_interval = 10
alpha = 0.3
delay = 10

build_animation(prjfile, 'Vz', step_interval, alpha, delay)