import numpy as np 

from seidart.routines import prjrun, sourcefunction
from seidart.routines.arraybuild import Array 
from seidart.visualization.im2anim import build_animation 


# file input definitions
prjfile = 'basal_melange.prj'
receiver_file = 'receivers.xyz' 

# Load the project file
domain, material, seismic, electromag = prjrun.domain_initialization(prjfile) 
prjrun.status_check(seismic, material, domain, prjfile, append_to_prjfile=True) # If editing values, set append_to_prjfile=False so they aren't overwritten
timevec, fx, fy, fz, srcfn = sourcefunction(seismic, 1e8, 'gaus1')


prjrun.runseismic(seismic, material, domain)

# Visualize the wavefield
step_interval = 10
alpha = 0.3
delay = 10

build_animation(
        prjfile, 
        'Vz', delay, step_interval, alpha, 
        is_single_precision = True
)

array_vz = Array('Vz', prjfile, receiver_file)
array_vz.exaggeration = 0.1
array_vz.sectionplot(amplitude_correction_type = 'AGC', colormap = 'seismic')