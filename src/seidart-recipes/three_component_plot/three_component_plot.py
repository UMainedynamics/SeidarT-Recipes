import numpy as np 
import matplotlib.pyplot as plt 

from seidart.routines.definitions import *
from seidart.routines import prjrun, sourcefunction
from seidart.routines.arraybuild import Array 
from seidart.visualization.im2anim import build_animation

prjfile = 'four_layer_iso.prj'
rcxfile = 'receivers.xyz'

domain, material, seismic,__ = prjrun.domain_initialization(prjfile)
timevec, fx, fy, fz, srcfn = sourcefunction(seismic, 100000, 'gaus1')
prjrun.status_check(
    seismic, material, domain, prjfile
)
prjrun.runseismic(
    seismic, material, domain
)

frame_delay = 3
frame_interval = 20 
alpha_value = 0.3 
build_animation(prjfile, 'Vz', frame_delay, frame_interval, alpha_value)
