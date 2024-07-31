import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from seidart.routines.definitions import *
from seidart.routines import prjrun, sourcefunction 
from seidart.simulations.common_offset import CommonOffset
from seidart.visualization.im2anim import build_animation
from seidart.routines.arraybuild import Array


prjfile = 'zop.prj' 
rcxfile = 'gusmeroli_zop2r.xyz'

channel = 'Ez'
is_complex = False

# ------------------------------------------------------------------------------
# 
domain, material, seismic, electromag = prjrun.domain_initialization(prjfile)
prjrun.status_check(
    electromag, material, domain, prjfile, append_to_prjfile = False
)
tv, fx, fy, fz, fn = sourcefunction(electromag, 1e6, 'gaus1')
prjrun.runelectromag(electromag, material, domain)
array_ez = Array(channel, prjfile, rcxfile)
array_ez.exaggeration = 0.03
array_ez.save() 
frame_delay = 5
frame_interval = 10 
alpha_value = 0.3 

build_animation(
    prjfile, channel, frame_delay, frame_interval, alpha_value,
    is_complex = is_complex, is_single_precision = True
)

from zop1 import *


from zop2 import * 