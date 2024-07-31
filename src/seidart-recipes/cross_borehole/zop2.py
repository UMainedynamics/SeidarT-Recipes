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
srcfile = 'gusmeroli_zop2s.xyz'

channel = 'Ez'
is_complex = False

# ------------------------------------------------------------------------------
# 
domain, material, seismic, electromag = prjrun.domain_initialization(prjfile)
prjrun.status_check(
    electromag, material, domain, prjfile, append_to_prjfile = False
)

# ------------------------------------------------------------------------------
#ZOP2 Survey

zop2 = CommonOffset(
    srcfile, 
    channel, 
    prjfile, 
    rcxfile, 
    receiver_indices = False, 
    is_complex = is_complex,
    single_precision = True,
    status_check = False
)

zop2.co_run()
zop2.exaggeration = 0.03
zop2.sectionplot() 

depth = (zop2.receiver_xyz_all[:,2] - domain.cpml ) * domain.dz
zop2_txrx_dist = np.sqrt( 
    np.sum( 
        (zop2.source_xyz - (zop2.receiver_xyz_all-domain.cpml)*domain.dz)**2,
        axis = 1
    )
)
fig, ax = plt.subplots(1,1, figsize = (4,4) )
ax.plot(depth, zop2_txrx_dist, lw = 2, c='k')
ax.set_ylim(20,40)
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Tx-Rx Distance (m)')