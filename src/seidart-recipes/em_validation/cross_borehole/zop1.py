import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model 
from seidart.simulations.common_offset import CommonOffset
from seidart.visualization.im2anim import build_animation
from seidart.routines.arraybuild import Array


project_file = 'zop.json' 
rcxfile = 'zop1r.xyz'
srcfile = 'zop1s.xyz'
channel = 'Ez'

# ------------------------------------------------------------------------------
# 
domain, material, seismic, electromag = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)
electromag.build(material, domain, recompute_tensors = False)
electromag.kband_check(domain)

zop1 = CommonOffset(
    srcfile, 
    channel, 
    project_file, 
    rcxfile, 
    receiver_indices = False, 
    single_precision = True,
)
zop1.output_basefile = 'zop1.ez'
zop1.co_run(parallel = False)
zop1.srcrcx_distance()

# ------------------------------- Custom Ploting -------------------------------
fig, (ax0, ax1) = plt.subplots(
    2,1,
    figsize = (6, 12), sharex = True,
    gridspec_kw = {'height_ratios':[4,1]},
)

exaggeration = 0.03
rcx_depth = (zop1.receiver_xyz_all[:,2] - 65)*domain.dz
rcx_tick_label = rcx_depth[::15]
rcx_tick_locs = np.arange(0, len(rcx_depth))[::15]
timevals = np.arange(0, zop1.electromag.time_steps)
timeval_locs = timevals[::200]
timeval_labels = np.round(timeval_locs * zop1.electromag.dt / 1e-9).astype(int)

# ax0 = plt.subplot(gs[0]) 
im = ax0.imshow(
    zop1.timeseries, cmap = 'Greys', aspect = 'auto', 
    extent = [5, 80, zop1.electromag.time_steps, 0],
    origin = 'upper'
)
ax0.set_xlabel(r'Depth (m)')
ax1.set_xlabel('')
ax0.xaxis.tick_top()
ax0.xaxis.set_label_position('top')
ax0.set_xticks(rcx_tick_locs)
ax0.set_xticklabels(rcx_tick_label)
ax0.set_ylabel(r'Two-way Travel Time (ns)')
ax0.set_yticks(timeval_locs)
ax0.set_yticklabels(timeval_labels)
ax0.set_ylim([2400, 600])      
# ax0.set_aspect(aspect = exaggeration)
# ax0.text(0, m + 0.03*m, 'x $10^{-6}$')
 # Enable x-tick labels on the top subplot
ax0.tick_params(
    axis='x', which='both', 
    bottom=False, top=True, 
    labelbottom=False, labeltop=True
)

# ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.plot(rcx_depth, zop1.distances, 'k', lw = 2)
ax1.set_ylabel(r'Tx-Rx Distance (m)')
plt.tight_layout()#pad = 1.5)
# fig.subplots_adjust(top = 0.7)

plt.show()

