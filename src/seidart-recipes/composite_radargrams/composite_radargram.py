import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from glob2 import glob 

from seidart.routines import prjrun,sourcefunction 
from seidart.routines.arraybuild import Array 
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------
prjfile = 'composite_radargram.prj' 
receiver_file = 'receivers2.xyz' 
is_complex = False

source_center_frequencies = np.array([
	5e7,7e7,8.5e7,1e8,1.5e8,2e8,3e8,4e8
])
# source_center_frequencies = np.array([
#    5e6,1e7,2e7,4e7,8e7,1e8,2e8,4e8,
# ])

# ------------------------------------------------------------------------------

domain, material, seis, em = prjrun.domain_initialization(prjfile)

# ------------------------------------------------------------------------------
n = len(source_center_frequencies)
m = int(em.time_steps)
# The source will be in the x-direction so we will only need the srcfn value. 
source_fn = np.zeros([m])
# ------------------------------------------------------------------------------

prjrun.NP = 4
prjrun.NPA = 4
ind = 5
# for ind in range(n):
em.f0 = source_center_frequencies[ind]
prjrun.status_check(em, material, domain, prjfile)
tv, fx, fy, fz, srcfn = sourcefunction(
	em, 1e7, 'gaus1'
)
source_fn = source_fn + srcfn
prjrun.runelectromag(
    em, material, domain, use_complex_equations = is_complex
)
arr = Array('Ex', prjfile, receiver_file, is_complex = is_complex)
    # arr.save( output_basefile = 'source_frequency-' + str(int(source_center_frequencies[ind])) )

#!
arr.exaggeration = 0.02
arr.gain = 501
arr.sectionplot()

frame_delay = 10 
frame_interval = 10 
alpha_value = 0.3

build_animation(
        prjfile, 
        'Ex', frame_delay, frame_interval, alpha_value, 
        is_complex = False, 
        is_single_precision = True
)

#!

# Plot the source function 
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.plot(tv, source_fn)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (m/s)')
plt.show()

composite_radargram = np.zeros(arr.timeseries.shape)
fn = glob('*.csv')
pklfn = glob('*.pkl')
for file in fn:
    composite_radargram += pd.read_csv(file, header = None).to_numpy()

arr.timeseries = composite_radargram 
arr.exaggeration = 0.02
# arr.gain = 101
arr.sectionplot()


