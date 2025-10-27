import numpy as np 
import pickle
from glob2 import glob 
import matplotlib.pyplot as plt 
from matplotlib.colors import TwoSlopeNorm

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
import seidart.routines.materials as mf 

# ------------------------------------------------------------------------------

## Initiate the model and domain objects
project_file = 'bw_validation.json' 
receiver_file = 'receivers.xyz'
dom, mat, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

pklfiles = glob('*-Vx-*.pkl')
f = open(pklfiles[0], 'rb')
array_vz = pickle.load(f)
f.close() 

array_vz.butterworth_filter(
    'highpass', lowcut = 50, highcut = 70, pad_samples = 2*seis.time_steps, order = 2
)

array_vz.exaggeration = 0.2 
array_vz.agc_gain_window = 501 
array_vz.sectionplot(colormap = 'seismic', amplitude_correction_type = 'AGC')

array_vz.fk_analysis(
    2, 
    ntfft = 2*array_vz.timeseries.shape[0],
    nxfft = 2*array_vz.timeseries.shape[1],
    taper = 'both',
    wavenumber_limits = (-0.04,0.04), 
    frequency_limits = (0, 120), 
    use_filtered = True,
    colormap = 'inferno',
    mask_db = -160,
)
