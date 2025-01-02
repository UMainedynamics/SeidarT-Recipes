import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from seidart.routines.definitions import *
from seidart.simulations.common_offset import CommonOffset
from seidart.visualization.im2anim import build_animation
from seidart.routines.classes import Domain, Material, Model 

project_file = 'common_offset.json' 
rcxfile = 'common_offset_receivers.xyz'
srcfile = 'common_offset_sources.xyz'
channel = 'Ex'

domain, material, seismic, electromag = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

#
electromag.build(material, domain, recompute_tensors = False)
# electromag.save_to_json()
electromag.kband_check(domain)

co = CommonOffset(
    srcfile, 
    channel, 
    project_file, 
    rcxfile, 
    receiver_indices = False, 
    single_precision = True,
)

co.co_run(parallel = False)

co.gain = 800
co.exaggeration = 0.05
co.sectionplot()
co.save()

