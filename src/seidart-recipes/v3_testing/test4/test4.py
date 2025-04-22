import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

# ------------------------------------------------------------------------------

## Initiate the model and domainain objects
project_file = 'air_ice.json' 
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)
# seis.sourcefunction(seis)
seis.CFL = seis.CFL/3
# seis.density_method = 'arithmetic'
seis.density_method = 'harmonic'
seis.build(material, domain) 
seis.kband_check(domain)
seis.run()


# Create the GIF animation so we can
if domain.dim == 2: 
    frame_delay = 5 
    frame_interval = 30 
    alpha_value = 0.3
    
    build_animation(
            project_file, 
            'Vz', frame_delay, frame_interval, alpha_value, 
            is_single_precision = True
    )
    
    build_animation(
            project_file, 
            'Vx', frame_delay, frame_interval, alpha_value, 
            is_single_precision = True
    )


