from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.visualization.im2anim import build_animation

project_file = 'test2.json'

domain, material, seismic, electromag = loadproject(
    project_file,
    Domain(),
    Material(), 
    Model(),
    Model()
)

## Compute the tensor coefficients
material.material_flag = True
seismic.build(material, domain, recompute_tensors = True)
seismic.kband_check(domain)
seismic.run()

# Create the GIF animation so we can 
frame_delay = 5 
frame_interval = 20 
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

# ------------------------------------------------------------------------------
material.material_flag = True
electromag.build(material, domain, recompute_tensors = True)
electromag.kband_check(domain)
electromag.run()

frame_delay = 8 
frame_interval = 10 
alpha_value = 0.3


build_animation(
        project_file, 
        'Ex', frame_delay, frame_interval, alpha_value, 
        is_single_precision = True
)