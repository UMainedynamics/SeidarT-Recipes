from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.visualization.im2anim import build_animation
from seidart.visualization.slice25d import slicer 

project_file = 'test_model.json'

domain, material, seismic, electromag = loadproject(
    project_file,
    Domain(),
    Material(), 
    Model(),
    Model()
)

# Compute the tensor coefficients
seismic.build(material, domain, recompute_tensors = False)
seismic.kband_check(domain)
seismic.run()

# Create the GIF animation so we can 
frame_delay = 5 
frame_interval = 20 
# alpha_value = 0.3

# build_animation(
#         project_file, 
#         'Vz', frame_delay, frame_interval, alpha_value, 
#         is_single_precision = True
# )

slicer(project_file, 'Vz', 15, frame_interval, frame_delay)

# ------------------------------------------------------------------------------
# electromag.build(material, domain, recompute_tensors = True)
# electromag.kband_check(domain)
# electromag.run()

# frame_delay = 5 
# frame_interval = 10 
# alpha_value = 0.3


# build_animation(
#         project_file, 
#         'Ex', frame_delay, frame_interval, alpha_value, 
#         is_single_precision = True
# )