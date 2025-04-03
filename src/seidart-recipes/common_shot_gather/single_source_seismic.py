from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

project_file = 'single_source.json' 
receiver_file = 'receivers.xyz'

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model() 
)

## Compute the permittivity coefficients
seis.build(material, domain)
seis.kband_check(domain)
seis.run() 

array_vx = Array('Vx', project_file, receiver_file)
array_vx.agc_gain = 101
array_vx.exaggeration = 0.1 
# array_vx.sectionplot()

# Let's plot a trace for the 10th receiver in the list of receivers. 
receiver_number = 10
# array_vx.wiggleplot(receiver_number, figure_size = (5,8))

# Pickle the object
# array_vx.save()

# Create the GIF animation so we can 
frame_delay = 6 
frame_interval = 10 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vx', frame_delay, frame_interval, alpha_value, 
)

# ----------
# We can do the same for the z-direction
array_vz = Array(Vz', project_file, receiver_file)
array_vz.agc_gain_window = 201 
array_vz.exaggeration = 0.1
array_vz.sectionplot()

array_vz.wiggleplot(receiver_number, figure_size = (5,8))
# array_vz.save()

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
)
