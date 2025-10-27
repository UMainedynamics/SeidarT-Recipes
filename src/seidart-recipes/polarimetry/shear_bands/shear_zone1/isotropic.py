from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

project_file = 'isotropic.json' 
receiver_file = 'receivers.xyz'

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model() 
)

## Compute the permittivity coefficients
em.CFL = em.CFL/1.0
em.build(material, domain)
em.kband_check(domain)
# em.density_method = "arithmetic"
em.run() 

# array_ex = Array('Ex', project_file, receiver_file)
# # array_ex.gain = int(em.time_steps/5)
# array_ex.exaggeration = 0.1 
# array_ex.sectionplot()

# # Let's plot a trace for the 10th receiver in the list of receivers. 
# receiver_number = 10
# array_ex.wiggleplot(receiver_number, figure_size = (5,8))

# # Pickle the object
# array_ex.save()

# Create the GIF animation so we can 
frame_delay = 10 
frame_interval = 20 
alpha_value = 0.3

build_animation(
        project_file, 
        'Ex', frame_delay, frame_interval, alpha_value, 
)

# ----------
# We can do the same for the z-direction
array_ez = Array('Ez', project_file, receiver_file)
array_ez.gain = int(em.time_steps/3)
array_ez.exaggeration = 0.1
array_ez.sectionplot()

array_ez.wiggleplot(receiver_number, figure_size = (5,8))
array_ez.save()

build_animation(
        project_file, 
        'Ez', frame_delay, frame_interval, alpha_value, 
)
