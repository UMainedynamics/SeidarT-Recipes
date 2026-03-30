import numpy as np

from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation


project_file = '0crevasse.json'
receiver_file = '1receivers.xyz' #'../../src/seidart/recipes/receivers.xyz'
complex_values = False

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

## Compute the elastic coefficients
seis.density_method = 'geometric'
seis.CFL = 1.0/( 6.0* np.sqrt(3))
seis.build(material, domain)
seis.kband_check(domain)
seis.run()

## Compute the permittivity coefficients
#em.build(material, domain)
#### em.kband_check(domain)
#em.run()

array_vx = Array('Vx', project_file, receiver_file)
# array_vx.gain = int(em.time_steps/5)
array_vx.exaggeration = 0.1
array_vx.sectionplot()

# Let's plot a trace for the 10th receiver in the list of receivers.
receiver_number = 60
array_vx.wiggleplot(receiver_number, figure_size = (5,8))

# Pickle the object
array_vx.save()

# Create the GIF animation so we can
frame_delay = 10
frame_interval = 20
alpha_value = 0.3

build_animation(
        project_file,
        'Vx', frame_delay, frame_interval, alpha_value,
)

# We can do the same for the z-direction
##array_ez = Array('Ez', project_file, receiver_file)
##array_ez.gain = int(em.time_steps/3)
##array_ez.exaggeration = 0.1
##array_ez.sectionplot()

##array_ez.wiggleplot(receiver_number, figure_size = (5,8))
##array_ez.save()

##build_animation(
##        project_file,
##        'Ez', frame_delay, frame_interval, alpha_value,
##)
