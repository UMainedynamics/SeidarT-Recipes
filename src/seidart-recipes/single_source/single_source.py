from seidart.routines import prjbuild, prjrun, sourcefunction
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

prjfile = 'single_source_test.prj' 
rcxfile = 'receivers.xyz'
complex_values = False

## Initiate the model and domain objects
dom, mat, seis, em = prjrun.domain_initialization(prjfile)

## Compute the permittivity coefficients
# prjrun.status_check(em, mat, dom, prjfile)
prjrun.status_check(
    em, mat, dom, prjfile, append_to_prjfile = True
)

timevec, fx, fy, fz, srcfn = sourcefunction(em, 10, 'gaus1')

prjrun.runelectromag(em, mat, dom, use_complex_equations = complex_values)

array_ex = Array('Ex', prjfile, rcxfile, is_complex = complex_values)
# array_ex.gain = int(em.time_steps/5)
array_ex.exaggeration = 0.1 
array_ex.sectionplot(
    plot_complex = False
)

# Let's plot a trace for the 10th receiver in the list of receivers. 
receiver_number = 10
array_ex.wiggleplot(receiver_number, figure_size = (5,8))

# Pickle the object
array_ex.save()

# Create the GIF animation so we can 
frame_delay = 10 
frame_interval = 10 
alpha_value = 0.3

build_animation(
        prjfile, 
        'Ex', frame_delay, frame_interval, alpha_value, 
        is_complex = complex_values, 
        is_single_precision = True
)

# We can do the same for the z-direction
array_ez = Array('Ez', prjfile, rcxfile, is_complex = complex_values)
array_ez.gain = int(em.time_steps/3)
array_ez.exaggeration = 0.1
array_ez.sectionplot(
    plot_complex = False
)

array_ez.wiggleplot(receiver_number, figure_size = (5,8))
array_ex.save()

build_animation(
        prjfile, 
        'Ex', frame_delay, frame_interval, alpha_value, 
        is_complex = complex_values, 
        is_single_precision = True,
        plottype = 'energy_density'
)
