from seidart.routines import prjbuild, prjrun, sourcefunction
from seidart.visualization.im2anim import build_animation

prjfile = 'wavefield_animation.prj' 
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

# Create the GIF animation so we can 
frame_delay = 5 
frame_interval = 10 
alpha_value = 0.3

build_animation(
        prjfile, 
        'Ex', frame_delay, frame_interval, alpha_value, 
        is_complex = complex_values, 
        is_single_precision = True
)

build_animation(
        prjfile, 
        'Ez', frame_delay, frame_interval, alpha_value, 
        is_complex = complex_values, 
        is_single_precision = True,
)
