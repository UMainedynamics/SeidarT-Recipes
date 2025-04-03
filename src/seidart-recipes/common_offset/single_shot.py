import numpy as np 
from seidart.routines import prjbuild, prjrun, sourcefunction
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

project_file = 'common_offset.json' 

## Initiate the model and domain objects
dom, mat, seis, em = prjrun.domain_initialization(project_file)

## Compute the permittivity coefficients
prjrun.status_check(em, mat, dom, project_file, seismic = False, append_to_project_file = False)
# prjrun.status_check(
#     em, mat, dom, project_file, seismic = True, append_to_project_file = True
# )

timevec, fx, fy, fz, srcfn = sourcefunction(em, 1e7, 'gaus1', 'e')
# timevec, fx, fy, fz, srcfn = sourcefunction(em, 1e7, 'gaus1', 's')

complex_values = False
# complex_values = True
prjrun.runelectromag(em, mat, dom, use_complex_equations = complex_values)
# prjrun.runseismic(seis, mat, dom)

build_animation(
        project_file, 
        'Ex', 10, 10, 0.3, 
        is_complex = complex_values, 
        is_single_precision = True
)



array_ex = Array('Ex', project_file, rcxfile, is_complex = complex_values)
# array_ez = Array('Ez', project_file, rcxfile, is_complex = complex_values)

array_ex.gain = int(em.time_steps/3)
array_ex.exaggeration = 0.1 
# array_ez.gain = int(em.time_steps/3)
# array_ez.exaggeration = 0.1

array_ex.sectionplot(
    plot_complex = False
)

# array_ez.sectionplot(
#     plot_complex = False
# )

