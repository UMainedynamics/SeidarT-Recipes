import numpy as np
from seidart.routines.definitions import *
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation
from seidart.routines.materials import rotator_zxz, ice_permittivity, snow_conductivity

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
# ------------------------------------------------------------------------------

make_animation = False

## Initiate the model and domain objects
project_files = ['strong_aniso.json', 'mod_aniso.json', 'aniso_ice.json' ]
receiver_file = 'receivers.xyz'

for pf in project_files:
    domain, material, seis, em = loadproject(
        pf, Domain(), Material(), Model(), Model()
    )
    em.CFL = em.CFL/1.25 
    em.build(material, domain, recompute_tensors = False)
    em.kband_check(domain)
    em.run()
    
    array_ex = Array('Ex', pf, receiver_file)
    array_ey = Array('Ey', pf, receiver_file)
    array_ez = Array('Ez', pf, receiver_file) 
    array_ex.save() 
    array_ey.save() 
    array_ez.save() 

    if make_animation:
        frame_delay = 20 
        frame_interval = 30
        alpha_value = 0.3
        if domain.dim == 2.5:
            yslice = em.yind + domain.cpml
            slicer(pf, 'Ex', yslice, frame_interval, frame_delay, plane = 'xz')
            slicer(pf, 'Ey', yslice, frame_interval, frame_delay, plane = 'xz')
            slicer(pf, 'Ez', yslice, frame_interval, frame_delay, plane = 'xz')
        else:
            build_animation(
                pf, 
                'Ez', frame_delay, frame_interval, alpha_value, 
                is_single_precision = True
            )

            build_animation(
                pf, 
                'Ex', frame_delay, frame_interval, alpha_value, 
                is_single_precision = True
            )

    

# energies = compute_polar_energy(E_x_f[950:1150], E_y_f[950:1150], em.dt, angles, to_db=True)

# fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
# ax.plot(angles, energies)
# ax.set_title("Polar anisotropy energy response")
# plt.show()

# E = compute_polar_energy(E_x_f, E_y_f, em.dt, angles, to_db=False)
# baseline = np.cos(angles)**2
# corr = E / baseline
# anisotropy_db = 10*np.log10(corr/corr.max())

# fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
# ax.plot(angles, E)
# plt.show()


# total_energy = np.sum(np.abs(E_x)**2 + np.abs(E_y)**2)*em.dt