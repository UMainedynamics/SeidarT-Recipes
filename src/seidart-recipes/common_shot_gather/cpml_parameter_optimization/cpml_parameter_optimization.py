import numpy as np
from glob2 import glob
from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

project_file = 'three_layer.json' 
receiver_file = '../receivers.xyz'

## Initiate the model and domain objects
domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model() 
)


na, ns, nk = 5, 5, 3
# na, ns, nk = 81, 50, 11
residual_norm = np.zeros([na-1, ns-1, nk-1])

alpha_max_scalar = np.linspace(1.0,20,na)
sig_opt_scalar = 10**np.linspace(-2,0.3,ns)
kappa_max = np.linspace(5, 10, nk)



## Run the reference model
domain.alpha_max_scalar = 1.0# alpha_max_scalar[0]
domain.sig_opt_scalar = 1.2#sig_opt_scalar[0]
domain.kappa_max = 5#kappa_max[0]

seis.build(material, domain)
seis.kband_check(domain)
seis.run() 


frame_delay = 5
frame_interval = 20 
alpha_value = 0.3

build_animation(
        project_file, 
        'Vz', frame_delay, frame_interval, alpha_value, 
)


fn = glob('Vz*.dat')
dat = read_dat(fn[1000], 'Vz', domain, single = True)