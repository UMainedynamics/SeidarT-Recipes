from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model
from seidart.routines.arraybuild import Array
from seidart.visualization.im2anim import build_animation

project_file = 'cmp1.json'
receiver_file = 'receivers.xyz'


domain, material, seis, em = loadproject(
    project_file, Domain(), Material(), Model(), Model()
)

em.build(material, domain)
em.kband_check(domain)
em.run() 

seis.build(material, domain)