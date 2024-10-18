from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model

project_file = 'test_model.json'

domain, material, seismic, electromag = loadproject(
    project_file,
    Domain(),
    Material(), 
    Model(),
    Model()
)

## Compute the tensor coefficients
# em.status_check(mat, dom, project_file, append_to_json = True)
material.material_flag = True
seismic.build(material, domain, recompute_tensors = True)
seismic.run()

electromag.build(material, domain, append_to_json = True)
electromag.run()