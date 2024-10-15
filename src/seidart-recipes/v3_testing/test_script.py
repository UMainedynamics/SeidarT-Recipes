from seidart.routines.definitions import * 
from seidart.routines.classes import Domain, Material, Model

project_file = 'test_model.json'

dom, mat, seis, em = loadproject(
    project_file,
    Domain(),
    Material(), 
    Model(),
    Model()
)

## Compute the tensor coefficients
# em.status_check(mat, dom, project_file, append_to_json = True)
mat.material_flag = True
seis.status_check(mat, dom, project_file, append_to_json = True)

