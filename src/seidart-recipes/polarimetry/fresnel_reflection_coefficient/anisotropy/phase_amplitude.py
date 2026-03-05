import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from seidart.routines.materials import * 
from seidart.routines.definitions import * 
from seidart.routines.fabricsynth import Fabric


# ------------------------------------------------------------------------------
eps0 = 8.85478782e-12
temperature = -10 
density = 910 
f0 = 1e8 


m, n = 45, 91
incidence = np.linspace(0, 90, m)
inc_rad = np.radians(incidence)
azimuth = np.linspace(0, 180, n)
azi_rad = np.radians(azimuth) 
std_scalar = 180.0/32.0
plunge_std = np.array([0.5, 1.0, 2.0, 4.0, 8.0]) * std_scalar
# vel = np.zeros([m, n])

kk = len(plunge_std)
refractive_indices = np.empty([m*n, 2, kk], dtype = complex)

def azi_inc_to_dir(azi, inc):    
    vx = np.sin(inc) * np.cos(azi)  
    vy = np.sin(inc) * np.sin(azi)
    vz = np.cos(inc)
    v = np.array([vx, vy, vz])
    return v





M = 6
# plunge = np.linspace(0.0, 90.0, M)
# div_orig = np.linspace(0.0, 22.5, M) 
# div_orig[0] = 1.0

fabric_parameters = {
    'npts': [500],
    'distribution': 'normal',
    'plunge': [0],
    'skew_plunge': [0],
    'plunge_std': [10],
    'trend': [0],
    'skew_trend': [0],
    'trend_std': [90],
    'orientation': [0],
    'skew_orientation': [0],
    'orientation_std': [0]
}


eps = ice_permittivity(temperature, density, f0)
eps_iso = np.trace(eps) / 3.0

cols_em = [
    'e11', 'e22', 'e33', 'e23', 'e13', 'e12',
    's11', 's22', 's33', 's23', 's13', 's12'
]

perm_cond_df = pd.DataFrame(columns = cols_em)
# elastic_df = pd.DataFrame(columns = cols_s)

anisotropy_parameters_perm = pd.DataFrame(columns = ['Ratio', 'Birefringence', 'Spread'])
anisotropy_parameters_cond = pd.DataFrame(columns = ['Ratio', 'Birefringence', 'Spread'])

wave_speeds = pd.DataFrame(columns = ['Incidence', 'Azimuth', 'Eig1', 'Eig2', 'Eig3'])

fabric = Fabric(fabric_parameters, plot = False)

for kk, std in enumerate(plunge_std):
    fabric_parameters['plunge_std'] = [std] 
    fabric = Fabric(fabric_parameters, plot = False)
    perm, cond = vrh2( 
        eps.real, -2.0*np.pi*f0*eps0*eps.imag, fabric.euler_angles.to_numpy() 
    )
        
    anisotropy_parameters_perm.loc[len(anisotropy_parameters_perm)] = anisotropy_parameters(perm)
    anisotropy_parameters_cond.loc[len(anisotropy_parameters_cond)] = anisotropy_parameters(cond)
    
    coefs = np.array([
        perm[0,0], perm[1,1], perm[2,2], perm[1,2], perm[0,2], perm[0,1],
        cond[0,0], cond[1,1], cond[2,2], cond[1,2], cond[0,2], cond[0,1]
    ])
    perm_cond_df.loc[ len(perm_cond_df) ] = coefs
    
    for ii, azi in enumerate(azi_rad):
        for jj, inc in enumerate(inc_rad):
            n_hat = azi_inc_to_dir(azi, inc)
            lam, vecs, refractive_indices[jj + (ii*m),:, kk] = get_complex_refractive_index(perm_cond_df.iloc[-1], f0, n_hat)
