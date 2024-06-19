import numpy as np
from glob2 import glob
import seaborn as sns

from seidart.routines import prjrun, sourcefunction 
from seidart.routines.definitions import * 


prjfile = 'em_boundary_performance.prj'
domain, material, __, em = prjrun.domain_initialization(prjfile)

# --------------------------------- Variables ----------------------------------
dk = 0.1
ki = 1.0 
kf = 15
kappa_max = np.arange(ki, kf+dk, dk)

sk = 0.05 
si = 0.1
sf = 10.0
sig_opt_scalar = np.arange(si,sf+sk, sk)

source_frequency = np.array([
    2.0e7
])

ref_kappa = 1.0 
ref_sigma = 1.0

m = len(kappa_max)
n = len(sig_opt_scalar)
p = len(source_frequency)
# ---------------------------------- Modeling ----------------------------------
# Prep
cumulative_power_density_x = np.zeros([m,n,p])
cumulative_power_density_z = np.zeros([m,n,p])
datx = np.zeros([domain.nx + 2 * domain.cpml, domain.nz + 2*domain.cpml])
datz = datx.copy()

# Load the project and get going
prjrun.status_check(
    em, material, domain, prjfile, append_to_prjfile = True
)
timevec, fx, fy, fz, srcfn = sourcefunction(em, 10, 'gaus1')

for ii in range(m):
    prjrun.kappa_max = kappa_max[ii]
    for jj in range(n):
        prjrun.sig_opt_scalar = sig_opt_scalar[jj] 
        for kk in range(p):
            print( f"{ii}/{m-1} - {jj}/{n-1} - {kk}/{p-1}" )
            # Zero out the 
            datx[:,:] = 0.0
            datz[:,:] = 0.0 
            em.f0 = source_frequency[kk]
            prjrun.runelectromag(
                em, material, domain
            )
            domain.nx = domain.nx + 2*domain.cpml
            domain.nz = domain.nz + 2*domain.cpml
            
            fnx = glob('Ex.*.dat')
            fnz = glob('Ez.*.dat')
            for filex, filez in zip(fnx, fnz):
                datx += ( read_dat(
                    filex, 'Ex', domain, is_complex = False, single = True
                ) )**2
                datz += ( read_dat(
                    filez, 'Ez', domain, is_complex = False, single = True
                ) )**2
            
            datx[domain.cpml:-(domain.cpml+1),domain.cpml:-(domain.cpml+1)] = 0.0
            datz[domain.cpml:-(domain.cpml+1),domain.cpml:-(domain.cpml+1)] = 0.0
            
            cumulative_power_density_x[ii,jj,kk] = datx.sum() 
            cumulative_power_density_z[ii,jj,kk] = datz.sum() 
            
            # Typically the domain dimensions are changed and stored in an Array
            # object, but in this case we have to manually change them so that 
            # an error isn't returned in runelectromag or read_dat
            domain.nx = domain.nx - 2*domain.cpml
            domain.nz = domain.nz - 2*domain.cpml

# This is setup to compute many different source frequencies. 
cpd_x = cumulative_power_density_x[:,:,0]
cpd_z = cumulative_power_density_z[:,:,0]

# Create the kde plots 
kappa_grid, sigma_grid = np.meshgrid(kappa_max, sig_opt_scalar)
kappa_flat = kappa_grid.ravel() 
sigma_flat = sigma_grid.ravel() 
cpd_x_flat = cpd_x.ravel()
cpd_z_flat = cpd_z.ravel()

fig, axs = plt.subplots(1,2, figsize = (12,6), sharey = True)
sns.kdeplot(x = sigma_flat, y = kappa_flat, weights = cpd_x_flat, fill = True, ax = axs[0], cmap = "Blues")
sns.kdeplot(x = sigma_flat, y = kappa_flat, weights = cpd_z_flat, fill = True, ax = axs[1], cmap = "Blues")

axs[0].set_xlabel(r'$\kappa_{\text{max}}$', fontsize = 16)
axs[0].set_ylabel(r'$\sigma_{\text{max}}/\sigma_{\text{opt}}$', fontsize = 16)
axs[0].set_title('Ex Cumulative Power Density')
axs[1].set_xlabel(r'$\kappa_{\text{max}}$', fontsize = 16)
axs[1].set_ylabel(r'$\sigma_{\text{max}}/\sigma_{opt}$', fontsize = 16)
axs[1].set_title('Ez Cumulative Power Density')
plt.show()

