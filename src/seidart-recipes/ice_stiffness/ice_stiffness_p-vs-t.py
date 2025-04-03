import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from seidart.routines.materials import ice_stiffness, ice_density, tensor2velocities


temperature = np.arange(-50, 0, 0.25) # Temperatures that are 
pressure = np.arange(0.01, 0.20, 0.005) #From 1 atm at the surface to the overburden at approximately 1000m
temperature_mesh, pressure_mesh = np.meshgrid(temperature, pressure)

rho = 917
n = len(temperature)
m = len(pressure)

vpv = np.zeros([m,n]) 
vph = np.zeros([m,n])
vsv = np.zeros([m,n])
vsh = np.zeros([m,n])

# ------------------------------------------------------------------------------
c_ref = ice_stiffness(temperature.max(), pressure.min() )
vpv_ref, vph_ref, vsv_ref, vsh_ref = tensor2velocities(
    c_ref, ice_density( temperature.max() ), seismic = True
)

for ii in range(m):
    for jj in range(n):
        C = ice_stiffness(temperature[jj], pressure[ii])
        rho = ice_density(temperature[jj])
        vpv[ii,jj], vph[ii,jj], vsv[ii,jj], vsh[ii,jj] = tensor2velocities(
            C, rho, seismic = True
        )

vpv_pct = (1 - vpv_ref/vpv)*100
vph_pct = (1 - vph_ref/vph)*100
vsv_pct = (1 - vsv_ref/vsv)*100
vsh_pct = (1 - vsh_ref/vsh)*100

dvpv = vpv - vpv_ref 
dvph = vph - vph_ref 
dvsv = vsv - vsv_ref 
dvsh = vsh - vsh_ref
# ------------------------------------------------------------------------------
# Create the plots 
xlab = 'Temperature (C)' 
ylab = 'Pressure (kbar)'
cmap_name = 'magma'

# Plot the subplots
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

# Vpv
sns.heatmap(
    dvpv, 
    ax=axes1[0, 0], 
    cmap=cmap_name, 
    cbar_kws={'label': r'$\Delta V_{pv}$ (m/s)'}
)
axes1[0, 0].set_title('Vpv')
axes1[0, 0].set_ylabel('Pressure (MPa)')
axes1[0, 0].set_xlabel(r'Temperature ($^o$C)')
axes1[0, 0].set_xticks(np.linspace(0, n-1, 5))
axes1[0, 0].set_xticklabels(np.round(np.linspace(temperature.min(), temperature.max(), 5), 2))
axes1[0, 0].set_yticks(np.linspace(0, m-1, 5))
axes1[0, 0].set_yticklabels(np.round(np.linspace(pressure.min(), pressure.max(), 5), 2))
axes1[0, 0].invert_yaxis()

# Vph
sns.heatmap(
    dvph, 
    ax=axes1[0, 1], 
    cmap=cmap_name, 
    cbar_kws={'label': r'$\Delta V_{ph}$ (m/s)'}
)
axes1[0, 1].set_title('Vph')
axes1[0, 1].set_xlabel('Pressure (MPa)')
axes1[0, 1].set_xlabel(r'Temperature ($^o$C)')
axes1[0, 1].set_xticks(np.linspace(0, n-1, 5))
axes1[0, 1].set_xticklabels(np.round(np.linspace(temperature.min(), temperature.max(), 5), 2))
axes1[0, 1].set_yticks(np.linspace(0, m-1, 5))
axes1[0, 1].set_yticklabels(np.round(np.linspace(pressure.min(), pressure.max(), 5), 2))
axes1[0, 1].invert_yaxis()

# Vsv
sns.heatmap(
    dvsv, 
    ax=axes1[1, 0], 
    cmap=cmap_name, 
    cbar_kws={'label': r'$\Delta V_{sv}$ (m/s)'}
)
axes1[1, 0].set_title('Vsv')
axes1[1, 0].set_xlabel('Pressure (MPa)')
axes1[1, 0].set_xlabel(r'Temperature ($^o$C)')
axes1[1, 0].set_xticks(np.linspace(0, n-1, 5))
axes1[1, 0].set_xticklabels(np.round(np.linspace(temperature.min(), temperature.max(), 5), 2))
axes1[1, 0].set_yticks(np.linspace(0, m-1, 5))
axes1[1, 0].set_yticklabels(np.round(np.linspace(pressure.min(), pressure.max(), 5), 2))
axes1[1, 0].invert_yaxis()

# Vsh
sns.heatmap(
    dvsh, 
    ax=axes1[1, 1], 
    cmap=cmap_name, 
    cbar_kws={'label': r'$\Delta V_{sh}$ (m/s)'}
)
axes1[1, 1].set_title('Vsh')
axes1[1, 1].set_xlabel('Pressure (MPa)')
axes1[1, 1].set_xlabel(r'Temperature ($^o$C)')
axes1[1, 1].set_xticks(np.linspace(0, n-1, 5))
axes1[1, 1].set_xticklabels(np.round(np.linspace(temperature.min(), temperature.max(), 5), 2))
axes1[1, 1].set_yticks(np.linspace(0, m-1, 5))
axes1[1, 1].set_yticklabels(np.round(np.linspace(pressure.min(), pressure.max(), 5), 2))
axes1[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# Plot the subplots
alpha_vp = (vph - vpv) / vpv * 100
alpha_vs = (vsh - vsv) / vsv * 100 
fig2, axes2 = plt.subplots(2, 1, figsize=(7, 10))

# Vpv
contour_vp = axes2[0].contourf(
    temperature_mesh, 
    pressure_mesh, 
    alpha_vp, 
    30,
    cmap = cmap_name,
    linewidths = 0.5
)
axes2[0].set_ylabel('Pressure (MPa)')
axes2[0].set_xlabel(r'Temperature ($^o$C)')
axes2[0].invert_yaxis()
cbar_vp = fig2.colorbar(contour_vp, ax=axes2[0])
cbar_vp.set_label(r'\alpha_{V_p} ($\%$)')

# Vsv
contour_levels = [5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]
contour_vs = axes2[1].contourf(
    temperature_mesh, 
    pressure_mesh, 
    alpha_vs, 
    30,
    cmap = cmap_name, 
    linewidths = 0.5
)
axes2[1].set_ylabel('Pressure (MPa)')
axes2[1].set_xlabel(r'Temperature ($^o$C)')
axes2[1].invert_yaxis()
cbar_vs = fig2.colorbar(contour_vs, ax=axes2[1])
cbar_vs.set_label(r'\alpha_{V_s} ($\%$)')

plt.tight_layout()
plt.show()

