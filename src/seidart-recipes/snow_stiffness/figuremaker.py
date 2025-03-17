import numpy as np
import matplotlib.pyplot as plt

# Import the snow_stiffness function from your materials file.
from seidart.routines.materials import *

# ---------------------------
# Parameters to explore
# ---------------------------

# Porosity range in percent (0% to 90%)
m = 100 
n = 100
porosities = np.linspace(0, 90, m)

# Liquid water content (lwc) range in percent
lwc = np.linspace(0, 90, n)
  
# Fixed temperature (°C) for this plot; adjust as needed
temperature = -10  

# Methods to compare: "Hill", "Gassmann", and "SCA"
methods = ["Hill", "Gassmann", "SCA"]

# Preallocate arrays for velocities.
# Dimensions: (porosity, lwc, method)
vp = np.zeros((m, n, len(methods)))
vs = np.zeros((m, n, len(methods)))

# Compute velocities for each method, porosity, and lwc
for kk in range(len(methods)):
    for ii in range(m):
        for jj in range(n):
            # The snow_stiffness function returns: K_snow, G_snow, C_snow, rho_eff
            K, G, C, rho_eff = snow_stiffness(
                temperature=temperature, 
                lwc=lwc[jj], 
                porosity=porosities[ii], 
                pressure=0.0, 
                method=methods[kk],
                epsilon=2e-8,  # if your function accepts an epsilon parameter
                exponent=5,
                small_denominator=1e-9
            )
            # tensor2velocities returns (vp, __, vs, __)
            vp[ii, jj, kk], __, vs[ii, jj, kk], __ = tensor2velocities(C, rho_eff)

# Extract arrays for each method:
vp_hill      = vp[:, :, 0]
vp_gassmann  = vp[:, :, 1]
vp_sca       = vp[:, :, 2]
vs_hill      = vs[:, :, 0]
vs_gassmann  = vs[:, :, 1]
vs_sca       = vs[:, :, 2]

# Determine common color limits for Vp and Vs across all methods.
vp_min = min(vp_hill.min(), vp_gassmann.min(), vp_sca.min())
vp_max = max(vp_hill.max(), vp_gassmann.max(), vp_sca.max())
vs_min = min(vs_hill.min(), vs_gassmann.min(), vs_sca.min())
vs_max = max(vs_hill.max(), vs_gassmann.max(), vs_sca.max())

# Create a meshgrid for plotting (porosity on y-axis, lwc on x-axis)
P_grid, L_grid = np.meshgrid(porosities, lwc, indexing='ij')

# Set manuscript font size (approx. 12 pt)
plt.rcParams.update({'font.size': 12})

# Create figure with 2 rows (Vp, Vs) and 3 columns (Hill, Gassmann, SCA) using constrained_layout
fig, axs = plt.subplots(2, 3, figsize=(9,7.5), constrained_layout=True)
colormap_name = 'cividis'

# Define subplot labels for top and bottom rows.
top_labels = ["a", "b", "c"]
bot_labels = ["d", "e", "f"]

# ---------------------------
# Top row: Vp plots
data_vp = [vp_hill, vp_gassmann, vp_sca]
for j in range(3):
    ax = axs[0, j]
    cfp = ax.contourf(L_grid, P_grid, data_vp[j], levels=30, cmap=colormap_name,
                     vmin=vp_min, vmax=vp_max)
    c_lines = ax.contour(L_grid, P_grid, data_vp[j], levels=15, colors='k', linewidths=0.5)
    ax.clabel(c_lines, inline=True, fontsize=10, fmt="%.0f")
    ax.set_xlabel("LWC (%)")
    if j == 0:
        ax.set_ylabel("Porosity (%)")
    # Place subplot label (a, b, or c) in the left margin
    ax.text(-0.2, 1.0, top_labels[j], transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='right')

# ---------------------------
# Bottom row: Vs plots
data_vs = [vs_hill, vs_gassmann, vs_sca]
for j in range(3):
    ax = axs[1, j]
    cfs = ax.contourf(L_grid, P_grid, data_vs[j], levels=30, cmap=colormap_name,
                     vmin=vs_min, vmax=vs_max)
    c_lines = ax.contour(L_grid, P_grid, data_vs[j], levels=15, colors='k', linewidths=0.5)
    ax.clabel(c_lines, inline=True, fontsize=10, fmt="%.0f")
    ax.set_xlabel("LWC (%)")
    if j == 0:
        ax.set_ylabel("Porosity (%)")
    # Place subplot label (d, e, or f) in the left margin
    ax.text(-0.2, 1.0, bot_labels[j], transform=ax.transAxes, fontsize=14,
            fontweight='bold', va='top', ha='right')

# Add a common colorbar for the top row (Vp)
cbar_vp = fig.colorbar(cfp, ax=axs[0, :], orientation='vertical', pad=0.02)
cbar_vp.set_label("Vp (m/s)")

# Add a common colorbar for the bottom row (Vs)
cbar_vs = fig.colorbar(cfs, ax=axs[1, :], orientation='vertical', pad=0.02)
cbar_vs.set_label("Vs (m/s)")

fig.savefig('snow_seismic_velocity_method_comparison.eps', format = 'eps', dpi = 300)
# plt.show()
