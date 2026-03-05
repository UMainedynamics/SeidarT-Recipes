import numpy as np 
import matplotlib.pyplot as plt 

from seidart.routines.materials import * 

temperature = -10
eps0 = 8.854e-12 
d = 0.03 # Layer thickness
c = 299792458
N = 120 
M = 80
theta = np.linspace(0, np.pi/4, num = M)
freq = np.linspace(10, 400, num = N)*1e6


dirty_perm = sand_silt_clay_permittivity_conductivity(
    temperature, 0, 0, {"sand": 0.20, "silt": 0.60, "clay": 0.20 })


rTE = np.zeros([M, N], dtype = complex)
rTM = rTE.copy() 
phi = np.zeros([M], dtype = complex)
w = 0.1

for ii, theta1 in enumerate(theta):
    for jj, f in enumerate(freq):
        
        epsilon = ice_permittivity(temperature, 910, f)
        epsilon_eff = np.trace(epsilon) / 3
        refr1 = np.sqrt(epsilon_eff)
        
        dirty_perm_c = dirty_perm[1] / (2.0 * np.pi * f * eps0) 
        refr2 = np.sqrt( np.complex128(dirty_perm[0], -dirty_perm_c) )
        refr2 = (1 - w ) * refr1 + w * refr2
        
        theta2 = np.arcsin( (refr1 * np.sin(theta1) / refr2 )).real
        theta3 = np.arcsin( (refr2 * np.sin(theta2) / refr1 )).real
        
        rte12 = (refr1 * np.cos(theta1) - refr2 * np.cos(theta2) )/ \
                (refr1 * np.cos(theta1) + refr2 * np.cos(theta2) )
        rte23 = (refr2 * np.cos(theta2) - refr1 * np.cos(theta3) )/ \
                (refr2 * np.cos(theta2) + refr1 * np.cos(theta3) )
        
        rtm12 = (refr2 * np.cos(theta1) - refr1 * np.cos(theta2) )/ \
                (refr2 * np.cos(theta1) + refr1 * np.cos(theta2) )
        rtm23 = (refr1 * np.cos(theta2) - refr2 * np.cos(theta3) )/ \
                (refr1 * np.cos(theta2) + refr2 * np.cos(theta3) )
        
        
        beta2 = 2.0 * np.pi * refr2 * d * np.cos(theta2) / (c / f ) 
        phase = np.exp(2j * beta2)
        rTE[ii,jj] = (rte12 + rte23 * phase) / (1 + rte12 * rte23 * phase)
        rTM[ii,jj] = (rtm12 + rtm23 * phase) / (1 + rtm12 * rtm23 * phase)
        

R_TE = np.abs(rTE)** 2 
R_TM = np.abs(rTM)** 2 
phaseTE = np.angle(rTE) 
phaseTM = np.angle(rTM)


def fmt_colorbar(cb, label, power_limits=(-2, 2)):
    cb.set_label(label)
    fmt = mtick.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits(power_limits)  # e.g. show 10^k factor if outside this range
    cb.ax.yaxis.set_major_formatter(fmt)
    cb.update_ticks()


theta_grid, freq_grid = np.meshgrid(theta*180/np.pi, freq*1e-6, indexing='ij')
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

# 1) R_TE
cmap1 = 'Reds'
cmap2 = 'seismic'
im0 = axes[0, 0].pcolormesh(
    freq_grid, theta_grid, R_TE, shading='auto', cmap=cmap1
)
# axes[0, 0].set_xscale('log')
axes[0, 0].set_xlabel('Frequency (MHz)')
axes[0, 0].set_ylabel('Dip / Incidence (deg)')
cb0 = fig.colorbar(im0, ax=axes[0, 0])
fmt_colorbar(cb0, r'$R_{TE}$')
# axes[0, 0].text(
#     0.98, 0.2, r'$\times 10^{-3}$'
# )


# 2) R_TM
im1 = axes[0, 1].pcolormesh(
    freq_grid, theta_grid, R_TM, shading='auto', cmap=cmap1
)
# axes[0, 1].set_xscale('log')
axes[0, 1].set_xlabel('Frequency (MHz)')
axes[0, 1].set_ylabel('Dip / Incidence (deg)')
cb1 = fig.colorbar(im1, ax=axes[0, 1])
fmt_colorbar(cb1, r'$R_{TM}$')
# axes[0, 1].text(
#     1.96, 0.02, r'$\times 10^{-4}$'
# )


# 3) phase of r_TE
im2 = axes[1, 0].pcolormesh(
    freq_grid, theta_grid, phaseTE, shading='auto', cmap=cmap2
)
# axes[1, 0].set_xscale('log')
axes[1, 0].set_xlabel('Frequency (MHz)')
axes[1, 0].set_ylabel('Dip / Incidence (deg)')
cb2 = fig.colorbar(im2, ax=axes[1, 0])
fmt_colorbar(cb2, r'$\phi_{TE}$ (rad)')

# 4) phase of r_TM
im3 = axes[1, 1].pcolormesh(
    freq_grid, theta_grid, phaseTM, shading='auto', cmap=cmap2
)
# axes[1, 1].set_xscale('log')
axes[1, 1].set_xlabel('Frequency (MHz)')
axes[1, 1].set_ylabel('Dip / Incidence (deg)')
cb3 = fig.colorbar(im3, ax=axes[1, 1])
fmt_colorbar(cb3, r'$\phi_{TM}$ (rad)')

plt.show()