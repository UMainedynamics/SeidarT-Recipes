import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from seidart.routines.materials import ice_stiffness, ice_density, tensor2velocities

temperature = np.arange(-50, 0, 0.25) # Temperatures that are 
pressure = np.array([0.01, 0.1, 0.4])

n = len(temperature)
m = len(pressure)

vpv = np.zeros([m,n]) 
vph = np.zeros([m,n])
vsv = np.zeros([m,n])
vsh = np.zeros([m,n])

for ii in range(m):
    for jj in range(n):
        C = ice_stiffness(temperature[jj], pressure[ii])
        rho = ice_density(temperature[jj])
        vpv[ii,jj], vph[ii,jj], vsv[ii,jj], vsh[ii,jj] = tensor2velocities(
            C, rho, seismic = True
        )



# ------------------------------------------------------------------------------
alpha_vp = (vpv - vph) / vpv * 100
alpha_vs = (vsh - vsv) / vsh * 100 

print(f'Alpha Vp min/max are {alpha_vp.min()}/{alpha_vp.max()}')
print(f'Alpha Vs min/max are {alpha_vs.min()}/{alpha_vs.max()}')

# ------------------------------------------------------------------------------
vp = (vpv + vph) / 2
vs = (vsh + vsv) / 2 
fig, ax1 = plt.subplots(1, 1, figsize = (7,4) )
#Vpv
wsp = np.round( (vp.max() - vp.min() ) * 0.35 )
ax1.fill_between(temperature, vp[0,:], vp[2,:], color = '#784444', alpha = 0.3)
ax1.plot(temperature, vp[1,:], c = '#ff1f1f', label = r'$V_p$') 
ax1.set_ylim( vp.min() - wsp, vp.max() + wsp )
ax1.set_xlabel(r'Temperature ($^o$C)')
ax1.set_ylabel(r'$V_p$ (m$\cdot$s$^{-1}$)')

ax2 = ax1.twinx() 
wss = np.round( (vs.max() - vs.min() ) * 0.1 )
vs_fill = ax2.fill_between(temperature, vs[0,:], vs[2,:], color = '#2d3457', alpha = 0.3)
vs_line = ax2.plot(temperature, vs[1,:], c = '#1f75ff', label = r'$V_s$') 
ax2.set_ylim(vs.min() - wss, vs.max() + wss)
ax2.set_ylabel(r'$V_s$ (m$\cdot$s$^{-1}$)')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('ps_velocity_comparison_PT.png')
plt.show()

