import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import copy 
import seaborn as sns

from seidart.routines import prjrun, sourcefunction
from seidart.routines.arraybuild import Array 
from seidart.routines.definitions import * 
from seidart.routines.fabricsynth import Fabric

# ------------------------------------------------------------------------------
# 

# ------------------------------------------------------------------------------

# Create fabrics 
sp1 = {
    'distribution': 'normal', 'npts': [200],
    'strike': [0], 'strike_std': [180], 'skew_strike': [0],
    'dip': [0], 'dip_std': [90], 'skew_dip': [0]
}
sp2 = {
    'distribution': 'normal', 'npts': [200],
    'strike': [0],'strike_std': [180],'skew_strike': [0],
    'dip': [0],'dip_std': [65],'skew_dip': [0]
}
sp3 = {
    'distribution': 'normal','npts': [200],
    'strike': [0],'strike_std': [180],'skew_strike': [0],
    'dip': [0],'dip_std': [40],'skew_dip': [0]
}
sp4 = {
    'distribution': 'normal','npts': [200],
    'strike': [0],'strike_std': [180],'skew_strike': [0],
    'dip': [0],'dip_std': [15],'skew_dip': [0]
}
sp5 = {
    'distribution': 'normal', 'npts': [200],
    'strike': [0],'strike_std': [180],'skew_strike': [0],
    'dip': [0],'dip_std': [5],'skew_dip': [0]
}
sp1 = Fabric(sp1, output_filename = 'single_pole1.txt', plot = False)
sp2 = Fabric(sp2, output_filename = 'single_pole2.txt', plot = False)
sp3 = Fabric(sp3, output_filename = 'single_pole3.txt', plot = False)
sp4 = Fabric(sp4, output_filename = 'single_pole4.txt', plot = False)
sp5 = Fabric(sp5, output_filename = 'single_pole5.txt', plot = False)

# Create a split violin plot to compare the population distributions 
columns = [
    'Homogeneous', 'Very Weak', 'Moderately Weak', 'Moderately Strong', 'Strong'
]
strikes = np.column_stack(
    [sp1.strikes, sp2.strikes, sp3.strikes, sp4.strikes, sp5.strikes]
) * np.pi / 180
dips = np.column_stack( 
    [sp1.dips, sp2.dips, sp3.dips, sp4.dips, sp5.dips] 
) * np.pi / 180

# We want to create a dataframe from the arrays 
population = np.tile(columns, 2*strikes.shape[0])
category = np.repeat(['strike', 'dip'], len(strike) )
data = {
    'Radians': np.concatenate([strikes.reshape(-1,1), dips.reshape(-1, 1) ]).flatten(),
    'Population': population.flatten(),
    'Category': category
}
strikes_and_dips = pd.DataFrame(data)

# Plot the distributions for comparison. 
# Set the theme for the plot
sns.set_theme(style="dark", rc = {'figure.figsize':(9,5)})
ax = sns.violinplot(
    data = strikes_and_dips, 
    x = "Population", y = "Radians", hue = "Category", 
    split = True, inner="quart", palette="muted", scale = 'width'
)
# Update the legend
ax.legend().set_title('')
plt.show()

# ------------------------------------------------------------------------------
# Run a set of seismic simulations for the different fabrics generated above

# Load the project 
prjfile = 'split_material.prj'
domain, material, seismic,__ = prjrun.domain_initialization(prjfile)
timevec, fx, fy, fz, srcfn = sourcefunction(seismic, 10000, 'gaus1')
angfiles = np.array(
    ['single_pole1.txt','single_pole2.txt','single_pole3.txt','single_pole4.txt','single_pole5.txt']
)
output_basefiles = np.array(['sp1', 'sp2', 'sp3', 'sp4', 'sp5'])
material.sort_material_list() 
material.is_anisotropic[1] = True

for ind in range(len(angfiles)):    
    mat = copy.deepcopy(material)
    seis = copy.deepcopy(seismic)
    mat.angfiles = np.array(['', angfiles[ind]])
    prjrun.status_check(
        seis, mat, domain, prjfile
    )
    # P waves
    prjrun.runseismic(
        seis, mat, domain
    )
    z = Array('Vz', prjfile, 'receivers.xyz')
    z.save(output_basefile = output_basefiles[ind] + 'z')
    # Sx waves 
    seis.theta = 0
    prjrun.runseismic(
        seis, mat, domain
    )
    x = Array('Vx', prjfile, 'receivers.xyz')
    x.save(output_basefile = output_basefiles[ind] + 'x')
    # Sy waves 
    seis.phi = 90 
    prjrun.runseismic(
        seis, mat, domain
    )
    y = Array('Vx', prjfile, 'receivers.xyz')
    y.save(output_basefile = output_basefiles[ind] + 'y')




# Set the anisotropic booleans to True and define 

# ------------------------------------------------------------------------------
