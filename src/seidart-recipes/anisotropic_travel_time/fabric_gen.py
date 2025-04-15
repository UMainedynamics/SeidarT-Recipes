
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from seidart.routines.fabricsynth import Fabric

# ------------------------------------------------------------------------------

# Create fabrics 
sp1 = {
    'distribution': 'normal', 
    'npts': [200],
    'trend': [0], 'trend_std': [180], 'skew_trend': [0],
    'plunge': [0], 'plunge_std': [90], 'skew_plunge': [0],
    'orientation': [0], 'orientation_std': [180], 'skew_orientation': [0]
}
sp2 = {
    'distribution': 'normal', 'npts': [200],
    'trend': [0],'trend_std': [180],'skew_trend': [0],
    'plunge': [0],'plunge_std': [65],'skew_plunge': [0],
    'orientation': [0], 'orientation_std': [180], 'skew_orientation': [0]
}
sp3 = {
    'distribution': 'normal','npts': [200],
    'trend': [0],'trend_std': [180],'skew_trend': [0],
    'plunge': [0],'plunge_std': [40],'skew_plunge': [0],
    'orientation': [0], 'orientation_std': [180], 'skew_orientation': [0]
}
sp4 = {
    'distribution': 'normal','npts': [200],
    'trend': [0],'trend_std': [180],'skew_trend': [0],
    'plunge': [0],'plunge_std': [15],'skew_plunge': [0],
    'orientation': [0], 'orientation_std': [180], 'skew_orientation': [0]
}
sp5 = {
    'distribution': 'normal', 'npts': [200],
    'trend': [0],'trend_std': [180],'skew_trend': [0],
    'plunge': [0],'plunge_std': [5],'skew_plunge': [0],
    'orientation': [0], 'orientation_std': [180], 'skew_orientation': [0]
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
trends = np.column_stack(
    [sp1.trends, sp2.trends, sp3.trends, sp4.trends, sp5.trends]
) * np.pi / 180
plunges = np.column_stack( 
    [sp1.plunges, sp2.plunges, sp3.plunges, sp4.plunges, sp5.plunges] 
) * np.pi / 180

# We want to create a dataframe from the arrays 
population = np.tile(columns, 2*trends.shape[0])
category = np.repeat(['trend', 'plunge'], np.prod(trends.shape) )
data = {
    'Radians': np.concatenate([trends.reshape(-1,1), plunges.reshape(-1, 1) ]).flatten(),
    'Population': population.flatten(),
    'Category': category
}
trends_and_plunges = pd.DataFrame(data)

# Plot the distributions for comparison. 
# Set the theme for the plot
sns.set_theme(style="dark", rc = {'figure.figsize':(9,5)})
ax = sns.violinplot(
    data = trends_and_plunges, 
    x = "Population", y = "Radians", hue = "Category", 
    split = True, inner="quart", palette="muted", density_norm = 'width'
)
# Update the legend
ax.legend().set_title('')
plt.show()