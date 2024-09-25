import numpy as np 
import pandas as pd
from scipy import signal
import scipy.fft as sf 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

from seidart.routines import prjrun
from seidart.routines.definitions import * 

# Function to format the ticks in engineering notation
def engineering_formatter(x, pos):
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(x))))
    base = x / 10**exponent
    return r"${:.1f}e{:d}$".format(base, exponent)

# ------------------------------------------------------------------------------


prjfile = 'stiffness_calculation.prj'
domain, material, seismic, electromag = prjrun.domain_initialization(prjfile)

prjrun.status_check(
    seismic, material, domain, prjfile, append_to_prjfile = True
)

tensor_labels = np.array(
    [
        r'$c_{11}$',  r'$c_{12}$',  r'$c_{13}$',  r'$c_{14}$',  r'$c_{15}$',  r'$c_{16}$', 
                    r'$c_{22}$',  r'$c_{23}$',  r'$c_{24}$',  r'$c_{25}$',  r'$c_{26}$',
                                r'$c_{33}$',  r'$c_{34}$',  r'$c_{35}$',  r'$c_{36}$',
                                            r'$c_{44}$',  r'$c_{45}$',  r'$c_{46}$',
                                                        r'$c_{55}$',  r'$c_{56}$',
                                                                    r'$c_{66}$'
    ]
)
tensor_colors = np.array(['#ff0040', '#8bad1a', '#f03ac8', '#1e3282'])

coeffs = seismic.tensor_coefficients[:, 1:-1]


# ------------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, sharex=True, figsize=(11, 4), gridspec_kw={'hspace': 0.08}
)

# Set up x-axis labels
x = np.arange(len(tensor_labels))
x_offset = np.array([-0.21, -0.07, 0.07, 0.21 ])
markers = ['o', 's', 'v', 'D']

# Labels for the legend
legend_labels = ['Multipole 1', 'Multipole 2', 'Multipole 3', 'Ref. Single Pole']

# Plot each row of coefficients with different colors
m, n = coeffs.shape
for ind in range(m):
    ax1.stem(
        x + x_offset[ind], coeffs[ind, :], 
        markerfmt=markers[ind], linefmt=tensor_colors[ind % len(tensor_colors)],
        basefmt = ' ', label=legend_labels[ind]# if ind == 0 else None
    )
    
    ax2.stem(
        x + x_offset[ind], coeffs[ind, :], 
        markerfmt=markers[ind], linefmt=tensor_colors[ind % len(tensor_colors)],
        basefmt = ' '
    )
    
    ax3.stem(
        x + x_offset[ind], coeffs[ind, :], 
        markerfmt=markers[ind], linefmt=tensor_colors[ind % len(tensor_colors)],
        basefmt = ' '
    )

# Set y-limits for each subplot
ax1.set_ylim(1.0e10, 1.52e10)
ax2.set_ylim(2.2e9, 8.5e9)
ax3.set_ylim(-1e8, 1.5e8)

# Add the tensor labels to the x-axis for the bottom plot
ax3.set_xticks(x)
ax3.set_xticklabels(tensor_labels, rotation=45)

# Format the y-axis to use offset notation and adjust label padding
# Format the y-axis to use engineering notation and set custom tick values
# for ax in [ax1, ax2, ax3]:
#     ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=True, useMathText=True))
#     ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))  # Enable engineering notation

# Format the y-axis using engineering notation directly on the tick labels
formatter = FuncFormatter(engineering_formatter)
ax1.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_major_formatter(formatter)
ax3.yaxis.set_major_formatter(formatter)

# Set specific tick values for each subplot
ax1.set_yticks([1.1e10, 1.3e10, 1.5e10])
ax2.set_yticks([2.5e9, 4e9, 6e9, 8e9])
ax3.set_yticks([-1e8, -0.5e8, 0, 0.5e8, 1e8])

# Add more space between the axis labels and the ticks
ax1.tick_params(axis='y', labelsize=9, pad=15)
ax2.tick_params(axis='y', labelsize=9, pad=15)
ax3.tick_params(axis='y', labelsize=9, pad=15)

# ------------------------------------------------------------------------------
# Hide the spines between the plots to create the break effect
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)

# Add diagonal lines to show the breaks
d = .015  # How big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal

kwargs.update(transform=ax2.transAxes)  # Switch to the middle subplot
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal for top break
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal for top break
ax2.plot((-d, +d), (-d, +d), **kwargs)        # Top-left diagonal for bottom break
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Top-right diagonal for bottom break

kwargs.update(transform=ax3.transAxes)  # Switch to the bottom subplot
ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Bottom-left diagonal
ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Bottom-right diagonal

# Adjust the grid lines for subtlety
for ax in [ax1, ax2, ax3]:
    ax.grid(True, linestyle='--', color='gray', alpha=0.2)

# Add legend to the first subplot
ax1.legend(loc='upper right', fontsize=8)

# Display the plot with better layout management
plt.tight_layout()
# plt.show()
plt.savefig('stiffness_tensor_stemplot.png', transparent = True, dpi = 400)