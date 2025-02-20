import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seidart.routines.materials import *
import plotly.figure_factory as ff 
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Convert data for plotting
def convert_to_ternary_dict(coordinates, values):
    """
    Converts a list of ternary coordinates and corresponding values into a dictionary.
    
    :param coordinates: List of (sand, silt, clay) tuples.
    :param values: List of corresponding scalar values (e.g., Vp or Vs).
    :return: Dictionary with ternary coordinates as keys and values as scalar data.
    """
    return {coord: value for coord, value in zip(coordinates, values)}

def compute_wave_velocities(K, G, rho):
    """
    Computes P-wave and S-wave velocities and density for a given sand-silt-clay mixture.

    :param P: Overburden pressure in Pascals (Pa).
    :param T: Temperature in degrees Celsius.
    :param lwc: Water content fraction (0 to 1).
    :param composition: Dictionary with fractions of sand, silt, and clay (must sum to 1).
    :return: (P-wave velocity in m/s, S-wave velocity in m/s, density in kg/m³)
    """
    # Compute wave velocities
    Vp = np.sqrt((K + (4/3) * G) / rho)
    Vs = np.sqrt(G / rho)
    
    return Vp, Vs


# Generate ternary grid data
resolution = 40
sand_silt_clay_combinations = []
data = []
# Vp_results = {0: [], 30: [], 60: [], 100: []}
# Vs_results = {0: [], 30: [], 60: [], 100: []}

T = 0.0 # Constant temperature at freezing 
overburden_pressure = 910 * 9.81 * 100
porosity = 35 # This an average between deformed tills and non-deformed tills from Fountain and Walder

for sand in np.linspace(0, 1, resolution):
    for silt in np.linspace(0, 1 - sand, resolution):
        clay = 1 - sand - silt
        if clay < 0:
            continue
        sand_silt_clay_combinations.append((sand, silt, clay))
        
        for lwc in [0, 30, 60, 100]:  # % lwc levels
            composition = {"sand": sand, "silt": silt, "clay": clay}
            K, G, rho_eff = moduli_sand_silt_clay(overburden_pressure, T, lwc, porosity = porosity, composition=composition)
            Vp, Vs = compute_wave_velocities(K, G, rho_eff)
            data.append([sand, silt, clay, lwc, Vp, Vs, rho_eff])


df = pd.DataFrame(data, columns=["sand", "silt", "clay", "lwc", "Vp", "Vs", "density"])

# Function to plot ternary diagram using Plotly
# Function to create an interpolated ternary contour plot
# Function to plot ternary contour
def plot_ternary_contour(df, color_col, lwc, colormap = "Blues", ncontours = 20):
    """
    Plots a ternary contour plot with a separate colorbar subplot.
    
    :param df: Pandas DataFrame with sand, silt, clay, lwc, Vp, Vs.
    :param color_col: The column to use for contour mapping (e.g., "Vp" or "Vs").
    :param lwc: Liquid water content (%), selects the appropriate subset.
    :param colormap: The sequential colormap to use.
    """
    # Filter data for selected LWC level
    df_filtered = df[df["lwc"] == lwc]
    
    sand = df_filtered["sand"].values
    silt = df_filtered["silt"].values
    clay = df_filtered["clay"].values
    values = df_filtered[color_col].values
    
    # Create the ternary contour plot
    fig = ff.create_ternary_contour(
        np.array([sand, silt, clay]), 
        values,
        pole_labels=['Sand', 'Silt', 'Clay'],
        colorscale=colormap,
        coloring=None,
        ncontours=ncontours,  # Increase number of contour levels
        showscale = True,
        title=f"{color_col} at {lwc}% Saturation"
    )
    
    # fig.show()
    pio.write_image(fig, f"ternary_plot_{color_col}_{lwc}.png", format="png", scale=2)


# Plot Vp and Vs ternary diagrams for each LWC level
for lwc in [0, 30, 60, 100]:
    plot_ternary_contour(df, "Vp", lwc, colormap = "Viridis", ncontours = 60)
    plot_ternary_contour(df, "Vs", lwc, colormap = "Viridis", ncontours = 60 )