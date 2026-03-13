import numpy as np 
import matplotlib.pyplot as plt 
from seidart.routines.materials import * 



def reflection_amplitude(n1, n2, theta1, theta2):
    Rs = np.abs( 
        (n1 * np.cos(theta1) - n2 * np.cos(theta2) ) /\
        (n1 * np.cos(theta1) + n2 * np.cos(theta2) )
    ) ** 2
    Rp = np.abs( 
        (n1 * np.cos(theta2) - n2 * np.cos(theta1) ) /\
        (n1 * np.cos(theta2) + n2 * np.cos(theta1) )
    ) ** 2
    return(Rs, Rp) 


# Calculate n1 and n2

theta1 = np.linspace(-np.pi, np.pi)
theta2 = np.linspace(0, 2*np.pi)