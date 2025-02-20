import numpy as np 
import pandas as pd 
from seidart.routine.materials import * 


temperature = -10 
lwc = np.linspace(0, 100, 40)
porosity = np.linspace(0, 90, 40)

methods = ('Hill', 'Gaussmann', 'SCA')
