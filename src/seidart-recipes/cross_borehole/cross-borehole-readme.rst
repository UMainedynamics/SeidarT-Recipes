Cross Borehole 
--------------

The cross borehole method is a geophysical technique that uses two boreholes to measure the velocity of seismic or electromagnetic waves between them. The program *cross-borehole.py* creates a simple 2.5D model with a vertical source at x=1m and a vertical receiver array at x=9m. For a tomography problem, the source can be moved via a for loop and array outputs saved similar to the common offset recipe. This script will evolve into a simulation module that will be contained in SeidarT. 

The script presents a straightforward workflow. The primary components are broken down as follows:

1. Import the necessary libraries and modules, and define the project and receiver files.

.. code-block:: python

    import numpy as np 

    from seidart.routines import prjrun, sourcefunction 
    from seidart.routines.arraybuild import Array 
    from seidart.visualization.slice25d import slice

    prjfile = 'cross_borehole.prj'
    rcxfile = 'receivers.xyz'
    is_complex = False


2. Initialize and prep the model then run the simulation. 

.. code-block:: python

    # Prep and run the model
    domain, material, __, em = prjrun.domain_initialization(prjfile)
    prjrun.status_check(em, material, domain, prjfile)
    tv, fx, fy, fz, srcfn = sourcefunction(em, 1e6, 'gaus1')

    prjrun.runelectromag(
        em, material, domain, use_complex_equations = is_complex
    )

3. Visualize the wavefield and receiver data. 
    a. Slice the wavefield at a specific indice in the xz plane. 
    b. Create the GIF and repeat for the other electric field directions.
    c. Initialize the array and build the set of time series. 
    d. Plot the section plot for each electric field direction. 

.. code-block:: python
	
    plane = 'xz' 
    indslice = 25 + domain.cpml #Inline with the source and corrected for the CPML

    # GIF parameters 
    num_steps = 10
    alpha = 0.3 
    delay = 10 

    slice(prjfile, 'Ex', indslice, num_steps, plane, alpha, delay)
    slice(prjfile, 'Ey', indslice, num_steps, plane, alpha, delay)
    slice(prjfile, 'Ez', indslice, num_steps, plane, alpha, delay)

    # ------------------------------------------------------------------------------
    # Visualize the receiver data
    arr_x = Array('Ex', prjfile, rcxfile) 
    arr_y = Array('Ey', prjfile, rcxfile)
    arr_z = Array('Ez', prjfile, rcxfile)

    arr_x.gain = 31 
    arr_y.gain = 31
    arr_z.gain = 31
    arr_x.exaggeration = 0.1 
    arr_y.exaggeration = 0.1
    arr_z.exaggeration = 0.1    

    arr_x.sectionplot()
    arr_y.sectionplot()
    arr_z.sectionplot()

