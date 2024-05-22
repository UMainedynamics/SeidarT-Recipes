Single Source/Common Midpoint
-----------------------------

The Python script, *single_source.py*, is designed to perform electromagnetic simulations using the *seidart* library. The script initializes the model and domain objects, computes permittivity coefficients, runs electromagnetic simulations, and visualizes the results through animations. 

Here is a breakdown of the program. The project file needs to be built and required values filled in. 

We need to import the necessary modules to create the source function, run the model, create a receiver array from the model outputs, and build a GIF animation.

.. code-block:: python
    
    from seidart.routines import prjrun, sourcefunction
    from seidart.routines.arraybuild import Array
    from seidart.visualization.im2anim import build_animation

Define files and load the values of the project file into their respective class variables. 

.. code-block:: python

    # Define the necessary files. Adjust the relative paths if necessary. 
    prjfile = 'single_source.prj' 
    rcxfile = 'receivers.xyz'

    # Initiate the model and domain objects
    dom, mat, seis, em = prjrun.domain_initialization(prjfile)


The *status_check* function will check to make sure that all fields have been completed for the model parameters. If not, it will return a message for which group of parameters are incomplete. If all parameters have been provided, then the tensor coefficients to the model object (seismic or electromagnetic use the same Model object class) will be computed. These will then overwrite the values in the single_source.prj file. In order to suppress overwriting then you will need to flag the *append_to_prjfile* to False. 

.. code-block:: python
    
    # Compute the permittivity coefficients and check to make sure the project file has all required values
    prjrun.status_check(
        em, mat, dom, prjfile, seismic = False, append_to_prjfile = True
    )

Before running the model, the necessary .dat files need to be created for the time series of the source function in each direction, x,y, and z. This needs to be built 

.. code-block:: python
 
    # Create the source function
    timevec, fx, fy, fz, srcfn = sourcefunction(em, 10, 'gaus1', 'e')

We are ready to run the model. We have the option to compute the complex valued model for the electric field. 
.. code-block:: python
    
    # The non-complex equations aren't necessary but are also a solution to the PDE
    complex_values = False
    prjrun.runelectromag(em, mat, dom, use_complex_equations = complex_values)

.. code-block:: python
 
    # Create the array object
    array_ex = Array('Ex', prjfile, rcxfile, is_complex = complex_values)
    # Add an AGC function for visualization
    array_ex.gain = int(em.time_steps/3)
    # We need to scale the axes
    array_ex.exaggeration = 0.1
    # Create the plot 
    array_ex.sectionplot(
        plot_complex = False
    )
    

.. code-block:: python  
    # Create the GIF so that we can view the wavefield
    build_animation(
            prjfile, 
            'Ex', 10, 10, 0.3, 
            is_complex = complex_values, 
            is_single_precision = True
    )

.. code-block:: python
 
    # --------------------------------------------------------------------------
    # We can do the same for the vertical electric field as above
    array_ez = Array('Ez', prjfile, rcxfile, is_complex = complex_values)
    array_ez.gain = int(em.time_steps/3)
    array_ez.exaggeration = 0.1
    array_ez.sectionplot(
        plot_complex = False
    )
    build_animation(
            prjfile, 
            'Ex', 10, 10, 0.3, 
            is_complex = complex_values, 
            is_single_precision = True,
            plottype = 'energy_density'
    )