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

We are ready to run the model. We have the option to compute the complex valued model for the electric field. For ice and snow, the complex permittivity is already calculated and used to compute the conductivity. No other materials have a complex representation so we will not use the complex solver. 

.. code-block:: python
    
    # The non-complex equations aren't necessary but are also a solution to the PDE
    complex_values = False
    prjrun.runelectromag(em, mat, dom, use_complex_equations = complex_values)

Next we want to initiate the Array object and load the time series for each receiver in the receiver file. We need to specify the channel. For the section plot, we will apply an auto-gain control to each time series to accomodate geometric spreading and attenuation. For this example, we will use a window of 1/3 the time series. The section plot will likely have quite a few more time steps than receivers so we want to apply an exaggeration to make the plot 

.. code-block:: python
    
    channel = 'Ex'
    # Create the array object
    array_ex = Array(channel, prjfile, rcxfile, is_complex = complex_values)
    # Add an AGC function for visualization
    array_ex.gain = int(em.time_steps/3)
    # We need to scale the axes
    array_ex.exaggeration = 0.1
    # Create the plot 
    array_ex.sectionplot(
        plot_complex = False
    )

.. note:: 
    
    If the .dat files already exist, you can skip the status check, creating the source function, and running the model unless you have edited the project file. 

A single trace can be plotted from the list of receivers by specifying the integer value/index of the receiver. Additional *matplotlib* arguments for Axes.plot can be passed. You can refer to the `*matplotlib.pyplot.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot>`_ documentation for more details.  

.. code-block:: python
    
    # Let's plot a trace for the 10th receiver in the list of receivers. 
    receiver_number = 10
    array_ex.wiggleplot(receiver_number, figure_size = (5,8))

Save the array object as a pickle (.pkl) file. This will store all of the information from the object, as well as, create a .csv file of the array time series. The .csv file will contain each time series per column and each row is a time step. 

.. code-block:: python
    
    # Pickle the object
    array_ex.save()

We can create a GIF animation to visualize the wavefield. We need to specify the delay between frames and the frame interval. The frame interval is the number of time steps between each frame. A smaller number will create a larger GIF file, and appear to be slower unless the frame delay is lowered. A larger frame interval will appear to jump between time steps. The alpha value is the transparency in the background image of model on which the electric field amplitude is overlayn. 

.. code-block:: python  
    
    # Create the GIF animation so we can 
    frame_delay = 10
    frame_interval = 10 
    alpha_value = 0.3
    
    # Create the GIF so that we can view the wavefield
    build_animation(
            prjfile, 
            channel, frame_delay, frame_interval, alpha, 
            is_complex = complex_values, 
            is_single_precision = True
    )


Finally, we can do the same plotting and create the animation for the vertical electric field. The only thing we need to change is the channel. 

.. code-block:: python
 
    # --------------------------------------------------------------------------
    # We can do the same for the vertical electric field as above
    channel = 'Ez'
    array_ez = Array(channel, prjfile, rcxfile, is_complex = complex_values)
    array_ez.gain = int(em.time_steps/3)
    array_ez.exaggeration = 0.1
    array_ez.sectionplot(
        plot_complex = False
    )
    
    array_ez.wiggleplot(receiver_number, figure_size = (5,8))
    array_ex.save()
    
    build_animation(
            prjfile, 
            channel, frame_delay, frame_interval, alpha, 
            is_complex = complex_values, 
            is_single_precision = True,
    )