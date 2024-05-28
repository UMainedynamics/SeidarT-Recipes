Common Offset
-------------

This script simulates a typical GPR survey that uses a common offset between the source and single receiver. SeidarT computes a model for each source-receiver pair and extracts the time series prior to deleting all of the model outputs. The source-receiver pairs are in the order that they are given in the CSV files containing their coordinates. There are 3 files that need to be declared. These are the project file, receiver coordinate file, and source coordinate file. 

First, we need to import necessary modules and define common variables. We will be simulating a common offset survey of a radial source and receiver orientation. We are interested in the electric field in the x-direction. 

.. code-block:: python
    
    from seidart.simulations.common_offset import CommonOffset
    from seidart.visualization.im2anim import build_animation

    prjfile = 'common_offset.prj' 
    rcxfile = 'common_offset_receivers.xyz'
    srcfile = 'common_offset_sources.xyz'

    channel = 'Ex'
    complex = False

We can initialize the object from the *CommonOffset* class which inherits much of its functionality from the *Array* class in *arraybuild*. We will use the receiver and source locations given in meters and not indices. We will not use the complex solver, and we want single precision outputs to limit hard drive requirements. After we have intialized the object, we can simply run it with the *co_run* function. 

.. code-block:: python

    co = CommonOffset(
        srcfile, 
        channel, 
        prjfile, 
        rcxfile, 
        receiver_indices = False, 
        is_complex = complex,
        single_precision = True
    )

    co.co_run(seismic = False)
    
Each source and receiver pair will run the model, pull the receiver locations, then delete all of the .dat files associated with the model. Following the termination of *co_run*, we will have an array built for each receiver-source pair which we can plot. Similarly to the common midpoint survey in `single_source.py <https://github.com/UMainedynamics/SeidarT-Recipes/blob/main/src/seidart-recipes/single_source/single-source-readme.rst>`_. The save function is inherited from the *Array* class and produces a CSV and pickle file for each receiver pair and the CommonOffset object, respectively.

.. code-block:: python
    
    co.gain = 800
    co.exaggeration = 0.05
    co.sectionplot(plot_complex = complex)
    co.save()

