Common Offset
-------------

This script simulates a typical GPR survey that uses a common offset between the source and single receiver. SeidarT computes a model for each source-receiver pair and extracts the time series prior to deleting all of the model outputs. The source-receiver pairs are in the order that they are given in the CSV files containing their coordinates. There are 3 files that need to be declared. These are the project file, receiver coordinate file, and source coordinate file. 

.. code-block:: python
    
    from seidart.simulations.common_offset import CommonOffset
    from seidart.visualization.im2anim import build_animation

    prjfile = 'common_offset.prj' 
    rcxfile = 'common_offset_receivers.xyz'
    srcfile = 'common_offset_sources.xyz'

    channel = 'Ex'
    complex = False

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

co.gain = 800
co.exaggeration = 0.05
co.sectionplot(plot_complex = complex)
co.save()


    
    


.. code-block:: python
    
    co.gain = 800
    co.exaggeration = 0.05
    co.sectionplot(plot_complex = complex)
    co.save()

    