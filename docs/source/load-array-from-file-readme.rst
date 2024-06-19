Load Array From File 
--------------------

The python script, *load_from_file.py*, provides the details for loading array objects that have been previously saved. An Array object contains a save function. Both a pickle formatted file and CSV file are written in the local directory. See the `save function <https://umainedynamics.github.io/SeidarT/docs/build/html/seidart.routines.arraybuild.html>`_ for more details. A CommonOffset object inherits the Array object so the process is also the same. 

When loading from csvfile you will need to supply the channel, project file, and receiver file. 

.. code-block:: python 

    prjfile = 'single_source.prj'
    rcxfile = 'receivers.xyz' 
    channel = 'Ex'
    csvfile = 'single_source-Ex-50.0-10.0-10.0.csv'
    array_csv = Array(channel, prjfile, rcxfile, csvfile = csvfile)
    array_csv.exaggeration = 0.1
    array_csv.sectionplot()


When loading from a pickle file, you need to open the file for reading since it is a binary file. All of the information from the project file, receiver file and channel along with the other seidart objects are also contained in the pickle file. 

.. code-block:: python 

    pickle_file = 'single_source-Ex-50.0-10.0-10.0.pkl'
    f = open(pickle_file, 'rb')
    array_pkl = pickle.load(f)
    array_pkl.exaggeration = 0.1 
    array_pkl.sectionplot()

Both CSV and pickle files are cross-platform and can be loaded using other programming languages. See the examples for the following languages for reading pickle files:

* `MATLAB <https://www.mathworks.com/matlabcentral/answers/1975844-how-to-import-pickle-data-using-python-interface>`_
* `Julia <https://stackoverflow.com/questions/65720584/how-to-load-python-pickle-from-julia>`_ 
* `R <https://www.ankuroh.com/programming/data-analysis/reading-pickle-file-in-r/>`_ 
    
