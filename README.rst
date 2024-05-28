SeidarT-Recipes
###############


This is an extension of the `SeidarT <https://github.com/UMainedynamics/SeidarT>`_ package to provide examples to models. When contributing, create a folder/submodule for the recipe which includes a Markdown or ReStructured Text file that summarizes the model, what is required, and directions on how to run the submodule. You will also need to include an *__init__.py* file and, if applicable, a YAML file called *<recipe_name>-requirements.yml* that includes any additional packages needing to be installed into the environment. To install packages use the command (after activating the environment)::
    
    conda env update --file requirements.yml
    
or (without activating the environment)::
    
    conda env update --name seidart --file requirements.yml
    
Please provide robust comments in any scripts.

The *main* branch of the repository is the only permanent branch. When creating new recipes that haven't been finalized, create branch(es) under the name of the recipe and delete it after merging to *main*. In the *docs/source* folder is a file called *index.rst*. Add the path to your .md or .rst file. 