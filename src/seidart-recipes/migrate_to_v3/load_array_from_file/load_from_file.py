import pickle
from seidart.routines.arraybuild import Array 

# When loading from csvfile you will need to supply the channel, project file, 
# and receiver file
prjfile = 'single_source.json'
rcxfile = 'receivers.xyz' 
channel = 'Ex'
csvfile = 'single_source-Ex-50.0-10.0-10.0.csv'
array_csv = Array(channel, prjfile, rcxfile, csvfile = csvfile)
array_csv.exaggeration = 0.1
array_csv.sectionplot()

# ==============================================================================
# When loading from a pickle file, you need to open the file for reading since
# it is a binary file. All of the information from the project file, receiver 
# file and channel along with the other seidart objects are also contained 
# in the pickle file. 
pickle_file = 'single_source-Ex-50.0-10.0-10.0.pkl'
f = open(pickle_file, 'rb')
array_pkl = pickle.load(f)
array_pkl.exaggeration = 0.1 
array_pkl.sectionplot()