import os
import scipy.io as sio 
import numpy as np

# Get matlab files from the folder storing all raw data.
files = os.listdir('./data/raw/')   # List all files in the raw data dump.
mat_files = [file for file in files if file[-4:] == '.mat'] # Get all files with the extension .mat

subject = {}    # Initialize an empty dictionary.
for filename in mat_files:  # Iterate across matlab files from our data/raw directory
    directory = 'data/raw/'
    recording = sio.loadmat(directory+filename)['newdata']
    id = int(filename[1])   # Get subject ID from filename (could be better, but it works) 
    subject[id] = np.array(recording) # Assign subject id to its recording numpy array with shape (Channels, Time)