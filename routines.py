import numpy as np
import pickle
import pandas as pd 
import scipy as sp 
import scipy.io as sio
from scipy import signal
import os

# Get matlab files from the folder storing all raw data.
def mats2dict(dir = './data/raw'):
    files = os.listdir(dir)   # List all files in the raw data dump.
    mat_files = [file for file in files if file[-4:] == '.mat'] # Get all files with the extension .mat

    subject = {}    # Initialize an empty dictionary.
    for filename in mat_files:  # Iterate across matlab files from our data/raw directory
        directory = 'data/raw/'
        recording = sio.loadmat(directory+filename)['newdata']
        id = int(filename[1])   # Get subject ID from filename (could be better, but it works) 
        subject[id] = np.array(recording) # Assign subject id to its recording numpy array with shape (Channels, Time)

    return subject


class Subject:
    # Initialize a subject class with 2 files: Raw recording from TMSi (.mat), 
    #   time stamps that dictate the swallow period (.csv). 
    def __init__(self,filename_Swallow,filename_TimeStamps,labels=[],fs=2000):
        #### Import EMG rest data.
        # Load .mat file
        file = scipy.io.loadmat(filename_Swallow)
        # Assign TMSi noise recording to variable "NoiseData"
        self.data = file['newdata'][:,:-150]            #Added [:,:-150] to get rid of the short at the end of the recording         

        #### Segment Swallows from EMG data
        # Load .csv file into a df
        df = pd.read_csv(filename_TimeStamps)
        # Import timestamp vector where row 1: start of swallow, row 2 end of swallow.
        timestamps = np.array([float(x[2:4])*60+float(x[5:]) for x in df['Time']])
        # Clean up timestamps by removing index x_n when x_n-x_(n-1)<500ms.
        timestamps_wrong = np.where((timestamps[1:]-timestamps[:timestamps.shape[0]-1])<0.5)[0]+1
        timestamps = np.delete(timestamps,timestamps_wrong)
        # Create sample indeces for easy access in the dataset
        swallowIDx = [np.arange(x[0]*fs,x[1]*fs,dtype=int) for x in timestamps.reshape(-1,2)]
        # Every subject has swallows and labels for future classification
        self.swallows = [self.data[:,IDx] for IDx in swallowIDx]
        self.labels = labels

# Function to plot TMSi data. Data should be a (Ch,Samples) matrix.
## 
def PlotTMSi(axs, Data, fs=2000, vline = []):
    N,M = Data.shape
    Time = np.arange(M)/fs

    # Normalize for plotting. 
    Max,Min = np.max(Data,axis=1),np.min(Data,axis=1)
    Data = (Data-Min.reshape(N,1))/(Max-Min).reshape(N,1)

    # Plotting.
    axs.plot(Time,Data.T+np.arange(N))
    axs.set_ylabel('Channels')
    axs.set_yticks(np.arange(N/4+1)*4)
    axs.set_xlabel('Time (sec)')

    # Sections. 
    if vline:
        for xc in vline:
            axs.axvline(x=xc, linestyle='--',color='r')

# Options:  btype{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
#           fc (Cutoff frequency), N (filter order), fs (sampling frequency)
def Butterworth(Signals,options,axis=-1):
    btype,fc,N,fs = options                                     
    b,a = sp.signal.butter(N,fc,btype=btype,fs=fs)   # Digital coefficients of highpass filter.
    Signals_filt = sp.signal.filtfilt(b,a,Signals,axis=axis)
    return Signals_filt

# Compute the frobenius norm of two matrices.
def Frobenius(A,B):
    return np.sqrt(np.trace(A.T.dot(B)))

# Compute the geodesic norm for two positive semidefinite matrices.
# More on this : https://www.stat.uchicago.edu/~lekheng/work/ellipsoids.pdf
def Geodesic(A_inv,B):
    C = A_inv.dot(B)
    eigvals = np.linalg.eigvals(C)
    log_sqr_eigvals = np.log(eigvals)**2
    return np.sqrt(np.sum(log_sqr_eigvals))

# This function computes the covariance of signals (Ch, Samples) in different ways: 'MLE', 'LASSO', 'Shrunkage'.
# It is also packed with a more robust way of computing the covariance structure by computing many covariances and
# using matrix distances to remove outliers.
from sklearn.covariance import EmpiricalCovariance, ShrunkCovariance, GraphicalLasso
def Covariance(Signals,Type='MLE',Robust=[]):  # Robust = [T:Period, Z_outlier:Number of z distances to be considered an outlier, kind: matrix norm (Frobenius, Geodesic)}
    if Type == 'MLE':
        Cov = EmpiricalCovariance(store_precision=True, assume_centered=False)
    elif Type == 'Shrunkage':
        Cov = ShrunkCovariance(store_precision=True, assume_centered=False, shrinkage=0.01)
    elif Type == 'LASSO':
        Cov = GraphicalLasso(alpha=0.01, assume_centered=False)

    
    # The goal is to remove outliers in a signal by splitting the large dataset into smaller subsets.
    # Data is then discarded depending on its distance away from the average covariance matrix.
    # To enter into this function, Robust must be passed as an array with options: {T,Z_outlier,kind}
    if Robust:
        # Unpack args into sampling period and outlier metric.
        T,Z_outlier,kind = Robust
        # Size of time window (samples)
        M = int(T)
        # Number of windows
        N = int(Signals.shape[1]/M)
        # Compute list of N covariance samples along with their inverses (precision matrix)
        Cov_Samples = []
        Precision_Samples = []
        for i in range(N):
            Cov.fit(Signals[:,M*i:M*(i+1)].T)
            Cov_Samples.append(Cov.covariance_)
            Precision_Samples.append(Cov.precision_)
        # Enter 'while' loop to discard outlier covariance matrices.
        i = 0
        while i < 100:  # Keep it from looping.
            i += 1; #print(i,len(Cov_Samples))
            # Take the mean of all covariance matrices
            Cov_Mean = np.mean(Cov_Samples,axis=0)
            # Type of metric for matrix distance {Frobenius and Geodesic}
            if kind == 'Frobenius':
                Distance = np.array([Frobenius(Cov, Cov_Mean) for Cov in Cov_Samples])
            elif kind == 'Geodesic':
                Distance = np.array([Geodesic(Precision, Cov_Mean) for Precision in Precision_Samples])
            # Normalize distances to z-scores.
            z_dist = (Distance-np.mean(Distance))/np.std(Distance)
            #if i == 1: plt.plot(z_dist)
            # Find outliers
            Index = np.where(np.abs(z_dist) > Zout)
            if Index[0].size > 0:
                Cov_Samples = np.delete(Cov_Samples,Index,axis=0)
                Precision_Samples = np.delete(Precision_Samples,Index,axis=0)
            else: break # if none, exit while loop
        #plt.plot(z_dist)
        print(np.mean(Distance),np.std(Distance))
    return Cov_Mean

