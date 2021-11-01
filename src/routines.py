import numpy as np
import pickle
import pandas as pd 
import scipy as sp 
import scipy.io as sio
from scipy import signal
import os

# Get matlab files from the folder storing all raw data and import them as dictionaries.
def mats2dict(dir = './data/raw/', sub_load = [7], fVerbose = True):
    files = os.listdir(dir)   # List all files in the raw data dump.
    mat_files = [file for file in files if file[-4:] == '.mat'] # Get all files with the extension .mat

    subjects = {}    # Initialize an empty dictionary.
    for filename in mat_files:  # Iterate across matlab files from our data/raw directory
        # Extract experiment information from filename.
        a,b,c = (filename.find(word) for word in ['S','E','.mat'])
        sub_id, expt_id = int(filename[a+1:b]), filename[b+1:c] 
        if sub_id not in sub_load:
            continue
        else:
            print(sub_id)

            # Initialize a subject if it does not yet exist. (Still need to add an else case, but it's not a problem yet.)
            if not sub_id in subjects.keys():
                subjects[sub_id] = {}

            # Add experiment information to the subject dictionary.
            subjects[sub_id]['Information'] = expt_id

            # Assign recording data to the dictionary structure. Stored as a numpy array because it is handled better.
            mat_file = sio.loadmat(dir+filename)
            variable_name = list(mat_file.keys())[-1]           # Some recordings have different variable names, but typically our data of interest is the last one.
            recording = mat_file[variable_name]
            subjects[sub_id]['Recording'] = np.array(recording)
            
            # For now we will assume they are all 2000. BUt some are 4000.
            subjects[sub_id]['Sampling Frequency'] = 2000

            # Add subject's clicker information to the subject dictionary. In samples.
            clicker_time = pd.read_csv(dir+filename[:-4]+'_events.csv')['Seconds']
            subjects[sub_id]['Timestamps'] = np.array(clicker_time)
        
    # Print information about the subject structure.
    if fVerbose:
        print('Subjects imported: ', *subjects.keys())
        print('Subject information: ', *subjects[next(iter(subjects))].keys(), sep='\t')
        
    if len(subjects.keys()) == 1:
        return subjects[next(iter(subjects))]
    else:
        return subjects

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
            Index = np.where(np.abs(z_dist) > Z_outlier)
            if Index[0].size > 0:
                Cov_Samples = np.delete(Cov_Samples,Index,axis=0)
                Precision_Samples = np.delete(Precision_Samples,Index,axis=0)
            else: break # if none, exit while loop
        #plt.plot(z_dist)
        print(np.mean(Distance),np.std(Distance))
    return Cov_Mean

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


