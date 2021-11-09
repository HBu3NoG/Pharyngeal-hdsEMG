import os
import json 
import numpy as np
import pandas as pd 
import scipy as sp 
import scipy.io as sio
from scipy import signal

# Get matlab files from the folder storing all raw data and import them as dictionaries.
def mats2dict(dir = './data/raw/', sub_load = [7], fVerbose = True):
    files = os.listdir(dir)   # List all files in the raw data dump.
    mat_files = [file for file in files if file[-4:] == '.mat'] # Get all files with the extension .mat
    with open(dir+'subject_information.txt', 'r') as json_file:
        subjects_fs = json.load(json_file)

    subjects = {}    # Initialize an empty dictionary.
    for filename in mat_files:  # Iterate across matlab files from our data/raw directory
        # Extract experiment information from filename.
        a,b,c = (filename.find(word) for word in ['S','E','.mat'])
        sub_id, expt_id = int(filename[a+1:b]), filename[b+1:c] 
        if sub_id not in sub_load:
            continue
        else:
            # Initialize a subject if it does not yet exist. (Still need to add an else case, but it's not a problem yet.)
            if not sub_id in subjects.keys():
                subjects[sub_id] = {}

            # Add experiment information to the subject dictionary.
            subjects[sub_id]['Information'] = expt_id

            # Assign recording data to the dictionary structure. Stored as a numpy array because it is handled better.
            mat_file = sio.loadmat(dir+filename)
            variable_name = list(mat_file.keys())[-1]           # Some recordings have different variable names, but typically our data of interest is the last one.
            recording = np.array(mat_file[variable_name])
            mask = recording.any(axis=0)                        # Apply mask to recording to remove shorted segments that occur at the end of the signal.
            recording = recording[:,mask]  
            subjects[sub_id]['Recording'] = np.array(recording)
            
            # For now we will assume they are all 2000. But some are 4000.
            fs = subjects_fs[str(sub_id)]  # Quick fix: For some reason json stores objects as strings, not bytes.
            subjects[sub_id]['Sampling Frequency'] = fs

            # Add subject's clicker information to the subject dictionary. In samples.
            clicker_time = pd.read_csv(dir+filename[:-4]+'_events.csv')['Seconds']
            sub_fixed = [3,5]
            if sub_id in sub_fixed:     # Fix subjects 3 and 5 whose experiments (rest/swallow) were appended.
                mask = np.logical_xor(mask[1:],mask[:-1])
                start_swallow_experiment = np.where(mask == 1)[0][1]
                clicker_time = clicker_time + start_swallow_experiment/fs
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

from scipy.ndimage import gaussian_filter1d
def return_features(subject,nChannels=None, sigma=400):
    # Get subject information
    fs = subject['Sampling Frequency']
    x = np.array(subject['Recording']);
    clicker_information = np.array(subject['Timestamps'])  

    # Pre-processing 
    if nChannels:
        idx = np.random.randint(len(x),size=(nChannels,))
        x = x[idx]
    x = Butterworth(x, ['bandpass', [70,250], 4, fs])
    cutoffs_swallows = clicker_information.reshape(-1,2)
    cutoffs_tasks = [(cutoffs_swallows[i][0]-3,cutoffs_swallows[i+5][1]+3) for i in np.arange(5)*6]

    # Compute activity time course with weighted norm
    z = np.linalg.norm(x.T-np.mean(x,axis=1), axis = 1)**2

    # Compute envelope with a gaussian filter.
    z_env = gaussian_filter1d(z, sigma=sigma)

    # Find peaks conditioned on the first 2 minutes of rest.
    z_noise = z[:120*fs]
    height_extrema = z_noise.mean() + 2*z_noise.std()
    peaks, _ = sp.signal.find_peaks(z_env, height=height_extrema)

    # Extract width of peaks.
    T_swallow = [cutoff[1]-cutoff[0] for cutoff in cutoffs_swallows]
    wlen = np.mean(T_swallow)*fs
    widths, width_heights, left_ips, right_ips = sp.signal.peak_widths(z_env, peaks, rel_height= 0.5, wlen=wlen)

    # Return features indexed by swallow. 
    y_tasks = np.linspace(1,6, num=30,dtype=int,endpoint=False)
    x_features = np.zeros((6,y_tasks.shape[-1]))
    for i,cutoff in enumerate(cutoffs_swallows):
        a,b = [int(x*fs) for x in cutoff]
        mask = np.logical_and(peaks>a,peaks<b)
        n = widths[mask].argmax()
        x_features[0,i] = len(peaks[mask])
        x_features[1,i] = widths[mask].max()    # Only return the biggest
        x_features[2,i], x_features[3,i] = np.vstack((left_ips,right_ips))[:,n]-peaks[n]
        x_features[4,i] = width_heights[mask].max()
        x_features[5,i] = (widths[mask]*width_heights[mask]).sum()

    return y_tasks, x_features
