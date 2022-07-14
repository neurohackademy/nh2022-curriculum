# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.io as spio

# Functions
def load_graydata(datadir,subject,session):
    """ Script to load Matlab structure data into separate numpy arrays and then combine into a dict

    Inputs:
    datadir: directory where data files are found
    subject: subject name as a string (e.g., 'MSC01')
    session: session number as a integer (e.g., 1, 1-10 on this dataset)

    Returns:
    QC: a dictionary with GM, WM, CSF, and MVM timecourses
    """

    top_str = '%s_sess%02d' % (subject,session)
    
    #Gray matter timecourses
    GMinput = spio.loadmat(datadir + top_str + '_GMtcs.mat')
    GMarray = np.array(GMinput['GMtcs'])

    #White matter timecourses
    WMinput = spio.loadmat(datadir + top_str + '_WMtcs.mat')
    WMarray = np.array(WMinput['WMtcs'])

    #CSF timecourses
    CSFinput = spio.loadmat(datadir + top_str + '_CSFtcs.mat')
    CSFarray = np.array(CSFinput['CSFtcs'])

    #Movement parameters
    MVMinput = spio.loadmat(datadir + top_str + '_MVM.mat')
    MVMarray = np.array(MVMinput['MVM'])

    # load all into a dictionary for easier handling
    QC = {'GMtcs' : GMarray, 'WMtcs' : WMarray, 'CSFtcs' : CSFarray, 'MVM' : MVMarray}

    return QC


    
def compute_FD(mvm_orig,convert_rot = False,radius = 50):
    """ This function will compute FD given a set of movement parameters

    Inputs:
    mvm_orig: movement parameters (time x 6 array; first 3 are translation, last 3 are rotation)
    convert_rot: True or False, convert rotational mvm params from deg to mm (default = False)
    radius: mm head radius to assume in converting rotation params (default = 50)

    Returns:
    mvm: converted movement parameters (if convert_rot = False will be same as input)
    ddt_mvm: diffed movement parameters (frame by frame change)
    FD: framewise displacememt, sum of absolute value of ddt_mvm 
    """

    # if needed, convert rotation parameters to mm
    if convert_rot:
        mvm = np.zeros(mvm_orig.shape) #initialize an empty array
        mvm[:,:3] = mvm_orig[:,:3] #translation parameters stay the same
        mvm[:,3:] = mvm_orig[:,3:]*(2*radius*np.pi/360) #rotation params are converted
    else:
        mvm = mvm_orig

    # take original movement parameters, demean and detrend
    ddt_mvm = np.diff(mvm,axis=0)
    ddt_mvm = np.vstack((np.zeros((1,6)),ddt_mvm)) #0 pad to make same shape; by def first val is 0

    #compute FD
    FD=np.sum(np.abs(ddt_mvm),axis=1)

    # return output values
    return mvm, ddt_mvm, FD


def compute_DVARS(GMtcs):
    """ This function will compute DVARS given a set of GM timecourses

    Inputs:
    GMtcs: a set of gray matter timecourses (voxel x time)

    Output:
    DVARS: a timeseries of dvars values
    """

    GMdiff = np.diff(GMtcs,axis=1)
    DVARS = np.sqrt(np.mean(GMdiff**2,axis=0)) #rms of GMdiff
    DVARS = np.hstack((0,DVARS)) #0 pad start

    return DVARS

def compute_GS(GMtcs):
    """ This function will compute the global signal given a set of GM timecourses
    
    Inputs:
    GMtcs: a set of gray matter timecourses (voxel x time)

    Output:
    GS: a global signal timecourse
    """

    GS = np.mean(GMtcs,axis=0) #average over voxels

    return GS
    


def grayplot_NH(QC,stage):
    """This function will make grayplots to look at timeseries after different
    stages of procesing. Additionally allows for bad timepoints to be masked
    out from visualization. This will allow one to inspect data for artifacts.

    Inputs:
    QCfile: QC variable (usually saved under 'QC.mat')
      this file contains gray ts, white ts, CSF ts, original mvm params
      at several different stages of processing
    stage: stage of processing, 1-7
      1 = original pre-processed data
      2 = demeaned, detrended
      3 = residuals from nuisance regression (GM, WM, CSF, motion, + derivatives)
      4 = interpolation of high-motion censored frames
      5 = bandpass temporal filter (0.009 - 0.08 Hz)
      6 = demean and detrend again

    Output:
      A figure

    Notes:
    - Current colorscale limits assume mode 1000 normalization for timeseries, and 
    show 2% signal change

    Made by CGratton, 5/21/14
    Edited for MIND, 8/1/17
    Modified for Neurohackademy, 7/9/19
    Based on FCPROCESS code, v4 (JD Power)
    """


    #set some constants
    numpts=QC['GMtcs'].shape[1] #number of timepoints
    rightsignallim = np.arange(-20,21,20) #GS, main plot signal limits - 2% assuming mode 1000 normalization
    leftsignallim = np.arange(0,21,10) #DVars limits
    rylimz=[np.min(rightsignallim),np.max(rightsignallim)]
    lylimz=[np.min(leftsignallim),np.max(leftsignallim)]
    FDmult = 10 #multiplier to FD to get in range of DVars values
    FDthresh = 0.2 #FD threshold to mark frame for scrubbing (use 0.1 for filtered FD)

    #compute data quality metrics -- CG: compute by hand to better understand (separated here for practice)
    [mvm,ddt_mvm,FD] = compute_FD(QC['MVM'])
    DVars = compute_DVARS(QC['GMtcs'][:,:,stage]) # compute DVARs for a particular processing stage
    GS = compute_GS(QC['GMtcs'][:,:,stage]) # compute global signal for a particular processing stage

    #create plot
    fig = plt.figure(figsize=(10,10),constrained_layout = True)
    gs = GridSpec(9,1,figure=fig)

    #plot individual mvm params
    ax1 = fig.add_subplot(gs[0:2])
    pointindex = np.arange(1,numpts+1)
    plt.plot(pointindex,mvm)

    plt.xlim([0, numpts])
    plt.ylim([-1.5, 1.5])
    plt.ylabel('mvm-XYZPYR')

    #Next, plot FD, DVARS and GS on the same plot
    ax2a = fig.add_subplot(gs[2:4])
    ax2b = ax2a.twinx()
    ax2a.plot(pointindex,DVars,color=[0,0,1],alpha=0.5)
    ax2b.plot(pointindex,GS,color=[0,1,0],alpha=0.5)
    ax2a.plot(pointindex,FD*FDmult,color=[1,0,0],alpha=0.5)
    ax2a.hlines(FDthresh*FDmult,pointindex[0],pointindex[-1],'k',alpha=0.5)
    
    plt.xlim([0, numpts])
    ax2a.set_ylim(lylimz)
    ax2a.set_yticks(leftsignallim)
    ax2b.set_ylim(rylimz)
    ax2b.set_yticks(rightsignallim)
    ax2a.set_ylabel('R:FD*' + str(FDmult) +' B:DV G:GS')

    #next plot gray matter signal
    ax3 = fig.add_subplot(gs[4:8])
    new_GMtcs = QC['GMtcs'][:,:,stage]
    plt.imshow(new_GMtcs,cmap='gray',vmin=-20,vmax=20,aspect='auto') #default: showing 2% signal on mode 1000 norm
    plt.ylabel('GRAY')

    #finally, plot WM and CSF ts
    ax4 = fig.add_subplot(gs[8:])
    new_WMCSF = np.vstack((QC['WMtcs'][:,:,stage],QC['CSFtcs'][:,:,stage]))
    plt.imshow(new_WMCSF,cmap='gray',vmin=-20,vmax=20,aspect='auto')
    plt.ylabel('WM CSF')
    plt.xlabel('frames')

    return fig

