#Imports
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

#Functions
def figure_simmat(simmat,nsubs,nsess,clims=(0,1)):
    """ This function will make a figure of the similarity among correlation matrices for a given 
    set of subjects and sessions. Assumes this comes from a set of subjects with an equal session num
    
    Inputs:
    simmat: a square matrix of similarity values
    nsubs: number of subjects (the primary dimension)
    nsess: number of sessions per subject (the secondary dimension)
    clims: (optional) limits to place on colormap

    Returns:
    fig: a figure handle for the figure that was madee
    """

    # main figure plotting
    fig, ax = plt.subplots()
    im = ax.imshow(simmat,cmap='plasma',vmin=clims[0],vmax=clims[1])
    plt.colorbar(im)

    # add some lines between the subjects
    transitions = np.arange(nsess,nsess*(nsubs-1)+1,nsess) - 0.5
    for tr in transitions:
        ax.axhline(tr,0,simmat.shape[1],color='k')
        ax.axvline(tr,0,simmat.shape[1],color='k')

    # alter how the tick marks are shown to plot network names
    trans_plusends = np.hstack((0,transitions,simmat.shape[1])) #add ends
    centers = trans_plusends[:-1] + ((trans_plusends[1:] - trans_plusends[:-1])/2)
    ax.set_xticks(centers)
    ax.set_yticks(centers)

    subnames = []
    for sub in range(nsubs):
        subnames.append('MSC%02d'%(sub+1))
    ax.set_xticklabels(subnames)
    ax.set_yticklabels(subnames)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

    plt.show()
    
    return fig

def figure_corrmat(corrmat,Parcel_params,clims=(-1,1)):
    """ This function will make a nice looking plot of a correlation matrix for a given parcellation, 
    labeling and demarkating networks.

    Inputs:
    corrmat: an roi X roi matrix for plotting
    Parcel_params: a dictionary with ROI information
    clims: (optional) limits to place on corrmat colormap

    Returns:
    fig: a figure handle for figure that was made
    """

    # some variables for ease
    roi_sort = np.squeeze(Parcel_params['roi_sort'])

    # main figure plotting
    fig, ax = plt.subplots()
    im = ax.imshow(corrmat[roi_sort,:][:,roi_sort],cmap='seismic',vmin=clims[0],vmax=clims[1],interpolation='none')
    plt.colorbar(im)

    # add some lines between networks
    for tr in Parcel_params['transitions']:
        ax.axhline(tr,0,Parcel_params['num_rois'],color='k')
        ax.axvline(tr,0,Parcel_params['num_rois'],color='k')

    # alter how the tick marks are shown to plot network names
    ax.set_xticks(Parcel_params['centers'])
    ax.set_yticks(Parcel_params['centers'])
    ax.set_xticklabels(Parcel_params['networks'],fontsize=8)
    ax.set_yticklabels(Parcel_params['networks'],fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right', rotation_mode = 'anchor')

    plt.show()

    return fig

def loadParcelParams(roiset,datadir):
    """ This function loads information about the ROIs and networks.
    For now, this is only set up to work with 333 Gordon 2014 Cerebral Cortex regions

    Inputs:
    roiset = string naming roi type to get parameters for (e.g. 'Gordon333')
    datadir = string path to the location where ROI files are stored

    Returns:
    Parcel_params: a dictionary with ROI information stored in it
    """

    #initialize a dictionary where info will be stored
    Parcel_params = {}

    # put some info into the dict that will work for all roi sets
    Parcel_params['roiset'] = roiset
    dataIn_types = {'dmat','mods_array','roi_sort','net_colors'}
    for dI in dataIn_types:
          dataIn = spio.loadmat(datadir + roiset + '_' + dI + '.mat')
          Parcel_params[dI] = np.array(dataIn[dI])
    Parcel_params['roi_sort'] = Parcel_params['roi_sort'] - 1 #orig indexing in matlab, need to subtract 1
    
    #transition points and centers for plotting
    transitions,centers = compute_trans_centers(Parcel_params['mods_array'],Parcel_params['roi_sort'])
    Parcel_params['transitions'] = transitions
    Parcel_params['centers'] = centers

    # some ROI specific info that needs to be added by hand
    # add to this if you have a new ROI set that you're using
    if roiset == 'Gordon333':
        Parcel_params['dist_thresh'] = 20 #exclusion distance to not consider in metrics
        Parcel_params['num_rois'] = 333
        Parcel_params['networks'] = ['unassign','default','visual','fp','dan','van','salience',
                                         'co','sm','sm-lat','auditory','pmn','pon']
    else:
        raise ValueError("roiset input is recognized.")

    return Parcel_params


def compute_trans_centers(mods_array,roi_sort):
    """ Function that computes transitions and centers of networks for plotting names

    Inputs:
    mods_array: a numpy vector with the network assignment for each ROI (indexed as a number)
    roi_sort: ROI sorting ordered to show each network in sequence

    Returns:
    transitions: a vector with transition points between networks
    centers: a vector with center points for each network
    """

    mods_sorted = np.squeeze(mods_array[roi_sort])
    transitions = np.nonzero((np.diff(mods_sorted,axis=0)))[0]+1 #transition happens 1 after

    trans_plusends = np.hstack((0,transitions,mods_array.size)) #add ends
    centers = trans_plusends[:-1] + ((trans_plusends[1:] - trans_plusends[:-1])/2)

    return transitions,centers
