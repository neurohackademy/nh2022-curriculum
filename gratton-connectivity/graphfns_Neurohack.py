#Imports
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

#Functions
def threshold_matrix_density(corrmat,thr,return_thr = False):
    """ Function that thresholds a given correlation matrix by density so a set proportion
    of edges passes threshold.

    Inputs:
    corrmat: a square symmetric matrix
    thr: density threshold to use
    return_thr = return r and density thresholds calculated from data (default = False)

    Returns:
    corrmat_thr = a thresholded correlation matrix
    [optionally also returns r and k density thresholds calculated from the data

    Based on a script originally by JD Power
    CG edited and converted to python
    """

    cshp = corrmat.shape
    if cshp[0] != cshp[1]:
        raise ValueError("input matrix is not square")
    nodnum = cshp[0]
    
    numpossibleedges = int(nodnum * (nodnum-1)/2)
    edgesleft = int(np.ceil(thr * numpossibleedges))
    

    # take upper triangle of the matrix
    corrmat_tinds = np.triu_indices(nodnum,1)
    corrmat_lin = corrmat[corrmat_tinds]

    # sort values in the upper triangle and select lowest thr%
    sorti = np.argsort(-corrmat_lin) #descending order
    keptedges = np.zeros(corrmat_lin.shape)
    keptedges[sorti[:edgesleft]] = corrmat_lin[sorti[:edgesleft]]
    
    #reshape into a matrix
    corrmat_thr = np.zeros(corrmat.shape)
    corrmat_thr[corrmat_tinds] = keptedges
    corrmat_thr_sym = corrmat_thr + np.transpose(corrmat_thr) # make it symmetric again

    if return_thr:
        #some quick checks on where threshold was set
        kthr = edgesleft/numpossibleedges
        rthr = corrmat_lin[sorti[edgesleft]]
        return corrmat_thr, corrmat_thr_sym, kthr, rthr
    else:       
        return corrmat_thr, corrmat_thr_sym

def make_gephi_node_inputfile(Parcel_params,nod_colors=[]):
    """ Function to make input csv files needed for gephi for nodes

    Inputs:
    Parcel_params: a dictionary with information about the nodes in a graph
    nod_color: optional input, node x 3 array of RGB values
      if not included, will instead use network colors for nodes

    Returns:
    node_data: a Pandas data frame with information for a gephi csv node file
    """

    # some constants we use a lot
    nrois = Parcel_params['num_rois']
    net_colors = Parcel_params['net_colors']
    net_nums = np.unique(Parcel_params['mods_array']) #to deal with numbers being non-sequential

    # some basic columns needed for gephi (for now, nothing important to add here)
    ID = np.arange(nrois) #np.squeeze(Parcel_params['roi_sort']) + 1 #index from 1
    Label = np.zeros(nrois)
    Interval = np.zeros(nrois)
    Network = np.squeeze(Parcel_params['mods_array'])
    Hub1 = np.zeros(nrois)
    Hub2 = np.zeros(nrois)
    Hub3 = np.zeros(nrois)

    # create a color array (one entry for each node)
    roi_rgb = [] #colors in a list of rgb strings as inputs for gephi
    for roi in range(nrois):
        if np.size(nod_colors) > 0:
            nod_color = np.squeeze(nod_colors[roi,:])
        else:
            rind = np.where(net_nums == Parcel_params['mods_array'][roi])
            nod_color = net_colors[rind[0][0]]
        rgb_str = '%.03f,%.03f,%.03f' %(nod_color[0]*255,nod_color[1]*255,nod_color[2]*255)
        roi_rgb.append(rgb_str)

    # make this into a Pandas DataFrame
    items = np.transpose([ID,Label,Interval,Network,roi_rgb,Hub1,Hub2,Hub3])
    node_data = pd.DataFrame(items,columns = ['ID','Label','Interval','Network','Color','Hub1','Hub2','Hub3'])

    return node_data

def make_gephi_edge_inputfile(corrmat):
    """ Function to make input csv files needed for gephi edges

    Inputs:
    corrmat: an NxN correlation matrix (can be weighted or unweighted)
      should be zeroed under upper tri so edges aren't double counted
      but assumes edges are undirected
      should be thresholded or output file will be very large

    Output:
    edge_data: a Panda data frame with information for a gephi edge file
    """

    # find edges and store these into lists
    edge_inds = np.where(corrmat>0)
    nedges = np.size(edge_inds[0])
    Source = edge_inds[0]
    Target = edge_inds[1]

    # fill in some other information about the edges
    Type = ['Undirected'] * nedges
    Weight = corrmat[edge_inds] # or enter 1 for unweighted?
    ID = np.arange(nedges)
    Label = np.zeros(nedges)
    Interval = np.zeros(nedges)
    Hub1 = np.ones(nedges)
    Hub2 = np.ones(nedges)
    Hub3 = np.ones(nedges)

    # make a pandas data frame with all of the information to save into csv
    items = np.transpose([Source, Target, Type, ID, Label, Interval, Weight, Hub1, Hub2, Hub3])
    edge_data = pd.DataFrame(items,columns = ['Source','Target','Type','ID','Label','Interval','Weight',
                                                  'Hub1','Hub2','Hub3'])
    return edge_data
    
def hub_metrics(adj_mat,mod_array):
    """ Function for calculating hub metrics 

    Inputs:
    adj_mat: a NxN unweighted symmetric matrix
    mod_array: a Nx1 array of community labels for each node

    Returns:
    deg: an Nx1 array of degree values for each node
    WD: an Nx1 array of within module degree for each node
    PC: an Nx1 array of participation coefficient for each node
    see Guimera & Amaral (2005) Nature for definitions
    """
   
    # some constants/initializations
    communities = np.unique(mod_array) # get community IDs
    nnod = mod_array.shape[0]
    nodes = np.arange(nnod)
    WD = np.zeros(nnod) # any nodes not in a community get WD=0
    PC = np.zeros(nnod)
    
    # prep input matrix:
    # make sure diagonal is zeroed (should be true already)
    # and make matrix binarized (below are unweighted calcs)
    adj_mat[np.diag_indices(nnod)] = 0
    adj_mat = adj_mat > 0

    # quick degree calculation
    deg = np.nansum(adj_mat,1)

    # WD calculation
    for comm in communities:

        comm_nodes = np.where(mod_array == comm)[0]
        comm_mat = adj_mat[:,comm_nodes][comm_nodes,:]

        ki = np.nansum(comm_mat,1) # density of connections within module
        mean_ks = np.mean(ki) #average across all nodes in a module
        sigma_ks = np.std(ki) #std across all nodes in a module
        zi = (ki - mean_ks)/sigma_ks #z-score per module
        WD[comm_nodes] = zi

        # if they're all the same, make them all WD = 0        
        if sigma_ks == 0:
            WD[comm_nodes] = 0

    # PC calculation
    for nod in nodes:
        kis = np.zeros(np.size(communities)) # num links of node i to mod s
        for c,comm in enumerate(communities):
            # get all nodes in community that are not n
            comm_nodes = list(set(np.where(mod_array == comm)[0]) - set([nod]))
            node_to_comm = adj_mat[:,nod][comm_nodes]
            kis[c] = np.sum(node_to_comm)

        PC[nod] = 1 - np.sum((kis/deg[nod])**2)
        # deg in mod / total deg : close to 1 if all in one mod
        # therefore PC is low if most connections are to one mod
        
        # for disconnected nodes, set PC = 0
        if deg[nod] == 0:
            PC[nod] = 0
    
    return PC, WD, deg

def figure_hubs(Parcel_params,thresholds,degree,wd,pc):
    """ Function for making a plot of hub measures across thresholds.

    Inputs:
    Parcel_params: a dictionary with information about the ROI choice
    thresholds: a 1D array with thresholds that the hub measures were taken at
    degree: a 1D array with degree hub measure per node (# of connections to each node)
    wd: a 1D array with within module hub measure per node
      z-scored measure of the number of connections of a node to its module
    pc: a 1D array with participation coefficient measure per node
      dispersion measue of proportion of connections node has across mods

    Returns:
    fig: a figure handle
    """

    # for ease:
    sorti = np.squeeze(np.transpose(Parcel_params['roi_sort']))
    transitions = Parcel_params['transitions']
    centers = Parcel_params['centers']
    networks = Parcel_params['networks']
    hub_measures = {'degree':degree,'within-mod':wd,'part-coeff':pc}
    thresh_str_fmt = ['%.2f' % (t) for t in thresholds]

    fig, axs = plt.subplots(1,3,figsize=(10,8), constrained_layout = True)
    
    for hm,hubm in enumerate(hub_measures.keys()):
        im = axs[hm].imshow(hub_measures[hubm][sorti,:],cmap='magma',aspect='auto')
        
        axs[hm].set_title(hubm)
        axs[hm].set(xlabel='thresholds', ylabel = 'networks')
        for tr in transitions:
            axs[hm].axhline(tr,0,thresholds.size,color='w')
        axs[hm].set_yticks(centers)
        axs[hm].set_yticklabels(networks)
        axs[hm].set_xticks(np.arange(0,thresholds.size,4)) #only plot every fifth
        axs[hm].set_xticklabels(thresh_str_fmt[:thresholds.size:4],fontsize='small')
        fig.colorbar(im,ax=axs[hm],location='bottom')
            
    for ax in axs.flat:
        ax.label_outer()

    plt.show()

    return fig


def hub_colormap(hubvals):
    """ Function to transform hub values into an RGB color map for plotting in a graph

    Inputs:
    hubvals: a 1D array with hub values per node. Will by default try to fit colormap to range.

    Returns:
    hub_colors: a nodex3 array with RGB color values per node 
    """

    hub_colors = np.zeros([hubvals.shape[0],3])
    
    map_vals = cm.get_cmap('jet',101)
    map_range = [np.min(hubvals), np.max(hubvals)]

    for r in range(hub_colors.shape[0]):
        cind = (hubvals[r]-map_range[0])/(map_range[1] - map_range[0])
        hub_colors[r,:] = map_vals(cind)[:3]

    return hub_colors
