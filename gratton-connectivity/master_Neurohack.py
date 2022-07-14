"""
This tutorial looks at different ways of visualizing your data
1. First, I present a way developed by JD Power for looking at your data
    at different stages of processing to detect artifacts
2. Next, we turn to FC measures. We look at correlation matrices and
    their variability.
3. Next we look at graph representations of these correlation matrices
4. Finally, we examine hub measures for these graphs.

Dataset is from Midnight Scan Club, 10 highly sampled individual subjects
Freely available on OpenfMRI: https://openfmri.org/dataset/ds000224/
See paper with description:
    Gordon et al. (2017) Precision Functional Mapping of Individual Human Brains. Neuron
    http://www.cell.com/neuron/fulltext/S0896-6273(17)30613-X
"""

# imports
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import importlib
import pandas as pd

import grayplot_Neurohackademy as gplt
import corrmat_fns_Neurohackademy as cfns
import graphfns_Neurohack as gfns
importlib.reload(gplt) #reload this during development
importlib.reload(cfns)
importlib.reload(gfns)

# Initialization of directory information:
thisDir = os.getcwd() + '/'
datadir = thisDir + 'data/'
outdir = thisDir + 'output/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
# What to do:
step1_grayplots = True
step2_corrmats = True
step3_springplots = True
step4_hubs = True


# --------------------------------------------------------------------------------------
# 1. Grayplots

if step1_grayplots:
    # Start with movement parameters and time-series data - opens into GrayplotInfo structure
    # for now only provided for MSC01, since files can be large.
    # See MSC openfMRI dataset for data from additional subjects

    # Go over how to calculate FD, DVars, and GS [CG: have students do this]

    # Make grayplots for single session, different stages of processing
    # Discuss consequences of each stage of processing for signals
    stages = ['orig','demeantrend1','regress','interp','bpfilt','demeantrend2'] #analysis steps
    subject = 'MSC01'
    session = 1
    for st in range(0,np.size(stages)):
        QC = gplt.load_graydata(datadir + 'grayplot_info/',subject,session)
        fig = gplt.grayplot_NH(QC,st)
        title_str ='%s_sess%02d_stage%d_%s' % (subject,session,st,stages[st])
        fig.suptitle(title_str)
        plt.show()
        plt.savefig(outdir + title_str + '.tiff')
    plt.close('all')
    st
    # Make grayplots for different sessions. Discuss variability.
    st = 5 #stage to plot (set last for now)
    for session in range(1,11):
        QC = gplt.load_graydata(datadir + 'grayplot_info/',subject,session)
        fig = gplt.grayplot_NH(QC,st)
        title_str ='%s_sess%02d_stage%d_%s' % (subject,session,st,stages[st])
        fig.suptitle(title_str)
        plt.show()
        plt.savefig(outdir + title_str + '.tiff')
    plt.close('all')

    # Discuss other finer points:
    #   1. How to quantitatively check for motion contamination - QC-FC measures
    #   2. Relatedly, how to set FD threshold
    #   3. Other issues: respiration artifact in movement parameters

    # Relevant references:
    # For more insight into what this sort of approach tells you, read:
    #   Power, J. D. (2017). A simple but useful way to assess fMRI scan qualities. Neuroimage, 154, 1, 150-158
    # And for more information on FC-processing strategies, read:
    #   Power, J.D., et al., (2014). Methods to detect, characterize, and remove motion artifact in resting state fMRI.
    #    Neuroimage, 84, 320-341
    #   Power J.D., et al. (2015). Recent progress and outstanding issues in motion correction in resting-state fMRI.
    #    Neuroimage, 105, 536-551.
    #   Power, J.D., et al. (2017). Sources and implications of whole-brain fMRI signals in humans.
    #    Neuroimage, 146, 609-625
    #   Ciric, R., et al., (2017). Benchmarking of participant-level confound regression strategies for the
    #    control of motion artifact in studies of functional connectivity. Neuroimage, 154, 174-187

# --------------------------------------------------------------------------------------
# Load some information you'll need for the next few modules

if step2_corrmats or step3_springplots or step4_hubs:
    # first load information about ROIs and networks
    Parcel_params = cfns.loadParcelParams('Gordon333',datadir + 'Parcel_info/')

    # some constants
    nsubs = 10
    nsess = 10
    ntime = 818
    nrois = Parcel_params['num_rois']

    # load subject timecourses and tmasks
    roiData = np.empty((nsubs,nsess,ntime,nrois)) #initialize arrays
    tmaskData = np.empty((nsubs,nsess,ntime))
    for sub in range(nsubs): 
        fname = '%sMSC%02d_parcel_timecourse.mat' % (datadir,sub+1)
        fin = spio.loadmat(fname) #loads a mat with parcel_time and tmask_all variables
        for sess in range(nsess): 
            roiData[sub,sess] = fin['parcel_time'][0,sess] #time X parcel matrix
            tmaskData[sub,sess] = np.squeeze(fin['tmask_all'][0,sess]) #time X 1 matrix of good/bad datapoints

    # compute correlations among TS ROIs, masking out bad frames
    corrmat = np.empty((nsubs,nsess,nrois,nrois)) # initialize
    for sub in range(nsubs): 
        for sess in range(nsess):
            # Fisher transform the data (arctanh) for easier math later
            corrmat[sub,sess,:,:] = np.arctanh(np.corrcoef(np.transpose(roiData[sub,sess,tmaskData[sub,sess]>0,:])))

    # average over sessions in a sub
    subavgmat = np.squeeze(np.mean(corrmat,axis=1))

    # make a group average matrix too
    groupmat = np.squeeze(np.mean(subavgmat,axis=0))

# --------------------------------------------------------------------------------------
# 2. Correlation matrices

if step2_corrmats:

    # Start with TS ROIs from 333 and tmask, along with network assignments
    #   1. Discuss what these ts, ROIs and tmasks are
    #   2. Discuss how we got these ROIs - surface mapping procedure in between
    #        [quickly make an image of the ROI timeseries, pre and post mask]
    #        [Discuss +/- of different types of ROIs (functional, anatomical, group, individual)]
    

    # plot a few ROI timeseries from a given subject/session, with and without masking
            
    # plot average matrix, order ROIs randomly
    plt.figure()
    rand_roi = np.random.permutation(nrois)
    plt.imshow(groupmat[rand_roi,:][:,rand_roi],cmap='seismic',vmin=-1.0,vmax=1.0)
    plt.colorbar()
    plt.title('Group corrmat, random ROI order')
    plt.show()

    # plot group matrix in network order
    fig = cfns.figure_corrmat(groupmat,Parcel_params)
    plt.title('Group corrmat, network order')
    plt.savefig(outdir + 'Corrmat_group.pdf')
    
    # A1. Discuss matrix structure
    # A2. Discuss community detection methods (Infomap, modularity optimization)
    # A3. Discuss multi-scale nature of networks
    
    # Now look at relationship between correlation matrices across sessions and
    # subjects
    # B1. Look by eye at group and a single subject across sessions
    for sess in range(nsess):
        fig = cfns.figure_corrmat(np.squeeze(corrmat[0,sess]),Parcel_params)
        plt.title('MSC01, session ' + str(sess+1))
        title_str = '%sCorrmat_MSC01_sess%02d.pdf' % (outdir,sess+1)
        plt.savefig(title_str)
    plt.close('all');

    # B2. Compare corrmats across subjects
    for sub in range(nsubs):
        fig = cfns.figure_corrmat(np.squeeze(subavgmat[sub]),Parcel_params)
        subnum = 'MSC%02d' %(sub + 1)
        plt.title(subnum + ', session avg')
        plt.savefig(outdir + 'Corrmat_' + subnum + '_sessavg')
    plt.close('all');

    # Make a similarity matrix to more formally compare corrmats
    # C1. First make an index of upper triangle of the matrix
    maskmat_inds = np.triu_indices(nrois,1)

    # C2. Create a linearized version of the upper triangle
    corrlin = np.empty((nsubs*nsess,int(nrois*(nrois-1)/2)))
    count = 0
    for sub in range(nsubs):
        for sess in range(nsess):
            tmp = corrmat[sub,sess]
            corrlin[count] = tmp[maskmat_inds]
            count = count + 1

    # C3. Calculate similarity and plot
    simmat = np.corrcoef(corrlin)
    fig = cfns.figure_simmat(simmat,nsubs,nsess)
    plt.savefig(outdir + 'Simmat_rest.pdf')

    # C4. Discuss structure in this matrix
    # C5. Discuss the advantages and disadvantags of doing these measures with individual ROIs
    # C6. If time, create some reliability curves

    # for more information, relevant references:
    # Power et al. (2011). Functional Network Organization of the Human Brain. Neuron, 72, 4, 665-678
    # Yeo, et al. (2011) "The organization of the human cerebral cortex estimated by intrinsic functional
    #   connectivity." Journal of neurophysiology 106.3: 1125-1165.

# --------------------------------------------------------------------------------------
# 3. Spring embedded plots

    
if step3_springplots:

    # for this section, we will be making some spring-embedded plots of the data using gephi
    # https://gephi.org/users/download/
    # make sure all plugins are installed

    # 1. Briefly overview spring embedding method: 
    # 2. Discuss different types of thresholding (r vs. density)
    # 3. Discuss weighted vs. not weighted graphs
    # 4. Discuss issues of network size and density for feasibility

    # start by thresholding the group matrix at a set threshold (2% edge density)
    thr = 0.02
    adj_mat,adj_mat_sym = gfns.threshold_matrix_density(groupmat,thr)

    # create a set of files for plotting spring embedding in gephi
    # first for nodes
    node_data = gfns.make_gephi_node_inputfile(Parcel_params)
    node_data.to_csv(outdir + 'Groupmat_gephi_nodedata.csv',index=False)

    # now for edges
    edge_data = gfns.make_gephi_edge_inputfile(adj_mat)
    outstr = '%sGroupmat_%sper' %(outdir,thr*100)
    edge_data.to_csv(outstr + '_gephi_edgedata.csv',index=False)

    # import these into gephi and play around different graph layouts
    # Gephi basic instructions (see pdf docs for more):  --- adapted from EM Gordon
    # 1) Open gephi [make sure all plugins are installed: Tools-> available plugins]
    # 2) Do "File->import spreadsheet" and select the nodes file
    # 3) Press "Next", and then scroll down and set the "Color" column to be a String
    # 4) Press "Finish", and say "Append to existing workspace"
    # 5) Do "File->import spreadsheet" and select the edges file
    # 6) Press "Next", "Finish", and say "Append to existing workspace". Ignore any errors.
    # 7) In the "Overview" tab, color the nodes by pressing the little N button with a colorful
    #    circle around it that's on the edge of the graph window
    # 8) In the "Layout" pane (bottom left), select the "Force Layout" option and select run
    # 9) Go to the "Preview" tab at the top, where you can change how the lines behave, plus a lot more
    # 10) Alter default options (see PDF for description of others) and see what happens to the graph

    # Now create across a range of thresholds (1% - 5%) for the group -- IF TIME
    # Discuss consequences of different thresholds
    thresholds = [0.01,0.02,0.03,0.04,0.05] # edit, but beware: these can get unweildy at the higher densities
    for thr in thresholds:
        #rethreshold the matrix
        adj_mat,adj_mat_sym = gfns.threshold_matrix_density(groupmat,thr)
        
        # only edge csv needs to be remade
        edge_data = gfns.make_gephi_edge_inputfile(adj_mat)
        outstr = '%sGroupmat_%sper' %(outdir,thr*100)
        edge_data.to_csv(outstr + '_gephi_edgedata.csv',index=False)

    # Pick your favorite threshold, and do the same for all subjects -- IF TIME
    # 1. Discuss differences
    # 2. Discuss consequences of group vs. individual ROIs and network assignments
    thr = 0.02
    for sub in range(nsubs):
        # threshold subject matrix
        adj_mat,adj_mat_sym = gfns.threshold_matrix_density(subavgmat[sub],thr)

        # create new edge file
        edge_data = gfns.make_gephi_edge_inputfile(adj_mat)
        outstr = '%sMSC%02dmat_%sper' %(outdir,sub+1,thr*100)
        edge_data.to_csv(outstr + '_gephi_edgedata.csv',index=False)


    # for more information, relevant references:
    # Bullmore & Sporns (2009). Complex brain networks: graph theoretical analysis of structural
    #    and functional systems. Nature Reviews Neuroscience, 10 (3), 186-198
    # Sporns (2010). Networks of the Brain. MIT Press.
    # Spring Embedding Methods (Kamada-Kawai): https://arxiv.org/pdf/1201.3011.pdf

# --------------------------------------------------------------------------------------
# 4. Hub measures

if step4_hubs:

    # a constant (already used for infomap, can't be changed)
    thresholds = np.arange(0.01,0.11,0.01)
    
    # 1. Discuss infomap approach
    # Infomap: http://www.mapequation.org/code.html
    #    Rosvall & Bergstrom (2008). Maps of information flow reveal
    #    community structure in complex networks. PNAS, 105, 1118
    # 2. Discuss other approaches to network definition
    # 3. Discuss parameters that must be set
    
    # Start with a set of thresholded correlation matrices and network assignments for each threshold
    # [In the interest of time/ease, I am pre-computing network assignments and providing them here]
    # mat file has key = 'clrs' that lists network assignment across thresholds
    Group_infomapcomm = spio.loadmat(datadir + 'Allsubavg_333parcels_infomapassn.mat')

    # Compute hub measures - degree, PC, and WD - in the group across different thresholds
    # See formula from Guimera & Amaral (2005). Functional Cartography of Complex Metabolic
    #    Networks. Nature, 433, 895-900
    #     http://www.nature.com/nature/journal/v433/n7028/full/nature03288.html?foxtrotcallback=true
    # Practice writing code on your own for this [?]
    group_pc = np.empty((nrois,thresholds.size))
    group_wd = np.empty((nrois,thresholds.size))
    group_degree = np.empty((nrois,thresholds.size))
    for t in range(thresholds.size):
        adj_mat,adj_mat_sym = gfns.threshold_matrix_density(groupmat,thresholds[t]) # get a thresholded matrix
        [group_pc[:,t], group_wd[:,t], group_degree[:,t]] = gfns.hub_metrics(adj_mat_sym,Group_infomapcomm['clrs'][:,t])

    fig = gfns.figure_hubs(Parcel_params,thresholds,group_degree,group_wd,group_pc)
    plt.savefig(outdir + 'Hubmeasures_group.pdf')

    # [If time], do this across subjects
    for sub in range(nsubs):
        subnum = 'MSC%02d' % (sub+1)
        sub_infomapcomm = spio.loadmat(datadir + subnum + '_333parcels_infomapassn.mat')

        sub_pc = np.empty((nrois,thresholds.size))
        sub_wd = np.empty((nrois,thresholds.size))
        sub_degree = np.empty((nrois,thresholds.size))
        for t in range(thresholds.size):
            adj_mat,adj_mat_sym = gfns.threshold_matrix_density(subavgmat[sub],thresholds[t]) # get a thresholded matrix
            [sub_pc[:,t], sub_wd[:,t], sub_degree[:,t]] = gfns.hub_metrics(adj_mat_sym,sub_infomapcomm['clrs'][:,t])

        fig = gfns.figure_hubs(Parcel_params,thresholds,sub_degree,sub_wd,sub_pc)
        plt.savefig(outdir + 'Hubmeasures_' + subnum + '.pdf')
    plt.close('all')
    # Discuss challenges of making hub measures per subject
    # See: Gordon, E, et al. (2018) "Three distinct sets of connector hubs integrate human brain function."
    #     Cell reports 24.7: 1687-1695.

    # [If time] Make a spring embedded plot, colored by hub measures rather than networks
    # Start with group and favorite threshold. Do other versions if time.
    t = 2;
    pc_hub_colors = gfns.hub_colormap(group_pc[:,t])
    node_data = gfns.make_gephi_node_inputfile(Parcel_params,nod_colors=pc_hub_colors)
    node_data.to_csv(outdir + 'Groupmat_gephi_nodedata_PC.csv',index=False)

    wd_hub_colors = gfns.hub_colormap(group_wd[:,t])
    node_data = gfns.make_gephi_node_inputfile(Parcel_params,nod_colors=wd_hub_colors)
    node_data.to_csv(outdir + 'Groupmat_gephi_nodedata_WD.csv',index=False)

    deg_hub_colors = gfns.hub_colormap(group_degree[:,t])
    node_data = gfns.make_gephi_node_inputfile(Parcel_params,nod_colors=deg_hub_colors)
    node_data.to_csv(outdir + 'Groupmat_gephi_nodedata_degree.csv',index=False)

    # For more work on hubs and their importance in brain function, in addition to the references above, see:
    # Gratton, C., et al., (2012). Focal brain lesions to critical locations cause widespread disruption of the
    #   modular organization of the brain. Journal of Cognitive Neuroscience, 24 (6), 1275-1285
    # Power, J.D. et al. (2013). Evidence for hubs in human functional brain networks. Neuron, 79 (4), 798-813
    # Warren, D.E., et al. (2014). Network measures predict neuropsychological outcome after brain injury. PNAS, 111 (39), 14247-14252

    # The following packages contain tools for graph theoretical analyses:
    # Brain Connectivity Toolbox (Sporns, Matlab/Python/C++): https://sites.google.com/site/bctnet/
    # NetworkX (Python): https://networkx.github.io/ 
    #   see also brainx extension: https://github.com/nipy/brainx

