# imports
import numpy as np
from scipy import stats

import functions.helperFunctions as helper


def create_time_vector(epoch_limits, srate):
    tt = np.arange(start=epoch_limits[0],
                   stop=epoch_limits[1]+1/srate, step=1/srate)
    tt = np.delete(tt, 0)
    return tt


def ccep_CAR64blocks(df_data_in, ttt, good_channels):

    signaldata = np.stack(df_data_in['data'].values)

    # which channels have lots of noise:
    #   Set a threshold for which channels to reject based on variance
    #   The result is a tuple with first all the row indices, then all the column indices.
    start_th = np.where(ttt == helper.find_nearest(ttt, .500))[0][0]+1
    end_th = np.where(ttt == helper.find_nearest(ttt, 2))[
        0][0]  # The result is a tuple with first all the row indices, then all the column indices.\

    data_var = signaldata[:, start_th:end_th]

    # “Delta Degrees of Freedom”: the divisor used in the calculation is N - ddof, where N represents the number of elements.
    # By default ddof is zero. Set it to 1 to get the MATLAB result
    chan_var = np.var(data_var, axis=1, ddof=1)
    df_data_in.insert(2, "channel_var", chan_var, True)

    good_channels_var = df_data_in[df_data_in['channel'].isin(
        good_channels['name'])]['channel_var']

    var_th = helper.quantile(good_channels_var, 0.75)  # costum function

    # set a threshold for which channels to reject based on response period
    # The result is a tuple with first all the row indices, then all the column indices.
    resp_start = np.where(ttt == helper.find_nearest(ttt, .010))[0][0]+1
    # The result is a tuple with first all the row indices, then all the column indices.
    resp_end = np.where(ttt == helper.find_nearest(ttt, .100))[0][0]

    resp_data = signaldata[:, resp_start:resp_end]

    resp_var = np.var(resp_data, axis=1, ddof=1)
    df_data_in.insert(3, "response_var", resp_var, True)

    good_channels_resp_var = df_data_in[df_data_in['channel'].isin(
        good_channels['name'])]['response_var']
    # chan_var OR resp_var???
    resp_th = helper.quantile(good_channels_var, 0.75)

    # these are the channels that are ok to include in the CAR for this
    # stimulation pair (should exclude stim pair, but not explicitly):
    curr_indexes_chan_var = good_channels_var[(
        good_channels_var > var_th)].index  # Get indexes
    # also exclude channels with a larger response
    curr_indexes_resp_vars = good_channels_resp_var[(
        good_channels_resp_var > resp_th)].index  # Get indexes
    df_chans_incl = good_channels[~good_channels.index.isin(curr_indexes_resp_vars.union(
        curr_indexes_chan_var))]  # Delete these row indexes from dataFrame

    # we split the original channels into 64 channel blocks and take the mean of each group of good channels
    df_data_chans_incl = df_data_in[df_data_in.index.isin(
        df_chans_incl.index.to_list())]

    car = df_data_chans_incl['data'].values.mean() # common average reference across all channels

    """
    # calc good channels mean per group
    car_sets = np.empty(len(df_data_in.groupNum.unique()), dtype=object)

    for ind in df_data_in.groupNum.unique():
        i = int(ind)
        car_sets[i] = (
            df_data_chans_incl[df_data_chans_incl['groupNum'] == i]['data'].values.mean())
    """

    # substract CAR from each channel
    for idx, row in df_data_in.iterrows():
        #groupmean = car_sets[row['groupNum']]
        df_data_in['data'].loc[idx] = row['data'] - car

    return df_data_in


def bpcVoltage(V_pre, tt, BPCs_epoch):
    tt_BPCs = []
    tt_BPCs.append(
        np.where(tt == helper.find_nearest(tt, BPCs_epoch[0]))[0][0]+1)
    tt_BPCs.append(
        np.where(tt == helper.find_nearest(tt, BPCs_epoch[1]))[0][0])

    V = []
    for i in range(len(V_pre)):
        trial_V = V_pre[i]
        BPC_V = trial_V[tt_BPCs[0]:tt_BPCs[1]]
        V.append(BPC_V)

    V = np.vstack(V).T  # Stack arrays in sequence vertically (row wise)

    return V, tt_BPCs


def nativeNormalized(pair_types, P):
    n = len(pair_types.index)
    S = []
    tmat = []

    for k in pair_types.index:
        for l in pair_types.index:
            if k == l:  # diagonal
                # gather all off-diagonal elements from self-submatrix
                a = P[np.ix_(pair_types['indices'][k],
                             pair_types['indices'][k])]
                b = []
                # for q=1:(size(a,2)-1), b=[b a(q,(q+1):end)]; end
                for q in range(a.shape[1]-1):
                    b.extend(a[q, q+1:])
                # for q=2:(size(a,2)), b=[b a(q,1:(q-1))]; end
                for q in range(1, a.shape[1]):
                    b.extend(a[q, :q])
            else:
                # b=reshape(P(pair_types(k).indices,pair_types(l).indices),1,[]);
                b = P[np.ix_(pair_types['indices'][k],
                             pair_types['indices'][l])].ravel()

            S.append(b)

            b_mean = np.mean(b)
            b_std = np.std(b, ddof=1)
            b_sqrt_n = np.sqrt(len(b))

            t = b_mean/(b_std/b_sqrt_n)  # calculate t-statistic
            tmat.append(t)

    S = np.array(S).reshape(-1, n)
    tmat = np.array(tmat).reshape(-1, n)

    return tmat


def curvesStatistics(B_struct, V, B, pair_types):
    B_struct['alphas'] = None
    B_struct['ep2'] = None
    B_struct['V2'] = None
    B_struct['errxproj'] = None

    for bb in range(B_struct.shape[0]):  # cycle through basis curves

        # alpha coefficient weights for basis curve bb into V
        al = B_struct.curve[bb] @ V
        # np.newaxis comes in handy when we want to explicitly convert a 1D array to either a row vector or a column vector!!!!
        # residual epsilon (error timeseries) for basis bb after alpha*B coefficient fit
        ep = V - B_struct.curve[bb][np.newaxis].T @ al[np.newaxis]
        errxproj = ep.T  @ ep  # calculate all projections of error
        V_selfproj = V.T @ V  # power in each trial

        B_struct['alphas'][bb] = []
        B_struct['ep2'][bb] = []
        B_struct['V2'][bb] = []
        B_struct['errxproj'][bb] = []

        pair_types = pair_types.reset_index(drop=True)

        # cycle through pair types represented by this basis curve
        for n in range(len(B_struct.pairs[bb])):
            ind = (B_struct.pairs[bb])[n]
            tmp_inds = pair_types['indices'][ind]  # indices for this pair type
            # alpha coefficient weights for basis curve bb into V
            (B_struct.alphas[bb]).append(al[tmp_inds])
            # self-submatrix of error projections
            a = errxproj[np.ix_(tmp_inds, tmp_inds)]
            (B_struct.ep2[bb]).append((np.diag(a)).T)  # sum-squared error
            # sum-squared individual trials
            (B_struct.V2[bb]).append(
                np.diag(V_selfproj[np.ix_(tmp_inds, tmp_inds)]).T)

            # gather all off-diagonal elements from self-submatrix
            b = []
            # for q=1:(size(a,2)-1), b=[b a(q,(q+1):end)]; end
            for q in range(a.shape[1]-1):
                b.extend(a[q, q+1:])
            # for q=2:(size(a,2)), b=[b a(q,1:(q-1))]; end
            for q in range(1, a.shape[1]):
                b.extend(a[q, :q-1])

            # systematic residual structure within a stim pair group for a given basis will be given by set of native normalized internal cross-projections
            B_struct.errxproj[bb] = b
    return B_struct


def projectionWeights(B_struct):

    B_struct['p'] = None
    B_struct['plotweights'] = None

    for q in range(B_struct.shape[0]):  # cycle through basis curves
        B_struct['p'][q] = []
        B_struct['plotweights'][q] = []
        # cycle through pair types represented by this basis curve
        for n in range(B_struct.pairs[q].shape[0]):
            curr_alphas = B_struct.alphas[q][n]
            curr_ep2_5 = (B_struct.ep2[q][n])**0.5
            # alphas normalized by error magnitude
            B_struct.plotweights[q].append(np.mean(curr_alphas / curr_ep2_5))

            # significance alphas normalized by error magnitude
            t, pVal = stats.ttest_1samp((curr_alphas / curr_ep2_5), 0)
            (B_struct.p[q]).append(pVal)

    return B_struct

def kpca(X):

    F,S,_=np.linalg.svd(X.T)  # Compute the eigenvalues and right eigenvectors.
    ES = X @ F # kernel trick
    E = ES / (np.ones((X.shape[0],1)) @ S[np.newaxis]) # divide through to obtain unit-normalized eigenvectors

    return E
