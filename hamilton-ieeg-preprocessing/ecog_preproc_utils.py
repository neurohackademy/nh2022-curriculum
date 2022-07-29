# ecog_preproc_utils

import numpy as np
import os
import mne
from pyfftw.interfaces.numpy_fft import fft, ifft, fftfreq
from tqdm import tqdm


def auto_bands(fq_min=4.0749286538265, fq_max=200., scale=7.):
    """
    Get the frequency bands of interest for the neural signal decomposition.
    Usually these are bands between 4 and 200 Hz, log spaced. 
    These filters were originally chosen by Erik Edwards for his
    thesis work (LH)

    Inputs:
    ----------------
    fq_min : minimum frequency center
    fq_max : maximum frequency center
    scale : scaling factor (don't change)
    
    Outputs:
    ----------------
    cts : center frequencies for the filtering
    sds : bandwidths for the filtering
    """
    cts = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
    sds = 10 ** (np.log10(.39) + .5 * (np.log10(cts)))
    return cts, sds


def applyHilbertTransform(X, rate, center, sd):
    """
    Apply bandpass filtering with Hilbert transform using a Gaussian kernel
    """
    # frequencies
    T = X.shape[-1]
    freq = fftfreq(T, 1/rate)
    # heaviside kernel
    h = np.zeros(len(freq))
    h[freq > 0] = 2.
    h[0] = 1.
    # bandpass transfer function
    k = np.exp((-(np.abs(freq)-center)**2)/(2*(sd**2)))
    # compute analytical signal
    Xc = ifft(fft(X)*h*k)
    return Xc


def transformData(raw, data_dir, band='high_gamma', notch=True, CAR=True,
                  car_chans='average', log_transform=True, do_zscore=True,
                  hg_fs=100, notch_freqs=[60,120,180],
                  ch_types='eeg', overwrite=False, save=False, out_dir=None):
    
    # The suffix that will be added to the file name as
    # different procedures occur
    full_suffix = ''

    raw.load_data()
    #raw.pick_types(meg=False, eeg=True, ecog=True) 
    nchans = raw.info['nchan']

    band_ranges = {'delta': [None, 4],
                   'theta': [4, 8],
                   'alpha': [8, 15],
                   'beta':  [15, 30],
                   'gamma': [32, 70],
                   'high_gamma': [70, 150]}

    if notch:
        full_suffix += '_notch'
        print("Doing notch filter")
        raw.plot_psd()
        raw.notch_filter(notch_freqs)
        raw.plot_psd()
        raw.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                 title='notch filtered raw data')
        # try:
        #     newfile = os.path.join(data_dir, 'Raw', f'ecog_raw{full_suffix}.fif')
        #     raw.save(newfile, overwrite=overwrite)
        # except:
        #     print(f"Can't save {newfile}. Do you need overwrite=True?")

    if CAR:
        full_suffix += '_car'
        print("Doing CAR on")
        print(car_chans)
        raw.set_eeg_reference(car_chans)
        raw.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                 title='after referencing (CAR)')
        # try:
        #     newfile = os.path.join(data_dir, 'Raw', f'ecog_raw{full_suffix}.fif')
        #     raw.save(newfile, overwrite=overwrite)
        # except:
        #     print(f"Can't save {newfile}. Do you need overwrite=True?")

    # Get center frequencies and standard deviations of the bands
    # for the Hilbert transform
    print("Getting frequency bands for Hilbert transform")
    cts, sds = auto_bands()

    if band == 'high_gamma':
        print(f"Getting {band} band data")
        hg_dir = os.path.join(data_dir, 'HilbAA_70to150_8band')
        if not os.path.isdir(hg_dir):
            print("Creating directory %s" %(hg_dir))
            os.mkdir(hg_dir)

        # determine size of our high gamma band
        f_low = band_ranges[band][0]
        f_high = band_ranges[band][1]

        sds = sds[(cts>=f_low) & (cts<=f_high)]
        cts = cts[(cts>=f_low) & (cts<=f_high)]

    elif (band == "alpha") or (band=="theta") or (band=="delta") or (band=="beta") or (band=="gamma"):
        f_low = band_ranges[band][0]
        f_high = band_ranges[band][1]

        if out_dir is None:
            out_dir = os.path.join(data_dir, f'{band}_{f_low}to{f_high}')
        if not os.path.isdir(out_dir):
            print("Creating directory %s" %(out_dir))
            os.mkdir(out_dir)

        fname = f'ecog_{band}_{f_low}to{f_high}{full_suffix}.fif'

        print(f"Filtering data in {band} band from {f_low} to {f_high} Hz")
        print("Note that this will *not* use the analytic amplitude like high gamma")
        raw.filter(l_freq=f_low, h_freq=f_high)
        if save:
            raw.save(os.path.join(out_dir, fname))
        raw.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                 title=f'after filtering in {band} band')
        transformed_data = raw.copy()

    elif (band == "broadband"):
        f_low = 4
        f_high = 200
        hg_dir = os.path.join(data_dir, f'HilbAA_{f_low}to{f_high}_40band')
        if not os.path.isdir(hg_dir):
            print("Creating directory %s" %(hg_dir))
            os.mkdir(hg_dir)

    if (band == "high_gamma") or (band == "broadband"):
        # do the Hilbert transform
        print("Getting the raw data array")
        raw_data = raw.get_data()
        dat = []
        
        for i, (ct, sd) in enumerate(tqdm(zip(cts, sds), 'applying Hilbert transform...', total=len(cts))):
            hilbdat = np.zeros((raw_data.shape))
            for ch in np.arange(nchans):
                hilbdat[ch,:] = applyHilbertTransform(raw_data[ch,:], raw.info['sfreq'], ct, sd)
            if log_transform:
                print("Taking log transform of high gamma")
                dat.append(np.log2(np.abs(hilbdat.real.astype('float32') + 1j*hilbdat.imag.astype('float32'))))
            else:
                dat.append(np.abs(hilbdat.real.astype('float32') + 1j*hilbdat.imag.astype('float32')))

        if log_transform:
            full_suffix+='_log'

        # hilbmat is now the analytic amplitude matrix
        hilbmat = np.array(np.hstack((dat))).reshape(dat[0].shape[0], -1, dat[0].shape[1])
        
    if band == "broadband":
        print("Saving the full 40 band matrices (not yet done...)")
        ### TO BE DEVELOPED

        for b in np.arange(hilbmat.shape[1]): 
            print(f"Doing band {b}")
            band_signal = hilbmat[:,b,:]
            #band_signal = (band_signal - )
            band_signal = (band_signal - np.expand_dims(np.nanmean(band_signal, axis=1), axis=1) )/np.expand_dims(np.nanstd(band_signal, axis=1), axis=1)

            band_info = mne.create_info(raw.info['ch_names'], raw.info['sfreq'], ch_types)

            hgdat = mne.io.RawArray(band_signal, band_info)

            if raw.annotations: # if we rejected something reject it in HG also
                for annotation in raw.annotations:
                    # Add annotations from raw to hg data
                    onset = (annotation['onset']-(raw.first_samp/raw.info['sfreq'])) # convert start time for clin
                    duration = annotation['duration']
                    description = annotation['description']
                    hgdat.annotations.append(onset,duration,description)
            hgdat.resample(hg_fs)
            nband = len(cts)
            fname = f'ecog_hilbAA_{f_low}to{f_high}_{nband}band_bandnum{b}{full_suffix}.fif'
            new_fname = os.path.join(hg_dir, fname) 

            if save:
                print(f"Saving to {new_fname}")
                try:
                    hgdat.save(new_fname, overwrite=overwrite)
                except:
                    print(f"Can't save {new_fname}. Do you need overwrite=True?")
                    
        return hilbmat
    else:
        # average across relevant bands
        print("Taking the mean across %d bands"%(hilbmat.shape[1]))
        hg_signal = hilbmat.mean(1) # Get the average across the relevant bands 
    
        if do_zscore:
            # Z-score
            print("Z-scoring signal")
            hg_signal = (hg_signal - np.expand_dims(np.nanmean(hg_signal, axis=1), axis=1) )/np.expand_dims(np.nanstd(hg_signal, axis=1), axis=1)

        hg_info = mne.create_info(raw.info['ch_names'], raw.info['sfreq'], ch_types)

        hgdat = mne.io.RawArray(hg_signal, hg_info)
        if raw.annotations: # if we rejected something reject it in HG also
            for annotation in raw.annotations:
                # Add annotations from raw to hg data
                onset = (annotation['onset']-(raw.first_samp/raw.info['sfreq'])) # convert start time for clinical data   
                duration = annotation['duration']
                description = annotation['description']
                hgdat.annotations.append(onset,duration,description)

        hgdat.resample(hg_fs)
        nband = len(cts)
        fname = f'ecog_hilbAA_{f_low}to{f_high}_{nband}band{full_suffix}.fif'
        new_fname = os.path.join(hg_dir, fname) 
        if save:
            print(f"Saving to {new_fname}")
            try:
                hgdat.save(new_fname, overwrite=overwrite)
            except:
                print(f"Can't save {new_fname}. Do you need overwrite=True?")
        transformed_data = hgdat.copy()
        hgdat.plot(scalings='auto', color=dict(eeg='b'), n_channels=64, block=True,
                   title=f'after analytic amplitude in {band} band')

        return transformed_data

