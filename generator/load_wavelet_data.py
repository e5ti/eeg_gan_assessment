import numpy as np
import scipy 
import os
import pandas as pd

def normalize(serie):
    return (serie - np.min(serie)) / (np.max(serie) - np.min(serie))

def split_series(series, n_past, n_future, filter_future = False, skip = 16):
    # n_past ==> no of past observations
    # n_future ==> no of future observations 

    X, y = list(), list()
    for window_start in range(0, series.shape[1], skip):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > series.shape[1]:
            break
        past, future = series[:, window_start:past_end], series[:, past_end:future_end]
        X.append(past)
        y.append(future)
    return np.transpose(np.array(X), (0,2,1)), np.transpose(np.array(y), (0,2,1))


def load_data(FS = 256, select_subj = None, my_channels = [4, 5, 11, 12, 15], 
                MAT_ARRAY = ["02", "04", "05", "07", "09", "10", "11", "13", "14", "16"], 
                true_loc = "/home/ettore/MyCode/EEGForecasting/Z_EEG_wavelets/",
                fixed_len = 256*60*21,
                dataname = "kuka", 
                key = "wav_result_alpha",
                load_synth=False, 
                skip_overlap = 16):

    n_features = len(my_channels)
    min_list = np.zeros((len(MAT_ARRAY), n_features))
    max_list = np.zeros((len(MAT_ARRAY), n_features))

    n_future = 32
    n_past = 256

    X_real = np.empty((0, n_past, int(n_features)))

    key = "EEG_wavelet"
    X_temp = np.empty((len(my_channels), FS, 0))

    count_j = 0

    for idx, i in enumerate(MAT_ARRAY):
        for j, eyes in enumerate(["EC", "EO"]):
            # ASSIGN SECOND BY SECOND OF ACQUISITION
            if select_subj is None or select_subj != idx:
                data_i = scipy.io.loadmat(f"{true_loc}cwt_coeff_{eyes}_{i}.mat")[key][my_channels, :, :]
                jj_idx = []
                for jj in range(data_i.shape[2]):
                    # esclude flat portions of the signal
                    if (np.min(data_i[:, :, jj]) == 0) and (np.max(data_i[:, :, jj]) == 0):
                        count_j +=1
                    else:
                        jj_idx.append(jj)

                data_i = data_i[:, :, jj_idx]
                data_i = normalize(data_i)
                X_temp = np.concatenate((X_temp, data_i), axis=2)

    # append data to arrays. Eval subject is found by idx comparison
    X_real = np.swapaxes(X_temp, 0, 2)
    print(X_real.shape)

    return X_real
