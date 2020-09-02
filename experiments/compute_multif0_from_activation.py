from experiments import config

import os
import numpy as np
import csv
import json
import scipy
import librosa

## FUNCTIONS

test_path = '/scratch/hc2945/data/test_data'

def get_hcqt_params():

    bins_per_octave = 60
    n_octaves = 6
    over_sample = 5
    harmonics = [1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length, over_sample


def get_freq_grid():
    """Get the hcqt frequency grid
    """
    (bins_per_octave, n_octaves, _, _, f_min, _, over_sample) = get_hcqt_params()
    freq_grid = librosa.cqt_frequencies(
        n_octaves*12*over_sample, f_min, bins_per_octave=bins_per_octave)
    return freq_grid

def get_time_grid(n_time_frames):
    """Get the hcqt time grid
    """
    (_, _, _, sr, _, hop_length, _) = get_hcqt_params()
    time_grid = librosa.core.frames_to_time(
        range(n_time_frames), sr=sr, hop_length=hop_length
    )
    return time_grid


def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert a pitch activation map to multif0 by thresholding peak values
    at thresh
    """
    freqs = get_freq_grid()
    times = get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return times, est_freqs

def load_json_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data

def main():

    print("Computing multi-f0 ground truth for training files...")
    data_splits = load_json_data(
        os.path.join(config.data_save_folder, 'data_splits.json'))

    for fn in data_splits['train']:

        targ = np.load(os.path.join(config.data_save_folder, 'outputs', fn.replace('.wav', '_output.npy')))
        ts, fs = pitch_activations_to_mf0(targ, 0.9)
        output_mf0 = os.path.join(
            test_path, fn.replace('.wav', '.csv')
        )

        with open(output_mf0, 'w') as fhandle:
            csv_writer = csv.writer(fhandle, delimiter='\t')
            for t, f in zip(ts, fs):
                row = [t]
                row.extend(f)
                csv_writer.writerow(row)

    print("Computing multi-f0 ground truth for validation files...")
    data_splits = load_json_data(
        os.path.join(config.data_save_folder, 'data_splits.json'))

    for fn in data_splits['validate']:

        targ = np.load(os.path.join(config.data_save_folder, 'outputs', fn.replace('.wav', '_output.npy')))
        ts, fs = pitch_activations_to_mf0(targ, 0.9)
        output_mf0 = os.path.join(
            test_path, fn.replace('.wav', '.csv')
        )
        with open(output_mf0, 'w') as fhandle:
            csv_writer = csv.writer(fhandle, delimiter='\t')
            for t, f in zip(ts, fs):
                row = [t]
                row.extend(f)
                csv_writer.writerow(row)

    print("Computing multi-f0 ground truth for training files...")
    data_splits = load_json_data(
        os.path.join(config.data_save_folder, 'data_splits.json'))

    for fn in data_splits['test']:

        targ = np.load(os.path.join(config.data_save_folder, 'outputs', fn.replace('.wav', '_output.npy')))
        ts, fs = pitch_activations_to_mf0(targ, 0.9)
        output_mf0 = os.path.join(
            test_path, fn.replace('.wav', '.csv')
        )
        with open(output_mf0, 'w') as fhandle:
            csv_writer = csv.writer(fhandle, delimiter='\t')
            for t, f in zip(ts, fs):
                row = [t]
                row.extend(f)
                csv_writer.writerow(row)


main()

