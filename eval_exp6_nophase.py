import os
import json

import numpy as np
import csv

import config
import scipy

import utils
import mir_eval
import pandas as pd

import ast


def load_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data

def pitch_activations_to_mf0(pitch_activation_mat, thresh):
    """Convert a pitch activation map to multif0 by thresholding peak values
    at thresh
    """
    freqs = utils.get_freq_grid()
    times = utils.get_time_grid(pitch_activation_mat.shape[1])

    peak_thresh_mat = np.zeros(pitch_activation_mat.shape)
    peaks = scipy.signal.argrelmax(pitch_activation_mat, axis=0)
    peak_thresh_mat[peaks] = pitch_activation_mat[peaks]

    idx = np.where(peak_thresh_mat >= thresh)

    est_freqs = [[] for _ in range(len(times))]
    for f, t in zip(idx[0], idx[1]):
        est_freqs[t].append(freqs[f])

    est_freqs = [np.array(lst) for lst in est_freqs]
    return times, est_freqs

def get_best_thresh(test_files, files_pth):
    """Use validation set to get the best threshold value
    """

    # get files for this test set
    test_set_path = '/scratch/hc2945/data/test_data'

    thresh_vals = np.arange(0.1, 0.8, 0.1)

    thresh_scores = {t: [] for t in thresh_vals}

    for npy_file in test_files:

        predicted_output = np.load(os.path.join(files_pth, npy_file), allow_pickle=True)

        fname_base = os.path.basename(npy_file).replace('_prediction.npy', '.csv')

        label_file = os.path.join(
                test_set_path, fname_base)

        # load ground truth labels
        ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(label_file)
        #ref_times, ref_freqs = load_broken_mf0(label_file)

        for thresh in thresh_vals:
            # get multif0 output from prediction
            est_times, est_freqs = pitch_activations_to_mf0(predicted_output, thresh)

            # get multif0 metrics and append
            scores = mir_eval.multipitch.evaluate(
                ref_times, ref_freqs, est_times, est_freqs)
            thresh_scores[thresh].append(scores['Accuracy'])

    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print("Best Threshold is {}".format(best_thresh))
    print("Best accuracy is {}".format(np.max(avg_thresh)))
    print("Validation accuracy at 0.5 is {}".format(np.mean(thresh_scores[0.5])))

    return best_thresh




def save_multif0_output(times, freqs, output_path):
    """save multif0 output to a csv file
    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)


def load_broken_mf0(annotpath):
    '''Equivalent function to load_ragged_time_series in mir_eval for bad-formatted csv files
    '''

    times = []
    freqs = []
    with open(annotpath, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            times.append(float(line[0]))
            fqs = ast.literal_eval(line[1])
            freqs.append(np.array(fqs))

    times = np.array(times)

    # get rid of zeros for input to mir_eval
    for i, (tms, fqs) in enumerate(zip(times, freqs)):
        if any(fqs == 0):
            freqs[i] = np.array([f for f in fqs if f > 0])

    return times, freqs

def score_on_test_set(test_files, files_pth, thresh, save_path):
    """score a model on all files in a named test set
    """

    # get files for this test set
    test_set_path = '/scratch/hc2945/data/test_data'

    all_scores = []
    for npy_file in sorted(test_files):

        predicted_output = np.load(os.path.join(files_pth, npy_file), allow_pickle=True)

        print(npy_file)

        # get input npy file and ground truth label pair
        fname_base = os.path.basename(npy_file).replace('_prediction.npy', '.csv')
        label_file = os.path.join(
                test_set_path, fname_base)


        # get multif0 output from prediction
        est_times, est_freqs = pitch_activations_to_mf0(
            predicted_output, thresh
        )

        # save multif0 output
        save_multif0_output(
            est_times, est_freqs,
            os.path.join(
                save_path,
                "{}_prediction.txt".format(fname_base)
            )
        )

        # load ground truth labels
        try:
            ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(label_file)
        except:
            ref_times, ref_freqs = load_broken_mf0(label_file)

        # get multif0 metrics and append
        scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        scores['track'] = fname_base
        all_scores.append(scores)

    # save scores to data frame
    scores_path = os.path.join(
        save_path, '{}_all_scores.csv'.format('test_set')
    )
    score_summary_path = os.path.join(
        save_path, "{}_score_summary.csv".format('test_set')
    )
    df = pd.DataFrame(all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())


def main():


    save_key = 'exp6multif0'
    pth_pred = os.path.join(config.exper_output, save_key)

    test_files = []
    for fname in os.listdir(pth_pred):
        if not '_prediction' in fname: continue
        else:
            test_files.append(fname)

    best_thresh = get_best_thresh(test_files, pth_pred)

    score_on_test_set(test_files, pth_pred, best_thresh, pth_pred)




main()
