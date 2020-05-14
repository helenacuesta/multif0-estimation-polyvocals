
import numpy as np
import mir_eval
import pandas as pd

import os

def score_on_file_list(pred_folder, save_path):

    """score a model on all files in the input folder
    """

    pth_gt = '/scratch/hc2945/data/test_data'

    all_scores = []

    for fname in os.listdir(pred_folder):

        if not fname.endswith('csv'): continue


        # load ground truth labels
        ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(
            os.path.join(pth_gt, fname.replace('_multif0_multif0.csv', '.csv'))
        )

        # load predicted labels
        est_times, est_freqs = mir_eval.io.load_ragged_time_series(
            os.path.join(pth_pred, fname)
        )

        # get multif0 metrics and append
        scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs)
        scores['track'] = fname.split('.')[0]
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



pth_pred = '/scratch/hc2945/data/deepsalience_output'
save_path = '/scratch/hc2945/data/deepsalience_output'

score_on_file_list(pth_pred, save_path)
