import models
import utils_train
import pandas as pd
from config import *
import mir_eval

from scipy.signal import medfilt2d

import os
import argparse



'''Predict multiple F0 values for the Barbershop quartets files using model3 trained in experiment 4
'''


''' Parameters
'''


def main(args):

    pth_model = args.pth_model
    save_path = args.save_path
    list_of_files = args.list_of_files

    gt_path = '/scratch/hc2945/data/test_data'

    audio_path = np.array(pd.read_csv(list_of_files, header=None))[0][0]
    fname_list = np.array(pd.read_csv(list_of_files, header=None))[1:]


    save_key = 'exp4multif0'
    model_path = os.path.join(pth_model, "{}.pkl".format(save_key))

    model = models.build_model3()
    model.load_weights(model_path)

    model.compile(
        loss=utils_train.bkld, metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )

    print("Model compiled")

    all_scores_100 = []
    all_scores_20 = []

    for fname in fname_list:

        fname = fname[0]

        if not fname.endswith('.wav'): continue

        # predict using trained model
        predicted_output, _, _ = utils_train.get_single_test_prediction(model,
                                                                  npy_file=None,
                                                                  audio_file=os.path.join(audio_path, fname))

        '''
        np.save(os.path.join(
            save_path, "{}_prediction.npy".format(fname.split('.')[0])),
            predicted_output.astype(np.float32))
        '''

        # load ground truth
        ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(
            os.path.join(gt_path, fname.replace('wav', 'csv'))
        )
        for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            if any(fqs <= 0):
                ref_freqs[i] = np.array([f for f in fqs if f > 0])

        # optimize threshold for this data
        '''
        accuracies = []

        # get multif0 output from prediction
        thresholds = [0.2, 0.3, 0.4, 0.5]
        for thresh in thresholds:

            est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, thresh)

            for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
                if any(fqs <= 0):
                    est_freqs[i] = np.array([f for f in fqs if f > 0])

            # evaluation
            scores = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, window=1)
            accuracies.append(scores['Accuracy'])

        mx_idx = np.argmax(accuracies)
        trsh = thresholds[mx_idx]
        '''

        trsh = 0.5

        est_times, est_freqs = utils_train.pitch_activations_to_mf0(predicted_output, trsh)
        for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
            if any(fqs <= 0):
                est_freqs[i] = np.array([f for f in fqs if f > 0])

        '''Save output multi-f0
        '''
        output_mf0 = os.path.join(save_path, "{}_{}.csv".format(fname.split('.')[0], trsh))
        utils_train.save_multif0_output(est_times, est_freqs, output_mf0)

        scores_100 = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, window=1)
        scores_100['track'] = fname.replace('.wav', '')
        all_scores_100.append(scores_100)

        scores_20 = mir_eval.multipitch.evaluate(ref_times, ref_freqs, est_times, est_freqs, window=0.2)
        scores_20['track'] = fname.replace('.wav', '')
        all_scores_20.append(scores_20)

        print("     Multiple F0 prediction exported and evaluated on 100 and 20 cents for {}".format(fname))

    # export results with 100 cents
    scores_path = os.path.join(
        save_path, '{}_all_scores_100_cents.csv'.format('test_set')
    )
    score_summary_path = os.path.join(
        save_path, "{}_score_100_cents_summary.csv".format('test_set')
    )
    df = pd.DataFrame(all_scores_100)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)

    # export results with 20 cents
    scores_path = os.path.join(
        save_path, '{}_all_scores_20_cents.csv'.format('test_set')
    )
    score_summary_path = os.path.join(
        save_path, "{}_score_20_cents_summary.csv".format('test_set')
    )
    df = pd.DataFrame(all_scores_20)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output using model 3 without reverb. Experiment 5.")

    parser.add_argument("--model_path",
                        dest='pth_model',
                        type=str,
                        help="Path to the model weights.")

    parser.add_argument("--save_path",
                        dest='save_path',
                        type=str,
                        help="Folder to save predicted outputs.")

    parser.add_argument("--list_of_files",
                        dest='list_of_files',
                        type=str,
                        help="Path to the text file with the list of files to process. The first line should contain"
                        "the folder where the files are located.")


    main(parser.parse_args())
