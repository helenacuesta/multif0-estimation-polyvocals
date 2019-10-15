import models
import utils
from utils import *
from config import *
import mir_eval

import matplotlib.pyplot as plt

import pandas
import os
import glob
import csv
import ast

'''Bunch of code that will be copied and adapted from the original pipeline for easier use in this context (basically from evaluate_models)
'''


def load_broken_mf0(annotpath):
    '''Equivalent function to load_ragged_time_series in mir_eval for bad-formatted csv files
        as the ones I have now.
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


class ExternalData(object):
    """Class that deals with all the data mess
    """

    def __init__(self, audio_folder, input_patch_size, batch_size,
                 active_str, muxrate):
        self.input_patch_size = input_patch_size

        self.audio_folder = audio_folder

        self.list_of_files = self.load_data_list()

        self.batch_size = batch_size
        self.active_str = active_str
        self.muxrate = muxrate

    def load_data_list(self):
        """load list of files to evaluate
        """
        files_list = os.listdir(self.audio_folder)
        return files_list

    def get_test_generator(self):
        """return a test data generator
        """
        return keras_generator(
            self.list_of_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )


def score_on_list_of_files(model, list_of_files, npy_folder, gt_folder, save_path, thresh):
    """score a model on all files in a list
    """
    #test_npy_files = os.listdir(npy_folder)
    test_npy_files = list_of_files

    all_scores = []

    for npy_file in sorted(test_npy_files):
        if not npy_file.endswith('npy'): continue
        gt_fname = os.path.basename(npy_file).replace('.npy', '_multif0.csv')
        npy_file = 'rev_' + npy_file[:-4] + '_multif0_input_dph.npy'
        print("Scoring for {}".format(npy_file))
        npy_file = os.path.join(npy_folder, npy_file)
        
        fname_base = gt_fname[:-4]

        label_file = os.path.join(gt_folder, gt_fname)
        ''''''

        # generate prediction on numpy file
        predicted_output, input_hcqt, input_dphase = \
            get_single_test_prediction(model, npy_file)

        # save prediction
        np.save(
            os.path.join(
                save_path,
                "{}_prediction.npy".format(fname_base)
            ),
            predicted_output.astype(np.float32)
        )

        # from scipy.signal import medfilt2d
        # predicted_output = medfilt2d(predicted_output, kernel_size=(1, 3))

        # predicted_output = np.load(os.path.join(save_path, "{}_prediction.npy".format(fname_base)))

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

        ref_times, ref_freqs = load_broken_mf0(label_file)
        # ref_times, ref_freqs = \
        #    mir_eval.io.load_ragged_time_series(label_file)

        for i, (tms, fqs) in enumerate(zip(ref_times, ref_freqs)):
            if any(fqs <= 0):
                ref_freqs[i] = np.array([f for f in fqs if f > 0])

        for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
            if any(fqs <= 0):
                est_freqs[i] = np.array([f for f in fqs if f > 0])

        # get multif0 metrics and append
        scores = mir_eval.multipitch.evaluate(
            ref_times, ref_freqs, est_times, est_freqs)
        scores['track'] = fname_base
        all_scores.append(scores)

    # save scores to data frame
    scores_path = os.path.join(
        save_path, '{}_all_scores.csv'.format('SubsetWithReverb')
    )
    score_summary_path = os.path.join(
        save_path, "{}_score_summary.csv".format('SubsetWithReverb')
    )
    df = pandas.DataFrame(all_scores)
    df.to_csv(scores_path)
    df.describe().to_csv(score_summary_path)
    print(df.describe())


''''''

# save_key should be the name of the model: multif0_exp5, multif0_exp5_single, multif0_exp6, multif0_exp6_single, multif0_exp7


input_patch_size = (360, 50)
rev_features_folder = '/scratch/hc2945/multif0/AudioMixtures/inputs_dph'
gt_folder = '/scratch/hc2945/multif0/VocalEnsembles/test_data'


'''Compute features
for fn in os.listdir(audio_folder):
    if not fn.endswith('wav'): continue
    compute_features.compute_features(audio_folder, fn, audio_folder)
'''

models_to_use = ['model5', 'model5filt', 'model6', 'model6filt', 'model7', 'model7filt']

for md in models_to_use:

    if md == 'model5':
        model = models.build_model5()
        save_key = 'multif0_exp5'

    elif md == 'model5filt':
        model = models.build_model5()
        save_key = 'multif0_exp5_filtered'

    elif md == 'model6':
        model = models.build_model6()
        save_key = 'multif0_exp6'

    elif md == 'model6filt':
        model = models.build_model6()
        save_key = 'multif0_exp6_filtered'

    elif md == 'model7':
        model = models.build_model7()
        save_key = 'multif0_exp7'

    elif md == 'model7filt':
        model = models.build_model7()
        save_key = 'multif0_exp7_filtered'

    else:
        print("Wrong model")
        continue

    model_path = os.path.join('../multif0-ds-singing/models', save_key + '.pkl')

    model.load_weights(model_path)
    print("{} built".format(md))

    model.compile(
        loss=bkld, metrics=['mse', soft_binary_accuracy],
        optimizer='adam'
    )

    print("Model compiled")

    thresholds = [0.2, 0.3, 0.4]

    #fnlist = np.loadtxt('./songs_to_evaluate.txt')
    fnlist = ['2_DG_take1_3_3_2_2.npy',
              '3_CSD_ER_3_3_2_4.npy', '0_CSD_ND_2_2_1_4.npy',
              '1_CSD_ER_3_2_2_1.npy', '1_WU_take1_1_2_1_1.npy',
              '2_CSD_ND_1_1_1_1.npy', '2_DG_take1_1_3_1_1.npy',
              '4_WU_take1_1_3_3_1.npy', '4_CSD_ND_2_2_2_4.npy',
              '3_DG_take2_1_2_3_1.npy', '3_CSD_LI_4_4_4_1.npy']

    for thrsh in thresholds:
        th_folder = './results/{}/{}'.format(md, thrsh)
        if not os.path.exists(th_folder):
            os.mkdir(th_folder)
        print("Using threshold = {}".format(thrsh))
        score_on_list_of_files(model, fnlist, rev_features_folder, gt_folder, th_folder, thresh=thrsh)



