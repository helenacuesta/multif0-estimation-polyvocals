import os
import json

import keras
import numpy as np
import csv
import pandas as pd

import config
import utils
import utils_train
import models

import mir_eval

import argparse

class Data(object):
    """Class that deals with all the data mess
    """
    def __init__(self, data_splits_path, data_path, input_patch_size, batch_size,
                 active_str, muxrate):

        self.data_splits_path = data_splits_path
        self.input_patch_size = input_patch_size

        self.data_path = data_path

        (self.train_set,
         self.validation_set,
         self.test_set) = self.load_data_splits()

        self.train_files = utils_train.get_file_paths(self.train_set, self.data_path)
        self.validation_files = utils_train.get_file_paths(
            self.validation_set, self.data_path
        )
        self.test_files = utils_train.get_file_paths(self.test_set, self.data_path)

        self.batch_size = batch_size
        self.active_str = active_str
        self.muxrate = muxrate

    def load_data_splits(self):

        with open(self.data_splits_path, 'r') as fhandle:
            data_splits = json.load(fhandle)

        return data_splits['train'], data_splits['validate'], data_splits['test']

    def get_train_generator(self):
        """return a training data generator
        """
        return utils_train.keras_generator_mag(
            self.train_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )



    def get_validation_generator(self):
        """return a validation data generator
        """
        return utils_train.keras_generator_mag(
            self.validation_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )

    def get_test_generator(self):
        """return a test data generator
        """
        return utils_train.keras_generator_mag(
            self.test_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )


def load_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data


def create_data_splits(path_to_metadata_file, exper_dir):

    metadata = load_data(path_to_metadata_file)

    utils.create_data_split(metadata,
                           os.path.join(exper_dir, 'data_splits.json'))


def train(model, model_save_path, data_splits_file, batch_size, active_str, muxrate):

    #data_path = utils.data_path_multif0()
    data_path = config.data_save_folder

    input_patch_size = (360, 50)
    data_splits_path = os.path.join(config.data_save_folder, data_splits_file)

    ## DATA MESS SETUP

    dat = Data(
        data_splits_path, data_path, input_patch_size,
        batch_size, active_str, muxrate
    )

    # instantiate train and validation generators

    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    model.compile(
        loss=utils_train.bkld,
        metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )

    print(model.summary(line_length=80))

    # hopefully fit model
    '''
    history = model.fit_generator(
        train_generator, config.SAMPLES_PER_EPOCH, epochs=config.NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=config.NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_save_path, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=25, verbose=1)
        ]
    )
    '''

    model.load_weights(model_save_path)
    history = ''

    return model, history, dat

def get_single_test_prediction_phase_free(model, npy_file=None, audio_file=None):
    """Generate output from a model given an input numpy file
    """
    if npy_file is not None:

        input_hcqt = np.load(npy_file, allow_pickle=True).item()['dphase/mag'][0]

    elif audio_file is not None:
        # should not be the case
        pump = utils.create_pump_object()
        features = utils.compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]


    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []
    # we need two inputs
    for t in t_slices:
        p = model.predict(np.transpose(input_hcqt[:, :, t:t+5000, :], (0, 1, 3, 2)))[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt


def get_best_thresh(dat, model):
    """Use validation set to get the best threshold value
    """

    # get files for this test set
    validation_files = dat.validation_files
    test_set_path = utils_train.test_path()

    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}
    for npy_file, _ in validation_files:

        fname_base = os.path.basename(npy_file).replace('_input.npy', '.csv')

        label_file = os.path.join(
                test_set_path, fname_base)

        print(label_file)

        # generate prediction on numpy file
        predicted_output, input_hcqt = get_single_test_prediction_phase_free(model=model, npy_file=npy_file)

        # load ground truth labels
        ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(label_file)
        #ref_times, ref_freqs = load_broken_mf0(label_file)

        for thresh in thresh_vals:
            # get multif0 output from prediction
            est_times, est_freqs = \
                utils_train.pitch_activations_to_mf0(predicted_output, thresh)

            # get multif0 metrics and append
            scores = mir_eval.multipitch.evaluate(
                ref_times, ref_freqs, est_times, est_freqs)
            thresh_scores[thresh].append(scores['Accuracy'])

    avg_thresh = [np.mean(thresh_scores[t]) for t in thresh_vals]
    best_thresh = thresh_vals[np.argmax(avg_thresh)]
    print("Best Threshold is {}".format(best_thresh))
    print("Best validation accuracy is {}".format(np.max(avg_thresh)))
    print("Validation accuracy at 0.5 is {}".format(np.mean(thresh_scores[0.5])))

    return best_thresh

def score_on_test_set(model, save_path, dat, thresh=0.5):
    """score a model on all files in a named test set
    """

    # get files for this test set
    test_set_path = utils_train.test_path()
    print('test set path {}'.format(test_set_path))

    test_npy_files = dat.test_files

    all_scores = []
    for npy_file in sorted(test_npy_files):
        npy_file = npy_file[0]
        print(npy_file)
        # get input npy file and ground truth label pair
        fname_base = os.path.basename(npy_file).replace('_input.npy', '.csv')
        print(fname_base)
        label_file = os.path.join(
                test_set_path, fname_base)

        print(label_file)

        # generate prediction on numpy file
        predicted_output, input_hcqt = get_single_test_prediction_phase_free(model, npy_file)


        # save prediction
        np.save(
            os.path.join(
                save_path,
                "{}_prediction.npy".format(fname_base)
            ),
            predicted_output.astype(np.float32)
        )

        # get multif0 output from prediction
        est_times, est_freqs = utils_train.pitch_activations_to_mf0(
            predicted_output, thresh
        )

        # save multif0 output
        utils_train.save_multif0_output(
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
            ref_times, ref_freqs = utils_train.load_broken_mf0(label_file)

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



def run_evaluation(exper_dir, save_key, history, dat, model):

    (save_path, _, plot_save_path,
     model_scores_path, _, _
     ) = utils_train.get_paths(exper_dir, save_key)

    ## Results plots
    #print("plotting results...")
    #utils_train.plot_metrics_epochs(history, plot_save_path)

    ## Evaluate
    print("getting model metrics...")
    utils_train.get_model_metrics(dat, model, model_scores_path)

    print("getting best threshold...")
    thresh = get_best_thresh(dat, model)


    print("scoring multif0 metrics on test sets...")
    utils_train.score_on_test_set(model, save_path, dat, thresh)


def experiment(save_key, model, data_splits_file, batch_size, active_str, muxrate):
    """
    This should be common code for all experiments
    """

    exper_dir = config.exper_output

    (save_path, _, plot_save_path,
     model_scores_path, _, _
     ) = utils_train.get_paths(exper_dir, save_key)


    model_save_path = '/scratch/hc2945/data/models/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model_save_path = os.path.join(model_save_path, "{}.pkl".format(save_key))

    model, history, dat = train(model, model_save_path, data_splits_file,
                                batch_size, active_str, muxrate)


    run_evaluation(exper_dir, save_key, history, dat, model)
    print("Done! Results saved to {}".format(save_path))



def main(args):

    batch_size = 32
    active_str = 100
    muxrate = 32

    save_key = args.save_key
    data_splits_file = args.data_splits_file

    model = models.build_model3_mag()

    experiment(save_key, model, data_splits_file, batch_size, active_str, muxrate)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model3 (Late/Deep) model without the phase, experiment 7.")

    parser.add_argument("--save_key",
                        dest='save_key',
                        type=str,
                        help="String to save model-related data.")

    parser.add_argument("--data_splits_file",
                        dest='data_splits_file',
                        type=str,
                        help="Filename of the data splits file to use in the experiment.")


    main(parser.parse_args())
