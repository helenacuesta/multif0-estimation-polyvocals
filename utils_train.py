import os
import glob
import json
import csv
import ast

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import utils

import pescador
import mir_eval

import keras.backend as K




''' TRAINING UTIL FUNCTIONS
    Some of the functions in this file are taken/adapted from deepsalience.
'''

RANDOM_STATE = 42


def patch_size():
    """Patch size used by all models for training
    """
    return (360, 50)


def experiment_output_path():
    return "/scratch/hc2945/data/experiment_output"


def data_path_multif0():
    """Data path for complete mulif0 data
    """
    return "/scratch/hc2945/data/audiomixtures"

def track_id_list():
    """List of tracks of the datasets
    """
    metadata_path = '/scratch/hc2945/data/audiomixtures/mtracks_info.json'

    data = utils.load_json_data(metadata_path)

    mtracks = list(
        data.keys()
    )

    return mtracks

def keras_loss():
    """Loss function used by all models
    """
    return bkld


def keras_metrics():
    """Metrics used by all models
    """
    return ['mse', soft_binary_accuracy]


def bkld(y_true, y_pred):
    """Brian's KL Divergence implementation
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)


def soft_binary_accuracy(y_true, y_pred):
    """Binary accuracy that works when inputs are probabilities
    """
    return K.mean(K.mean(
        K.equal(K.round(y_true), K.round(y_pred)), axis=-1), axis=-1)


def keras_generator(data_list, input_patch_size, batch_size=16, active_str=200, muxrate=20):
    """Generator to be passed to a keras model
    """
    streams = []
    for fpath_in, fpath_out in data_list:

        print("Data list shape is {}".format(len(data_list)))

        streams.append(
            pescador.Streamer(
                patch_generator, fpath_in, fpath_out,
                input_patch_size=input_patch_size
            )
        )

    stream_mux = pescador.StochasticMux(streams, active_str, rate=muxrate, mode='with_replacement', random_state=RANDOM_STATE)

    batch_generator = pescador.buffer_stream(stream_mux, batch_size)

    for batch in batch_generator:
        print("\n Batch length: ".format(len(batch['X1'])))
        yield [batch['X1'], batch['X2']], batch['Y']

def keras_generator_mag(data_list, input_patch_size, batch_size=16, active_str=200, muxrate=20):
    """Generator to be passed to a keras model
    """
    streams = []
    for fpath_in, fpath_out in data_list:

        print("Data list shape is {}".format(len(data_list)))

        streams.append(
            pescador.Streamer(
                patch_generator_mag, fpath_in, fpath_out,
                input_patch_size=input_patch_size
            )
        )

    stream_mux = pescador.StochasticMux(streams, active_str, rate=muxrate, mode='with_replacement', random_state=RANDOM_STATE)

    batch_generator = pescador.buffer_stream(stream_mux, batch_size)

    for batch in batch_generator:
        print("\n Batch length: ".format(len(batch['X1'])))
        yield batch['X1'], batch['Y']



def grab_patch_output(f, t, n_f, n_t, y_data):
    """Get a time-frequency patch from an output file
    """
    return y_data[f: f + n_f, t: t + n_t][np.newaxis, :, :]


def grab_patch_input(f, t, n_f, n_t, x_data_1, x_data_2):
    """Get a time-frequency patch from an input file
    """
    return np.transpose(
        x_data_1[:, f: f + n_f, t: t + n_t], (1, 2, 0)
    )[np.newaxis, :, :, :], np.transpose(
            x_data_2[:, f: f + n_f, t: t + n_t], (1, 2, 0)
        )[np.newaxis, :, :, :]

def grab_patch_input_mag(f, t, n_f, n_t, x_data_1):
    """Get a time-frequency patch from an input file
    """
    return np.transpose(
        x_data_1[:, f: f + n_f, t: t + n_t], (1, 2, 0)
    )[np.newaxis, :, :, :]


def patch_generator(fpath_in, fpath_out, input_patch_size):
    """Generator that yields an infinite number of patches
       for a single input, output pair
    """
    try:
        data_in_1 = np.load(fpath_in, allow_pickle=True).item()['dphase/mag'][0]
        data_in_2 = np.load(fpath_in, allow_pickle=True).item()['dphase/dphase'][0]
        data_out = np.load(fpath_out, allow_pickle=True)

        data_in_1 = np.transpose(data_in_1, (2, 1, 0))
        data_in_2 = np.transpose(data_in_2, (2, 1, 0))


        _, _, n_times = data_in_1.shape
        n_f, n_t = input_patch_size

        t_vals = np.arange(0, n_times - n_t)
        np.random.shuffle(t_vals)

        for t in t_vals:
            f = 0
            #t = np.random.randint(0, n_times - n_t)

            x1, x2 = grab_patch_input(
                f, t, n_f, n_t, data_in_1, data_in_2
            )
            y = grab_patch_output(
                f, t, n_f, n_t, data_out
            )
            #print(x1.shape, x2.shape, y.shape)
            yield dict(X1=x1[0], X2=x2[0], Y=y[0])
    except:
        pass

def patch_generator_mag(fpath_in, fpath_out, input_patch_size):
    """Generator that yields an infinite number of patches
       for a single input, output pair
    """
    try:
        data_in_1 = np.load(fpath_in, allow_pickle=True).item()['dphase/mag'][0]
        data_out = np.load(fpath_out, allow_pickle=True)

        data_in_1 = np.transpose(data_in_1, (2, 1, 0))

        _, _, n_times = data_in_1.shape
        n_f, n_t = input_patch_size

        t_vals = np.arange(0, n_times - n_t)
        np.random.shuffle(t_vals)

        for t in t_vals:
            f = 0
            #t = np.random.randint(0, n_times - n_t)

            x1 = grab_patch_input_mag(
                f, t, n_f, n_t, data_in_1)

            y = grab_patch_output(
                f, t, n_f, n_t, data_out
            )
            #print(x1.shape, x2.shape, y.shape)
            yield dict(X1=x1[0], Y=y[0])
    except:
        pass


def get_paths(save_dir, save_key):

    save_path = os.path.join(save_dir, save_key)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_save_path = os.path.join(save_path, "{}.pkl".format(save_key))
    plot_save_path = os.path.join(save_path, "{}_loss.pdf".format(save_key))
    model_scores_path = os.path.join(
        save_path, "{}_model_scores.csv".format(save_key))
    scores_path = os.path.join(save_path, "{}_scores.csv".format(save_key))
    score_summary_path = os.path.join(
        save_path, "{}_score_summary.csv".format(save_key))
    return (save_path, model_save_path, plot_save_path,
            model_scores_path, scores_path, score_summary_path)


def get_file_paths(mtrack_list, data_path):
    """Get the absolute paths to input/output pairs for
       a list of multitracks given a data path
    """
    file_paths = []
    for track_id in mtrack_list:
        input_path = glob.glob(
            os.path.join(data_path, 'inputs', "{}*_input.npy".format(track_id[:-4]))
        )

        output_path = glob.glob(
            os.path.join(
                data_path, 'outputs', "{}*_output.npy".format(track_id[:-4])
            )
        )

        if len(input_path) == 1 and len(output_path) == 1:
            input_path = input_path[0]
            output_path = output_path[0]
            file_paths.append((input_path, output_path))

    return file_paths

def plot_metrics_epochs(history, plot_save_path):
    """create and save plot of loss and metrics across epochs
    """
    plt.figure(figsize=(15, 15))

    plt.subplot(3, 1, 1)
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('mean squared error')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.subplot(3, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.subplot(3, 1, 3)
    plt.plot(history.history['soft_binary_accuracy'])
    plt.plot(history.history['val_soft_binary_accuracy'])
    plt.title('soft_binary_accuracy')
    plt.ylabel('soft_binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')

    plt.savefig(plot_save_path, format='pdf')
    plt.close()


def create_data_split(mtrack_dict, output_path):

    mtracks = mtrack_dict.keys()

    all_tracks = [
        m for m in mtracks
    ]
    Ntracks = len(all_tracks)


    train_perc = 0.75
    validation_perc = 0.1
    test_perc = 1 - train_perc - validation_perc

    # consider doing the training taking into account the songs
    # maybe leaving one song out for evaluation

    mtracks_randomized = np.random.permutation(all_tracks)

    train_set = mtracks_randomized[:int(train_perc * Ntracks)]
    validation_set = mtracks_randomized[int(train_perc * Ntracks):int(train_perc * Ntracks) + int(validation_perc * Ntracks)]
    test_set = mtracks_randomized[int(train_perc * Ntracks) + int(validation_perc * Ntracks):]

    data_splits = {
        'train': list(train_set),
        'validate': list(validation_set),
        'test': list(test_set)
    }

    with open(output_path, 'w') as fhandle:
        fhandle.write(json.dumps(data_splits, indent=2))

def get_model_metrics(data_object, model, model_scores_path):
    """Get model loss and metrics on train, validation and test generators
    """
    train_generator = data_object.get_train_generator()
    validation_generator = data_object.get_validation_generator()
    test_generator = data_object.get_test_generator()

    train_eval = model.evaluate_generator(
        train_generator, 1000, max_q_size=10
    )
    valid_eval = model.evaluate_generator(
        validation_generator, 1000, max_q_size=10
    )
    test_eval = model.evaluate_generator(
        test_generator, 1000, max_q_size=10
    )

    df = pd.DataFrame(
        [train_eval, valid_eval, test_eval],
        index=['train', 'validation', 'test']
    )
    print(df)
    df.to_csv(model_scores_path)

def get_single_test_prediction_phase_free(model, npy_file=None, audio_file=None):
    """Generate output from a model given an input numpy file
    """
    if npy_file is not None:

        input_hcqt = np.load(npy_file, allow_pickle=True).item()['dphase/mag'][0]
        input_dphase = np.load(npy_file, allow_pickle=True).item()['dphase/dphase'][0]

    elif audio_file is not None:
        # should not be the case
        pump = utils.create_pump_object()
        features = utils.compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]
        input_dphase = features['dphase/dphase'][0]

        # replace phase info by zeros
        dim_phase = input_dphase.shape
        input_dphase = np.zeros(dim_phase)
        print("     >> Phase replaced by zeros!")

    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []
    # we need two inputs
    for t in t_slices:
        p = model.predict([np.transpose(input_hcqt[:, :, t:t+5000, :], (0, 1, 3, 2)),
                           np.transpose(input_dphase[:, :, t:t+5000, :], (0, 1, 3, 2))]
                          )[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt, input_dphase


def get_single_test_prediction(model, npy_file=None, audio_file=None):
    """Generate output from a model given an input numpy file
    """
    if npy_file is not None:

        input_hcqt = np.load(npy_file, allow_pickle=True).item()['dphase/mag'][0]
        input_dphase = np.load(npy_file, allow_pickle=True).item()['dphase/dphase'][0]

    elif audio_file is not None:
        # should not be the case
        pump = utils.create_pump_object()
        features = utils.compute_pump_features(pump, audio_file)
        input_hcqt = features['dphase/mag'][0]
        input_dphase = features['dphase/dphase'][0]

    else:
        raise ValueError("one of npy_file or audio_file must be specified")

    input_hcqt = input_hcqt.transpose(1, 2, 0)[np.newaxis, :, :, :]
    input_dphase = input_dphase.transpose(1, 2, 0)[np.newaxis, :, :, :]

    n_t = input_hcqt.shape[2]
    t_slices = list(np.arange(0, n_t, 5000))
    output_list = []
    # we need two inputs
    for t in t_slices:
        p = model.predict([np.transpose(input_hcqt[:, :, t:t+5000, :], (0, 1, 3, 2)),
                           np.transpose(input_dphase[:, :, t:t+5000, :], (0, 1, 3, 2))]
                          )[0, :, :]

        output_list.append(p)

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt, input_dphase

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

def test_path():
    """top level path for test data
    """
    return '/scratch/hc2945/data/test_data'

def get_best_thresh(dat, model):
    """Use validation set to get the best threshold value
    """

    # get files for this test set
    validation_files = dat.validation_files
    test_set_path = test_path()

    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}
    for npy_file, _ in validation_files:

        fname_base = os.path.basename(npy_file).replace('_input.npy', '.csv')

        label_file = os.path.join(
                test_set_path, fname_base)

        print(label_file)

        # generate prediction on numpy file
        predicted_output, input_hcqt, input_dph = \
            get_single_test_prediction(model=model, npy_file=npy_file)

        # load ground truth labels
        ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(label_file)
        #ref_times, ref_freqs = load_broken_mf0(label_file)

        for thresh in thresh_vals:
            # get multif0 output from prediction
            est_times, est_freqs = \
                pitch_activations_to_mf0(predicted_output, thresh)

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

def save_multif0_output(times, freqs, output_path):
    """save multif0 output to a csv file
    """
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter='\t')
        for t, f in zip(times, freqs):
            row = [t]
            row.extend(f)
            csv_writer.writerow(row)

def score_on_test_set(model, save_path, dat, thresh=0.5):
    """score a model on all files in a named test set
    """

    # get files for this test set
    test_set_path = test_path()
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
