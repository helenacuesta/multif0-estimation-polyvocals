import os
import glob

import json
import csv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import ast

import pumpp
import jams
import librosa
import pescador
import mir_eval

import keras.backend as K
from scipy.ndimage import filters


'''General util functions
'''

def save_json_data(data, save_path):
    with open(save_path, 'w') as fp:
        json.dump(data, fp)

def load_json_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data

def get_hcqt_params():

    bins_per_octave = 60
    n_octaves = 6
    over_sample = 5
    harmonics = [1, 2, 3, 4, 5]
    sr = 22050
    fmin = 32.7
    hop_length = 256
    return bins_per_octave, n_octaves, harmonics, sr, fmin, hop_length, over_sample

import numpy as np
import pandas as pd
import mir_eval
import librosa

import os

def pyin_to_unvoiced(pyin_path, pyin_fname, audio_path, audio_fname, fs=44100):
    '''This function takes a CSV file with smoothedpitchtrack info from pYIN
    and adds zeros in the unvoiced frames.
    '''
    x, fs = librosa.core.load(os.path.join(audio_path, audio_fname), sr=fs)
    import pdb; pdb.set_trace()
    if pyin_fname.endswith('csv'):
        pyi = pd.read_csv(os.path.join(pyin_path, pyin_fname), header=None).values

    elif pyin_fname.endswith('f0'):
        pyi = np.loadtxt(os.path.join(pyin_path, pyin_fname))

    else:
        print("Wrong annotation file format found.")
        quit()

    hopsize = 256
    l_samples = len(x)
    del x
    time_pyin = mir_eval.melody.constant_hop_timebase(hop=hopsize, end_time=l_samples) / fs

    # times_pyin uses the same hopsize as the original pyin so we can directly compare them
    pyin_new = np.zeros([len(time_pyin), 2])
    _, _, idx_y = np.intersect1d(np.around(pyi[:, 0], decimals=5), np.around(time_pyin, decimals=5), return_indices=True)
    pyin_new[idx_y, 1] = pyi[:, 1]
    pyin_new[:, 0] = time_pyin

    pd.DataFrame(pyin_new).to_csv(os.path.join(pyin_path, 'constant_timebase', pyin_fname), header=None, index=False)


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


def grid_to_bins(grid, start_bin_val, end_bin_val):
    """Compute the bin numbers from a given grid
    """
    bin_centers = (grid[1:] + grid[:-1])/2.0
    bins = np.concatenate([[start_bin_val], bin_centers, [end_bin_val]])
    return bins

def save_data(save_path, prefix, X, Y, f, t):

    input_path = os.path.join(save_path, 'inputs_dph')
    output_path = os.path.join(save_path, 'outputs_dph')

    if not os.path.exists(input_path):
        os.mkdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(os.path.join(input_path, "{}_input_dph.npy".format(prefix))):
        np.save(os.path.join(input_path, "{}_input_dph.npy".format(prefix)), X, allow_pickle=True)
        np.save(os.path.join(output_path, "{}_output_dph.npy".format(prefix)), Y, allow_pickle=True)
        print("    Saved inputs and targets targets for {} to {}".format(prefix, save_path))

    else:
        np.save(os.path.join(output_path, "{}_output_dph.npy".format(prefix)), Y, allow_pickle=True)
        print("    Saved only targets for {} to {}".format(prefix, save_path))

def get_all_pitch_annotations(mtrack):
    '''Load annotations
    '''

    annot_times = []
    annot_freqs = []
    stem_annot_activity = {}
    for stem in mtrack['annot_files']:

        '''
        Annotations are loaded HERE
        '''
        d = jams.load(os.path.join(mtrack['annot_folder'], stem))
        data = np.array(d.annotations[0].data)[:, [0, 2]]

        times = data[:, 0]
        freqs = []
        for d in data[:, 1]:
            freqs.append(d['frequency'])
        freqs = np.array(freqs)


        if data is not None:
            idx_to_use = np.where(freqs > 0)[0]
            times = times[idx_to_use]
            freqs = freqs[idx_to_use]

            annot_times.append(times)
            annot_freqs.append(freqs)
        else:
            print('Data not available for {}.'.format(mtrack))
            continue

    if len(annot_times) > 0:
        annot_times = np.concatenate(annot_times)
        annot_freqs = np.concatenate(annot_freqs)

        return annot_times, annot_freqs, stem_annot_activity
    else:
        return None, None, None, stem_annot_activity

def create_annotation_target(freq_grid, time_grid, annotation_times, annotation_freqs):
    """Create the binary annotation target labels
    """
    time_bins = grid_to_bins(time_grid, 0.0, time_grid[-1])
    freq_bins = grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    annot_time_idx = np.digitize(annotation_times, time_bins) - 1
    annot_freq_idx = np.digitize(annotation_freqs, freq_bins) - 1

    n_freqs = len(freq_grid)
    n_times = len(time_grid)

    idx = annot_time_idx < n_times
    annot_time_idx = annot_time_idx[idx]
    annot_freq_idx = annot_freq_idx[idx]

    idx2 = annot_freq_idx < n_freqs
    annot_time_idx = annot_time_idx[idx2]
    annot_freq_idx = annot_freq_idx[idx2]

    annotation_target = np.zeros((n_freqs, n_times))
    annotation_target[annot_freq_idx, annot_time_idx] = 1


    annotation_target_blur = filters.gaussian_filter1d(
        annotation_target, 1, axis=0, mode='constant'
    )
    if len(annot_freq_idx) > 0:
        min_target = np.min(
            annotation_target_blur[annot_freq_idx, annot_time_idx]
        )
    else:
        min_target = 1.0

    annotation_target_blur = annotation_target_blur / min_target
    annotation_target_blur[annotation_target_blur > 1.0] = 1.0

    return annotation_target_blur


def create_pump_object():

    (bins_per_octave, n_octaves, harmonics,
     sr, f_min, hop_length, over_sample) = get_hcqt_params()

    p_phdif = pumpp.feature.HCQTPhaseDiff(name='dphase', sr=sr, hop_length=hop_length,
                                   fmin=f_min, n_octaves=n_octaves, over_sample=over_sample, harmonics=harmonics, log=True)

    pump = pumpp.Pump(p_phdif)

    return pump

def compute_pump_features(pump, audio_fpath):

    data = pump(audio_f=audio_fpath)

    return data


def get_input_output_pairs_pump(audio_fpath, annot_times, annot_freqs):

    print("    > computing HCQT and Phase Differentials for {}".format(os.path.basename(audio_fpath)))
    pump_hcqt_dph = create_pump_object()
    hcqt = compute_pump_features(pump_hcqt_dph, audio_fpath)


    freq_grid = get_freq_grid()
    time_grid = get_time_grid(len(hcqt['dphase/mag'][0]))

    annot_target = create_annotation_target(
        freq_grid, time_grid, annot_times, annot_freqs)

    return hcqt, annot_target, freq_grid, time_grid

def compute_multif0_complete(mtrack, save_dir, wavmixes_path):

    prefix = "{}_multif0".format(mtrack['filename'].split('.')[0])

    input_path = os.path.join(save_dir, 'inputs_dph', "{}_input_dph.npy".format(prefix))
    output_path = os.path.join(save_dir, 'outputs_dph', "{}_output_dph.npy".format(prefix))

    if os.path.exists(input_path) and os.path.exists(output_path):
        print("    > already done!")
        return

    multif0_mix_path = os.path.join(
        wavmixes_path, mtrack['filename']
    )

    if os.path.exists(multif0_mix_path):

        (times, freqs, stem_annot_activity) = get_all_pitch_annotations(
            mtrack)
    else:
        print("{} does not exist".format(mtrack))
        return

    if times is not None:

        X, Y, f, t = get_input_output_pairs_pump(
            multif0_mix_path, times, freqs)

        save_data(save_dir, prefix, X, Y, f, t)

    else:
        print("    {} No multif0 data".format(mtrack['filename']))

def compute_features_mtrack(mtrack, save_dir, wavmixes_path, idx):

    print(mtrack['filename'])
    compute_multif0_complete(mtrack, save_dir, wavmixes_path)


''' TRAINING UTIL FUNCTIONS
'''

RANDOM_STATE = 42


def patch_size():
    """Patch size used by all models for training
    """
    return (360, 50)


def experiment_output_path():
    return "/scratch/hc2945/multif0/experiment_output"


def data_path_multif0():
    """Data path for complete mulif0 data
    """
    return "/scratch/hc2945/multif0/AudioMixtures"

def track_id_list():
    """List of tracks of the three datasets
    """
    metadata_path = '/scratch/hc2945/multif0/VocalEnsembles/mtracks_info.json'

    data = load_json_data(metadata_path)

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
        streams.append(
            pescador.Streamer(
                patch_generator, fpath_in, fpath_out,
                input_patch_size=input_patch_size
            )
        )

    print(len(streams))

    stream_mux = pescador.StochasticMux(streams, active_str, rate=muxrate, random_state=RANDOM_STATE)

    batch_generator = pescador.buffer_stream(stream_mux, batch_size)

    for batch in batch_generator:
        print(len(batch['X1']))
        yield [batch['X1'], batch['X2']], batch['Y']

def keras_generator_single(data_list, input_patch_size, batch_size=16, active_str=200, muxrate=20):
    """Generator to be passed to a keras model. Specifically for single HCQT input, no phase
    """
    streams = []
    for fpath_in, fpath_out in data_list:
        streams.append(
            pescador.Streamer(
                patch_generator_single, fpath_in, fpath_out,
                input_patch_size=input_patch_size
            )
        )

    stream_mux = pescador.StochasticMux(streams, active_str, rate=muxrate, random_state=RANDOM_STATE)

    batch_generator = pescador.buffer_stream(stream_mux, batch_size)

    for batch in batch_generator:
        yield batch['X'], batch['Y']


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

def grab_patch_input_single(f, t, n_f, n_t, x_data):
    """Get a time-frequency patch from an input file
    """
    return np.transpose(
        x_data[:, f: f + n_f, t: t + n_t], (1, 2, 0)
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
            t = np.random.randint(0, n_times - n_t)

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

def patch_generator_single(fpath_in, fpath_out, input_patch_size):
    """Generator that yields an infinite number of patches
       for a single input, output pair. For single HCQT input
    """
    try:
        data_in = np.load(fpath_in, allow_pickle=True).item()['dphase/mag'][0]
        data_out = np.load(fpath_out, allow_pickle=True)

        data_in = np.transpose(data_in, (2, 1, 0))


        _, _, n_times = data_in.shape
        n_f, n_t = input_patch_size

        t_vals = np.arange(0, n_times - n_t)
        np.random.shuffle(t_vals)

        for t in t_vals:
            f = 0
            t = np.random.randint(0, n_times - n_t)

            x = grab_patch_input_single(
                f, t, n_f, n_t, data_in
            )
            y = grab_patch_output(
                f, t, n_f, n_t, data_out
            )
            yield dict(X=x[0], Y=y[0])
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
            os.path.join(data_path, 'inputs_dph', "{}*_input_dph.npy".format(track_id[:-4]))
        )

        output_path = glob.glob(
            os.path.join(
                data_path, 'outputs_dph', "{}*_output_dph.npy".format(track_id[:-4])
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

    train_perc = 0.8
    validation_perc = 0.1
    test_perc = 1 - train_perc - validation_perc

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


def get_single_test_prediction(model, npy_file=None, audio_file=None):
    """Generate output from a model given an input numpy file
    """
    if npy_file is not None:

        input_hcqt = np.load(npy_file, allow_pickle=True).item()['dphase/mag'][0]
        input_dphase = np.load(npy_file, allow_pickle=True).item()['dphase/dphase'][0]

    elif audio_file is not None:
        # should not be the case
        pump = create_pump_object()
        features = compute_pump_features(pump, audio_file)
        input_hcqt = features.item()['dphase/mag'][0]
        input_dphase = features.item()['dphase/dphase'][0]

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
        '''
        output_list.append(
            model.predict([input_hcqt[:, :, t:t+5000, :], input_dphase[:, :, t:t+5000, :]]
                          )[0, :, :]
        )
        '''

    predicted_output = np.hstack(output_list)
    return predicted_output, input_hcqt, input_dphase

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

def test_path():
    """top level path for test data
    """
    return '/scratch/hc2945/multif0/VocalEnsembles/test_data'

def get_best_thresh(dat, model):
    """Use validation set to get the best threshold value
    """

    # get files for this test set
    validation_files = dat.validation_files
    test_set_path = test_path()

    thresh_vals = np.arange(0.1, 1.0, 0.1)
    thresh_scores = {t: [] for t in thresh_vals}
    for npy_file, _ in validation_files:

        fname_base = os.path.basename(npy_file).replace('_input_dph.npy', '.csv')

        if 'rev_' in fname_base:
            fname_base = fname_base[4:]

        label_file = os.path.join(
                test_set_path, fname_base)

        print(label_file)

        # generate prediction on numpy file
        predicted_output, input_hcqt, input_dph = \
            get_single_test_prediction(model=model, npy_file=npy_file)

        # load ground truth labels
        #ref_times, ref_freqs = \
        #    mir_eval.io.load_ragged_time_series(label_file)
        ref_times, ref_freqs = load_broken_mf0(label_file)

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
        fname_base_gt = os.path.basename(npy_file).replace('_input_dph.npy', '.csv')
        fname_base_gt = 'rev_' + fname_base_gt
        print(fname_base_gt)
        fname_base = fname_base_gt[:-4]
        print(fname_base)
        label_file = os.path.join(
                test_set_path, fname_base_gt)

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
        #ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(label_file)
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



class Data_single(object):
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

        self.train_files = get_file_paths(self.train_set, self.data_path)
        self.validation_files = get_file_paths(
            self.validation_set, self.data_path
        )
        self.test_files = get_file_paths(self.test_set, self.data_path)

        self.batch_size = batch_size
        self.active_str = active_str
        self.muxrate = muxrate

    def load_data_splits(self):
        """Get randomized artist-conditional splits
        """
        with open(self.data_splits_path, 'r') as fhandle:
            data_splits = json.load(fhandle)

        return data_splits['train'], data_splits['validate'], data_splits['test']

    def get_train_generator(self):
        """return a training data generator
        """
        return keras_generator_single(
            self.train_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )



    def get_validation_generator(self):
        """return a validation data generator
        """
        return keras_generator_single(
            self.validation_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )

    def get_test_generator(self):
        """return a test data generator
        """
        return keras_generator_single(
            self.test_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )

