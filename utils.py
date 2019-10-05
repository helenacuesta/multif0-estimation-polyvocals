import os
import glob

import numpy as np
import json

import pumpp
import jams
import librosa
import pescador

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

