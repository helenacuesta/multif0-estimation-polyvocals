import os
import json

import numpy as np
import pandas as pd

import pumpp
import jams
import librosa
import mir_eval
import muda


from scipy.ndimage import filters


'''General util functions
   Some of the functions in this file are taken/adapted from deepsalience.
'''


def shift_annotations(jams_path, jams_fname, audio_path, audio_fname):

    '''
    Use the IRConvolution deformer to shift F0 annotations according to
    the estimated group delay introduced by impulse response
    '''

    ir_muda = muda.deformers.IRConvolution(ir_files='./ir/IR_greathall.wav', n_fft=2048, rolloff_value=-24)

    # make sure the duration field in the jams file is not null
    jm = jams.load(os.path.join(jams_path, jams_fname))
    jm.annotations[0].duration = jm.file_metadata.duration
    jm.save(os.path.join(jams_path, jams_fname))

    # load jam and associated audio
    jam = muda.load_jam_audio(os.path.join(jams_path, jams_fname), os.path.join(audio_path, audio_fname))

    for s in ir_muda.states(jam):
        ir_muda.deform_times(jam.annotations[0], s)

    # store deformed annotations in the reverb folder
    jam.save(os.path.join(jams_path, 'reverb', jams_fname))


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


def pyin_to_unvoiced(pyin_path, pyin_fname, audio_path, audio_fname, fs=22050.0):

    '''This function takes a CSV file with smoothedpitchtrack info from pYIN
    and adds zeros in the unvoiced frames.
    '''

    x, fs = librosa.core.load(os.path.join(audio_path, audio_fname), sr=fs)

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
        n_octaves * 12 * over_sample, f_min, bins_per_octave=bins_per_octave)
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


def save_data(save_path, input_path, output_path, prefix, X, Y, f, t):

    i_path = os.path.join(save_path, 'inputs')
    o_path = os.path.join(save_path, 'outputs')

    if not os.path.exists(i_path):
        os.mkdir(i_path)
    if not os.path.exists(o_path):
        os.mkdir(o_path)

    if not os.path.exists(input_path):

        np.save(input_path, X, allow_pickle=True)
        np.save(output_path, Y, allow_pickle=True)
        print("    Saved inputs and targets targets for {} to {}".format(prefix, save_path))

    else:
        np.save(output_path, Y, allow_pickle=True)
        print("    Saved only targets for {} to {}".format(prefix, save_path))

def get_all_pitch_annotations(mtrack):
    '''Load annotations
    '''

    annot_times = []
    annot_freqs = []

    for stem in mtrack['annot_files']:

        '''Load annotations for each singer in the mixture
        '''
        d = jams.load(os.path.join(mtrack['annot_folder'], stem))
        data = np.array(d.annotations[0].data)[:, [0, 2]]


        times = data[:, 0]
        freqs = []
        for d in data[:, 1]:
            freqs.append(d['frequency'])
        freqs = np.array(freqs)

        '''
        times = data[:, 0]
        freqs = data[:, 1]
        '''

        if data is not None:
            idx_to_use = np.where(freqs > 0)[0]
            times = times[idx_to_use]
            freqs = freqs[idx_to_use]

            annot_times.append(times)
            annot_freqs.append(freqs)
        else:
            print('Data not available for {}.'.format(mtrack))
            continue

    # putting all annotations together
    if len(annot_times) > 0:
        annot_times = np.concatenate(annot_times)
        annot_freqs = np.concatenate(annot_freqs)

        return annot_times, annot_freqs

    else:
        return None, None

def create_annotation_target(freq_grid, time_grid, annotation_times, annotation_freqs):
    """Create the binary annotation target labels with Gaussian blur
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

    prefix = "{}".format(mtrack['filename'].split('.')[0])

    input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
    output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))

    '''
    if 'reverb' in mtrack['audiopath']:

        input_path = os.path.join(save_dir, 'inputs', "rev_{}_input.npy".format(prefix))
        output_path = os.path.join(save_dir, 'outputs', "rev_{}_output.npy".format(prefix))
    else:
        input_path = os.path.join(save_dir, 'inputs', "{}_input.npy".format(prefix))
        output_path = os.path.join(save_dir, 'outputs', "{}_output.npy".format(prefix))
    '''


    if os.path.exists(input_path) and os.path.exists(output_path):
        print("    > already done!")
        return

    if 'rev_' in prefix:
        multif0_mix_path = os.path.join(
            mtrack['audiopath'], mtrack['filename'][4:]
        )

    else:
        multif0_mix_path = os.path.join(
            mtrack['audiopath'], mtrack['filename']
        )


    if os.path.exists(multif0_mix_path):

        times, freqs = get_all_pitch_annotations(
            mtrack)
    else:
        print("{} audio file does NOT exist".format(mtrack))
        return

    if times is not None:

        X, Y, f, t = get_input_output_pairs_pump(
            multif0_mix_path, times, freqs)

        save_data(save_dir, input_path, output_path, prefix, X, Y, f, t)

    else:
        print("    {} No multif0 data".format(mtrack['filename']))

def compute_features_mtrack(mtrack, save_dir, wavmixes_path, idx):

    print("Processing {}...".format(mtrack['filename']))

    compute_multif0_complete(mtrack, save_dir, wavmixes_path)

def create_data_split(mtrack_dict, output_path):

    mtracks = mtrack_dict.keys()

    all_tracks = [
        m for m in mtracks
    ]

    '''
    for m in mtracks:
        if 'reverb' in mtrack_dict[m]['audiopath']:
            all_tracks.append('rev_' + m)
        else:
            all_tracks.append(m)
    '''


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



