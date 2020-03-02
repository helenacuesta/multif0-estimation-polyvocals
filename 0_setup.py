'''
This script creates vocal quartets using the three target datasets: CSD, ECS, DCS, which are augmented
using pitch-shifting from MUDA.

The code is specifically written for these datasets, but can be easily adapted to other multitracks.
'''

import sox
import os

import pandas as pd
import librosa
import soundfile

import scipy

import config
import utils


def combine_audio_files(piece, filenames, output_fname, mode, reverb):

    #### Choral Singing Dataset
    if mode == 'csd':

        cmb = sox.Combiner()
        cmb.convert(samplerate=22050)
        cmb.build(
            [config.csd_folder + filenames[0], config.csd_folder + filenames[1],
             config.csd_folder + filenames[2], config.csd_folder + filenames[3]],
            os.path.join(config.audio_save_folder, output_fname), 'mix')  # , 'mix', input_volumes=[0.6, 0.3, 0.3, 0.3])

        # if the reverb option is active, this creates the reverb audio files using an IR from Isophonics
        if reverb:
            y_ir, sr_ir = librosa.load('./ir/IR_greathall.wav', sr=22050)
            y_sig, sr_sig = librosa.load(os.path.join(config.audio_save_folder, output_fname), sr=22050)
            y_rev = scipy.signal.convolve(y_sig, y_ir, mode="full")
            soundfile.write(os.path.join(config.audio_save_folder, 'reverb', output_fname), y_rev, samplerate=22050)


    # TODO

    #### ESMUC Choral Set
    elif mode == 'ecs':


        filenames = [
            '{}S{}.wav'.format(piece, singer_idxs[0]),
            '{}A{}.wav'.format(piece, singer_idxs[1]),
            '{}T{}.wav'.format(piece, singer_idxs[2]),
            '{}B{}.wav'.format(piece, singer_idxs[3]),
        ]

        cmb = sox.Combiner()
        cmb.convert(samplerate=22050)
        cmb.build(
            [config.ecs_folder + filenames[0], config.ecs_folder + filenames[1],
             config.ecs_folder + filenames[2], config.ecs_folder + filenames[3]],
            os.path.join(config.audio_save_folder, output_fname), 'mix', input_volumes=[0.55, 0.3, 0.4, 0.3]
        )

        tf = sox.Transformer()
        tf.reverb(reverberance=100, pre_delay=30, high_freq_damping=80, wet_gain=4)
        tf.build(os.path.join(config.audio_save_folder, output_fname),
                 os.path.join(config.audio_save_folder, 'rev_' + output_fname))

    #### Dagstuhl ChoirSet

    elif mode == 'dcs':

        cmb = sox.Combiner()
        cmb.convert(samplerate=22050)
        cmb.build(
            [config.dcs_folder + filenames[0], config.dcs_folder + filenames[1],
             config.dcs_folder + filenames[2], config.dcs_folder + filenames[3]],

            os.path.join(config.audio_save_folder, output_fname), 'mix')

        # if the reverb option is active, this creates the reverb audio files using an IR from Isophonics
        if reverb:
            y_ir, sr_ir = librosa.load('./ir/IR_greathall.wav', sr=22050)
            y_sig, sr_sig = librosa.load(os.path.join(config.audio_save_folder, output_fname), sr=22050)
            y_rev = scipy.signal.convolve(y_sig, y_ir, mode="full")
            soundfile.write(os.path.join(config.audio_save_folder, 'reverb', output_fname), y_rev, samplerate=22050)



def create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder):

    diction[audiofname] = dict()
    diction[audiofname]['audiopath'] = audiopath
    diction[audiofname]['annot_files'] = annot_files
    diction[audiofname]['annot_folder'] = annot_folder

    return diction


def create_full_dataset_mixes(dataset, mixes_wavpath, reverb=True, exclude_dataset=None, compute_audio_mix=True, compute_metadata=True):

    mtracks = dict()

    #TODO: figure out how to handle this
    if exclude_dataset is not None:
        print("Sth needs to be done here!")


    dataset_ids = ['CSD', 'ECS', 'DCS', 'BC', 'BSQ']


    # ------------ Process Choral Singing Dataset ------------ #

    D1 = dataset_ids[0] # CSD
    mode = 'csd'


    for song in dataset['CSD']['songs']:

        for combo in dataset['CSD']['combos']:

            filenames = [
                '{}_soprano_{}.wav'.format(song, combo[0]),
                '{}_alto_{}.wav'.format(song, combo[1]),
                '{}_tenor_{}.wav'.format(song, combo[2]),
                '{}_bass_{}.wav'.format(song, combo[3]),
            ]

            output_fname = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(song, filenames, output_fname, mode, reverb)

            if compute_metadata:
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_soprano_{}.jams'.format(song, combo[0]), '{}_alto_{}.jams'.format(song, combo[1]),
                    '{}_tenor_{}.jams'.format(song, combo[2]), '{}_bass_{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.csd_folder)

                if reverb:
                    print("Reverb annotations not created for the reverb versions. Working on annotation shift.")

        print("Mixtures for {} have been created.".format(song))

    # ------------ Process ESMUC ChoralSet ------------ #

    # In this dataset we do not have a uniform number of singers per part.
    # We have: 4 sopranos, 3 altos, 3 tenors and 2 basses. Singer combinations need to be done accordingly.
    '''
    for song in dataset['CSD']['songs']:

        for combo in dataset['CSD']['combos']:

            filenames = [
                '{}_soprano_{}.wav'.format(song, combo[0]),
                '{}_alto_{}.wav'.format(song, combo[1]),
                '{}_tenor_{}.wav'.format(song, combo[2]),
                '{}_bass_{}.wav'.format(song, combo[3]),
            ]

            output_fname = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(song, filenames, output_fname, mode, reverb)

            if compute_metadata:
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_soprano_{}.jams'.format(song, combo[0]), '{}_alto_{}.jams'.format(song, combo[1]),
                    '{}_tenor_{}.jams'.format(song, combo[2]), '{}_bass_{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.csd_folder)

                if reverb:
                    print("Reverb annotations not created for the reverb versions. Working on annotation shift.")
    '''
    mode = 'ecs'

    # Der Greis
    for song in dataset['ECS']['DG_songs']:
        for combo in dataset['ECS']['DG_combos']:

            filenames = [
                "{}_S{}.wav".format(song, combo[0]),
                "{}_A{}.wav".format(song, combo[1]),
                "{}_T{}.wav".format(song, combo[2]),
                "{}_B{}.wav".format(song, combo[3])
            ]

            output_fname = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(song, filenames, output_fname, mode, reverb)

            if compute_metadata:
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_S{}.jams'.format(song, combo[0]), '{}_A{}.jams'.format(song, combo[1]),
                    '{}_T{}.jams'.format(song, combo[2]), '{}_B{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.ecs_folder)

                if reverb:
                    print("Reverb annotations not created for the reverb versions. Working on annotation shift.")

        print('{} quartets mixed and exported'.format(song))


    # Die Himmel

    for song in dataset['ECS']['DH_songs']:
        for combo in dataset['ECS']['DG_combos']:

            filenames = [
                "{}_{}.wav".format(song, dataset['ECS']['DH_singers'][combo[0]]),
                "{}_{}.wav".format(song, combo[1]),
                "{}_{}.wav".format(song, combo[2]),
                "{}_{}.wav".format(song, combo[3])
            ]

            output_fname = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(song, filenames, output_fname, mode, reverb)

            if compute_metadata:
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_S{}.jams'.format(song, combo[0]), '{}_A{}.jams'.format(song, combo[1]),
                    '{}_T{}.jams'.format(song, combo[2]), '{}_B{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.ecs_folder)

                if reverb:
                    print("Reverb annotations not created for the reverb versions. Working on annotation shift.")

        print('{} quartets mixed and exported'.format(song))

    # ------------ Process Dagstuhl ChoirSet ------------ #

    mode = 'dcs'

    # Full Choir setting
    for song in dataset['DCS']['FC_songs']:

        filenames = [
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][0]),
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][1]),
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][2]),
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][3])
        ]

        # no combos here, there are only four singers per song
        output_fname = "{}_1_2_2_2.wav".format(song)

        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, output_fname)):
            combine_audio_files(song, filenames, output_fname, mode=mode, reverb=reverb)

        if compute_metadata:
            annotation_files = [
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][0]),
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][1]),
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][2]),
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.dcs_folder)

            if reverb:
                print("Reverb annotations not created for the reverb versions. Working on annotation shift.")

        print('{} quartets mixed and exported'.format(song))

    # Quartet A setting
    for song in dataset['DCS']['QA_songs']:

        filenames = [
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][0]),
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][1]),
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][2]),
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][3])
        ]

        # no combos here, there are only four singers per song
        output_fname = "{}_2_1_1_1.wav".format(song)

        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, output_fname)):
            combine_audio_files(song, filenames, output_fname, mode=mode, reverb=reverb)

        if compute_metadata:
            annotation_files = [
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][0]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][1]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][2]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.dcs_folder)

            if reverb:
                print("Reverb annotations not created for the reverb versions. Working on annotation shift.")

        print('{} quartets mixed and exported'.format(song))

    # Quartet B setting
    for song in dataset['DCS']['QB_songs']:

        filenames = [
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][0]),
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][1]),
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][2]),
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][3])
        ]

        # no combos here, there are only four singers per song
        output_fname = "{}_1_2_2_2.wav".format(song)

        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, output_fname)):
            combine_audio_files(song, filenames, output_fname, mode=mode, reverb=reverb)

        if compute_metadata:
            annotation_files = [
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][0]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][1]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][2]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, output_fname, annotation_files, config.dcs_folder)

            if reverb:
                print("Reverb annotations not created for the reverb versions. Working on annotation shift.")

        print('{} quartets mixed and exported'.format(song))


    # Barbershop quartets
    bq = pd.read_csv('BQ_info.csv').values

    dict_bq = dict()

    voices = [bq[0, 1], bq[0, 2], bq[0, 3], bq[0, 4]]
    endname = bq[0, 7]

    idx = 0
    for song in bq[:, 0]:
        idx += 1
        for parts in bq[:, 6]:
            P = int(parts) + 1

            for i in range(1, P):
                basename = "{}_{}_part{}".format(idx, song, i)
                dict_bq[basename + '_mix.wav'] = dict()
                dict_bq[basename + '_mix.wav']['audiopath'] = bq_audio_folder
                dict_bq[basename + '_mix.wav']['annot_folder'] = bq_audio_folder
                dict_bq[basename + '_mix.wav']['annot_files'] = []

                for voice in voices:
                    fname = "{}_{}{}".format(basename, voice, endname)
                    dict_bq[basename + '_mix.wav']['annot_files'].append(fname)

    # Bach Chorales

    bc = pd.read_csv('BC_info.csv').values

    dict_bc = dict()

    voices = [bc[0, 1], bc[0, 2], bc[0, 3], bc[0, 4]]
    endname = bc[0, 7]

    idx = 0
    for song in bc[:, 0]:
        idx += 1
        for parts in bc[:, 6]:
            P = int(parts) + 1

            for i in range(1, P):
                basename = "{}_{}_part{}".format(idx, song, i)
                dict_bc[basename + '_mix.wav'] = dict()
                dict_bc[basename + '_mix.wav']['audiopath'] = bc_audio_folder
                dict_bc[basename + '_mix.wav']['annot_folder'] = bc_audio_folder
                dict_bc[basename + '_mix.wav']['annot_files'] = []

                for voice in voices:
                    fname = "{}_{}{}".format(basename, voice, endname)
                    dict_bc[basename + '_mix.wav']['annot_files'].append(fname)

    # Store the metadata file
    if compute_metadata:
        utils.save_json_data(mtracks, os.path.join(mixes_wavpath, 'mtracks_info.json'))


def main():

    # load the dataset info
    dataset = config.dataset

    # use the dataset information to create audio mixtures and annotations
    create_full_dataset_mixes(dataset, config.audio_save_folder, reverb=True, exclude_dataset=None,
                              compute_audio_mix=True, compute_metadata=True)


if __name__ == '__main__':

    main()

