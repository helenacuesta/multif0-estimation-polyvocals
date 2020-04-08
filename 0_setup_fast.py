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


def combine_audio_files(params):

    cmb = sox.Combiner()
    cmb.convert(samplerate=22050)
    cmb.build(
        [
            os.path.join(params['audio_folder'], params['filenames'][0]),
            os.path.join(params['audio_folder'], params['filenames'][1]),
            os.path.join(params['audio_folder'], params['filenames'][2]),
            os.path.join(params['audio_folder'], params['filenames'][3])
        ],
        os.path.join(config.audio_save_folder, params['output_fname']), 'mix')  # , 'mix', input_volumes=[0.6, 0.3, 0.3, 0.3])

    # if the reverb option is active, this creates the reverb audio files using an IR from Isophonics
    if params['reverb']:
        y_ir, sr_ir = librosa.load('./ir/IR_greathall.wav', sr=params['sr'])
        y_sig, sr_sig = librosa.load(os.path.join(config.audio_save_folder, params['output_fname']), sr=params['sr'])
        y_rev = scipy.signal.convolve(y_sig, y_ir, mode="full")
        soundfile.write(os.path.join(config.audio_save_folder, 'reverb', params['output_fname']), y_rev, samplerate=params['sr'])


    '''cmb = sox.Combiner()
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
        soundfile.write(os.path.join(config.audio_save_folder, 'reverb', output_fname), y_rev, samplerate=22050)'''




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

    print("Processing Choral Singing Dataset...")

    for song in dataset['CSD']['songs']:
        for combo in dataset['CSD']['combos']:

            params = {}
            params['audio_folder'] = config.csd_folder
            params['annot_folder'] = config.csd_folder
            params['sr'] = 44100
            params['reverb'] = True

            params['filenames'] = [
                '{}_soprano_{}.wav'.format(song, combo[0]),
                '{}_alto_{}.wav'.format(song, combo[1]),
                '{}_tenor_{}.wav'.format(song, combo[2]),
                '{}_bass_{}.wav'.format(song, combo[3]),
            ]

            params['output_fname'] = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            '''
            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(params)
            '''

            if compute_metadata:
                print("Annotations for {}".format(song))
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_soprano_{}.jams'.format(song, combo[0]), '{}_alto_{}.jams'.format(song, combo[1]),
                    '{}_tenor_{}.jams'.format(song, combo[2]), '{}_bass_{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files, params['annot_folder'])
                '''
                if reverb:
                    idx=-1
                    for annot in annotation_files:
                        idx+=1
                        utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])
                '''

        print("Mixtures for {} have been created.".format(song))

    # ------------ Process ESMUC ChoralSet ------------ #

    print("Processing Choral Singing Dataset...")

    # Der Greis
    for song in dataset['ECS']['DG_songs']:
        for combo in dataset['ECS']['DG_combos']:

            params = {}
            params['audio_folder'] = config.ecs_folder
            params['annot_folder'] = config.ecs_folder
            params['sr'] = 22050
            params['reverb'] = True

            params['filenames'] = [
                "{}_S{}.wav".format(song, combo[0]),
                "{}_A{}.wav".format(song, combo[1]),
                "{}_T{}.wav".format(song, combo[2]),
                "{}_B{}.wav".format(song, combo[3])
            ]

            params['output_fname'] = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            '''
            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(params)
            '''

            if compute_metadata:
                print("Annotations for {}".format(song))
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_S{}.jams'.format(song, combo[0]), '{}_A{}.jams'.format(song, combo[1]),
                    '{}_T{}.jams'.format(song, combo[2]), '{}_B{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files, params['annot_folder'])
                '''
                if reverb:
                    idx=-1
                    for annot in annotation_files:
                        idx+=1
                        utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])
                    #print("Reverb annotations not created for the reverb versions. Working on annotation shift.")
                '''
        print('{} quartets mixed and exported'.format(song))


    # Die Himmel

    for song in dataset['ECS']['DH_songs']:
        for combo in dataset['ECS']['DG_combos']:

            params = {}
            params['audio_folder'] = config.ecs_folder
            params['annot_folder'] = config.ecs_folder
            params['sr'] = 22050
            params['reverb'] = True

            params['filenames'] = [
                "{}_{}.wav".format(song, dataset['ECS']['DH_singers'][combo[0]-1]),
                "{}_{}.wav".format(song, dataset['ECS']['DH_singers'][combo[1]-1+5]),
                "{}_{}.wav".format(song, dataset['ECS']['DH_singers'][combo[2]-1+7]),
                "{}_{}.wav".format(song, dataset['ECS']['DH_singers'][combo[3]-1+10])
            ]

            params['output_fname'] = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            '''
            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):

                # create audio mixture and its reverb version if indicated
                combine_audio_files(params)
            '''

            if compute_metadata:
                print("Annotations for {}".format(song))
                # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                annotation_files = [
                    '{}_{}.jams'.format(song, dataset['ECS']['DH_singers'][combo[0]-1]),
                    '{}_{}.jams'.format(song, dataset['ECS']['DH_singers'][combo[1]-1+5]),
                    '{}_{}.jams'.format(song, dataset['ECS']['DH_singers'][combo[2]-1+7]),
                    '{}_{}.jams'.format(song, dataset['ECS']['DH_singers'][combo[3]-1+10])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files, params['annot_folder'])
                '''
                if reverb:
                    idx=-1
                    for annot in annotation_files:
                        idx+=1
                        utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'], params['filenames'][idx])
                '''
        print('{} quartets mixed and exported'.format(song))

        # Seele Christi

        for song in dataset['ECS']['SC_songs']:
            for combo in dataset['ECS']['SC_combos']:

                params = {}
                params['audio_folder'] = config.ecs_folder
                params['annot_folder'] = config.ecs_folder
                params['sr'] = 22050
                params['reverb'] = True

                params['filenames'] = [
                    "{}_S{}.wav".format(song, combo[0]),
                    "{}_A{}.wav".format(song, combo[1]),
                    "{}_T{}.wav".format(song, combo[2]),
                    "{}_B{}.wav".format(song, combo[3])
                ]

                params['output_fname'] = '{}_{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])
                '''
                if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
                    # create audio mixture and its reverb version if indicated
                    combine_audio_files(params)
                
                '''

                if compute_metadata:
                    print("Annotations for {}".format(song))
                    # create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder)
                    annotation_files = [
                        "{}_S{}.jams".format(song, combo[0]),
                        "{}_A{}.jams".format(song, combo[1]),
                        "{}_T{}.jams".format(song, combo[2]),
                        "{}_B{}.jams".format(song, combo[3])
                    ]

                    mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files,
                                                params['annot_folder'])
                    '''
                    if reverb:
                        idx = -1
                        for annot in annotation_files:
                            idx += 1
                            utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'],
                                                    params['filenames'][idx])
                    '''

            print('{} quartets mixed and exported'.format(song))

    # ------------ Process Dagstuhl ChoirSet ------------ #


    # Full Choir setting
    for song in dataset['DCS']['FC_songs']:

        params = {}
        params['audio_folder'] = config.dcs_folder_audio
        params['annot_folder'] = config.dcs_folder_annot
        params['sr'] = 22050
        params['reverb'] = True

        params['filenames'] = [
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][0]),
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][1]),
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][2]),
            "{}_{}.wav".format(song, dataset['DCS']['FC_singers'][3])
        ]

        # no combos here, there are only four singers per song
        params['output_fname'] = "{}_1_2_2_2.wav".format(song)

        '''
        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
            combine_audio_files(params)
        '''

        if compute_metadata:
            print("Annotations for {}".format(song))
            annotation_files = [
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][0]),
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][1]),
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][2]),
                "{}_{}.jams".format(song, dataset['DCS']['FC_singers'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files, params['annot_folder'])
            '''
            if reverb:
                idx = -1
                for annot in annotation_files:
                    idx += 1
                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'],
                                            params['filenames'][idx])
            '''
        print('{} quartets mixed and exported'.format(song))

    # Quartet A setting
    for song in dataset['DCS']['QA_songs']:

        params = {}
        params['audio_folder'] = config.dcs_folder_audio
        params['annot_folder'] = config.dcs_folder_annot
        params['sr'] = 22050
        params['reverb'] = True

        params['filenames'] = [
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][0]),
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][1]),
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][2]),
            "{}_{}.wav".format(song, dataset['DCS']['QA_singers'][3])
        ]

        # no combos here, there are only four singers per song
        params['output_fname'] = "{}_2_1_1_1.wav".format(song)

        '''
        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
            combine_audio_files(params)
        '''

        if compute_metadata:
            print("Annotations for {}".format(song))
            annotation_files = [
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][0]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][1]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][2]),
                "{}_{}.jams".format(song, dataset['DCS']['QA_singers'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files, params['annot_folder'])
            '''
            if reverb:
                idx = -1
                for annot in annotation_files:
                    idx += 1
                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'],
                                            params['filenames'][idx])
            '''
        print('{} quartets mixed and exported'.format(song))

    # Quartet B setting
    for song in dataset['DCS']['QB_songs']:

        params = {}
        params['audio_folder'] = config.dcs_folder_audio
        params['annot_folder'] = config.dcs_folder_annot
        params['sr'] = 22050
        params['reverb'] = True

        params['filenames'] = [
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][0]),
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][1]),
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][2]),
            "{}_{}.wav".format(song, dataset['DCS']['QB_singers'][3])
        ]

        # no combos here, there are only four singers per song
        params['output_fname'] = "{}_1_2_2_2.wav".format(song)
        '''
        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
            combine_audio_files(params)
        '''

        if compute_metadata:
            print("Annotations for {}".format(song))
            annotation_files = [
                "{}_{}.jams".format(song, dataset['DCS']['QB_singers'][0]),
                "{}_{}.jams".format(song, dataset['DCS']['QB_singers'][1]),
                "{}_{}.jams".format(song, dataset['DCS']['QB_singers'][2]),
                "{}_{}.jams".format(song, dataset['DCS']['QB_singers'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files, params['annot_folder'])
            '''
            if reverb:
                idx = -1
                for annot in annotation_files:
                    idx += 1
                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'],
                                            params['filenames'][idx])
            '''

        print('{} quartets mixed and exported'.format(song))

    # ------------ Process Barbershop Quartets ------------ #
    song_idx = -1

    for song in dataset['BSQ']['songs']:
        song_idx += 1
        parts = dataset['BSQ']['num_parts'][song_idx]

        params = {}
        params['audio_folder'] = config.bsq_folder_audio
        params['annot_folder'] = config.bsq_folder_annot
        params['sr'] = 44100
        params['reverb'] = True

        params['filenames'] = [
            "{}_part{}_s_1ch.wav".format(song, parts),
            "{}_part{}_a_1ch.wav".format(song, parts),
            "{}_part{}_t_1ch.wav".format(song, parts),
            "{}_part{}_b_1ch.wav".format(song, parts)
        ]

        params['output_fname'] = "{}_{}_satb.wav".format(song, parts)

        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
            combine_audio_files(params)

        if compute_metadata:
            print("Annotations for {}".format(song))
            annotation_files = [
                "{}_part{}_s_1ch_pyin.jams".format(song, parts),
                "{}_part{}_a_1ch_pyin.jams".format(song, parts),
                "{}_part{}_t_1ch_pyin.jams".format(song, parts),
                "{}_part{}_b_1ch_pyin.jams".format(song, parts)
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files,
                                        params['annot_folder'])

            if reverb:
                idx = -1
                for annot in annotation_files:
                    idx += 1
                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'],
                                            params['filenames'][idx])

        print('{} quartets mixed and exported'.format(song))

    # ------------ Process Bach Chorales ------------ #
    song_idx = -1
    for song in dataset['BC']['songs']:
        song_idx += 1
        parts = dataset['BC']['num_parts'][song_idx]

        params = {}
        params['audio_folder'] = config.bc_folder_audio
        params['annot_folder'] = config.bc_folder_annot
        params['sr'] = 44100
        params['reverb'] = True

        params['filenames'] = [
            "{}_part{}_s_1ch.wav".format(song, parts),
            "{}_part{}_a_1ch.wav".format(song, parts),
            "{}_part{}_t_1ch.wav".format(song, parts),
            "{}_part{}_b_1ch.wav".format(song, parts)
        ]

        params['output_fname'] = "{}_{}_satb.wav".format(song, parts)

        if compute_audio_mix and os.path.exists(os.path.join(mixes_wavpath, params['output_fname'])):
            combine_audio_files(params)

        if compute_metadata:
            print("Annotations for {}".format(song))
            annotation_files = [
                "{}_part{}_s_1ch_pyin.jams".format(song, parts),
                "{}_part{}_a_1ch_pyin.jams".format(song, parts),
                "{}_part{}_t_1ch_pyin.jams".format(song, parts),
                "{}_part{}_b_1ch_pyin.jams".format(song, parts)
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, params['output_fname'], annotation_files,
                                        params['annot_folder'])

            if reverb:
                idx = -1
                for annot in annotation_files:
                    idx += 1
                    utils.shift_annotations(params['annot_folder'], annot, params['audio_folder'],
                                            params['filenames'][idx])

    print('{} quartets mixed and exported'.format(song))


    # Store the metadata file
    if compute_metadata:
        utils.save_json_data(mtracks, os.path.join(mixes_wavpath, 'mtracks_info.json'))


def main():

    # load the dataset info
    dataset = config.dataset

    print("Dataset info loaded.")

    # use the dataset information to create audio mixtures and annotations
    create_full_dataset_mixes(dataset, config.audio_save_folder, reverb=True, exclude_dataset=None,
                              compute_audio_mix=True, compute_metadata=True)


if __name__ == '__main__':

    main()

