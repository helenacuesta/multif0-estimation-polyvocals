'''
This script creates vocal quartets using the three target datasets: CSD, ECS, DCS, which are augmented
using pitch-shifting from MUDA.

The code is specifically written for these datasets, but can be easily adapted to other multitracks.
'''

import sox
import os

import pandas as pd

import config
import utils


def combine_audio_files(piece, singer_idxs, output_fname, mode):

    if mode == 'csd':

        filenames = [
            '{}soprano_{}.wav'.format(piece, singer_idxs[0]),
            '{}alto_{}.wav'.format(piece, singer_idxs[1]),
            '{}tenor_{}.wav'.format(piece, singer_idxs[2]),
            '{}bass_{}.wav'.format(piece, singer_idxs[3]),
        ]

        cmb = sox.Combiner()
        cmb.convert(samplerate=22050)
        cmb.build(
            [config.csd_folder + filenames[0], config.csd_folder + filenames[1],
             config.csd_folder + filenames[2], config.csd_folder + filenames[3]],
            os.path.join(config.audio_save_folder, output_fname), 'mix', input_volumes=[0.6, 0.3, 0.3, 0.3]
        )

        tf = sox.Transformer()
        tf.reverb(reverberance=100, pre_delay=30, high_freq_damping=80, wet_gain=4)
        tf.build(os.path.join(config.audio_save_folder, output_fname),
                 os.path.join(config.audio_save_folder, 'rev_' + output_fname))

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


    elif mode == 'dcs':

        filenames = [
            '{}S{}_LRX.wav'.format(piece, singer_idxs[0]),
            '{}A{}_LRX.wav'.format(piece, singer_idxs[1]),
            '{}T{}_LRX.wav'.format(piece, singer_idxs[2]),
            '{}B{}_LRX.wav'.format(piece, singer_idxs[3]),
        ]


        cmb = sox.Combiner()
        cmb.convert(samplerate=22050)

        try:
            cmb.build(
                [config.dcs_folder + filenames[0], config.dcs_folder + filenames[1],
                 config.dcs_folder + filenames[2], config.dcs_folder + filenames[3]],
                os.path.join(config.audio_save_folder, output_fname), 'mix'
            )

            tf = sox.Transformer()
            tf.reverb(reverberance=100, pre_delay=30, high_freq_damping=80, wet_gain=4)
            tf.build(os.path.join(config.audio_save_folder, output_fname),
                     os.path.join(config.audio_save_folder, 'rev_' + output_fname))

            return 1
        except:
            print('Combination not found! {}'.format(output_fname))
            return 0

    elif mode == 'dcs_dyn':

        filenames = [
            '{}S{}_DYN.wav'.format(piece, singer_idxs[0]),
            '{}A{}_DYN.wav'.format(piece, singer_idxs[1]),
            '{}T{}_DYN.wav'.format(piece, singer_idxs[2]),
            '{}B{}_DYN.wav'.format(piece, singer_idxs[3]),
        ]

        cmb = sox.Combiner()
        cmb.convert(samplerate=22050)

        try:
            cmb.build(
                [config.dcs_folder + filenames[0], config.dcs_folder + filenames[1],
                 config.dcs_folder + filenames[2], config.dcs_folder + filenames[3]],
                os.path.join(config.audio_save_folder, output_fname), 'mix'
            )

            tf = sox.Transformer()
            tf.reverb(reverberance=100, pre_delay=30, high_freq_damping=80, wet_gain=4)
            tf.build(os.path.join(config.audio_save_folder, output_fname),
                     os.path.join(config.audio_save_folder, 'rev_' + output_fname))

            return 1

        except:
            print('Combination not found! {}'.format(output_fname))
            return 0


def create_dict_entry(diction, audiopath, audiofname, annot_files, annot_folder):

    diction[audiofname] = dict()
    diction[audiofname]['audiopath'] = audiopath
    diction[audiofname]['annot_files'] = annot_files
    diction[audiofname]['annot_folder'] = annot_folder

    return diction


def create_full_dataset_mixes(dataset, mixes_wavpath, compute_audio_mix=True, compute_metadata=True):

    mtracks = dict()

    # ------------ Process Choral Singing Dataset ------------ #

    mode = 'csd'

    for song in dataset['CSD']['songs']:

        for combo in dataset['CSD']['combos']:

            output_fname = '{}{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):

                combine_audio_files(song, combo, output_fname, mode)

            if compute_metadata:
                # add location of individual jams files
                annotation_files = [
                    '{}soprano_{}.jams'.format(song, combo[0]), '{}alto_{}.jams'.format(song, combo[1]),
                    '{}tenor_{}.jams'.format(song, combo[2]), '{}bass_{}.jams'.format(song, combo[3])
                ]

                mtracks = create_dict_entry(mtracks, mixes_wavpath, 'rev_'+ output_fname, annotation_files, config.csd_folder)

        print('{} quartets mixed'.format(song))

    # ------------ Process ESMUC ChoralSet ------------ #

    # In this dataset we do not have a uniform number of singers per part.
    # We have: 4 sopranos, 3 altos, 3 tenors and 2 basses. Singer combinations need to be done accordingly.

    mode = 'ecs'

    for song in dataset['ECS']['songs']:

        for combo in dataset['ECS']['combos']:

            output_fname = '{}{}_{}_{}_{}.wav'.format(song, combo[0], combo[1], combo[2], combo[3])

            if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):
                # create audio mixture
                combine_audio_files(song, combo, output_fname, mode)

            if compute_metadata:
                # add location of individual jams files

                if 'WU_' in song:
                    annotation_files = [
                        '{}S{}.jams'.format(song, combo[0]), '{}T{}.jams'.format(song, combo[2])]

                else:
                    annotation_files = [
                        '{}S{}.jams'.format(song, combo[0]), '{}A{}.jams'.format(song, combo[1]),
                        '{}T{}.jams'.format(song, combo[2]), '{}B{}.jams'.format(song, combo[3])
                    ]


                mtracks = create_dict_entry(mtracks, mixes_wavpath, 'rev_'+ output_fname, annotation_files, config.ecs_folder)

        print('{} quartets mixed and exported'.format(song))

    # ------------ Process Dagstuhl ChoirSet ------------ #

    mode = 'dcs'

    for song in dataset['DCS']['All']['songs']:

        # combo dynamic
        output_fname = '{}{}_{}_{}_{}_DYN.wav'.format(song, dataset['DCS']['All']['dyn_combos'][0],
                                                      dataset['DCS']['All']['dyn_combos'][1],
                                                      dataset['DCS']['All']['dyn_combos'][2],
                                                      dataset['DCS']['All']['dyn_combos'][3])

        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):
            # create audio mixture
            o = combine_audio_files(song, dataset['DCS']['All']['dyn_combos'], output_fname, 'dcs_dyn')
            if o == 0: continue

        if compute_metadata:
            # add location of individual jams files
            annotation_files = [
                '{}S{}_DYN.jams'.format(song, dataset['DCS']['All']['dyn_combos'][0]),
                '{}A{}_DYN.jams'.format(song, dataset['DCS']['All']['dyn_combos'][1]),
                '{}T{}_DYN.jams'.format(song, dataset['DCS']['All']['dyn_combos'][2]),
                '{}B{}_DYN.jams'.format(song, dataset['DCS']['All']['dyn_combos'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, 'rev_'+ output_fname, annotation_files,
                                            config.dcs_folder)

        print('{} quartets mixed and exported'.format(song))

    for song in dataset['DCS']['QuartetA']['songs']:

        # combo dynamic
        output_fname = '{}{}_{}_{}_{}_DYN.wav'.format(song, dataset['DCS']['QuartetA']['combos'][0],
                                                      dataset['DCS']['QuartetA']['combos'][1],
                                                      dataset['DCS']['QuartetA']['combos'][2],
                                                      dataset['DCS']['QuartetA']['combos'][3])

        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):
            # create audio mixture
            o = combine_audio_files(song, dataset['DCS']['QuartetA']['combos'], output_fname, 'dcs_dyn')
            if o == 0: continue

        if compute_metadata:
            # add location of individual jams files
            annotation_files = [
                '{}S{}_DYN.jams'.format(song, dataset['DCS']['QuartetA']['combos'][0]),
                '{}A{}_DYN.jams'.format(song, dataset['DCS']['QuartetA']['combos'][1]),
                '{}T{}_DYN.jams'.format(song, dataset['DCS']['QuartetA']['combos'][2]),
                '{}B{}_DYN.jams'.format(song, dataset['DCS']['QuartetA']['combos'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, 'rev_'+ output_fname, annotation_files,
                                            config.dcs_folder)

        print('{} quartets mixed and exported'.format(song))

    for song in dataset['DCS']['QuartetB']['songs']:


        # combo dynamic
        output_fname = '{}{}_{}_{}_{}_DYN.wav'.format(song, dataset['DCS']['QuartetB']['combos'][0],
                                                      dataset['DCS']['QuartetB']['combos'][1],
                                                      dataset['DCS']['QuartetB']['combos'][2],
                                                      dataset['DCS']['QuartetB']['combos'][3])

        if compute_audio_mix and not os.path.exists(os.path.join(mixes_wavpath, output_fname)):
            # create audio mixture
            o = combine_audio_files(song, dataset['DCS']['QuartetB']['combos'], output_fname, 'dcs_dyn')
            if o == 0: continue

        if compute_metadata:
            # add location of individual jams files
            annotation_files = [
                '{}S{}_DYN.jams'.format(song, dataset['DCS']['QuartetB']['combos'][0]),
                '{}A{}_DYN.jams'.format(song, dataset['DCS']['QuartetB']['combos'][1]),
                '{}T{}_DYN.jams'.format(song, dataset['DCS']['QuartetB']['combos'][2]),
                '{}B{}_DYN.jams'.format(song, dataset['DCS']['QuartetB']['combos'][3])
            ]

            mtracks = create_dict_entry(mtracks, mixes_wavpath, 'rev_'+ output_fname, annotation_files,
                                            config.dcs_folder)

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
    create_full_dataset_mixes(dataset, config.audio_save_folder) #compute_audio_mix=False, compute_metadata=False)


if __name__ == '__main__':

    main()

