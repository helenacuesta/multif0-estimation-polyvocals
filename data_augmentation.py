'''
This script augments the dataset by pitch-shifting the individual singers recordings 2 semitones above and below the original
pitch. It uses the MUDA python package for music data augmentation.

This is a stand-alone script that is executed even before the setup process.

'''

import muda
import jams
import numpy as np
import pandas as pd

import os
import argparse
import csv

import utils


def create_jams(times, freqs, outfile):

    track_duration = times[-1] + (times[-1] - times[-2])

    jam = jams.JAMS()
    jam.file_metadata.duration = track_duration

    pitch_a = jams.Annotation(namespace='pitch_contour')
    pitch_a.annotation_metadata.data_source = "Tony pitch estimation + manual correction"
    pitch_a.annotation_metadata.annotation_tools = "Tony"
    #pitch_a.annotation_metadata.curator = jams.Curator(name="Helena Cuesta",
    #                                                       email="helena.cuesta@upf.edu")

    for t, p in zip(times, freqs):

        if p != 0:
            pitch_a.append(
                time=t,
                duration=0.0,
                value={'index': 0, 'frequency': p, 'voiced': True},
                confidence=1.0
            )
        else:
            pitch_a.append(
                time=t,
                duration=0.0,
                value={'index': 0, 'frequency': p, 'voiced': False},
                confidence=1.0
            )

    jam.annotations.append(pitch_a)

    jam.save(outfile)

#####################################################################################

def read_annotations_f0(annot_fname, annot_path, dataset=None):

    if annot_fname.endswith('f0'):

        if dataset == 'ECS':
            # loadtxt fails with some ECS files
            # annotation = np.loadtxt(os.path.join(annot_path, annot_fname))
            annotation = []
            with open(os.path.join(annot_path, annot_fname), newline='\n') as f:
                reader = csv.reader(f, delimiter='\t')
                for line in reader:
                    annotation.append([float(line[0]), float(line[1])])
                annotation = np.array(annotation)
        else:
            print("CSD: {}".format(annot_path))
            annotation = np.loadtxt(os.path.join(annot_path, annot_fname))

    elif annot_fname.endswith('csv'):
        annotation = pd.read_csv(os.path.join(annot_path, annot_fname), header=None).values
    else:
        print("Invalid annotation file format for {}".format(annot_fname))

    return annotation[:, 0], annotation[:, 1]

#####################################################################################

def pitch_shifting(audio_fname, jams_fname, audio_folder, jams_folder, n_samples=5, l=-2, u=2):

    print(audio_folder, audio_fname)

    orig = muda.load_jam_audio(os.path.join(jams_folder, jams_fname),
                               os.path.join(audio_folder, audio_fname)
                               )

    pitchshift = muda.deformers.LinearPitchShift(n_samples=n_samples, lower=l, upper=u)

    for i, jam_out in enumerate(pitchshift.transform(orig)):

        muda.save(os.path.join(audio_folder, '{}_{}'.format(i, audio_fname)),
                  os.path.join(jams_folder, '{}_{}'.format(i, jams_fname)),
                  jam_out)


#####################################################################################

def add_unvoiced_frames(annot_path, audio_path, format='csv'):

    if not os.path.exists(os.path.join(annot_path, 'constant_timebase')):
        os.mkdir(os.path.join(annot_path, 'constant_timebase'))

    for fname in os.listdir(annot_path):
        if not fname.endswith(format): continue
        if not 'smoothedpitchtrack' in fname: continue

        utils.pyin_to_unvoiced(annot_path, fname, audio_path, fname.replace('_vamp_pyin_pyin_smoothedpitchtrack.csv',
                                                                            '.wav'))


#####################################################################################

def main(args):

    # fix pYIN annotations for BC and BSQ and then keep going. After this, annot folder becomes constant_timebase
    if args.dataset == 'BC' or args.dataset == 'BSQ':

        add_unvoiced_frames(args.path_to_annotations, args.path_to_audio)

        # once unvoiced frames are fixed, data augmentation steps for audio and annotations (CSV)
        for fn in os.listdir(os.path.join(args.path_to_annotations, 'constant_timebase')):

            if not fn.endswith('csv'): continue

            orig_times, orig_freqs = read_annotations_f0(fn, os.path.join(args.path_to_annotations, 'constant_timebase'))

            new_fname = fn.replace('_vamp_pyin_pyin_smoothedpitchtrack.csv', '_pyin.jams')
            outfile = os.path.join(args.path_to_annotations, 'constant_timebase', new_fname)
            create_jams(orig_times, orig_freqs, outfile)

            # step 3 is pitch-shifting audio and annotations accordingly
            pitch_shifting(new_fname.replace('_pyin.jams', '.wav'), new_fname, args.path_to_audio,
                           os.path.join(args.path_to_annotations, 'constant_timebase'))


    elif args.dataset == 'ECS':
        # step 2 is converting annotation files to jams
        for fn in os.listdir(args.path_to_annotations):

            if not fn.endswith('f0'): continue

            orig_times, orig_freqs = read_annotations_f0(fn, args.path_to_annotations, dataset='ECS')

            outfile = os.path.join(args.path_to_annotations, fn.replace('f0', 'jams'))
            create_jams(orig_times, orig_freqs,  outfile)

            # step 3 is pitch-shifting audio and annotations accordingly

            pitch_shifting(fn.replace('f0', 'wav'), fn.replace('f0', 'jams'), args.path_to_audio,
                           args.path_to_annotations)

    else:
        # step 2 is converting annotation files to jams
        for fn in os.listdir(args.path_to_annotations):

            if not fn.endswith('f0') or fn.endswith('csv'): continue

            orig_times, orig_freqs = read_annotations_f0(fn, args.path_to_annotations)

            if fn.endswith('f0'):
                outfile = os.path.join(args.path_to_annotations, fn.replace('f0', 'jams'))
                create_jams(orig_times, orig_freqs,  outfile)

            elif fn.endswith('csv'):
                outfile = os.path.join(args.path_to_annotations, fn.replace('csv', 'jams'))
                create_jams(orig_times, orig_freqs, outfile)

            # step 3 is pitch-shifting audio and annotations accordingly
            if fn.endswith('f0'):
                pitch_shifting(fn.replace('f0', 'wav'), fn.replace('f0', 'jams'), args.path_to_audio,
                               args.path_to_annotations)
            elif fn.endswith('csv'):
                pitch_shifting(fn.replace('csv', 'wav'), fn.replace('csv', 'jams'), args.path_to_audio,
                               args.path_to_annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Several steps: (1) Convert f0 annotations to jams format for further use, (2) pitch-shifting for dataugm.")

    parser.add_argument("--f0-path",
                        dest='path_to_annotations',
                        type=str,
                        help="Path to folder with f0 files. ")

    parser.add_argument("--audio-path",
                        dest='path_to_audio',
                        type=str,
                        help="Path to folder with audio files. ")

    parser.add_argument("--dataset",
                        dest="dataset",
                        type=str,
                        help="Indicate the dataset to process: BC / BSQ / CSD / ECS / DCS")
    '''
    parser.add_argument("--pyin",
                        dest='pyin',
                        type=str,
                        help="If F0-trajectories come from pYIN they might not have unvoiced frames as 0. If 'yes', the code takes care of this. If 'no', "
                             "it assumes unvoiced frames are OK and skips this.")
    '''

    main(parser.parse_args())
