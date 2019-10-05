'''
This script augments the dataset by pitch-shifting the individual singers recordings 2 semitones above and below the original
pitch. It uses the MUDA python package for music data augmentation.

This is a stand-alone script that is executed even before the setup process.

'''

import muda
import jams
import numpy as np

import os
import argparse


def create_jams(times, freqs, outfile):

    track_duration = times[-1] + (times[-1] - times[-2])

    jam = jams.JAMS()
    jam.file_metadata.duration = track_duration

    pitch_a = jams.Annotation(namespace='pitch_contour')
    pitch_a.annotation_metadata.data_source = "Tony pitch estimation + manual correction"
    pitch_a.annotation_metadata.annotation_tools = "Tony"
    pitch_a.annotation_metadata.curator = jams.Curator(name="Helena Cuesta",
                                                       email="helena.cuesta@upf.edu")

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

def read_annotations_f0(path_to_file):

    annotation = np.loadtxt(path_to_file)

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

#path_to_annotations = '/Volumes/MTGMIR/ChoralSingingDataset'


def main(args):

    for fn in os.listdir(args.path_to_annotations):

        if not fn.endswith('f0'): continue

        infile = os.path.join(args.path_to_annotations, fn)

        orig_times, orig_freqs = read_annotations_f0(infile)
        create_jams(orig_times, orig_freqs, infile[:-2] + 'jams')

        pitch_shifting(fn[:-2] + 'wav', fn[:-2]+'jams', args.path_to_annotations, args.path_to_annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert f0 annotations to jams format for further use.")

    parser.add_argument("--f0-path",
                        dest='path_to_annotations',
                        type=str,
                        help="Path to folder with f0 files. ")

    main(parser.parse_args())
