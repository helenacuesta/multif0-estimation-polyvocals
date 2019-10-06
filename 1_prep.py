'''

This script generates inputs and targets for the multif0 learning algorithm.
Inputs are HCQTs and targets are blurred activation functions.

'''

import argparse
from joblib import Parallel, delayed

import utils
import config

import os



def define_parameters():

    wavmixes_path = config.audio_save_folder
    save_dir = config.data_save_folder

    return save_dir, wavmixes_path


def main(args):

    metad = utils.load_json_data(args.metadata_file)

    splits_path = os.path.join(config.data_save_folder, 'data_splits.json')
    utils.create_data_split(metad, splits_path)

    mtracks = []
    for ky in metad.keys():
        mtrack = metad[ky]
        mtrack['filename'] = ky
        mtracks.append(mtrack)


    nmixes = len(metad.keys())
    print("{} mixes to be processed".format(nmixes))
    idx=0

    Parallel(n_jobs=4, verbose=5)(
            delayed(utils.compute_features_mtrack)(
                mtrack, args.save_dir, args.wavmixes_path, idx
            ) for mtrack in mtracks)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate input files and targets for multif0 learning.")

    parser.add_argument("--audio-path",
                        dest='wavmixes_path',
                        type=str,
                        help="Path to folder with audio mixes. ")

    parser.add_argument("--metadata-path",
                        dest='metadata_file',
                        type=str,
                        help="Path to file with metadata of the dataset.")

    parser.add_argument("--save-dir",
                        dest='save_dir',
                        type=str,
                        help="Path to save generated npy files.")


    main(parser.parse_args())