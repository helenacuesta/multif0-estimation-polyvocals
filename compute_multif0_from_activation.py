import utils
import config

import os
import numpy as np
import csv

def main():

    print("Computing multi-f0 ground truth for training files...")
    data_splits = utils.load_json_data(
        os.path.join(config.data_save_folder, 'data_splits.json'))

    for fn in data_splits['train']:

        targ = np.load(os.path.join(config.data_save_folder, 'outputs', fn.replace('.wav', '_output.npy')))
        ts, fs = utils.pitch_activations_to_mf0(targ, 0.9)
        output_mf0 = os.path.join(
            utils.test_path(), fn.replace('.wav', '.csv')
        )
        with open(output_mf0, 'w') as f:
            writer = csv.writer(f)
            for i in range(len(ts)):
                writer.writerow(np.array([ts[i], fs[i]]))

    print("Computing multi-f0 ground truth for validation files...")
    data_splits = utils.load_json_data(
        os.path.join(config.data_save_folder, 'data_splits.json'))

    for fn in data_splits['validate']:

        targ = np.load(os.path.join(config.data_save_folder, 'outputs', fn.replace('.wav', '_output.npy')))
        ts, fs = utils.pitch_activations_to_mf0(targ, 0.9)
        output_mf0 = os.path.join(
            utils.test_path(), fn.replace('.wav', '.csv')
        )
        with open(output_mf0, 'w') as f:
            writer = csv.writer(f)
            for i in range(len(ts)):
                writer.writerow(np.array([ts[i], fs[i]]))

    print("Computing multi-f0 ground truth for training files...")
    data_splits = utils.load_json_data(
        os.path.join(config.data_save_folder, 'data_splits.json'))

    for fn in data_splits['test']:

        targ = np.load(os.path.join(config.data_save_folder, 'outputs', fn.replace('.wav', '_output.npy')))
        ts, fs = utils.pitch_activations_to_mf0(targ, 0.9)
        output_mf0 = os.path.join(
            utils.test_path(), fn.replace('.wav', '.csv')
        )
        with open(output_mf0, 'w') as f:
            writer = csv.writer(f)
            for i in range(len(ts)):
                writer.writerow(np.array([ts[i], fs[i]]))


main()

