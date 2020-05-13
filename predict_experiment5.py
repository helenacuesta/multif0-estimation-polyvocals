import models
import utils
from config import *

import os
import argparse


'''Predict a multiple F0 output using the specified model from an input audio file
'''


''' Parameters
'''


def main(args):

    pth_model = args.pth_model
    save_path = args.save_path
    list_of_files = args.list_of_files

    audio_path = np.loadtxt(list_of_files)
    print(audio_path, type(audio_path))
    #fname_list = np.loadtxt(list_of_files)[1:]


    save_key = 'exp5multif0'
    model_path = os.path.join(pth_model, "{}.pkl".format(save_key))

    model = models.build_model3()
    model.load_weights(model_path)

    model.compile(
        loss=utils.bkld, metrics=['mse', utils.soft_binary_accuracy],
        optimizer='adam'
    )

    print("Model compiled")

    thresh = 0.5

    for fname in fname_list:

        if not fname.endswith('.wav'): continue

        # predict using trained model
        predicted_output, _, _ = utils.get_single_test_prediction(model,
                                                                  npy_file=None,
                                                                  audio_file=os.path.join(audio_path, fname))

        np.save(os.path.join(
            save_path, "{}_prediction.npy".format(fname.split('.')[0])),
            predicted_output.astype(np.float32))

        # get multif0 output from prediction
        est_times, est_freqs = utils.pitch_activations_to_mf0(predicted_output, thresh)

        for i, (tms, fqs) in enumerate(zip(est_times, est_freqs)):
            if any(fqs <= 0):
                est_freqs[i] = np.array([f for f in fqs if f > 0])

        output_mf0 = os.path.join(save_path, "{}.csv".format(fname.split('.')[0]))
        utils.save_multif0_output(est_times, est_freqs, output_mf0)

        print("     Multiple F0 prediction exported for {}".format(fname))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict multiple F0 output using model 3 without reverb. Experiment 5.")

    parser.add_argument("--model_path",
                        dest='pth_model',
                        type=str,
                        help="Path to the model weights.")

    parser.add_argument("--save_path",
                        dest='save_path',
                        type=str,
                        help="Folder to save predicted outputs.")

    parser.add_argument("--list_of_files",
                        dest='list_of_files',
                        type=str,
                        help="Path to the text file with the list of files to process. The first line should contain"
                        "the folder where the files are located.")


    main(parser.parse_args())
