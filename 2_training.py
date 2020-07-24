import os
import json

import keras
import numpy as np
import csv

import config
import utils
import utils_train
import models


import argparse

class Data(object):
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

        self.train_files = utils_train.get_file_paths(self.train_set, self.data_path)
        self.validation_files = utils_train.get_file_paths(
            self.validation_set, self.data_path
        )
        self.test_files = utils_train.get_file_paths(self.test_set, self.data_path)

        self.batch_size = batch_size
        self.active_str = active_str
        self.muxrate = muxrate

    def load_data_splits(self):

        with open(self.data_splits_path, 'r') as fhandle:
            data_splits = json.load(fhandle)

        return data_splits['train'], data_splits['validate'], data_splits['test']

    def get_train_generator(self):
        """return a training data generator
        """
        return utils_train.keras_generator(
            self.train_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )



    def get_validation_generator(self):
        """return a validation data generator
        """
        return utils_train.keras_generator(
            self.validation_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )

    def get_test_generator(self):
        """return a test data generator
        """
        return utils_train.keras_generator(
            self.test_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )


def load_data(load_path):
    with open(load_path, 'r') as fp:
        data = json.load(fp)
    return data


def create_data_splits(path_to_metadata_file, exper_dir):

    metadata = load_data(path_to_metadata_file)

    utils.create_data_split(metadata,
                           os.path.join(exper_dir, 'data_splits.json'))


def train(model, model_save_path, data_splits_file, batch_size, active_str, muxrate):

    #data_path = utils.data_path_multif0()
    data_path = config.data_save_folder

    input_patch_size = (360, 50)
    data_splits_path = os.path.join(config.data_save_folder, data_splits_file)

    ## DATA MESS SETUP

    dat = Data(
        data_splits_path, data_path, input_patch_size,
        batch_size, active_str, muxrate
    )

    # instantiate train and validation generators

    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    model.compile(
        loss=utils_train.bkld,
        metrics=['mse', utils_train.soft_binary_accuracy],
        optimizer='adam'
    )

    print(model.summary(line_length=80))

    # hopefully fit model

    history = model.fit_generator(
        train_generator, config.SAMPLES_PER_EPOCH, epochs=config.NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=config.NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_save_path, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=25, verbose=1)
        ]
    )

    model.load_weights(model_save_path)

    return model, history, dat


def run_evaluation(exper_dir, save_key, history, dat, model):

    (save_path, _, plot_save_path,
     model_scores_path, _, _
     ) = utils_train.get_paths(exper_dir, save_key)

    ## Results plots
    print("plotting results...")
    utils_train.plot_metrics_epochs(history, plot_save_path)

    ## Evaluate
    print("getting model metrics...")
    utils_train.get_model_metrics(dat, model, model_scores_path)

    print("getting best threshold...")
    thresh = utils_train.get_best_thresh(dat, model)


    print("scoring multif0 metrics on test sets...")
    utils_train.score_on_test_set(model, save_path, dat, thresh)


def experiment(save_key, model, data_splits_file, batch_size, active_str, muxrate):
    """
    This should be common code for all experiments
    """

    exper_dir = config.exper_output

    (save_path, _, plot_save_path,
     model_scores_path, _, _
     ) = utils_train.get_paths(exper_dir, save_key)


    model_save_path = '/scratch/hc2945/data/models/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    model_save_path = os.path.join(model_save_path, "{}.pkl".format(save_key))

    '''
    # create data splits file if it doesnt exist
    if not os.path.exists(
        os.path.join(exper_dir, 'data_splits.json')):
        create_data_splits(path_to_metadata_file='./mtracks_info.json', exper_dir=exper_dir)
    '''

    model, history, dat = train(model, model_save_path, data_splits_file,
                                batch_size, active_str, muxrate)


    run_evaluation(exper_dir, save_key, history, dat, model)
    print("Done! Results saved to {}".format(save_path))



def main(args):

    batch_size = 32
    active_str = 100
    muxrate = 32

    save_key = args.save_key
    data_splits_file = args.data_splits_file

    if args.model_name == 'model1':
        model = models.build_model1()
    elif args.model_name == 'model2':
        model = models.build_model2()
    elif args.model_name == 'model3':
        model = models.build_model3()
    else:
        print("Specified model does not exist. Please choose an valid model: model1, model2 or model3.")
        return

    experiment(save_key, model, data_splits_file, batch_size, active_str, muxrate)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train specified model with training set.")

    parser.add_argument("--model",
                        dest='model_name',
                        type=str,
                        help="Name of the model you want to train.")

    parser.add_argument("--save_key",
                        dest='save_key',
                        type=str,
                        help="String to save model-related data.")

    parser.add_argument("--data_splits_file",
                        dest='data_splits_file',
                        type=str,
                        help="Filename of the data splits file to use in the experiment.")


    main(parser.parse_args())
