import os
import json

import keras

import config
import utils


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

        self.train_files = utils.get_file_paths(self.train_set, self.data_path)
        self.validation_files = utils.get_file_paths(
            self.validation_set, self.data_path
        )
        self.test_files = utils.get_file_paths(self.test_set, self.data_path)

        self.batch_size = batch_size
        self.active_str = active_str
        self.muxrate = muxrate

    def load_data_splits(self):
        """Get randomized artist-conditional splits
        """
        with open(self.data_splits_path, 'r') as fhandle:
            data_splits = json.load(fhandle)

        return data_splits['train'], data_splits['validate'], data_splits['test']

    def get_train_generator(self):
        """return a training data generator
        """
        return utils.keras_generator(
            self.train_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )



    def get_validation_generator(self):
        """return a validation data generator
        """
        return utils.keras_generator(
            self.validation_files, self.input_patch_size,
            self.batch_size, self.active_str, self.muxrate
        )

    def get_test_generator(self):
        """return a test data generator
        """
        return utils.keras_generator(
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


def train(model, model_save_path, exper_dir, batch_size, active_str, muxrate):

    data_path = utils.data_path_multif0()

    input_patch_size = (360, 50)
    data_splits_path = os.path.join(exper_dir, 'data_splits.json')

    ## DATA MESS SETUP


    # create data object with
    # data_splits_path, data_path, input_patch_size, batch_size, active_str, muxrate

    dat = Data(
        data_splits_path, data_path, input_patch_size,
        batch_size, active_str, muxrate
    )



    # instantiate train and validation generators

    train_generator = dat.get_train_generator()
    validation_generator = dat.get_validation_generator()

    model.compile(
        loss=utils.bkld, metrics=['mse', utils.soft_binary_accuracy],
        optimizer='adam'
    )

    print(model.summary(line_length=80))

    ## HOPEFULLY FIT MODEL

    history = model.fit_generator(
        train_generator, config.SAMPLES_PER_EPOCH, epochs=config.NB_EPOCHS, verbose=1,
        validation_data=validation_generator, validation_steps=config.NB_VAL_SAMPLES,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_save_path, save_best_only=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(patience=5, verbose=1),
            keras.callbacks.EarlyStopping(patience=25, verbose=0)
        ]
    )

    model.load_weights(model_save_path)

    return model, history, dat


def run_evaluation(exper_dir, save_key, history, dat, model):

    (save_path, _, plot_save_path,
     model_scores_path, _, _
     ) = eval_utils.get_paths(exper_dir, save_key)

    ## Results plots
    print("plotting results...")
    eval_utils.plot_metrics_epochs(history, plot_save_path)

    ## Evaluate
    print("getting model metrics...")
    eval_utils.get_model_metrics(dat, model, model_scores_path)

    print("getting best threshold...")
    thresh = eval_utils.get_best_thresh(dat, model)


    print("scoring multif0 metrics on test sets...")
    eval_utils.score_on_test_set(model, save_path, thresh)


def experiment(save_key, model, batch_size, active_str, muxrate):
    """
    This should be common code for all experiments
    """

    exper_dir = train_utils.experiment_output_path()

    (save_path, _, plot_save_path,
     model_scores_path, _, _
     ) = eval_utils.get_paths(exper_dir, save_key)


    model_save_path_rt = './models'
    if not os.path.exists(model_save_path_rt):
        os.mkdir(model_save_path_rt)

    model_save_path = os.path.join(model_save_path_rt, "{}.pkl".format(save_key))


    # create data splits file if it doesnt exist
    if not os.path.exists(
        os.path.join(exper_dir, 'mtracks_metadata_augm.json')):
        create_data_splits(path_to_metadata_file='./mtracks_metadata_augm.json', exper_dir=exper_dir)


    model, history, dat = train(model, model_save_path, exper_dir,
                                batch_size, active_str, muxrate)

    run_evaluation(exper_dir, save_key, history, dat, model)
    print("Done! Results saved to {}".format(save_path))
