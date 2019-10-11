# Multi-f0 Estimation in Vocal Polyphonic Recordings

This README file is a work in progress. We start by stating the order that the user should
follow when running the code.

1. *data_augmentation*: pitch-shifting the individual audio recordings for the three datasets. This
is an automated process and if you plan to use other data the code needs to be adapted to it. This script 
also converts annotations in text format (.f0 files) into jams format.

2. *0_setup*: create the dataset structure, and create audio mixtures and associated annotations.

3. *1_prep*: create the input features (HCQT+Phase differentials) and targets (blurred activation map) and also 
the json file with the data splits: train, test, validation.

4. *2_training*: train specified model (model5, model6, model7, see script _models.py_) with the training
set, choose the threshold that maximizes accuracy on the validation set
and evaluate the model on the test set.

5. *config*: the config file contains most of the fixed variables needed in the code: 
paths, training parameters, and all the information to create the datasert structure.

6. *models*: definition of the models that are part of the experiments.