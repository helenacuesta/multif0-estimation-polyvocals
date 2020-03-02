# Multi-F0 Estimation in Vocal Polyphonic Recordings

This README file is a work in progress. We start by stating the order that the user should
follow when running the code.

1. ```data_augmentation```: first applying some corrections to the pYIN annotations of BC and BSQ datasets. 
Then, convert all annotation files (either f0 files or csv files) to **jams** for convenience.
Finally, pitch-shifting the individual audio recordings for all datasets. 
All the code is specific for the datasets used in the original project. If using other data, it needs
to be adapted to the new formats.

2. ```config```: this script helps making the data preparation easier. It covers the basics of the structure
of each dataset: song names, singers, 

3. ```0_setup```: using the info from ```config```, this script creates the whole dataset structure, and generates
the audio mixtures and associated annotations for further processing. It's the first step of data preparation,
before feature computation. Use the mtracks 

4. ```1_prep```: feature and target computation from all audio mixtures

3. *1_prep*: create the input features (HCQT+Phase differentials) and targets (blurred activation map) and also 
the json file with the data splits: train, test, validation.

4. *2_training*: train specified model (model5, model6, model7, see script _models.py_) with the training
set, choose the threshold that maximizes accuracy on the validation set
and evaluate the model on the test set.

5. *config*: the config file contains most of the fixed variables needed in the code: 
paths, training parameters, and all the information to create the datasert structure.

6. *models*: definition of the models that are part of the experiments.