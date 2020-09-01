# Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks

This repo contains the companion code for the ISMIR paper:

Cuesta, H., McFee, B., & Gómez, E. (2020). Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural
Networks. In _Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR)._ 
Montreal, Canada (virtual).

Please, cite the aforementioned paper if using this material.

## Description

### Main scripts
```predict_on_audio.py```:
This is the main script of the repo if you want to use the models "out-of-the-box".
Run the script specifying the working mode, i.e., an audio file or a folder that contains several audio files, and
which model to run:

* Early/Shallow --> model1
* Early/Deep --> model2
* Late/Deep --> model3

An example command to predict multiple F0 values of an audio file using the Late/Deep model. Note that we use a "0"
flag for the working mode we do don't use.

```
python predict.py --model model3 --audiofile poly_sing.wav --audio_folder 0
```

And an example command for the folder mode:
```
python predict.py --model model3 --audiofile 0 --audio_folder ../data/vocal_music
```

The system will save a CSV file with the output in the same directory where the input audio file is located.

```utils.py``` and ```utils_train.py``` both contain util functions for the experiments.

### experiments

This folder contains all the code we developed to carry out the experiments: data augmentation, feature extraction, 
training, evaluation...etc. Here's a short description of each of them. **This part of the documentation is 
a work in progress and we keep updating it.**

_**Note**: The scripts related to data structure and preparation are specifically designed for our working datasets, 
some of them not publicly-available (see paper for more info on this). Although the code can be adapted to other datasets, note that the user needs
to modify the dataset structure that we define in ```setup.py```._

```data_augmentation.py```: this script first converts all F0 annotations of individual audio files from TXT/CSV 
files to **jams** format. Then, each individual audio recording and its associated F0 curve is pitch-shifted using 
± 2 semitones using the MUDA package (https://github.com/bmcfee/muda). Remember that this code is specific for our 
working datasets. The main pitch-shifting code can be found in the function ```pitch_shifting``` inside the script.

```config.py```: this is the main configuration file for the whole project. It contains all the working directories, 
some training parameters, as well as the whole dataset structure, which is stored in a big Python dictionary 
named _dataset_. This dictionary covers all filenames for the different files of the various datasets, number of singers,
annotation files...etc. It is further used to generate all the audio mixtures and associated annotations.

```0_setup.py```: here we first load the dataset structure from ```config```, and then use the info to create all 
the audio mixtures combining singers. We create the audio mixtures, a version of the audio mixtures with additional 
reverb, and then we prepare the associated annotations. A json file (mtracks_info.json) that contains all the
information for every audio mixture is also created.

```1_prep.py```: feature computation for all audio mixtures, targets computation for all annotations, and generation
of the data splits json file with train/validation/test data splits for all experiments (data_splits.json).

```2_training.py```

```3_training_nophase.py```





