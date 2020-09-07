# Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural Networks

This repo contains the companion code for the ISMIR paper:

Helena Cuesta, Brian McFee and Emilia Gómez (2020). **Multiple F0 Estimation in Vocal Ensembles using Convolutional Neural
Networks**. In _Proceedings of the 21st International Society for Music Information Retrieval Conference (ISMIR)._ 
Montreal, Canada (virtual).

Please, cite the aforementioned paper if using this material.

**Note 1:** This documentation is a work in progress and will be updated regularly. In addition, improved
models might be added to the repo in the future.

**Note 2:** This project builds upon the **DeepSalience** project:

Rachel M. Bittner, Brian McFee, Justin Salamon, Peter Li, and Juan P. Bello "Deep Salience Representations 
for F0 Estimation in Polyphonic Music”, ISMIR 2017, Suzhou, China.

Parts of this code are taken/adapted from their scripts, publicly available in 
https://github.com/rabitt/ismir2017-deepsalience.




## Usage / requirements

To use this framework, please follow these first steps: 

* Clone the repo

* ```cd ./multif0-estimation-polyvocals```

* Create your favourite environment (or not)
and install the required packages: ```pip install -r requirements.txt```


Note that this code runs Keras with Tensorflow as backend in Python 3.6.

In the requirements we specify ```tensorflow-gpu==1.15.2```, which runs the code in the GPU.
To use CPU instead, please install ```tensorflow==1.15```. 
Both can be installed using ```pip```.

Note that the experiments were done using tensorflow-gpu 
and we have only tested them with versions 1.13.1 and 1.15.2. 

The default version of tensorflow is currently 2.x. 
In further tests we plan to shift to tensorflow 2.

## Description

### main scripts: compute the output from these models
```predict_on_audio.py```:
This is the main script of the repo if you want to use the models "out-of-the-box".
Run the script specifying the working mode, i.e., an audio file or a folder that contains several audio files, and
which model to run:

* Early/Deep --> model1
* Early/Shallow --> model2
* Late/Deep --> model3 _(recommended)_

An example command to predict multiple F0 values of an audio file using the Late/Deep model. Note that we use a "0"
flag for the working mode we do don't use.

```
python predict_on_audio.py --model model3 --audiofile poly_sing.wav --audio_folder 0
```

And an example command for the folder mode:
```
python predict_on_audio.py --model model3 --audiofile 0 --audio_folder ../data/vocal_music
```

The system will save a CSV file with the output in the same directory where the input audio file is located.

```utils.py``` and ```utils_train.py``` both contain util functions for the experiments.

### experiments/

This folder contains all the code we developed to carry out the experiments: data augmentation, feature extraction, 
training, evaluation...etc. Here's a short description of each of them.

_**Note**: The scripts related to data structure and preparation are specifically designed for our working datasets, 
some of them not publicly-available (see paper for more info on this). Although the code can be adapted to 
other datasets, note that the user needs to modify the dataset structure that we define in ```setup.py```._

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
It's called from the command line with three parameters: the directory of the audio mixtures, the path to the 
mtracks_info.json file (generated during setup), and the directory to save features. Example:
```
python 1_prep.py --audio-path ./audiomixtures --metadata-path ./mtracks_info.json --save-dir ./features
```

```2_training.py```: this is the script for training, threshold optimization, and evaluation. It takes the 
_data_splits.json_ file, and uses the train/validation/test subsets by default. An example of training the 
Early/Shallow model:
```
python 2_training.py --model model2 --save_key exp2multif0 --data_splits_file data_splits.json
```

```3_training_nophase.py```: additional train/test script for the experiments without phase information at the input 
with the Late/Deep model.

```compute_multif0_from_activation.py```: generate (quantized) joint multiple F0 annotations
from the targets computed during data preparation. 

```exp4_bsq.py``` and ```predict_experiment5.py``` are the prediction scripts for experiments 2 and 3 in the paper.
They use models trained with ```2_training.py``` but different data splits. Note that the latter does not include
the code for the evaluation (will be added soon).
Due to data access concerns, these two experiments are barely reproducible. 
However, in the ```models``` folder we provide both trained models.

### models/

**exp1multif0.pkl**, Early/Deep (model1), default data splits

**exp2multif0.pkl**, Early/Shallow (model2), default data splits

**exp3multif0.pkl**, Late/Deep (model3), default data splits

**exp4multif0.pkl**, Late/Deep (model3), default data splits except BSQ (experiment 2 in the paper)

**exp5multif0.pkl**, Late/Deep (model3), default data splits except reverb files (experiment 3 in the paper)

### bsh/

Most of the bash scripts used for the experiments are inside this folder. exp1_1, exp1_2, exp1_3 refer to the 
three models in experiment 1; exp2 and exp3 belong to experiment 2 and 3 from the paper; exp4 is the no-phase
experiment. The versions of the modules we load in these scripts are the ones we used during our experiments, but
if attempting to re-run the code, the user might have to change them according to their own configuration. 





