import training_temp
import numpy as np


dat = training_temp.Data(
    '/scratch/hc2945/data/features_targets/data_splits.json',
    '/scratch/hc2945/data/features_targets/',
    (360, 50), 16, 300, 30)


durations = 0
for fname in dat.train_files:
    out = np.load(fname[1])
    durations += out.shape[1] * 256 / 22050.0

print("Total audio duration for TRAINING SET: {}".format(durations))


durations = 0
for fname in dat.validation_files:
    out = np.load(fname[1])
    durations += out.shape[1] * 256 / 22050.0

print("Total audio duration for VALIDATION SET: {}".format(durations))

durations = 0
for fname in dat.test_files:
    out = np.load(fname[1])
    durations += out.shape[1] * 256 / 22050.0

print("Total audio duration for TEST SET: {}".format(durations))