'''
This script intends to slice the long audio files (quartets) into short segments with their associated gt annotations
to make them suitable for the MuSingers multif0 estimation method
'''


import sox
import numpy as np
import mir_eval

import config
import utils

import os



def find_nearest_arg(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# 1. Load gound truth
# 2. Locate frames close to steps of 10 seconds
# 3. Save exact time of these frames (in secs)
# 4. Iteratively cut audio files using SoX and store segments of gt in individual files: filename_segX.wav/txt/npy
#    To cut the gt we need to handle the timestamps: subtract the first value from all values

annotations_path = '/scratch/hc2945/data/test_data'
export_path = '/scratch/hc2945/data/musingers_data'
audiopath = '/scratch/hc2945/data/audiomixtures'

data_splits_path = os.path.join(config.data_save_folder, 'data_splits.json')
_, _, test_set = utils.load_json_data(data_splits_path)

for audiofile in test_set:

    if not audiofile.endswith('wav'): continue

    #piece = audiofile.split('_')[0]

    ref_times, ref_freqs = mir_eval.io.load_ragged_time_series(
        os.path.join(annotations_path, audiofile.replace('wav', 'csv'))
    )

    time_annot = ref_times

    #[time_annot.append(annot[i][0]) for i in range(len(annot))]

    lensec = time_annot[-1] + (time_annot[-1] - time_annot[-2])
    slices = np.arange(start=0, stop=lensec, step=10) # cut 10 seconds segments
    idx_slice = 0

    for i in range(len(slices)-1):

        idx_slice+=1

        idx1, sec1 = find_nearest_arg(time_annot, slices[i])
        idx2, sec2 = find_nearest_arg(time_annot, slices[i+1])

        store_times = ref_times[idx1:idx2]
        store_freqs = ref_freqs[idx1:idx2]

        first_val = store_times[0]
        for i in range(len(store_times)):
            store_times[i] = store_times[i] - first_val

        output_annot_fname = os.path.join(export_path, "{}_{}.csv".format(audiofile.replace('.wav', ''), idx_slice))
        utils.save_multif0_output(store_times, store_freqs, output_annot_fname)


        output_audiofile = '{}_{}.wav'.format(audiofile[:-4], idx_slice)
        tfm = sox.Transformer()
        tfm.trim(sec1, sec2)
        tfm.build(
            os.path.join(audiopath, audiofile),
            os.path.join(export_path,'{}_{}.wav'.format(audiofile[:-4], idx_slice))
        )
    print("All slices from {} have been exported".format(audiofile))





