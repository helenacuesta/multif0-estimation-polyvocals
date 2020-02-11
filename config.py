'''
Config file
Dataset structure is created here
'''

import itertools
import numpy as np


'''Paths
'''
audio_save_folder = '/scratch/hc2945/multif0/AudioMixtures/'
data_save_folder = '/scratch/hc2945/multif0/AudioMixtures/'

bq_audio_folder = '/scratch/hc2945/multif0/'

csd_folder = '/scratch/hc2945/multif0/CSD/individuals/'
ecs_folder = '/scratch/hc2945/multif0/ECS/individuals/'
dcs_folder = '/scratch/hc2945/multif0/DCS/individuals/'

'''All variables and parameters related to the dataset creation
'''

dataset = dict()
dataset['CSD'] = dict()
dataset['DCS'] = dict()
dataset['ECS'] = dict()

augmentation_idx = ['0_', '1_', '2_', '3_', '4_']

'''Choral Singing Dataset
'''
csd_songs = ['CSD_ER_', 'CSD_LI_', 'CSD_ND_']

singers_csd = [
        'soprano_1', 'soprano_2', 'soprano_3', 'soprano_4',
        'alto_1', 'alto_2', 'alto_3', 'alto_4',
        'tenor_1', 'tenor_2', 'tenor_3', 'tenor_4',
        'bass_1', 'bass_2', 'bass_3', 'bass_4']

dataset['CSD']['songs'] = []
for song in csd_songs:
    for idx in augmentation_idx:
        dataset['CSD']['songs'].append(idx + song)

dataset['CSD']['singers'] = singers_csd

singers_per_section = 4
x = np.arange(1, singers_per_section + 1).astype(np.int32)
combos = [p for p in itertools.product(x, repeat=4)]
dataset['CSD']['combos'] = combos

'''ESMUC ChoralSet
'''
ecs_songs = ['WU_take1_', 'WU_take2_', 'DG_take1_', 'DG_take2_', 'DG_take3_mixed_', 'DG_take4_mixed_']

singers_ecs = [
    'S1', 'S2', 'S3', 'S4',
    'A1', 'A2', 'A3',
    'T1', 'T2', 'T3',
    'B1', 'B2'
]

dataset['ECS']['songs'] = []
for song in ecs_songs:
    for idx in augmentation_idx:
        dataset['ECS']['songs'].append(idx + song)

dataset['ECS']['singers'] = singers_ecs

sop = np.arange(1, 4 + 1)
alto = np.arange(1, 3 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['combos'] = combos

'''Dagstuhl ChoirSet
'''

dcs_settings = ['All', 'QuartetA', 'QuartetB']

singers_QB = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']
singers_QA = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']
singers_all_dyn = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']
#singers_all_lrx = ['S1_LRX', 'S2_LRX', 'A1_LRX', 'A2_LRX', 'T1_LRX', 'T2_LRX', 'B1_LRX', 'B2_LRX']

dcs_songs = [
    'DLI_All_Take1_', 'DLI_All_Take2_', 'DLI_All_Take3_',
    'DLI_QuartetA_Take1_', 'DLI_QuartetA_Take2_', 'DLI_QuartetA_Take3_', 'DLI_QuartetA_Take4_', 'DLI_QuartetA_Take5_',
    'DLI_QuartetA_Take6_', 'DLI_QuartetB_Take1_', 'DLI_QuartetB_Take2_', 'DLI_QuartetB_Take2_', 'DLI_QuartetB_Take3_',
    'DLI_QuartetB_Take4_', 'DLI_QuartetB_Take5_']

for setting in dcs_settings:
    dataset['DCS'][setting] = dict()
    dataset['DCS'][setting]['songs'] = []
    dataset['DCS'][setting]['singers'] = dict()

#dataset['DCS']['All']['singers']['lrx'] = singers_all_lrx
dataset['DCS']['All']['singers']['dyn'] = singers_all_dyn
dataset['DCS']['QuartetA']['singers'] = singers_QA
dataset['DCS']['QuartetB']['singers'] = singers_QB

combo_dyn = np.array([1, 2, 2, 2], dtype=np.int32)
dataset['DCS']['All']['dyn_combos'] = combo_dyn

'''
singers_per_section = 2
x = np.arange(1, singers_per_section + 1).astype(np.int32)
combos_lrx = [p for p in itertools.product(x, repeat=4)]
dataset['DCS']['All']['lrx_combos'] = combos_lrx
'''

dataset['DCS']['QuartetA']['combos'] = np.array([2, 1, 1, 1], dtype=np.int32)
dataset['DCS']['QuartetB']['combos'] = np.array([1, 2, 2, 2], dtype=np.int32)


for song in dcs_songs:
    for idx in augmentation_idx:
        if 'All' in song:
            dataset['DCS']['All']['songs'].append(idx + song)

        elif 'QuartetA' in song:
            dataset['DCS']['QuartetA']['songs'].append(idx + song)

        elif 'QuartetB' in song:
            dataset['DCS']['QuartetB']['songs'].append(idx + song)

        else:
            print("Wrong setting for the DCS")

'''Training parameters
'''
SAMPLES_PER_EPOCH = 2048
NB_EPOCHS = 100
NB_VAL_SAMPLES = 128