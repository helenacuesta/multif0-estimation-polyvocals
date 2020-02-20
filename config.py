'''
Config file
Dataset structure is created here
'''

import itertools
import numpy as np


'''Paths
'''
audio_save_folder = '/scratch/hc2945/data/audiomixtures/'
data_save_folder = '/scratch/hc2945/data/mf0annotations/'

bq_audio_folder = '/scratch/hc2945/multif0/'

csd_folder = '/scratch/hc2945/data/CSD/'
ecs_folder = '/scratch/hc2945/data/ECS/'
dcs_folder = '/scratch/hc2945/data/DagstuhlChoirSet_V1.0/audio_wav_22050_mono/'


'''All variables and parameters related to the dataset creation
'''

dataset = dict()
dataset['CSD'] = dict()
dataset['DCS'] = dict()
dataset['ECS'] = dict()
dataset['BC'] = dict()
dataset['BSQ'] = dict()

augmentation_idx = ['0_', '1_', '2_', '3_', '4_']

'''Choral Singing Dataset
'''
csd_songs = ['CSD_ER', 'CSD_LI', 'CSD_ND']

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



'''ESMUC ChoralSet (divided by songs for convenience)
'''

''' Der Greis
'''

ecs_dg = ['DG_take1', 'DG_take2', 'DG_take3_mixed', 'DG_take4_mixed']

singers_ecs_dg = [
    'S1', 'S2', 'S3', 'S4',
    'A1', 'A2', 'A3',
    'T1', 'T2', 'T3',
    'B1', 'B2']

dataset['ECS']['DG_singers'] = singers_ecs_dg

dataset['ECS']['DG_songs'] = []
for song in ecs_dg:
    for idx in augmentation_idx:
        dataset['ECS']['DG_songs'].append(idx + song)

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
dataset['ECS']['DG_combos'] = combos

''' Die Himmel
'''

ecs_dh = ['DH1_take2', 'DH2_take2']

singers_ecs_dh = [
    'S1-1', 'S2-1', 'S3-2', 'S4-2', 'S5-2',
    'A1', 'A2',
    'T1-1', 'T2-1', 'T3-2',
    'B1', 'B2'
]

dataset['ECS']['DH_singers'] = singers_ecs_dh

dataset['ECS']['DH_songs'] = []
for song in ecs_dh:
    for idx in augmentation_idx:
        dataset['ECS']['DH_songs'].append(idx + song)

sop = np.arange(1, 5 + 1)
alto = np.arange(1, 2 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['DH_combos'] = combos

''' Seele Christi
'''

ecs_sc = ['SC1_take1', 'SC1_take2', 'SC1_take3_mixed', 'SC2_take1',
          'SC2_take2', 'SC2_take3_mixed', 'SC3_take1', 'SC3_take2_mixed']


singers_ecs_sc = [
    'S1', 'S2', 'S3', 'S4', 'S5',
    'A1', 'A2',
    'T1', 'T2', 'T3',
    'B1', 'B2'
]

dataset['ECS']['SC_singers'] = singers_ecs_sc

dataset['ECS']['SC_songs'] = []
for song in ecs_sc:
    for idx in augmentation_idx:
        dataset['ECS']['SC_songs'].append(idx + song)

sop = np.arange(1, 5 + 1)
alto = np.arange(1, 2 + 1)
ten = np.arange(1, 3 + 1)
bass = np.arange(1, 2 + 1)

combos = []
for s in sop:
    for a in alto:
        for t in ten:
            for b in bass:
                combos.append(np.array([s, a, t, b]))

combos = np.array(combos, dtype=np.int32)
dataset['ECS']['SC_combos'] = combos


'''Dagstuhl ChoirSet
'''

dcs_settings = ['All', 'QuartetA', 'QuartetB']

singers_QB = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']
singers_QA = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']
singers_all_dyn = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']
#singers_all_lrx = ['S1_LRX', 'S2_LRX', 'A1_LRX', 'A2_LRX', 'T1_LRX', 'T2_LRX', 'B1_LRX', 'B2_LRX']

'''dcs_songs = [
    'DLI_All_Take1_', 'DLI_All_Take2_', 'DLI_All_Take3_',
    'DLI_QuartetA_Take1_', 'DLI_QuartetA_Take2_', 'DLI_QuartetA_Take3_', 'DLI_QuartetA_Take4_', 'DLI_QuartetA_Take5_',
    'DLI_QuartetA_Take6_', 'DLI_QuartetB_Take1_', 'DLI_QuartetB_Take2_', 'DLI_QuartetB_Take2_', 'DLI_QuartetB_Take3_',
    'DLI_QuartetB_Take4_', 'DLI_QuartetB_Take5_']
'''

# no combos because these are quartets (inside the full choir)
dcs_songs_fc = ['DCS_LI_FullChoir_Take01', 'DCS_LI_FullChoir_Take02', 'DCS_LI_FullChoir_Take03']
dcs_singers_fc = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

dataset['DCS']['FC_singers'] = dcs_singers_fc
dataset['DCS']['FC_songs'] = []
for song in dcs_songs_fc:
    for idx in augmentation_idx:
        dataset['DCS']['FC_songs'].append(idx + song)



dcs_songs_qa = ['DCS_LI_QuartetA_Take01', 'DCS_LI_QuartetA_Take02', 'DCS_LI_QuartetA_Take03',
                              'DCS_LI_QuartetA_Take04', 'DCS_LI_QuartetA_Take05', 'DCS_LI_QuartetA_Take06']
dcs_singers_qa = ['S2_DYN', 'A1_DYN', 'T1_DYN', 'B1_DYN']

dataset['DCS']['QA_singers'] = dcs_singers_qa
dataset['DCS']['QA_songs'] = []

for song in dcs_songs_qa:
    for idx in augmentation_idx:
        dataset['DCS']['QA_songs'].append(idx + song)


dcs_songs_qb = ['DCS_LI_QuartetB_Take01', 'DCS_LI_QuartetB_Take02', 'DCS_LI_QuartetB_Take03',
               'DCS_LI_QuartetB_Take04', 'DCS_LI_QuartetB_Take05']
dcs_singers_qb = ['S1_DYN', 'A2_DYN', 'T2_DYN', 'B2_DYN']

dataset['DCS']['QB_singers'] = dcs_singers_qb
dataset['DCS']['QB_songs'] = []
for song in dcs_songs_qb:
    for idx in augmentation_idx:
        dataset['DCS']['QB_songs'].append(idx + song)

''' Bach Chorales
'''
dataset['BC']['songs'] = []

'''
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


singers_per_section = 2
x = np.arange(1, singers_per_section + 1).astype(np.int32)
combos_lrx = [p for p in itertools.product(x, repeat=4)]
dataset['DCS']['All']['lrx_combos'] = combos_lrx

dataset['DCS']['QuartetA']['combos'] = np.array([2, 1, 1, 1], dtype=np.int32)
dataset['DCS']['QuartetB']['combos'] = np.array([1, 2, 2, 2], dtype=np.int32)'''



'''Training parameters
'''
SAMPLES_PER_EPOCH = 2048
NB_EPOCHS = 100
NB_VAL_SAMPLES = 128