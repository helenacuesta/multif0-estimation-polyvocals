'''In this script we implement a convolution between an impulse response and a set of audio signals to add the reverb effect to them.
This is done in the context of a multi-f0 estimation system, and we also modify pitch annotations according to the group delay
introduced by this kind of effect.

IMPORTANT NOTE: Most of this code comes from the muda package repo, written by Brian McFee

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, freqz, group_delay
from scipy.io import wavfile

import jams
import muda
import librosa

import os


def deform_times(annotation, ir_groupdelay):
    # Deform time values for all annotations.

    for obs in annotation.pop_data():
        # Drop obervation that fell off the end

        if obs.time + ir_groupdelay > annotation.duration:
            # Drop the annotation if its delayed onset out of the range of duration
            annotation = annotation.slice(0, annotation.duration, strict=False)
        else:
            # truncate observation's duration if its offset fell off the end of annotation
            if obs.time + obs.duration + ir_groupdelay > annotation.duration:
                deformed_duration = annotation.duration - obs.time - ir_groupdelay
            else:
                deformed_duration = obs.duration

        annotation.append(
            time=obs.time + ir_groupdelay,
            duration=deformed_duration,
            value=obs.value,
            confidence=obs.confidence,
        )

def f0_annotation_shift(f0_vec, gd):
    '''

    :param f0_vec: vector with the original f0 annotations, [timestamp, f0 value]
    :param gd: group delay estimated by muda
    :return: vector with the modified f0 annotations, shifted in time according to the group delay
    '''
    # see if it's this straightforward...I don't think so
    ts = f0_vec[:, 0]
    ts_sh = ts + gd
    f0_vec[:, 0] = ts_sh

    return f0_vec


def read_annotations(path_to_jams):
    annotation = jams.load(path_to_jams)
    return annotation

def read_impulse(ir_path, fs):

    ir_sig, ir_sr = librosa.core.load(
        ir_path, sr=fs)

    return ir_sig, ir_sr

def read_audio(audio_folder, audio_fname, fs):

    y_sig, sr_sig = librosa.core.load(
        os.path.join(audio_folder, audio_fname),sr=fs)

    return y_sig, sr_sig


def conv_with_ir(y_sig, y_ir, mode='same'):

    y_conv = fftconvolve(y_sig, y_ir, mode=mode)

    return y_conv



# Main code
audio_folder = '/scratch/hc2945/multif0/AudioMixtures'
new_audio_folder = os.path.join(audio_folder, 'ir_rev')
ir_audio = './IR_greathall.wav'

fs=22050

fnlist = ['2_DG_take1_3_3_2_2.wav',
          '3_CSD_ER_3_3_2_4.wav', '0_CSD_ND_2_2_1_4.wav',
          '1_CSD_ER_3_2_2_1.wav', '1_WU_take1_1_2_1_1.wav',
          '2_CSD_ND_1_1_1_1.wav', '2_DG_take1_1_3_1_1.wav',
          '4_WU_take1_1_3_3_1.wav', '4_CSD_ND_2_2_2_4.wav',
          '3_DG_take2_1_2_3_1.wav', '3_CSD_LI_4_4_4_1.wav']


'''call functions here'''

# test different rolloff values
# rolloff = np.arange(-15, -30)

y_ir, sr_ir = read_impulse(ir_audio, fs=fs)
gd = muda.deformers.median_group_delay(y_ir, fs, n_fft=2048, rolloff_value=-24)
print("GroupDelay for GreatHall from ISOPHONICS is {}.".format(gd))


for fn in fnlist:

    # read original audio file
    y_sig, sr_sig = read_audio(audio_folder, fn, fs)

    # read IR
    y_ir, _ = read_impulse(ir_audio, fs=fs)

    # convolution between audio and IR
    y_conv = conv_with_ir(y_sig, y_ir)

    # save output of the convolution
    wavfile.write(os.path.join(new_audio_folder, fn), rate=fs, data=y_conv)












