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

def read_annotations(path_to_jams):
    annotation = jams.load(path_to_jams)
    return annotation


def median_group_delay(y, sr, n_fft=2048, rolloff_value=-24):
    """Compute the average group delay for a signal

    Parameters
    ----------
    y : np.ndarray
        the signal

    sr : int > 0
        the sampling rate of `y`

    n_fft : int > 0
        the FFT window size

    rolloff_value : int > 0
        If provided, only estimate the groupd delay of the passband that
        above the threshold, which is the rolloff_value below the peak
        on frequency response.

    Returns
    -------
    mean_delay : float
        The mediant group delay of `y` (in seconds)

    """
    if rolloff_value > 0:
        # rolloff_value must be strictly negative
        rolloff_value = -rolloff_value

    # frequency response
    _, h_ir = freqz(y, a=1, worN=n_fft, whole=False, plot=None)

    # convert to dB(clip function avoids the zero value in log computation)
    power_ir = 20 * np.log10(np.clip(np.abs(h_ir), 1e-8, 1e100))

    # set up threshold for valid range
    threshold = max(power_ir) + rolloff_value

    _, gd_ir = group_delay((y, 1), n_fft)

    return np.median(gd_ir[power_ir > threshold]) / sr



def read_data(audio_folder, audio_fname, ir_fname, fs):

    y_sig, sr_sig = librosa.core.load(
        os.path.join(audio_folder, audio_fname),sr=fs)

    y_ir, sr_ir = librosa.core.load(ir_fname, sr=fs)

    return y_sig, sr_sig, y_ir, sr_ir


def convolove_with_ir(y_sig, y_ir, mode='same'):

    y_conv = fftconvolve(y_sig, y_ir, mode=mode)

    return y_conv



# Main code
audio_folder = '/scratch/hc2945/multif0/AudioMixtures'
new_audio_folder = os.path.join(audio_folder, 'ir_rev')
ir_audio = './IR_greathall.wav'

fs=22050

'''call functions here'''

fnlist = ['2_DG_take1_3_3_2_2.wav',
          '3_CSD_ER_3_3_2_4.wav', '0_CSD_ND_2_2_1_4.wav',
          '1_CSD_ER_3_2_2_1.wav', '1_WU_take1_1_2_1_1.wav',
          '2_CSD_ND_1_1_1_1.wav', '2_DG_take1_1_3_1_1.wav',
          '4_WU_take1_1_3_3_1.wav', '4_CSD_ND_2_2_2_4.wav',
          '3_DG_take2_1_2_3_1.wav', '3_CSD_LI_4_4_4_1.wav']



for fn in fnlist:

    y_sig, sr_sig, y_ir, sr_ir = read_data(audio_folder, fn, ir_audio, fs)
    y_conv = convolove_with_ir(y_sig, y_ir)

    wavfile.write(os.path.join(new_audio_folder, fn), rate=fs, data=y_conv)

    gd = median_group_delay(y_sig, fs)

    print("GroupDelay for {} is {}.".format(fn, gd))








