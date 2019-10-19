'''In this script we implement a convolution between an impulse response and a set of audio signals to add the reverb effect to them.
This is done in the context of a multi-f0 estimation system, and we also modify pitch annotations according to the group delay
introduced by this kind of effect.
'''
import numpy as np
import matplotlib.pyplot as plt

import muda
import jams


audio_folder = ''
jams_folder = ''

def combine_pitch_curves()
