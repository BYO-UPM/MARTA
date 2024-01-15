"""This script will do the following:

1. Read the pataka audios from labeled/NeuroVozPATAKA/
    1.1 They are stored in two folders: "HC" and "PD" (healthy controls and parkinson's disease) and in WAV format.
2. For each audio, we have to detect how many times "pataka" is said. To do so, we will do the envelope of the audio and then we will detect the peaks. Each peak is either a "pa, ta or ka". We will use the librosa library to do so.
3. We will then generate a .txt file that will say "pataka" for each 3 peaks detected.
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Read the pataka audios from labeled/NeuroVozPATAKA/
#     1.1 They are stored in two folders: "HC" and "PD" (healthy controls and parkinson's disease) and in WAV format.
path_to_pataka = "labeled/NeuroVoz/PATAKA/"
path_to_save = "labeled/NeuroVoz/PATAKA/"
pataka_audios = []
pataka_labels = []
for folder in os.listdir(path_to_pataka):
    if folder == "HC":
        for file in os.listdir(path_to_pataka + folder):
            if file.endswith(".wav"):
                pataka_audios.append(path_to_pataka + folder + "/" + file)
                pataka_labels.append(0)
    elif folder == "PD":
        for file in os.listdir(path_to_pataka + folder):
            if file.endswith(".wav"):
                pataka_audios.append(path_to_pataka + folder + "/" + file)
                pataka_labels.append(1)

# 2. For each audio, we have to detect how many times "pataka" is said. To do so, we will do the envelope of the audio and then we will detect the peaks. Each peak is either a "pa, ta or ka". We will use the librosa library to do so.

# Calculate the envelope and plot the wav signal and its envelope

envelopes = []
x_array = []
audios = []
sr = 16000


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin] < s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax] > s_mid]

    # global min of dmin-chunks of locals min
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # global max of dmax-chunks of locals max
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    return lmin, lmax


for path in pataka_audios:
    audio = librosa.load(path, sr=sr)[0]
    envelope_idx = hl_envelopes_idx(audio)
    upper_envelope = audio[envelope_idx[1]]
    time_vector = np.arange(len(audio))[envelope_idx[1]]
    lower_envelope = audio[envelope_idx[0]]
    audios += [audio]
    envelopes += [upper_envelope]
    x_array += [time_vector]

y = audios[0]
envelope = envelopes[0]

# Time vectors for plotting
t = np.arange(len(y))

# Plotting the audio signal and its envelope
plt.figure(figsize=(15, 6))

# Plot the original audio signal
plt.plot(t[:100], y[:100], label="Audio Signal")

# Plot the envelope of the audio signal
plt.plot(x_array[0][:100], envelope[:100], label="Envelope", color="red")

plt.title("Audio Signal and its Envelope")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()

plt.show()
