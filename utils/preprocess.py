import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
import scipy.signal as signal
import numpy as np
import wandb
import librosa
import seaborn as sns

# Add rasta_py to the path
import sys

sys.path.append("./rasta_py")
from rasta import rastaplp


def analyse_distribution(data):
    # For all folds, plot a sns.distplot of the plps_with_derivatives. For each fold, plot a different color
    folds = np.unique(data["fold"])
    fig, ax = plt.subplots()
    for f in folds:
        data_fold = data[data["fold"] == f]["plps_with_derivatives"].mean(axis=0)
        sns.histplot(data_fold, label="Fold " + str(f), ax=ax)
    plt.title("Distribution of the plps_with_derivatives")
    plt.legend()
    plt.savefig("./results/plps_with_derivatives_distribution.png")
    plt.close()


def read_data(path_to_data, wandb_flag=False):
    # List all the folders in the path_to_data directory
    folders = os.listdir(path_to_data)

    # Initialize lists to store file paths and corresponding labels
    file_paths = []
    labels = []

    # Loop through the folders
    for folder in folders:
        # Get the full folder path
        folder_path = os.path.join(path_to_data, folder)

        # Check if the folder is a directory
        if os.path.isdir(folder_path):
            # Get the label from the folder name
            label = folder

            # Get the file names in the folder
            files = os.listdir(folder_path)

            # Loop through the files
            for file in files:
                # Get the file path
                file_path = os.path.join(folder_path, file)

                # Append the file path and label to the lists
                file_paths.append(file_path)
                labels.append(label)

    # Create a dataframe with all the data
    data = pd.DataFrame({"file_path": file_paths, "label": labels})

    # Read the .wav files and store the signals and sampling rates
    data["sr"], data["signal"] = zip(*data["file_path"].map(wavfile.read))

    # ============== Downsampling ===========

    # Downsample the signals to 16kHz using the resample function
    # data["signal"] = data["signal"].apply(
    #     lambda x: signal.resample(x, int(x.shape[0] * 16000 / data["sr"][0]))
    # )
    # data["sr"] = 16000

    # Plot in wandb a random signal
    random_signal = np.random.randint(0, len(data))
    if wandb_flag:
        wandb.log(
            {
                "signal": wandb.Audio(
                    data["signal"][random_signal],
                    sample_rate=data["sr"][random_signal],
                    caption="Random raw signal " + data["label"][random_signal],
                )
            }
        )

    # Generate ten folds for cross-validation. Store them as a label in the dataframe
    data["fold"] = data.groupby("label").cumcount() % 10

    # Check the len of each signal
    data["len"] = data["signal"].apply(len)
    data["max"] = data["signal"].apply(np.abs).apply(np.max)

    # ============== Preprocessing ===========
    # Normalize the signals applying the normalize_audio function with receives as input the signal
    data["norm_signal"] = data.apply(lambda x: normalize_audio(x["signal"]), axis=1)

    # Plot in wandb a random signal
    if wandb_flag:
        wandb.log(
            {
                "norm_signal": wandb.Audio(
                    data["norm_signal"][random_signal],
                    sample_rate=data["sr"][random_signal],
                    caption="Random normalised signal" + data["label"][random_signal],
                )
            }
        )

    return data


# Normalize the signals
def normalize_audio(audio_data, max=None):
    if max is None:
        max_value = np.max(np.abs(audio_data))
    else:
        max_value = max
    normalized_data = audio_data / max_value
    return normalized_data


def frame_audio(audio_data, frame_size_ms, hop_size_percent, sample_rate):
    frame_size = int(frame_size_ms * sample_rate / 1000)
    hop_size = int(frame_size * hop_size_percent / 100)
    num_samples = len(audio_data)
    num_frames = int(np.ceil((num_samples - frame_size) / hop_size) + 1)

    framed_data = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = audio_data[start:end]
        if len(frame) < frame_size:
            frame = np.append(frame, np.zeros(frame_size - len(frame)))
        else:  # TODO: preguntar si descartamos la que se queda corta o la relleno con 0s
            frame = np.hanning(frame_size) * frame
        framed_data[i] = frame

    return framed_data


def cmn(mfcc):
    # Compute the mean of each feature
    mean = np.mean(mfcc, axis=0)

    # Compute the variance of each feature
    var = np.var(mfcc, axis=0)

    # Noramlize
    cmn_mfcc = (mfcc - mean) / np.sqrt(var)

    return cmn_mfcc


def extract_rasta_plp_with_derivatives(audio, sample_rate, frame_length_ms, n_plps=10):
    # Compute the PLPs
    plps = rastaplp(
        audio,
        fs=sample_rate,
        win_time=frame_length_ms * 1e-3,
        hop_time=int(frame_length_ms / 2) * 1e-3,
        modelorder=n_plps,
    )

    # Compute the derivatives
    plps_d1 = librosa.feature.delta(plps, order=1)
    plps_d2 = librosa.feature.delta(plps, order=2)

    # Concatenate the features
    plps = np.concatenate((plps, plps_d1, plps_d2), axis=0)

    # TODO: Preguntar: es necesario separar por frames? o se puede hacer directamente com heatmaps?
    return plps.T


def extract_mfcc_with_derivatives(audio, sample_rate, frame_length_ms, n_mfcc=10):
    frame_length = int(sample_rate * frame_length_ms * 1e-3)  # Convert ms to samples
    hop_length = int(frame_length / 2)  # 50% overlap
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    # Apply hanning windows
    frames = frames * np.hanning(frame_length)[:, None]

    # N_fft is the next number in power of 2 of the frame size
    n_fft = 2 ** (int(np.log2(frames.shape[1])) + 1)
    # Compute MFCC for each frame
    mfccs = []
    for frame in frames:
        mfcc = librosa.feature.mfcc(y=frame, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft)
        mfccs.append(mfcc)

    mfccs = np.hstack(mfccs)

    # Normalize the MFCCs
    mfccs = cmn(mfccs)

    # Compute derivatives
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

    # Concatenate the features
    mfcc_features = np.concatenate((mfccs, mfccs_delta, mfccs_delta2))

    # TODO: Preguntar: es necesario separar por frames? o se puede hacer directamente com heatmaps?
    return mfcc_features.T
