"""
Phoneme-Based Audio Feature Extraction and Classification Dataset Preparation

This script is designed for preparing datasets for phoneme-based audio feature extraction and classification. 
It focuses on processing audio files, extracting relevant features, and aligning these with phonetic 
annotations from TextGrid files. The primary aim is to facilitate the training of machine learning models 
for tasks like speech and language pathology analysis, particularly in the context of Parkinson's Disease.

Main Components:
1. Data Preparation: Reads and processes audio files from specified directories, aligning audio 
   signals with corresponding phoneme annotations from TextGrid files.
2. Feature Extraction: Supports extraction of various audio features such as PLPs, MFCCs, and spectrograms.
3. Dataset Creation: Constructs a PyTorch Dataset with extracted features, labels, and additional metadata.
4. DataLoader Generation: Splits the dataset into training, validation, and testing sets and prepares 
   DataLoaders for model training and evaluation.

Functions:
- collapse_to_most_repeated: Collapses a matrix into a vector with the most repeated string.
- match_phonemes: Align phoneme annotations with audio signals.
- read_neurovoz, read_albayzin: Read and process specific datasets (Neurovoz, Albayzin).
- normalize_audio: Normalize audio signals.

Usage:
- The script is expected to be used as part of a larger pipeline for training speech and language pathology models.
- It requires the directory paths for the audio files and TextGrid files, along with specified hyperparameters.

Output:
- Prepared PyTorch DataLoaders with processed audio features and labels, ready for model training and evaluation.

Requirements:
- The script assumes a specific directory structure and file naming convention for the audio and TextGrid files.
- Libraries: torch, numpy, pandas, librosa, textgrids.

Author: Guerrero-López, Alejandro
Date: 25/01/2024

Note:
- Ensure that the required data directories and files are correctly structured and accessible.
- Hyperparameter configuration may need adjustment based on specific requirements of the dataset and the model.
"""

import torch
import numpy as np
import os
import pandas as pd
import librosa
import textgrids as tg


# Function to collapse the matrix into a 24x1 vector with the most repeated string
def collapse_to_most_repeated(row):
    from collections import Counter

    collapsed_vector = []
    for row_values in row:
        # Count the occurrences of each string in the row
        count = Counter(row_values)
        # Find the most common string
        most_common = count.most_common(1)
        if most_common:
            collapsed_vector.append(most_common[0][0])
        else:
            collapsed_vector.append(None)  # Handle the case when there are no strings
    return collapsed_vector


class Dataset_AudioFeatures(torch.utils.data.Dataset):
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.spectrogram = self.hyperparams["spectrogram"]

        # Check if the data has been already processed and saved
        name_save = (
            "local_results/data_frame_with_phonemes"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + ".pkl"
        )

        if os.path.exists(name_save):
            self.data = pd.read_pickle(name_save)["data"]
        else:
            self.data = self.read_dataset()

    def __len__(self):
        return len(self.data)

    def read_gita(self):
        file_paths = []
        labels = []
        id_patient = []
        texts = []
        phonemes = []

        datapath = "/media/my_ftp/BasesDeDatos_Voz_Habla/PC-GITA/ "

        for root, dirs, files in os.walk(datapath):
            for file in files:
                # If the file does not end with .wav, skip it
                if not file.endswith(".wav"):
                    continue
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                # Each file is named as follows: XXXXXXXXAC0001_text.wav where XXXXXXXX is not important, A or AC is the condition (A =PD, AC = HC), the id patient is the four digits, and text is the text
                # ID patient is always a 4 digit number
                keys = file.split(".")[0].split("_")
                # Remove duplicated keys
                keys = set(keys)
                for key in keys:
                    # if they key contains "AC" then it is the label
                    if "AVPEPUDEAC" in key:
                        labels.append(0)
                        id_patient.append(int("0" + key[-4:]))
                    elif "AVPEPUDEA" in key:
                        labels.append(1)
                        id_patient.append(int("1" + key[-4:]))
                    # in any other case, it should be the name of the "task" performed
                    else:
                        texts.append(key)

                # Read the text grid file
                tg_file = os.path.join(root, file).replace(".wav", ".TextGrid")
                # Check if the file exists
                if not os.path.exists(tg_file):
                    print("File does not exist: ", tg_file)
                    phonemes.append(None)
                    continue
                tg_file = tg.TextGrid(tg_file)
                phonemes.append(tg_file["speaker : phones"])

        # Generate a dataframe with all the data
        data = pd.DataFrame(
            {
                "file_path": file_paths,
                "label": labels,
                "text": texts,
                "phonemes": phonemes,
                "id_patient": id_patient,
            }
        )
        # Drop na
        data = data.dropna()
        # sort by id_patient
        data = data.sort_values(by=["id_patient"])
        # reset index
        data = data.reset_index(drop=True)

        return data

    def read_neurovoz(self):
        file_paths = []
        labels = []
        id_patient = []
        texts = []
        phonemes = []

        datapath = "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/neurovoz_htk_forced_alignment/"

        for root, dirs, files in os.walk(datapath):
            for file in files:
                # If the file does not end with .wav, skip it
                if not file.endswith(".wav"):
                    continue
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

                # ID patient is always a 4 digit number
                keys = file.split(".")[0].split("_")
                # Remove duplicated keys
                keys = set(keys)
                for key in keys:
                    # if the key is PD or HC, then it is the label
                    if key == "PD":
                        labels.append(1)
                    elif key == "HC":
                        labels.append(0)
                    # if the key is a number, then it is the id_patient
                    elif key.isdigit() and len(key) == 4:
                        id_patient.append(int(key))
                    # in any other case, it should be the name of the "task" performed
                    else:
                        texts.append(key)

                # Read the text grid file
                tg_file = os.path.join(root, file).replace(".wav", ".TextGrid")
                # Check if the file exists
                if not os.path.exists(tg_file):
                    print("File does not exist: ", tg_file)
                    phonemes.append(None)
                    continue
                tg_file = tg.TextGrid(tg_file)
                phonemes.append(tg_file["speaker : phones"])

        # Generate a dataframe with all the data
        data = pd.DataFrame(
            {
                "file_path": file_paths,
                "label": labels,
                "text": texts,
                "phonemes": phonemes,
                "id_patient": id_patient,
            }
        )
        # Drop na
        data = data.dropna()
        # sort by id_patient
        data = data.sort_values(by=["id_patient"])
        # reset index
        data = data.reset_index(drop=True)

        return data

    def read_albayzin(self):
        file_paths = []
        labels = []
        id_patient = []
        texts = []
        phonemes = []

        datapath_wav = "/media/my_ftp/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/albayzin_htk_forced_alignment"

        i = 0
        for file in os.listdir(datapath_wav):
            # If the file does not end with .wav, skip it
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(datapath_wav, file)
            file_paths.append(file_path)
            # Each file is named as follows: aabbXXXX.wav where aa is the id_patient, bb is the train/test partition, XXXX is the text
            labels.append(0)  # In albayzin all are healthy
            id_patient.append(file.split(".")[0][0:4])
            texts.append(file.split(".")[0][4:])

            # Read the text grid file
            tg_file = os.path.join(datapath_wav, file.split(".")[0] + ".TextGrid")
            # Check if the file exists
            if not os.path.exists(tg_file):
                print("File does not exist: ", tg_file)
                i += 1
                phonemes.append(None)
                continue
            tg_file = tg.TextGrid(tg_file)
            phonemes.append(tg_file["speaker : phones"])

        print("Total WAV files: ", len(os.listdir(datapath_wav)))
        print("Total TextGrid files: ", len(os.listdir(datapath_wav)))
        print("Total files without textgrid: ", i)

        # Generate a dataframe with all the data
        data = pd.DataFrame(
            {
                "file_path": file_paths,
                "label": labels,
                "text": texts,
                "phonemes": phonemes,
                "id_patient": id_patient,
            }
        )
        # Drop na
        data = data.dropna()
        # sort by id_patient
        data = data.sort_values(by=["id_patient"])
        # reset index
        data = data.reset_index(drop=True)

        return data

    def match_phonemes(self, phonemes, signal, sr):
        phoneme_labels = []

        for timestamp in range(len(signal)):
            # Convert the timestamp to seconds
            timestamp_seconds = timestamp / sr

            # Find the Interval object that corresponds to the current timestamp
            matching_interval = None
            for interval in phonemes:
                if interval.xmin <= timestamp_seconds <= interval.xmax:
                    matching_interval = interval
                    break
            # Append the phoneme label to the phoneme_labels list (or None if no match is found)
            if matching_interval is not None:
                phoneme_labels.append(matching_interval.text)
            else:
                phoneme_labels.append(None)

        return phoneme_labels

    def read_dataset(self):
        print("Reading the data...")

        data_alb = self.read_albayzin()
        # add a column with the dataset name
        data_alb["dataset"] = "albayzin"

        data_neuro = self.read_neurovoz()
        data_neuro["dataset"] = "neurovoz"

        data_gita = self.read_gita()
        data_gita["dataset"] = "gita"
        # Sum 1000 to all id_patient in gita to avoid overlapping with neurovoz
        data_gita["id_patient"] = data_gita["id_patient"] + 1000

        data = pd.concat([data_alb, data_neuro, data_gita])

        # Categorise label to 0 and 1
        data["label"] = data["label"].astype("category").cat.codes

        print("Reading the .wav files...")
        target_sr = 16000  # Because Albayzin has 16kHz sampling rate and is the lowest sampling rate in our datasets
        # Read the .wav files and store the signals and sampling rates
        data["signal"], data["sr"] = zip(
            *data["file_path"].map(lambda x: librosa.load(x, sr=target_sr))
        )

        # Normalize the audio
        data["signal"] = data["signal"].apply(self.normalize_audio)

        # Frame the signals into 400ms frames with 50% overlap
        frame_length = int(
            data["sr"].iloc[0] * self.hyperparams["frame_size_ms"]
        )  # 400ms
        hop_length = int(frame_length * self.hyperparams["hop_size_percent"])
        # 200ms = 50% overlap

        # Drop all signals that are shorter than frame_length
        data = data[data["signal"].apply(lambda x: len(x) >= frame_length)]

        # Frame the signals
        data["signal_framed"] = data["signal"].apply(
            lambda x: librosa.util.frame(
                x, frame_length=frame_length, hop_length=hop_length, axis=0
            )
        )

        print("Matching text grid phonemes to the signals...")

        # Match phonemes
        data["phonemes_matched"] = data.apply(
            lambda x: self.match_phonemes(x["phonemes"], x["signal"], x["sr"]), axis=1
        )

        # Frame the phonemes with 50% overlap
        data["phonemes_framed_overlap"] = data["phonemes_matched"].apply(
            lambda x: librosa.util.frame(
                x, frame_length=frame_length, hop_length=hop_length, axis=0
            )
        )

        # Remove unused columns
        data = data.drop(columns=["signal", "phonemes", "phonemes_matched"])

        # Explode the DataFrame by signal_framed and phonemes_framed_overlap
        data = data.explode(["signal_framed", "phonemes_framed_overlap"])

        # Reset index
        data.reset_index(drop=True, inplace=True)

        if self.spectrogram:
            # Calculate the spectrogram. We want that each spectrogram is 400ms long with X windows of 30ms each.
            win_length = int(
                self.hyperparams["spectrogram_win_size"] * data["sr"].iloc[0]
            )
            hop_length = win_length // 2  # 50% overlap

            n_fft = 512
            n_mels = 65  # if hifigan use 80

            # Calculate the melspectrogram using librosa
            data["spectrogram"] = data["signal_framed"].apply(
                lambda x: librosa.feature.melspectrogram(
                    y=x,
                    sr=data["sr"].iloc[0],
                    win_length=win_length,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    center=False,
                )
            )

            # Calculate the power to db for each frame
            data["spectrogram"] = data["spectrogram"].apply(
                lambda x: librosa.power_to_db(x, ref=np.max)
            )

            # Normalise each spectrogram by substraction the mean and dividing by the standard deviation
            data["spectrogram"] = data["spectrogram"].apply(
                lambda x: (x - x.mean()) / x.std()
            )

            # Frame again the phonemes to match spectrogram frames
            data["phonemes_framed_spectrogram"] = data["phonemes_framed_overlap"].apply(
                lambda x: librosa.util.frame(
                    x, frame_length=win_length, hop_length=hop_length, axis=0
                )
            )

            # Collapse to most common phoneme
            data["collapsed_phonemes"] = data["phonemes_framed_spectrogram"].apply(
                collapse_to_most_repeated
            )

        # Save data to this to not compute this again if it is not necessary. This is a heavy process.
        name_save = (
            "local_results/data_frame_with_phonemes"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + ".pkl"
        )

        if not os.path.exists(name_save):
            # Save the data
            pd.to_pickle({"data": data}, name_save)

        return data

    def get_dataloaders(
        self,
        experiment="fisrt",
        supervised=False,
        verbose=True,
    ):
        # Map phonemes to manner classes
        manner_classes = {
            "p": 0,  # plosives
            "t": 0,
            "k": 0,
            "b": 1,  # plosives voiced
            "B": 1,
            "d": 1,
            "D": 1,
            "g": 1,
            "G": 1,
            "n": 2,  # nasals
            "N": 2,
            "m": 2,
            "NY": 2,
            "J": 2,
            "f": 3,  # fricatives
            "s": 3,
            "z": 3,
            "x": 3,
            "h": 3,
            "T": 3,
            "R": 4,  # liquids
            "r": 4,
            "4": 4,
            "l": 4,
            "y": 4,
            "jj": 4,
            "L": 4,
            "a": 5,  # vowels
            "e": 5,
            "e ": 5,
            "i": 5,
            "j": 5,
            "o": 5,
            "u": 5,
            "w": 5,
            "CH": 6,  # affricates
            "tS": 6,
            "sil": 7,  # silence
            "_": 7,
            "sp": 7,  # short pause
        }

        # Map all self.data["collapsed_phonemes"] to manner classes
        self.data["manner_class"] = self.data["collapsed_phonemes"].apply(
            lambda x: [manner_classes[phoneme] for phoneme in x]
        )

        # Current data has 3 ["label"] values: 0, 1, 2. However, 0 and 1 are both Healthy and 2 is Parkinson. We need to map that: 0 and 1 to 0 and 2 to 1
        # self.data["label"] = self.data["label"].apply(lambda x: 0 if x < 2 else 1)

        # Modify the manner class: sum to each manner class the label multiplied by the number of manner classes (8)
        if supervised:
            self.data["manner_class"] = self.data.apply(
                lambda x: [x["label"] * 8 + y for y in x["manner_class"]], axis=1
            )

        # Print unique values in manner_class
        print("Unique values in manner_class: ", np.unique(self.data["manner_class"]))

        albayzin_data = self.data[self.data["dataset"] == "albayzin"]
        neurovoz_data = self.data[self.data["dataset"] == "neurovoz"]
        gita_data = self.data[self.data["dataset"] == "gita"]

        albayzin_patients = albayzin_data["id_patient"].unique()
        np.random.shuffle(albayzin_patients)

        # Split in 80% train and 20% val
        albayzin_train = albayzin_patients[: int(len(albayzin_patients) * 0.8)]
        albayzin_val = albayzin_patients[int(len(albayzin_patients) * 0.8) :]

        train_data = albayzin_data[albayzin_data["id_patient"].isin(albayzin_train)]
        val_data = albayzin_data[albayzin_data["id_patient"].isin(albayzin_val)]

        if experiment == "first":
            # Test data is all neurovoz and gita  patients
            test_data = pd.concat([neurovoz_data, gita_data])

            f = "first_experiment"

            self.create_dataloader(
                train_data,
                val_data,
                test_data,
                f,
            )

        elif experiment == "fourth":
            # Create 10 folds of neurovoz and gita patients
            healthy_gita_patients = gita_data[gita_data["label"] == 0][
                "id_patient"
            ].unique()
            pd_gita_patients = gita_data[gita_data["label"] == 1]["id_patient"].unique()
            healthy_neurovoz_patients = neurovoz_data[neurovoz_data["label"] == 0][
                "id_patient"
            ].unique()
            pd_neurovoz_patients = neurovoz_data[neurovoz_data["label"] == 1][
                "id_patient"
            ].unique()
            # Randomly shuffle the patients
            np.random.shuffle(healthy_gita_patients)
            np.random.shuffle(pd_gita_patients)
            np.random.shuffle(healthy_neurovoz_patients)
            np.random.shuffle(pd_neurovoz_patients)

            # Split in 10 folds all lists
            healthy_gita_patients = np.array_split(healthy_gita_patients, 10)
            pd_gita_patients = np.array_split(pd_gita_patients, 10)
            healthy_neurovoz_patients = np.array_split(healthy_neurovoz_patients, 10)
            pd_neurovoz_patients = np.array_split(pd_neurovoz_patients, 10)

            # Create the 10 folds
            for f in range(10):
                # Get the test patients (just 1 fold)
                test_patients = np.concatenate(
                    [
                        healthy_gita_patients[f],
                        pd_gita_patients[f],
                        healthy_neurovoz_patients[f],
                        pd_neurovoz_patients[f],
                    ]
                )
                # Get the val patients, the fold just before the test fold (if it is the first fold, then the last fold is the val fold)
                if f == 0:
                    val_patients = np.concatenate(
                        [
                            healthy_gita_patients[-1],
                            pd_gita_patients[-1],
                            healthy_neurovoz_patients[-1],
                            pd_neurovoz_patients[-1],
                        ]
                    )
                else:
                    val_patients = np.concatenate(
                        [
                            healthy_gita_patients[f - 1],
                            pd_gita_patients[f - 1],
                            healthy_neurovoz_patients[f - 1],
                            pd_neurovoz_patients[f - 1],
                        ]
                    )

                # Get the train patients, which are the rest of the patients (not in test not in val)
                train_patients = np.concatenate(
                    [
                        np.concatenate(healthy_gita_patients),
                        np.concatenate(pd_gita_patients),
                        np.concatenate(healthy_neurovoz_patients),
                        np.concatenate(pd_neurovoz_patients),
                    ]
                )
                # Drop now the patients athat already are in test and val
                train_patients = np.concatenate(
                    [np.setdiff1d(x, test_patients) for x in train_patients]
                )
                train_patients = np.concatenate(
                    [np.setdiff1d(x, val_patients) for x in train_patients]
                )

                # Check that train, val and test are all diferent (no patient is repeated)
                assert len(np.intersect1d(train_patients, val_patients)) == 0
                assert len(np.intersect1d(train_patients, test_patients)) == 0
                assert len(np.intersect1d(val_patients, test_patients)) == 0

                # The train data then is all albayzin + the train patients of the other datasets
                train_data = albayzin_data
                train_data = pd.concat(
                    [
                        train_data,
                        neurovoz_data[neurovoz_data["id_patient"].isin(train_patients)],
                        gita_data[gita_data["id_patient"].isin(train_patients)],
                    ]
                )

                # The val data is the val patients of the other datasets
                val_data = pd.concat(
                    [
                        neurovoz_data[neurovoz_data["id_patient"].isin(val_patients)],
                        gita_data[gita_data["id_patient"].isin(val_patients)],
                    ]
                )

                # The test data is the test patients of the other datasets
                test_data = pd.concat(
                    [
                        neurovoz_data[neurovoz_data["id_patient"].isin(test_patients)],
                        gita_data[gita_data["id_patient"].isin(test_patients)],
                    ]
                )

                self.create_dataloader(
                    train_data,
                    val_data,
                    test_data,
                    supervised,
                    f,
                )

    def create_dataloader(self, train_data, val_data, test_data, supervised, f=0):

        audio_features = "spectrogram"

        # Make sure that x_train is shape N, Channels, Height, Width (N,C,H,W) where C is 1
        x_train = np.stack(train_data[audio_features].values)
        x_train = np.expand_dims(x_train, axis=1)
        y_train = train_data["label"].values
        p_train = np.array([np.array(x) for x in train_data["manner_class"]])
        # Make a new label that is the combination of p and y. That is: p has 7 classes and y has 2 classes. So the new label will have 14 classes
        z_train = np.array(
            [np.array([np.repeat(x, len(y)), y]) for x, y in zip(y_train, p_train)]
        )
        n_classes = np.unique(p_train).shape[0]
        z_train = np.array([np.array(x[1] + n_classes * x[0]) for x in z_train])
        d_train = np.array([np.array(x) for x in train_data["dataset"]])

        x_val = np.stack(val_data[audio_features].values)
        x_val = np.expand_dims(x_val, axis=1)
        y_val = val_data["label"].values
        p_val = np.array([np.array(x) for x in val_data["manner_class"]])
        z_val = np.array(
            [np.array([np.repeat(x, len(y)), y]) for x, y in zip(y_val, p_val)]
        )
        z_val = np.array([np.array(x[1] + n_classes * x[0]) for x in z_val])
        d_val = np.array([np.array(x) for x in val_data["dataset"]])
        id_val = np.array([np.array(x) for x in val_data["id_patient"]])

        x_test = np.stack(test_data[audio_features].values)
        x_test = np.expand_dims(x_test, axis=1)
        y_test = test_data["label"].values
        p_test = np.array([np.array(x) for x in test_data["manner_class"]])
        z_test = np.array(
            [np.array([np.repeat(x, len(y)), y]) for x, y in zip(y_test, p_test)]
        )
        z_test = np.array([np.array(x[1] + n_classes * x[0]) for x in z_test])
        d_test = np.array([np.array(x) for x in test_data["dataset"]])

        print("Creating dataloaders...")
        train_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_train,
                    y_train,
                    p_train,
                    d_train,
                )
            ),
            drop_last=False,
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_val,
                    y_val,
                    p_val,
                    d_val,
                    id_val,
                )
            ),
            drop_last=False,
            batch_size=self.hyperparams["batch_size"],
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_test,
                    y_test,
                    p_test,
                    d_test,
                )
            ),
            drop_last=False,
            batch_size=self.hyperparams["batch_size"],
            shuffle=False,
        )

        # Save the dataloaders to a file
        train_name = (
            "local_results/folds/train_loader_supervised_"
            + str(self.hyperparams["supervised"])
            + "_frame_size_"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + "fold"
            + str(f)
            + ".pt"
        )
        val_name = (
            "local_results/folds/val_loader_supervised_"
            + str(self.hyperparams["supervised"])
            + "_frame_size_"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + "fold"
            + str(f)
            + ".pt"
        )
        test_name = (
            "local_results/folds/test_loader_supervised_"
            + str(self.hyperparams["supervised"])
            + "_frame_size_"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + "fold"
            + str(f)
            + ".pt"
        )
        test_data_name = (
            "local_results/folds/test_data_supervised_"
            + str(self.hyperparams["supervised"])
            + "_frame_size_"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + "fold"
            + str(f)
            + ".pt"
        )

        torch.save(train_loader, train_name)
        torch.save(val_loader, val_name)
        torch.save(test_loader, test_name)
        torch.save(test_data, test_data_name)
        return (
            train_loader,
            val_loader,
            test_loader,
            train_data,
            val_data,
            test_data,
        )

    def normalize_audio(self, audio_data, max=None):
        if max is None:
            max_value = np.max(np.abs(audio_data))
        else:
            max_value = max
        normalized_data = audio_data / max_value

        return normalized_data
