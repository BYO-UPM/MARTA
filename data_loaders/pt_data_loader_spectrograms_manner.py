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
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import bisect


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

        datapath = (
            "/media/my_ftp/BasesDeDatos_Voz_Habla/PC-GITA/gita_htk_forced_alignment/"
        )

        for root, dirs, files in os.walk(datapath):
            for file in files:
                # If the file does not end with .wav, skip it
                if not file.endswith(".wav"):
                    continue
                if file.endswith("_normalized.wav"):
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

    def read_italian(self):
        file_paths = []
        labels = []
        id_patient = []
        texts = []
        phonemes = []

        datapath = "/home/aguerrero@gaps_domain.ssr.upm.es/Projects/MARTA/data_loaders/italian_data/anonymized_italian"

        for root, dirs, files in os.walk(datapath):
            for file in files:
                # If the file does not end with .wav, skip it
                if not file.endswith(".wav"):
                    continue
                if file.endswith("_normalized.wav"):
                    continue
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

                # ID patient
                id_name = root.split("/")[-1]
                parent_folder = root.split("/")[-2]
                id_patient.append(parent_folder + id_name)
                # Label
                if "Healthy" in root:
                    labels.append(0)
                elif "Parkinson" in root:
                    labels.append(1)
                # Text
                texts.append(file.split(".")[0][:2])

                # Read the text grid file
                tg_file = os.path.join(root, file).replace(".wav", ".TextGrid")
                # Check if the file exists
                if not os.path.exists(tg_file):
                    print("File does not exist: ", tg_file)
                    phonemes.append(None)
                    continue
                tg_file = tg.TextGrid(tg_file)
                phonemes.append(tg_file["phones"])

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

        # Keep only texts= ["PR", "FB", "B1", "B2"], that is removing vowels and DDKs
        data = data[data["text"].isin(["PR", "FB", "B1", "B2"])]

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
                # if file ends with _normalized.wav continue
                if file.endswith("_normalized.wav"):
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

        datapath_wav = "/media/my_ftp/BasesDeDatos_Voz_Habla/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/albayzin_htk_forced_alignment"

        i = 0
        for file in os.listdir(datapath_wav):
            # If the file does not end with .wav, skip it
            if not file.endswith(".wav"):
                continue
            if file.endswith("_normalized.wav"):
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
        # Prepare a list of (start, end, label) tuples and sort it
        intervals = sorted(
            (interval.xmin, interval.xmax, interval.text) for interval in phonemes
        )
        starts = [interval[0] for interval in intervals]  # Extract the start times

        phoneme_labels = []

        for timestamp in range(len(signal)):
            # Convert the timestamp to seconds
            timestamp_seconds = timestamp / sr

            # Find the index of the first interval whose start time is greater than the timestamp
            idx = bisect.bisect_left(starts, timestamp_seconds)

            # Check if the previous interval (if any) contains the timestamp
            if (
                idx > 0
                and intervals[idx - 1][0] <= timestamp_seconds <= intervals[idx - 1][1]
            ):
                phoneme_labels.append(intervals[idx - 1][2])
            else:
                phoneme_labels.append(None)

        return phoneme_labels

    def ebu_r128_normalize(self, file_path):
        normalized_file_path = file_path.replace(".wav", "_normalized.wav")
        if not os.path.exists(normalized_file_path):
            command = [
                "ffmpeg-normalize",
                file_path,
                "-o",
                normalized_file_path,
            ]
            subprocess.run(command, check=True)
        return normalized_file_path

    def read_dataset(self):
        print("Reading the data...")

        # Italian removed from study as it has bad quality
        # data_it = self.read_italian()
        # data_it["dataset"] = "italian"

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
        # data = pd.concat([data_alb, data_neuro, data_gita, data_it])

        # Assert that all datasets have been read
        assert len(data[data["dataset"] == "albayzin"]) > 0
        assert len(data[data["dataset"] == "neurovoz"]) > 0
        assert len(data[data["dataset"] == "gita"]) > 0
        # assert len(data[data["dataset"] == "italian"]) > 0

        print("Data read successfully...")

        # Categorise label to 0 and 1
        data["label"] = data["label"].astype("category").cat.codes

        print("Reading the .wav files...")
        target_sr = 16000  # Because Albayzin has 16kHz sampling rate and is the lowest sampling rate in our datasets

        # Normalize all audio files using EBU R128 with a progress bar and parallel processing
        file_paths = data["file_path"].tolist()

        # Define a function to handle the normalization in parallel
        def parallel_ebu_r128_normalize(file_paths):
            with ThreadPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(self.ebu_r128_normalize, file_paths),
                        total=len(file_paths),
                        desc="Normalizing audio files",
                    )
                )
            return results

        # Apply parallel normalization
        data["normalized_file_path"] = parallel_ebu_r128_normalize(file_paths)

        # Read the normalized .wav files and store the signals and sampling rates
        print("Loading .wav files...")
        data["signal"], data["sr"] = zip(
            *tqdm(
                data["normalized_file_path"].map(
                    lambda x: librosa.load(x, sr=target_sr)
                ),
                total=len(data),
                desc="Reading normalized .wav files",
            )
        )

        # Check that there is no nan signal
        assert len(data[data["signal"].isna()]) == 0

        # Apply logmmse to all signals
        print("Applying logmmse to all signals...")
        import logmmse

        data["signal"] = data["signal"].apply(
            lambda x: logmmse.logmmse(x, target_sr, output_file=None)
        )

        # Normalize the audio (assuming self.normalize_audio is defined elsewhere in your code)
        data["signal"] = data["signal"].apply(self.normalize_audio)

        # Check that there is no nan signal
        assert len(data[data["signal"].isna()]) == 0

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

        # Check that there is no nan signal framed
        assert len(data[data["signal_framed"].isna()]) == 0

        print("Matching text grid phonemes to the signals...")

        # Match phonemes (this takes too long, we have to optimise it)
        # Use the optimized match_phonemes function
        tqdm.pandas(desc="Matching phonemes")
        data["phonemes_matched"] = data.progress_apply(
            lambda x: self.match_phonemes(x["phonemes"], x["signal"], x["sr"]), axis=1
        )

        # Frame the phonemes with 50% overlap
        print("Framing the phonemes with 50% overlap...")
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

            # Check that no spectrogram gave nan
            assert (
                len(data[data["spectrogram"].apply(lambda x: np.isnan(x).any())]) == 0
            )

            # Calculate the power to db for each frame
            data["spectrogram"] = data["spectrogram"].apply(
                lambda x: librosa.power_to_db(x, ref=np.max)
            )

            # Assert nans after power to db
            assert (
                len(data[data["spectrogram"].apply(lambda x: np.isnan(x).any())]) == 0
            )

            # Normalise each spectrogram by substraction the mean and dividing by the standard deviation
            data["spectrogram"] = data["spectrogram"].apply(
                lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
            )

            assert (
                len(data[data["spectrogram"].apply(lambda x: np.isnan(x).any())]) == 0
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
        experiment="fourth",
        supervised=False,
        verbose=True,
    ):
        # Load manner class map from yaml
        import yaml

        with open(
            "/home/aguerrero@gaps_domain.ssr.upm.es/Projects/MARTA/data_loaders/manner_classes.yaml",
            "r",
        ) as file:
            manner_classes = yaml.load(file, Loader=yaml.FullLoader)

        # First the spanish data
        spanish_data = self.data[self.data["dataset"] != "italian"]

        # Map all self.data["collapsed_phonemes"] to manner classes
        spanish_data["manner_class"] = spanish_data["collapsed_phonemes"].apply(
            lambda x: [
                manner_classes["manner_classes"]["spanish"][phoneme] for phoneme in x
            ]
        )

        # Italian data
        # italian_data = self.data[self.data["dataset"] == "italian"]

        # # Map all self.data["collapsed_phonemes"] to manner classes
        # italian_data["manner_class"] = italian_data["collapsed_phonemes"].apply(
        #     lambda x: [
        #         phoneme.split("_")[0]  # Split by '_' and get the manner class
        #         for phoneme in x
        #     ]
        # )
        # italian_data["manner_class"] = italian_data["manner_class"].apply(
        #     lambda x: [
        #         manner_classes["manner_classes"]["italian"][phoneme] for phoneme in x
        #     ]
        # )

        # Merge
        # self.data = pd.concat([spanish_data, italian_data])
        self.data = spanish_data

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
        # italian_data = self.data[self.data["dataset"] == "italian"]

        # # Italian IDs are strings, categorize it to integers above 2000 (neurovoz are under 100, gita above 1000)
        # italian_data["id_patient"] = (
        #     italian_data["id_patient"].astype("category").cat.codes + 2000
        # )

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

            (
                train_loader,
                val_loader,
                test_loader,
                train_data,  # train_data, not used
                val_data,  # val_data, not used
                test_data,
            ) = self.create_dataloader(
                train_data,
                val_data,
                test_data,
                f,
            )

            return (
                train_loader,
                val_loader,
                test_loader,
                train_data,  # train_data, not used
                val_data,  # val_data, not used
                test_data,
            )

        elif experiment == "fourth":
            # Create 10 folds of neurovoz and gita patients and italian patients
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
            # healthy_italian_patients = italian_data[italian_data["label"] == 0][
            #     "id_patient"
            # ].unique()
            # pd_italian_patients = italian_data[italian_data["label"] == 1][
            #     "id_patient"
            # ].unique()

            # Randomly shuffle the patients
            np.random.shuffle(healthy_gita_patients)
            np.random.shuffle(pd_gita_patients)
            np.random.shuffle(healthy_neurovoz_patients)
            np.random.shuffle(pd_neurovoz_patients)
            # np.random.shuffle(healthy_italian_patients)
            # np.random.shuffle(pd_italian_patients)

            # Split in 10 folds all lists
            healthy_gita_patients = np.array_split(healthy_gita_patients, 10)
            pd_gita_patients = np.array_split(pd_gita_patients, 10)
            healthy_neurovoz_patients = np.array_split(healthy_neurovoz_patients, 10)
            pd_neurovoz_patients = np.array_split(pd_neurovoz_patients, 10)
            # healthy_italian_patients = np.array_split(healthy_italian_patients, 10)
            # pd_italian_patients = np.array_split(pd_italian_patients, 10)

            # Create the 10 folds
            for f in range(10):
                # Get the test patients (just 1 fold)
                test_patients = np.concatenate(
                    [
                        healthy_gita_patients[f],
                        pd_gita_patients[f],
                        healthy_neurovoz_patients[f],
                        pd_neurovoz_patients[f],
                        # healthy_italian_patients[f],
                        # pd_italian_patients[f],
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
                            # healthy_italian_patients[-1],
                            # pd_italian_patients[-1],
                        ]
                    )
                else:
                    val_patients = np.concatenate(
                        [
                            healthy_gita_patients[f - 1],
                            pd_gita_patients[f - 1],
                            healthy_neurovoz_patients[f - 1],
                            pd_neurovoz_patients[f - 1],
                            # healthy_italian_patients[f - 1],
                            # pd_italian_patients[f - 1],
                        ]
                    )

                # Get the train patients, which are the rest of the patients (not in test not in val)
                train_patients = np.concatenate(
                    [
                        np.concatenate(healthy_gita_patients),
                        np.concatenate(pd_gita_patients),
                        np.concatenate(healthy_neurovoz_patients),
                        np.concatenate(pd_neurovoz_patients),
                        # np.concatenate(healthy_italian_patients),
                        # np.concatenate(pd_italian_patients),
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
                        # italian_data[italian_data["id_patient"].isin(train_patients)],
                    ]
                )

                # The val data is the val patients of the other datasets
                val_data = pd.concat(
                    [
                        neurovoz_data[neurovoz_data["id_patient"].isin(val_patients)],
                        gita_data[gita_data["id_patient"].isin(val_patients)],
                        # italian_data[italian_data["id_patient"].isin(val_patients)],
                    ]
                )

                # The test data is the test patients of the other datasets
                test_data = pd.concat(
                    [
                        neurovoz_data[neurovoz_data["id_patient"].isin(test_patients)],
                        gita_data[gita_data["id_patient"].isin(test_patients)],
                        # italian_data[italian_data["id_patient"].isin(test_patients)],
                    ]
                )

                (
                    train_loader,
                    val_loader,
                    test_loader,
                    train_data,  # train_data, not used
                    val_data,  # val_data, not used
                    test_data,
                ) = self.create_dataloader(
                    train_data,
                    val_data,
                    test_data,
                    f,
                )

                return (
                    train_loader,
                    val_loader,
                    test_loader,
                    train_data,  # train_data, not used
                    val_data,  # val_data, not used
                    test_data,
                )

    def create_dataloader(self, train_data, val_data, test_data, supervised, f=0):

        audio_features = "spectrogram"

        # X = Spectrograms (N, C, H, W)
        # Y = Labels (N)
        # P = Manner classes (N, 8)
        # Z = New labels (N, 16) (Y + P)
        # D = Dataset (N)

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
