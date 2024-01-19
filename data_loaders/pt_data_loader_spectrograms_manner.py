import torch
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
from sklearn.preprocessing import StandardScaler
import torchaudio
from torchaudio import transforms
import textgrids as tg
import time
from torch.utils.data import Dataset, DataLoader, sampler


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
    def __init__(self, data_path, hyperparams):
        self.hyperparams = hyperparams
        self.plps = self.hyperparams["n_plps"] > 0
        self.mfcc = self.hyperparams["n_mfccs"] > 0
        self.spectrogram = self.hyperparams["spectrogram"]
        self.data_path = data_path

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

    def read_neurovoz(self, dataset="albayzin"):
        file_paths = []
        labels = []
        id_patient = []
        texts = []
        phonemes = []

        datapath = "labeled/NeuroVoz"

        for file in os.listdir(datapath):
            # If the file does not end with .wav, skip it
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(datapath, file)
            file_paths.append(file_path)
            # Each file is named as follows: <speakerid>_<idpatient>_<text>_<condition>.wav
            labels.append(file.split("_")[3].split(".")[0])
            id_patient.append(file.split("_")[1])
            texts.append(file.split("_")[2])

            # Read the text grid file
            tg_file = os.path.join(datapath, file + ".TextGrid")
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
        datapath_textgrid = "/media/my_ftp/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/albayzin_htk_forced_alignment"

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
            tg_file = os.path.join(datapath_textgrid, file.split(".")[0] + ".TextGrid")
            # Check if the file exists
            if not os.path.exists(tg_file):
                print("File does not exist: ", tg_file)
                i += 1
                phonemes.append(None)
                continue
            tg_file = tg.TextGrid(tg_file)
            phonemes.append(tg_file["speaker : phones"])

        print("Total WAV files: ", len(os.listdir(datapath_wav)))
        print("Total TextGrid files: ", len(os.listdir(datapath_textgrid)))
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

    def match_phonemes2(self, phonemes, signal, sr):
        # Create a dictionary for fast interval lookup
        phoneme_dict = {interval.xmin: interval.text for interval in phonemes}

        # Convert signal timestamps to seconds
        timestamps_seconds = np.arange(len(signal)) / sr

        # Initialize an array to store phoneme labels
        phoneme_labels = np.empty(len(signal), dtype=object)

        # Iterate through timestamps and fill in phoneme labels
        for timestamp_index, timestamp_seconds in enumerate(timestamps_seconds):
            # Use binary search to find the matching interval
            interval_start_times = np.array(list(phoneme_dict.keys()))
            matching_interval_index = (
                np.searchsorted(interval_start_times, timestamp_seconds) - 1
            )

            if matching_interval_index >= 0:
                matching_interval = phonemes[matching_interval_index]
                phoneme_labels[timestamp_index] = matching_interval.text
            else:
                phoneme_labels[timestamp_index] = None

        return phoneme_labels

    def read_dataset(self):
        print("Reading the data...")

        data_train = self.read_albayzin()
        # add a column with the dataset name
        data_train["dataset"] = "albayzin"
        data_test = self.read_neurovoz()
        # add a column with the dataset name
        data_test["dataset"] = "neurovoz"

        data = pd.concat([data_train, data_test])

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
            # Calculate the spectrogram. We want that each spectrogram is 400ms long with 20 windows of 23ms each.
            win_length = 512
            hop_length = win_length // 2  # 50% overlap

            n_fft = 512
            n_mels = 65  # if hifigan use 80

            # Calculate the melspectrogram using librosa
            data["spectrogram"] = data["signal_framed"].apply(
                lambda x: librosa.feature.melspectrogram(
                    y=x,
                    sr=data["sr"].iloc[0],
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

    def get_dataloaders(self, train_albayzin=False, verbose=True, supervised=False):
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
        self.data["label"] = self.data["label"].apply(lambda x: 0 if x < 2 else 1)

        # Modify the manner class: sum to each manner class the label multiplied by the number of manner classes (8)
        if supervised:
            self.data["manner_class"] = self.data.apply(
                lambda x: [x["label"] * 8 + y for y in x["manner_class"]], axis=1
            )

        # Print unique values in manner_class
        print("Unique values in manner_class: ", np.unique(self.data["manner_class"]))

        for f in range(10):
            print("Splitting train, val and test for fold: ", f)
            # Set all random seeds to "f"
            torch.manual_seed(f)
            np.random.seed(f)

            # Split the data into train and test.
            if (
                train_albayzin
            ):  # If we want to train with albayzin + 0.9 of neurovoz healthy patients
                # =============== Prepare data ===============
                albayzin_data = self.data[self.data["dataset"] == "albayzin"]

                # Get all healthy patients from neurovoz and shuffle them
                healthy_patients_neurovoz = self.data[
                    (self.data["dataset"] == "neurovoz") & (self.data["label"] == 0)
                ]["id_patient"].unique()
                # Suffle the numpy array
                np.random.shuffle(healthy_patients_neurovoz)

                # == Split nwurovoz healthy patients in 90-10 partitions ==
                # Firt 90%
                half_90_hp_neurovoz = healthy_patients_neurovoz[
                    : int(len(healthy_patients_neurovoz) * 0.9)
                ]
                half_10_hp_neurovoz = healthy_patients_neurovoz[
                    int(len(healthy_patients_neurovoz) * 0.9) :
                ]
                neurovoz_healthy_90_data = self.data[
                    (self.data["dataset"] == "neurovoz")
                    & (self.data["label"] == 0)
                    & (self.data["id_patient"].isin(half_90_hp_neurovoz))
                ]
                # Second half
                neurovoz_healthy_10_data = self.data[
                    (self.data["dataset"] == "neurovoz")
                    & (self.data["label"] == 0)
                    & (self.data["id_patient"].isin(half_10_hp_neurovoz))
                ]

                # Get all parkinsonian patients from neurovoz and shuffle
                parkinsonian_patients_neurovoz = self.data[
                    (self.data["dataset"] == "neurovoz") & (self.data["label"] == 1)
                ]["id_patient"].unique()
                # Suffle the numpy array
                np.random.shuffle(parkinsonian_patients_neurovoz)

                # == Split neurovoz parkinsonian patients in two halfs: 90-10 ==
                # Firt 90%
                half_90_pk_neurovoz = parkinsonian_patients_neurovoz[
                    : int(len(parkinsonian_patients_neurovoz) * 0.9)
                ]
                half_10_pk_neurovoz = parkinsonian_patients_neurovoz[
                    int(len(parkinsonian_patients_neurovoz) * 0.9) :
                ]
                neurovoz_parkinson_90_data = self.data[
                    (self.data["dataset"] == "neurovoz")
                    & (self.data["label"] == 1)
                    & (self.data["id_patient"].isin(half_90_pk_neurovoz))
                ]
                # Second 10%
                neurovoz_parkinson_10_data = self.data[
                    (self.data["dataset"] == "neurovoz")
                    & (self.data["label"] == 1)
                    & (self.data["id_patient"].isin(half_10_pk_neurovoz))
                ]

                # =========================================== TRAIN DATA ===========================================
                if supervised:
                    # Albayzin + 90% of healthy neurovoz + 90% of parkinson neurovoz
                    train_data = pd.concat(
                        [
                            albayzin_data,
                            neurovoz_healthy_90_data,
                            neurovoz_parkinson_90_data,
                        ]
                    )
                else:
                    # Albayzin + 90% of healthy neurovoz
                    train_data = pd.concat([albayzin_data, neurovoz_healthy_90_data])

                # =========================================== VALIDATION DATA ===========================================
                if supervised:
                    # Get 20% of train data ensuring that: 20% of albayzin PATIENTS + 20% of healthy neurovoz PATIENTS + 20% of parkinson neurovoz PATIENTS
                    # Get 20% of albayzin
                    albayzin_patients = albayzin_data["id_patient"].unique()
                    # Shuffle it
                    np.random.shuffle(albayzin_patients)
                    # Select 20% of the patients
                    albayzin_val = albayzin_data[
                        albayzin_data["id_patient"].isin(
                            albayzin_patients[: int(len(albayzin_patients) * 0.2)]
                        )
                    ]
                    # Get 20% of healthy neurovoz
                    neurovoz_healthy_patients = neurovoz_healthy_90_data[
                        "id_patient"
                    ].unique()
                    # Shuffle it
                    np.random.shuffle(neurovoz_healthy_patients)
                    # Select 20% of the patients
                    neurovoz_healthy_val = neurovoz_healthy_90_data[
                        neurovoz_healthy_90_data["id_patient"].isin(
                            neurovoz_healthy_patients[
                                : int(len(neurovoz_healthy_patients) * 0.2)
                            ]
                        )
                    ]

                    # Get 20% of parkinson neurovoz
                    neurovoz_parkinson_patients = neurovoz_parkinson_90_data[
                        "id_patient"
                    ].unique()
                    # Shuffle it
                    np.random.shuffle(neurovoz_parkinson_patients)
                    # Select 20% of the patients
                    neurovoz_parkinson_val = neurovoz_parkinson_90_data[
                        neurovoz_parkinson_90_data["id_patient"].isin(
                            neurovoz_parkinson_patients[
                                : int(len(neurovoz_parkinson_patients) * 0.2)
                            ]
                        )
                    ]

                    # Concatenate all
                    val_data = pd.concat(
                        [
                            albayzin_val,
                            neurovoz_healthy_val,
                            neurovoz_parkinson_val,
                        ]
                    )
                    # Remove from train_data the val_data
                    train_data = train_data.drop(val_data.index)
                else:
                    # Get 20% of train data ensuring that: 20% of albayzin PATIENTS + 20· of healthy neurovoz PATIENTS
                    # Get 20% of albayzin
                    albayzin_patients = albayzin_data["id_patient"].unique()
                    albayzin_val = albayzin_data[
                        albayzin_data["id_patient"].isin(
                            np.random.choice(
                                albayzin_patients, int(len(albayzin_patients) * 0.2)
                            )
                        )
                    ]
                    # Get 20% of healthy neurovoz
                    neurovoz_healthy_patients = neurovoz_healthy_90_data[
                        "id_patient"
                    ].unique()
                    neurovoz_healthy_val = neurovoz_healthy_90_data[
                        neurovoz_healthy_90_data["id_patient"].isin(
                            np.random.choice(
                                neurovoz_healthy_patients,
                                int(len(neurovoz_healthy_patients) * 0.2),
                            )
                        )
                    ]

                    # Concatenate all
                    val_data = pd.concat(
                        [
                            albayzin_val,
                            neurovoz_healthy_val,
                        ]
                    )

                    # Remove from train_data the val_data
                    train_data = train_data.drop(val_data.index)

                # =========================================== TEST DATA ===========================================
                # Test data is alway: 10% of healthy neurovoz + 10% of parkinson neurovoz
                test_data = pd.concat(
                    [
                        neurovoz_healthy_10_data,
                        neurovoz_parkinson_10_data,
                    ]
                )

                # =========================================== ASSERTIONS ===========================================
                # Get all id patients from train, val and test data
                train_patients = train_data["id_patient"].unique()
                val_patients = val_data["id_patient"].unique()
                test_patients = test_data["id_patient"].unique()

                # Assert that there are no patients shared between train, val and test
                assert (
                    len(np.intersect1d(train_patients, val_patients)) == 0
                ), "There are patients in both train and val data!"
                assert (
                    len(np.intersect1d(train_patients, test_patients)) == 0
                ), "There are patients in both train and test data!"
                assert (
                    len(np.intersect1d(val_patients, test_patients)) == 0
                ), "There are patients in both val and test data!"

            else:  # If we want to train with only neurovoz healthy patients and not use albayzin for enriching the training data
                # Train data will be 0.8 of neurovoz healthy patients
                train_data = self.data[
                    (self.data["dataset"] == "neurovoz") & (self.data["label"] == 0)
                ].sample(frac=0.8, random_state=f)
                # Get the rest healhty patients not used for training
                rest_data = self.data[
                    (self.data["dataset"] == "neurovoz") & (self.data["label"] == 0)
                ].drop(train_data.index)
                test_data = self.data[
                    (self.data["dataset"] == "neurovoz") & (self.data["label"] == 1)
                ]
                # Concatenate the rest of healthy patients with the test data
                test_data = pd.concat([test_data, rest_data])

            if verbose:
                # Print the number of patients in each set
                print("Number of patients in train set: ", len(train_patients))
                print("Number of patients in val set: ", len(val_patients))
                print(
                    "Number of patients in test set: ",
                    len(test_data["id_patient"].unique()),
                )

                # Get nº of patients of dataset per set
                albayzin_train = len(
                    train_data[train_data["dataset"] == "albayzin"][
                        "id_patient"
                    ].unique()
                )
                albayzin_val = len(
                    val_data[val_data["dataset"] == "albayzin"]["id_patient"].unique()
                )
                albayzin_test = len(
                    test_data[test_data["dataset"] == "albayzin"]["id_patient"].unique()
                )

                print("Number of patients in train set from albayzin: ", albayzin_train)
                print("Number of patients in val set from albayzin: ", albayzin_val)
                print("Number of patients in test set from albayzin: ", albayzin_test)

                neurovoz_train = len(
                    train_data[train_data["dataset"] == "neurovoz"][
                        "id_patient"
                    ].unique()
                )
                neurovoz_val = len(
                    val_data[val_data["dataset"] == "neurovoz"]["id_patient"].unique()
                )
                neurovoz_test = len(
                    test_data[test_data["dataset"] == "neurovoz"]["id_patient"].unique()
                )

                print("Number of patients in train set from neurovoz: ", neurovoz_train)
                print("Number of patients in val set from neurovoz: ", neurovoz_val)
                print("Number of patients in test set from neurovoz: ", neurovoz_test)

            # Create the dataloaders
            if self.plps:
                audio_features = "plps"
            if self.mfcc:
                audio_features = "mfccs"
            if self.spectrogram:
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

        return train_loader, val_loader, test_loader, train_data, val_data, test_data

    def normalize_audio(self, audio_data, max=None):
        if max is None:
            max_value = np.max(np.abs(audio_data))
        else:
            max_value = max
        normalized_data = audio_data / max_value

        return normalized_data
