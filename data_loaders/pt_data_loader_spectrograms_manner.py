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


# Function to collapse the matrix into a 33x1 vector with the most repeated string
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

        datapath_wav = (
            "/media/my_ftp/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/SUB_APRE_WAV/"
        )
        datapath_textgrid = (
            "/media/my_ftp/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/textgrid/"
        )

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
            phonemes.append(tg_file["phones"])

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
        # Read the .wav files and store the signals and sampling rates
        data["signal"], data["sr"] = zip(*data["file_path"].map(librosa.load))

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
        t1 = time.time()
        # Match phonemes
        data["phonemes_matched"] = data.apply(
            lambda x: self.match_phonemes(x["phonemes"], x["signal"], x["sr"]), axis=1
        )

        t2 = time.time()
        print("Time to match phonemes: ", t2 - t1)

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
            n_mels = 65

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

    def get_dataloaders(self, fold=0):
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
            "J": 2,  # enye
            "f": 3,  # fricatives
            "s": 3,
            "z": 3,
            "x": 3,
            "h": 3,
            "T": 3,  # theta
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

        print("Splitting in train, test and validation sets...")
        # Split the data into train and test. We will train only with albayzin data
        train_data = self.data[self.data["dataset"] == "albayzin"]
        test_data = self.data[self.data["dataset"] != "albayzin"]

        # Split the train data into train and validation sets
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42, stratify=train_data["text"]
        )

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

        x_val = np.stack(val_data[audio_features])
        x_val = np.expand_dims(x_val, axis=1)
        y_val = val_data["label"].values
        p_val = np.array([np.array(x) for x in val_data["manner_class"]])
        z_val = np.array(
            [np.array([np.repeat(x, len(y)), y]) for x, y in zip(y_val, p_val)]
        )
        z_val = np.array([np.array(x[1] + n_classes * x[0]) for x in z_val])

        x_test = np.stack(test_data[audio_features])
        x_test = np.expand_dims(x_test, axis=1)
        y_test = test_data["label"].values
        p_test = np.array([np.array(x) for x in test_data["manner_class"]])
        z_test = np.array(
            [np.array([np.repeat(x, len(y)), y]) for x, y in zip(y_test, p_test)]
        )
        z_test = np.array([np.array(x[1] + n_classes * x[0]) for x in z_test])

        # Normalise the spectrograms which are 2D using standard scaler
        std = StandardScaler()

        x_train = np.stack(
            [
                std.fit_transform(x.reshape(-1, x.shape[1])).reshape(x.shape)
                for x in x_train
            ]
        )
        x_val = np.stack(
            [std.transform(x.reshape(-1, x.shape[1])).reshape(x.shape) for x in x_val]
        )
        x_test = np.stack(
            [std.transform(x.reshape(-1, x.shape[1])).reshape(x.shape) for x in x_test]
        )

        # Min max scaler between -1 and 1
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # x_train = scaler.fit_transform(x_train)
        # x_val = scaler.transform(x_val)
        # x_test = scaler.transform(x_test)

        print("Creating dataloaders...")
        train_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_train,
                    y_train,
                    p_train,
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
                )
            ),
            drop_last=False,
            batch_size=self.hyperparams["batch_size"],
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_test,
                    y_test,
                    p_test,
                )
            ),
            drop_last=False,
            batch_size=self.hyperparams["batch_size"],
            shuffle=False,
        )

        return train_loader, val_loader, test_loader, train_data, val_data, test_data

    def normalize_audio(self, audio_data, max=None):
        if max is None:
            max_value = np.max(np.abs(audio_data))
        else:
            max_value = max
        normalized_data = audio_data / max_value

        return normalized_data
