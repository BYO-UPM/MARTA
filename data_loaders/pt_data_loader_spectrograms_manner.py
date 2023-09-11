import torch
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
from sklearn.preprocessing import StandardScaler
import torchaudio
from torchaudio import transforms


class Dataset_AudioFeatures(torch.utils.data.Dataset):
    def __init__(self, data_path, hyperparams):
        self.hyperparams = hyperparams
        self.plps = self.hyperparams["n_plps"] > 0
        self.mfcc = self.hyperparams["n_mfccs"] > 0
        self.spectrogram = self.hyperparams["spectrogram"]
        self.data_path = data_path

        # Check if the data has been already processed and saved
        name_save = (
            "local_results/data_frame"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + ".csv"
        )

        if os.path.exists(name_save):
            self.data = pd.read_csv(name_save)
        else:
            self.data = self.read_dataset(self.data_path)

    def __len__(self):
        return len(self.data)

    def read_manner(self, datapath):
        file_paths = []
        labels = []
        id_patient = []
        texts = []

        for text in os.listdir(datapath):
            # If the folder's name lenght is less than 3, is a vowel so skip it
            if len(text) < 3:
                continue
            # If os.path.data.join is not a folder, skip it
            if not os.path.isdir(os.path.join(datapath, text)):
                continue
            for condition in os.listdir(os.path.join(datapath, text)):
                for file in os.listdir(os.path.join(datapath, text, condition)):
                    # If the file does not end with .wav, skip it
                    if not file.endswith(".wav"):
                        continue
                    file_path = os.path.join(datapath, text, condition, file)
                    file_paths.append(file_path)
                    labels.append(condition)
                    # Each file is named as follows: <condition>_<text>_<id_patient>.wav
                    id_patient.append(file.split("_")[2].split(".")[0])
                    texts.append(text)
        # Generate a dataframe with all the data
        data = pd.DataFrame(
            {
                "file_path": file_paths,
                "label": labels,
                "text": texts,
                "id_patient": id_patient,
            }
        )
        # sort by id_patient
        data = data.sort_values(by=["id_patient"])
        # reset index
        data = data.reset_index(drop=True)

        return data

    def read_dataset(self, path_to_data):
        print("Reading the data...")

        data = self.read_manner(path_to_data)
        # Generate the folds by splitting the data by id_patient
        data["fold"] = data.groupby("id_patient").cumcount() % 10
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

        if self.spectrogram:
            # Calculate the spectrogram. We want that each spectrogram is 400ms long with 20 windows of 20ms each.
            win_length = int(
                data["sr"].iloc[0] * self.hyperparams["spectrogram_win_size"]
            )  # 20ms (default) to capture each phoneme
            hop_length = int(win_length * self.hyperparams["hop_size_percent"])

            n_fft = 2048
            n_mels = 65

            # Calculate spectorgram with torchaudio and normalize with transforms
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=data["sr"].iloc[0],
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm="slaney",
                onesided=True,
                mel_scale="htk",
            )
            # Power to db
            power_to_db = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=None
            )
            # Calculate the spectrogram for each frame
            data["spectrogram"] = data["signal_framed"].apply(
                lambda x: mel_spec(torch.tensor(x).unsqueeze(0))
            )
            # First, reshape the "spectrogram" column to be a list of tensors instead of a single tensor
            data["spectrogram"] = data["spectrogram"].apply(
                lambda x: x.reshape(-1, x.shape[2], x.shape[3])
            )

            # Explode the DataFrame by "n_frames"
            data = data.explode("spectrogram")
            data.reset_index(drop=True, inplace=True)

            # Calculate the power to db for each frame
            data["spectrogram"] = data["spectrogram"].apply(lambda x: power_to_db(x))

            # Normalise each spectrogram by substraction the mean and dividing by the standard deviation
            data["spectrogram"] = data["spectrogram"].apply(
                lambda x: (x - x.mean()) / x.std()
            )

        # Save data to this to not compute this again if it is not necessary. This is a heavy process.
        name_save = (
            "local_results/data_frame"
            + str(self.hyperparams["frame_size_ms"])
            + "spec_winsize_"
            + str(self.hyperparams["spectrogram_win_size"])
            + "hopsize_"
            + str(self.hyperparams["hop_size_percent"])
            + ".csv"
        )
        data.to_csv(name_save, index=False)
        return data

    def get_dataloaders(self, fold=0):
        print("Splitting in train, test and validation sets...")
        # Split the data into train, validation and test sets
        train_data = self.data[self.data["fold"] != fold]
        test_data = self.data[self.data["fold"] == fold]

        # Split the train data into train and validation sets
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42, stratify=train_data["label"]
        )

        # Create the dataloaders
        label_counts = train_data["label"].value_counts()
        total_samples = len(train_data)
        class_weights = 1.0 / torch.Tensor(label_counts.values / total_samples)
        sample_weights = class_weights[train_data["label"].values]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
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

        x_val = np.stack(val_data[audio_features])
        x_val = np.expand_dims(x_val, axis=1)
        y_val = val_data["label"].values

        x_test = np.stack(test_data[audio_features])
        x_test = np.expand_dims(x_test, axis=1)
        y_test = test_data["label"].values

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
                )
            ),
            drop_last=False,
            batch_size=self.hyperparams["batch_size"],
            sampler=sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_val,
                    y_val,
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
