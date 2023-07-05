import torch
import numpy as np
import os
from .rasta_py.rasta import rastaplp
from scipy.io import wavfile
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
from sklearn.preprocessing import StandardScaler


class Dataset_PLPs(torch.utils.data.Dataset):
    def __init__(self, data_path, hyperparams, material):
        self.hyperparams = hyperparams
        self.material = material
        if self.material == "PATAKA":
            self.data_path = os.path.join(data_path, self.material)
        elif self.material == "VOWELS":
            self.data_path = data_path
        else:
            raise ValueError("Material not recognized")
        self.data = self.read_dataset(self.data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the data for the given index
        data = self.data.iloc[index]

        # Get the plps
        plps = data["plps"]

        # Get the label
        label = data["label"]

        # Convert the plps to a numpy array
        plps = np.array(plps)

        # Convert the plps to a torch tensor
        plps = torch.from_numpy(plps).float()

        # Return the plps and the label
        return plps, label

    def read_pataka(self, path_to_data):
        # Initialize lists to store file paths and corresponding labels
        file_paths = []
        labels = []
        id_patient = []

        # List all the folders in the path_to_data directory
        folders = os.listdir(path_to_data)

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
                    # Each file is named as follows: <condition>_<PATAKA>_<id_patient>.wav
                    id_patient.append(file.split("_")[2].split(".")[0])
        # Create a dataframe with all the data
        data = pd.DataFrame(
            {"file_path": file_paths, "label": labels, "id_patient": id_patient}
        )

        return data

    def read_vowels(self, path_to_data):
        file_paths = []
        labels = []
        vowels = []
        id_patient = []

        # Read the error files it is a txt file
        error_files = pd.read_csv("error_files.txt", header=None)
        # Remove the path get only the filename
        error_files = error_files[0].apply(lambda x: x.split("/")[-1]).to_list()

        folders = [
            "A1",
            "A2",
            "A3",
            "E1",
            "E2",
            "E3",
            "I1",
            "I2",
            "I3",
            "O1",
            "O2",
            "O3",
            "U1",
            "U2",
            "U3",
        ]
        for folder_vowel in folders:
            folder_path = os.path.join(path_to_data, folder_vowel)
            for folder in os.listdir(folder_path):
                folder_condition = os.path.join(path_to_data, folder_vowel, folder)
                if os.path.isdir(folder_condition):
                    label = folder
                    for file in os.listdir(folder_condition):
                        # If the file does not end with .wav, skip it
                        if not file.endswith(".wav"):
                            continue
                        # If the file is an error file, skip it
                        if file in error_files:
                            print("Skipping file: ", file)
                            continue
                        file_path = os.path.join(folder_condition, file)
                        file_paths.append(file_path)
                        labels.append(label)
                        vowel = [*folder_vowel][0]
                        vowels.append(vowel)
                        # Each file is named as follows: <condition>_<vowel>_<id_patient>.wav
                        id_patient.append(file.split("_")[2].split(".")[0])
        # Generate a dataframe with all the data
        data = pd.DataFrame(
            {
                "file_path": file_paths,
                "label": labels,
                "vowel": vowels,
                "id_patient": id_patient,
            }
        )
        # sort by id_patient
        data = data.sort_values(by=["id_patient"])
        # reset index
        data = data.reset_index(drop=True)

        return data

    def read_dataset(self, path_to_data, plps=False, mfcc=False):
        print("Reading the data...")

        if self.material == "PATAKA":
            data = self.read_pataka(path_to_data)
            data["fold"] = data.groupby("label").cumcount() % 10
        elif self.material == "VOWELS":
            data = self.read_vowels(path_to_data)
            # Generate the folds by splitting the data by id_patient
            data["fold"] = data.groupby("id_patient").cumcount() % 10
            # Categorise the vowels to 0,1,2,3,4
            data["vowel"] = data["vowel"].astype("category").cat.codes

        print("Reading the .wav files...")
        # Read the .wav files and store the signals and sampling rates
        data["signal"], data["sr"] = zip(*data["file_path"].map(librosa.load))

        # ======= Checking errors in loading files =======
        # sr_list = []
        # signal_list = []
        # error_files = []
        # for file in data["file_path"]:
        #     try:
        #         signal, sr = librosa.load(file)
        #         sr_list.append(sr)
        #         signal_list.append(signal)
        #     except:
        #         sr_list.append(0 * sr_list[-1])
        #         signal_list.append(0 * signal_list[-1])
        #         error_files.append(file)
        #         print("Error reading file: ", file)

        # data["sr"] = sr_list
        # data["signal"] = signal_list

        # Store in a txt file the files that could not be read
        # with open("error_files.txt", "w") as f:
        #     for file in error_files:
        #         f.write("%s\n" % file)
        # Print the total number of files that could not be read
        # print("Total number of files that could not be read: ", len(error_files))
        # # Check how many samples per patient are not read
        # print(
        #     "Number of samples per patient that could not be read: ",
        #     data[data["sr"] == 0]["id_patient"].value_counts(),
        # )
        # # Remove the files that could not be read
        # data = data[data["sr"] != 0]
        # ======= END =======

        print("Normalising the signals...")
        data["max"] = data["signal"].apply(np.abs).apply(np.max)
        data["norm_signal"] = data.apply(
            lambda x: self.normalize_audio(x["signal"]), axis=1
        )

        if plps:
            print("Extracting RASTA-PLP features...")
            # Extract the RASTA-PLP features
            data["plps"] = data.apply(
                lambda x: self.extract_rasta_plp_with_derivatives(
                    x["norm_signal"],
                    x["sr"],
                    self.hyperparams["frame_size_ms"],
                    self.hyperparams["n_plps"],
                ),
                axis=1,
            )
        if mfcc:
            print("Extracting MFCC features...")
            # Extract the MFCC features
            data["mfcc"] = data.apply(
                lambda x: self.extract_mfcc_with_derivatives(
                    x["norm_signal"],
                    x["sr"],
                    self.hyperparams["frame_size_ms"],
                    self.hyperparams["n_mfcc"],
                ),
                axis=1,
            )

        # Binarise the labels
        data["label"] = data["label"].apply(lambda x: 1 if x == "PD" else 0)

        print("Exploding data...")
        # Data explode
        if plps:
            data = data.explode("plps")
        if mfcc:
            data = data.explode("mfcc")

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
        x_train = np.vstack(train_data["plps"])
        y_train = train_data["label"].values
        z_train = train_data["vowel"].values

        x_val = np.vstack(val_data["plps"])
        y_val = val_data["label"].values
        z_val = val_data["vowel"].values

        x_test = np.vstack(test_data["plps"])
        y_test = test_data["label"].values
        z_test = test_data["vowel"].values

        # Normalise the data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)

        print(train_data.columns)
        print("Creating dataloaders...")
        train_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_train,
                    y_train,
                    z_train,
                )
            ),
            batch_size=self.hyperparams["batch_size"],
            sampler=sampler,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_val,
                    y_val,
                    z_val,
                )
            ),
            batch_size=self.hyperparams["batch_size"],
            shuffle=False,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=list(
                zip(
                    x_test,
                    y_test,
                    z_test,
                )
            ),
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

    def extract_rasta_plp_with_derivatives(
        self, audio, sample_rate, frame_length_ms, n_plps=10
    ):
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

        return plps.T

    def extract_mfcc_with_derivatives(audio, sample_rate, frame_length_ms, n_mfcc=10):
        frame_length = int(
            sample_rate * frame_length_ms * 1e-3
        )  # Convert ms to samples
        hop_length = int(frame_length / 2)  # 50% overlap
        frames = librosa.util.frame(
            audio, frame_length=frame_length, hop_length=hop_length
        )
        # Apply hanning windows
        frames = frames * np.hanning(frame_length)[:, None]

        # N_fft is the next number in power of 2 of the frame size
        n_fft = 2 ** (int(np.log2(frames.shape[1])) + 1)
        # Compute MFCC for each frame
        mfccs = []
        for frame in frames:
            mfcc = librosa.feature.mfcc(
                y=frame, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft
            )
            mfccs.append(mfcc)

        mfccs = np.hstack(mfccs)

        # Normalize the MFCCs
        # Compute the mean of each feature
        mean = np.mean(mfcc, axis=0)

        # Compute the variance of each feature
        var = np.var(mfcc, axis=0)

        # Noramlize
        mfcc = (mfcc - mean) / np.sqrt(var)

        # Compute derivatives
        mfccs_delta = librosa.feature.delta(mfccs)
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        # Concatenate the features
        mfcc_features = np.concatenate((mfccs, mfccs_delta, mfccs_delta2))

        return mfcc_features.T
