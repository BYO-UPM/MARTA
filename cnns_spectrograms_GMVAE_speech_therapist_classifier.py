from models.pt_models import SpeechTherapist
from training.pt_training import SpeechTherapist_trainer, SpeechTherapist_tester
from utils.utils import (
    plot_logopeda,
    calculate_distances_manner,
    plot_logopeda_alb_neuro,
)
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import numpy as np
import sys
import os
import random
from collections import Counter
import copy
import argparse


def find_minority_class(dataset):
    labels = [data[1] for data in dataset]
    label_counts = Counter(labels)
    minority_class = min(label_counts, key=label_counts.get)
    return minority_class, label_counts


def augment_data(dataset, validation=False):
    augmented_data = []
    for data in dataset:
        if validation:
            spectrogram, label, manner, ds, id = data
        else:
            spectrogram, label, manner, ds = data

        spectrogram_to_modify1 = copy.deepcopy(spectrogram)
        spectrogram_to_modify2 = copy.deepcopy(spectrogram)

        # First augmentation
        augmented_spectrogram_1 = augment_spectrogram(
            spectrogram_to_modify1, p=0.8, q=0.8, r=0.2
        )
        if validation:
            augmented_data.append((augmented_spectrogram_1, label, manner, ds, id))
        else:
            augmented_data.append((augmented_spectrogram_1, label, manner, ds))

        # Second augmentation
        augmented_spectrogram_2 = augment_spectrogram(
            spectrogram_to_modify2, p=0.8, q=0.8, r=0.2
        )
        if validation:
            augmented_data.append((augmented_spectrogram_2, label, manner, ds, id))
        else:
            augmented_data.append((augmented_spectrogram_2, label, manner, ds))

    dataset += augmented_data

    return dataset


# Function for frequency-based data augmentation
def augment_spectrogram(spectrogram, p=0.5, q=0.5, r=0.2):
    _, freq_dimension, time_dimension = spectrogram.shape

    # With probability p, mask 15% of frequency bands
    if random.random() < p:
        mask_percentage = 0.15
        mask_size = int(freq_dimension * mask_percentage)
        mask_start = random.randint(0, freq_dimension - mask_size)
        spectrogram[0, mask_start : mask_start + mask_size, :] = 0

    # With probability q, mask 15% of time windows
    if random.random() < q:
        mask_percentage = 0.15
        mask_size = int(time_dimension * mask_percentage)
        mask_start = random.randint(0, time_dimension - mask_size)
        spectrogram[0, :, mask_start : mask_start + mask_size] = 0

    # With probability r, add Gaussian noise
    if random.random() < r:
        noise = np.random.normal(0, 1, spectrogram.shape)
        spectrogram += 0.10 * noise
        # Renormalize the spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

    return spectrogram


def stratify_dataset(dataset):
    # Check stratification of labels (they are the second element of the triplet)
    minority_class, label_counts = find_minority_class(dataset)
    majority_class_count = max(label_counts.values())

    if label_counts[minority_class] < majority_class_count:
        additional_augmented_data = []
        augmentations_needed = majority_class_count - label_counts[minority_class]
        minority_class_data = [data for data in dataset if data[1] == minority_class]

        while augmentations_needed > 0:
            for data in minority_class_data:
                if augmentations_needed <= 0:
                    break
                spectrogram, label, manner, ds = data

                spectrogram_to_modify3 = copy.deepcopy(spectrogram)

                augmented_spectrogram = augment_spectrogram(spectrogram_to_modify3)

                additional_augmented_data.append(
                    (augmented_spectrogram, label, manner, ds)
                )
                augmentations_needed -= 1

        # Combine additional augmented data
        balanced_dataset = dataset + additional_augmented_data
    else:
        balanced_dataset = dataset

    return balanced_dataset


def make_sampler(dataset, validation=False):
    # Count the occurrences of each class
    class_counts = {}
    if validation:
        dataset = [data[:4] for data in dataset]

    for _, label, _, _ in dataset:
        label = label.item()  # Assuming label is a tensor
        class_counts[label] = class_counts.get(label, 0) + 1

    # Assign weights inversely proportional to class frequencies
    weights = []
    for _, label, _, _ in dataset:
        label = label.item()
        weight = 1.0 / class_counts[label]
        weights.append(weight)

    # Create a WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return sampler


def main(args, hyperparams):
    gpu = args.gpu
    fold = args.fold
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    hyperparams["path_to_save"] = (
        "local_results/spectrograms/manner_gmvae_alb_neurovoz_"
        + str(hyperparams["latent_dim"])
        + "final_model_classifier"
        + "_LATENTSPACE+manner_CNN_fold"
        + str(hyperparams["fold"])
    )

    # Create the path if does not exist
    if not os.path.exists(hyperparams["path_to_save"]):
        os.makedirs(hyperparams["path_to_save"])

    old_stdout = sys.stdout
    log_file = open(hyperparams["path_to_save"] + "/log.txt", "w")
    sys.stdout = log_file

    if hyperparams["wandb_flag"]:
        gname = (
            "SPECTROGRAMS_GMVAE_"
            + hyperparams["material"]
            + "_final_model_unsupervised"
        )
        wandb.finish()
        wandb.init(
            project="parkinson",
            config=hyperparams,
            group=gname,
        )

    # First check in local_results/ if there eist any .pt file with the dataloaders
    # If not, create them and save them in local_results/

    if not hyperparams["new_data_partition"]:
        print("Reading train, val and test loaders from local_results/...")
        train_loader = torch.load(
            "local_results/folds/train_loader_supervised_True_frame_size_0.4spec_winsize_0.023hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        val_loader = torch.load(
            "local_results/folds/val_loader_supervised_True_frame_size_0.4spec_winsize_0.023hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        test_loader = torch.load(
            "local_results/folds/test_loader_supervised_True_frame_size_0.4spec_winsize_0.023hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        test_data = torch.load(
            "local_results/folds/test_data_supervised_True_frame_size_0.4spec_winsize_0.023hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )

        # Remove all albayzin samples from train_loader
        new_train = [data for data in train_loader.dataset if data[3] == "neurovoz"]

        # First, remove all albayzin samples from val_loader
        new_val = [data for data in val_loader.dataset if data[3] == "neurovoz"]

        # Split the new val dataset into train and validation. Use "id_patient" to ensure that all samples from the same patient are in the same set
        # patients = set([data[4] for data in new_val])
        # patients = list(patients)
        # np.random.shuffle(patients)
        # train_patients = patients[: int(0.6 * len(patients))]
        # val_patients = patients[int(0.6 * len(patients)) :]
        # new_train = [data[:4] for data in new_val if data[4] in train_patients]
        # new_val = [data for data in new_val if data[4] in val_patients]

        # Augment the train dataset
        extended_dataset = augment_data(new_train)

        # # Agment the validation dataset
        # new_val = augment_data(new_val, validation=True)

        # Stratify train dataset
        balanced_dataset = stratify_dataset(extended_dataset)

        train_sampler = make_sampler(balanced_dataset)
        val_sampler = make_sampler(new_val, validation=True)

        # Create new dataloaders
        new_train_loader = torch.utils.data.DataLoader(
            balanced_dataset,
            batch_size=val_loader.batch_size,
            sampler=train_sampler,
        )
        val2_loader = torch.utils.data.DataLoader(
            new_val,
            batch_size=val_loader.batch_size,
            sampler=val_sampler,
        )

        #
    else:
        print("Reading data...")
        # Read the data

        dataset = Dataset_AudioFeatures(
            "labeled/NeuroVoz",
            hyperparams,
        )
        print("Creating train, val and test loaders...")
        (
            train_loader,
            val_loader,
            test_loader,
            _,  # train_data, not used
            _,  # val_data, not used
            test_data,
        ) = dataset.get_dataloaders(
            train_albayzin=hyperparams["train_albayzin"],
            supervised=hyperparams["supervised"],
        )

    print("Defining models...")
    # Create the model
    model = SpeechTherapist(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_gaussians=hyperparams["n_gaussians"],
        n_manner=16,
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        classifier=hyperparams["classifier_type"],
        weights=hyperparams["weights"],
        device=device,
    )

    # model = torch.compile(model)

    if hyperparams["train"]:
        # Load the best unsupervised model to supervise it
        name = (
            "local_results/spectrograms/results_90_10_da_latentspace+manner_cnn/manner_gmvae_alb_neurovoz_32supervised90-10-fold"
            + str(hyperparams["fold"])
            + "/GMVAE_cnn_best_model_2d.pt"
        )
        tmp = torch.load(name)
        model.load_state_dict(tmp["model_state_dict"])

        # Freeze all the network
        for param in model.parameters():
            param.requires_grad = False
        # Add a classifier to the model
        model.class_dims = [64, 32, 16]
        model.classifier()
        # Unfreeze the classifier
        for param in model.hmc.parameters():
            # Unfreezing the manner class embeddings
            param.requires_grad = False
        for param in model.clf_cnn.parameters():
            # Unfreezing the cnn classifier
            param.requires_grad = True
        for param in model.clf_mlp.parameters():
            # Unfreezing the mlp classifier
            param.requires_grad = True
        model.to(device)

        print("Training GMVAE...")
        # Train the model
        SpeechTherapist_trainer(
            model=model,
            trainloader=new_train_loader,
            validloader=val2_loader,
            epochs=hyperparams["epochs"],
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
            path_to_save=hyperparams["path_to_save"],
            supervised=hyperparams["supervised"],
            classifier=hyperparams["classifier"],
        )

        print("Training finished!")
    else:
        print("Loading model...")

    # Restoring best model
    name = hyperparams["path_to_save"] + "/GMVAE_cnn_best_model_2d.pt"
    tmp = torch.load(name)
    model.load_state_dict(tmp["model_state_dict"])

    print("Testing GMVAE...")

    # Read the best threshold
    path = hyperparams["path_to_save"] + "/best_threshold.txt"
    with open(path, "r") as f:
        threshold = float(f.read())

    # Test the model
    SpeechTherapist_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        supervised=True,  # Not implemented yet
        wandb_flag=hyperparams["wandb_flag"],
        path_to_plot=hyperparams["path_to_save"],
        best_threshold=threshold,
    )
    print("Testing finished!")

    # Create an empty pd dataframe with three columns: data, label and manner
    # df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    # df_train[audio_features] = [t[0] for t in train_loader.dataset]
    # df_train["label"] = [t[1] for t in train_loader.dataset]
    # df_train["manner"] = [t[2] for t in train_loader.dataset]

    # # Select randomly 1000 samples of dftrain
    # # df_train = df_train.sample(n=1000)

    # # Create an empty pd dataframe with three columns: data, label and manner
    # df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    # df_test[audio_features] = [t[0] for t in test_loader.dataset]
    # df_test["label"] = [t[1] for t in test_loader.dataset]
    # df_test["manner"] = [t[2] for t in test_loader.dataset]

    # print("Starting to calculate distances...")
    # plot_logopeda_alb_neuro(
    #     model,
    #     df_train,
    #     df_test,
    #     hyperparams["wandb_flag"],
    #     name="test",
    #     supervised=hyperparams["supervised"],
    #     samples=5000,
    #     path_to_plot=hyperparams["path_to_save"],
    # )

    if hyperparams["wandb_flag"]:
        wandb.finish()

    sys.stdout = old_stdout
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument(
        "--fold", type=int, default=1, help="Fold number for the experiment"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU number for the experiment"
    )

    args = parser.parse_args()

    hyperparams = {
        "frame_size_ms": 0.400,  # 400ms
        "n_plps": 0,
        "n_mfccs": 0,
        "spectrogram_win_size": 0.023,  # 23ms as it is recommended in the librosa library for speech recognition
        "material": "MANNER",
        "hop_size_percent": 0.5,
        "spectrogram": True,
        "wandb_flag": False,
        "epochs": 500,
        "batch_size": 128,
        "lr": 1e-3,
        "latent_dim": 32,
        "hidden_dims_enc": [64, 1024, 64],
        "hidden_dims_gmvae": [256],
        "weights": [
            1,  # w1 is rec loss,
            1,  # w2 is gaussian kl loss,
            1,  # w3 is categorical kl loss,
            10,  # w5 is metric loss
        ],
        "classifier_type": "cnn",  # "cnn" or "mlp"
        "n_gaussians": 16,  # 2 per manner class
        "semisupervised": False,
        "train": True,
        "train_albayzin": True,
        "new_data_partition": False,
        "supervised": True,
        "classifier": True,
        "fold": args.fold,
        "gpu": args.gpu,
    }

    main(args, hyperparams)
