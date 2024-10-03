"""
Unsupervised GMVAE (MARTA) for Parkinson's Disease Analysis

This script implements an unsupervised Gaussian Mixture Variational Autoencoder (GMVAE), 
named MARTA, for analyzing speech features related to Parkinson's Disease. The model is 
trained exclusively on healthy patients and then used to evaluate the latent space 
distances between healthy and Parkinsonian patients (who were never seen by the model 
during training). This approach aims to identify distinct patterns in speech that could 
be indicative of Parkinson's Disease.

Main Components:
1. Data Preparation: Uses 'Dataset_AudioFeatures' for loading and preprocessing spectrogram data.
2. Model Definition: Sets up the MARTA model with specified hyperparameters.
3. Training: Trains the MARTA model in an unsupervised manner using healthy patient data.
4. Evaluation: Evaluates the model by computing distances in the latent space between healthy 
   and Parkinsonian spectrograms.
5. Visualization: Plots and analyzes the results to understand the model's performance.

Usage:
- The script is configured for running in environments with CUDA-capable devices.
- Hyperparameters for the model and training process are adjustable.

Key Functions:
- MARTA_trainer: Handles the training loop of the MARTA model.
- MARTA_tester: Manages the testing and evaluation of the model.
- plot_logopeda_alb_neuro: Visualization utility for plotting model outputs and analysis.

Output:
- Trained GMVAE model capable of distinguishing between healthy and Parkinsonian speech features.
- Log files and performance metrics for model training and testing.
- Visualizations of latent space representations and distances.

Requirements:
- Torch, pandas, and other dependencies for model building and data handling.
- Properly structured and preprocessed data in expected formats.

Author: Guerrero-López, Alejandro 
Date: 25/01/2024

Note:
- The script assumes a specific structure and format for input data.
- The hyperparameters and model configurations may need tuning based on the specific characteristics 
  of the input data and the computational resources available.
"""

from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester
from utils.utils import plot_logopeda_alb_neuro, stratify_per_dataset
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import pandas as pd
import sys
import os
import numpy as np

# Select the free GPU if there is one available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


def main(args, hyperparams):

    hyperparams["path_to_save"] = (
        "local_results/spectrograms/marta_"
        + str(hyperparams["latent_dim"])
        + "unsupervised"
        + "_latentdim_"
        + str(hyperparams["latent_dim"])
        + "_gaussians_"
        + str(hyperparams["n_gaussians"])
        + "_experiment_"
        + str(hyperparams["experiment"])
    )

    # Create the path if does not exist
    if not os.path.exists(hyperparams["path_to_save"]):
        os.makedirs(hyperparams["path_to_save"])

    old_stdout = sys.stdout
    log_file = open(hyperparams["path_to_save"] + "/log.txt", "w")
    sys.stdout = log_file

    if hyperparams["new_data_partition"]:
        print("Reading data...")
        # Read the data
        dataset = Dataset_AudioFeatures(
            hyperparams,
        )
        (
            train_loader,
            val_loader,
            test_loader,
            train_data,  # train_data, not used
            val_data,  # val_data, not used
            test_data,
        ) = dataset.get_dataloaders(experiment=hyperparams["experiment"])
    else:
        print("Reading train, val and test loaders from local_results/...")
        train_loader = torch.load(
            "local_results/folds/train_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )
        val_loader = torch.load(
            "local_results/folds/val_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )
        test_loader = torch.load(
            "local_results/folds/test_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )
        test_data = torch.load(
            "local_results/folds/test_data_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )

    # Modify the dataloaders to have, in train, only healthy patients and move all parkinsonan  to test

    # Remove from training all parkinsonian
    parkinsonian_train = [i for i, t in enumerate(train_loader.dataset) if t[1] == 1]
    train_dataset = torch.utils.data.Subset(
        train_loader.dataset,
        [i for i in range(len(train_loader.dataset)) if i not in parkinsonian_train],
    )
    # Do the same for train_data which is a dataframe
    train_data_park = train_data.loc[train_data["label"] == 1]
    train_data = train_data.loc[train_data["label"] == 0]

    # Remove from validation all parkinsonian
    parkinsonian_val = [i for i, t in enumerate(val_loader.dataset) if t[1] == 1]
    val_dataset = torch.utils.data.Subset(
        val_loader.dataset,
        [i for i in range(len(val_loader.dataset)) if i not in parkinsonian_val],
    )
    val_data_park = val_data.loc[val_data["label"] == 1]
    val_data = val_data.loc[val_data["label"] == 0]

    # Add to test_loader all train and val parkinsonian
    parkinsonian_train_val = parkinsonian_train + parkinsonian_val
    test_dataset = torch.utils.data.ConcatDataset(
        [
            test_loader.dataset,
            torch.utils.data.Subset(train_loader.dataset, parkinsonian_train_val),
        ]
    )
    park_data = train_data_park.append(val_data_park)
    test_data = test_data.append(park_data)

    # Stratify dataset to have same number of samples per dataset
    train_dataset = stratify_per_dataset(train_dataset)

    # Redefine the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    print("Defining models...")
    # Create the model
    model = MARTA(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_gaussians=hyperparams["n_gaussians"],
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        weights=hyperparams["weights"],
        device=device,
        reducer="sum",
        domain_adversarial_bool=True,
    )

    # model = torch.compile(model)

    if hyperparams["train"]:
        print("Training GMVAE...")
        # Train the model
        MARTA_trainer(
            model=model,
            trainloader=train_loader,
            validloader=val_loader,
            epochs=hyperparams["epochs"],
            wandb_flag=None,
            lr=hyperparams["lr"],
            path_to_save=hyperparams["path_to_save"],
            supervised=False,
            classifier=False,
        )

        print("Training finished!")
    else:
        print("Loading model...")

    # Restoring best model
    name = hyperparams["path_to_save"] + "/GMVAE_cnn_best_model_2d.pt"
    tmp = torch.load(name)
    model.load_state_dict(tmp["model_state_dict"])

    audio_features = "spectrogram"
    print("Testing GMVAE...")

    # Drop all samples that have any nan in the test_loader.dataset and count how many
    # Initialize a list to hold indices of samples to be removed
    indices_to_remove = []

    # Iterate through the dataset to find samples with nan values
    for i, t in enumerate(test_loader.dataset):
        if np.isnan(t[0]).any() or np.isnan(t[1]).any() or np.isnan(t[2]).any():
            indices_to_remove.append(i)

    # Print the number of samples to be removed
    print(f"Number of samples with nan values to be removed: {len(indices_to_remove)}")

    # Remove samples with nan values from the dataset
    new_test = [
        t for i, t in enumerate(test_loader.dataset) if i not in indices_to_remove
    ]

    test_loader = torch.utils.data.DataLoader(
        new_test, batch_size=hyperparams["batch_size"], shuffle=False
    )

    # Test the model
    # MARTA_tester(
    #     model=model,
    #     testloader=test_loader,
    #     test_data=test_data,
    #     supervised=False,
    #     wandb_flag=None,
    #     path_to_plot=hyperparams["path_to_save"],
    # )
    # print("Testing finished!")

    # Create an empty pd dataframe with three columns: data, label and manner
    df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_train[audio_features] = [t[0] for t in train_loader.dataset]
    df_train["label"] = [t[1] for t in train_loader.dataset]
    df_train["manner"] = [t[2] for t in train_loader.dataset]
    df_train["dataset"] = [t[3] for t in train_loader.dataset]

    # Create an empty pd dataframe with three columns: data, label and manner
    df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_test[audio_features] = [t[0] for t in test_loader.dataset]
    df_test["label"] = [t[1] for t in test_loader.dataset]
    df_test["manner"] = [t[2] for t in test_loader.dataset]
    df_test["dataset"] = [t[3] for t in test_loader.dataset]

    # Plot 5 spectrograms from plosives from "italian" and "neurovoz"

    italian = df_test[df_test["dataset"] == "italian"]
    neurovoz = df_test[df_test["dataset"] == "neurovoz"]
    gita = df_test[df_test["dataset"] == "gita"]
    albayzin = df_test[df_test["dataset"] == "albayzin"]

    print("Starting to calculate distances...")
    plot_logopeda_alb_neuro(
        model,
        df_train,
        df_test,
        None,
        name="test",
        supervised=hyperparams["supervised"],
        samples=5000,
        path_to_plot=hyperparams["path_to_save"],
    )

    sys.stdout = old_stdout
    log_file.close()


if __name__ == "__main__":
    args = {}

    hyperparams = {
        # ================ Spectrogram parameters ===================
        "spectrogram": True,  # If true, use spectrogram.
        "frame_size_ms": 0.400,  # Size of each spectrogram frame
        "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
        "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
        # ================ GMVAE parameters ===================
        "epochs": 500,  # Number of epochs to train the model (at maximum, we have early stopping)
        "batch_size": 512,  # Batch size
        "lr": 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
        "latent_dim": 3,  # Latent dimension of the z vector (remember it is also the input to the classifier)
        "n_gaussians": 16,  # Number of gaussians in the GMVAE
        "hidden_dims_enc": [
            64,
            1024,
            64,
        ],  # Hidden dimensions of encoder/decoder (from audio framed to spectrogram and viceversa)
        "hidden_dims_gmvae": [256],  # Hidden dimensions of the GMVAE encoder/decoder
        "weights": [  # Weights for the different losses
            1,  # w1 is rec loss,
            1,  # w2 is gaussian kl loss,
            1,  # w3 is categorical kl loss,
            10,  # w5 is metric loss
        ],
        # ================ Experiment parameters ===================
        "experiment": "fourth",  # Experiment name, from 1 to 6. It is used to load the correct data.
        # ================ Classifier parameters ===================
        "cnn_classifier": False,  # Here no classifier is used
        "supervised": False,  # Here no classifier is used
        # ================ Training parameters ===================
        "train": False,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "new_data_partition": True,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
    }

    main(args, hyperparams)
