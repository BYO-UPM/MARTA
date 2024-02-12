"""
Unsupervised GMVAE (SThVAE) for Parkinson's Disease Analysis

This script implements an unsupervised Gaussian Mixture Variational Autoencoder (GMVAE), 
named SThVAE, for analyzing speech features related to Parkinson's Disease. The model is 
trained exclusively on healthy patients and then used to evaluate the latent space 
distances between healthy and Parkinsonian patients (who were never seen by the model 
during training). This approach aims to identify distinct patterns in speech that could 
be indicative of Parkinson's Disease.

Main Components:
1. Data Preparation: Uses 'Dataset_AudioFeatures' for loading and preprocessing spectrogram data.
2. Model Definition: Sets up the SThVAE model with specified hyperparameters.
3. Training: Trains the SThVAE model in an unsupervised manner using healthy patient data.
4. Evaluation: Evaluates the model by computing distances in the latent space between healthy 
   and Parkinsonian spectrograms.
5. Visualization: Plots and analyzes the results to understand the model's performance.

Usage:
- The script is configured for running in environments with CUDA-capable devices.
- Hyperparameters for the model and training process are adjustable.

Key Functions:
- SpeechTherapist_trainer: Handles the training loop of the SThVAE model.
- SpeechTherapist_tester: Manages the testing and evaluation of the model.
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

from models.pt_models import SpeechTherapist
from training.pt_training import SpeechTherapist_trainer, SpeechTherapist_tester
from utils.utils import (
    plot_logopeda_alb_neuro,
)
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import pandas as pd
import sys
import os

# Select the free GPU if there is one available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


def main(args, hyperparams):
    if hyperparams["train_albayzin"]:
        hyperparams["path_to_save"] = (
            "local_results/spectrograms/manner_gmvae_alb_neurovoz_"
            + str(hyperparams["latent_dim"])
            + "unsupervised"
            + "32d_final_model2"
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

    if hyperparams["train"] and hyperparams["new_data_partition"]:
        print("Reading data...")
        # Read the data
        dataset = Dataset_AudioFeatures(
            "labeled/NeuroVoz",
            hyperparams,
        )
        (
            train_loader,
            val_loader,
            test_loader,
            _,  # train_data, not used
            _,  # val_data, not used
            test_data,
        ) = dataset.get_dataloaders(train_albayzin=hyperparams["train_albayzin"])
    else:
        print("Reading train, val and test loaders from local_results/...")
        train_loader = torch.load(
            "local_results/folds/folds30ms/train_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5fold0.pt"
        )
        val_loader = torch.load(
            "local_results/folds/folds30ms/val_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5fold0.pt"
        )
        test_loader = torch.load(
            "local_results/folds/folds30ms/test_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5fold0.pt"
        )
        test_data = torch.load(
            "local_results/folds/folds30ms/test_data_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5fold0.pt"
        )

    print("Defining models...")
    # Create the model
    model = SpeechTherapist(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_gaussians=hyperparams["n_gaussians"],
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        weights=hyperparams["weights"],
        device=device,
        reducer="sum",
    )

    # model = torch.compile(model)

    if hyperparams["train"]:
        print("Training GMVAE...")
        # Train the model
        SpeechTherapist_trainer(
            model=model,
            trainloader=train_loader,
            validloader=val_loader,
            epochs=hyperparams["epochs"],
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
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

    # Test the model
    SpeechTherapist_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        supervised=False,
        wandb_flag=hyperparams["wandb_flag"],
        path_to_plot=hyperparams["path_to_save"],
    )
    print("Testing finished!")

    # Create an empty pd dataframe with three columns: data, label and manner
    df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_train[audio_features] = [t[0] for t in train_loader.dataset]
    df_train["label"] = [t[1] for t in train_loader.dataset]
    df_train["manner"] = [t[2] for t in train_loader.dataset]

    # Create an empty pd dataframe with three columns: data, label and manner
    df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_test[audio_features] = [t[0] for t in test_loader.dataset]
    df_test["label"] = [t[1] for t in test_loader.dataset]
    df_test["manner"] = [t[2] for t in test_loader.dataset]

    print("Starting to calculate distances...")
    plot_logopeda_alb_neuro(
        model,
        df_train,
        df_test,
        hyperparams["wandb_flag"],
        name="test",
        supervised=hyperparams["supervised"],
        samples=5000,
        path_to_plot=hyperparams["path_to_save"],
    )

    if hyperparams["wandb_flag"]:
        wandb.finish()

    sys.stdout = old_stdout
    log_file.close()


if __name__ == "__main__":
    args = {}

    hyperparams = {
        # ================ Spectrogram parameters ===================
        "spectrogram": True,  # If true, use spectrogram. If false, use plp (In this study we only use spectrograms)
        "frame_size_ms": 0.400,  # Size of each spectrogram frame
        "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
        "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
        # ================ GMVAE parameters ===================
        "epochs": 500,  # Number of epochs to train the model (at maximum, we have early stopping)
        "batch_size": 128,  # Batch size
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
        # ================ Classifier parameters ===================
        "cnn_classifier": False,  # Here no classifier is used
        "supervised": False,  # Here no classifier is used
        # ================ Training parameters ===================
        "train": False,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "train_albayzin": True,  # If true, train with albayzin data. If false, only train with neurovoz data.
        "new_data_partition": False,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
        # ================ UNUSED PARAMETERS (we should fix this) ===================
        # These parameters are not used at all and they are from all versions of the code, we should fix this.
        "material": "MANNER",  # not used here
        "n_plps": 0,  # Not used here
        "n_mfccs": 0,  # Not used here
        "wandb_flag": False,  # Not used here
        "semisupervised": False,  # Not used here
    }

    main(args, hyperparams)
