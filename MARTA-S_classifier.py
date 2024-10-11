"""
MARTA VAE-based Parkinson's Disease Classification from Spectrograms

This script implements a pipeline for classifying Parkinsonian and healthy control spectrograms using a 
pre-trained Gaussian Mixture Variational Autoencoder (GMVAE) and a subsequent classifier. The GMVAE model 
is first trained in a supervised manner (using MARTA_supervised.py) and then frozen. A classifier is trained on the latent space 
outputs of the MARTA to distinguish between Parkinsonian and healthy spectrograms. Finally, postprocessing 
is conducted to calculate joint probability predictions, providing a unified prediction for each patient 
based on all available 400ms spectrogram segments.

The main steps include:
1. Initializing the environment and setting up GPU for computations.
2. Loading pre-processed data or creating new data partitions.
3. Defining and loading the MARTA supervised model and classifier architecture.
4. Training the classifier on the latent space representations provided by the MARTA.
5. Evaluating the model on test data and calculating joint probability predictions for patients.

Requirements:
- This script assumes the existence of a pre-trained MARTA model.
- Data should be pre-processed and organized in specific formats for effective loading and training.

Outputs:
- A trained classifier model capable of differentiating between Parkinsonian and healthy control spectrograms.
- Log files and model checkpoints saved in specified directories.
- (Optional) Weights and Biases (wandb) integration for experiment tracking.

Usage:
- The script is configured via command-line arguments and a set of hyperparameters.
- GPU selection and fold number for experiments are among the configurable parameters.

Author: Guerrero-López, Alejandro
Date: 25/01/2024

Note: 
- The script includes several hardcoded paths and parameters, which might need to be adjusted 
  based on the specific setup and data organization.
"""

import os
import sys
import torch
import argparse
import pandas as pd
import wandb
from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester, check_latent_space
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
from utils.utils import make_balanced_sampler, stratify_per_dataset


def main(args, hyperparams):
    gpu = args.gpu
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device being used:", device)

    # Create the path if does not exist
    path_to_save = f"local_results/spectrograms/classifier_{hyperparams['crosslingual']}_{hyperparams['latent_dim']}latent_dim_{hyperparams['domain_adversarial']}domain_adversarial_fold_{hyperparams['fold']}"
    os.makedirs(path_to_save, exist_ok=True)

    # Redirect standard output to a log file
    log_file = open(f"{path_to_save}/log.txt", "w")
    sys.stdout = log_file

    # First check in local_results/ if there eist any .pt file with the dataloaders
    # If not, create them and save them in local_results/

    if not hyperparams["new_data_partition"]:
        print("Reading train, val and test loaders from local_results/...")
        loader_path = f"local_results/folds/{{}}_loader_supervised_True_frame_size_0.4spec_winsize_{hyperparams['spectrogram_win_size']}hopsize_0.5fold{hyperparams['fold']}.pt"
        data_path = f"local_results/folds/{{}}_data_supervised_True_frame_size_0.4spec_winsize_{hyperparams['spectrogram_win_size']}hopsize_0.5fold{hyperparams['fold']}.pt"
        train_loader, val_loader, test_loader = (
            torch.load(loader_path.format(ds)) for ds in ["train", "val", "test"]
        )
        train_data, val_data, test_data = (
            torch.load(data_path.format(ds)) for ds in ["train", "val", "test"]
        )

        # Get all gita data
        gita_data_test = [data for data in test_loader.dataset if data[3] == "gita"]
        gita_data_val = [data for data in val_loader.dataset if data[3] == "gita"]
        gita_data_train = [data for data in train_loader.dataset if data[3] == "gita"]

        # Get all neurovoz data
        neurovoz_data_test = [
            data for data in test_loader.dataset if data[3] == "neurovoz"
        ]
        neurovoz_data_val = [
            data for data in val_loader.dataset if data[3] == "neurovoz"
        ]
        neurovoz_data_train = [
            data for data in train_loader.dataset if data[3] == "neurovoz"
        ]

        # Get all albayzin data
        albayzin_data_test = [
            data for data in test_loader.dataset if data[3] == "albayzin"
        ]
        albayzin_data_val = [
            data for data in val_loader.dataset if data[3] == "albayzin"
        ]
        albayzin_data_train = [
            data for data in train_loader.dataset if data[3] == "albayzin"
        ]

        if hyperparams["crosslingual"] == "testing_gita":
            new_train = stratify_per_dataset(neurovoz_data_train + albayzin_data_train)
            new_val = neurovoz_data_val + albayzin_data_val
            new_test = gita_data_test
        elif hyperparams["crosslingual"] == "testing_neurovoz":
            new_train = stratify_per_dataset(gita_data_train + albayzin_data_train)
            new_val = gita_data_val + albayzin_data_val
            new_test = neurovoz_data_test
        else:
            new_train = stratify_per_dataset(train_loader.dataset)
            new_val = val_loader.dataset
            new_test = test_loader.dataset

        train_sampler = make_balanced_sampler(new_train, validation=False)
        val_sampler = make_balanced_sampler(new_val, validation=True)

        # Create new dataloaders
        train_loader = torch.utils.data.DataLoader(
            new_train, batch_size=hyperparams["batch_size"], sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            new_val, batch_size=hyperparams["batch_size"], sampler=val_sampler
        )
        test_loader = torch.utils.data.DataLoader(
            new_test, batch_size=hyperparams["batch_size"], shuffle=False
        )

    print("Defining models...")
    # Create the model
    model = MARTA(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_gaussians=hyperparams["n_gaussians"],
        n_manner=16,
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        classifier=hyperparams["classifier_type"],
        weights=hyperparams["weights"],
        device=device,
        domain_adversarial_bool=hyperparams["domain_adversarial"],
        datasets=3,
    )

    if hyperparams["train"]:
        # Load the best unsupervised model to supervise it
        model_path = f"local_results/spectrograms/marta_{hyperparams['latent_dim']}_experiment_{hyperparams['crosslingual']}_supervised__domain_adversarial_{hyperparams['domain_adversarial']}_fold_{hyperparams['fold']}/GMVAE_cnn_best_model_2d.pt"
        model.load_state_dict(torch.load(model_path)["model_state_dict"])

        # Freeze all the network
        for param in model.parameters():
            param.requires_grad = False
        # Add a classifier to the model
        model.class_dims = [64, 32, 16]
        model.classifier()
        # Unfreeze the classifier
        for param in model.hmc.parameters():
            # Unfreezing the manner class embeddings
            param.requires_grad = True
        for param in model.clf_cnn.parameters():
            # Unfreezing the cnn classifier
            param.requires_grad = True
        for param in model.clf_mlp.parameters():
            # Unfreezing the mlp classifier
            param.requires_grad = True
        model.to(device)

        print("Training GMVAE...")
        # Train the model
        MARTA_trainer(
            model=model,
            trainloader=train_loader,
            validloader=val_loader,
            epochs=hyperparams["epochs"],
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
            path_to_save=path_to_save,
            supervised=hyperparams["supervised"],
            classifier=hyperparams["classifier"],
        )

        print("Training finished!")
    else:
        print("Loading model...")

    # Restoring best model
    model.load_state_dict(
        torch.load(f"{path_to_save}/GMVAE_cnn_best_model_2d.pt")["model_state_dict"]
    )

    # Check latent space
    # print("Checking latent space...")
    # check_latent_space(
    #     model,
    #     train_loader.dataset,
    #     test_loader.dataset,
    #     path_to_save,
    #     latent_dim=hyperparams["latent_dim"],
    # )

    print("Testing GMVAE...")

    # Read the best threshold
    with open(f"{path_to_save}/best_threshold.txt", "r") as f:
        threshold = float(f.read())

    # Test the model
    print("================ TESTING DATA ================")
    MARTA_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        supervised=True,  # Not implemented yet
        wandb_flag=hyperparams["wandb_flag"],
        path_to_plot=path_to_save,
        best_threshold=threshold,
    )
    print("Testing finished!")

    # Close log file and reset stdout
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument(
        "--fold", type=int, default=0, help="Fold number for the experiment"
    )
    parser.add_argument(
        "--gpu", type=int, default=2, help="GPU number for the experiment"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Latent dimension of the model"
    )
    parser.add_argument(
        "--domain_adversarial", type=int, default=1, help="Use domain adversarial"
    )
    parser.add_argument(
        "--cross_lingual",
        type=str,
        default="testing_gita",
        choices=["multilingual", "testing_gita", "testing_neurovoz", "testing_italian"],
        help="Select one choice of crosslingual scenario",
    )
    args = parser.parse_args()

    hyperparams = {
        # ================ Spectrogram parameters ===================
        "spectrogram": True,  # If true, use spectrogram. If false, use plp (In this study we only use spectrograms)
        "frame_size_ms": 0.400,  # Size of each spectrogram frame
        "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
        "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
        # ================ GMVAE parameters ===================
        "epochs": 200,  # Number of epochs to train the model (at maximum, we have early stopping)
        "batch_size": 128,  # Batch size
        "lr": 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
        "latent_dim": args.latent_dim,  # Latent dimension of the z vector (remember it is also the input to the classifier)
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
        "domain_adversarial": args.domain_adversarial,  # If true, use domain adversarial model
        "crosslingual": args.cross_lingual,  # Crosslingual scenario
        # ================ Classifier parameters ===================
        "classifier_type": "cnn",  # classifier architecture (cnn or mlp)-.Their dimensions are hard-coded in pt_models.py (we should fix this)
        "classifier": True,  # If true, train the classifier
        "supervised": True,  # It must be true
        # ================ Training parameters ===================
        "train": True,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "train_albayzin": False,  # If true, train with albayzin data. If false, only train with neurovoz data.
        "new_data_partition": False,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
        "fold": args.fold,  # Which fold to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        "gpu": args.gpu,  # Which gpu to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        # ================ UNUSED PARAMETERS (we should fix this) ===================
        # These parameters are not used at all and they are from all versions of the code, we should fix this.
        "material": "MANNER",  # not used here
        "n_plps": 0,  # Not used here
        "n_mfccs": 0,  # Not used here
        "wandb_flag": False,  # Not used here
        "semisupervised": False,  # Not used here
    }

    main(args, hyperparams)
