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
from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester, check_latent_space
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
from utils.utils import make_balanced_sampler, stratify_per_dataset
import textgrids as tg
import bisect
import librosa


def main(args, hyperparams):
    gpu = args.gpu
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device being used:", device)

    # Create the path if does not exist
    if hyperparams["crosslingual"] == "multilingual":
        path_to_load = f"local_results/spectrograms/multilingual/classifier_{hyperparams['crosslingual']}_{hyperparams['latent_dim']}latent_dim_{hyperparams['domain_adversarial']}domain_adversarial_fold_{hyperparams['fold']}"
        path_to_save = f"local_results/spectrograms/whisper_multilingual/classifier_{hyperparams['crosslingual']}_{hyperparams['latent_dim']}latent_dim_{hyperparams['domain_adversarial']}domain_adversarial_fold_{hyperparams['fold']}"
    elif (
        hyperparams["crosslingual"] == "testing_gita"
        or hyperparams["crosslingual"] == "testing_neurovoz"
    ):
        path_to_load = f"local_results/spectrograms/crosslingual/classifier_{hyperparams['crosslingual']}_{hyperparams['latent_dim']}latent_dim_{hyperparams['domain_adversarial']}domain_adversarial_fold_{hyperparams['fold']}"
        path_to_save = f"local_results/spectrograms/whisper_crosslingual/classifier_{hyperparams['crosslingual']}_{hyperparams['latent_dim']}latent_dim_{hyperparams['domain_adversarial']}domain_adversarial_fold_{hyperparams['fold']}"
    os.makedirs(path_to_save, exist_ok=True)

    # Redirect standard output to a log file
    log_file = open(f"{path_to_save}/log.txt", "w")
    sys.stdout = log_file

    # First check in local_results/ if there eist any .pt file with the dataloaders
    # If not, create them and save them in local_results/

    if not hyperparams["new_data_partition"]:
        print("Reading train, val and test loaders from local_results/...")
        data_path_new = f"local_results/folds/{{}}_data_supervised_True_frame_size_0.4spec_winsize_{hyperparams['spectrogram_win_size']}hopsize_0.5fold{hyperparams['fold']}.pt"
        data_path_old = f"local_results/folds/manually_transcribed/{{}}_data_supervised_True_frame_size_0.4spec_winsize_{hyperparams['spectrogram_win_size']}hopsize_0.5fold{hyperparams['fold']}.pt"

        # Load both train, val and test data from old a new path
        train_data_new, val_data_new, test_data_new = (
            torch.load(data_path_new.format("train")),
            torch.load(data_path_new.format("val")),
            torch.load(data_path_new.format("test")),
        )
        test_data_old = torch.load(data_path_old.format("test"))

        # Rearrange new test data. To do so: create a unique dataset new with is all_data_new. Then, create a new test_data_new which have to match same patients as test_data_old but with the new data
        total_data_new = pd.concat([train_data_new, val_data_new, test_data_new])
        test_data = total_data_new[
            total_data_new["id_patient"].isin(test_data_old["id_patient"].unique())
        ]
        # print test_data shape
        print("Test data shape: ", test_data.shape)
        print(test_data.head())
        # now create the testloader from test_data

        import numpy as np

        x_test = np.stack(test_data["spectrogram"].values)
        x_test = np.expand_dims(x_test, axis=1)
        y_test = test_data["label"].values
        p_test = np.array([np.array(x) for x in test_data["manner_class"]])
        d_test = np.array([np.array(x) for x in test_data["dataset"]])
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
            batch_size=hyperparams["batch_size"],
            shuffle=False,
        )

        # if experiment is testing_gita
        if hyperparams["crosslingual"] == "testing_gita":
            # Test data should be only gita
            test_data = test_data[test_data["dataset"] == "gita"]
            test_loader.dataset.data = test_loader.dataset.data[
                test_loader.dataset.data["dataset"] == "gita"
            ]

        # if experiment is testing_neurovoz
        if hyperparams["crosslingual"] == "testing_neurovoz":
            # Test data should be only neurovoz
            test_data = test_data[test_data["dataset"] == "neurovoz"]
            test_loader.dataset.data = test_loader.dataset.data[
                test_loader.dataset.data["dataset"] == "neurovoz"
            ]

        else:
            # Test data should be all data as is the multilingual scenario
            test_data = test_data
    else:
        print("Creating new data partition...")
        dataset = Dataset_AudioFeatures(
            hyperparams,
        )
        (
            train_loader,
            val_loader,
            test_loader,
            train_data,  # train_data, not used
            _,  # val_data, not used
            test_data,
        ) = dataset.get_dataloaders(
            experiment=hyperparams["experiment"], supervised=hyperparams["supervised"]
        )
    print("Defining models...")
    # Create the model
    model = MARTA(
        x_dim=test_loader.dataset[0][0].shape,
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

    # Load the model
    print("Loading model...")

    # Restoring best model
    model.load_state_dict(
        torch.load(f"{path_to_load}/GMVAE_cnn_best_model_2d.pt")["model_state_dict"]
    )

    print("Testing GMVAE...")

    # Read the best threshold
    with open(f"{path_to_load}/best_threshold.txt", "r") as f:
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
        default="multilingual",
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
        "experiment": "fourth",
        # ================ Classifier parameters ===================
        "classifier_type": "cnn",  # classifier architecture (cnn or mlp)-.Their dimensions are hard-coded in pt_models.py (we should fix this)
        "classifier": True,  # If true, train the classifier
        "supervised": True,  # It must be true
        "whisper": True,  # If true, use whisper data
        # ================ Training parameters ===================
        "train": False,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "new_data_partition": False,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
        "fold": args.fold,  # Which fold to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        "gpu": args.gpu,  # Which gpu to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        # ================ UNUSED PARAMETERS (we should fix this) ===================
        # These parameters are not used at all and they are from all versions of the code, we should fix this.
        "wandb_flag": False,  # Not used here
    }

    main(args, hyperparams)
