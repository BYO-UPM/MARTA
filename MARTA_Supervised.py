"""
Supervised MARTA for Speech Feature Analysis in Parkinson's Disease

This script implements a supervised Gaussian Mixture Variational Autoencoder (GMVAE) to analyze 
speech features for Parkinson's Disease research. Unlike traditional classifiers, this approach 
focuses on maximizing the distance between clusters in the latent space for different manner classes. 
The model is trained on healthy patient data while being supervised with manner classes split between 
Parkinsonian (8 classes) and healthy (8 classes). The aim is to observe how the model distinguishes 
between these classes in a latent space representation.

Key Features:
1. Data Processing: Utilizes 'Dataset_AudioFeatures' for loading and preprocessing spectrogram data.
2. MARTA Model: Constructs and trains a MARTA model with 16 manner classes (split between healthy 
   and Parkinsonian classes).
3. Supervised Learning: The model is trained in a supervised manner without being a traditional classifier.
4. Latent Space Analysis: Focuses on examining the distances between clusters in the latent space.
5. Visualization: Utilizes 'plot_logopeda_alb_neuro' for visualizing the results and understanding 
   the model's performance.

Usage:
- The script is designed for execution in environments with CUDA-compatible GPUs.
- Hyperparameters for the model and training process can be adjusted according to the requirements.

Output:
- A trained MARTA model capable of differentiating speech features based on manner classes.
- Log files and performance metrics for model training and testing.
- Visualizations highlighting the separation in the latent space.

Requirements:
- Libraries like torch, pandas, wandb, etc., for model building, data handling, and logging.
- Properly formatted and pre-processed speech data.

Author: [Your Name]
Date: [Creation/Modification Date]

Note:
- The script assumes a specific format and structure for the input data.
- Adjustments may be necessary for hyperparameters and model configuration based on the 
  characteristics of the data and available computational resources.
"""

from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester
from utils.utils import (
    plot_logopeda_alb_neuro,
)
from utils.definitions import *
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import pandas as pd
import sys
import os
import argparse


def main(args, hyperparams):
    gpu = "cuda:" + str(hyperparams["gpu"])
    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    hyperparams["path_to_save"] = (
        "local_results/spectrograms/manner_gmvae_alb_neurovoz_"
        + str(hyperparams["latent_dim"])
        + "supervised"
        + "90-10-fold"
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

    if hyperparams["train"] and hyperparams["new_data_partition"]:
        # Create path if does not exist
        if not os.path.exists(PROCESSED_DATA_LOCAL + "folds"):
            os.makedirs(
                PROCESSED_DATA_LOCAL + "folds"
            )

        print("Reading data...")
        # Read the data
        dataset = Dataset_AudioFeatures(
            NEUROVOZ_LABELS_LOCAL,
            hyperparams,
        )
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
    else:
        print("Reading train, val and test loaders from local_results/...")
        train_loader = torch.load(
            "local_results/folds/train_loader_supervised_True_frame_size_0.4spec_winsize_"
            + str(hyperparams["spectrogram_win_size"])
            + "hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        val_loader = torch.load(
            "local_results/folds/val_loader_supervised_True_frame_size_0.4spec_winsize_"
            + str(hyperparams["spectrogram_win_size"])
            + "hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        test_loader = torch.load(
            "local_results/folds/test_loader_supervised_True_frame_size_0.4spec_winsize_"
            + str(hyperparams["spectrogram_win_size"])
            + "hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        test_data = torch.load(
            "local_results/folds/test_data_supervised_True_frame_size_0.4spec_winsize_"
            + str(hyperparams["spectrogram_win_size"])
            + "hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )

    print("Defining models...")
    # Create the model
    model = MARTA(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_manner=16,
        n_gaussians=hyperparams["n_gaussians"],
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        weights=hyperparams["weights"],
        device=device,
        reducer="sum",
    )

    if hyperparams["train"]:
        print("Training GMVAE...")
        # Train the model
        MARTA_trainer(
            model=model,
            trainloader=train_loader,
            validloader=val_loader,
            epochs=hyperparams["epochs"],
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
            path_to_save=hyperparams["path_to_save"],
            supervised=hyperparams["supervised"],
            classifier=hyperparams["classifier"],
            method=hyperparams["method"],
        )

        print("Training finished!")
    else:
        print("Loading model...")

    # # Restoring best model
    # name = hyperparams["path_to_save"] + "/GMVAE_cnn_best_model_2d.pt"
    # tmp = torch.load(name)
    # model.load_state_dict(tmp["model_state_dict"])

    # audio_features = "spectrogram"
    # print("Testing GMVAE...")

    # # Test the model
    # MARTA_tester(
    #     model=model,
    #     testloader=test_loader,
    #     test_data=test_data,
    #     supervised=False,  # Not implemented yet
    #     wandb_flag=hyperparams["wandb_flag"],
    #     path_to_plot=hyperparams["path_to_save"],
    # )
    # print("Testing finished!")

    # # Create an empty pd dataframe with three columns: data, label and manner
    # df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    # df_train[audio_features] = [t[0] for t in train_loader.dataset]
    # df_train["label"] = [t[1] for t in train_loader.dataset]
    # df_train["manner"] = [t[2] for t in train_loader.dataset]

    # # Substract 8 to manner if their corresponidng label is 1
    # df_train["manner"] = df_train.apply(
    #     lambda x: x["manner"] - 8 if x["label"] == 1 else x["manner"], axis=1
    # )

    # # Create an empty pd dataframe with three columns: data, label and manner
    # df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    # df_test[audio_features] = [t[0] for t in test_loader.dataset]
    # df_test["label"] = [t[1] for t in test_loader.dataset]
    # df_test["manner"] = [t[2] for t in test_loader.dataset]

    # # Substract 8 to manner if their corresponidng label is 1
    # df_test["manner"] = df_test.apply(
    #     lambda x: x["manner"] - 8 if x["label"] == 1 else x["manner"], axis=1
    # )

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

    # if hyperparams["wandb_flag"]:
    #     wandb.finish()

    sys.stdout = old_stdout
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument(
        "--fold", type=int, default=1, help="Fold number for the experiment"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU number to use in the experiment"
    )

    parser.add_argument(
        "--method", type=str, default='sumloss', help="Gradients manipulation method"
    )

    args = parser.parse_args()

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
        "latent_dim": 32,  # Latent dimension of the z vector (remember it is also the input to the classifier)
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
        "classifier_type": "cnn",  # classifier architecture (cnn or mlp)-.Their dimensions are hard-coded in pt_models.py (we should fix this)
        "classifier": False,  # It must be False in this script.
        "supervised": True,  # It must be true
        # ================ Training parameters ===================
        "method": args.method, 
        "train": True,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "train_albayzin": True,  # If true, train with albayzin data. If false, only train with neurovoz data.
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
