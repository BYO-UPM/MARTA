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

from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import sys
import os
import argparse
from utils.utils import make_balanced_sampler, augment_data, stratify_dataset


def main(args, hyperparams):
    gpu = args.gpu
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    hyperparams["path_to_save"] = (
        "local_results/spectrograms/manner_gmvae_alb_neurovoz_"
        + str(hyperparams["latent_dim"])
        + "final_model_classifier"
        + "_LATENTSPACE+manner_MLP_fold"
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

        # Remove all albayzin samples from train_loader
        if not hyperparams["train_albayzin"]:
            new_train = [data for data in train_loader.dataset if data[3] == "neurovoz"]
            new_val = [data for data in val_loader.dataset if data[3] == "neurovoz"]
        else:
            new_train = train_loader.dataset
            new_val = val_loader.dataset

        # Augment the train dataset
        extended_dataset = augment_data(new_train)

        # Stratify train dataset
        balanced_dataset = stratify_dataset(extended_dataset)

        train_sampler = make_balanced_sampler(balanced_dataset)
        val_sampler = make_balanced_sampler(new_val, validation=True)

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

    else:
        print("Reading data...")
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
    )

    if hyperparams["train"]:
        # Load the best unsupervised model to supervise it
        name = (
            "local_results/spectrograms/manner_gmvae_alb_neurovoz_32supervised90-10-fold"
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
    MARTA_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        supervised=True,  # Not implemented yet
        wandb_flag=hyperparams["wandb_flag"],
        path_to_plot=hyperparams["path_to_save"],
        best_threshold=threshold,
    )
    print("Testing finished!")

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
        "classifier_type": "mlp",  # classifier architecture (cnn or mlp)-.Their dimensions are hard-coded in pt_models.py (we should fix this)
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
