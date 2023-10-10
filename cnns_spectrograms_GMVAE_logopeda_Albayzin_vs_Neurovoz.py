from models.pt_models import GMVAE
from training.pt_training import GMVAE_trainer, GMVAE_tester
from utils.utils import (
    plot_logopeda,
    calculate_distances_manner,
)
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import numpy as np
import pandas as pd

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Print the cuda device to use
print("Using cuda device: ", torch.cuda.current_device())
fold = 0  # Not used, just for compatibility with the other scripts #### this should be improved xD


def main(args):
    hyperparams = {
        "frame_size_ms": 0.400,  # 400ms
        "spectrogram_win_size": 0.023,  # 23ms as it is recommended in the librosa library for speech recognition
        "material": "MANNER",
        "hop_size_percent": 0.5,
        "n_plps": 0,
        "n_mfccs": 0,
        "spectrogram": True,
        "wandb_flag": False,
        "epochs": 1,
        "batch_size": 64,
        "lr": 1e-3,
        "latent_dim": 32,
        "hidden_dims_enc": [64, 128, 64, 32],
        "hidden_dims_dec": [32, 64, 128, 64],
        "supervised": False,
        "n_gaussians": 9,
        "semisupervised": False,
    }

    print("Reading data...")
    # Read the data
    dataset = Dataset_AudioFeatures(
        "labeled/NeuroVoz",
        hyperparams,
    )

    if hyperparams["wandb_flag"]:
        gname = "SPECTROGRAMS_GMVAE_" + hyperparams["material"]
        if hyperparams["n_gaussians"] > 0:
            if hyperparams["n_gaussians"] == 2:
                gname += "_supervised_PD"
            elif hyperparams["n_gaussians"] == 5:
                gname += "_supervised_vowels"
            elif hyperparams["n_gaussians"] == 10:
                gname += "_supervised_2labels"
            elif hyperparams["n_gaussians"] > 10:
                gname += "_logopeda_9gaussians_9mannerclasses"
        else:
            gname += "_UNsupervised"
        wandb.finish()
        wandb.init(
            project="parkinson",
            config=hyperparams,
            group=gname,
            name="fold_" + str(fold),
        )
    print("Training a VAE for fold: ", fold)

    (
        train_loader,
        val_loader,
        test_loader,
        train_data,
        val_data,
        test_data,
    ) = dataset.get_dataloaders(fold)

    print("Defining models...")
    # Create the model
    model = GMVAE(
        x_dim=train_loader.dataset[0][0].shape[0],
        n_gaussians=hyperparams["n_gaussians"],
        z_dim=hyperparams["latent_dim"],
        hidden_dims=hyperparams["hidden_dims_enc"],
        ss=hyperparams["semisupervised"],
        supervised=hyperparams["supervised"],
        weights=[
            1,  # w1 is rec loss,
            1,  # w2 is gaussian kl loss,
            1,  # w3 is categorical kl loss,
            1,  # w4 is supervised loss, # not implemented for n_gaussians != 2,5
            100,  # w5 is metric loss
        ],
        cnn=hyperparams["spectrogram"],
    )

    model = torch.compile(model)

    print("Training GMVAE...")
    # Train the model
    GMVAE_trainer(
        model=model,
        trainloader=train_loader,
        validloader=val_loader,
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
        supervised=hyperparams["supervised"],
        wandb_flag=hyperparams["wandb_flag"],
    )

    print("Training finished!")

    # Restoring best model
    if hyperparams["supervised"]:
        name = "local_results/spectrograms/manner_gmvae/GMVAE_cnn_best_model.pt"
    else:
        name = "local_results/spectrograms/manner_gmvae/GMVAE_cnn_best_model_2d.pt"
    tmp = torch.load(name)
    model.load_state_dict(tmp["model_state_dict"])

    if hyperparams["n_plps"] > 0:
        audio_features = "plps"
    elif hyperparams["n_mfccs"] > 0:
        audio_features = "mfccs"
    elif hyperparams["spectrogram"]:
        audio_features = "spectrogram"
    print("Testing GMVAE...")

    # Test the model
    GMVAE_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        audio_features=audio_features,
        supervised=False,  # Not implemented yet
        wandb_flag=hyperparams["wandb_flag"],
    )

    # Create an empty pd dataframe with three columns: data, label and manner
    df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_train[audio_features] = [t[0] for t in train_loader.dataset]
    df_train["label"] = [t[1] for t in train_loader.dataset]
    df_train["manner"] = [t[2] for t in train_loader.dataset]

    # Select randomly 1000 samples of dftrain
    df_train = df_train.sample(n=1000)

    # Create an empty pd dataframe with three columns: data, label and manner
    df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_test[audio_features] = [t[0] for t in test_loader.dataset]
    df_test["label"] = [t[1] for t in test_loader.dataset]
    df_test["manner"] = [t[2] for t in test_loader.dataset]

    if hyperparams["material"] == "MANNER":
        # Plot the latent space in test
        plot_logopeda(
            model,
            df_train,
            df_test,
            hyperparams["wandb_flag"],
            name="test",
            supervised=hyperparams["supervised"],
            samples=2000,
        )

    if hyperparams["wandb_flag"]:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser(description="VAE training")
    # parser.add_argument(
    #     "--data_path",
    #     type=str,
    #     default="/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2",
    #     help="Path to the data",
    # )
    # parser.add_argument(
    #     "--material",
    #     type=str,
    #     default="VOWELS",
    #     choices=["PATAKA", "VOWELS"],
    #     help="Acoustic material to use",
    # )
    # parser.add_argument(
    #     "--frame_size_ms",
    #     type=int,
    #     default=40,
    #     help="Frame size in milliseconds",
    # )
    # parser.add_argument(
    #     "--hop_size_percent",
    #     type=float,
    #     default=0.5,
    #     help="Hop size in percent",
    # )
    # parser.add_argument(
    #     "--n_plps",
    #     type=int,
    #     default=0,
    #     help="Number of RASTA-PLP coefficients. If 0, use MFCCs",
    # )
    # parser.add_argument(
    #     "--n_mfccs",
    #     type=int,
    #     default=0,
    #     help="Number of MFCC coefficients. If 0, use RASTA-PLPs",
    # )
    # parser.add_argument(
    #     "--wandb_flag",
    #     action="store_true",
    #     help="Flag to use wandb",
    # )
    # parser.add_argument(
    #     "--epochs",
    #     type=int,
    #     default=300,
    #     help="Number of epochs",
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     default=32,
    #     help="Batch size",
    # )
    # parser.add_argument(
    #     "--lr",
    #     type=float,
    #     default=0.001,
    #     help="Learning rate",
    # )
    # parser.add_argument(
    #     "--latent_dim",
    #     type=int,
    #     default=2,
    #     help="Latent dimension",
    # )
    # parser.add_argument(
    #     "--hidden_dims_enc",
    #     type=list,
    #     default=[20, 10],
    #     help="Hidden dimensions of the encoder",
    # )
    # parser.add_argument(
    #     "--hidden_dims_dec",
    #     type=list,
    #     default=[10],
    #     help="Hidden dimensions of the decoder",
    # )
    # parser.add_argument(
    #     "--supervised",
    #     action="store_true",
    #     help="Flag to use supervised training",
    # )
    # parser.add_argument(
    #     "--n_gaussians",
    #     type=int,
    #     choices=[2, 5],
    #     default=2,
    #     help="Number of classes to supervise. Only used if supervised=True. 2 for PD, 5 for VOWELS.",
    # )
    # args = parser.parse_args()

    args = {}

    main(args)
