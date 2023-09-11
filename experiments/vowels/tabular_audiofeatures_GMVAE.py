from models.pt_models import GMVAE
from training.pt_training import GMVAE_trainer, GMVAE_tester
from utils.utils import plot_latent_space, plot_latent_space_vowels, calculate_distances
from data_loaders.pt_data_loader_audiofeatures import Dataset_AudioFeatures
import torch
import wandb
import numpy as np
import pandas as pd

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Print the cuda device to use
print("Using cuda device: ", torch.cuda.current_device())


def main(args):
    hyperparams = {
        "frame_size_ms": 40,
        "material": "VOWELS",
        "hop_size_percent": 0.5,
        "n_plps": 13,
        "n_mfccs": 0,
        "wandb_flag": True,
        "epochs": 300,
        "batch_size": 64,
        "lr": 1e-3,
        "latent_dim": 2,
        "hidden_dims_enc": [64, 128, 64, 32],
        "hidden_dims_dec": [32, 64, 128, 64],
        "supervised": False,
        "n_gaussians": 10,
        "semisupervised": False,
    }

    print("Reading data...")
    # Read the data
    dataset = Dataset_AudioFeatures(
        "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2",
        hyperparams,
    )

    for fold in dataset.data["fold"].unique():
        if hyperparams["wandb_flag"]:
            gname = "rasta_PLPs_GMVAE_" + hyperparams["material"]
            if hyperparams["n_gaussians"] > 0:
                if hyperparams["n_gaussians"] == 2:
                    gname += "_naive_PD"
                elif hyperparams["n_gaussians"] == 5:
                    gname += "_supervised_vowels"
                elif hyperparams["n_gaussians"] == 10:
                    gname += "_unsupervised_10Gaussians_x_and_y_sameshape_in_inference"
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
                2,  # w2 is gaussian kl loss,
                5,  # w3 is categorical kl loss,
                1,  # w4 is supervised loss, # not implemented for n_gaussians != 2,5
                10,  # w5 is metric loss
            ],
        )

        model = torch.compile(model)

        print("Training VAE...")
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
            name = "local_results/plps/vae_supervised/VAE_best_model.pt"
        else:
            name = "local_results/plps/vae_unsupervised/VAE_best_model.pt"
        tmp = torch.load(name)
        model.load_state_dict(tmp["model_state_dict"])

        if hyperparams["n_plps"] > 0:
            audio_features = "plps"
        elif hyperparams["n_mfccs"] > 0:
            audio_features = "mfccs"
        print("Testing VAE...")

        # Test the model  #TODO: supervised version is not yet implemented for testing
        GMVAE_tester(
            model=model,
            testloader=test_loader,
            test_data=test_data,
            audio_features=audio_features,
            supervised=False,  # Not implemented yet
            wandb_flag=hyperparams["wandb_flag"],
        )

        # Create an empty pd dataframe with two columns: data and label
        df = pd.DataFrame(columns=["plps", "label", "vowel"])
        df["plps"] = [t[0] for t in test_loader.dataset]
        df["label"] = [t[1] for t in test_loader.dataset]
        df["vowel"] = [t[2] for t in test_loader.dataset]

        if hyperparams["material"] == "PATAKA":
            # Plot the latent space in test
            plot_latent_space(
                model,
                df,
                fold,
                hyperparams["wandb_flag"],
                name="test",
                supervised=hyperparams["supervised"],
            )
        elif hyperparams["material"] == "VOWELS":
            plot_latent_space_vowels(
                model,
                df,
                fold,
                hyperparams["wandb_flag"],
                name="test",
                supervised=hyperparams["supervised"],
                gmvae=True,
            )
            calculate_distances(
                model, df, fold, hyperparams["wandb_flag"], name="test", gmvae=True
            )

        # df = pd.DataFrame(columns=["plps", "label", "vowel"])
        # df["plps"] = [t[0] for t in train_loader.dataset]
        # df["label"] = [t[1] for t in train_loader.dataset]
        # df["vowel"] = [t[2] for t in train_loader.dataset]

        # # Plot the latent space in train
        # if hyperparams["material"] == "PATAKA":
        #     # Plot the latent space in test
        #     plot_latent_space(
        #         model,
        #         df,
        #         fold,
        #         hyperparams["wandb_flag"],
        #         supervised=hyperparams["supervised"],
        #         name="train",
        #     )
        # elif hyperparams["material"] == "VOWELS":
        #     plot_latent_space_vowels(
        #         model,
        #         df,
        #         fold,
        #         hyperparams["wandb_flag"],
        #         supervised=hyperparams["supervised"],
        #         name="train",
        #     )

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