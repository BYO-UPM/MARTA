from DBLR.DBLR.DeepBLR import DeepBLR
from training.pt_training import VAE_trainer, VAE_tester
from utils.utils import plot_latent_space, plot_latent_space_vowels, calculate_distances
from data_loaders.pt_data_loader_audiofeatures import Dataset_AudioFeatures
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm

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
        "wandb_flag": False,
        "epochs": 300,
        "batch_size": 64,
        "lr": 1e-3,
        "latent_dim": 32,
        "hidden_dims_enc": [64, 128, 64],
        "hidden_dims_dec": [64, 128, 64],
        "supervised": True,
        "n_classes": 5,
    }

    print("Reading data...")
    # Read the data
    dataset = Dataset_AudioFeatures(
        "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2",
        hyperparams,
    )

    for fold in dataset.data["fold"].unique():
        if hyperparams["wandb_flag"]:
            gname = "rasta_PLPs_vae_" + hyperparams["material"]
            if hyperparams["supervised"]:
                gname += "_supervised_by_VOWELS_deeper"
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
        dblr = DeepBLR(
            D=42,
            hdim_mean=30,
            hdim_var=30,
            K=2,
            input_type="real",
            percentage=0.1,
        )

        print("Training DBLR...")

        x = torch.Tensor(np.vstack(train_data["plps"]))
        y = torch.Tensor(train_data["label"].values).unsqueeze(1)

        model = torch.compile(dblr)

        # Make a for loop using tqdm
        for e in tqdm(range(20)):
            # Train the model
            dblr.train_epoch(x, y, batch_size=64, mc=1, verbose=True)

            # Update tqdm
            tqdm.write("Epoch: " + str(e) + " Loss: " + str(-dblr.mean_ELBO_loss_epoch))

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
        # Test the model by frame
        VAE_tester(
            model,
            test_loader,
            test_data,
            audio_features,
            supervised=hyperparams["supervised"],
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
            )
            calculate_distances(model, df, fold, hyperparams["wandb_flag"], name="test")

        df = pd.DataFrame(columns=["plps", "label", "vowel"])
        df["plps"] = [t[0] for t in train_loader.dataset]
        df["label"] = [t[1] for t in train_loader.dataset]
        df["vowel"] = [t[2] for t in train_loader.dataset]

        # Plot the latent space in train
        if hyperparams["material"] == "PATAKA":
            # Plot the latent space in test
            plot_latent_space(
                model,
                df,
                fold,
                hyperparams["wandb_flag"],
                supervised=hyperparams["supervised"],
                name="train",
            )
        elif hyperparams["material"] == "VOWELS":
            plot_latent_space_vowels(
                model,
                df,
                fold,
                hyperparams["wandb_flag"],
                supervised=hyperparams["supervised"],
                name="train",
            )


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
    #     "--n_classes",
    #     type=int,
    #     choices=[2, 5],
    #     default=2,
    #     help="Number of classes to supervise. Only used if supervised=True. 2 for PD, 5 for VOWELS.",
    # )
    # args = parser.parse_args()

    args = {}

    main(args)
