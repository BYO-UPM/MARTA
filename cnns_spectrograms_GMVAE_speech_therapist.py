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
            + "final_model"
            + "testingsupervised"
        )

    # Create the path if does not exist
    if not os.path.exists(hyperparams["path_to_save"]):
        os.makedirs(hyperparams["path_to_save"])

    old_stdout = sys.stdout
    log_file = open(hyperparams["path_to_save"] + "/log.txt", "w")
    sys.stdout = log_file

    print("Reading data...")
    # Read the data
    dataset = Dataset_AudioFeatures(
        "labeled/NeuroVoz",
        hyperparams,
    )

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

    (
        train_loader,
        val_loader,
        test_loader,
        _,  # train_data, not used
        _,  # val_data, not used
        test_data,
    ) = dataset.get_dataloaders(train_albayzin=hyperparams["train_albayzin"])

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
        supervised=False,  # Not implemented yet
        wandb_flag=hyperparams["wandb_flag"],
        path_to_plot=hyperparams["path_to_save"],
    )
    print("Testing finished!")

    # Create an empty pd dataframe with three columns: data, label and manner
    df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_train[audio_features] = [t[0] for t in train_loader.dataset]
    df_train["label"] = [t[1] for t in train_loader.dataset]
    df_train["manner"] = [t[2] for t in train_loader.dataset]

    ## ALERT: REMOVE THIS LINE AFTER TESTING!!!!
    # Substract 8 to manner if their corresponidng label is 1
    df_train["manner"] = df_train.apply(
        lambda x: x["manner"] - 8 if x["label"] == 1 else x["manner"], axis=1
    )

    # Select randomly 1000 samples of dftrain
    # df_train = df_train.sample(n=1000)

    # Create an empty pd dataframe with three columns: data, label and manner
    df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    df_test[audio_features] = [t[0] for t in test_loader.dataset]
    df_test["label"] = [t[1] for t in test_loader.dataset]
    df_test["manner"] = [t[2] for t in test_loader.dataset]

    ## ALERT: REMOVE THIS LINE AFTER TESTING!!!!
    # Substract 8 to manner if their corresponidng label is 1
    df_test["manner"] = df_test.apply(
        lambda x: x["manner"] - 8 if x["label"] == 1 else x["manner"], axis=1
    )

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
        "frame_size_ms": 0.400,  # 400ms
        "n_plps": 0,
        "n_mfccs": 0,
        "spectrogram_win_size": 0.023,  # 23ms as it is recommended in the librosa library for speech recognition
        "material": "MANNER",
        "hop_size_percent": 0.5,
        "spectrogram": True,
        "wandb_flag": False,
        "epochs": 1000,
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
        "supervised": False,
        "cnn_classifier": True,
        "n_gaussians": 16,  # 2 per manner class
        "semisupervised": False,
        "train": False,
        "train_albayzin": True,  # If True, only albayzin+neuro is used to train. If False only neuro are used for training
    }

    main(args, hyperparams)
