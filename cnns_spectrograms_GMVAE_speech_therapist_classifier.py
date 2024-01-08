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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device being used:", device)


def main(args, hyperparams):
    if hyperparams["train_albayzin"]:
        hyperparams["path_to_save"] = (
            "local_results/spectrograms/manner_gmvae_alb_neurovoz_"
            + str(hyperparams["latent_dim"])
            + "final_model_classifier"
            + "_LATENTSPACE+manner_CNN3"
        )

    else:
        hyperparams["path_to_save"] = (
            "local_results/spectrograms/manner_gmvae_only_neurovoz_"
            + str(hyperparams["latent_dim"])
            + "final_model_classifier"
            + "adding_"
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

    # First check in local_results/ if there eist any .pt file with the dataloaders
    # If not, create them and save them in local_results/

    if os.path.exists("local_results/train_loader0.4spec_winsize_0.023hopsize_0.5.pt"):
        print("Reading train, val and test loaders from local_results/...")
        train_loader = torch.load(
            "local_results/train_loader0.4spec_winsize_0.023hopsize_0.5.pt"
        )
        val_loader = torch.load(
            "local_results/val_loader0.4spec_winsize_0.023hopsize_0.5.pt"
        )
        test_loader = torch.load(
            "local_results/test_loader0.4spec_winsize_0.023hopsize_0.5.pt"
        )
        test_data = torch.load(
            "local_results/test_data0.4spec_winsize_0.023hopsize_0.5.pt"
        )

        # Split val_loader in two: val_loader and test_loader
        new_train, new_val = torch.utils.data.random_split(
            val_loader.dataset,
            [
                int(0.5 * len(val_loader.dataset)),
                int(0.5 * len(val_loader.dataset)),
            ],
        )
        new_train_loader = torch.utils.data.DataLoader(
            dataset=new_train,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val2_loader = torch.utils.data.DataLoader(
            dataset=new_val,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        new_test_set = torch.utils.data.DataLoader(
            dataset=test_loader.dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        #
    else:
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
    model = SpeechTherapist(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_gaussians=hyperparams["n_gaussians"],
        n_manner=8,
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        classifier=hyperparams["classifier"],
        weights=hyperparams["weights"],
        device=device,
    )

    # model = torch.compile(model)

    if hyperparams["train"]:
        # Load the best unsupervised model to supervise it
        name = "local_results/spectrograms/manner_gmvae_alb_neurovoz_32final_modeltesting_supervised2/GMVAE_cnn_best_model_2d.pt"
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
        SpeechTherapist_trainer(
            model=model,
            trainloader=train_loader,
            validloader=val_loader,
            epochs=hyperparams["epochs"],
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
            path_to_save=hyperparams["path_to_save"],
            supervised=hyperparams["supervised"],
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
    SpeechTherapist_tester(
        model=model,
        testloader=test_loader,
        test_data=test_data,
        supervised=True,  # Not implemented yet
        wandb_flag=hyperparams["wandb_flag"],
        path_to_plot=hyperparams["path_to_save"],
        best_threshold=threshold,
    )
    print("Testing finished!")

    # Create an empty pd dataframe with three columns: data, label and manner
    # df_train = pd.DataFrame(columns=[audio_features, "label", "manner"])
    # df_train[audio_features] = [t[0] for t in train_loader.dataset]
    # df_train["label"] = [t[1] for t in train_loader.dataset]
    # df_train["manner"] = [t[2] for t in train_loader.dataset]

    # # Select randomly 1000 samples of dftrain
    # # df_train = df_train.sample(n=1000)

    # # Create an empty pd dataframe with three columns: data, label and manner
    # df_test = pd.DataFrame(columns=[audio_features, "label", "manner"])
    # df_test[audio_features] = [t[0] for t in test_loader.dataset]
    # df_test["label"] = [t[1] for t in test_loader.dataset]
    # df_test["manner"] = [t[2] for t in test_loader.dataset]

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
        "epochs": 500,
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
        "supervised": True,
        "classifier": "cnn",  # "cnn" or "mlp"
        "n_gaussians": 16,  # 2 per manner class
        "semisupervised": False,
        "train": True,
        "train_albayzin": True,  # If True, only albayzin+neuro is used to train. If False only neuro are used for training
    }

    main(args, hyperparams)
