from models.pt_models import SpeechTherapist
from training.pt_training import SpeechTherapist_trainer, SpeechTherapist_tester
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import sys
import os
import argparse
from utils.utils import make_balanced_sampler, augment_data, stratify_dataset


def main(args, hyperparams):
    gpu = args.gpu
    fold = args.fold
    device = torch.device("cuda:" + str(gpu) if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    hyperparams["path_to_save"] = (
        "local_results/spectrograms/manner_gmvae_alb_neurovoz_"
        + str(hyperparams["latent_dim"])
        + "final_model_classifier"
        + "_LATENTSPACE+manner_CNN_fold"
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
        new_train = [data for data in train_loader.dataset if data[3] == "neurovoz"]

        # First, remove all albayzin samples from val_loader
        new_val = [data for data in val_loader.dataset if data[3] == "neurovoz"]

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

        #
    else:
        print("Reading data...")
        # Read the data

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
    model = SpeechTherapist(
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

    # model = torch.compile(model)

    if hyperparams["train"]:
        # Load the best unsupervised model to supervise it
        name = (
            "local_results/spectrograms/results_30ms/manner_gmvae_alb_neurovoz_32supervised90-10-fold"
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
        SpeechTherapist_trainer(
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
        "frame_size_ms": 0.400,  # 400ms
        "n_plps": 0,
        "n_mfccs": 0,
        "spectrogram_win_size": 0.030,  # 23ms as it is recommended in the librosa library for speech recognition
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
        "classifier_type": "cnn",  # "cnn" or "mlp"
        "n_gaussians": 16,  # 2 per manner class
        "semisupervised": False,
        "train": True,
        "train_albayzin": True,
        "new_data_partition": False,
        "supervised": True,
        "classifier": True,
        "fold": args.fold,
        "gpu": args.gpu,
    }

    main(args, hyperparams)
