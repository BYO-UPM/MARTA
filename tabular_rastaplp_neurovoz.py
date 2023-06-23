from utils.rasta_py import rastaplp
from utils.preprocess import read_data, extract_rasta_plp_with_derivatives
from models.pt_models import VAE
from training.pt_training import VAE_trainer, VAE_tester
from utils.utils import plot_latent_space
import torch
import wandb
import numpy as np
import pandas as pd


def main(args):
    hyperparams = {
        "frame_size_ms": args.frame_size_ms,
        "hop_size_percent": args.hop_size_percent,
        "n_plps": args.n_plps,
        "wandb_flag": args.wandb_flag,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "latent_dim": args.latent_dim,
        "hidden_dims_enc": args.hidden_dims_enc,
        "hidden_dims_dec": args.hidden_dims_dec,
    }

    # Read the data
    # data = read_data(args.data_path, args.wandb_flag)

    # Extract the RASTA-PLP features
    # data["plps"] = data.apply(
    #    lambda x: extract_rasta_plp_with_derivatives(
    #        x["norm_signal"],
    #        x["sr"],
    #        hyperparams["frame_size_ms"],
    #        hyperparams["n_plps"],
    #    ),
    #    axis=1,
    # )

    # Data explode
    # data = data.explode("plps")

    # Generate a synthetic dataset with random values. It has to have 3 columns: fold, plps and labels
    data = pd.DataFrame(
        {
            "fold": np.random.randint(0, 5, 1000),
            "plps": [np.random.rand(39) for _ in range(1000)],
            "label": np.random.randint(0, 2, 1000),
        }
    )

    folds = data["fold"].unique()
    for fold in folds:
        training_data = data[data["fold"] != fold]
        # Subsample the train data to have a valid data
        train_data = training_data.sample(
            n=int(len(training_data) * 0.8), random_state=42
        )
        valid_data = training_data.drop(train_data.index)
        test_data = data[data["fold"] == fold]

        label_counts = train_data["label"].value_counts()
        total_samples = len(train_data)
        class_weights = 1.0 / torch.Tensor(label_counts.values / total_samples)
        sample_weights = class_weights[train_data["label"].values]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        # Create DataLoaders
        train_dataloader = torch.utils.data.DataLoader(
            dataset=np.vstack(train_data["plps"]),
            batch_size=hyperparams["batch_size"],
            sampler=sampler,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            dataset=np.vstack(valid_data["plps"]),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=np.vstack(test_data["plps"]),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
        )

        # Create the model
        model = VAE(
            hyperparams["n_plps"]
            * 3,  # 3 is because we have the derivatives (1st and 2nd)
            latent_dim=hyperparams["latent_dim"],
            hidden_dims_enc=hyperparams["hidden_dims_enc"],
            hidden_dims_dec=hyperparams["hidden_dims_dec"],
        )

        # TRain the model
        VAE_trainer(
            model,
            train_dataloader,
            valid_dataloader,
            epochs=hyperparams["epochs"],
            beta=1.0,
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
        )

        # Test the model
        test_loss_mse, test_loss_nll = VAE_tester(model, test_dataloader)

        # Plot the latent space in test
        plot_latent_space(model, test_data, fold, hyperparams["wandb_flag"])

        # Plot the latent space in train
        plot_latent_space(model, train_data, fold, hyperparams["wandb_flag"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2",
        help="Path to the data",
    )
    parser.add_argument(
        "--frame_size_ms",
        type=int,
        default=15,
        help="Frame size in milliseconds",
    )
    parser.add_argument(
        "--hop_size_percent",
        type=float,
        default=0.5,
        help="Hop size in percent",
    )
    parser.add_argument(
        "--n_plps",
        type=int,
        default=13,
        help="Number of RASTA-PLP coefficients",
    )
    parser.add_argument(
        "--wandb_flag",
        type=bool,
        default=False,
        help="Flag to use wandb",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2,
        help="Latent dimension",
    )
    parser.add_argument(
        "--hidden_dims_enc",
        type=list,
        default=[30, 30, 30, 30],
        help="Hidden dimensions of the encoder",
    )
    parser.add_argument(
        "--hidden_dims_dec",
        type=list,
        default=[30, 30, 30, 30],
        help="Hidden dimensions of the decoder",
    )
    args = parser.parse_args()

    main(args)
