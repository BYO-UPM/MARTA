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

    if hyperparams["wandb_flag"]:
        wandb.init(
            project="parkinson",
            config=hyperparams,
            group="rasta_plp_vae",
            name="preprocess",
        )

    print("Reading data...")
    # Read the data
    data = read_data(args.data_path, args.wandb_flag)

    print("Extracting RASTA-PLP features...")
    # Extract the RASTA-PLP features
    data["plps"] = data.apply(
        lambda x: extract_rasta_plp_with_derivatives(
            x["norm_signal"],
            x["sr"],
            hyperparams["frame_size_ms"],
            hyperparams["n_plps"],
        ),
        axis=1,
    )

    print("Exploding data...")
    # Data explode
    data = data.explode("plps")

    # Selecting only relevant columns: fold, plps and labels
    data = data[["fold", "plps", "label"]]

    # Binarise the labels
    data["label"] = data["label"].apply(lambda x: 1 if x == "PD" else 0)

    # Generate a synthetic dataset with random values. It has to have 3 columns: fold, plps and labels
    # data = pd.DataFrame(
    #     {
    #         "fold": np.random.randint(0, 5, 1000),
    #         "plps": [np.random.rand(39) for _ in range(1000)],
    #         "label": np.random.randint(0, 2, 1000),
    #     }
    # )

    folds = data["fold"].unique()
    for fold in folds:
        if hyperparams["wandb_flag"]:
            wandb.finish()
            wandb.init(
                project="parkinson",
                config=hyperparams,
                group="rasta_plp_vae",
                name="fold_" + str(fold),
            )
        print("Training a VAE for fold: ", fold)

        print("Generating train, valid and test data...")
        training_data = data[data["fold"] != fold]
        # Subsample the train data to have a valid data

        print("Sampling from training")
        from sklearn.model_selection import train_test_split

        # Running sklearn
        print("Training + val data shape: ", training_data.shape)
        train_data, valid_data = train_test_split(
            training_data, test_size=0.2, random_state=42
        )
        print("Spllited, generating samplers for unbalanced data")
        test_data = data[data["fold"] == fold]

        label_counts = train_data["label"].value_counts()
        total_samples = len(train_data)
        class_weights = 1.0 / torch.Tensor(label_counts.values / total_samples)

        # Cast to integer to avoid problems with the sampler
        sample_weights = class_weights[train_data["label"].values]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        print("Creating DataLoaders...")
        # Create DataLoaders
        print("Train data shape: ", np.vstack(train_data["plps"]).shape)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=np.vstack(train_data["plps"]),
            batch_size=hyperparams["batch_size"],
            sampler=sampler,
        )
        print("Valid data shape: ", np.vstack(valid_data["plps"]).shape)
        valid_dataloader = torch.utils.data.DataLoader(
            dataset=np.vstack(valid_data["plps"]),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
        )
        print("Test data shape: ", np.vstack(test_data["plps"]).shape)
        test_dataloader = torch.utils.data.DataLoader(
            dataset=np.vstack(test_data["plps"]),
            batch_size=hyperparams["batch_size"],
            shuffle=True,
        )

        print("Defining models...")
        # Create the model
        model = VAE(
            np.vstack(train_data["plps"]).shape[1],
            latent_dim=hyperparams["latent_dim"],
            hidden_dims_enc=hyperparams["hidden_dims_enc"],
            hidden_dims_dec=hyperparams["hidden_dims_dec"],
        )

        print("Training VAE...")
        # TRain the model
        VAE_trainer(
            model,
            train_dataloader,
            valid_dataloader,
            epochs=hyperparams["epochs"],
            beta=0.5,
            lr=hyperparams["lr"],
            wandb_flag=hyperparams["wandb_flag"],
        )
        print("Training finished!")

        print("Testing VAE...")
        # Test the model
        test_loss_mse, test_loss_nll = VAE_tester(model, test_dataloader)

        # Plot the latent space in test
        plot_latent_space(
            model, test_data, fold, hyperparams["wandb_flag"], name="test"
        )

        # Plot the latent space in train
        plot_latent_space(
            model, train_data, fold, hyperparams["wandb_flag"], name="train"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VAE training")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2/PATAKA",
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
        default=[20, 10],
        help="Hidden dimensions of the encoder",
    )
    parser.add_argument(
        "--hidden_dims_dec",
        type=list,
        default=[10],
        help="Hidden dimensions of the decoder",
    )
    args = parser.parse_args()

    main(args)
