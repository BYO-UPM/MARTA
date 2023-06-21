from utils.rasta_py import rastaplp
from utils.preprocess import read_data, extract_rasta_plp_with_derivatives
from models.vae import VAE
from training.vae_trainer import VAE_trainer
import torch



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
    data = read_data(args.data_path, args.wandb_flag)

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

    # Data explode
    data = data.explode("plps")

    # Create DataLoader with stratisfied sampling
    train


