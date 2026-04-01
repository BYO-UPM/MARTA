"""
MARTA latent distance analysis

This script loads a pretrained MARTA model and reproduces the latent-space
analysis performed in `MARTA_Supervised_international.py`, generating the
Jensen-Shannon distance heatmaps alongside additional Fréchet distance
visualizations per manner class and dataset combination.
"""

import argparse
import os
import torch
import pandas as pd

from models.pt_models import MARTA
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
from utils.utils import (
    make_balanced_sampler,
    stratify_per_dataset,
    plot_logopeda_alb_neuro,
)


def load_dataloaders(hyperparams):
    if hyperparams["new_data_partition"]:
        dataset = Dataset_AudioFeatures(hyperparams)
        (
            train_loader,
            val_loader,
            test_loader,
            _,
            _,
            test_data,
        ) = dataset.get_dataloaders(
            experiment=hyperparams["experiment"], supervised=hyperparams["supervised"]
        )
    else:
        print("Reading train, val and test loaders from local_results/...")
        suffix = (
            "frame_size_0.4spec_winsize_"
            + str(hyperparams["spectrogram_win_size"])
            + "hopsize_0.5fold"
            + str(hyperparams["fold"])
            + ".pt"
        )
        base = "local_results/folds/"
        train_loader = torch.load(base + "train_loader_supervised_True_" + suffix)
        val_loader = torch.load(base + "val_loader_supervised_True_" + suffix)
        test_loader = torch.load(base + "test_loader_supervised_True_" + suffix)
        test_data = torch.load(base + "test_data_supervised_True_" + suffix)
    return train_loader, val_loader, test_loader, test_data


def build_crosslingual_splits(train_loader, val_loader, test_loader, hyperparams):
    gita_data_test = [data for data in test_loader.dataset if data[3] == "gita"]
    gita_data_val = [data for data in val_loader.dataset if data[3] == "gita"]
    gita_data_train = [data for data in train_loader.dataset if data[3] == "gita"]

    neurovoz_data_test = [data for data in test_loader.dataset if data[3] == "neurovoz"]
    neurovoz_data_val = [data for data in val_loader.dataset if data[3] == "neurovoz"]
    neurovoz_data_train = [data for data in train_loader.dataset if data[3] == "neurovoz"]

    albayzin_data_test = [data for data in test_loader.dataset if data[3] == "albayzin"]
    albayzin_data_val = [data for data in val_loader.dataset if data[3] == "albayzin"]
    albayzin_data_train = [data for data in train_loader.dataset if data[3] == "albayzin"]

    italian_data_test = [data for data in test_loader.dataset if data[3] == "italian"]
    italian_data_val = [data for data in val_loader.dataset if data[3] == "italian"]
    italian_data_train = [data for data in train_loader.dataset if data[3] == "italian"]

    librispeech_data_test = [data for data in test_loader.dataset if data[3] == "librispeech"]
    librispeech_data_val = [data for data in val_loader.dataset if data[3] == "librispeech"]
    librispeech_data_train = [data for data in train_loader.dataset if data[3] == "librispeech"]

    crosslingual = hyperparams["crosslingual"]
    if crosslingual == "testing_gita":
        new_train = (
            neurovoz_data_train
            + albayzin_data_train
            + neurovoz_data_test
            + librispeech_data_train
        )
        new_val = neurovoz_data_val + albayzin_data_val + librispeech_data_val
        gita_data_val = [(data[0], data[1], data[2], data[3]) for data in gita_data_val]
        new_test = (
            gita_data_test
            + gita_data_train
            + gita_data_val
            + albayzin_data_test
            + librispeech_data_test
        )
        print("Crosslingual scenario: everything -> gita")
    elif crosslingual == "testing_neurovoz":
        new_train = (
            gita_data_train
            + gita_data_test
            + albayzin_data_train
            + librispeech_data_train
        )
        new_val = gita_data_val + albayzin_data_val + librispeech_data_val
        neurovoz_data_val = [(data[0], data[1], data[2], data[3]) for data in neurovoz_data_val]
        new_test = (
            neurovoz_data_test
            + neurovoz_data_train
            + neurovoz_data_val
            + albayzin_data_test
            + librispeech_data_test
        )
        print("Crosslingual scenario: everything -> neurovoz")
    elif crosslingual == "testing_italian":
        new_train = (
            gita_data_train
            + gita_data_test
            + albayzin_data_train
            + albayzin_data_test
            + neurovoz_data_train
            + neurovoz_data_test
            + librispeech_data_train
        )
        new_val = gita_data_val + albayzin_data_val + neurovoz_data_val + librispeech_data_val
        italian_data_val = [(data[0], data[1], data[2], data[3]) for data in italian_data_val]
        new_test = (
            italian_data_test
            + italian_data_train
            + italian_data_val
            + librispeech_data_test
        )
        print("Crosslingual scenario: everything -> italian")
    else:
        new_train = (
            gita_data_train + neurovoz_data_train + albayzin_data_train + librispeech_data_train
        )
        new_val = gita_data_val + neurovoz_data_val + albayzin_data_val + librispeech_data_val
        new_test = gita_data_test + neurovoz_data_test + albayzin_data_test + librispeech_data_test
        print("Multilingual scenario:")

    new_train = stratify_per_dataset(new_train)
    train_sampler = make_balanced_sampler(new_train, validation=False)
    val_sampler = make_balanced_sampler(new_val, validation=True)

    train_loader = torch.utils.data.DataLoader(
        new_train,
        batch_size=512,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        new_val,
        batch_size=512,
        sampler=val_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        new_test,
        batch_size=512,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def build_dataframe(loader, feature_key):
    df = pd.DataFrame(columns=[feature_key, "label", "manner"])
    df[feature_key] = [t[0] for t in loader.dataset]
    df["label"] = [t[1] for t in loader.dataset]
    df["manner"] = [t[2] for t in loader.dataset]
    df["dataset"] = [t[3] for t in loader.dataset]
    df["manner"] = df.apply(lambda x: x["manner"] - 8 if x["label"] == 1 else x["manner"], axis=1)
    return df


def main(args):
    hyperparams = {
        "spectrogram": True,
        "frame_size_ms": 0.400,
        "spectrogram_win_size": 0.030,
        "hop_size_percent": 0.5,
        "epochs": 200,
        "batch_size": 128,
        "lr": 1e-3,
        "latent_dim": args.latent_dim,
        "n_gaussians": 16,
        "hidden_dims_enc": [64, 1024, 64],
        "hidden_dims_gmvae": [256],
        "weights": [1, 1, 1, 10],
        "domain_adversarial": args.domain_adversarial,
        "crosslingual": args.cross_lingual,
        "classifier_type": "cnn",
        "classifier": False,
        "supervised": True,
        "experiment": "fourth",
        "train": False,
        "train_albayzin": True,
        "new_data_partition": args.new_data_partition,
        "fold": args.fold,
        "gpu": args.gpu,
    }

    if args.model_dir:
        hyperparams["path_to_save"] = args.model_dir
    else:
        hyperparams["path_to_save"] = (
            "local_results/spectrograms/marta_"
            + str(hyperparams["latent_dim"])
            + "_supervised__domain_adversarial_"
            + str(hyperparams["domain_adversarial"])
            + "_fold_"
            + str(hyperparams["fold"])
        )

    os.makedirs(hyperparams["path_to_save"], exist_ok=True)

    device = torch.device(
        "cuda:" + str(hyperparams["gpu"]) if torch.cuda.is_available() else "cpu"
    )
    print("Device being used:", device)

    train_loader, val_loader, test_loader, _ = load_dataloaders(hyperparams)
    train_loader, _, test_loader = build_crosslingual_splits(
        train_loader, val_loader, test_loader, hyperparams
    )

    model = MARTA(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_manner=16,
        n_gaussians=hyperparams["n_gaussians"],
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        weights=hyperparams["weights"],
        device=device,
        reducer="sum",
        domain_adversarial_bool=hyperparams["domain_adversarial"],
        datasets=3,
    )

    checkpoint_name = (
        args.model_checkpoint if args.model_checkpoint else "GMVAE_cnn_best_model_2d.pt"
    )
    checkpoint_path = os.path.join(hyperparams["path_to_save"], checkpoint_name)
    print(f"Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    audio_features = "spectrogram"
    df_train = build_dataframe(train_loader, audio_features)
    df_test = build_dataframe(test_loader, audio_features)

    print("Starting latent distance analysis (JSD + Fréchet)...")
    plot_logopeda_alb_neuro(
        model,
        df_train,
        df_test,
        wandb_flag=False,
        name="distance_analysis",
        supervised=hyperparams["supervised"],
        samples=5000,
        path_to_plot=hyperparams["path_to_save"],
        compute_frechet=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MARTA latent space distance analysis")
    parser.add_argument("--fold", type=int, default=0, help="Fold number for the experiment")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--latent_dim", type=int, default=3, help="Latent dimension of the pretrained model"
    )
    parser.add_argument(
        "--domain_adversarial",
        type=int,
        default=0,
        help="Domain adversarial flag used for the pretrained model",
    )
    parser.add_argument(
        "--cross_lingual",
        type=str,
        default="multilingual",
        choices=["multilingual", "testing_gita", "testing_neurovoz", "testing_italian"],
        help="Cross-lingual scenario to reproduce",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory containing the pretrained checkpoint and where plots will be saved",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Checkpoint filename inside model_dir (defaults to GMVAE_cnn_best_model_2d.pt)",
    )
    parser.add_argument(
        "--new_data_partition",
        action="store_true",
        help="Rebuild folds instead of loading cached dataloaders from disk",
    )

    args = parser.parse_args()
    main(args)
