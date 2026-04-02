#!/usr/bin/env python3
"""
Standalone reviewer experiment for the unsupervised MARTA GMVAE.

This script wraps the original unsupervised training pipeline in one entrypoint that can:
1. Reuse cached fold loaders or rebuild them from raw data.
2. Train or load an unsupervised MARTA checkpoint with a configurable number of Gaussians.
3. Recompute reconstruction metrics on the test set.
4. Export latent-space projections and JSD heatmaps for the train/healthy/PD comparison.

It defaults to the reviewer baseline requested in the paper review: a single Gaussian.

Examples
--------
Train the 1-Gaussian baseline on fold 0:
    python reviewer_unsupervised_gmvae_baseline.py --fold 0 --n-gaussians 1 --device cuda:0

Run the same pipeline with the original multi-Gaussian setup for comparison:
    python reviewer_unsupervised_gmvae_baseline.py --fold 0 --n-gaussians 8 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon

from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
from models.pt_models import MARTA
from training.pt_training import MARTA_trainer
from utils.definitions import NEUROVOZ_LABELS_LOCAL, PROCESSED_DATA_LOCAL


MANNER_NAMES = {
    0: "Plosives",
    1: "Plosives voiced",
    2: "Nasals",
    3: "Fricatives",
    4: "Liquids",
    5: "Vowels",
}

MANNER_COLORS = {
    0: "#1f77b4",
    1: "#d62728",
    2: "#2ca02c",
    3: "#9467bd",
    4: "#ff7f0e",
    5: "#17becf",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the unsupervised MARTA analysis with a configurable number of Gaussians."
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="Fold tag appended after 'fold' in the cached filenames, for example '0' or 'first_experiment'.",
    )
    parser.add_argument(
        "--folds-dir",
        type=str,
        default=str(Path(PROCESSED_DATA_LOCAL) / "folds"),
        help="Directory that contains the cached fold artifacts. Recursive lookup is supported.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, for example cuda:0 or cpu.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index used when --device is not set.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for sampling and t-SNE.")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and analysis.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension.")
    parser.add_argument("--n-gaussians", type=int, default=1, help="Number of Gaussian components.")
    parser.add_argument(
        "--hidden-dims-enc",
        type=int,
        nargs="+",
        default=[64, 1024, 64],
        help="Encoder/decoder hidden dimensions.",
    )
    parser.add_argument(
        "--hidden-dims-gmvae",
        type=int,
        nargs="+",
        default=[256],
        help="GMVAE hidden dimensions.",
    )
    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs=4,
        default=[1.0, 1.0, 1.0, 10.0],
        metavar=("REC", "GAUSS", "CAT", "METRIC"),
        help="Loss weights in the same order as the original MARTA code.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sumloss",
        choices=["sumloss", "pcgrad", "mgd", "graddrop", "cagrad"],
        help="Gradient manipulation method used by the original trainer.",
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the model before evaluation. Use --no-train to only evaluate an existing checkpoint.",
    )
    parser.add_argument(
        "--new-data-partition",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force a rebuild of the cached folds from the raw datasets.",
    )
    parser.add_argument(
        "--train-albayzin",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass the original train_albayzin flag into the fold-building code.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to <output_dir>/GMVAE_cnn_best_model_2d.pt.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory used for checkpoints, plots, and metrics.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name used when --output-dir is not provided.",
    )
    parser.add_argument(
        "--tsne-max-per-group",
        type=int,
        default=500,
        help="Maximum number of points per (label, manner) group used in the t-SNE projection.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=40.0,
        help="Requested t-SNE perplexity. The script will clamp it when the sample is small.",
    )
    parser.add_argument(
        "--jsd-samples",
        type=int,
        default=1000,
        help="Number of positions sampled when estimating each JSD value.",
    )
    parser.add_argument(
        "--train-loader-path",
        type=str,
        default=None,
        help="Optional explicit path to the cached train loader.",
    )
    parser.add_argument(
        "--val-loader-path",
        type=str,
        default=None,
        help="Optional explicit path to the cached validation loader.",
    )
    parser.add_argument(
        "--test-loader-path",
        type=str,
        default=None,
        help="Optional explicit path to the cached test loader.",
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        default=None,
        help="Optional explicit path to the cached test dataframe.",
    )
    return parser.parse_args()


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device is not None:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu}")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return ensure_dir(Path(args.output_dir))

    run_name = args.run_name
    if run_name is None:
        run_name = (
            f"reviewer_unsup_fold{args.fold}_ld{args.latent_dim}_k{args.n_gaussians}"
        )
    return ensure_dir(Path(PROCESSED_DATA_LOCAL) / "reviewer_baselines" / run_name)


def make_hyperparams(args: argparse.Namespace, output_dir: Path) -> Dict:
    return {
        "spectrogram": True,
        "frame_size_ms": 0.400,
        "spectrogram_win_size": 0.030,
        "hop_size_percent": 0.5,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "latent_dim": args.latent_dim,
        "n_gaussians": args.n_gaussians,
        "hidden_dims_enc": args.hidden_dims_enc,
        "hidden_dims_gmvae": args.hidden_dims_gmvae,
        "weights": args.loss_weights,
        "cnn_classifier": False,
        "supervised": False,
        "method": args.method,
        "train": args.train,
        "train_albayzin": args.train_albayzin,
        "new_data_partition": args.new_data_partition,
        "material": "MANNER",
        "n_plps": 0,
        "n_mfccs": 0,
        "wandb_flag": False,
        "semisupervised": False,
        "path_to_save": str(output_dir),
    }


def default_folds_dir() -> Path:
    return Path(PROCESSED_DATA_LOCAL) / "folds"


def resolve_cached_artifact(base_dir: Path, expected_name: str, explicit_path: str | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    exact_path = base_dir / expected_name
    if exact_path.exists():
        return exact_path

    exact_matches = sorted(base_dir.rglob(expected_name))
    if len(exact_matches) == 1:
        print(f"Resolved cached artifact {expected_name} -> {exact_matches[0]}")
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise FileNotFoundError(
            "Multiple cached artifacts matched "
            f"{expected_name}: {', '.join(str(path) for path in exact_matches)}. "
            "Use --folds-dir or the explicit --*-path arguments."
        )

    fuzzy_matches = sorted(base_dir.rglob(f"*{expected_name}"))
    if len(fuzzy_matches) == 1:
        print(f"Resolved cached artifact {expected_name} -> {fuzzy_matches[0]}")
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        raise FileNotFoundError(
            "Multiple cached artifacts loosely matched "
            f"{expected_name}: {', '.join(str(path) for path in fuzzy_matches)}. "
            "Use --folds-dir or the explicit --*-path arguments."
        )

    return exact_path


def fold_artifact_paths(args: argparse.Namespace, hyperparams: Dict) -> Dict[str, Path]:
    base = Path(args.folds_dir)
    suffix = (
        f"_supervised_{hyperparams['supervised']}"
        f"_frame_size_{hyperparams['frame_size_ms']}"
        f"spec_winsize_{hyperparams['spectrogram_win_size']}"
        f"hopsize_{hyperparams['hop_size_percent']}"
        f"fold{args.fold}.pt"
    )
    names = {
        "train_loader": f"train_loader{suffix}",
        "val_loader": f"val_loader{suffix}",
        "test_loader": f"test_loader{suffix}",
        "test_data": f"test_data{suffix}",
    }
    return {
        "train_loader": resolve_cached_artifact(base, names["train_loader"], args.train_loader_path),
        "val_loader": resolve_cached_artifact(base, names["val_loader"], args.val_loader_path),
        "test_loader": resolve_cached_artifact(base, names["test_loader"], args.test_loader_path),
        "test_data": resolve_cached_artifact(base, names["test_data"], args.test_data_path),
    }


def maybe_build_folds(args: argparse.Namespace, hyperparams: Dict, artifact_paths: Dict[str, Path]) -> None:
    have_all_folds = all(path.exists() for path in artifact_paths.values())
    if have_all_folds and not args.new_data_partition:
        print("Using cached fold artifacts.")
        return

    using_custom_fold_source = (
        Path(args.folds_dir) != default_folds_dir()
        or args.train_loader_path is not None
        or args.val_loader_path is not None
        or args.test_loader_path is not None
        or args.test_data_path is not None
        or args.fold != "0"
    )

    if using_custom_fold_source:
        missing = [str(path) for path in artifact_paths.values() if not path.exists()]
        raise FileNotFoundError(
            "The requested cached fold artifacts were not found, so the script would fall back to rebuilding "
            "raw folds. That is disabled for custom fold sources.\nMissing paths:\n- "
            + "\n- ".join(missing)
            + "\nPass the correct --folds-dir / --fold tag, or the explicit --*-path options."
        )

    ensure_dir(Path(PROCESSED_DATA_LOCAL))
    ensure_dir(Path(PROCESSED_DATA_LOCAL) / "folds")

    if have_all_folds and args.new_data_partition:
        print("Rebuilding folds because --new-data-partition was set.")
    else:
        print("Cached fold artifacts are missing. Rebuilding them from the raw data.")

    dataset = Dataset_AudioFeatures(NEUROVOZ_LABELS_LOCAL, hyperparams)
    dataset.get_dataloaders(
        train_albayzin=args.train_albayzin,
        verbose=True,
        supervised=False,
    )

    missing = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Fold generation finished but the requested fold artifacts were not written: "
            + ", ".join(missing)
        )


def load_fold_data(artifact_paths: Dict[str, Path]):
    print(f"Loading fold artifacts from {artifact_paths['train_loader'].parent}")
    return (
        torch_load(artifact_paths["train_loader"]),
        torch_load(artifact_paths["val_loader"]),
        torch_load(artifact_paths["test_loader"]),
        torch_load(artifact_paths["test_data"]),
    )


def build_model(train_loader, hyperparams: Dict, device: torch.device) -> MARTA:
    model = MARTA(
        x_dim=train_loader.dataset[0][0].shape,
        z_dim=hyperparams["latent_dim"],
        n_gaussians=hyperparams["n_gaussians"],
        hidden_dims_spectrogram=hyperparams["hidden_dims_enc"],
        hidden_dims_gmvae=hyperparams["hidden_dims_gmvae"],
        weights=hyperparams["weights"],
        device=device,
        reducer="sum",
    )
    return model


def checkpoint_path(output_dir: Path, args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    return output_dir / "GMVAE_cnn_best_model_2d.pt"


def train_if_requested(
    model: MARTA,
    train_loader,
    val_loader,
    hyperparams: Dict,
    output_dir: Path,
) -> None:
    if not hyperparams["train"]:
        print("Skipping training and expecting an existing checkpoint.")
        return

    MARTA_trainer(
        model=model,
        trainloader=train_loader,
        validloader=val_loader,
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
        wandb_flag=False,
        path_to_save=str(output_dir),
        supervised=False,
        classifier=False,
        method=hyperparams["method"],
    )


def load_checkpoint(model: MARTA, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch_load(ckpt_path)
    model.load_state_dict(state["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")


def dataframe_from_loader(loader) -> pd.DataFrame:
    df = pd.DataFrame(columns=["spectrogram", "label", "manner", "dataset"])
    df["spectrogram"] = [item[0] for item in loader.dataset]
    df["label"] = [item[1] for item in loader.dataset]
    df["manner"] = [item[2] for item in loader.dataset]
    df["dataset"] = [item[3] for item in loader.dataset]
    return df


def extract_latent_mu(model: MARTA, data: pd.DataFrame, batch_size: int) -> np.ndarray:
    model.eval()
    latent_batches: List[np.ndarray] = []
    stacked = torch.as_tensor(np.vstack(data["spectrogram"]), dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, len(stacked), batch_size):
            batch = stacked[start : start + batch_size].to(model.device).unsqueeze(1)
            encoded = model.spec_encoder_forward(batch)
            _, _, _, latent_mu_batch, _, _, _ = model.inference_forward(encoded)
            latent_batches.append(latent_mu_batch.detach().cpu().numpy())

    latent_mu = np.vstack(latent_batches)
    print(f"Extracted latent vectors with shape {latent_mu.shape}")
    return latent_mu


def filter_latent_by_manner(
    latent_mu: np.ndarray,
    data: pd.DataFrame,
    supervised: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    manner = np.array([np.array(x) for x in data["manner"]], dtype=int)
    labels = np.array(data["label"].values, dtype=int)

    repeated_labels = np.repeat(labels, manner.shape[1], axis=0)
    flattened_manner = manner.reshape(-1)

    if supervised:
        flattened_manner = flattened_manner.copy()
        flattened_manner[flattened_manner > 7] -= 8

    keep_mask = (flattened_manner != 6) & (flattened_manner != 7)

    filtered_latent = latent_mu[keep_mask]
    filtered_manner = flattened_manner[keep_mask]
    filtered_labels = repeated_labels[keep_mask]

    if filtered_latent.shape[0] != filtered_manner.shape[0]:
        raise ValueError("Latent vectors and manner labels are misaligned after filtering.")

    return filtered_latent, filtered_manner, filtered_labels


def evaluate_reconstruction(model: MARTA, test_loader, test_data: pd.DataFrame, output_dir: Path) -> Dict:
    model.eval()
    x_batches: List[np.ndarray] = []
    xhat_batches: List[np.ndarray] = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(model.device).float()
            _, x_hat, _, _, _, _, _, _, _, _, _ = model(x)
            x_batches.append(x.detach().cpu().numpy())
            xhat_batches.append(x_hat.detach().cpu().numpy())

    x_array = np.concatenate(x_batches, axis=0)
    x_hat_array = np.concatenate(xhat_batches, axis=0)
    mse = float(np.mean((x_array - x_hat_array) ** 2))

    patient_rows = []
    for patient_id in test_data["id_patient"].unique():
        idx = test_data["id_patient"] == patient_id
        patient_mse = float(np.mean((x_array[idx] - x_hat_array[idx]) ** 2))
        patient_rows.append({"id_patient": str(patient_id), "mse": patient_mse})

    patient_df = pd.DataFrame(patient_rows).sort_values("id_patient")
    patient_df.to_csv(output_dir / "reconstruction_per_patient.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].imshow(x_array[0].squeeze(), cmap="viridis", aspect="auto")
    axes[0].set_title("Original test spectrogram")
    axes[1].imshow(x_hat_array[0].squeeze(), cmap="viridis", aspect="auto")
    axes[1].set_title("Reconstructed spectrogram")
    for axis in axes:
        axis.set_xlabel("Frame")
        axis.set_ylabel("Mel bin")
    fig.tight_layout()
    fig.savefig(output_dir / "rec_img.png", dpi=200)
    plt.close(fig)

    metrics = {
        "frame_mse": mse,
        "patient_mse_mean": float(patient_df["mse"].mean()),
        "patient_mse_std": float(patient_df["mse"].std(ddof=0)),
        "num_test_frames": int(x_array.shape[0]),
        "num_test_patients": int(test_data["id_patient"].nunique()),
    }
    return metrics


def sample_group_indices(
    manner: np.ndarray,
    labels: np.ndarray,
    max_per_group: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []

    for label in sorted(np.unique(labels)):
        for manner_class in sorted(np.unique(manner)):
            idx = np.flatnonzero((labels == label) & (manner == manner_class))
            if idx.size == 0:
                continue
            if idx.size > max_per_group:
                idx = rng.choice(idx, size=max_per_group, replace=False)
            selected.append(np.sort(idx))

    if not selected:
        return np.array([], dtype=int)
    return np.concatenate(selected)


def effective_perplexity(requested: float, n_samples: int) -> float:
    if n_samples < 3:
        raise ValueError("t-SNE needs at least 3 samples.")
    max_recommended = max(2.0, (n_samples - 1) / 3.0)
    return float(min(requested, max_recommended))


def compute_tsne_projection(
    train_latent: np.ndarray,
    train_manner: np.ndarray,
    train_labels: np.ndarray,
    test_latent: np.ndarray,
    test_manner: np.ndarray,
    test_labels: np.ndarray,
    max_per_group: int,
    perplexity: float,
    seed: int,
) -> pd.DataFrame:
    train_idx = sample_group_indices(train_manner, train_labels, max_per_group, seed)
    healthy_idx = sample_group_indices(
        test_manner[test_labels == 0],
        test_labels[test_labels == 0],
        max_per_group,
        seed + 1,
    )
    pd_idx = sample_group_indices(
        test_manner[test_labels == 1],
        test_labels[test_labels == 1],
        max_per_group,
        seed + 2,
    )

    splits = []
    if train_idx.size > 0:
        splits.append(
            {
                "name": "train_reference",
                "latent": train_latent[train_idx],
                "manner": train_manner[train_idx],
                "label": train_labels[train_idx],
            }
        )
    if healthy_idx.size > 0:
        splits.append(
            {
                "name": "test_healthy",
                "latent": test_latent[test_labels == 0][healthy_idx],
                "manner": test_manner[test_labels == 0][healthy_idx],
                "label": test_labels[test_labels == 0][healthy_idx],
            }
        )
    if pd_idx.size > 0:
        splits.append(
            {
                "name": "test_pd",
                "latent": test_latent[test_labels == 1][pd_idx],
                "manner": test_manner[test_labels == 1][pd_idx],
                "label": test_labels[test_labels == 1][pd_idx],
            }
        )

    all_latent = np.concatenate([split["latent"] for split in splits], axis=0)
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        perplexity=effective_perplexity(perplexity, all_latent.shape[0]),
    )
    all_proj = tsne.fit_transform(all_latent)

    rows = []
    start = 0
    for split in splits:
        end = start + split["latent"].shape[0]
        split_proj = all_proj[start:end]
        for idx, coords in enumerate(split_proj):
            rows.append(
                {
                    "split": split["name"],
                    "x": float(coords[0]),
                    "y": float(coords[1]),
                    "manner": int(split["manner"][idx]),
                    "label": int(split["label"][idx]),
                }
            )
        start = end

    return pd.DataFrame(rows)


def plot_projection_by_manner(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for manner_class, manner_name in MANNER_NAMES.items():
        subset = df[df["manner"] == manner_class]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            s=14,
            alpha=0.8,
            color=MANNER_COLORS[manner_class],
            label=manner_name,
            linewidths=0,
        )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def plot_reference_overlay(reference: pd.DataFrame, query: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for manner_class in MANNER_NAMES:
        ref_subset = reference[reference["manner"] == manner_class]
        if not ref_subset.empty:
            ax.scatter(
                ref_subset["x"],
                ref_subset["y"],
                s=10,
                alpha=0.10,
                color=MANNER_COLORS[manner_class],
                linewidths=0,
            )

    for manner_class, manner_name in MANNER_NAMES.items():
        query_subset = query[query["manner"] == manner_class]
        if query_subset.empty:
            continue
        ax.scatter(
            query_subset["x"],
            query_subset["y"],
            s=18,
            alpha=0.9,
            color=MANNER_COLORS[manner_class],
            label=manner_name,
            edgecolors="black",
            linewidths=0.2,
        )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def scott_bandwidth(data: np.ndarray) -> float:
    n_samples, n_dims = data.shape
    if n_samples <= 1:
        return 0.1
    std = np.std(data, axis=0)
    scale = float(np.mean(std[std > 0])) if np.any(std > 0) else 0.1
    bandwidth = scale * np.power(n_samples, -1.0 / (n_dims + 4))
    return float(max(bandwidth, 1e-3))


def fit_kde(data: np.ndarray):
    if data.size == 0:
        return None
    bandwidth = scott_bandwidth(data)
    return KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data)


def jsd_between_kdes(kde_one, kde_two, positions: np.ndarray) -> float:
    if kde_one is None or kde_two is None:
        return 0.0
    p = np.exp(kde_one.score_samples(positions))
    q = np.exp(kde_two.score_samples(positions))
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(jensenshannon(p, q))


def sample_positions(latent_one: np.ndarray, latent_two: np.ndarray, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = np.concatenate((latent_one, latent_two), axis=0)
    if positions.shape[0] == 0:
        latent_dim = 1
        if latent_one.ndim == 2 and latent_one.shape[1] > 0:
            latent_dim = latent_one.shape[1]
        elif latent_two.ndim == 2 and latent_two.shape[1] > 0:
            latent_dim = latent_two.shape[1]
        return np.zeros((num_samples, latent_dim), dtype=np.float32)
    replace = positions.shape[0] < num_samples
    idx = rng.choice(positions.shape[0], size=num_samples, replace=replace)
    return positions[idx]


def compute_jsd_results(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    train_manner: np.ndarray,
    test_manner: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    num_samples: int,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    unique_manners = sorted(np.unique(train_manner))

    matrices = {
        "js_dist_Albayzin_Albayzin": np.zeros((len(unique_manners), len(unique_manners))),
        "js_dist_Albayzin_h_Neurovoz_h": np.zeros((len(unique_manners), len(unique_manners))),
        "js_dist_Albayzin_h_Neurovoz_pd": np.zeros((len(unique_manners), len(unique_manners))),
        "js_dist_Neurovoz_h_Neurovoz_pd": np.zeros((len(unique_manners), len(unique_manners))),
    }

    train_kdes = {
        manner: fit_kde(train_latent[(train_labels == 0) & (train_manner == manner)])
        for manner in unique_manners
    }
    healthy_kdes = {
        manner: fit_kde(test_latent[(test_labels == 0) & (test_manner == manner)])
        for manner in unique_manners
    }
    pd_kdes = {
        manner: fit_kde(test_latent[(test_labels == 1) & (test_manner == manner)])
        for manner in unique_manners
    }

    for i, manner_i in enumerate(unique_manners):
        for j, manner_j in enumerate(unique_manners):
            train_i = train_latent[(train_labels == 0) & (train_manner == manner_i)]
            train_j = train_latent[(train_labels == 0) & (train_manner == manner_j)]
            healthy_j = test_latent[(test_labels == 0) & (test_manner == manner_j)]
            pd_j = test_latent[(test_labels == 1) & (test_manner == manner_j)]
            healthy_i = test_latent[(test_labels == 0) & (test_manner == manner_i)]

            matrices["js_dist_Albayzin_Albayzin"][i, j] = jsd_between_kdes(
                train_kdes[manner_i],
                train_kdes[manner_j],
                sample_positions(train_i, train_j, num_samples, seed + 10 * i + j),
            )
            matrices["js_dist_Albayzin_h_Neurovoz_h"][i, j] = jsd_between_kdes(
                train_kdes[manner_i],
                healthy_kdes[manner_j],
                sample_positions(train_i, healthy_j, num_samples, seed + 100 + 10 * i + j),
            )
            matrices["js_dist_Albayzin_h_Neurovoz_pd"][i, j] = jsd_between_kdes(
                train_kdes[manner_i],
                pd_kdes[manner_j],
                sample_positions(train_i, pd_j, num_samples, seed + 200 + 10 * i + j),
            )
            matrices["js_dist_Neurovoz_h_Neurovoz_pd"][i, j] = jsd_between_kdes(
                healthy_kdes[manner_i],
                pd_kdes[manner_j],
                sample_positions(healthy_i, pd_j, num_samples, seed + 300 + 10 * i + j),
            )

    healthy_diag = np.diag(matrices["js_dist_Albayzin_h_Neurovoz_h"])
    pd_diag = np.diag(matrices["js_dist_Albayzin_h_Neurovoz_pd"])

    diff = pd_diag - healthy_diag
    mape = float(np.mean(np.abs((healthy_diag - pd_diag) / np.clip(healthy_diag, 1e-8, None))) * 100.0)

    summary = {
        "train_vs_train_diag": matrices["js_dist_Albayzin_Albayzin"].diagonal().tolist(),
        "train_vs_healthy_diag": healthy_diag.tolist(),
        "train_vs_pd_diag": pd_diag.tolist(),
        "healthy_vs_pd_diag": matrices["js_dist_Neurovoz_h_Neurovoz_pd"].diagonal().tolist(),
        "pd_minus_healthy_diag": diff.tolist(),
        "train_vs_train_mean": float(np.mean(np.diag(matrices["js_dist_Albayzin_Albayzin"]))),
        "train_vs_healthy_mean": float(np.mean(healthy_diag)),
        "train_vs_pd_mean": float(np.mean(pd_diag)),
        "healthy_vs_pd_mean": float(np.mean(np.diag(matrices["js_dist_Neurovoz_h_Neurovoz_pd"]))),
        "pd_minus_healthy_mean": float(np.mean(diff)),
        "pd_vs_healthy_mape": mape,
    }
    return matrices, summary


def save_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > 0.5 else "black",
                fontsize=11,
            )
    ax.set_xticks(range(len(MANNER_NAMES)))
    ax.set_yticks(range(len(MANNER_NAMES)))
    ax.set_xticklabels([MANNER_NAMES[idx] for idx in sorted(MANNER_NAMES)])
    ax.set_yticklabels([MANNER_NAMES[idx] for idx in sorted(MANNER_NAMES)])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def save_diff_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(matrix[row, col]) > 0.05 else "black",
                fontsize=11,
            )
    ax.set_xticks(range(len(MANNER_NAMES)))
    ax.set_yticks(range(len(MANNER_NAMES)))
    ax.set_xticklabels([MANNER_NAMES[idx] for idx in sorted(MANNER_NAMES)])
    ax.set_yticklabels([MANNER_NAMES[idx] for idx in sorted(MANNER_NAMES)])
    ax.set_title(title)
    ax.set_xlabel("Reference healthy train clusters")
    ax.set_ylabel("Test clusters")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)


def export_jsd_outputs(matrices: Dict[str, np.ndarray], output_dir: Path) -> None:
    titles = {
        "js_dist_Albayzin_Albayzin": "JSD between reference healthy clusters",
        "js_dist_Albayzin_h_Neurovoz_h": "JSD between reference healthy clusters and NeuroVoz healthy test clusters",
        "js_dist_Albayzin_h_Neurovoz_pd": "JSD between reference healthy clusters and NeuroVoz PD test clusters",
        "js_dist_Neurovoz_h_Neurovoz_pd": "JSD between NeuroVoz healthy and PD test clusters",
    }

    for name, matrix in matrices.items():
        pd.DataFrame(matrix).to_csv(output_dir / f"{name}.csv", index=False)
        save_heatmap(matrix, titles[name], output_dir / f"{name}.png")

    diff = matrices["js_dist_Albayzin_h_Neurovoz_pd"] - matrices["js_dist_Albayzin_h_Neurovoz_h"]
    pd.DataFrame(diff).to_csv(output_dir / "diff_healthy_park_test.csv", index=False)
    save_diff_heatmap(
        diff,
        "Difference between train-vs-PD and train-vs-healthy JSD",
        output_dir / "diff_healthy_park_test.png",
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args)
    output_dir = make_output_dir(args)
    hyperparams = make_hyperparams(args, output_dir)
    artifact_paths = fold_artifact_paths(args, hyperparams)
    ckpt_path = checkpoint_path(output_dir, args)

    ensure_dir(output_dir)
    log_path = output_dir / "log.txt"

    set_seed(args.seed)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_handle:
        sys.stdout = Tee(original_stdout, log_handle)
        try:
            print(f"Device: {device}")
            print(f"Output directory: {output_dir}")
            print(f"Requested fold: {args.fold}")
            print(f"Requested number of Gaussians: {args.n_gaussians}")

            save_json(output_dir / "config.json", {"args": vars(args), "hyperparams": hyperparams})

            maybe_build_folds(args, hyperparams, artifact_paths)
            train_loader, val_loader, test_loader, test_data = load_fold_data(artifact_paths)

            model = build_model(train_loader, hyperparams, device)
            train_if_requested(model, train_loader, val_loader, hyperparams, output_dir)
            load_checkpoint(model, ckpt_path)

            reconstruction_metrics = evaluate_reconstruction(model, test_loader, test_data, output_dir)
            save_json(output_dir / "reconstruction_metrics.json", reconstruction_metrics)
            print("Reconstruction metrics:", json.dumps(reconstruction_metrics, indent=2))

            df_train = dataframe_from_loader(train_loader)
            df_test = dataframe_from_loader(test_loader)

            train_latent_raw = extract_latent_mu(model, df_train, args.batch_size)
            test_latent_raw = extract_latent_mu(model, df_test, args.batch_size)

            train_latent, train_manner, train_labels = filter_latent_by_manner(
                train_latent_raw, df_train, supervised=False
            )
            test_latent, test_manner, test_labels = filter_latent_by_manner(
                test_latent_raw, df_test, supervised=False
            )

            np.savez_compressed(
                output_dir / "latent_space_filtered.npz",
                train_latent=train_latent,
                train_manner=train_manner,
                train_labels=train_labels,
                test_latent=test_latent,
                test_manner=test_manner,
                test_labels=test_labels,
            )

            projection_df = compute_tsne_projection(
                train_latent=train_latent,
                train_manner=train_manner,
                train_labels=train_labels,
                test_latent=test_latent,
                test_manner=test_manner,
                test_labels=test_labels,
                max_per_group=args.tsne_max_per_group,
                perplexity=args.tsne_perplexity,
                seed=args.seed,
            )
            projection_df.to_csv(output_dir / "tsne_projection.csv", index=False)

            reference_df = projection_df[projection_df["split"] == "train_reference"].copy()
            healthy_df = projection_df[projection_df["split"] == "test_healthy"].copy()
            pd_df = projection_df[projection_df["split"] == "test_pd"].copy()

            if not reference_df.empty:
                plot_projection_by_manner(
                    reference_df,
                    "Reference healthy train clusters",
                    output_dir / "tsne_train_reference_by_manner.png",
                )
            if not healthy_df.empty:
                plot_projection_by_manner(
                    healthy_df,
                    "NeuroVoz healthy test latent space",
                    output_dir / "tsne_test_healthy_by_manner.png",
                )
            if not pd_df.empty:
                plot_projection_by_manner(
                    pd_df,
                    "NeuroVoz PD test latent space",
                    output_dir / "tsne_test_pd_by_manner.png",
                )
            if not reference_df.empty and not healthy_df.empty:
                plot_reference_overlay(
                    reference_df,
                    healthy_df,
                    "Reference healthy clusters vs NeuroVoz healthy test samples",
                    output_dir / "tsne_reference_vs_test_healthy.png",
                )
            if not reference_df.empty and not pd_df.empty:
                plot_reference_overlay(
                    reference_df,
                    pd_df,
                    "Reference healthy clusters vs NeuroVoz PD test samples",
                    output_dir / "tsne_reference_vs_test_pd.png",
                )

            jsd_matrices, jsd_summary = compute_jsd_results(
                train_latent=train_latent,
                test_latent=test_latent,
                train_manner=train_manner,
                test_manner=test_manner,
                train_labels=train_labels,
                test_labels=test_labels,
                num_samples=args.jsd_samples,
                seed=args.seed,
            )
            export_jsd_outputs(jsd_matrices, output_dir)
            save_json(output_dir / "jsd_summary.json", jsd_summary)
            print("JSD summary:", json.dumps(jsd_summary, indent=2))
            print("Finished reviewer baseline run.")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
