import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import wandb


def StratifiedGroupKFold_local(class_labels, groups, n_splits=2, shuffle=True):
    unique_groups = np.unique(groups)
    train_idx = []
    test_idx = []
    class_labels_g = []

    for i in range(len(unique_groups)):
        indx = np.argwhere(groups == unique_groups[i])
        if len(indx) > 1:
            indx = indx[0]
        class_labels_g.append(class_labels[indx])
    class_labels_g = np.stack(class_labels_g).ravel()
    train_idx_p, _ = next(
        StratifiedKFold(n_splits=n_splits, shuffle=shuffle).split(
            np.zeros(len(class_labels_g)), class_labels_g
        )
    )

    for i in range(len(class_labels_g)):
        indx = np.argwhere(groups == unique_groups[i])
        if i in train_idx_p:
            train_idx.append(indx)
        else:
            test_idx.append(indx)

    train_idx = np.concatenate(train_idx).ravel().tolist()
    test_idx = np.concatenate(test_idx).ravel().tolist()

    return train_idx, test_idx


def plot_latent_space(model, data, fold, wandb_flag, name="default"):
    from matplotlib import pyplot as plt

    # Generate mu and sigma in training
    model.eval()
    with torch.no_grad():
        latent_mu, latent_sigma = model.encoder(
            torch.Tensor(np.vstack(data["plps"])).to(model.device)
        )

    # Check latent_mu shape, if greater than 2 do a t-SNE
    if latent_mu.shape[1] > 2:
        from sklearn.manifold import TSNE

        latent_mu = TSNE(n_components=2).fit_transform(latent_mu.detach().cpu().numpy())
        xlabel = "t-SNE dim 1"
        ylabel = "t-SNE dim 2"
    else:
        latent_mu = latent_mu.detach().cpu().numpy()
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    plt.figure(figsize=(10, 10))

    # Scatter plot
    scatter = plt.scatter(
        latent_mu[:, 0],
        latent_mu[:, 1],
        c=data["label"].values,
        cmap="viridis",
    )

    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Latent space in " + str(name) + " for fold {fold}")

    # Create custom legend
    classes = ["Healthy", "PD"]
    class_labels = np.unique(data["label"].values)
    class_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=scatter.cmap(scatter.norm(cls)),
            markersize=10,
        )
        for cls in class_labels
    ]
    plt.legend(class_handles, classes)
    plt.savefig(f"local_results/latent_space_{fold}_{name}.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space": plt})
    plt.show()
    plt.close()


def plot_latent_space_vowels(
    model, data, fold, wandb_flag, name="default", supervised=False, vqvae=False
):
    from matplotlib import pyplot as plt

    # Generate mu and sigma in training
    model.eval()
    with torch.no_grad():
        if not vqvae:
            latent_mu, latent_sigma = model.encoder(
                torch.Tensor(np.vstack(data["plps"])).to(model.device)
            )
        else:
            latent_mu = model.encoder(
                torch.Tensor(np.vstack(data["plps"])).to(model.device)
            )
            latent_code, _ = model.vq(latent_mu)


    # Check latent_mu shape, if greater than 2 do a t-SNE
    if latent_mu.shape[1] > 2:
        from sklearn.manifold import TSNE

        latent_mu = TSNE(n_components=2).fit_transform(latent_mu.detach().cpu().numpy())
        xlabel = "t-SNE dim 1"
        ylabel = "t-SNE dim 2"
        if vqvae:
            latent_code = TSNE(n_components=2).fit_transform(latent_code.detach().cpu().numpy())
    else:
        latent_mu = latent_mu.detach().cpu().numpy()
        if vqvae: latent_code = latent_code.detach().cpu().numpy()
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    labels = data["label"].values
    vowels = data["vowel"].values

    # Make two different plots, one for vowels and one for labels

    # TODO: repeat this code for VQVAE latent_code

    # PLot latent space by vowels
    plt.figure(figsize=(10, 10))
    idxA = np.argwhere(vowels == 0).ravel()
    idxE = np.argwhere(vowels == 1).ravel()
    idxI = np.argwhere(vowels == 2).ravel()
    idxO = np.argwhere(vowels == 3).ravel()
    idxU = np.argwhere(vowels == 4).ravel()
    plt.scatter(latent_mu[idxA, 0], latent_mu[idxA, 1], c="r", label="a", alpha=0.5)
    plt.scatter(latent_mu[idxE, 0], latent_mu[idxE, 1], c="b", label="e", alpha=0.5)
    plt.scatter(latent_mu[idxI, 0], latent_mu[idxI, 1], c="g", label="i", alpha=0.5)
    plt.scatter(latent_mu[idxO, 0], latent_mu[idxO, 1], c="y", label="o", alpha=0.5)
    plt.scatter(latent_mu[idxU, 0], latent_mu[idxU, 1], c="m", label="u", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Latent space in " + str(name) + " for fold {fold} by vowels")
    plt.legend()
    if supervised:
        savepath = "local_results/vae_supervised/"
    elif vqvae:
        savepath = "local_results/vqvae/"
    else:
        savepath = "local_results/vae_unsupervised/"
    plt.savefig(f"{savepath}latent_space_vowels_{fold}_{name}.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_vowels": plt})
    plt.show()
    plt.close()

    # Plot latent space by labels
    plt.figure(figsize=(10, 10))
    idxH = np.argwhere(labels == 0).ravel()
    idxPD = np.argwhere(labels == 1).ravel()
    plt.scatter(latent_mu[idxH, 0], latent_mu[idxH, 1], c="r", label="Healthy", alpha=0.5)
    plt.scatter(latent_mu[idxPD, 0], latent_mu[idxPD, 1], c="b", label="PD", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Latent space in " + str(name) + " for fold {fold} by labels")
    # Create a custom legend where 0="healthy", 1="PD"
    plt.legend()
    save_path = savepath + f"latent_space_labels_{fold}_{name}.png"
    plt.savefig(
        save_path,
    )
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_labels": plt})
    plt.show()
    plt.close()
