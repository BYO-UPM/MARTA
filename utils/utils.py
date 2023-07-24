import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import pandas as pd
import wandb
from matplotlib import pyplot as plt


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
    plt.savefig(f"local_results/plps/latent_space_{fold}_{name}.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space": plt})
    plt.show()
    plt.close()


def plot_latent_space_vowels(
    model, data, fold, wandb_flag, name="default", supervised=False, vqvae=False
):
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
            latent_code, vq_loss, enc_idx = model.vq(latent_mu)

    # Check latent_mu shape, if greater than 2 do a t-SNE
    if latent_mu.shape[1] > 2:
        from sklearn.manifold import TSNE

        latent_mu = TSNE(n_components=2).fit_transform(latent_mu.detach().cpu().numpy())
        xlabel = "t-SNE dim 1"
        ylabel = "t-SNE dim 2"
        if vqvae:
            latent_code = TSNE(n_components=2).fit_transform(
                latent_code.detach().cpu().numpy()
            )
    else:
        latent_mu = latent_mu.detach().cpu().numpy()
        if vqvae:
            latent_code = latent_code.detach().cpu().numpy()
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    labels = data["label"].values
    vowels = data["vowel"].values

    # Make two different plots, one for vowels and one for labels

    if vqvae:
        plt.figure(figsize=(10, 10))
        unique_codes = np.unique(enc_idx.cpu())
        for i in range(len(unique_codes)):
            idx = np.argwhere(enc_idx.cpu() == unique_codes[i]).ravel()
            plt.scatter(
                latent_mu[idx, 0], latent_mu[idx, 1], label="Code " + str(i), alpha=0.5
            )
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Latent space by CODE in " + str(name) + " for fold " + str(fold))
        plt.savefig(f"local_results/plps/vqvae/latent_space_code_{fold}_{name}.png")
        if wandb_flag:
            wandb.log({str(name) + "/latent_space_code": plt})
        plt.show()
        plt.close()

    # PLot latent space by vowels
    fig, ax = plt.subplots(figsize=(20, 20))
    unique_vowels = np.unique(vowels)
    vowel_dict = {0: "a", 1: "e", 2: "i", 3: "o", 4: "u"}
    colors = ["red", "blue", "green", "orange", "purple"]
    for i in range(len(unique_vowels)):
        idx = np.argwhere(vowels == unique_vowels[i]).ravel()
        label = "Vowel " + vowel_dict[unique_vowels[i]]
        # Get for each vowel, the label
        idxH = np.argwhere(labels[idx] == 0).ravel()
        idxPD = np.argwhere(labels[idx] == 1).ravel()
        ax.scatter(
            latent_mu[idxH, 0],
            latent_mu[idxH, 1],
            label=label,
            marker="$H$",
            c=colors[i],
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxPD, 0],
            latent_mu[idxPD, 1],
            label=label,
            marker="$P$",
            alpha=0.5,
            c=colors[i],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Latent space in " + str(name) + " for fold {fold} by vowels")
    ax.legend()
    if supervised:
        savepath = "local_results/plps/vae_supervised/"
    if vqvae:
        savepath = "local_results/plps/vqvae/"
    if not supervised and not vqvae:
        savepath = "local_results/plps/vae_unsupervised/"

    fig.savefig(savepath + f"latent_space_vowels_{fold}_{name}.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_vowels": wandb.Image(fig)})
    plt.close(fig)

    # PLot latent space by vowels but inversing the legend
    fig, ax = plt.subplots(figsize=(20, 20))
    unique_vowels = np.unique(vowels)
    vowel_dict = {0: "a", 1: "e", 2: "i", 3: "o", 4: "u"}
    colors = ["red", "blue"]
    for i in range(len(np.unique(labels))):
        idx = np.argwhere(labels == i).ravel()
        if i == 0:
            label = "Healty"
        else:
            label = "PD"
        # For each label, plot a scatter for each vowel
        idxA = np.argwhere(vowels[idx] == 0).ravel()
        idxE = np.argwhere(vowels[idx] == 1).ravel()
        idxI = np.argwhere(vowels[idx] == 2).ravel()
        idxO = np.argwhere(vowels[idx] == 3).ravel()
        idxU = np.argwhere(vowels[idx] == 4).ravel()
        ax.scatter(
            latent_mu[idxA, 0],
            latent_mu[idxA, 1],
            label=label,
            marker="$A$",
            s=1000,
            c=colors[i],
        )
        ax.scatter(
            latent_mu[idxE, 0],
            latent_mu[idxE, 1],
            label=label,
            marker="$E$",
            s=800,
            c=colors[i],
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxI, 0],
            latent_mu[idxI, 1],
            label=label,
            marker="$I$",
            c=colors[i],
            s=600,
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxO, 0],
            latent_mu[idxO, 1],
            label=label,
            marker="$O$",
            s=400,
            c=colors[i],
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxU, 0],
            latent_mu[idxU, 1],
            label=label,
            marker="$U$",
            s=200,
            c=colors[i],
            alpha=0.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Latent space in " + str(name) + " for fold {fold} by vowels")
    ax.legend()
    if supervised:
        savepath = "local_results/plps/vae_supervised/"
    if vqvae:
        savepath = "local_results/plps/vqvae/"
    if not supervised and not vqvae:
        savepath = "local_results/plps/vae_unsupervised/"

    fig.savefig(savepath + f"latent_space_vowels_{fold}_{name}_inverse.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_vowels_inverse": wandb.Image(fig)})
    plt.close(fig)

    # Plot latent space by labels
    fig, ax = plt.subplots(figsize=(20, 20))
    idxH = np.argwhere(labels == 0).ravel()
    idxPD = np.argwhere(labels == 1).ravel()
    ax.scatter(latent_mu[idxH, 0], latent_mu[idxH, 1], label="Healthy", alpha=0.5)
    ax.scatter(latent_mu[idxPD, 0], latent_mu[idxPD, 1], label="PD", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Latent space in " + str(name) + " for fold {fold} by labels")
    # Create a custom legend where 0="healthy", 1="PD"
    ax.legend()
    save_path = savepath + f"latent_space_labels_{fold}_{name}.png"
    fig.savefig(
        save_path,
    )
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_labels": wandb.Image(fig)})
    plt.close(fig)


def calculate_distances(model, data, fold, wandb_flag, name="default", vqvae=False):
    # Import KDE
    from scipy.stats import gaussian_kde

    # Import jensen-shannon distance
    from scipy.spatial.distance import jensenshannon

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
            latent_code, vq_loss, enc_idx = model.vq(latent_mu)

    latent_mu = latent_mu.detach().cpu().numpy()
    if vqvae:
        latent_code = latent_code.detach().cpu().numpy()

    labels = data["label"].values
    vowels = data["vowel"].values

    idxH = np.argwhere(labels == 0).ravel()
    idxPD = np.argwhere(labels == 1).ravel()

    idxAH = np.argwhere(vowels[idxH] == 0).ravel()
    idxEH = np.argwhere(vowels[idxH] == 1).ravel()
    idxIH = np.argwhere(vowels[idxH] == 2).ravel()
    idxOH = np.argwhere(vowels[idxH] == 3).ravel()
    idxUH = np.argwhere(vowels[idxH] == 4).ravel()

    idxAPD = np.argwhere(vowels[idxPD] == 0).ravel()
    idxEPD = np.argwhere(vowels[idxPD] == 1).ravel()
    idxIPD = np.argwhere(vowels[idxPD] == 2).ravel()
    idxOPD = np.argwhere(vowels[idxPD] == 3).ravel()
    idxUPD = np.argwhere(vowels[idxPD] == 4).ravel()

    all_idx = [
        idxAH,
        idxEH,
        idxIH,
        idxOH,
        idxUH,
        idxAPD,
        idxEPD,
        idxIPD,
        idxOPD,
        idxUPD,
    ]

    distances = np.zeros((10, 10))
    for i in range(len(all_idx)):
        kde1 = gaussian_kde(latent_mu[all_idx[i]].T)
        for j in range(len(all_idx)):
            kde2 = gaussian_kde(latent_mu[all_idx[j]].T)

            # Sample from a uniform distribution of the limits of the latent_mu space
            x = np.linspace(
                np.min(latent_mu[:, 0]), np.max(latent_mu[:, 0]), 1000, endpoint=False
            )
            y = np.linspace(
                np.min(latent_mu[:, 1]), np.max(latent_mu[:, 1]), 1000, endpoint=False
            )
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])

            # Calculate logprob of each kde
            logprob1 = kde1.logpdf(positions)
            logprob2 = kde2.logpdf(positions)

            # Calculate the jensen-shannon distance
            distances[i, j] = jensenshannon(logprob1, logprob2)

    # Plot the distances as a heatmap using seaborn
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(distances, annot=True, ax=ax)
    ax.set_title(f"Jensen-Shannon distance between all vowels in fold {fold}")
    ax.set_xticklabels(
        ["A-H", "E-H", "I-H", "O-H", "U-H", "A-PD", "E-PD", "I-PD", "O-PD", "U-PD"]
    )
    ax.set_yticklabels(
        ["A-H", "E-H", "I-H", "O-H", "U-H", "A-PD", "E-PD", "I-PD", "O-PD", "U-PD"]
    )
    ax.set_xlabel("Vowels (Healthy / PD)")
    ax.set_ylabel("Vowels (Healthy / PD)")
    save_path = "local_results/plps/vae_supervised/" + f"js_dist_{fold}_{name}.png"
    fig.savefig(save_path)

    if wandb_flag:
        wandb.log({str(name) + "/distances": wandb.Image(fig)})

    plt.close()

    dist2 = np.zeros((10, 10))
    for i in range(len(all_idx)):
        kde1 = gaussian_kde(latent_mu[all_idx[i]].T)
        for j in range(len(all_idx)):
            kde2 = gaussian_kde(latent_mu[all_idx[j]].T)

            x1 = latent_mu[all_idx[i]].T
            x2 = latent_mu[all_idx[j]].T

            # Distance between kde1 and kde2: logprob(x2 | kde1) - logprob(x2 | kde2)
            d1 = kde1.logpdf(x2) - kde2.logpdf(x2)

            # Check if x2 | kde2 is almost zero
            print(np.sum(kde2.logpdf(x2)))

            # Distance between kde2 and kde1: logprob(x1 | kde2) - logprob(x1 | kde1)
            d2 = kde2.logpdf(x1) - kde1.logpdf(x1)

            print(np.sum(kde1.logpdf(x1)))

            # symmetric distance
            dist2[i, j] = np.mean([d1, d2])

    # Plot the distances as a heatmap using seaborn
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(dist2, annot=True, ax=ax)
    ax.set_title(f"Symmetric distance between all vowels in fold {fold}")
    ax.set_xticklabels(
        ["A-H", "E-H", "I-H", "O-H", "U-H", "A-PD", "E-PD", "I-PD", "O-PD", "U-PD"]
    )
    ax.set_yticklabels(
        ["A-H", "E-H", "I-H", "O-H", "U-H", "A-PD", "E-PD", "I-PD", "O-PD", "U-PD"]
    )

    ax.set_xlabel("Vowels (Healthy / PD)")
    ax.set_ylabel("Vowels (Healthy / PD)")
    save_path = "local_results/plps/vae_supervised/" + f"sym_dist_{fold}_{name}.png"
    fig.savefig(save_path)

    if wandb_flag:
        wandb.log({str(name) + "/distances": wandb.Image(fig)})

    plt.close()
