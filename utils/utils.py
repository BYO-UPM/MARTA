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


def plot_latent_space_vowels(model, data, fold, wandb_flag, name="default"):
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

    labels = data["label"].values
    vowels = data["vowel"].values

    markers_vowels = ["^", "o", "s", "P", "X"]

    plt.figure(figsize=(10, 10))
    for v in data["vowel"].unique():
        alpha = 0.2 * (v + 1)
        idx_vowel = np.where(data["vowel"] == v)
        idx_label0 = np.where(data["label"] == 0)
        idx_label1 = np.where(data["label"] == 1)
        idx0 = np.intersect1d(idx_vowel, idx_label0)
        idx1 = np.intersect1d(idx_vowel, idx_label1)

        plt.scatter(
            latent_mu[idx0, 0],
            latent_mu[idx0, 1],
            c="r",
            alpha=alpha,
            marker=markers_vowels[v],
            label="Vowel " + str(v) + " PD",
        )
        plt.scatter(
            latent_mu[idx1, 0],
            latent_mu[idx1, 1],
            c="b",
            alpha=alpha,
            marker=markers_vowels[v],
            label="Vowel " + str(v) + " PD",
        )
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Latent space in " + str(name) + " for fold " + str(fold))
    plt.savefig(f"local_results/latent_space_{fold}_{name}.png")
