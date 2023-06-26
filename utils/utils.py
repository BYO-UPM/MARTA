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

    plt.figure(figsize=(10, 10))

    # Scatter plot
    scatter = plt.scatter(
        latent_mu[:, 0].detach().cpu().numpy(),
        latent_mu[:, 1].detach().cpu().numpy(),
        c=data["label"].values,
        cmap="viridis",
    )

    # Add labels and title
    plt.xlabel("Latent dim 1")
    plt.ylabel("Latent dim 2")
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
    plt.savefig(f"local_results/latent_space_{fold}.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space": plt})
    plt.show()
    plt.close()
