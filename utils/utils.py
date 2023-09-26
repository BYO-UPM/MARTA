import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from scipy import linalg
import matplotlib as mpl


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


def plot_latent_space(model, data, fold, wandb_flag, name="default", supervised=False):
    # Generate mu and sigma in training
    model.eval()
    with torch.no_grad():
        data_input = (
            torch.Tensor(np.vstack(data["spectrogram"])).to(model.device).unsqueeze(1)
        )
        z, qy_logits, qy, latent_mu, qz_logvar, x_hat, x_hat_unflatten = model.infere(
            data_input
        )
    labels = torch.Tensor(data["label"].values)
    manner = torch.Tensor(np.array([np.array(x) for x in data["manner"]]))

    manner_labels = manner.reshape(-1)

    # Subsample 1000 points to the plotting and move them to numpy and cpu
    idx = np.random.choice(len(labels), 1000)
    labels = labels[idx].cpu().numpy()
    manner_labels = manner_labels[idx].cpu().numpy()
    latent_mu = latent_mu[idx].cpu().numpy()

    # p(y) = Cat(10)
    py = torch.eye(model.k).to(model.device)
    # Sample from generative model
    z_mu, z_logvar = torch.chunk(model.generative_pz_y(py), 2, dim=1)
    z_var = torch.nn.functional.softplus(z_logvar)

    # Check latent_mu shape, if greater than 2 do a t-SNE
    if latent_mu.shape[1] > 2:
        from sklearn.manifold import TSNE
        inference_mu_shape = latent_mu.shape
        generative_mu_shape = z_mu.shape

        # Convert all to 2D

        # Concat all info
        all = np.concatenate((latent_mu, z_mu.cpu().detach().numpy(), z_var.cpu().detach().numpy()), axis=0)

        all_2D = TSNE(n_components=2).fit_transform(all)
        

        # Separate info
        latent_mu = all_2D[:inference_mu_shape[0]]
        z_mu = all_2D[inference_mu_shape[0]:inference_mu_shape[0]+generative_mu_shape[0]]
        z_var = all_2D[inference_mu_shape[0]+generative_mu_shape[0]:]

        xlabel = "t-SNE dim 1"
        ylabel = "t-SNE dim 2"

    else:
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    fig, ax = plt.subplots(figsize=(20, 20))

    # Scatter ax

    # Divide the scatter in two scatters: one for label=0 and one for label=1. The difference will be the alpha
    idxH = np.argwhere(labels == 0).ravel()
    idxPD = np.argwhere(labels == 1).ravel()
    scatter1= ax.scatter(latent_mu[idxH, 0], latent_mu[idxH, 1], c = manner_labels[idxH], alpha=0.2, cmap="Set1")
    scatter2= ax.scatter(latent_mu[idxPD, 0], latent_mu[idxPD, 1], c = manner_labels[idxPD], alpha=1, cmap="Set1")

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Latent space in " + str(name) + " for fold {fold}")

    # Create custom legend
    classes = [
        "Plosives",
        "Plosives voiced",
        "Nasals",
        "Fricatives",
        "Vowels",
        "Affricates",
        "Silence",
    ]
    class_labels = np.unique(manner_labels)
    class_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=scatter1.cmap(scatter1.norm(cls)),
            markersize=10,
        )
        for cls in class_labels
    ]
    ax.legend(
        class_handles,
        classes,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        title="Classes",
    )
    fig.savefig(
        f"local_results/spectrograms/manner_gmvae/latent_space_{fold}_{name}.png",
    )

    if wandb_flag:
        wandb.log({str(name) + "/latent_space": wandb.Image(fig)})

    

    for i in range(model.k):
        mu = z_mu[i]
        var = z_var[i]
        cov = np.diag(var)

        x = np.linspace(
            np.min(latent_mu[:, 0]),
            np.max(latent_mu[:, 0]),
        )
        y = np.linspace(
            np.min(latent_mu[:, 1]),
            np.max(latent_mu[:, 1]),
        )
        X, Y = np.meshgrid(x, y)

        v, w = linalg.eigh(cov)
        v1 = 2.0 * 1 * np.sqrt(v)  # 1 std
        v2 = 2.0 * 2 * np.sqrt(v)  # 2 std
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            mu, v1[0], v1[1], angle=180.0 + angle, facecolor="none", edgecolor="green"
        )
        ell2 = mpl.patches.Ellipse(
            mu, v2[0], v2[1], angle=180.0 + angle, facecolor="none", edgecolor="orange"
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ell2.set_clip_box(ax.bbox)
        ell2.set_alpha(0.5)
        ax.add_artist(ell2)
        # Use star as a marker
        ax.scatter(mu[0], mu[1], label="Gaussian " + str(i), marker="*", s=100)
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(f"Latent space with Gaussians distributions")
    ax.legend()
    save_path = f"local_results/spectrograms/manner_gmvae/gaussians_generative_and_test_vowels_{fold}_{name}.png"
    fig.savefig(
        save_path,
    )
    if wandb_flag:
        wandb.log(
            {str(name) + "/gaussians_generative_and_test_vowels_": wandb.Image(fig)}
        )

    plt.close()


def plot_latent_space_vowels(
    model,
    data,
    fold,
    wandb_flag,
    name="default",
    supervised=False,
    vqvae=False,
    gmvae=False,
    audio_features="plps",
):
    # Generate mu and sigma in training
    model.eval()
    with torch.no_grad():
        if vqvae:
            latent_mu = model.encoder(
                torch.Tensor(np.vstack(data[audio_features])).to(model.device)
            )
            latent_code, vq_loss, enc_idx = model.vq(latent_mu)
        elif gmvae:
            if audio_features == "spectrogram":
                z, _, qy, latent_mu, latent_logvar, _ = model.infere(
                    torch.Tensor(np.expand_dims(np.vstack(data[audio_features]), 1)).to(
                        model.device
                    )
                )
            else:
                _, _, _, latent_mu, latent_logvar, _ = model.infere(
                    torch.Tensor(np.vstack(data[audio_features])).to(model.device)
                )
        else:
            latent_mu, latent_logvar = model.encoder(
                torch.Tensor(np.vstack(data[audio_features])).to(model.device)
            )

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

    if gmvae:
        savepath = "local_results/spectrograms/gmvae/"
    if supervised:
        savepath = "local_results/plps/vae_supervised/"
    if vqvae:
        savepath = "local_results/plps/vqvae/"

    if gmvae:
        plot_gaussians_generative(
            model,
            latent_mu,
            wandb_flag,
            name,
            fold,
            savepath=savepath,
        )

    # PLot latent space by vowels
    plot_latent_space_by_vowels(
        labels,
        vowels,
        latent_mu,
        fold,
        wandb_flag,
        name,
        xlabel,
        ylabel,
        savepath,
    )

    # PLot latent space by vowels but inversing the legend
    plot_latent_space_vowels_inverse(
        labels,
        vowels,
        latent_mu,
        fold,
        wandb_flag,
        name,
        xlabel,
        ylabel,
        savepath,
    )

    # Plot latent space by labels
    plot_latent_space_by_labels(
        labels, latent_mu, fold, wandb_flag, name, xlabel, ylabel, savepath
    )

    # Plot everything
    plot_gaussians_generative_over_vowels(
        labels,
        vowels,
        latent_mu,
        fold,
        wandb_flag,
        name,
        xlabel,
        ylabel,
        savepath,
        model,
    )


def plot_latent_space_by_labels(
    labels, latent_mu, fold, wandb_flag, name, xlabel, ylabel, savepath
):
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


def plot_latent_space_vowels_3D(
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
    if latent_mu.shape[1] > 3:
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
        zlabel = "Latent dim 3"

    labels = data["label"].values
    vowels = data["vowel"].values

    # Make two different plots, one for vowels and one for labels

    if vqvae:
        plt.figure(figsize=(10, 10))
        unique_codes = np.unique(enc_idx.cpu())
        for i in range(len(unique_codes)):
            idx = np.argwhere(enc_idx.cpu() == unique_codes[i]).ravel()
            plt.scatter(
                latent_mu[idx, 0],
                latent_mu[idx, 1],
                latent_mu[idx, 2],
                label="Code " + str(i),
                alpha=0.5,
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
            latent_mu[idxH, 2],
            label=label,
            marker="$H$",
            c=colors[i],
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxPD, 0],
            latent_mu[idxPD, 1],
            latent_mu[idxPD, 2],
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
            latent_mu[idxA, 2],
            label=label,
            marker="$A$",
            s=1000,
            c=colors[i],
        )
        ax.scatter(
            latent_mu[idxE, 0],
            latent_mu[idxE, 1],
            latent_mu[idxE, 2],
            label=label,
            marker="$E$",
            s=800,
            c=colors[i],
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxI, 0],
            latent_mu[idxI, 1],
            latent_mu[idxI, 2],
            label=label,
            marker="$I$",
            c=colors[i],
            s=600,
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxO, 0],
            latent_mu[idxO, 1],
            latent_mu[idxO, 2],
            label=label,
            marker="$O$",
            s=400,
            c=colors[i],
            alpha=0.5,
        )
        ax.scatter(
            latent_mu[idxU, 0],
            latent_mu[idxU, 1],
            latent_mu[idxU, 2],
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
    ax.scatter(
        latent_mu[idxH, 0],
        latent_mu[idxH, 1],
        latent_mu[idxH, 2],
        label="Healthy",
        alpha=0.5,
    )
    ax.scatter(
        latent_mu[idxPD, 0],
        latent_mu[idxPD, 1],
        latent_mu[idxPD, 2],
        label="PD",
        alpha=0.5,
    )
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


def calculate_distances(
    model,
    data,
    fold,
    wandb_flag,
    name="default",
    vqvae=False,
    gmvae=False,
    audio_features="plps",
):
    print("Calculating distances...")
    # Import KDE
    from scipy.stats import gaussian_kde

    # Import jensen-shannon distance
    from scipy.spatial.distance import jensenshannon

    model.eval()
    with torch.no_grad():
        if vqvae:
            latent_mu = model.encoder(
                torch.Tensor(np.vstack(data[audio_features])).to(model.device)
            )
            latent_code, vq_loss, enc_idx = model.vq(latent_mu)
        elif gmvae:
            if audio_features == "spectrogram":
                _, _, _, latent_mu, latent_sigma, _ = model.infere(
                    torch.Tensor(np.expand_dims(np.vstack(data[audio_features]), 1)).to(
                        model.device
                    )
                )
            else:
                _, _, _, latent_mu, latent_sigma, _ = model.infere(
                    torch.Tensor(np.vstack(data[audio_features])).to(model.device)
                )
        else:
            latent_mu, latent_sigma = model.encoder(
                torch.Tensor(np.vstack(data[audio_features])).to(model.device)
            )

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
    print("Calculating jensen-shannon distances sampling uniformly from the space...")
    for i in range(len(all_idx)):
        kde1 = gaussian_kde(latent_mu[all_idx[i]].T)
        for j in range(len(all_idx)):
            kde2 = gaussian_kde(latent_mu[all_idx[j]].T)

            # Sample from a uniform distribution of the limits of the latent_mu space which can be N-Dimensional
            positions = np.random.uniform(
                low=latent_mu.min(),
                high=latent_mu.max(),
                size=(1000, latent_mu.shape[1]),
            )

            positions = positions.T

            # Calculate logprob of each kde
            logprob1 = kde1.logpdf(positions)
            logprob2 = kde2.logpdf(positions)

            # Calculate the jensen-shannon distance
            distances[i, j] = jensenshannon(logprob1, logprob2)

    # Plot the distances as a heatmap using seaborn
    print("Plotting distances...")
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
        wandb.log({str(name) + "/js_distances": wandb.Image(fig)})

    plt.close()
    # print("Calculating symmetric distances...")
    # dist2 = np.zeros((10, 10))
    # for i in range(len(all_idx)):
    #     kde1 = gaussian_kde(latent_mu[all_idx[i]].T)
    #     for j in range(len(all_idx)):
    #         kde2 = gaussian_kde(latent_mu[all_idx[j]].T)

    #         x1 = latent_mu[all_idx[i]].T
    #         x2 = latent_mu[all_idx[j]].T

    #         # Distance between kde1 and kde2: logprob(x2 | kde1) - logprob(x2 | kde2)
    #         d1 = np.mean(kde1.logpdf(x2) - kde2.logpdf(x2))

    #         # Distance between kde2 and kde1: logprob(x1 | kde2) - logprob(x1 | kde1)
    #         d2 = np.mean(kde2.logpdf(x1) - kde1.logpdf(x1))

    #         # symmetric distance
    #         dist2[i, j] = np.abs((d1 + d2)) / 2

    # # Plot the distances as a heatmap using seaborn
    # print("Plotting distances...")
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(dist2, annot=True, ax=ax)
    # ax.set_title(f"Symmetric distance between all vowels in fold {fold}")
    # ax.set_xticklabels(
    #     ["A-H", "E-H", "I-H", "O-H", "U-H", "A-PD", "E-PD", "I-PD", "O-PD", "U-PD"]
    # )
    # ax.set_yticklabels(
    #     ["A-H", "E-H", "I-H", "O-H", "U-H", "A-PD", "E-PD", "I-PD", "O-PD", "U-PD"]
    # )

    # ax.set_xlabel("Vowels (Healthy / PD)")
    # ax.set_ylabel("Vowels (Healthy / PD)")
    # save_path = "local_results/plps/vae_supervised/" + f"sym_dist_{fold}_{name}.png"
    # fig.savefig(save_path)

    # if wandb_flag:
    #     wandb.log({str(name) + "/sym_distances": wandb.Image(fig)})

    # plt.close()


def get_formants(audio, sr, n_formants=2):
    import librosa
    from scipy.signal import lfilter, hamming

    y = audio

    # Hamming window
    y = y * np.hamming(len(y))

    # Pre-emphasis to enhance high-frequency content
    y = lfilter([1], [1.0, 0.63], y)

    # LPC analysis to estimate the formants
    order = int(2 + sr / 1000)
    a_coeffs = librosa.lpc(y, order=order)

    # Compute the roots of the LPC coefficients to get formant frequencies
    roots = np.roots(a_coeffs)
    roots = roots[roots.imag >= 0.01]  # Discard complex roots
    formant_frequencies = np.sort(
        np.arctan2(roots.imag, roots.real) * (sr / (2 * np.pi))
    )

    return formant_frequencies[:2]


def plot_formants(dataset):
    dataset.data["formants"] = dataset.data.apply(
        lambda x: get_formants(x["norm_signal"], x["sr"]), axis=1
    )

    # Formants is a list of 2 elements, so we need to create two columns
    dataset.data["formant1"] = dataset.data.apply(lambda x: x["formants"][0], axis=1)
    dataset.data["formant2"] = dataset.data.apply(lambda x: x["formants"][1], axis=1)

    # compute the mean for each formant grouping by vowels and labels
    formants = (
        dataset.data.groupby(["vowel", "label"])
        .mean()[["formant1", "formant2"]]
        .reset_index()
    )

    # Plot the vowel diagram where x-axis is F1 and y-axis is F2. The vowels label is the vowel column and the label column is the color

    import matplotlib.pyplot as plt

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Iterate through each vowel category and plot using plt.scatter with a label
    for vowel in range(5):
        data = formants[formants["vowel"] == vowel].iloc[0]
        label = data["label"]
        marker = ["$A$", "$E$", "$I$", "$O$", "$U$"][vowel]
        ax.scatter(
            data["formant2"],
            data["formant1"],
            c="blue",
            marker=marker,
            s=500,
            label="PD" if label == 1 else "Healthy",
        )

        data = formants[formants["vowel"] == vowel].iloc[1]
        label = data["label"]
        marker = ["$A$", "$E$", "$I$", "$O$", "$U$"][vowel]
        ax.scatter(
            data["formant2"],
            data["formant1"],
            c="red",
            marker=marker,
            s=1000,
            label="PD" if label == 1 else "Healthy",
        )

    # Invert both axes to make them decreasing
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Add a legend
    ax.legend()

    # Show the combined plot
    plt.show()

    # VaDE (Variational Deep Embedding:A Generative Approach to Clustering)


def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment

    Y_pred, Y = np.array(Y_pred).astype(np.int64), np.array(Y).astype(np.int64)
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return sum([w[row[i], col[i]] for i in range(row.shape[0])]) * 1.0 / Y_pred.size


def nmi(Y_pred, Y):
    from sklearn.metrics.cluster import normalized_mutual_info_score

    Y_pred, Y = np.array(Y_pred), np.array(Y)
    assert Y_pred.size == Y.size
    return normalized_mutual_info_score(Y_pred, Y, average_method="arithmetic")


def plot_gaussians_generative(model, latent_mu, wandb_flag, name, fold, savepath):
    # p(y) = Cat(10)
    py = torch.eye(model.k).to(model.device)
    # Sample from generative model
    z_mu, z_logvar = torch.chunk(model.generative_pz_y(py), 2, dim=1)
    z_var = torch.nn.functional.softplus(z_logvar)

    fig, ax = plt.subplots(figsize=(20, 20))
    for i in range(model.k):
        mu = z_mu[i].cpu().detach().numpy()
        var = z_var[i].cpu().detach().numpy()
        cov = np.diag(var)

        x = np.linspace(
            np.min(latent_mu[:, 0]),
            np.max(latent_mu[:, 0]),
        )
        y = np.linspace(
            np.min(latent_mu[:, 1]),
            np.max(latent_mu[:, 1]),
        )
        X, Y = np.meshgrid(x, y)

        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)

        coe = 1 / (2 * np.pi * cov_det) ** (1 / 2)
        w = coe * np.exp(
            -0.5
            * coe
            * np.e
            ** (
                -0.5
                * (
                    cov_inv[0, 0] * (X - mu[0]) ** 2
                    + (cov_inv[0, 1] + cov_inv[1, 0]) * (X - mu[0]) * (Y - mu[1])
                    + cov_inv[1, 1] * (Y - mu[1]) ** 2
                )
            )
        )
        ax.contour(
            X,
            Y,
            w,
        )
        # Use star as a marker
        ax.scatter(mu[0], mu[1], label="Gaussian " + str(i), alpha=0.5, marker="*")
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(f"Gaussians in " + str(name))
    ax.legend()
    save_path = savepath + f"gaussians_generative_{fold}_{name}.png"
    fig.savefig(
        save_path,
    )

    if wandb_flag:
        wandb.log({str(name) + "/gaussians_generative": wandb.Image(fig)})

    plt.close()


def plot_latent_space_by_vowels(
    labels,
    vowels,
    latent_mu,
    fold,
    wandb_flag,
    name,
    xlabel,
    ylabel,
    savepath,
):
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

    fig.savefig(savepath + f"latent_space_vowels_{fold}_{name}.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_vowels": wandb.Image(fig)})
    plt.close(fig)


def plot_latent_space_vowels_inverse(
    labels, vowels, latent_mu, fold, wandb_flag, name, xlabel, ylabel, savepath
):
    fig, ax = plt.subplots(figsize=(20, 20))
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

    fig.savefig(savepath + f"latent_space_vowels_{fold}_{name}_inverse.png")
    if wandb_flag:
        wandb.log({str(name) + "/latent_space_vowels_inverse": wandb.Image(fig)})
    plt.close(fig)


def plot_gaussians_generative_over_vowels(
    labels, vowels, latent_mu, fold, wandb_flag, name, xlabel, ylabel, savepath, model
):
    fig, ax = plt.subplots(figsize=(20, 20))
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

    # p(y) = Cat(10)
    py = torch.eye(model.k).to(model.device)
    # Sample from generative model
    z_mu, z_logvar = torch.chunk(model.generative_pz_y(py), 2, dim=1)
    z_var = torch.nn.functional.softplus(z_logvar)

    for i in range(model.k):
        mu = z_mu[i].cpu().detach().numpy()
        var = z_var[i].cpu().detach().numpy()
        cov = np.diag(var)

        x = np.linspace(
            np.min(latent_mu[:, 0]),
            np.max(latent_mu[:, 0]),
        )
        y = np.linspace(
            np.min(latent_mu[:, 1]),
            np.max(latent_mu[:, 1]),
        )
        X, Y = np.meshgrid(x, y)

        v, w = linalg.eigh(cov)
        v1 = 2.0 * 1 * np.sqrt(v)  # 1 std
        v2 = 2.0 * 2 * np.sqrt(v)  # 2 std
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(
            mu, v1[0], v1[1], angle=180.0 + angle, facecolor="none", edgecolor="green"
        )
        ell2 = mpl.patches.Ellipse(
            mu, v2[0], v2[1], angle=180.0 + angle, facecolor="none", edgecolor="orange"
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ell2.set_clip_box(ax.bbox)
        ell2.set_alpha(0.5)
        ax.add_artist(ell2)
        # Use star as a marker
        ax.scatter(mu[0], mu[1], label="Gaussian " + str(i), marker="*", s=100)
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(f"Latent space with Gaussians distributions")
    ax.legend()
    save_path = savepath + f"gaussians_generative_and_test_vowels_{fold}_{name}.png"
    fig.savefig(
        save_path,
    )
    if wandb_flag:
        wandb.log(
            {str(name) + "/gaussians_generative_and_test_vowels_": wandb.Image(fig)}
        )

    plt.close()
