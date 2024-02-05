import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import wandb
from matplotlib import pyplot as plt
from scipy import linalg
import matplotlib as mpl
import random
from collections import Counter
import copy


def augment_data(dataset, validation=False):
    augmented_data1 = []
    augmented_data2 = []
    for data in dataset:
        if validation:
            spectrogram, label, manner, ds, id, audio_id = data
        else:
            spectrogram, label, manner, ds, audio_id = data

        spectrogram_to_modify1 = copy.deepcopy(spectrogram)
        spectrogram_to_modify2 = copy.deepcopy(spectrogram)

        # First augmentation
        augmented_spectrogram_1 = augment_spectrogram(
            spectrogram_to_modify1, p=0.8, q=0.8, r=0.2
        )
        if validation:
            augmented_data1.append(
                (augmented_spectrogram_1, label, manner, ds, id, audio_id)
            )
        else:
            augmented_data1.append(
                (augmented_spectrogram_1, label, manner, ds, audio_id)
            )

        # Second augmentation
        augmented_spectrogram_2 = augment_spectrogram(
            spectrogram_to_modify2, p=0.8, q=0.8, r=0.2
        )
        if validation:
            augmented_data2.append(
                (augmented_spectrogram_2, label, manner, ds, id, audio_id)
            )
        else:
            augmented_data2.append(
                (augmented_spectrogram_2, label, manner, ds, audio_id)
            )

    dataset += augmented_data1
    dataset += augmented_data2

    return dataset


# Function for frequency-based data augmentation
def augment_spectrogram(spectrogram, p=0.5, q=0.5, r=0.2):
    _, freq_dimension, time_dimension = spectrogram.shape

    # With probability p, mask 15% of frequency bands
    if random.random() < p:
        mask_percentage = 0.15
        mask_size = int(freq_dimension * mask_percentage)
        mask_start = random.randint(0, freq_dimension - mask_size)
        spectrogram[0, mask_start : mask_start + mask_size, :] = 0

    # With probability q, mask 15% of time windows
    if random.random() < q:
        mask_percentage = 0.15
        mask_size = int(time_dimension * mask_percentage)
        mask_start = random.randint(0, time_dimension - mask_size)
        spectrogram[0, :, mask_start : mask_start + mask_size] = 0

    # With probability r, add Gaussian noise
    if random.random() < r:
        noise = np.random.normal(0, 1, spectrogram.shape)
        spectrogram += 0.10 * noise
        # Renormalize the spectrogram
        spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()

    return spectrogram


def find_minority_class(dataset):
    labels = [data[1] for data in dataset]
    label_counts = Counter(labels)
    minority_class = min(label_counts, key=label_counts.get)
    return minority_class, label_counts


def stratify_dataset(dataset):
    # Check stratification of labels (they are the second element of the triplet)
    minority_class, label_counts = find_minority_class(dataset)
    majority_class_count = max(label_counts.values())

    if label_counts[minority_class] < majority_class_count:
        additional_augmented_data = []
        augmentations_needed = majority_class_count - label_counts[minority_class]
        minority_class_data = [data for data in dataset if data[1] == minority_class]

        while augmentations_needed > 0:
            for data in minority_class_data:
                if augmentations_needed <= 0:
                    break
                spectrogram, label, manner, ds, audio_id = data

                spectrogram_to_modify3 = copy.deepcopy(spectrogram)

                augmented_spectrogram = augment_spectrogram(spectrogram_to_modify3)

                additional_augmented_data.append(
                    (augmented_spectrogram, label, manner, ds, audio_id)
                )
                augmentations_needed -= 1

        # Combine additional augmented data
        balanced_dataset = dataset + additional_augmented_data
    else:
        balanced_dataset = dataset

    return balanced_dataset


def make_balanced_sampler(dataset, validation=False):
    # Count the occurrences of each class
    class_counts = {}
    if validation:
        dataset = [data[:5] for data in dataset]

    for data in dataset:
        label = data[1]
        label = label.item()  # Assuming label is a tensor
        class_counts[label] = class_counts.get(label, 0) + 1

    # Assign weights inversely proportional to class frequencies
    weights = []
    for data in dataset:
        label = data[1]
        label = label.item()
        weight = 1.0 / class_counts[label]
        weights.append(weight)

    # Create a WeightedRandomSampler
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return sampler


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


def log_normal(x, mu, var):
    return -0.5 * torch.sum(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1
    )


def KL_cat(qy, qy_logits, k, reducer="mean"):
    # KL Divergence between an arbitrary categorical (q(y)) and a prior uniform distribution (U(0,1))
    #  loss = (1/n) * Σ(qx * log(qx/px)), because we use a uniform prior px = 1/k
    #  loss = (1/n) * Σ(qx * (log(qx) - log(1/k)))
    log_q = torch.log_softmax(qy_logits, dim=-1)
    log_p = torch.log(1 / torch.tensor(k))
    cat_loss = torch.sum(qy * (log_q - log_p), dim=-1)
    if reducer == "mean":
        cat_loss = torch.mean(cat_loss)
    else:
        cat_loss = torch.sum(cat_loss)
    return cat_loss


def plot_logopeda_alb_neuro(
    model,
    data_train,
    data_test,
    wandb_flag,
    name="default",
    supervised=False,
    samples=1000,
    path_to_plot="local_results/spectrograms/manner_gmvae_neurovoz",
):
    import copy

    # Generate mu and sigma in training
    model.eval()
    with torch.no_grad():
        data_input = torch.Tensor(np.vstack(data_train["spectrogram"]))
        # do model_infere by batches
        batch_size = 128
        latent_mu_train_original_space = []
        print("Calculating latent space for train samples")
        for i in range(0, len(data_input), batch_size):
            e_s = model.spec_encoder_forward(
                data_input[i : i + batch_size].to(model.device).unsqueeze(1)
            )
            (
                _,
                _,
                _,
                latent_mu_train_original_space_batch,
                _,
                _,
                _,
            ) = model.inference_forward(e_s)
            latent_mu_train_original_space.append(
                latent_mu_train_original_space_batch.cpu().detach().numpy()
            )
            del _, latent_mu_train_original_space_batch
        latent_mu_train_original_space = np.vstack(latent_mu_train_original_space)
        torch.cuda.empty_cache()
        print("Calculating latent space for test samples")
        data_input = torch.Tensor(np.vstack(data_test["spectrogram"]))
        latent_mu_test_original_space = []
        # do model_infere by batches
        for i in range(0, len(data_input), batch_size):
            e_s = model.spec_encoder_forward(
                data_input[i : i + batch_size].to(model.device).unsqueeze(1)
            )
            (
                _,
                _,
                _,
                latent_mu_test_original_space_batch,
                _,
                _,
                _,
            ) = model.inference_forward(e_s)
            latent_mu_test_original_space.append(
                latent_mu_test_original_space_batch.cpu().detach().numpy()
            )
            del _, latent_mu_test_original_space_batch
        latent_mu_test_original_space = np.vstack(latent_mu_test_original_space)
        torch.cuda.empty_cache()
        del data_input
        print("Latent space calculated")

    manner_train = np.array([np.array(x) for x in data_train["manner"]], dtype=int)
    manner_test = np.array([np.array(x) for x in data_test["manner"]], dtype=int)

    labels_train = np.array(data_train["label"].values, dtype=int)
    labels_test = np.array(data_test["label"].values, dtype=int)

    # Repeat labels manner.shape[1] times
    latent_mu_train = copy.copy(latent_mu_train_original_space)
    lm_train_original = latent_mu_train_original_space
    labels_train = np.repeat(labels_train, manner_train.shape[1], axis=0)
    manner_train = manner_train.reshape(-1)
    # Remove all affricates
    idx = np.argwhere(manner_train == 6).ravel()
    manner_train = np.delete(manner_train, idx)
    labels_train = np.delete(labels_train, idx)
    latent_mu_train = np.delete(latent_mu_train, idx, axis=0)
    lm_train_original = np.delete(lm_train_original, idx, axis=0)
    # Remove all silence
    idx = np.argwhere(manner_train == 7).ravel()
    manner_train = np.delete(manner_train, idx)
    labels_train = np.delete(labels_train, idx)
    latent_mu_train = np.delete(latent_mu_train, idx, axis=0)
    lm_train_original = np.delete(lm_train_original, idx, axis=0)

    # Repeat labels manner.shape[1] times
    latent_mu_test = copy.copy(latent_mu_test_original_space)
    lm_test_original = latent_mu_test_original_space
    labels_test = np.repeat(labels_test, manner_test.shape[1], axis=0)
    manner_test = manner_test.reshape(-1)
    # Remove all affricates
    idx = np.argwhere(manner_test == 6).ravel()
    manner_test = np.delete(manner_test, idx)
    labels_test = np.delete(labels_test, idx)
    latent_mu_test = np.delete(latent_mu_test, idx, axis=0)
    lm_test_original = np.delete(lm_test_original, idx, axis=0)
    # Remove all silence
    idx = np.argwhere(manner_test == 7).ravel()
    manner_test = np.delete(manner_test, idx)
    labels_test = np.delete(labels_test, idx)
    latent_mu_test = np.delete(latent_mu_test, idx, axis=0)
    lm_test_original = np.delete(lm_test_original, idx, axis=0)

    # Select randomly samples of each dataset
    # idx = np.random.choice(len(latent_mu_train), samples)
    # manner_train = manner_train[idx]
    # labels_train = labels_train[idx]
    # latent_mu_train = latent_mu_train[idx]

    # # Select only SAMPLES from test
    # idx = np.random.choice(len(latent_mu_test), samples)
    # manner_test = manner_test[idx]
    # labels_test = labels_test[idx]
    # latent_mu_test = latent_mu_test[idx]

    # Check latent_mu shape, if greater than 2 do a t-SNE
    if latent_mu_train.shape[1] > 2:
        # Import UMAP
        import umap

        train_mu_shape = latent_mu_train.shape

        print("Fitting UMAP")

        # Train the UMAP model with only "samples"  samples
        idx = np.random.choice(len(latent_mu_train), samples)

        # # TSNE only albayzin
        # umapmodel = umap.UMAP(
        #     n_components=2, metric="mahalanobis", n_neighbors=200
        # ).fit(latent_mu_train[idx])
        umapmodel = None

        # Convert test to 2d
        print("Calculating UMAP...")
        # latent_mu_train = umapmodel.transform(latent_mu_train)
        # latent_mu_test = umapmodel.transform(latent_mu_test)

        print("The shape of the train latent space is now: ", latent_mu_train.shape)
        print("The shape of the test latent space is now: ", latent_mu_test.shape)

        xlabel = "t-SNE dim 1"
        ylabel = "t-SNE dim 2"

    else:
        umapmodel = None
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    # =========================================== TRAIN SAMPLES AKA TRAIN CLUSTERS AKA HEALTHY CLUSTERS FROM ALBAYZIN ===========================================
    import matplotlib

    # cmap = matplotlib.cm.get_cmap("Set1")

    # fig, ax = plt.subplots(figsize=(20, 20))
    # class_labels = {
    #     0: "Plosives",
    #     1: "Plosives voiced",
    #     2: "Nasals",
    #     3: "Fricatives",
    #     4: "Liquids",
    #     5: "Vowels",
    #     # 6: "Affricates",
    #     # 7: "Silence",
    #     # 8: "Short pause",
    # }
    # for i in range(5):
    #     ax.scatter(
    #         latent_mu_train[:, 0][np.where(manner_train == i)],
    #         latent_mu_train[:, 1][np.where(manner_train == i)],
    #         color=cmap(i),
    #         label=class_labels[i],
    #     )
    # # Add labels and title
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_title(f"Latent space in " + str(name) + " for fold {fold}")
    # ax.legend()
    # fig.savefig(
    #     path_to_plot + f"/latent_space_albayzin_wo_vowels.png",
    # )
    # ax.scatter(
    #     latent_mu_train[:, 0][np.where(manner_train == 5)],
    #     latent_mu_train[:, 1][np.where(manner_train == 5)],
    #     color=cmap(5),
    #     label=class_labels[5],
    # )
    # ax.legend()
    # fig.savefig(
    #     path_to_plot + f"/latent_space_albayzin_w_vowels.png",
    # )

    # if wandb_flag:
    #     wandb.log({str(name) + "/latent_space": wandb.Image(fig)})

    # # =========================================== HEALTHY TEST SAMPLES from NEUROVOZ. We are going to do 7 plots. One per phoneme. ===========================================
    # # select only latent_mu_test with labels_test = 0
    # idx = np.argwhere(labels_test == 0).ravel()
    # latent_mu_test_healthy = latent_mu_test[idx]
    # manner_test_copy = copy.copy(manner_test[idx])

    # for i in range(5):
    #     i = int(i)

    #     idx = np.argwhere(manner_train == i).ravel()

    #     fig, ax = plt.subplots(figsize=(20, 20))

    #     # Scatter ax

    #     # Divide the scatter in two scatters: frst all healhty samples.
    #     for j in range(5):
    #         ax.scatter(
    #             latent_mu_train[:, 0][np.where(manner_train == j)],
    #             latent_mu_train[:, 1][np.where(manner_train == j)],
    #             color=cmap(j),
    #             label=class_labels[j],
    #             alpha=0.2,
    #         )

    #     idx = np.argwhere(manner_test_copy == i).ravel()

    #     ax.scatter(
    #         latent_mu_test_healthy[idx, 0],
    #         latent_mu_test_healthy[idx, 1],
    #         alpha=1,
    #         color=cmap(i),
    #         label=class_labels[i],
    #     )
    #     # Add labels and title
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(f"Latent space in " + str(name) + " for fold {fold}")
    #     ax.legend()
    #     fig.savefig(
    #         path_to_plot + f"/latent_space_healthy_albayzin_vs_neurovoz_class_{i}.png",
    #     )

    #     if wandb_flag:
    #         wandb.log(
    #             {
    #                 str(name)
    #                 + "/latent_space_healthy_albayzin_vs_neurovoz_class_"
    #                 + str(i): wandb.Image(fig)
    #             }
    #         )

    #     plt.close()

    # # =========================================== PARKINSONIAN TEST SAMPLES from NEUROVOZ. We are going to do 7 plots. One per phoneme. ===========================================

    # # select only latent_mu_test with ["label"] = 1
    # idx = np.argwhere(labels_test == 1).ravel()
    # latent_mu_test_park = latent_mu_test[idx]
    # manner_test_copy = copy.copy(manner_test[idx])

    # for i in range(5):
    #     i = int(i)
    #     fig, ax = plt.subplots(figsize=(20, 20))

    #     # Scatter ax
    #     # Divide the scatter in two scatters: frst all healhty samples.
    #     for j in range(5):
    #         sct = ax.scatter(
    #             latent_mu_train[:, 0][np.where(manner_train == j)],
    #             latent_mu_train[:, 1][np.where(manner_train == j)],
    #             color=cmap(j),
    #             label=class_labels[j],
    #             alpha=0.2,
    #         )
    #     idx = np.argwhere(manner_test_copy == i).ravel()

    #     ax.scatter(
    #         latent_mu_test_park[idx, 0],
    #         latent_mu_test_park[idx, 1],
    #         alpha=1,
    #         color=cmap(i),
    #         label=class_labels[i],
    #     )
    #     # Add labels and title
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(f"Latent space in " + str(name) + " for fold {fold}")

    #     ax.legend()

    #     fig.savefig(
    #         path_to_plot + f"/latent_space_park_albayzin_vs_neurovoz_class_{i}.png",
    #     )

    #     if wandb_flag:
    #         wandb.log(
    #             {
    #                 str(name)
    #                 + "/latent_space_park_albayzin_vs_neurovoz_class_"
    #                 + str(i): wandb.Image(fig)
    #             }
    #         )

    #     plt.close()

    print("Calculating distance in a space of dimensions: ", lm_train_original.shape[1])

    calculate_euclidean_distances_manner(
        lm_train_original,
        lm_test_original,
        manner_train,
        manner_test,
        labels_train,
        labels_test,
        wandb_flag,
        path_to_plot=path_to_plot,
    )

    print("Calculating jensen shannon")
    print("Calculating distances in a space of dimensions: ", lm_test_original.shape[1])
    calculate_distances_manner(
        model,
        lm_train_original,
        lm_test_original,
        manner_train,
        manner_test,
        labels_train,
        labels_test,
        umapmodel,
        wandb_flag,
        path_to_plot=path_to_plot,
    )


def plot_logopeda(
    model,
    data_train,
    data_test,
    wandb_flag,
    name="default",
    supervised=False,
    samples=1000,
):
    # Generate mu and sigma in training
    model.eval()
    with torch.no_grad():
        data_input = (
            torch.Tensor(np.vstack(data_train["spectrogram"]))
            .to(model.device)
            .unsqueeze(1)
        )
        (
            z,
            qy_logits,
            qy,
            latent_mu_train,
            qz_logvar,
            x_hat,
            x_hat_unflatten,
        ) = model.infere(data_input)
        data_input = (
            torch.Tensor(np.vstack(data_test["spectrogram"]))
            .to(model.device)
            .unsqueeze(1)
        )
        (
            z,
            qy_logits,
            qy,
            latent_mu_test,
            qz_logvar,
            x_hat,
            x_hat_unflatten,
        ) = model.infere(data_input)

    manner_train = np.array([np.array(x) for x in data_train["manner"]], dtype=int)
    manner_test = np.array([np.array(x) for x in data_test["manner"]], dtype=int)

    # # p(y) = Cat(10)
    # py = torch.eye(model.k).to(model.device)
    # # Sample from generative model
    # z_mu, z_logvar = torch.chunk(model.generative_pz_y(py), 2, dim=1)
    # z_var = torch.nn.functional.softplus(z_logvar)

    # Select only 100 samples from train
    idx = np.random.choice(len(latent_mu_train), samples)
    manner_train = manner_train.reshape(-1)[idx]
    latent_mu_train = latent_mu_train[idx].cpu().detach().numpy()

    # Select only SAMPLES from test
    idx = np.random.choice(len(latent_mu_test), samples)
    manner_test = manner_test.reshape(-1)[idx]
    latent_mu_test = latent_mu_test[idx].cpu().detach().numpy()

    # Check latent_mu shape, if greater than 2 do a t-SNE
    if latent_mu_train.shape[1] > 2:
        from sklearn.manifold import TSNE

        train_mu_shape = latent_mu_train.shape

        # Convert all to 2D
        all_vec = np.concatenate(
            (latent_mu_train, latent_mu_test),
            axis=0,
        )

        all_2D = TSNE(n_components=2).fit_transform(all_vec)

        # Separate info
        latent_mu_train = all_2D[: train_mu_shape[0]]
        latent_mu_test = all_2D[train_mu_shape[0] :]

        xlabel = "t-SNE dim 1"
        ylabel = "t-SNE dim 2"

    else:
        xlabel = "Latent dim 1"
        ylabel = "Latent dim 2"

    # =========================================== TRAIN SAMPLES AKA TRAIN CLUSTERS AKA HEALTHY CLUSTERS ===========================================
    import matplotlib

    cmap = matplotlib.cm.get_cmap("Set1")

    fig, ax = plt.subplots(figsize=(20, 20))
    sct = ax.scatter(
        latent_mu_train[:, 0],
        latent_mu_train[:, 1],
        c=manner_train,
        cmap=cmap,
    )
    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Latent space in " + str(name) + " for fold {fold}")
    legend_elements = []
    class_labels = {
        0: "Plosives",
        1: "Plosives voiced",
        2: "Nasals",
        3: "Fricatives",
        4: "Liquids",
        5: "Vowels",
        # 6: "Affricates",
        # 7: "Silence",
        # 8: "Short pause",
    }
    for class_value, class_label in enumerate(class_labels):
        # Get the label for the class from the list
        # Get the color from the colormap
        color = cmap(class_value / (len(class_labels) - 1))
        # Create a dummy scatter plot for each class with a single point
        dummy_scatter = matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=class_labels[class_label],
            markersize=10,
        )
        legend_elements.append(dummy_scatter)

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        title="Classes",
    )
    fig.savefig(
        path_to_plot + f"/latent_space_healthy.png",
    )

    if wandb_flag:
        wandb.log({str(name) + "/latent_space": wandb.Image(fig)})

    # =========================================== TEST SAMPLES. We are going to do 7 plots. One per phoneme. ===========================================

    # First plot. How the parkinsonian plosives are distributed in the latent space

    for i in range(8):
        i = int(i)
        fig, ax = plt.subplots(figsize=(20, 20))

        # Scatter ax

        # Divide the scatter in two scatters: frst all healhty samples.
        ax.scatter(
            latent_mu_train[:, 0],
            latent_mu_train[:, 1],
            c=manner_train,
            alpha=0.2,
            cmap=cmap,
        )
        idx = np.argwhere(manner_test == i).ravel()

        ax.scatter(
            latent_mu_test[idx, 0],
            latent_mu_test[idx, 1],
            alpha=1,
            color=cmap(i / (len(class_labels))),
        )
        # Add labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Latent space in " + str(name) + " for fold {fold}")

        legend_elements = []
        class_labels = {
            0: "Plosives",
            1: "Plosives voiced",
            2: "Nasals",
            3: "Fricatives",
            4: "Liquids",
            5: "Vowels",
            # 6: "Affricates",
            # 7: "Silence",
            # 8: "Short pause",
        }
        for class_value, class_label in enumerate(class_labels):
            # Get the label for the class from the list
            # Get the color from the colormap
            color = cmap(class_value / len(class_labels))
            # Create a dummy scatter plot for each class with a single point
            dummy_scatter = matplotlib.lines.Line2D(
                [0],
                [0],
                marker="o",
                color=color,
                label=class_labels[class_label],
                markersize=10,
            )
            legend_elements.append(dummy_scatter)

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(1.15, 1),
            title="Classes",
        )
        fig.savefig(
            path_to_plot + f"/latent_space_parkinsonian_class_{i}.png",
        )

        if wandb_flag:
            wandb.log(
                {
                    str(name)
                    + "/latent_space_parkinsonian_class_"
                    + str(i): wandb.Image(fig)
                }
            )

        plt.close()

    # for i in range(model.k):
    #     mu = z_mu[i]
    #     var = z_var[i]
    #     cov = np.diag(var)

    #     x = np.linspace(
    #         np.min(latent_mu[:, 0]),
    #         np.max(latent_mu[:, 0]),
    #     )
    #     y = np.linspace(
    #         np.min(latent_mu[:, 1]),
    #         np.max(latent_mu[:, 1]),
    #     )
    #     X, Y = np.meshgrid(x, y)

    #     v, w = linalg.eigh(cov)
    #     v1 = 2.0 * 1 * np.sqrt(v)  # 1 std
    #     v2 = 2.0 * 2 * np.sqrt(v)  # 2 std
    #     u = w[0] / linalg.norm(w[0])

    #     # Plot an ellipse to show the Gaussian component
    #     angle = np.arctan(u[1] / u[0])
    #     angle = 180.0 * angle / np.pi  # convert to degrees
    #     ell = mpl.patches.Ellipse(
    #         mu, v1[0], v1[1], angle=180.0 + angle, facecolor="none", edgecolor="green"
    #     )
    #     ell2 = mpl.patches.Ellipse(
    #         mu, v2[0], v2[1], angle=180.0 + angle, facecolor="none", edgecolor="orange"
    #     )
    #     ell.set_clip_box(ax.bbox)
    #     ell.set_alpha(0.5)
    #     ax.add_artist(ell)
    #     ell2.set_clip_box(ax.bbox)
    #     ell2.set_alpha(0.5)
    #     ax.add_artist(ell2)
    #     # Use star as a marker
    #     ax.scatter(mu[0], mu[1], label="Gaussian " + str(i), marker="*", s=100)
    # ax.set_xlabel("Latent dim 1")
    # ax.set_ylabel("Latent dim 2")
    # ax.set_title(f"Latent space with Gaussians distributions")
    # ax.legend()
    # save_path = path_to_plot + f"/gaussians_generative_and_test_vowels_{fold}_{name}.png"
    # fig.savefig(
    #     save_path,
    # )
    # if wandb_flag:
    #     wandb.log(
    #         {str(name) + "/gaussians_generative_and_test_vowels_": wandb.Image(fig)}
    #     )

    # plt.close()


def plot_latent_space(
    model, data, fold, wandb_flag, name="default", supervised=False, samples=1000
):
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
    idx = np.random.choice(len(labels), samples)
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
        all = np.concatenate(
            (latent_mu, z_mu.cpu().detach().numpy(), z_var.cpu().detach().numpy()),
            axis=0,
        )

        all_2D = TSNE(n_components=2).fit_transform(all)

        # Separate info
        latent_mu = all_2D[: inference_mu_shape[0]]
        z_mu = all_2D[
            inference_mu_shape[0] : inference_mu_shape[0] + generative_mu_shape[0]
        ]
        z_var = all_2D[inference_mu_shape[0] + generative_mu_shape[0] :]

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
    scatter1 = ax.scatter(
        latent_mu[idxH, 0],
        latent_mu[idxH, 1],
        c=manner_labels[idxH],
        alpha=0.2,
        cmap="Set1",
    )
    scatter2 = ax.scatter(
        latent_mu[idxPD, 0],
        latent_mu[idxPD, 1],
        c=manner_labels[idxPD],
        alpha=1,
        cmap="Set1",
    )

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
        "Liquids",
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
        path_to_plot + f"/latent_space_{fold}_{name}.png",
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
    save_path = (
        path_to_plot + f"/gaussians_generative_and_test_vowels_{fold}_{name}.png"
    )
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


def calculate_distances_manner(
    model,
    latent_mu_train,
    latent_mu_test,
    manner_train,
    manner_test,
    labels_train,
    labels_test,
    umapmodel,
    wandb_flag,
    path_to_plot,
):
    from sklearn.neighbors import KernelDensity
    from scipy.spatial.distance import jensenshannon

    print("Calculating distances...")

    def calculate_kde(data):
        # if data.shape[0] < 2 * data.shape[1]:
        #     return None
        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(data)
        return kde

    def calculate_js_distance(kde1, kde2, positions):
        if kde1 is None or kde2 is None:
            return 0
        p = kde1.score_samples(positions)  # This returns the log-likelihood of the data
        p = np.exp(p)  # Convert to probabilities
        p = p / np.sum(p)
        q = kde2.score_samples(positions)
        q = np.exp(q)
        q = q / np.sum(q)
        return jensenshannon(p, q)

    def calculate_cluster_distance(
        latent_mu_one, latent_mu_two, kde_train, kde_test, umapmodel
    ):
        # Sample from the GMM of the generative model
        # First generate uniformly distributed samples up to model.k
        cat_samples = np.random.choice(
            model.k, size=1000 * latent_mu_one.shape[1], replace=True
        )
        # # Convert them to one-hot-encoder
        # positions = (
        #     torch.chunk(
        #         model.generative_pz_y(torch.eye(model.k)[cat_samples].to(model.device)),
        #         2,
        #         dim=1,
        #     )[0]
        #     .cpu()
        #     .detach()
        #     .numpy()
        # )
        # UMAP to 2D
        # positions = umapmodel.transform(positions)

        # Uniformly sampling from the latent space restricted to the min and max of the classes we are comparing
        positions = np.concatenate((latent_mu_one, latent_mu_two), axis=0)
        # Get randomly 32000 points
        positions = positions[
            np.random.choice(
                positions.shape[0], size=1000, replace=len(positions) < 1000
            )
        ]
        # positions = np.random.uniform(low=latent_mu.min(), high=latent_mu.max(), size=(5000 * latent_mu.shape[1], latent_mu.shape[1]))

        distance = calculate_js_distance(kde_train, kde_test, positions)
        return distance

    # Get unique manner classes
    unique_manner_train = np.unique(manner_train)
    unique_manner_test = np.unique(manner_test)

    distances_albayzin = np.zeros((len(unique_manner_train), len(unique_manner_train)))
    distances_healthy = np.zeros((len(unique_manner_train), len(unique_manner_test)))
    distances_healthy_parkinson = np.zeros(
        (len(unique_manner_train), len(unique_manner_test))
    )
    distances_neuropark_neurosanos = np.zeros(
        (len(unique_manner_train), len(unique_manner_test))
    )

    kde_albayzin = [
        calculate_kde(latent_mu_train[(labels_train == 0) & (manner_train == manner)])
        for manner in unique_manner_train
    ]
    kde_neurovoz_healhty = [
        calculate_kde(latent_mu_test[(labels_test == 0) & (manner_test == manner)])
        for manner in unique_manner_test
    ]
    # latent_park = latent_mu_test[(labels_test == 1)]
    # manner_test = manner_test[(labels_test == 1)]
    # # assert that latent_park and manner_test have the same shape
    # assert latent_park.shape[0] == manner_test.shape[0]
    # # Sample randomly to have the same amount as healthy
    # latent_park = latent_park[
    #     np.random.choice(
    #         latent_park.shape[0],
    #         size=latent_mu_test[(labels_test == 0)].shape[0],
    #         replace=False,
    #     )
    # ]
    # kde_neurovoz_parkinson = [
    #     calculate_kde(latent_park[manner_test == manner])
    #     for manner in unique_manner_test
    # ]
    kde_neurovoz_parkinson = [
        calculate_kde(latent_mu_test[(labels_test == 1) & (manner_test == manner)])
        for manner in unique_manner_test
    ]

    for i, manner_i in enumerate(unique_manner_train):
        for j, manner_j in enumerate(unique_manner_test):
            print(
                "Calculating distance for Albayzin vs Albayzin for manner classes "
                + str(manner_i)
                + " and "
                + str(manner_j)
                + "..."
            )
            distances_albayzin[i, j] = calculate_cluster_distance(
                latent_mu_train[(labels_train == 0) & (manner_train == manner_i)],
                latent_mu_train[(labels_train == 0) & (manner_train == manner_j)],
                kde_albayzin[i],
                kde_albayzin[j],
                umapmodel=umapmodel,
            )
            print(
                "Calculating distance for Albayzin Healthy vs Neurovoz Healthy for manner classes "
                + str(manner_i)
                + " and "
                + str(manner_j)
                + "..."
            )
            distances_healthy[i, j] = calculate_cluster_distance(
                latent_mu_train[(labels_train == 0) & (manner_train == manner_i)],
                latent_mu_test[(labels_test == 0) & (manner_test == manner_j)],
                kde_albayzin[i],
                kde_neurovoz_healhty[j],
                umapmodel=umapmodel,
            )
            print(
                "Calculating distance for Albayzin Healthy vs Neurovoz Parkinson for manner classes "
                + str(manner_i)
                + " and "
                + str(manner_j)
                + "..."
            )
            distances_healthy_parkinson[i, j] = calculate_cluster_distance(
                latent_mu_train[(labels_train == 0) & (manner_train == manner_i)],
                latent_mu_test[(labels_test == 1) & (manner_test == manner_j)],
                kde_albayzin[i],
                kde_neurovoz_parkinson[j],
                umapmodel=umapmodel,
            )

            print("Caculating distance for Neurovoz Healthy vs Neurovoz Parkinson")
            distances_neuropark_neurosanos[i, j] = calculate_cluster_distance(
                latent_mu_test[(labels_test == 0) & (manner_test == manner_i)],
                latent_mu_test[(labels_test == 1) & (manner_test == manner_j)],
                kde_neurovoz_healhty[i],
                kde_neurovoz_parkinson[j],
                umapmodel=umapmodel,
            )

    distances = [
        distances_albayzin,
        distances_healthy,
        distances_healthy_parkinson,
        distances_neuropark_neurosanos,
    ]

    # Calculate the mean and std of the diagonals
    alb_diag = np.diag(distances_albayzin)
    healthy_diag = np.diag(distances_healthy)
    parkinson_diag = np.diag(distances_healthy_parkinson)
    neuro_diag = np.diag(distances_neuropark_neurosanos)

    print("Diagonal distances in train vs train:")
    print("Diagonal: " + str(alb_diag))
    print("Mean: " + str(np.mean(alb_diag)))
    print("Std: " + str(np.std(alb_diag)))

    print("Diagonal distance in train vs healthy test")
    print("Diagonal: " + str(healthy_diag))
    print("Mean: " + str(np.mean(healthy_diag)))
    print("Std: " + str(np.std(healthy_diag)))

    print("Diagonal distance in train vs parkinson test")
    print("Diagonal: " + str(parkinson_diag))
    print("Mean: " + str(np.mean(parkinson_diag)))
    print("Std: " + str(np.std(parkinson_diag)))

    print("Diagonal distance in healthy neurovoz test vs parkinson neurovoz test")
    print("Diagonal: " + str(neuro_diag))
    print("Mean: " + str(np.mean(neuro_diag)))
    print("Std: " + str(np.std(neuro_diag)))

    print("Difference between train vs parkinson and train vs healthy")
    print("Diagonal: " + str(parkinson_diag - healthy_diag))
    print("Mean: " + str(np.mean(parkinson_diag - healthy_diag)))
    print("Std: " + str(np.std(parkinson_diag - healthy_diag)))

    # Calculate MAPE of the difference
    mape = np.mean(np.abs((healthy_diag - parkinson_diag) / healthy_diag)) * 100
    print("MAPE: " + str(mape))

    # Plot the distances as a heatmap using seaborn
    print("Plotting distances...")
    import seaborn as sns

    for i in range(len(distances)):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(distances[i], annot=True, ax=ax, vmin=0, vmax=1)
        if i == 0:
            title = "Jensen-Shannon distance Albayzin vs Albayzin"
            savename = "js_dist_Albayzin_Albayzin"
        elif i == 1:
            title = "Jensen-Shannon distance Albayzin Healthy vs NeuroVoz Healthy"
            savename = "js_dist_Albayzin_h_Neurovoz_h"
        elif i == 2:
            title = "Jensen-Shannon distance Albayzin Healthy vs NeuroVoz Parkinson"
            savename = "js_dist_Albayzin_h_Neurovoz_pd"
        else:
            title = "Jensen-Shannon distance NeuroVoz Healthy vs NeuroVoz Parkinson"
            savename = "js_dist_Neurovoz_h_Neurovoz_pd"
        ax.set_title(title)
        ax.set_xticklabels(
            [
                "Plosives",
                "Plosives voiced",
                "Nasals",
                "Fricatives",
                "Liquids",
                "Vowels",
                # "Affricates",
                # "Silence",
            ],
            rotation=45,
        )
        ax.set_yticklabels(
            [
                "Plosives",
                "Plosives voiced",
                "Nasals",
                "Fricatives",
                "Liquids",
                "Vowels",
                # "Affricates",
                # "Silence",
            ]
        )
        ax.set_xlabel("Manner classes (Albayzin / Neurovoz)")
        ax.set_ylabel("Manner classes (Albayzin / Neurovoz)")

        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])

        save_path = path_to_plot + "/" + f"{savename}.png"
        fig.savefig(save_path)

        if wandb_flag:
            wandb.log({"test/" + savename: wandb.Image(fig)})

        plt.close()

    plt.close()

    # Plot difference of the distances as a heatmap using seaborn. Make a colorbar where the positive values are red, and the negative are blue.
    # Make the color change smooth.
    # Difference of the distances
    distances_diff = [
        distances_healthy_parkinson - distances_healthy,
    ]
    print("Plotting distances...")
    import seaborn as sns

    for i in range(len(distances_diff)):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            distances_diff[i], annot=True, ax=ax, cmap="RdBu_r", vmax=0.1, vmin=-0.1
        )
        if i == 0:
            title = "Diff. of JSD of Healthy in test vs Parkinsonian in test"
            savename = "diff_healthy_park_test"
        ax.set_title(title)
        ax.set_xticklabels(
            [
                "Plosives",
                "Plosives voiced",
                "Nasals",
                "Fricatives",
                "Liquids",
                "Vowels",
                # "Affricates",
                # "Silence",
            ],
            rotation=45,
        )
        ax.set_yticklabels(
            [
                "Plosives",
                "Plosives voiced",
                "Nasals",
                "Fricatives",
                "Liquids",
                "Vowels",
                # "Affricates",
                # "Silence",
            ]
        )
        ax.set_xlabel("Manner classes (Healthy train clusters)")
        ax.set_ylabel("Manner classes (Test clusters)")
        save_path = path_to_plot + "/" + f"{savename}.png"
        fig.savefig(save_path)
        if wandb_flag:
            wandb.log({"test/" + savename: wandb.Image(fig)})

        plt.close()
    plt.close()


def calculate_euclidean_distances_manner(
    latent_mu_train,
    latent_mu_test,
    manner_train,
    manner_test,
    labels_train,
    labels_test,
    wandb_flag,
    path_to_plot,
):
    from scipy.spatial.distance import euclidean

    print("Calculating distances...")

    # Index each dataset

    # Get unique manner classes
    unique_manner_classes = np.unique(manner_train)

    # Create dictionaries to store distances
    distances_albayzin = np.zeros(
        (len(unique_manner_classes), len(unique_manner_classes))
    )
    distances_healthy = np.zeros(
        (len(unique_manner_classes), len(unique_manner_classes))
    )
    distances_healthy_parkinson = np.zeros(
        (len(unique_manner_classes), len(unique_manner_classes))
    )

    print("Calculating Euclidean distances...")

    for i in range(len(unique_manner_classes)):
        for j in range(len(unique_manner_classes)):
            manner_i = unique_manner_classes[i]
            manner_j = unique_manner_classes[j]

            # Get the indices of data points with manner_i in train and test sets
            indices_albayzin_i = np.where(
                (manner_train == manner_i) & (labels_train == 0)
            )[0]

            # Get the indices of data points with manner_j in train and test sets
            indices_albayzin_j = np.where(
                (manner_train == manner_j) & (labels_train == 0)
            )[0]
            indices_neurovoz_healthy = np.where(
                (manner_test == manner_j) & (labels_test == 0)
            )[0]
            indices_neurovoz_parkinson = np.where(
                (manner_test == manner_j) & (labels_test == 1)
            )[0]

            # Calculate the cluster centroids for manner_i and manner_j
            centroid_albayzin_i = latent_mu_train[indices_albayzin_i].mean(axis=0)
            centroid_albayzin_j = latent_mu_train[indices_albayzin_j].mean(axis=0)
            centroid_neurovoz_healhty = latent_mu_test[indices_neurovoz_healthy].mean(
                axis=0
            )
            centroid_neurovoz_parkinson = latent_mu_test[
                indices_neurovoz_parkinson
            ].mean(axis=0)

            # Calculate Euclidean distance between centroids
            distances_albayzin[i, j] = euclidean(
                centroid_albayzin_i, centroid_albayzin_j
            )
            distances_healthy[i, j] = euclidean(
                centroid_albayzin_i, centroid_neurovoz_healhty
            )
            distances_healthy_parkinson[i, j] = euclidean(
                centroid_albayzin_i, centroid_neurovoz_parkinson
            )

            # If you want to calculate distances for other scenarios (e.g., healthy vs. parkinson),
            # you can add similar calculations here.

    distances = [distances_albayzin, distances_healthy, distances_healthy_parkinson]

    # Plot the distances as a heatmap using seaborn
    print("Plotting distances...")
    import seaborn as sns

    for i in range(len(distances)):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(distances[i], annot=True, ax=ax)
        if i == 0:
            title = "Euclidean distance Albayzin vs Albayzin"
            savename = "eu_dist_Albayzin_Albayzin"
        elif i == 1:
            title = "Euclidean distance Albayzin Healthy vs NeuroVoz Healthy"
            savename = "eu_dist_Albayzin_h_Neurovoz_h"
        else:
            title = "Euclidean distance Albayzin Healthy vs NeuroVoz Parkinson"
            savename = "eu_dist_Albayzin_h_Neurovoz_pd"
        ax.set_title(title)
        ax.set_xticklabels(
            [
                "Plosives",
                "Plosives voiced",
                "Nasals",
                "Fricatives",
                "Liquids",
                "Vowels",
                # "Affricates",
                # "Silence",
            ],
            rotation=45,
        )
        ax.set_yticklabels(
            [
                "Plosives",
                "Plosives voiced",
                "Nasals",
                "Fricatives",
                "Liquids",
                "Vowels",
                # "Affricates",
                # "Silence",
            ]
        )
        ax.set_xlabel("Manner classes (Albayzin / Neurovoz)")
        ax.set_ylabel("Manner classes (Albayzin / Neurovoz)")
        save_path = path_to_plot + "/" + f"{savename}.png"
        fig.savefig(save_path)

        if wandb_flag:
            wandb.log({"test/" + savename: wandb.Image(fig)})

        plt.close()
    plt.close()


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
    # try to assert, if fail convert ohe to labels
    if not Y_pred.size == Y.size:
        Y_pred = np.argmax(Y_pred, axis=1)  # convert to ohe
    if len(Y.shape) == 2:
        # flatten Y
        Y = Y.reshape(-1)
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return sum([w[row[i], col[i]] for i in range(row.shape[0])]) * 1.0 / Y_pred.size


def nmi(Y_pred, Y):
    from sklearn.metrics.cluster import normalized_mutual_info_score

    Y_pred, Y = np.array(Y_pred), np.array(Y)
    # try to assert, if fail convert ohe to labels
    if not Y_pred.size == Y.size:
        Y_pred = np.argmax(Y_pred, axis=1)  # convert to ohe
    if len(Y.shape) == 2:
        # flatten Y
        Y = Y.reshape(-1)
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
