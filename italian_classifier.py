"""
Unsupervised GMVAE (MARTA) for Parkinson's Disease Analysis

This script implements an unsupervised Gaussian Mixture Variational Autoencoder (GMVAE), 
named MARTA, for analyzing speech features related to Parkinson's Disease. The model is 
trained exclusively on healthy patients and then used to evaluate the latent space 
distances between healthy and Parkinsonian patients (who were never seen by the model 
during training). This approach aims to identify distinct patterns in speech that could 
be indicative of Parkinson's Disease.

Main Components:
1. Data Preparation: Uses 'Dataset_AudioFeatures' for loading and preprocessing spectrogram data.
2. Model Definition: Sets up the MARTA model with specified hyperparameters.
3. Training: Trains the MARTA model in an unsupervised manner using healthy patient data.
4. Evaluation: Evaluates the model by computing distances in the latent space between healthy 
   and Parkinsonian spectrograms.
5. Visualization: Plots and analyzes the results to understand the model's performance.

Usage:
- The script is configured for running in environments with CUDA-capable devices.
- Hyperparameters for the model and training process are adjustable.

Key Functions:
- MARTA_trainer: Handles the training loop of the MARTA model.
- MARTA_tester: Manages the testing and evaluation of the model.
- plot_logopeda_alb_neuro: Visualization utility for plotting model outputs and analysis.

Output:
- Trained GMVAE model capable of distinguishing between healthy and Parkinsonian speech features.
- Log files and performance metrics for model training and testing.
- Visualizations of latent space representations and distances.

Requirements:
- Torch, pandas, and other dependencies for model building and data handling.
- Properly structured and preprocessed data in expected formats.

Author: Guerrero-López, Alejandro 
Date: 25/01/2024

Note:
- The script assumes a specific structure and format for input data.
- The hyperparameters and model configurations may need tuning based on the specific characteristics 
  of the input data and the computational resources available.
"""

from models.pt_models import MARTA
from training.pt_training import MARTA_trainer, MARTA_tester
from utils.utils import plot_logopeda_alb_neuro, stratify_per_dataset
from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import torch
import wandb
import pandas as pd
import sys
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

# Select the free GPU if there is one available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


def main(args, hyperparams):

    hyperparams["path_to_save"] = (
        "local_results/spectrograms/marta_"
        + str(hyperparams["latent_dim"])
        + "unsupervised"
        + "_latentdim_"
        + str(hyperparams["latent_dim"])
        + "_gaussians_"
        + str(hyperparams["n_gaussians"])
        + "_experiment_"
        + str(hyperparams["experiment"])
    )

    # Create the path if does not exist
    if not os.path.exists(hyperparams["path_to_save"]):
        os.makedirs(hyperparams["path_to_save"])

    old_stdout = sys.stdout
    log_file = open(hyperparams["path_to_save"] + "/log.txt", "w")
    sys.stdout = log_file

    if hyperparams["new_data_partition"]:
        print("Reading data...")
        # Read the data
        dataset = Dataset_AudioFeatures(
            hyperparams,
        )
        (
            train_loader,
            val_loader,
            test_loader,
            train_data,  # train_data, not used
            val_data,  # val_data, not used
            test_data,
        ) = dataset.get_dataloaders(experiment=hyperparams["experiment"])
    else:
        print("Reading train, val and test loaders from local_results/...")
        train_loader = torch.load(
            "local_results/folds/train_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )
        val_loader = torch.load(
            "local_results/folds/val_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )
        test_loader = torch.load(
            "local_results/folds/test_loader_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )
        test_data = torch.load(
            "local_results/folds/test_data_supervised_False_frame_size_0.4spec_winsize_0.03hopsize_0.5foldsecond_experiment.pt"
        )

    # Modify the dataloaders to have, in train, only healthy patients and move all parkinsonan  to test

    # Define a simple CNN for classification of the spectrograms
    class SimpleCNN(nn.Module):

        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(
                64 * 8 * 3, 128
            )  # Adjust based on input size after convolutions
            self.fc2 = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.pool(nn.ReLU()(self.conv1(x)))
            x = self.pool(nn.ReLU()(self.conv2(x)))
            x = self.pool(nn.ReLU()(self.conv3(x)))
            x = x.view(-1, 64 * 8 * 3)  # Flatten
            x = nn.ReLU()(self.fc1(x))
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for data in tqdm(trainloader):
                (
                    inputs,
                    labels,
                    _,
                    _,
                ) = data
                inputs, labels = inputs.float().to(device), labels.float().to(
                    device
                )  # Adjusting the shape and type
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

        print("Training complete")

    def test_model(model, testloader):
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data in testloader:
                (
                    inputs,
                    labels,
                    _,
                    _,
                ) = data
                inputs = inputs.float().to(device)
                outputs = model(inputs)
                probs = outputs.squeeze().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_labels.extend(labels.numpy())
                all_preds.extend(preds)
                all_probs.extend(probs)

        accuracy = accuracy_score(all_labels, all_preds)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        print(f"Accuracy: {accuracy}")
        print(f"Balanced Accuracy: {balanced_acc}")
        print(f"AUC: {auc}")

    # Example of how to use the model
    model = SimpleCNN()
    print(model)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer)
    test_model(model, test_loader)

    def plot_mean_spectrograms(dataset):
        import matplotlib.pyplot as plt

        parkinsonian_spectrograms = []
        healthy_spectrograms = []

        for data, label, _, _ in dataset:
            for spec, label in zip(data, label):
                if label == 0:
                    healthy_spectrograms.append(spec.numpy().squeeze())
                else:
                    parkinsonian_spectrograms.append(spec.numpy().squeeze())

        parkinsonian_mean = np.mean(parkinsonian_spectrograms, axis=0)
        healthy_mean = np.mean(healthy_spectrograms, axis=0)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title("Mean Parkinsonian Spectrogram")
        plt.imshow(parkinsonian_mean, aspect="auto", origin="lower")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("Mean Healthy Spectrogram")
        plt.imshow(healthy_mean, aspect="auto", origin="lower")
        plt.colorbar()

        plt.tight_layout()
        plt.savefig("mean_spectrograms.png")
        plt.show()

    # Plot mean spectrograms
    plot_mean_spectrograms(test_loader)

    from sklearn.decomposition import PCA

    def plot_pca(dataset):
        import matplotlib.pyplot as plt

        spectrograms = []
        labels = []

        for data, label, _, _ in dataset:
            for spec, label in zip(data, label):
                spectrograms.append(spec.numpy().flatten())
                labels.append(label)

        spectrograms = np.array(spectrograms)
        labels = np.array(labels)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(spectrograms)

        plt.figure(figsize=(8, 8))
        plt.scatter(
            pca_results[labels == 0, 0],
            pca_results[labels == 0, 1],
            label="Healthy",
            alpha=0.5,
        )
        plt.scatter(
            pca_results[labels == 1, 0],
            pca_results[labels == 1, 1],
            label="Parkinsonian",
            alpha=0.5,
        )
        plt.title("PCA of Spectrograms")
        plt.legend()
        plt.savefig("pca_spectrograms.png")
        plt.show()

    # Plot PCA
    plot_pca(test_loader)

    from sklearn.manifold import TSNE

    def plot_tsne_from_batches(dataloader):
        from matplotlib import pyplot as plt

        spectrograms = []
        labels = []

        for data_batch, label_batch, _, _ in dataloader:
            for data, label in zip(data_batch, label_batch):
                spectrograms.append(data.numpy().flatten())
                labels.append(label.numpy())

        spectrograms = np.array(spectrograms)
        labels = np.array(labels)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(spectrograms)

        plt.figure(figsize=(8, 8))
        plt.scatter(
            tsne_results[labels == 0, 0],
            tsne_results[labels == 0, 1],
            label="Healthy",
            alpha=0.5,
        )
        plt.scatter(
            tsne_results[labels == 1, 0],
            tsne_results[labels == 1, 1],
            label="Parkinsonian",
            alpha=0.5,
        )
        plt.title("t-SNE of Spectrograms")
        plt.legend()
        plt.savefig("tsne_spectrograms.png")
        plt.show()

    # Plot t-SNE from batched data
    plot_tsne_from_batches(test_loader)

    sys.stdout = old_stdout
    log_file.close()


if __name__ == "__main__":
    args = {}

    hyperparams = {
        # ================ Spectrogram parameters ===================
        "spectrogram": True,  # If true, use spectrogram.
        "frame_size_ms": 0.400,  # Size of each spectrogram frame
        "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
        "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
        # ================ GMVAE parameters ===================
        "epochs": 500,  # Number of epochs to train the model (at maximum, we have early stopping)
        "batch_size": 128,  # Batch size
        "lr": 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
        "latent_dim": 3,  # Latent dimension of the z vector (remember it is also the input to the classifier)
        "n_gaussians": 16,  # Number of gaussians in the GMVAE
        "hidden_dims_enc": [
            64,
            1024,
            64,
        ],  # Hidden dimensions of encoder/decoder (from audio framed to spectrogram and viceversa)
        "hidden_dims_gmvae": [256],  # Hidden dimensions of the GMVAE encoder/decoder
        "weights": [  # Weights for the different losses
            1,  # w1 is rec loss,
            1,  # w2 is gaussian kl loss,
            1,  # w3 is categorical kl loss,
            10,  # w5 is metric loss
        ],
        # ================ Experiment parameters ===================
        "experiment": "fourth",  # Experiment name, from 1 to 6. It is used to load the correct data.
        # ================ Classifier parameters ===================
        "cnn_classifier": False,  # Here no classifier is used
        "supervised": False,  # Here no classifier is used
        # ================ Training parameters ===================
        "train": True,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
        "new_data_partition": True,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
    }

    main(args, hyperparams)
