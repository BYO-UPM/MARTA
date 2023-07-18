import torch
import copy
from models.pt_models import VQVAE
from training.pt_training import VQVAE_trainer, VQVAE_tester
from utils.utils import plot_latent_space, plot_latent_space_vowels
from data_loaders.pt_data_loader_audiofeatures import Dataset_AudioFeatures
import torch
import wandb
import numpy as np
import pandas as pd


# Define hyperparams for debugger
hyperparams = {
    "frame_size_ms": 40,
    "material": "VOWELS",
    "hop_size_percent": 0.5,
    "n_plps": 13,
    "n_mfccs": 0,
    "wandb_flag": False,
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.001,
    "latent_dim": 16,
    "K": 10,
    "hidden_dims_enc": [128, 256],
    "hidden_dims_dec": [256, 128],
    "supervised": True,
}
data_path = "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2"

print("Reading data...")
# Read the data
dataset = Dataset_AudioFeatures(data_path, hyperparams, hyperparams["material"])

fold = 0

(
    train_loader,
    val_loader,
    test_loader,
    train_data,
    val_data,
    test_data,
) = dataset.get_dataloaders(fold)


# Define a GaussianMixture model of sklearn
from sklearn.mixture import GaussianMixture

K_list = [2, 5, 10]
bic_score = []

for K in K_list:
    print("K = ", K)
    # Define the model
    model = GaussianMixture(n_components=K, covariance_type="diag")

    # Fit the model
    model.fit(np.vstack(train_data["plps"]))

    # Calcualte bic 
    bic_score.append(model.bic(np.vstack(train_data["plps"])))

    # For k=2, check if the model is able to separate the labels (PD and Healty)
    if K == 2:
        # Get the labels
        labels = np.array(train_data["label"])

        # Get the probabilities
        predictions = model.predict(np.vstack(train_data["plps"]))

        # Plot the latent space
        plt.figure()
        plt.scatter(predictions, labels)
        plt.xlabel("Predictions")
        plt.ylabel("Labels")
        plt.title("K = " + str(K))
        plt.show()

    # for K=5, check if the model is able to separate the vowels
    if K == 5:
        # Get the labels
        labels = np.array(train_data["vowel"])

        # Get the probabilities
        predictions = model.predict(np.vstack(train_data["plps"]))

        # Plot the latent space
        plt.figure()
        plt.scatter(predictions, labels)
        plt.xlabel("Predictions")
        plt.ylabel("Labels")
        plt.title("K = " + str(K))
        plt.show()






    
