from data_loaders.pt_data_loader_spectrograms_manner import Dataset_AudioFeatures
import numpy as np
import matplotlib.pyplot as plt

hyperparams = {
    # ================ Spectrogram parameters ===================
    "spectrogram": True,  # If true, use spectrogram. If false, use plp (In this study we only use spectrograms)
    "frame_size_ms": 0.400,  # Size of each spectrogram frame
    "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
    "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
    # ================ GMVAE parameters ===================
    "epochs": 300,  # Number of epochs to train the model (at maximum, we have early stopping)
    "batch_size": 128,  # Batch size
    "lr": 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
    "latent_dim": 32,  # Latent dimension of the z vector (remember it is also the input to the classifier)
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
    "domain_adversarial": 0,  # If true, use domain adversarial model
    "crosslingual": "gita_nv",  # Crosslingual scenario
    # ================ Classifier parameters ===================
    "classifier_type": "mlp",  # classifier architecture (cnn or mlp)-.Their dimensions are hard-coded in pt_models.py (we should fix this)
    "classifier": True,  # If true, train the classifier
    "supervised": True,  # It must be true
    # ================ Training parameters ===================
    "train": True,  # If false, the model should have been trained (you have a .pt file with the model) and you only want to evaluate it
    "train_albayzin": False,  # If true, train with albayzin data. If false, only train with neurovoz data.
    "new_data_partition": False,  # If True, new folds are created. If False, the folds are read from local_results/folds/. IT TAKES A LOT OF TIME TO CREATE THE FOLDS (5-10min aprox).
    "fold": 2,  # Which fold to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
    "gpu": 0,  # Which gpu to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
    # ================ UNUSED PARAMETERS (we should fix this) ===================
    # These parameters are not used at all and they are from all versions of the code, we should fix this.
    "material": "MANNER",  # not used here
    "n_plps": 0,  # Not used here
    "n_mfccs": 0,  # Not used here
    "wandb_flag": False,  # Not used here
    "semisupervised": False,  # Not used here
}


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
) = dataset.get_dataloaders(
    supervised=hyperparams["supervised"],
)


gita_data_test = [data for data in test_loader.dataset if data[3] == "gita"]
gita_data_val = [data for data in val_loader.dataset if data[3] == "gita"]
gita_data_val = [(data[0], data[1], data[2], data[3]) for data in gita_data_val]
gita_data_train = [data for data in train_loader.dataset if data[3] == "gita"]

# Get all neurovoz data
neurovoz_data_test = [data for data in test_loader.dataset if data[3] == "neurovoz"]
neurovoz_data_val = [data for data in val_loader.dataset if data[3] == "neurovoz"]
neurovoz_data_val = [(data[0], data[1], data[2], data[3]) for data in neurovoz_data_val]
neurovoz_data_train = [data for data in train_loader.dataset if data[3] == "neurovoz"]


# Define frequency band ranges
low_freq_range = slice(0, 22)  # Bands 1 to 21 (0-indexed)
medium_freq_range = slice(22, 44)  # Bands 22 to 43
high_freq_range = slice(44, 65)  # Bands 44 to 65

# Concatenate all gita data
gita_data = gita_data_train + gita_data_val + gita_data_test
neurovoz_data = neurovoz_data_train + neurovoz_data_val + neurovoz_data_test


# Initialize dictionaries to hold the histograms for both datasets
low_freq_histograms_gita = {i: [] for i in range(8)}
medium_freq_histograms_gita = {i: [] for i in range(8)}
high_freq_histograms_gita = {i: [] for i in range(8)}

low_freq_histograms_neurovoz = {i: [] for i in range(8)}
medium_freq_histograms_neurovoz = {i: [] for i in range(8)}
high_freq_histograms_neurovoz = {i: [] for i in range(8)}

# Process each spectrogram in the gita dataset
for spectrogram, _, manner_classes, _ in gita_data:
    manner_classes = [
        manner_class - 8 if manner_class >= 8 else manner_class
        for manner_class in manner_classes
    ]
    for i, manner_class in enumerate(manner_classes):
        column = spectrogram[0, :, i]
        low_freq_histograms_gita[manner_class].append(column[low_freq_range])
        medium_freq_histograms_gita[manner_class].append(column[medium_freq_range])
        high_freq_histograms_gita[manner_class].append(column[high_freq_range])

# Process each spectrogram in the neurovoz dataset
for spectrogram, _, manner_classes, _ in neurovoz_data:
    manner_classes = [
        manner_class - 8 if manner_class >= 8 else manner_class
        for manner_class in manner_classes
    ]
    for i, manner_class in enumerate(manner_classes):
        column = spectrogram[0, :, i]
        low_freq_histograms_neurovoz[manner_class].append(column[low_freq_range])
        medium_freq_histograms_neurovoz[manner_class].append(column[medium_freq_range])
        high_freq_histograms_neurovoz[manner_class].append(column[high_freq_range])

# Convert lists to numpy arrays for histogram calculation
for manner_class in range(8):
    low_freq_histograms_gita[manner_class] = np.concatenate(
        low_freq_histograms_gita[manner_class]
    )
    medium_freq_histograms_gita[manner_class] = np.concatenate(
        medium_freq_histograms_gita[manner_class]
    )
    high_freq_histograms_gita[manner_class] = np.concatenate(
        high_freq_histograms_gita[manner_class]
    )

    low_freq_histograms_neurovoz[manner_class] = np.concatenate(
        low_freq_histograms_neurovoz[manner_class]
    )
    medium_freq_histograms_neurovoz[manner_class] = np.concatenate(
        medium_freq_histograms_neurovoz[manner_class]
    )
    high_freq_histograms_neurovoz[manner_class] = np.concatenate(
        high_freq_histograms_neurovoz[manner_class]
    )

# Define manner class names
manner_class_names = [
    "Plosives",
    "Voiced Plosive",
    "Nasals",
    "Fricative",
    "Liquid",
    "Vowel",
    "Affricate",
    "Silences",
]


# Plot histograms
def plot_histograms(histograms_gita, histograms_neurovoz, title):
    plt.figure(figsize=(20, 10))
    for manner_class in range(8):
        plt.subplot(4, 2, manner_class + 1)
        plt.hist(histograms_gita[manner_class], bins=30, alpha=0.5, label="Gita")
        plt.hist(
            histograms_neurovoz[manner_class], bins=30, alpha=0.5, label="Neurovoz"
        )
        plt.title(f"{manner_class_names[manner_class]}")
        plt.legend()
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Plot the histograms for low, medium, and high frequency bands
plot_histograms(
    low_freq_histograms_gita,
    low_freq_histograms_neurovoz,
    "Low Frequency Mel Bands (1-22)",
)
plot_histograms(
    medium_freq_histograms_gita,
    medium_freq_histograms_neurovoz,
    "Medium Frequency Mel Bands (23-43)",
)
plot_histograms(
    high_freq_histograms_gita,
    high_freq_histograms_neurovoz,
    "High Frequency Mel Bands (44-65)",
)
