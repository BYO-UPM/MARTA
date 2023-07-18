from scipy.cluster import hierarchy
from data_loaders.pt_data_loader_audiofeatures import Dataset_AudioFeatures
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


hyperparams = {
    "frame_size_ms": 40,
    "material": "VOWELS",
    "hop_size_percent": 0.5,
    "n_plps": 13,
    "n_mfccs": 0,
    "wandb_flag": False,
    "batch_size": 32,
    "data_path": "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2",
}


dataset = Dataset_AudioFeatures(
    hyperparams["data_path"], hyperparams, hyperparams["material"]
)

# Generate a new label named label_complete which is the combination of label and vowel
dataset.data["label_complete"] = dataset.data["label"] * 5 + dataset.data["vowel"]


plps = np.vstack(dataset.data["plps"])
condition = dataset.data["label"]
vowels = dataset.data["vowel"]
patient = dataset.data["id_patient"]
n_patients = len(np.unique(patient))
labels = dataset.data["label_complete"]


# To reduce the number of samples, lets compute the mean plp for each vowel for each patient
# Using groupby
import pandas as pd

df = pd.DataFrame(plps)
df["condition"] = condition.values
df["vowel"] = vowels.values
df["patient"] = patient.values
df["label"] = labels.values

df_grouped = df.groupby(["patient", "vowel"]).mean()
df_grouped = df_grouped.reset_index()

X = df_grouped.drop(["vowel", "label", "condition"], axis=1)
y = df_grouped[["patient", "label"]]

# The first 80% of the patients will be used for training and the last 20% for testing
n_train = int(0.8 * n_patients)
train_patients = np.unique(y["patient"])[:n_train]
test_patients = np.unique(y["patient"])[n_train:]

# Split the data
X_train = X[y["patient"].isin(train_patients)]
y_train = y[y["patient"].isin(train_patients)]
X_test = X[y["patient"].isin(test_patients)]
y_test = y[y["patient"].isin(test_patients)]

# remove patient column
X_train = X_train.drop(["patient"], axis=1)
X_test = X_test.drop(["patient"], axis=1)

# Assert that a patient is not in both train and test
assert len(set(X_train.index).intersection(set(X_test.index))) == 0


# Standarize
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Using Agglomerative Clusering from sklearn to cluster the data
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=10, compute_distances=True).fit(X_train)

# Plot the histogram of the clusters for each label

df_train = pd.DataFrame(X_train)
df_train["cluster"] = clustering.labels_
df_train["label"] = y_train["label"].values
df_train["patient"] = y_train["patient"].values

# Plot the histogram of the label for each cluster
for cluster in np.unique(clustering.labels_):
    df_cluster = df_train[df_train["cluster"] == cluster]
    plt.hist(df_cluster["label"], bins=10)
    plt.title(f"Cluster {cluster}")
    plt.show()


   


plot_dendrogram(clustering, truncate_mode="level", p=2)

# Compute the accuracy
from sklearn.metrics import accuracy_score

accuracy_score(labels_train, clustering.labels_)
