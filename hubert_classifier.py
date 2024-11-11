import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
import argparse
import os
from contextlib import redirect_stdout
import torchaudio
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
import pandas as pd


def threshold_selection(y_true, y_pred_soft, verbose=0):
    from sklearn.metrics import roc_curve

    # Select best threshold by youden index
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_soft)
    j_scores = tpr - fpr
    youden_th = thresholds[np.argmax(j_scores)]

    # Select best threshold by EER
    fnr = 1 - tpr
    eer_threshold = thresholds[np.argmin(np.absolute((fnr - fpr)))]

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred_soft)

    return youden_th, eer_threshold, auc


def soft_output_by_subject(output_test, Y_test, subject_group_test):
    unique_subjects = np.unique(subject_group_test)
    Y_test_bySubject = []
    mean_probabilities = np.zeros(len(unique_subjects))
    Y_test = np.array(Y_test)

    for i, subject in enumerate(unique_subjects):
        subject_indices = np.where(subject_group_test == subject)[0]
        subject_outputs = np.array(output_test)[subject_indices]

        # Calculate mean probability for the subject
        mean_probabilities[i] = np.mean(subject_outputs)

        # Assert that Y_test[subject_indices] is the same value for all indices (same patient is always the same)
        assert np.all(Y_test[subject_indices] == Y_test[subject_indices][0])

        # Store the first label found for the subject
        Y_test_bySubject.append(Y_test[subject_indices][0])

    # Estimate labels based on mean probability
    estimated_labels = np.zeros(mean_probabilities.shape)
    estimated_labels[mean_probabilities >= 0.5] = 1

    Y_test_tensor_bySubject = torch.tensor(Y_test_bySubject, dtype=torch.long)

    return mean_probabilities, Y_test_tensor_bySubject, estimated_labels


def predict_frames(model, test_loader):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for embeddings, true_labels in tqdm(
            test_loader, total=len(test_loader), desc="Predicting"
        ):
            embeddings = embeddings.to(device)

            # Forward pass to get logits
            outputs = model(embeddings).squeeze(1)
            soft_preditions = torch.sigmoid(outputs)

            # Move predictions to CPU and append to the list
            all_predictions.extend(soft_preditions.cpu().numpy())

            # Move true labels to CPU and append to the list
            all_true_labels.extend(true_labels.numpy())

    return all_predictions, all_true_labels


def select_threshold_val(model, val_loader, val_data):
    y_pred_frames, y_true_frames = predict_frames(model, val_loader)

    # Compute consensus results
    consensus_soft, consensus_true, consensus_hard = soft_output_by_subject(
        y_pred_frames, y_true_frames, val_data["id_patient"]
    )

    # Select best threshold
    best_threshold, _, _ = threshold_selection(consensus_true, consensus_soft)

    return best_threshold


def evaluate_model(model, test_loader, test_data, best_threshold=0.5):
    # Predict all frame-level results
    y_pred_frames, y_true_frames = predict_frames(model, test_loader)

    # Print shape of predictions
    y_pred_frames = np.array(y_pred_frames)
    y_true_frames = np.array(y_true_frames)

    # Compute consensus results
    consensus_soft, consensus_true, consensus_hard = soft_output_by_subject(
        y_pred_frames, y_true_frames, test_data["id_patient"]
    )

    # Step 3: Calculate Accuracy and Balanced Accuracy for threshold 0.5
    acc_hard = accuracy_score(consensus_true, consensus_hard)
    balanced_acc_hard = balanced_accuracy_score(consensus_true, consensus_hard)

    # Step 4: Calculate the best threshold using the soft consensus
    consensus_best_thresh = np.where(consensus_soft >= best_threshold, 1, 0)

    acc_best = accuracy_score(consensus_true, consensus_best_thresh)
    balanced_acc_best = balanced_accuracy_score(consensus_true, consensus_best_thresh)

    # Step 5: Calculate AUC using the soft consensus
    auc_score = roc_auc_score(consensus_true, consensus_soft)

    # Step 6: compute F1 scores
    f1s = f1_score(consensus_true, consensus_best_thresh)

    # Print the evaluation results
    print("RESULTS FOR BEST THRESHOLD:", best_threshold)
    print(f"Accuracy : {acc_best}")
    print(f"Balanced Accuracy: {balanced_acc_best}")
    print(f"AUC: {auc_score}")
    print(f"F1 Score: {f1s}")

    return {
        "acc_hard_0.5": acc_hard,
        "balanced_acc_hard_0.5": balanced_acc_hard,
        "acc_best_threshold": acc_best,
        "balanced_acc_best_threshold": balanced_acc_best,
        "auc_soft": auc_score,
        "f1_score": f1s,
    }


import argparse


parser = argparse.ArgumentParser(description="Script configuration")
parser.add_argument(
    "--fold", type=int, default=0, help="Fold number for the experiment"
)
parser.add_argument(
    "--experiment",
    type=str,
    default="testing_gita",
    help="Can be either multilingual, testing_gita or testing_neurovoz",
)
parser.add_argument(
    "--gpu", type=int, default=0, help="GPU number to use in the experiment"
)
args = parser.parse_args()

hyperparams = {
    # ================ Spectrogram parameters ===================
    "spectrogram": True,  # If true, use spectrogram. If false, use plp (In this study we only use spectrograms)
    "frame_size_ms": 0.400,  # Size of each spectrogram frame
    "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
    "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
    # ================ GMVAE parameters ===================
    "batch_size": 128,  # Batch size
    "lr": 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
    "latent_dim": 3,  # Latent dimension of the z vector (remember it is also the input to the classifier)
    "n_gaussians": 16,  # Number of gaussians in the GMVAE
    "fold": args.fold,  # Which fold to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
    "gpu": args.gpu,  # Which gpu to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
    # ================ Experiments parameters ===================
    "experiment": args.experiment,  # Experiment name
}


print("Reading train, val and test loaders from local_results/...")
data_path = f"local_results/folds/{{}}_data_supervised_True_frame_size_0.4spec_winsize_{hyperparams['spectrogram_win_size']}hopsize_0.5fold{hyperparams['fold']}.pt"

train_data, val_data, test_data = (
    torch.load(data_path.format(ds)) for ds in ["train", "val", "test"]
)

print("Data loaded, converting to datasets...")

# count total hours of speech per dataset
print("Total hours of speech per dataset:")

# Select gita data from all data
gita_data = pd.concat(
    [
        test_data[test_data["dataset"] == "gita"],
        val_data[val_data["dataset"] == "gita"],
        train_data[train_data["dataset"] == "gita"],
    ]
)
neurovoz = pd.concat(
    [
        test_data[test_data["dataset"] == "neurovoz"],
        val_data[val_data["dataset"] == "neurovoz"],
        train_data[train_data["dataset"] == "neurovoz"],
    ]
)
albayzin = pd.concat(
    [
        test_data[test_data["dataset"] == "albayzin"],
        val_data[val_data["dataset"] == "albayzin"],
        train_data[train_data["dataset"] == "albayzin"],
    ]
)

# calcule duration of each signal framed, the length is df["signal_framed"].shape[1] and the sr is df["sr"]
gita_data["duration"] = gita_data["signal_framed"].apply(
    lambda x: x.shape[0] / gita_data["sr"].values[0]
)
neurovoz["duration"] = neurovoz["signal_framed"].apply(
    lambda x: x.shape[0] / neurovoz["sr"].values[0]
)
albayzin["duration"] = albayzin["signal_framed"].apply(
    lambda x: x.shape[0] / albayzin["sr"].values[0]
)

# Print total hours of speech per dataset
print("GITA:", gita_data["duration"].sum() / 3600)
print("NEUROVOZ:", neurovoz["duration"].sum() / 3600)
print("ALBAYZIN:", albayzin["duration"].sum() / 3600)

if hyperparams["experiment"] == "multilingual":
    print("Running multilingual experiment")
elif hyperparams["experiment"] == "testing_gita":
    train_data_final = pd.concat(
        [
            train_data[train_data["dataset"] == "neurovoz"],
            train_data[train_data["dataset"] == "albayzin"],
            test_data[test_data["dataset"] == "neurovoz"],
            test_data[test_data["dataset"] == "albayzin"],
        ]
    )
    val_data_final = pd.concat(
        [
            val_data[val_data["dataset"] == "neurovoz"],
            val_data[val_data["dataset"] == "albayzin"],
        ]
    )
    test_data_final = pd.concat(
        [
            train_data[train_data["dataset"] == "gita"],
            val_data[val_data["dataset"] == "gita"],
            test_data[test_data["dataset"] == "gita"],
        ]
    )
    train_data = train_data_final.copy()
    val_data = val_data_final.copy()
    test_data = test_data_final.copy()
elif hyperparams["experiment"] == "testing_neurovoz":
    train_data_final = pd.concat(
        [
            train_data[train_data["dataset"] == "gita"],
            train_data[train_data["dataset"] == "albayzin"],
            test_data[test_data["dataset"] == "gita"],
            test_data[test_data["dataset"] == "albayzin"],
        ]
    )
    val_data_final = pd.concat(
        [
            val_data[val_data["dataset"] == "gita"],
            val_data[val_data["dataset"] == "albayzin"],
        ]
    )
    test_data_final = pd.concat(
        [
            train_data[train_data["dataset"] == "neurovoz"],
            val_data[val_data["dataset"] == "neurovoz"],
            test_data[test_data["dataset"] == "neurovoz"],
        ]
    )
    train_data = train_data_final.copy()
    val_data = val_data_final.copy()
    test_data = test_data_final.copy()


# For each signal framed in the dataset, get features using HUBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.HUBERT_BASE
hubert = bundle.get_model().to(device)

# Load the model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torchaudio.pipelines.HUBERT_BASE
hubert = bundle.get_model().to(device)


# Custom Dataset class to create DataLoader for the dataframe
class SignalDataset(Dataset):
    def __init__(self, data):
        self.signals = data["signal_framed"].values
        self.labels = data["label"].values

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(
            label, dtype=torch.long
        )


def dataframe_to_dataloader(data, shuffle=False):
    batch_size = hyperparams["batch_size"]

    # Create DataLoader for input data
    dataset = SignalDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    embeddings = []
    labels_list = []

    # Iterate over DataLoader in batches
    for signals, labels in tqdm(data_loader, total=len(data_loader)):
        signals = signals.to(device)

        # Extract features in batches
        with torch.no_grad():

            embedding, _ = hubert.extract_features(signals)
            embeddings.append(
                embedding[-1].cpu()
            )  # Move embedding to CPU to save memory
            # assert that embeddings is shape 2
            labels_list.append(labels)

    # Concatenate all embeddings and labels to create final dataset
    embeddings = torch.cat(embeddings, dim=0)
    # Average over shape 1 of embeddings
    embeddings = torch.mean(embeddings, dim=1)
    labels = torch.cat(labels_list, dim=0)

    # Create a DataLoader with the extracted features
    final_dataset = torch.utils.data.TensorDataset(embeddings, labels)
    return torch.utils.data.DataLoader(
        final_dataset, batch_size=batch_size, shuffle=shuffle
    )


train_loader = dataframe_to_dataloader(train_data, shuffle=True)
val_loader = dataframe_to_dataloader(val_data)
test_loader = dataframe_to_dataloader(test_data)


# Define a simple classifier neural network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.classifier(x)


# Assuming the embedding size is obtained from the output of feature extraction
input_size = 768  # Example input size for HuBERT embeddings

# Create the model
model = SimpleClassifier(input_size).to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
optimizer = optim.Adam(model.parameters(), lr=0.001)


path_to_save = "local_results/noninter/hubert_experiment_{}_fold{}.h5".format(
    hyperparams["experiment"],
    hyperparams["fold"],
)

# Asser that path exists
assert os.path.exists(os.path.dirname(path_to_save))


def train_classifier(
    train_loader,
    val_loader,
    model,
    criterion,
    optimizer,
    num_epochs=50,
    patience=10,
    model_save_path="best_model.pth",
):
    model.train()
    best_val_accuracy = 0.0
    epochs_no_improve = 0

    # Early stopping parameters
    early_stopping_triggered = False

    for epoch in range(num_epochs):
        if early_stopping_triggered:
            print("Early stopping triggered. Stopping training.")
            break

        epoch_loss = 0.0
        model.train()  # Set model to training mode

        # Training loop
        for embeddings, labels in tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs} - Training",
        ):
            embeddings, labels = embeddings.to(device), labels.to(device).float()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(embeddings).squeeze(1)  # Output logits
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_loss += loss.item()

        # Validation loop
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():
            for embeddings, labels in tqdm(
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {epoch+1}/{num_epochs} - Validation",
            ):
                embeddings, labels = embeddings.to(device), labels.to(device).float()

                outputs = model(embeddings).squeeze(1)
                predictions = (
                    torch.sigmoid(outputs) > 0.5
                )  # Convert logits to binary predictions
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total

        # Print statistics
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Check if current validation accuracy is the best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation accuracy improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve >= patience:
            print("No improvement for 10 epochs. Triggering early stopping.")
            early_stopping_triggered = True


train = True
if train:
    train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1000,
        patience=10,
        model_save_path=path_to_save,
    )
    # Load the best model
    model.load_state_dict(torch.load(path_to_save))
else:
    model.load_state_dict(torch.load(path_to_save))


# Save all prints to a .txt file
with open(
    "local_results/noninter/hubert_experiment_{}_fold{}.txt".format(
        hyperparams["experiment"],
        hyperparams["fold"],
    ),
    "w",
) as f:
    with redirect_stdout(f):
        # Select best threshold using validation set
        best_threshold = select_threshold_val(model, val_loader, val_data)
        # Evaluate the model: globally
        print("RESULTS GLOBAL")
        evaluate_model(
            test_loader=test_loader,
            model=model,
            test_data=test_data,
            best_threshold=best_threshold,
        )
        print("===================================================")

        if (
            hyperparams["experiment"] == "multilingual"
            or hyperparams["experiment"] == "testing_neurovoz"
        ):
            # Evaluate the model only for Neurovoz RS: column "dataset" must be "neurovoz" and column "text" must be ESPONTANEA
            test_data_neurovoz = test_data[
                (test_data["dataset"] == "neurovoz")
                & (test_data["text"] == "ESPONTANEA")
            ]
            test_loader_neurovoz = dataframe_to_dataloader(test_data_neurovoz)
            val_data_neurovoz = val_data[
                (val_data["dataset"] == "neurovoz") & (val_data["text"] == "ESPONTANEA")
            ]
            print("RESULTS NEUROVOZ")
            print("Running Speech:")
            evaluate_model(
                test_loader=test_loader_neurovoz,
                model=model,
                test_data=test_data_neurovoz,
                best_threshold=best_threshold,
            )

            # Evaluate the model only for Neurovoz TDUs: column "dataset" must be "neurovoz" and column "text" must be different of ESPONTANEA
            test_data_neurovoz = test_data[
                (test_data["dataset"] == "neurovoz")
                & (test_data["text"] != "ESPONTANEA")
            ]
            test_loader_neurovoz = dataframe_to_dataloader(test_data_neurovoz)
            print("TDUs:")
            evaluate_model(
                test_loader=test_loader_neurovoz,
                model=model,
                test_data=test_data_neurovoz,
                best_threshold=best_threshold,
            )
        print("===================================================")
        if (
            hyperparams["experiment"] == "multilingual"
            or hyperparams["experiment"] == "testing_gita"
        ):
            # Evaluate the model only for GITA RS: column "dataset" must be "gita" and column "text" must be Monologo
            test_data_gita = test_data[
                (test_data["dataset"] == "gita") & (test_data["text"] == "Monologo")
            ]
            test_loader_gita = dataframe_to_dataloader(test_data_gita)
            print("RESULTS GITA")
            print("Running Speech:")
            evaluate_model(
                test_loader=test_loader_gita,
                model=model,
                test_data=test_data_gita,
                best_threshold=best_threshold,
            )

            # Evaluate the model only for GITA TDUs: column "dataset" must be "gita" and column "text" must different of Monologo
            test_data_gita = test_data[
                (test_data["dataset"] == "gita") & (test_data["text"] != "Monologo")
            ]
            test_loader_gita = dataframe_to_dataloader(test_data_gita)
            print("RESULTS GITA")
            print("TDUs:")
            evaluate_model(
                test_loader=test_loader_gita,
                model=model,
                test_data=test_data_gita,
                best_threshold=best_threshold,
            )
