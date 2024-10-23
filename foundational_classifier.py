import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import argparse

trillsson_layer = hub.KerasLayer(
    "https://tfhub.dev/google/trillsson1/1", trainable=False
)


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
    mean_probabilities = torch.zeros(len(unique_subjects))

    for i, subject in enumerate(unique_subjects):
        subject_indices = np.where(subject_group_test == subject)
        subject_outputs = output_test[subject_indices]

        # Calculate mean probability for the subject
        mean_probabilities[i] = torch.mean(subject_outputs)

        # Store the first label found for the subject
        Y_test_bySubject.append(Y_test[subject_indices][0])

    # Estimate labels based on mean probability
    estimated_labels = torch.zeros_like(mean_probabilities)
    estimated_labels[mean_probabilities >= 0.5] = 1

    Y_test_tensor_bySubject = torch.tensor(Y_test_bySubject, dtype=torch.long)

    return mean_probabilities, Y_test_tensor_bySubject, estimated_labels


def predict_frames(model, test_dataset):
    all_predictions = []
    all_true_labels = []

    for batch_x, batch_y in test_dataset:
        y_pred = model.predict(batch_x)
        all_predictions.extend(y_pred)  # Store predictions in order
        all_true_labels.extend(batch_y)  # Store true labels in order

    return np.array(all_predictions), np.array(all_true_labels)


def evaluate_model(model, test_dataset, test_data):
    # Predict all frame-level results
    y_pred_frames, y_true_frames = predict_frames(model, test_dataset)

    # Compute consensus results
    consensus_soft, consensus_true, consensus_hard = soft_output_by_subject(
        y_pred_frames, y_true_frames, test_data["id_patient"]
    )

    # Step 3: Calculate Accuracy and Balanced Accuracy for threshold 0.5
    acc_hard = accuracy_score(consensus_true, consensus_hard)
    balanced_acc_hard = balanced_accuracy_score(consensus_true, consensus_hard)

    # Step 4: Calculate the best threshold using the soft consensus
    best_threshold = threshold_selection(consensus_true, consensus_soft, verbose=0)
    consensus_best_thresh = np.where(consensus_soft >= best_threshold, 1, 0)

    acc_best = accuracy_score(consensus_true, consensus_best_thresh)
    balanced_acc_best = balanced_accuracy_score(consensus_true, consensus_best_thresh)

    # Step 5: Calculate AUC using the soft consensus
    auc_score = roc_auc_score(consensus_true, consensus_soft)

    # Print the evaluation results
    print(f"Accuracy (Hard, 0.5 threshold): {acc_hard}")
    print(f"Balanced Accuracy (Hard, 0.5 threshold): {balanced_acc_hard}")
    print(f"Accuracy (Best threshold): {acc_best}")
    print(f"Balanced Accuracy (Best threshold): {balanced_acc_best}")
    print(f"AUC (Soft consensus): {auc_score}")

    return {
        "acc_hard_0.5": acc_hard,
        "balanced_acc_hard_0.5": balanced_acc_hard,
        "acc_best_threshold": acc_best,
        "balanced_acc_best_threshold": balanced_acc_best,
        "auc_soft": auc_score,
    }


# Define the MLP model
def create_model():
    model = models.Sequential()

    # Input layer (assuming input shape is [None, frame_length])
    model.add(
        layers.Input(shape=(None,))
    )  # You can modify the shape based on your framed signal shape

    # TRILLsson layer to extract embeddings (output shape: [None, 1024])
    model.add(trillsson_layer)

    # MLP layers
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(16, activation="relu"))

    # Output layer for binary classification
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def dataframe_to_dataset(df):
    """
    Converts a Pandas DataFrame to a TensorFlow dataset.
    Args:
        df: Pandas DataFrame with 'signal_framed' and 'labels' columns.
    Returns:
        A tf.data.Dataset for training.
    """
    # Convert the DataFrame to two separate lists (signals and labels)
    signal_framed = list(df["signal_framed"].values)
    labels = list(df["labels"].values)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((signal_framed, labels))

    # Shuffle, batch, and prefetch for performance
    dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return dataset


def main(args, hyperparams):
    print("Reading train, val and test loaders from local_results/...")
    data_path = f"local_results/folds/{{}}_data_supervised_True_frame_size_0.4spec_winsize_{hyperparams['spectrogram_win_size']}hopsize_0.5fold{hyperparams['fold']}.pt"

    train_data, val_data, test_data = (
        torch.load(data_path.format(ds)) for ds in ["train", "val", "test"]
    )

    train_dataset = dataframe_to_dataset(train_data)
    val_dataset = dataframe_to_dataset(val_data)
    test_dataset = dataframe_to_dataset(test_data)

    # Create the model
    model = create_model()

    # Print model summary
    model.summary()

    # Fit the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # Evaluate the model: globally
    print("RESULTS GLOBAL")
    evaluate_model(test_dataset=test_dataset, model=model, test_data=test_data)
    print("===================================================")

    # Evaluate the model only for Neurovoz RS: column "dataset" must be "neurovoz" and column "text" must be ESPONTANEA
    test_data_neurovoz = test_data[
        (test_data["dataset"] == "neurovoz") & (test_data["text"] == "ESPONTANEA")
    ]
    test_dataset_neurovoz = dataframe_to_dataset(test_data_neurovoz)
    print("RESULTS NEUROVOZ")
    print("Running Speech:")
    evaluate_model(
        test_dataset=test_dataset_neurovoz, model=model, test_data=test_data_neurovoz
    )

    # Evaluate the model only for Neurovoz TDUs: column "dataset" must be "neurovoz" and column "text" must be different of ESPONTANEA
    test_data_neurovoz = test_data[
        (test_data["dataset"] == "neurovoz") & (test_data["text"] != "ESPONTANEA")
    ]
    test_dataset_neurovoz = dataframe_to_dataset(test_data_neurovoz)
    print("TDUs:")
    evaluate_model(
        test_dataset=test_dataset_neurovoz, model=model, test_data=test_data_neurovoz
    )
    print("===================================================")

    # Evaluate the model only for GITA RS: column "dataset" must be "gita" and column "text" must be Monologo
    test_data_gita = test_data[
        (test_data["dataset"] == "gita") & (test_data["text"] == "Monologo")
    ]
    test_dataset_gita = dataframe_to_dataset(test_data_gita)
    print("RESULTS GITA")
    print("Running Speech:")
    evaluate_model(
        test_dataset=test_dataset_gita, model=model, test_data=test_dataset_gita
    )

    # Evaluate the model only for GITA TDUs: column "dataset" must be "gita" and column "text" must different of Monologo
    test_data_gita = test_data[
        (test_data["dataset"] == "gita") & (test_data["text"] != "Monologo")
    ]
    test_dataset_gita = dataframe_to_dataset(test_data_gita)
    print("RESULTS GITA")
    print("TDUs:")
    evaluate_model(
        test_dataset=test_dataset_gita, model=model, test_data=test_dataset_gita
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script configuration")
    parser.add_argument(
        "--fold", type=int, default=0, help="Fold number for the experiment"
    )
    parser.add_argument(
        "--gpu", type=int, default=2, help="GPU number for the experiment"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=64, help="Latent dimension of the model"
    )
    parser.add_argument(
        "--domain_adversarial", type=int, default=1, help="Use domain adversarial"
    )
    parser.add_argument(
        "--cross_lingual",
        type=str,
        default="testing_gita",
        choices=["multilingual", "testing_gita", "testing_neurovoz", "testing_italian"],
        help="Select one choice of crosslingual scenario",
    )
    args = parser.parse_args()

    hyperparams = {
        # ================ Spectrogram parameters ===================
        "spectrogram": True,  # If true, use spectrogram. If false, use plp (In this study we only use spectrograms)
        "frame_size_ms": 0.400,  # Size of each spectrogram frame
        "spectrogram_win_size": 0.030,  # Window size of each window in the spectrogram
        "hop_size_percent": 0.5,  # Hop size (0.5 means 50%) between each window in the spectrogram
        # ================ GMVAE parameters ===================
        "epochs": 200,  # Number of epochs to train the model (at maximum, we have early stopping)
        "batch_size": 128,  # Batch size
        "lr": 1e-3,  # Learning rate: we use cosine annealing over ADAM optimizer
        "latent_dim": args.latent_dim,  # Latent dimension of the z vector (remember it is also the input to the classifier)
        "n_gaussians": 16,  # Number of gaussians in the GMVAE
        "fold": args.fold,  # Which fold to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
        "gpu": args.gpu,  # Which gpu to use, it is said as an argument to automatize the running for all folds using ./run_parallel.sh
    }

    main(args, hyperparams)
