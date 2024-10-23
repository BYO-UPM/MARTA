import os
import librosa
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import os
from jiwer import wer, cer
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np


# Function to load all .wav files from a parent folder
def load_wav_files(parent_folder):
    print("Loading wav files...")
    wav_files = []
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".wav"):
                if file.endswith("_normalized.wav"):
                    continue
                full_path = os.path.join(root, file)
                audio, sr = librosa.load(full_path, sr=16000)
                wav_files.append((audio, sr, full_path))
    print("Wav files loaded successfully!")
    return wav_files


# Function to chunk audio into 30-second segments
def chunk_audio(audio, sr, chunk_duration=30):
    chunk_length = chunk_duration * sr
    chunks = []
    for i in range(0, len(audio), chunk_length):
        chunk = audio[i : i + chunk_length]
        chunks.append(chunk)
    return chunks


# Function to transcribe audio files using Whisper model
def transcribe_audio_files(wav_files):
    transcriber = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=0 if torch.cuda.is_available() else -1,
    )

    transcriptions = []
    for audio, sr, path in tqdm(wav_files):
        print("Transcribing file:", path)
        chunks = chunk_audio(audio, sr)
        transcription = ""
        for chunk in chunks:
            inputs = {"array": chunk, "sampling_rate": sr, "language": "es-ES"}
            result = transcriber(inputs)
            transcription += result["text"] + " "

        transcription = transcription.strip()

        # Save transcription to a text file with the same name as the audio file
        text_file_path = path.replace(".wav", ".txt")
        with open(text_file_path, "w") as text_file:
            text_file.write(transcription)

        transcriptions.append(transcription)

    return transcriptions


# Function to read .txt files
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def evaluate_transcriptions(true_transcriptions_folder, whisper_transcriptions_folder):

    # Collect the .txt files from both directories
    auto_files = []
    manual_files = []

    for root, _, files in os.walk(whisper_transcriptions_folder):
        for file in files:
            if file.endswith(".txt"):
                auto_files.append(os.path.join(root, file))

    for root, _, files in os.walk(true_transcriptions_folder):
        for file in files:
            if file.endswith(".txt"):
                manual_files.append(os.path.join(root, file))

    # Ensure files match by name
    auto_files.sort()
    manual_files.sort()

    # Do a set of both and check which are the missing files
    auto_set = set([os.path.basename(file) for file in auto_files])
    manual_set = set([os.path.basename(file) for file in manual_files])

    missing_files = auto_set - manual_set

    # Check if both folders have the same number of files
    assert len(auto_files) == len(manual_files), "Mismatch in the number of files."

    # Metrics Calculation
    results = []

    for auto_file, manual_file in zip(auto_files, manual_files):
        auto_text = read_txt(auto_file)
        manual_text = read_txt(manual_file)

        # Word Error Rate (WER)
        wer_value = wer(manual_text, auto_text)

        # Character Error Rate (CER)
        cer_value = cer(manual_text, auto_text)

        # Levenshtein Distance (Edit Distance)
        levenshtein_value = 1 - SequenceMatcher(None, manual_text, auto_text).ratio()

        # BLEU Score
        bleu_value = sentence_bleu([manual_text.split()], auto_text.split())

        # Jaccard Similarity for words
        vectorizer = CountVectorizer().fit_transform([manual_text, auto_text])
        vectors = vectorizer.toarray()
        jaccard_value = jaccard_score(vectors[0], vectors[1], average="macro")

        # Store the results
        results.append(
            {
                "file": os.path.basename(auto_file),
                "WER": wer_value,
                "CER": cer_value,
                "Levenshtein": levenshtein_value,
                "BLEU": bleu_value,
                "Jaccard": jaccard_value,
            }
        )

    # save results to a csv file
    results_df = pd.DataFrame(results)

    # Calculate the mean and std of the metrics for all files that include "ESPONTANEA" in their name
    espontanea_df = results_df[results_df["file"].str.contains("Monologo")]
    espontanea_mean = espontanea_df.mean()
    espontanea_std = espontanea_df.std()
    espontanea_mean["file"] = "RS Mean"
    espontanea_std["file"] = "RS Std"
    results_df = results_df.append(espontanea_mean, ignore_index=True)
    results_df = results_df.append(espontanea_std, ignore_index=True)

    # Calculate the madn and std within ESPONTANEA files if the name contians
    hc_df = espontanea_df[espontanea_df["file"].str.contains("AVPEPUDEAC")]
    # the pd are all that not contain avpepudeac
    pd_df = espontanea_df[~espontanea_df["file"].str.contains("AVPEPUDEAC")]
    pd_mean = pd_df.mean()
    pd_std = pd_df.std()
    pd_mean["file"] = "RS PD Mean"
    pd_std["file"] = "RS PD Std"
    hc_mean = hc_df.mean()
    hc_std = hc_df.std()
    hc_mean["file"] = "RS HC Mean"
    hc_std["file"] = "RS HC Std"
    results_df = results_df.append(pd_mean, ignore_index=True)
    results_df = results_df.append(pd_std, ignore_index=True)
    results_df = results_df.append(hc_mean, ignore_index=True)
    results_df = results_df.append(hc_std, ignore_index=True)

    # Calculate the mean and stf of the metrics for all the other files (not including "ESPONTANEA")
    other_df = results_df[~results_df["file"].str.contains("Monologo")]
    other_mean = other_df.mean()
    other_std = other_df.std()
    other_mean["file"] = "TDU Mean"
    other_std["file"] = "TDU Std"
    results_df = results_df.append(other_mean, ignore_index=True)
    results_df = results_df.append(other_std, ignore_index=True)

    # Calculate the mean and std within the other files if the name contains "PD" or "HC"
    pd_df = other_df[~other_df["file"].str.contains("AVPEPUDEAC")]
    hc_df = other_df[other_df["file"].str.contains("AVPEPUDEAC")]
    pd_mean = pd_df.mean()
    pd_std = pd_df.std()
    pd_mean["file"] = "TDU PD Mean"
    pd_std["file"] = "TDU PD Std"
    hc_mean = hc_df.mean()
    hc_std = hc_df.std()
    hc_mean["file"] = "TDU HC Mean"
    hc_std["file"] = "TDU HC Std"
    results_df = results_df.append(pd_mean, ignore_index=True)
    results_df = results_df.append(pd_std, ignore_index=True)
    results_df = results_df.append(hc_mean, ignore_index=True)
    results_df = results_df.append(hc_std, ignore_index=True)

    # Add a final row with the mean and std of the metrics
    mean_row = results_df.mean()
    std_row = results_df.std()
    mean_row["file"] = "Total Mean"
    std_row["file"] = "Total Std"
    results_df = results_df.append(mean_row, ignore_index=True)
    results_df = results_df.append(std_row, ignore_index=True)

    results_df.to_csv("results.csv", index=False)

    return results_df


def transcript(parent_folder):
    # Load wav files
    wav_files = load_wav_files(parent_folder)

    # Transcribe audio files
    transcriptions = transcribe_audio_files(wav_files)

neurovoz=False

if neurovoz:
    # transcript(
    #     "/media/my_ftp/BasesDeDatos_Voz_Habla/PC-GITA/gita_automatically_transcribed_with_whisperv3turbo/"
    # )

    evaluate_transcriptions(
        true_transcriptions_folder="/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/neurovoz_htk_forced_alignment",
        whisper_transcriptions_folder="/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/neurovoz_automatically_transcribed_with_whisperv3turbo",
    )
else:
    # transcript(
    #     "/media/my_ftp/BasesDeDatos_Voz_Habla/PC-GITA/gita_automatically_transcribed_with_whisperv3turbo/"
    # )

    evaluate_transcriptions(
        true_transcriptions_folder="/media/my_ftp/BasesDeDatos_Voz_Habla/PC-GITA/gita_htk_forced_alignment",
        whisper_transcriptions_folder="/media/my_ftp/BasesDeDatos_Voz_Habla/PC-GITA/gita_automatically_transcribed_with_whisperv3turbo",
    )


