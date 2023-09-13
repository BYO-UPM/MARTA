import pandas as pd
import os
import librosa

# This code is used to generate the necessary files for KALDI forced alignment


def find_wav_path(text, condition="HC"):
    folder = text.split("_")[1].split(" ")[0]
    patientid = text.split(condition)[1].split("_")[0]
    path_to_wav = (
        path_to_all_audio
        + "/"
        + folder
        + "/"
        + condition
        + "/"
        + condition
        + "_"
        + folder
        + "_"
        + patientid
        + ".wav"
    )

    # Assert that the wav exists
    try:
        open(path_to_wav)
    except FileNotFoundError:
        print("File not found: ", path_to_wav)
        return None
    return path_to_wav


for condition in ["HC", "PD"]:
    print("Processing: ", condition)

    # Read text in dat
    path = "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/text" + condition + ".dat"

    df = pd.read_csv(
        path,
        encoding="latin1",
        header=None,
    )

    # Substitute "-" for ""
    df = df.replace("-", "", regex=True)

    # Subtitute "  " for " "
    df = df.replace("  ", " ", regex=True)

    # Generate utt_id column by removing the text after the first space
    df["utt_id"] = df[0].apply(lambda x: x.split(" ")[0])
    df["utt_id"] = df["utt_id"].apply(
        lambda x: x.split(condition)[1].split("_")[0] + "_" + x.split(condition)[1]
    )

    # utt2spk
    df["spkr_id"] = df["utt_id"].apply(lambda x: x.split("_")[0])

    # Modify utt_id so all of them starts by spkr_id

    # Generate the text column by removing the utt_id
    df["text"] = df[0].apply(lambda x: x.split(" ", 1)[1])

    # Find the path to each audio file
    path_to_all_audio = (
        "/media/my_ftp/BasesDeDatos_Voz_Habla/Neurovoz/PorMaterial_limpios1_2"
    )

    # Generate the path to wav column
    df["path_to_wav"] = df[0].apply(find_wav_path, condition=condition)

    # Remove nones
    df = df.dropna()

    # For each audio file, detect when the speech starts and end in seconds
    # and save it in the segments file
    # Save the segments file as a file named "segments" with the format:
    # utt_id wav_id start end

    # Concat the two dataframes
    if condition == "HC":
        df_all = df
    else:
        df_all = pd.concat([df_all, df])


# Get sampling rate
df_all["sampling_rate"] = df_all["path_to_wav"].apply(
    lambda x: librosa.get_samplerate(path=x)
)


# ============================================================================== #
# Drop the first column
df_all = df_all.drop(columns=[0])

# Make utt_id the index and sort by it
df_all = df_all.set_index("utt_id").sort_index()

# TEXT FILE
# Save text file as a file named "text" with the format:
# utt_id text
df_all[["text"]].to_csv("text", sep=" ", index=True, header=False)

# Generate words.txt file with all single words in the corpus
words = df_all["text"].str.split(" ", expand=True).stack().unique()
words = pd.DataFrame(words, columns=["word"])
words.to_csv("words.txt", sep=" ", index=False, header=False)

# SEGMENTS FILE
# Save segments file as a file named "segments" with the format:
# utt_id wav_id start end
# uttd_id and wav_id is the same and make start 0 and end the duration of the audio
df_all["wav_id"] = df_all.index
df_all["start"] = 0
df_all["end"] = df_all["path_to_wav"].apply(lambda x: librosa.get_duration(path=x))
df_all[["wav_id", "start", "end"]].to_csv("segments", sep=" ", index=True, header=False)

# WAV FILE
# Save wav.scp file as a file named "wav.scp" with the format:
# utt_id path_to_wav
df_all[["path_to_wav"]].to_csv("wav.scp", sep=" ", index=True, header=False)

# UTT2SPK FILE
# Save utt2spk file as a file named "utt2spk" with the format:
# utt_id spkr_id
df_all[["spkr_id"]].to_csv("utt2spk", sep=" ", index=True, header=False)

# SPK2UTT FILE
# Save spk2utt file as a file named "spk2utt" with the format:
# spkr_id utt_id1 utt_id2 ...
# Keep in mind that utt is the index
df_all.groupby("spkr_id").apply(lambda x: " ".join(x.index.tolist())).to_csv(
    "spk2utt", sep=" ", index=True, header=False
)
