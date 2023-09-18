import pandas as pd
import os
import librosa

# Thiss code gens the necessary files for FASE aligner


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

    # Add a new column with the condition
    df["condition"] = condition

    # Concat the two dataframes
    if condition == "HC":
        df_all = df
    else:
        df_all = pd.concat([df_all, df])


# Create a local folder under /labeled/ named NeuroVoz
os.makedirs("../labeled/NeuroVoz", exist_ok=True)

# Copy all wavs to the new folder and rename it as utt_id_condition.wav
for index, row in df_all.iterrows():
    os.system(
        "cp "
        + row["path_to_wav"]
        + " ./labeled/NeuroVoz/"
        + row["utt_id"]
        + "_"
        + row["condition"]
        + ".wav"
    )
    # Generate a .txt file with the text named equally
    with open(
        "../labeled/NeuroVoz/" + row["utt_id"] + "_" + row["condition"] + ".txt", "w"
    ) as f:
        f.write(row["text"])
