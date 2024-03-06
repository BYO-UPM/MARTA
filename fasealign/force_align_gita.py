"""
Author: Alejandro Guerrero-López

This Python script is designed for extracting text from TextGrid files (manual segmented from gita dataset), generating a final dataset
of pairs containing .txt and .wav files, and performing forced alignment using the FaseAlign tool.
The primary goal is to streamline the process of preparing and aligning audio and transcription data
for further linguistic analysis and research.

The script consists of three main functions:

1. extract_text_from_textgrids: Extracts text from TextGrid files and saves them as .txt files.
2. generate_final_dataset: Organizes pairs of .txt and .wav files and copies them to a new directory.
3. forced_alignment: Performs forced alignment on the pairs using the FaseAlign tool.

For more information on how to use this script and the FaseAlign tool, please refer to the respective function
documentation and the FaseAlign documentation: https://fasealign.readthedocs.io/en/latest/aligning.html
"""

import os


def generate_final_dataset(path_wav: str) -> None:
    """
    Find pairs of .txt and .wav files with the same name and copy them to a new folder named "gita_htk_forced_alignment."

    Args:
        path_texts (str): The path to the directory containing the .txt files
        path_wavs (str): The path to the directory containing the .wav files

    Returns:
        None
    """
    # Make the directory if it does not exist
    if not os.path.exists("./gita_htk_forced_alignment"):
        os.makedirs("./gita_htk_forced_alignment")

    pairs = []
    # First, find all .wav files under the directory using walk
    for root, dirs, files in os.walk(path_wav):
        for file in files:
            if file.endswith(".wav"):
                # Copy the files to the new directory
                # Replace spaces from root for "\ " to avoid problems with the command line
                source_path = os.path.join(root.replace(" ", "\ "), file)
                destination_path = os.path.join("./gita_htk_forced_alignment", file)
                os.system(f"cp {source_path} {destination_path}")

                # Generate a txt file with the same name
                txt_name = file.replace(".wav", ".txt")
                # Find the filenname in wav_filenames.txt and copy the text to the new directory
                for line in open("wav_filenames.txt", "r"):
                    if file in line:
                        with open(f"./gita_htk_forced_alignment/{txt_name}", "w") as f:
                            f.write(line.split("-")[1])


def forced_alignment() -> None:
    """
    Performs forced alignment using the FaseAlign tool on .wav and .txt files in the "gita_htk_forced_alignment" directory.

    This script works by calling "faseAlign" from the command line with the following arguments:
    -w <path to the wav files>
    -t <path to the text files>
    -n <path to the output directory>
    -p (optional) if there are missing words in the dictionary

    For more information, refer to the FaseAlign documentation: https://fasealign.readthedocs.io/en/latest/aligning.html

    Returns:
        None
    """
    # Read all .txt files in 'gita_htk_forced_alignment' directory
    path_to_text = [
        os.path.join("./gita_htk_forced_alignment", f)
        for f in os.listdir("./gita_htk_forced_alignment")
        if f.endswith(".txt")
    ]

    # Substitute .txt for .wav
    path_to_wav = [path.replace(".txt", ".wav") for path in path_to_text]

    # Substitute the .txt for .TextGrid to generate the path to output
    path_to_output = [path.replace(".txt", ".TextGrid") for path in path_to_text]

    for wav, text, output in zip(path_to_wav, path_to_text, path_to_output):
        os.system(f"faseAlign -w {wav} -t {text} -n {output} -p")


if __name__ == "__main__":
    path_wav = "/media/my_ftp/BasesDeDatos_Voz_Habla/BD Parkinson Spanish Full/Recordings/sentences/Resample"
    print("Generating the final dataset...")
    generate_final_dataset(path_wav)
    print("Performing forced alignment...")
    forced_alignment()
