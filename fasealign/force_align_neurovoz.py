"""
Author: Alejandro Guerrero-López

This Python script is designed for performing forced alignment using the FaseAlign tool.
The primary goal is to streamline the process of preparing and aligning audio and transcription data
for further linguistic analysis and research.

The script consists of three main functions:

1. extract_text_from_textgrids: Extracts text from TextGrid files and saves them as .txt files.
2. generate_final_dataset: Organizes pairs of .txt and .wav files and copies them to a new directory.
3. forced_alignment: Performs forced alignment on the pairs using the FaseAlign tool.

For more information on how to use this script and the FaseAlign tool, please refer to the respective function
documentation and the FaseAlign documentation: https://fasealign.readthedocs.io/en/latest/aligning.html
"""

import textgrids
import os


def forced_alignment() -> None:
    """
    Performs forced alignment using the FaseAlign tool on .wav and .txt files in the "albayzin_htk_forced_alignment" directory.

    This script works by calling "faseAlign" from the command line with the following arguments:
    -w <path to the wav files>
    -t <path to the text files>
    -n <path to the output directory>
    -p (optional) if there are missing words in the dictionary

    For more information, refer to the FaseAlign documentation: https://fasealign.readthedocs.io/en/latest/aligning.html

    Returns:
        None
    """
    # Read all .txt files
    path_to_text = "../labeled/NeuroVoz/runningspeech"

    # Substitute .txt for .wav
    path_to_wav = [path.replace(".txt", ".wav") for path in path_to_text]

    # Substitute the .txt for .TextGrid to generate the path to output
    path_to_output = [path.replace(".txt", ".TextGrid") for path in path_to_text]

    for wav, text, output in zip(path_to_wav, path_to_text, path_to_output):
        os.system(f"faseAlign -w {wav} -t {text} -n {output} -p")


if __name__ == "__main__":
    print("Performing forced alignment...")
    forced_alignment()
