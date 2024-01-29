"""
Author: Alejandro Guerrero-López

This Python script is designed for extracting text from TextGrid files (manual segmented from Albayzin dataset), generating a final dataset
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
import textgrids
import os

def extract_text_from_textgrids(path_texts: str) -> None:
    """
    Extracts the text from TextGrid files in the specified directory and saves them as .txt files.

    Args:
        path_texts (str): The path to the directory containing the TextGrid files.

    Returns:
        None
    """
    # Get all paths to TextGrid files
    paths = [os.path.join(path_texts, f) for f in os.listdir(path_texts) if f.endswith('.TextGrid')]

    # Read the texts from TextGrid files
    texts = [textgrids.TextGrid(path)["ortho"][1].text for path in paths]

    # Store the texts in a file with the same name as the TextGrid files but with .txt extension
    for path, text in zip(paths, texts):
        path = path.replace('.TextGrid', '.txt')
        with open(path, 'w') as f:
            f.write(text)

def generate_final_dataset(path_texts: str) -> None:
    """
    Find pairs of .txt and .wav files with the same name and copy them to a new folder named "albayzin_htk_forced_alignment."

    Args:
        path_texts (str): The path to the directory containing the .txt and .wav files.

    Returns:
        None
    """
    # Make the directory if it does not exist
    if not os.path.exists('./albayzin_htk_forced_alignment'):
        os.makedirs('./albayzin_htk_forced_alignment')
    
    pairs = []
    for file in os.listdir(path_texts):
        if file.endswith('.txt'):
            pairs.append(file.replace('.txt', ''))

    for pair in pairs:
        os.system(f'cp {os.path.join(path_texts, pair + ".txt")} {os.path.join("./albayzin_htk_forced_alignment", pair + ".txt")}')
        os.system(f'cp {os.path.join(path_texts, pair + ".wav")} {os.path.join("./albayzin_htk_forced_alignment", pair + ".wav")}')


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
    # Read all .txt files in 'albayzin_htk_forced_alignment' directory
    path_to_text = [os.path.join('./albayzin_htk_forced_alignment', f) for f in os.listdir('./albayzin_htk_forced_alignment') if f.endswith('.txt')]

    # Substitute .txt for .wav
    path_to_wav = [path.replace('.txt', '.wav') for path in path_to_text]

    # Substitute the .txt for .TextGrid to generate the path to output
    path_to_output = [path.replace('.txt', '.TextGrid') for path in path_to_text]

    for wav, text, output in zip(path_to_wav, path_to_text, path_to_output):
        os.system(f'faseAlign -w {wav} -t {text} -n {output} -p')

if __name__ == '__main__':
    path_texts = os.path.join('.', 'albayzin')
    print("Extracting text from TextGrids...")
    extract_text_from_textgrids(path_texts)
    print("Generating the final dataset...")
    generate_final_dataset(path_texts)
    print("Performing forced alignment...")
    forced_alignment()
