import os
from collections import Counter
import textgrids as tg

# STUPID CODE TO COUNT PHONEMES IN A DATASET

datapath = (
    "/media/my_ftp/BasesDeDatos_Voz_Habla/LibriSpeech/LibriSpeech/train-clean-100"
)
phonemes = []

# Walk through the directory and get all .TextGrid files
for root, _, files in os.walk(datapath):
    for file in files:
        if file.endswith(".TextGrid"):
            tg_file_path = os.path.join(root, file)
            tg_file = tg.TextGrid(tg_file_path)
            phonemes.extend([interval.text for interval in tg_file["phones"]])

# Count the unique phonemes
phoneme_counts = Counter(phonemes)
