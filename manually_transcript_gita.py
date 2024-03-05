import os

# Define the path to the directory containing the .wav files
directory_path = "/media/my_ftp/BasesDeDatos_Voz_Habla/BD Parkinson Spanish Full/Recordings/sentences/Resample"

# Define the path for the output txt file where filenames will be stored
output_txt_path = "wav_filenames.txt"

# Initialize an empty list to hold the filenames
wav_filenames = []

# Walk through the directory and subdirectories
for root, dirs, files in os.walk(directory_path):
    for file in files:
        # Check if the file is a .wav file
        if file.endswith(".wav"):
            # Add the full path of the .wav file to the list
            wav_filenames.append(file)

# Sort filenames just based on what is after "_" in the filename
wav_filenames.sort(key=lambda x: x.split("_")[1])

# Write the filenames to the output txt file, one per line
with open(output_txt_path, "w") as file:
    for filename in wav_filenames:
        # if last word after "_" is laura, the phrase is "laura sube al tren que pasa"
        if filename.split("_")[-1] == "laura.wav":
            file.write(filename + "- laura sube al tren que pasa\n")
        # if last word after "_" is loslibros, the phrase is "los libros nuevos no caben en la mesa de la oficina"
        elif filename.split("_")[-1] == "loslibros.wav":
            file.write(
                filename + "- los libros nuevos no caben en la mesa de la oficina\n"
            )
        # if last word after "_" is luisa, the phrase is "luisa rey compra el colchón duro que tanto le gusta"
        elif filename.split("_")[-1] == "luisa.wav":
            file.write(
                filename + "- luisa rey compra el colchón duro que tanto le gusta\n"
            )
        # if last word after "_" is micasa, the phrase is "Mi casa tiene tres cuartos"
        elif filename.split("_")[-1] == "micasa.wav":
            file.write(filename + "- Mi casa tiene tres cuartos\n")
        # if last word after "_" is omar, the phrase is "Omar, que vive cerca, trajo miel"
        elif filename.split("_")[-1] == "omar.wav":
            file.write(filename + "- Omar, que vive cerca, trajo miel\n")
        # if last word after "_" is rosita, the phrase is "Rosita Niño, que pinta bien, donó sus cuadros ayer"
        elif filename.split("_")[-1] == "rosita.wav":
            file.write(
                filename + "- Rosita Niño, que pinta bien, donó sus cuadros ayer\n"
            )
print(f"Done writing filenames to {output_txt_path}")


# read the excel which has the manual transcriptions if the patient didnt pronounce the phrase correctly
import pandas as pd

df = pd.read_excel("manual_transcriptions.xlsx")

# The fisrt column is the ID. If the id is less than 1000, it is a patient with PD, then, add to the number "AVPEPUDEA00" to the filename to match the filename in the txt file
df["ID"] = df["Unnamed: 0"].apply(lambda x: "AVPEPUDEA00" + str(x) if x < 1000 else x)
# If the id is more than 1000, it is a control patient, add "AVPEPUDEAC00" and the last two numbers of the id
df["ID"] = df["Unnamed: 0"].apply(
    lambda x: "AVPEPUDEAC00" + str(x)[-2:] if x > 1000 else x
)
