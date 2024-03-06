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
df["ID"] = df["Unnamed: 0"].apply(
    lambda x: (
        "AVPEPUDEA" + str(x).zfill(4)
        if x < 1000
        else ("AVPEPUDEAC" + str(x)[-2:].zfill(4) if x >= 1000 else x)
    )
)


# Rename LauraNorm column to "laura"
df.rename(columns={"LauraNorm": "laura"}, inplace=True)
# Rename LosLibros norm column to "loslibros"
df.rename(columns={"LosLibrosNorm": "loslibros"}, inplace=True)
# Rename LuisaNorm column to "luisa"
df.rename(columns={"LuisaNorm": "luisa"}, inplace=True)
# Rename MiCasaNorm column to "micasa"
df.rename(columns={"MiCasaNorm": "micasa"}, inplace=True)
# Rename OmarNorm column to "omar"
df.rename(columns={"OmarNorm": "omar"}, inplace=True)
# Rename RositaNorm column to "rosita"
df.rename(columns={"RositaNorm": "rosita"}, inplace=True)


# For these columns, check for every filename if the cell is Na.
# If it is not NA, replace the old transcription with the new one in the wav_filenames.txt
for column in df.columns[2:]:
    for index, row in df.iterrows():
        if not pd.isna(row[column]):
            with open(output_txt_path, "r") as file:
                lines = file.readlines()
            for i in range(len(lines)):
                if row["ID"] == lines[i].split("_")[0] and column in lines[i]:
                    lines[i] = row["ID"] + "_" + column + ".wav- " + row[column] + "\n"
                    break
            with open(output_txt_path, "w") as file:
                file.writelines(lines)

print("Done writing manual transcriptions to wav_filenames.txt")
