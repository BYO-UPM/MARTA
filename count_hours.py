import os
from pydub import AudioSegment

def count_audio_hours(directory):
    total_seconds = 0

    # Traverse the directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Load the audio file
                audio = AudioSegment.from_wav(file_path)
                # Add its duration to the total (in milliseconds)
                total_seconds += audio.duration_seconds

    # Convert total milliseconds to hours
    total_hours = total_seconds / 3600

    return total_hours

# Replace 'path_to_directory' with the actual directory path
directory_path = '/media/my_ftp/ALBAYZIN/ALBAYZIN/corpora/Albayzin1/CF/albayzin_htk_forced_alignment'
total_hours = count_audio_hours(directory_path)
print(f'Total hours of audio: {total_hours}')
