import os
import subprocess

def convert_m4a_to_wav(folder_path):
    # Get a list of all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.m4a'):
            # Define the full path for the input .m4a and output .wav files
            old_ext = os.path.join(folder_path, filename)
            wav_file = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.wav")
            
            # Run FFmpeg command to convert .m4a to .wav
            subprocess.run(['ffmpeg', '-i', old_ext, wav_file])
            print(f"Converted: {old_ext} to {wav_file}")

if __name__ == "__main__":
    # Specify the folder containing the .m4a files
    folder_path = "C:\Users\prash\.cache\huggingface\datasets\downloads\extracted\46da4d94bea6a89203c499f996e08c56bfc317143ef9e4a28fee1fee4a595e40"  
    convert_m4a_to_wav(folder_path)
