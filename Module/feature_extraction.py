import os
import librosa
import numpy as np

#this file will have the feature extraction function
'''
So 
'''



def extract_features(audio_file):
    # Load audio
    audio, sample_rate = librosa.load(audio_file, sr=16000)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    
    # Mean normalize the MFCCs
    mfccs_normalized = mfccs.mean(axis=1)
    
    return mfccs_normalized

def main():
    folder_path = "E:/New-Codes/Repositories/IdentifyMe-AI/VoiceData/P1/" 
    for wav_file in os.listdir(folder_path):


if __name__ == "__main__":
    main()
