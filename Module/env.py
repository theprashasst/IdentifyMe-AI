import pyaudio
from pyannote.audio import Inference
from pyannote.audio import Model

folder_path = "E:/New-Codes/Repositories/IdentifyMe-AI/VoiceData/data" 


# Define parameters
FORMAT = pyaudio.paInt16   # Format of audio
CHANNELS = 1               # Number of channels (1 for mono, 2 for stereo)
RATE = 44100               # Sample rate
CHUNK = 1024               # Buffer size
RECORD_SECONDS = 6        # Duration of recording


model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM",cache_dir=r"C:\Users\prash\.cache\huggingface\hub\models--pyannote--wespeaker-voxceleb-resnet34-LM")
inference = Inference(model, window="whole")