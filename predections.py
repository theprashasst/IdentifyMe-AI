from tensorflow.keras.models import load_model
from Module.Record import record
from Module.env import inference
from Module.Results import correctly_identify, could_not_identify
import numpy as np
import pyaudio
import json


#loading the names

with open("names_identified.json", 'r') as f:
    loaded_json = json.load(f)
identified_names={int(k):v for k,v in loaded_json.items()}

#loading the updated model

model=load_model("updated_model.keras")



#testing the model

audio = pyaudio.PyAudio()
frames=record(audio=audio)

test=inference(r"E:\New-Codes\Repositories\IdentifyMe-AI\recorded.wav")
predection=model.predict(test.reshape(1,-1))

threshold = 0.6  

# Checking if the highest probability exceeds the threshold
result = np.argmax(predection) if  np.max(predection) > threshold else -1


identified= True if result!=-1 else False

if result==-1:print("Could not identify")
else:print(f"Person identified as {identified_names[result]}")


#if could not identify

if not identified:
    could_not_identify(embeddings=test,audio=audio,frames=frames,names=identified_names)
else:
    correctly_identify(embeddings=test,the_label=result,audio=audio,frames=frames,names=identified_names)


