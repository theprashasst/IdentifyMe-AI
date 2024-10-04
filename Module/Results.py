from Module.train import finetune
from Module.env import folder_path, CHANNELS,FORMAT,RATE,CHUNK,RECORD_SECONDS
import numpy as np
import os
import wave


#if couldnt identify
def could_not_identify(embeddings,audio,frames,names):
    

    que=input(f"Are you a new Speaker?   (y/n)")
    is_new= False if (que=="n" or que=="n") else True
    new_name="Prashasst"
    new_label=0

    if not is_new:
        print(names)
        new_label=int(input("Enter your id : "))
        new_name=names[new_label]
        np.save("X_new_embeddings.npy",embeddings)
        np.save("y_new_labels.npy",new_label)
        finetune(embeddings,np.array(new_label))
    else:
        new_name=input("Enter your correct name: ")
        new_label= list(names.keys())[-1]+1
        new_name_folder_path=f"{folder_path}/{new_label}_{new_name}"
        os.makedirs(new_name_folder_path)

        print(" Retrain the model from scratch ")
    
    

    new_name_folder_path=f"{folder_path}/{new_label}_{new_name}"

    # Saving the recorded data as a WAV file
    with wave.open(f"{new_name_folder_path}/{np.random.randint(100)}_{new_name}.wav" , 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recorded audio saved in {new_label}_{new_name} in voiceData/data folder")


#if identifued correctly
def correctly_identify(embeddings,the_label,audio,frames,names):

    
    que=input(f"Were you identifed correctly? (y/n)")
    correct= True if (que=="y" or que=="Y") else False

    if not correct:
        print(names)
        the_label=int(input("Enter your correct id  or -1 if not here: "))
        if the_label==-1:return
        the_name=names[the_label]


    the_name=names[the_label]

    np.save("X_new_embeddings.npy",embeddings)
    np.save("y_new_labels.npy",the_label)

    finetune(embeddings,np.array(the_label))

    

    the_name_folder_path=f"{folder_path}/{the_label}_{the_name}"

   

    # Saving the recorded data as a WAV file
    with wave.open(f"{the_name_folder_path}/{np.random.randint(100)}_{the_name}.wav" , 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Recorded audio saved in {the_label}_{the_name} in voiceData/data folder")
