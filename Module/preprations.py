import os
import json
import numpy as np
from Module.env import folder_path, inference



#making the X and y for model

def load_dataset(names=dict()):


    final=[]
    y_label=[]
    for labeled_name in os.listdir(folder_path):
        label=int(labeled_name[0])
        name =labeled_name[2:]

        if label not in names:names[label]=name
        
        
        file_path=f"{folder_path}/{labeled_name}/"
        # print(file_path)
        for wav_file in os.listdir(file_path):
            # print(f"{file_path}{wav_file}")
            final.append(inference(f"{file_path}{wav_file}"))
            y_label.append(label)

    X=np.array(final)
    y=np.array(y_label)

    # Saving the labels to a json
    with open('names_identified.json', 'w') as f:
        json.dump(names, f)


    np.save("X_embeddings.npy",X)
    np.save("y_labels.npy",y)
    return X,y,names
