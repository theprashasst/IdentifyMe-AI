import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from sklearn.model_selection import train_test_split
from Module.preprations import load_dataset






#spliting into training and validation
X,y,names=load_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=49)


def finetune(X_new,y_new):

    X_old=np.load("X_embeddings.npy")
    y_old=np.load("y_labels.npy")


    X_finetune = np.concatenate([X_old, X_new.reshape(1,-1)], axis=0)
    y_finetune = np.concatenate([y_old, y_new.reshape(1)], axis=0)

    model=load_model("updated_model.keras")

    history = model.fit(X_finetune, y_finetune, epochs=10, batch_size=8, validation_split=0.2)

    model.save("updated_model.keras")


def train():
    

    # Define the model
    model = tf.keras.Sequential([
        InputLayer(input_shape=(256,), name='input_layer'),  
        tf.keras.layers.Reshape((1, 256, 1)),  

        # Convolutional layers
        Conv2D(32, (1, 3), activation='relu', padding='same'),  # 1x3 convolution for embedding
        MaxPooling2D(pool_size=(1, 2)),  # Downsample

        Conv2D(64, (1, 3), activation='relu', padding='same'),  # 1x3 convolution
        MaxPooling2D(pool_size=(1, 2)),  # Downsample

        Conv2D(128, (1, 3), activation='relu', padding='same'),  # 1x3 convolution
        MaxPooling2D(pool_size=(1, 2)),  # Downsample

        Flatten(),  # Flatten the output for dense layers

        # Fully connected layers
        Dense(256, activation='relu'),
        Dropout(0.5),  # Regularization to prevent overfitting
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularization
        Dense(len(names), activation='softmax')  # Output layer for classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()

    model.save("updated_model.keras")
    return model
