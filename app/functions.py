# -*- coding: utf-8 -*-
# Bibliotecas para o modelo
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import pickle
import numpy as np

def treino():
    diretorio_raiz = '../data/dados_tratados'

    # Defina os diretórios de treinamento e validação
    train_path = diretorio_raiz+'/train_path'
    validation_path = diretorio_raiz+'/validation_path'

    TRAINING_DIR = train_path
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = validation_path
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(96, 96),
        class_mode='categorical',
        batch_size=126
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(96, 96),
        class_mode='categorical',
        batch_size=126
    )

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 96x96 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(96, 96, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(17, activation='softmax')
    ])

    summary = model.summary()

    score = model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop', metrics=['accuracy'])
    
    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, 
                        validation_data = validation_generator, verbose = 1, validation_steps=3)
    
    with open("../models/toxic_to_pet.pkl", "wb") as f:
        pickle.dump(model, f)

    return score, history, summary

if __name__ == "__main__":
    treino()




    
