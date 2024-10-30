# src/treinamento.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

def criar_modelo():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emoções
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

data_gen = ImageDataGenerator(rescale=1./255)
train_data = data_gen.flow_from_directory('../dataset/train/', target_size=(48, 48), color_mode='grayscale', batch_size=64)
test_data = data_gen.flow_from_directory('../dataset/test/', target_size=(48, 48), color_mode='grayscale', batch_size=64)

model = criar_modelo()

# checkpoint pra salvar o melhor modelo
checkpoint = ModelCheckpoint('../models/modelo_treinamento.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
model.fit(train_data, validation_data=test_data, epochs=30, callbacks=[checkpoint])
