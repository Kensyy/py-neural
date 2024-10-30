import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('../models/modelo_treinamento.keras')

for layer in model.layers[:-4]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '../dataset/train/',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)
test_generator = test_datagen.flow_from_directory(
    '../dataset/test/',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale'
)

model.fit(train_generator, epochs=50, validation_data=test_generator)
model.save('../models/modelo_emocoes_finetuned.keras')
