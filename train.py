from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization
import os
import json
import tensorflow as tf
import gc
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

if tf.config.list_physical_devices('GPU'):
    print("✅ GPU detected, training will be faster!")
else:
    print("⚠️ No GPU detected, training might be slow!")

sz = 128

tf.keras.backend.clear_session()
gc.collect()

classifier = Sequential([
    Input(shape=(sz, sz, 1)),  
    Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(units=128, activation='relu', kernel_initializer='he_uniform'),
    Dropout(0.5),
    Dense(units=64, activation='relu', kernel_initializer='he_uniform'),
    Dropout(0.4),
])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.05,  
    zoom_range=0.05,  
    rotation_range=10,  
    width_shift_range=0.05,  
    height_shift_range=0.05,  
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_path = 'data/train'

selected_classes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]

batch_size = 32
training_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(sz, sz),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=selected_classes
)

validation_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(sz, sz),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    classes=selected_classes
)

num_classes = len(training_set.class_indices)
classifier.add(Dense(units=num_classes, activation='softmax'))

classifier.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

classifier.summary()

steps_per_epoch = max(math.ceil(training_set.samples / batch_size), 1)
validation_steps = max(math.ceil(validation_set.samples / batch_size), 1)

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "class_indices.json"), "w") as f:
    json.dump(training_set.class_indices, f)
print("✅ Class indices saved to 'model/class_indices.json'")

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

classifier.fit(
    training_set,
    steps_per_epoch=int(steps_per_epoch),
    validation_data=validation_set,
    validation_steps=int(validation_steps),
    epochs=30,
    verbose=1,
    callbacks=[lr_reduction, early_stopping]
)

model_json = classifier.to_json()
with open(os.path.join(model_dir, "model-bw.json"), "w") as json_file:
    json_file.write(model_json)
print('✅ Model saved in model/model-bw.json')

classifier.save_weights(os.path.join(model_dir, "model-bw.weights.h5"))
print('✅ Weights saved in model/model-bw.weights.h5')