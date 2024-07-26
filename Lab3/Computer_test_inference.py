#Jetson nano 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, AvgPool2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import time
import pathlib
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from memory_profiler import profile
import numpy as np

from tensorflow.keras.backend import clear_session
clear_session()



@profile
def prediction():
   predictions = model.predict(test_data)
   return


# rescale = tf.keras.layers.Rescaling(1./255) # No soportado por TF 2.4
rescale = tf.keras.layers.Lambda(lambda x: x / 255.0)
def preprocess_dataset(dataset):
    # Apply rescaling to each batch in the dataset
    return dataset.map(lambda x, y: (rescale(x), y))


# Define image dimensions
img_height = 128  # Reduced image size
img_width = 128   # Reduced image size
batch_size = 1


#loading
test_data = image_dataset_from_directory(
    r"/home/jetson/Documents/LAb3Embedded/test_images",
    label_mode="categorical",
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=12,
    validation_split=0.2,
    subset="validation",
    color_mode='grayscale'
)


# Preprocess the datasets
test_batches = preprocess_dataset(test_data)

# Define the main model
model = Sequential([
    tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 1)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    AvgPool2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

# Enable mixed precision if your hardware supports it
if tf.test.is_gpu_available():
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# Enable mixed precision if your hardware supports it
if tf.test.is_gpu_available():
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
 
 
# Load model weights
model_name = "model_weights.h5"
try:
    model.load_weights(model_name)
except ValueError as e:
    print(f"Error loading weights: {e}")

start_time = time.time()
predictions = prediction()
testing_time = time.time() - start_time
print(f"\nInference time: {testing_time} seconds.\n")

evaluation = model.evaluate(test_data)
print(f'\nTest accuracy: {evaluation[1]}')
print("---Inference tensorflow lite time %s seconds ---" % (time.time() - start_time))
