
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from memory_profiler import profile
import warnings

warnings.filterwarnings('ignore')


# Import the Rescaling layer from the correct module
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

@profile
def prediction_fun():
   interpreter.invoke()
   return
#----------------------------------Loading Dataset----------------------------------------

# rescale = tf.keras.layers.Rescaling(1./255) # No soportado por TF 2.4
rescale = tf.keras.layers.Lambda(lambda x: x / 255.0)

def preprocess_dataset(dataset):
    # Aplicar la reescalada a cada lote en el dataset
    return dataset.map(lambda x, y: (rescale(x), y))

# Define image dimensions
img_height = 128  # Reduced image size
img_width = 128   # Reduced image size
batch_size = 1

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


#load tflite mode
tflite_model_file = "Model_tf"
interpreter = tf.lite.Interpreter(model_path=tflite_model_file+".tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]


score = 0
testing_time = 0

for img, label in test_data.take(10):
  interpreter.set_tensor(input_index, img)
  start_time = time.time()
  prediction_fun()
  testing_time += time.time() - start_time
  
  prediction = np.argmax(interpreter.get_tensor(output_index))
  true_label = np.argmax(label.numpy(), axis=-1)
  if prediction == true_label:
      score += 1

accuracy = score / 10  # Assuming you're testing on 10 images
print(f"Accuracy: {accuracy}")
print(f"\nInference time: {testing_time} seconds.\n")
