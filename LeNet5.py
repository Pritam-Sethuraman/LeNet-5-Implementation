# Importing Necessary Libraries & Frameworks
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Training the Dataset
(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()
train_x = train_x / 255.0
test_x = test_x / 255.0
train_x = tf.expand_dims(train_x, 3)
test_x = tf.expand_dims(test_x, 3)
val_x = train_x[:5000]
val_y = train_y[:5000]

# Implementation of leNet 5 Model
lenet_5_model = keras.models.Sequential([
    keras.layers.Conv2D(6, kernel_size = 5, strides = 1,  activation = 'tanh', input_shape = train_x[0].shape, padding = 'same'), #C1: Feature Maps 6@28x28
    keras.layers.AveragePooling2D(), #S2: Feature Maps 6@14x14
    keras.layers.Conv2D(16, kernel_size = 5, strides = 1, activation = 'tanh', padding = 'valid'), #C3: Feature Maps 16@10x10
    keras.layers.AveragePooling2D(), #S4: Feature Maps 16@5x5
    keras.layers.Flatten(), #Flatten
    keras.layers.Dense(120, activation = 'tanh'), #C5: Layer
    keras.layers.Dense(84, activation = 'tanh'), #F6: Layer
    keras.layers.Dense(10, activation = 'softmax') #Output layer
])

# Compiling and Building the Model
lenet_5_model.compile(optimizer= 'adam', loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
  
# Evaluating the Model
lenet_5_model.fit(train_x, train_y, epochs=10, validation_data=(val_x, val_y))
lenet_5_model.evaluate(test_x, test_y)
