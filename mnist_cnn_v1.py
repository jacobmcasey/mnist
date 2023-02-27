# Jacob Casey / MNIST Ensemble 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Average
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

input_shape = (28,28,1) 
num_classes = 10

# 3x3 Kernel Size
mnist_cnn_v1 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the optimizer and loss function for training
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Compile the model
mnist_cnn_v1.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Define hyper-parameters
batch_size = 128
epochs = 10

# Train weights and biases
mnist_cnn_v1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the models on the test set
test_preds_1 = mnist_cnn_v1.predict(x_test)

# Evaluate the  model accuracy
test_loss, test_accuracy = mnist_cnn_v1.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

mnist_cnn_v1.save('mnist_cnn_v1.h5')
