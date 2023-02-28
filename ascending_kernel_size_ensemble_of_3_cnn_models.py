# Jacob Casey / MNIST Ensemble 
# Test accuracy = 0.992

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

input_shape = (28,28,1) 

num_classes = 10

model_1 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Model 2: 5x5 kernels
model_2 = Sequential([
    Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Model 3: 7x7 kernels
model_3 = Sequential([
    Conv2D(32, kernel_size=(7, 7), activation='relu', input_shape=input_shape),
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

# Compile the models
model_1.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_2.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_3.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the models
batch_size = 128
epochs = 10

# Define the data augmentation generator
datagen = ImageDataGenerator(rotation_range=15)

# Fit the models using data augmentation
batch_size = 128
epochs = 10

model_1.fit(datagen.flow(x_train, y_train, batch_size=batch_size), 
            epochs=epochs, validation_data=(x_test, y_test))
model_2.fit(datagen.flow(x_train, y_train, batch_size=batch_size), 
            epochs=epochs, validation_data=(x_test, y_test))
model_3.fit(datagen.flow(x_train, y_train, batch_size=batch_size), 
            epochs=epochs, validation_data=(x_test, y_test))

# Evaluate the models on the test set
test_preds_1 = model_1.predict(x_test)
test_preds_2 = model_2.predict(x_test)
test_preds_3 = model_3.predict(x_test)

# Combine the predictions using majority voting
test_preds = np.argmax(test_preds_1 + test_preds_2 + test_preds_3, axis=1)

# Evaluate the ensemble model accuracy
test_accuracy = np.mean(test_preds == np.argmax(y_test, axis=1))
print('Ensemble model accuracy:', test_accuracy)
