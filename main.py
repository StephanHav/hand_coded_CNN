# Keras is imported from TensorFlow in order to to load the MNIST dataset.

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
from CNN_funtions import *

## Classifying the MNIST dataset

# Train the model

# Load MINST data, splitting into training and test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape for CNN and rescale to 0-1
x_train = np.reshape(x_train, (60000,28,28,1))
x_test = np.reshape(x_test, (10000,28,28,1))
x_train = x_train / 255
x_test = x_test / 255

# Create one-hot vectors for labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Initialise filters
random.seed(1)
f1_a = initialiseFilter((3,3))
random.seed(2)
f1_b = initialiseFilter((3,3))

f1 = np.transpose(np.array([np.reshape(f1_a, (1, 3, 3)),
                            np.reshape(f1_b, (1, 3, 3))]),
                  (2, 3, 1, 0))

f1_orig = np.transpose(np.array([np.reshape(f1_a, (1, 3, 3)),
                                 np.reshape(f1_b, (1, 3, 3))]),
                       (2, 3, 1, 0))

random.seed(3)
f2_a = initialiseFilter((3,3))
random.seed(4)
f2_b = initialiseFilter((3,3))
random.seed(5)
f2_c = initialiseFilter((3,3))
random.seed(6)
f2_d = initialiseFilter((3,3))

f2 = np.transpose(np.array([np.broadcast_to(f2_a, (2, 3, 3)),
                            np.broadcast_to(f2_b, (2, 3, 3)),
                            np.broadcast_to(f2_c, (2, 3, 3)),
                            np.broadcast_to(f2_d, (2, 3, 3))]),
                  (2, 3, 1, 0))

f2_orig = np.transpose(np.array([np.broadcast_to(f2_a, (2, 3, 3)),
                                 np.broadcast_to(f2_b, (2, 3, 3)),
                                 np.broadcast_to(f2_c, (2, 3, 3)),
                                 np.broadcast_to(f2_d, (2, 3, 3))]),
                       (2, 3, 1, 0))

# Initialise weights
random.seed(7)
w = initialiseWeights((10,576))
random.seed(7)
w_orig = initialiseWeights((10,576))

# Set epoch number
epochs = 20

# Select number of images for training and set traininig/validation split
number_of_images = 12000
ratio = 0.8
x_train_sub = x_train[:int(number_of_images * ratio)]
y_train_sub = y_train[:int(number_of_images * ratio)]
x_val = x_train[int(number_of_images * ratio):number_of_images]
y_val = y_train[int(number_of_images * ratio):number_of_images]

# Initialise variables to store progress and validation dataset predictions
training_accuracy = []
validation_accuracy = []
predictions = []

# Train the model, tracking the training/val accuracy
for i in range(epochs):
    
    # Train for an epoch
    epoch_tr_output = cnn(x_train_sub, y_train_sub, f1, f2, w, (2,2), 0.01, 0.01, 128)
    f1 = epoch_tr_output[0]
    f2 = epoch_tr_output[1]
    w = epoch_tr_output[2]
    training_accuracy.append(epoch_tr_output[3])
    
    # Assess predictions for a validation dataset
    epoch_val_output = predict(x_val, y_val, f1, f2, w, (2,2))
    validation_accuracy.append(epoch_val_output[0])
    predictions.append(epoch_val_output[1])
    #print(" Training accuracy: {}%, Validation accuracy: {}%".format(epoch_tr_output[3], epoch_val_output[0]))

# Plot a graph that tracks the accuracy
x = [i+1 for i in range(epochs)]
plt.plot(x, training_accuracy, label = "train")
plt.plot(x, validation_accuracy, label = "val")
plt.xlabel("Epoch")
plt.ylabel("Classification accuracy (%)")
plt.title("Training history")
plt.legend()
plt.show()
