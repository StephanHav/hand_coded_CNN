# Keras is imported from TensorFlow in order to to load the CIFAR10 dataset of which the images are used 
# to test the CNN and its classifying capabilities.

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
from CNN_funtions import *

## Test the functions
 
# Load CIFAR10 data and select a test image
(x_train_obj, y_train_obj), (x_test_obj, y_test_obj) = keras.datasets.cifar10.load_data()
test_image = x_train_obj[4]

# Sobel operators - filters to find horizontal and vertical edges/contrasts
filters = np.transpose(np.array([np.broadcast_to(np.array([[-1,-2,-1], 
                                                           [ 0, 0, 0], 
                                                           [ 1, 2, 1]]), (3, 3, 3)),
                                 np.broadcast_to(np.array([[-1, 0, 1], 
                                                           [-2, 0, 2], 
                                                           [-1, 0, 1]]), (3, 3, 3))]),(2, 3, 1, 0))

# Run through and plot the output for the test image
plt.imshow(test_image)

conv_out = convolution(test_image, filters)
plt.imshow(conv_out[:,:,0], cmap='gray')
plt.imshow(conv_out[:,:,1], cmap='gray')

relu_out = relu(conv_out)
plt.imshow(relu_out[:,:,0], cmap='gray')
plt.imshow(relu_out[:,:,1], cmap='gray')

mp_out = max_pooling(relu_out, (2,2))
plt.imshow(mp_out[:,:,0], cmap='gray')
plt.imshow(mp_out[:,:,1], cmap='gray')

norm_out = normalise(mp_out)

weights = np.random.random((10,450))

fcl_out = fully_connected_layer(norm_out, weights)

final_classification = softmax(fcl_out).argmax()
