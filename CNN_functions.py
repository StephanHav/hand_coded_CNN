#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:58:26 2021

@author: stephanhavermans
"""

# The following script contains the code for a convolutional neural network (CNN) that uses backpropagation written without the use of functions from traditional
# Deep learning packages such as TensorFlow or PyTorch. The code was originally created as part of an assignment in a deep learning course that would reward extra 
# credits if completed successfully. 

from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm


## CNN functions

def convolution(inputs, kernels):
    
    spread = math.floor(kernels.shape[0]/2)
    
    assert kernels.shape[2] == inputs.shape[2]
     
    feature_maps = np.zeros((inputs.shape[0] - 2 * spread, inputs.shape[0] - 2 * spread,
                             kernels.shape[3]))
    
    # Apply each convolution in turn
    for k in range(kernels.shape[3]):
        
        # Iterate over each image
        for fm_number in range(inputs.shape[2]):

            # Iterate over each potential focus cell in the input
            for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    
                    # Ignore the cell if the kernel cannot be applied (too close to border)
                    if ((i-spread)<0 or (i+spread)>inputs.shape[0]-1
                        ) or ((j-spread)<0 or (j+spread)>inputs.shape[1]-1):
                        continue
                    
                    else:
                        # Update the output cell for the ouput feature map
                        # corresponding to that convolution and location
                        feature_maps[i-spread, j-spread, k] += np.sum(
                            kernels[:, :, :, k][:, :, fm_number] *
                            inputs[i-spread:i+spread+1, j-spread:j+spread+1, fm_number])
                                
    return feature_maps


#Rectified Linear Unit (ReLU) activation function
def relu(feature_maps):        
    return np.maximum(0,feature_maps)



def max_pooling(feature_maps, pool_size):
    
    fm_max_pooled = np.zeros((math.floor(feature_maps.shape[0]/pool_size[0]), 
                              math.floor(feature_maps.shape[0]/pool_size[0]), 
                              feature_maps.shape[2]))
    
    # Iterate over the input feature maps
    for z in range(feature_maps.shape[2]):
        curr_x = output_x = 0
        # Iterate over each location in the map, set x coordinate
        while curr_x + pool_size[0] <= feature_maps.shape[0]:
            curr_y = output_y = 0
            # Then iterate over the y coordinates
            while curr_y + pool_size[0] <= feature_maps.shape[0]:
                # Pick up the maximum value within the window and update the output
                fm_max_pooled[output_x, output_y, z] = np.max(feature_maps[curr_x:curr_x+pool_size[0],
                                                                           curr_y:curr_y+pool_size[0],
                                                                           z])
                curr_y += pool_size[0]
                output_y += 1
            curr_x += pool_size[0]
            output_x += 1
            
    return fm_max_pooled



def normalise(feature_maps):
    
    for fm_number in range(feature_maps.shape[2]):
        
        if np.std(feature_maps[:, :, fm_number]) == 0:
            feature_maps[:, :, fm_number] = feature_maps[:, :, fm_number] - np.mean(feature_maps[:, :, fm_number])
        else:
              feature_maps[:, :, fm_number] = (feature_maps[:, :, fm_number] - np.mean(feature_maps[:, :, fm_number])) / np.std(feature_maps[:, :, fm_number])
                                    
    return feature_maps



#Weights have to be in order (# of output nodes, weight values) --> (10, 450)
def fully_connected_layer(feature_maps, weights):
   return weights.dot(feature_maps.reshape((np.prod(feature_maps.shape),1)))

    

def softmax(activation_layer):
    return np.exp(activation_layer) / np.sum(np.exp(activation_layer))



def convolutionBackward(dconv_prev, conv_input, f):

    # Dimension information
    (f_dim, _, f_per_input, f_count) = f.shape
    (fm_dim, _, _) = conv_input.shape
    
    # Initialise variables to contain gradients/derivatives
    dout = np.zeros(conv_input.shape) 
    dfilt = np.zeros(filters.shape)
    
    # Iterate over each filter and image location
    for f in range(f_count):
        x = 0
        while x + f_dim <= fm_dim:    
            y = 0
            while y + f_dim <= fm_dim:
                # Loss gradient of filter
                dfilt[:, :, :, f] += dconv_prev[x, y, f] * conv_input[x:x+f_dim, y:y+f_dim, :]
                # Loss gradient of the input to the convolution operation
                dout[x:x+f_dim, y:y+f_dim, :] += dconv_prev[x, y, f] * filters[:, :, :, f]             
                y += 1
            x += 1
    
    return dout, dfilt


def nanargmax(arr):
    
    # Return index of the largest non-nan value in the array.
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    
    return idxs 


def maxpoolBackward(dpool, orig, pool_dim, s=2):

    # Dimensions and output    
    (fm_dim, _, fm_count) = orig.shape  
    dout = np.zeros(orig.shape)
    
    # Iterate over the feature_maps (and locations)
    for fm in range(fm_count):
        curr_x = output_x = 0
        while curr_x + pool_dim <= fm_dim:
            curr_y = output_y = 0
            while curr_y + pool_dim <= fm_dim:
                # Find the index of max value from the input for focus area
                (a, b) = nanargmax(orig[curr_x:curr_x+pool_dim, curr_y:curr_y+pool_dim, fm])
                # Update the reverse max pool output at that location 
                dout[curr_x+a, curr_y+b, fm] = dpool[output_x, output_y, fm]
                curr_y += s
                output_y += 1
            curr_x += s
            output_x += 1
        
    return dout


# A convolutional neural network with two convolutional layers, a fully connected layer, 
# and an output layer with pooling, thresholding and normalisation. The function gives 
# the accuracy of the labels as an output.

def cnn(images, labels, f1, f2, w, pool_size, lr, decay, batch_size):
    
    # Set variable to assess accuracy
    acc_count = 0
    
    # Number of batches
    batches = math.ceil(images.shape[0] / batch_size)
    batch_count = 0
    
    # Shuffle images/labels
    p = np.random.permutation(len(images))
    images = images[p]
    labels = labels[p]
    
    for n in tqdm(range(batches)):
        
        # Images and labels for the current batch
        images_batch = images[n * batch_size:(n * batch_size) + batch_size]
        labels_batch = labels[n * batch_size:(n * batch_size) + batch_size]
    
        # Initialise gradient totals for parameters     
        df1_batch = np.zeros((f1.shape))
        df2_batch = np.zeros((f2.shape))
        dw_batch = np.zeros((w.shape))
        
        for i in range(len(images_batch)):
            image = images_batch[i]
            label = labels_batch[i]
                
            #### Feed-forward
            conv1 = convolution(image, f1) # first convolutional layer
            conv1 = relu(conv1) # ReLU
            conv2 = convolution(conv1, f2) # second convolutional layer
            conv2 = relu(conv2) # ReLU
            pooled = max_pooling(conv2, pool_size) # max pooling
            norm = normalise(pooled) # normalisation
            fc = fully_connected_layer(norm, w) # fully connected layer
            probs = softmax(fc) # output probs
            # Check prediction
            if np.argmax(probs) == np.argmax(label):
                acc_count += 1
                         
            #### Backpropagation
            dout = probs - label.reshape(len(label), 1) # derivs of CCE and softmax 
            # Gradients of fc layer weights, product of output derivs and (flat) activations
            dw = (dout.reshape(len(dout),1)).dot(norm.reshape(np.prod(norm.shape),1).T)
            dfc = w.T.dot(dout) # derivative of the activations of the fully connected layer
            dpool = dfc.reshape(pooled.shape) # un-flatten
            dconv2 = maxpoolBackward(dpool, conv2, 2, s=2) # reverse max pooling
            dconv2[conv2<=0] = 0 # ReLU
            dconv1, df2 = convolutionBackward(dconv2, conv1, f2) # backpropagation conv2
            dconv1[conv1<=0] = 0 # ReLU
            dimage, df1 = convolutionBackward(dconv1, image, f1) # backpropagation conv1
            # Add image gradients to batch gradient totals
            df1_batch += df1
            df2_batch += df2
            dw_batch += dw
        
        # Calculate mean update for mini-batch gradient descent
        df1_update = df1_batch / len(images_batch)
        df2_update = df2_batch / len(images_batch)
        dw_update = dw_batch / len(images_batch)
        # Update parameters
        f1 -= lr * df1_update
        f2 -= lr * df2_update
        w -= lr * dw_update
        # Update learning rate
        batch_count += 1
        lr = lr * (1.0 / (1.0 + decay * batch_count))
    
    # Calculate accuracy
    accuracy = round((acc_count / len(images)) * 100, 2)
    
    return [f1, f2, w, accuracy]

# Define functions to initialise parameters
def initialiseFilter(size, scale = 1.0):
    
    # Initialise filter: normal dist. with sd inversely proportional the square root of the number of units
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)


def initialiseWeights(size):
    
    # Initialize weights with a random normal distribution
    return np.random.standard_normal(size=size) * 0.01


# Define a function to predict categories and output test accuracy
def predict(images, labels, f1, f2, w, pool_size):
    
    # Set variables to assess accuracy and store predictions
    acc_count = 0
    predictions = []
        
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        
        #### Feed-forward
        conv1 = convolution(image, f1) # first convolutional layer
        conv1 = relu(conv1) # ReLU
        conv2 = convolution(conv1, f2) # second convolutional layer
        conv2 = relu(conv2) # ReLU
        pooled = max_pooling(conv2, pool_size) # max pooling
        norm = normalise(pooled) # normalisation
        fc = fully_connected_layer(norm, w) # fully connected layer
        probs = softmax(fc) # output probs
        # Check prediction
        if np.argmax(probs) == np.argmax(label):
            acc_count += 1
        # Add prediction to list 
        predictions.append(np.argmax(probs))
    
    # Calculate accuracy
    accuracy = round((acc_count / len(images)) * 100, 2)
    
    return [accuracy, predictions]
