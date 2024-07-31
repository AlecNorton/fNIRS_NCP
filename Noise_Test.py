import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
import csv
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import glob
import time 
from sklearn.model_selection import train_test_split
import keras_tuner as kt


import sys

import argparse

keras = tf.keras

#Load a model for LTC_NCP
def LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.AutoNCP(ncp_size, ncp_output_size, ncp_sparsity_level)
    #Begin constructing layer, starting with input
    
    '''model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape = (None, 8)),
            CfC(wiring),
            tf.keras.layers.Dense(1)
        ]
    )'''
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = LTC(wiring, return_sequences= True)(x)
    x = keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(4)(x)

    model = tf.keras.Model(inputs = input, outputs = output)
    
    
    #Return model
    return model

#Load dataset
csv_files = glob.glob('/home/arnorton/NCP_Testing/size_30sec_150ts_stride_03ts/*.csv')



x_train = pd.DataFrame()
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])

#Collect labeled data
y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')


x_train = np.array(x_train)
print(x_train.shape)
#Reshape based on amount of samples in the window 
reshape = int(x_train.shape[0]/150)
print(reshape)
x_train = x_train.reshape(reshape, 150, 8)


x_train = x_train.astype(np.float32)

y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 150, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]


y_train = array
y_train = y_train.astype(np.int8)

input = tf.keras.layers.Input(shape = (150, 8))

#Initialize model to place weights in
LTC_NCP_model = LTC_NCP(input, 100, 5, .5)

print("\n")
print("Loading Models: ")
#Load models from folders. It's important that the LTC_NCP weights match the LTC_NCP_model configuration (i.e. same amount of units, sparsity level,etc.)
CNN_model = keras.models.load_model('CNN_Model/saved_model' )
LTC_NCP_model.load_weights('LTC_NCP_Model/saved_model.weights.h5')



base_lr = .02
train_steps = reshape // 64
decay_lr = .66
clipnorm = .9999

learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )


ncp_optimizer = tf.keras.optimizers.Adam(learning_rate_fn, clipnorm = clipnorm)

ncp_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

LTC_NCP_model.compile(ncp_optimizer, ncp_loss,  metrics = tf.keras.metrics.SparseCategoricalAccuracy())

cnn_optimizer = tf.keras.optimizers.Adam()

cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
CNN_model.compile(optimizer = cnn_optimizer, loss = cnn_loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

#Shuffle dataset randomly
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = .66, shuffle = True)
noise_x = []
CNN_accuracy = []
LTC_NCP_accuracy = []
LTC_FC_accuracy = []

print("Noise_Testing: ")

noise_x.append(0)
CNN_results = CNN_model.evaluate(x_valid, y_valid, verbose = 1)
LTC_NCP_results = LTC_NCP_model.evaluate(x_valid, y_valid, verbose = 1)


print("Begin iterative noise testing: ")

for i in range(0, 100, 1):
    #Add a guassian noise distribution, slowly increasing the amount of noise affecting the test data.
    noise_copy = x_valid + np.random.normal(0, float(float(i) / 100), x_valid.shape)
    #Pre-process so everything is between zero and one, this makes it faster to evaluate
    noise_copy = (noise_copy - np.mean(noise_copy, axis = 0)) / np.std(noise_copy, axis = 0)
    CNN_results = CNN_model.evaluate(noise_copy, y_valid, verbose = 1)
    LTC_NCP_results = LTC_NCP_model.evaluate(noise_copy, y_valid, verbose = 1)
    #Add values to list
    noise_x.append(float(float(i) / 100))
    CNN_accuracy.append(CNN_results[1])
    LTC_NCP_accuracy.append(LTC_NCP_results[1])

    



#Print out results
print("CNN accuracy: ")
for i in range(0, 100, 1):
    print(CNN_accuracy[i])

print("LTC_NCP accuracy")
for i in range(0, 100, 1):
    print(LTC_NCP_accuracy[i])

print("LTC_FC_accuracy")
for i in range(0, 100, 1):
    print(LTC_FC_accuracy[i])

print("Finished, did guassian noise distribution!")







