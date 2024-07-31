import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
from ncps.tf import CfC
from ncps.tf import LTC
from ncps.tf import LTCCell
import matplotlib.pyplot as plt
import glob
import time
import seaborn as sns
from sklearn.model_selection import train_test_split

import keras_spiking

keras = tf.keras

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

def LTC_FullyConnected(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.FullyConnected(ncp_size, ncp_output_size)
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

def CNN(input):
    #Algorithm make up is CNN2-a from Trakoolqilaiwan et all.
    #x = tf.keras.layers.Conv2D(32, 3)(input)
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    #x = tf.keras.layers.Conv1D(32, 3)(input)
    #x = tf.keras.layers.MaxPool1D(2)(x)
    #x = tf.keras.layers.Dropout(.5)(x)

    #x = tf.keras.layers.Conv1D(32, 3)(input)
    #x = tf.keras.layers.MaxPool1D(3)(x)
    #x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(384, activation = "relu")(x)
    x = tf.keras.layers.Dense(384, activation = "relu")(x)

    output = tf.keras.layers.Dense(4)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)


input = tf.keras.layers.Input(shape = (150, 8))
LTC_NCP_model = LTC_NCP(input, 100, 5, .5)
LTC_NCP_model.summary()

LTC_FC_model = LTC_FullyConnected(input, 100, 5, .2)
LTC_FC_model.summary()

CNN_model = CNN(input)
CNN_model.summary()

wiring = ncps.wirings.AutoNCP(100, 5, 0.5)
wiring.build(32)
connections = wiring.synapse_count + wiring.sensory_synapse_count
print("NCP_Connections: " + str(connections))

wiring = ncps.wirings.FullyConnected(100, 5)
wiring.build(32)
connections = wiring.synapse_count + wiring.sensory_synapse_count
print("FC_Connections: " + str(connections))


energy = keras_spiking.ModelEnergy(CNN_model)
energy.summary(print_warnings=False)

energy = keras_spiking.ModelEnergy(LTC_NCP_model)
energy.summary(print_warnings=False)

energy = keras_spiking.ModelEnergy(LTC_FC_model)
energy.summary(print_warnings=False)



