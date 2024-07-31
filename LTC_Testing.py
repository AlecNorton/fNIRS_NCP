import tensorflow as tf

import pandas as pd
import numpy as np
import ncps 
from ncps.tf import CfC
from ncps.tf import LTC
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import time 

import sys

keras = tf.keras

#define a function to return a NCP LTC Model
def LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level):
    #Set up architecture for Neural Circuit Policy
    wiring = ncps.wirings.AutoNCP(ncp_size, ncp_output_size, ncp_sparsity_level)
    #Begin constructing layer, starting with input
    x = LTC(wiring)(input)
    output = tf.keras.layers.Dense(4)(x)
    #Return model
    return tf.keras.Model(inputs = input, outputs = output)

def LTC_FullConnected(input, size):
    #Create fully connected architecture for CfC layer
    wiring = ncps.wirings.FullyConnected(size)
    #Begin constructing layers, starting with input
    x = LTC(wiring)(input)
    output = tf.keras.layers.Dense(4)(x)
    #Return model
    return tf.keras.Model(inputs = input, outputs = output)

def eval(model, index_arg, train_x, train_y, opt, loss_fun, validation_split, batch_size, epochs):
    #Compile the Model
    model.compile(optimizer = opt, loss = loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

    #Return a summary of the model and its (non)trainable paramaters
    model.summary()

    #Fit the model and return accuracy
    #Get beginning of time
    start = time.process_time()
    hist = model.fit(x = train_x, y = train_y, validation_split = validation_split, batch_size = batch_size, epochs = epochs, verbose =1)
    end = time.process_time()
    test_accuracies = hist.history["val_sparse_categorical_accuracy"]
    print("Max Accuracy Of Model: " + str(np.max(test_accuracies)))
    return np.max(test_accuracies), end-start

#Based on the model_number, create a model and train on specified optimizer, loss_function, validation_split, batch_size, and some epochs
#Then return the mean and standard deviation of the accuracy of these models. 
def score(model, train_x, train_y, opt, loss_fun, model_number, validation_split, batch_size, epochs):
    acc = []
    dur = []
    for i in range(model_number):
        print("Model: " + str(i))
        max_accuracy, time = eval(model, i, train_x, train_y, opt, loss_fun, validation_split, batch_size, epochs)
        dur.append(time)
        acc.append(100 * max_accuracy)
    acc_average = np.mean(acc)
    acc_std = np.std(acc)
    dur_average = np.mean(dur)
    dur_std = np.std(dur)
    print("Average Test Accuracy: " + str(acc_average) + " Standard Deviation Test Accuracy: " + str(acc_std))
    print("Average Time Training: " + str(dur_average) + " Standard Deviation Time: " + str(dur_std))

    #print(f"Test Accuracy: {np.mean(acc):(1/model_number)}\\% $\\pm$ {np.std(acc):(1/model_number)}")

#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

csv_files = glob.glob('size_30sec_150ts_stride_03ts\*.csv')


x_train = pd.DataFrame()

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    x_train = pd.concat([x_train, df])




#csv_file = pd.read_csv('size_30sec_150ts_stride_03ts\sub_1.csv')
#x_train = csv_file.copy()

y_train = x_train.loc[:, ['chunk', 'label']]
x_train.pop('chunk')
x_train.pop('label')


x_train = np.array(x_train)
print(x_train.shape)
reshape = int(x_train.shape[0]/150)
print(reshape)
x_train = x_train.reshape(reshape, 150, 8)


y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 150, 2)

array = np.zeros(reshape, )

for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array


input = tf.keras.layers.Input(shape = (150, 8))



#Pre-Processing:


#LTC NCP
ncp_size = 40
ncp_output_size = 10
ncp_sparsity_level = .5
ltc_optimizer = tf.keras.optimizers.Adam()
ltc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
number_of_models = 5
validation_split = .3
batch_size = 128
epochs = 5

print("LTC-NCP Testing: ")
print("NCP Size: " + str(ncp_size))
print("NCP Output Size: " + str(ncp_output_size))
print("NCP Sparisty Level: " + str(ncp_sparsity_level))
print("Optimizer: " + str(ltc_optimizer))
print("Loss_fun" + str(ltc_loss))
score(LTC_NCP(input, ncp_size, ncp_output_size, ncp_sparsity_level), x_train, y_train, ltc_optimizer, ltc_loss, number_of_models, validation_split, batch_size, epochs)

#LTC Fully Connected
size = 40
ltc_optimizer = tf.keras.optimizers.Adam()
ltc_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
number_of_models = 5
validation_split = .3
batch_size = 128
epochs = 5

print("LTC- Fully Connected Testing: ")
print("Size: " + str(size))
print("Optimizer: " + str(ltc_optimizer))
print("Loss_fun" + str(ltc_loss))
score(LTC_FullConnected(input, size), x_train, y_train, ltc_optimizer, ltc_loss, number_of_models, validation_split, batch_size, epochs)
