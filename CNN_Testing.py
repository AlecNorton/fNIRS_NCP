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
from sklearn.model_selection import train_test_split
import keras_tuner as kt


import sys

import argparse

parser = argparse.ArgumentParser()


parser.add_argument("batch_size")
parser.add_argument("epochs")
parser.add_argument("model_number")


args = parser.parse_args()


keras = tf.keras

def CNN(input):
    #Algorithm make up is CNN2-a from Trakoolqilaiwan et all.
    #x = tf.keras.layers.Conv2D(32, 3)(input)
    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Conv1D(32, 3)(input)
    x = tf.keras.layers.MaxPool1D(3)(x)
    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(384, activation = "relu")(x)
    x = tf.keras.layers.Dense(384, activation = "relu")(x)

    output = tf.keras.layers.Dense(4)(x)
    
    return tf.keras.Model(inputs = input, outputs = output)

def eval(model, index_arg, train_x, train_y, x_valid, y_valid, opt, loss_fun, batch_size, epochs):
    #Compile the Model
    model.compile(optimizer = opt, loss = loss_fun, metrics = tf.keras.metrics.SparseCategoricalAccuracy())

    #Return a summary of the model and its (non)trainable paramaters
    model.summary()

    #Fit the model and return accuracy
    #Get beginning of time
    start = time.process_time()
    hist = model.fit(x = train_x, y = train_y, validation_data = (x_valid, y_valid), batch_size = batch_size, epochs = epochs, verbose =1)
    end = time.process_time()
    test_accuracies = hist.history["val_sparse_categorical_accuracy"]
    print("Max Accuracy Of Model: " + str(np.max(test_accuracies)))
    return np.max(test_accuracies), end-start

#Based on the model_number, create a model and train on specified optimizer, loss_function, validation_split, batch_size, and some epochs
#Then return the mean and standard deviation of the accuracy of these models. 
def score(model, train_x, train_y, x_valid, y_valid, opt, loss_fun, model_number, batch_size, epochs):
    acc = []
    dur = []
    for i in range(model_number):
        print("Model: " + str(i))
        max_accuracy, time = eval(model, i, train_x, train_y, x_valid, y_valid, opt, loss_fun, batch_size, epochs)
        dur.append(time)
        acc.append(100 * max_accuracy)
    acc_average = np.mean(acc)
    acc_std = np.std(acc)
    dur_average = np.mean(dur)
    dur_std = np.std(dur)
    print("-------------------------------------------------------------------")
    print("Average Test Accuracy: " + str(acc_average) + " Standard Deviation Test Accuracy: " + str(acc_std))
    print("Average Time Training: " + str(dur_average) + " Standard Deviation Time: " + str(dur_std))
    print("-------------------------------------------------------------------")

    results = model.evaluate(x_valid, y_valid, batch_size=32)
    print("test loss, test acc:", results)



#Actual Execution of Code: 

#Load Data Here

#TODO: Load a Time-Series Application

csv_files = glob.glob('size_02sec_10ts_stride_03ts\*.csv')


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
reshape = int(x_train.shape[0]/10)
print(reshape)
x_train = x_train.reshape(reshape, 10, 8)

x_train = (x_train - np.mean(x_train, axis = 0)) / np.std(x_train, axis = 0)

x_train = x_train.astype(np.float32)

y_train = np.array(y_train)
y_train = y_train.reshape(reshape, 10, 2)
array = np.zeros(reshape, )
for i in range(0, reshape - 1):
    array[i] = y_train[i][0][1]

y_train = array
y_train = y_train.astype(np.int8)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = .33, shuffle = True)



input = tf.keras.layers.Input(shape = (10, 8))


#Pre-Processing:

#CNN

number_of_models = int(args.model_number)
batch_size = int(args.batch_size)
epochs = int(args.epochs)

cnn_optimizer = tf.keras.optimizers.Adam()

cnn_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)



score(CNN(input), x_train, y_train, x_valid, y_valid, cnn_optimizer, cnn_loss_fun, number_of_models, batch_size, epochs)
print("\n")
print("Epochs: " + str(epochs) + " Batch Size: " + str(batch_size) + " Number Of Models: " + str(number_of_models))

